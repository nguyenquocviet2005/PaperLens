"""Request ids, access logs, EMF metrics and LLM token metrics."""

from __future__ import annotations

import json
import logging
import uuid

import pytest
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from app.config import Settings
from app.infrastructure import observability as obs


@pytest.fixture()
def metrics_on(monkeypatch):
    settings = Settings(database_url="sqlite:///:memory:", metrics_enabled=True)
    monkeypatch.setattr(obs, "get_settings", lambda: settings)
    return settings


def _emf_lines(capsys) -> list[dict]:
    out = capsys.readouterr().out
    return [json.loads(line) for line in out.splitlines() if line.startswith('{"_aws"')]


def test_emf_record_shape(metrics_on, capsys):
    obs.emit_metrics({"Requests": (1, "Count"), "RequestLatency": (12.5, "Milliseconds")},
                     dimensions={"Route": "GET /papers"})
    (record,) = _emf_lines(capsys)
    spec = record["_aws"]["CloudWatchMetrics"][0]
    assert spec["Namespace"] == "PaperLens"
    assert spec["Dimensions"] == [["Route"], []]
    assert {m["Name"] for m in spec["Metrics"]} == {"Requests", "RequestLatency"}
    assert record["Route"] == "GET /papers" and record["RequestLatency"] == 12.5


def test_metrics_off_by_default(monkeypatch, capsys):
    monkeypatch.setattr(obs, "get_settings", lambda: Settings(database_url="sqlite:///:memory:"))
    obs.emit_metrics({"Requests": (1, "Count")})
    assert _emf_lines(capsys) == []


def _app() -> FastAPI:
    app = FastAPI()
    app.add_middleware(obs.RequestContextMiddleware)

    @app.get("/papers/{paper_id}")
    def get_paper(paper_id: str):
        logging.getLogger("test").info("inside handler")
        return {"id": paper_id, "request_id": obs.current_request_id()}

    @app.get("/stream")
    def stream():
        return StreamingResponse(iter(["a", "b"]), media_type="text/event-stream")

    @app.get("/boom")
    def boom():
        raise RuntimeError("fail")

    return app


def test_request_id_is_generated_returned_and_logged(metrics_on, capsys):
    client = TestClient(_app())
    response = client.get("/papers/p1")
    request_id = response.headers["x-request-id"]
    assert len(request_id) == 32 and response.json()["request_id"] == request_id

    records = _emf_lines(capsys)
    assert records[-1]["Route"] == "GET /papers/{paper_id}"  # template, not the raw path
    assert records[-1]["request_id"] == request_id
    assert records[-1]["Errors5xx"] == 0


def test_incoming_request_id_is_kept_only_if_safe(metrics_on):
    client = TestClient(_app())
    good = uuid.uuid4().hex
    assert client.get("/papers/p1", headers={"X-Request-ID": good}).headers["x-request-id"] == good
    bad = client.get("/papers/p1", headers={"X-Request-ID": "x\nInjected: 1"}).headers["x-request-id"]
    assert bad != "x\nInjected: 1" and len(bad) == 32


def test_streaming_and_errors_are_measured(metrics_on, capsys):
    client = TestClient(_app(), raise_server_exceptions=False)
    assert client.get("/stream").text == "ab"
    assert client.get("/boom").status_code == 500
    records = {r["Route"]: r for r in _emf_lines(capsys) if "Route" in r}
    assert "TimeToFirstByte" in records["GET /stream"]
    assert records["GET /boom"]["Errors5xx"] == 1


def test_json_log_lines_carry_request_id():
    formatter = obs.JsonFormatter()
    token = obs._request_id.set("req-12345678")
    try:
        record = logging.makeLogRecord({"name": "x", "levelname": "INFO", "msg": "hello %s",
                                        "args": ("world",), "route": "GET /a"})
        payload = json.loads(formatter.format(record))
    finally:
        obs._request_id.reset(token)
    assert payload["message"] == "hello world"
    assert payload["request_id"] == "req-12345678"
    assert payload["route"] == "GET /a"


def test_llm_callback_reports_tokens_per_model(metrics_on, capsys):
    run_id = uuid.uuid4()
    cb = obs.LLMMetricsCallback()
    cb.on_chat_model_start({}, [[]], run_id=run_id, invocation_params={"model": "gpt-5.4-mini"})
    message = AIMessage(content="hi", usage_metadata={"input_tokens": 120, "output_tokens": 30,
                                                      "total_tokens": 150})
    cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id=run_id)
    (record,) = _emf_lines(capsys)
    assert record["Model"] == "gpt-5.4-mini"
    assert record["LLMInputTokens"] == 120 and record["LLMOutputTokens"] == 30
    assert record["LLMCalls"] == 1


def test_llm_callback_reports_errors(metrics_on, capsys):
    run_id = uuid.uuid4()
    cb = obs.LLMMetricsCallback()
    cb.on_chat_model_start({}, [[]], run_id=run_id, invocation_params={"model": "m"})
    cb.on_llm_error(TimeoutError("slow"), run_id=run_id)
    (record,) = _emf_lines(capsys)
    assert record["LLMErrors"] == 1 and record["error_type"] == "TimeoutError"


def test_guardrail_blocks_are_counted(metrics_on, capsys):
    from app.harness.guardrails import GuardrailError

    GuardrailError("no", code="exfiltration_pattern")
    (record,) = _emf_lines(capsys)
    assert record["Code"] == "exfiltration_pattern" and record["GuardrailBlocks"] == 1
