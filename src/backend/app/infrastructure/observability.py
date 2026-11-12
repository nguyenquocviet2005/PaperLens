"""Logging, request correlation and application metrics.

- JSON logs (LOG_FORMAT=json) carry the request id of the request being served, so one
  CloudWatch Logs Insights query returns every line of a request.
- Metrics use the CloudWatch Embedded Metric Format (METRICS_ENABLED=true): a JSON line on
  stdout that CloudWatch turns into metrics. No agent, no API calls, nothing to fail.
- LLM tracing goes to LangSmith when the standard LANGSMITH_* variables are set; LangChain
  and LangGraph pick them up on their own.

Every function here swallows its own errors: observability must never break a request.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
import uuid
from contextvars import ContextVar
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

NAMESPACE = "PaperLens"
REQUEST_ID_HEADER = "x-request-id"
_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{8,64}$")
_LOG_ATTRS = set(vars(logging.makeLogRecord({})))


def current_request_id() -> str | None:
    return _request_id.get()


# --------------------------------------------------------------------------- logging


class JsonFormatter(logging.Formatter):
    """One JSON object per line; extra= fields and the request id are included."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S") + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = current_request_id()
        if request_id:
            payload["request_id"] = request_id
        for key, value in vars(record).items():
            if key not in _LOG_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id() or "-"
        return True


def configure_logging(settings: Settings) -> None:
    handler = logging.StreamHandler(sys.stdout)
    if settings.log_format == "json":
        handler.setFormatter(JsonFormatter())
        logging.Formatter.converter = time.gmtime
    else:
        handler.addFilter(_RequestIdFilter())
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] [%(request_id)s] %(message)s")
        )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))


# --------------------------------------------------------------------------- metrics

_emit_lock = threading.Lock()


def emit_metrics(
    metrics: dict[str, tuple[float, str]],
    *,
    dimensions: dict[str, str] | None = None,
    aggregate: bool = True,
    properties: dict[str, Any] | None = None,
    settings: Settings | None = None,
) -> None:
    """Publish metrics as one EMF line: {"Name": (value, "Unit")}.

    `dimensions` adds a per-dimension series; `aggregate` also publishes the series
    without dimensions so dashboards can show totals. Keep dimension values low
    cardinality (route templates, model names, status codes; never ids).
    """
    try:
        cfg = settings or get_settings()
        if not cfg.metrics_enabled or not metrics:
            return
        dims = {k: str(v) for k, v in (dimensions or {}).items()}
        dimension_sets: list[list[str]] = []
        if dims:
            dimension_sets.append(sorted(dims))
        if aggregate or not dims:
            dimension_sets.append([])
        record: dict[str, Any] = {
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [
                    {
                        "Namespace": NAMESPACE,
                        "Dimensions": dimension_sets,
                        "Metrics": [{"Name": name, "Unit": unit} for name, (_, unit) in metrics.items()],
                    }
                ],
            },
            **dims,
            **{name: value for name, (value, _) in metrics.items()},
            "env": cfg.app_env,
        }
        request_id = current_request_id()
        if request_id:
            record["request_id"] = request_id
        if properties:
            record.update({k: v for k, v in properties.items() if k not in record})
        line = json.dumps(record, default=str)
        with _emit_lock:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
    except Exception:  # pragma: no cover - never let metrics break the caller
        logger.debug("metric emission failed", exc_info=True)


def count(name: str, *, dimensions: dict[str, str] | None = None, value: float = 1) -> None:
    emit_metrics({name: (value, "Count")}, dimensions=dimensions)


# --------------------------------------------------------------------------- HTTP


class RequestContextMiddleware:
    """Pure ASGI middleware: request id, one access-log line and metrics per request.

    Pure ASGI (not BaseHTTPMiddleware) so streamed responses (SSE) are timed until the
    last byte, and nothing is buffered.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        incoming = dict(scope.get("headers") or []).get(REQUEST_ID_HEADER.encode(), b"").decode()
        request_id = incoming if _SAFE_REQUEST_ID.match(incoming or "") else uuid.uuid4().hex
        token = _request_id.set(request_id)
        started = time.perf_counter()
        state: dict[str, Any] = {"status": 500, "first_byte": None}

        async def send_wrapper(message: dict) -> None:
            if message["type"] == "http.response.start":
                state["status"] = message["status"]
                state["first_byte"] = time.perf_counter()
                headers = list(message.get("headers") or [])
                headers.append((REQUEST_ID_HEADER.encode(), request_id.encode()))
                message = {**message, "headers": headers}
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            self._record(scope, state, started)
            _request_id.reset(token)

    @staticmethod
    def _record(scope: dict, state: dict[str, Any], started: float) -> None:
        try:
            route = getattr(scope.get("route"), "path", None) or "unmatched"
            method = scope.get("method", "GET")
            if method == "OPTIONS" or route == "/health":
                return
            status = int(state["status"])
            duration_ms = round((time.perf_counter() - started) * 1000, 1)
            first_byte_ms = (
                round((state["first_byte"] - started) * 1000, 1) if state["first_byte"] else None
            )
            logging.getLogger("paperlens.access").info(
                "%s %s %s %.0fms",
                method,
                route,
                status,
                duration_ms,
                extra={
                    "http_method": method,
                    "route": route,
                    "status": status,
                    "duration_ms": duration_ms,
                    "first_byte_ms": first_byte_ms,
                },
            )
            metrics = {
                "Requests": (1, "Count"),
                "RequestLatency": (duration_ms, "Milliseconds"),
                "Errors5xx": (1 if status >= 500 else 0, "Count"),
                "Errors4xx": (1 if 400 <= status < 500 else 0, "Count"),
            }
            if first_byte_ms is not None:
                metrics["TimeToFirstByte"] = (first_byte_ms, "Milliseconds")
            emit_metrics(metrics, dimensions={"Route": f"{method} {route}"})
        except Exception:  # pragma: no cover
            logger.debug("request metrics failed", exc_info=True)


# --------------------------------------------------------------------------- LLM calls


class LLMMetricsCallback(BaseCallbackHandler):
    """Latency, errors and token usage of every LangChain chat-model call, per model."""

    raise_error = False

    def __init__(self) -> None:
        self._started: dict[UUID, tuple[float, str]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _model_name(serialized: dict[str, Any] | None, kwargs: dict[str, Any]) -> str:
        params = kwargs.get("invocation_params") or {}
        name = params.get("model") or params.get("model_name")
        if not name and serialized:
            name = (serialized.get("kwargs") or {}).get("model_name") or (
                serialized.get("kwargs") or {}
            ).get("model")
        return str(name or "unknown")

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):  # type: ignore[override]
        with self._lock:
            self._started[run_id] = (time.perf_counter(), self._model_name(serialized, kwargs))

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):  # type: ignore[override]
        with self._lock:
            self._started[run_id] = (time.perf_counter(), self._model_name(serialized, kwargs))

    def _pop(self, run_id: UUID) -> tuple[float, str]:
        with self._lock:
            return self._started.pop(run_id, (time.perf_counter(), "unknown"))

    def on_llm_end(self, response, *, run_id, **kwargs):  # type: ignore[override]
        started, model = self._pop(run_id)
        input_tokens = output_tokens = 0
        for generations in response.generations or []:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                input_tokens += int(usage.get("input_tokens") or 0)
                output_tokens += int(usage.get("output_tokens") or 0)
        if not (input_tokens or output_tokens):
            usage = (response.llm_output or {}).get("token_usage") or {}
            input_tokens = int(usage.get("prompt_tokens") or 0)
            output_tokens = int(usage.get("completion_tokens") or 0)
        emit_metrics(
            {
                "LLMCalls": (1, "Count"),
                "LLMLatency": (round((time.perf_counter() - started) * 1000, 1), "Milliseconds"),
                "LLMInputTokens": (input_tokens, "Count"),
                "LLMOutputTokens": (output_tokens, "Count"),
            },
            dimensions={"Model": model},
        )

    def on_llm_error(self, error, *, run_id, **kwargs):  # type: ignore[override]
        _started, model = self._pop(run_id)
        emit_metrics(
            {"LLMCalls": (1, "Count"), "LLMErrors": (1, "Count")},
            dimensions={"Model": model},
            properties={"error_type": type(error).__name__},
        )


LLM_METRICS = LLMMetricsCallback()


def llm_callbacks() -> list[BaseCallbackHandler]:
    return [LLM_METRICS]


def record_vlm_call(*, model: str, seconds: float, usage: dict[str, Any] | None, ok: bool) -> None:
    """Luna uses the OpenAI SDK directly, so it reports its calls here."""
    usage = usage or {}
    emit_metrics(
        {
            "LLMCalls": (1, "Count"),
            "LLMErrors": (0 if ok else 1, "Count"),
            "LLMLatency": (round(seconds * 1000, 1), "Milliseconds"),
            "LLMInputTokens": (int(usage.get("prompt_tokens") or 0), "Count"),
            "LLMOutputTokens": (int(usage.get("completion_tokens") or 0), "Count"),
        },
        dimensions={"Model": model},
    )


def trace_config(run_name: str, **metadata: Any) -> dict[str, Any]:
    """Run config so LangSmith traces carry a readable name and filterable metadata.

    Callbacks are attached where models are built, not here, to avoid counting twice.
    """
    values = {k: v for k, v in metadata.items() if v is not None}
    request_id = current_request_id()
    if request_id:
        values["request_id"] = request_id
    return {"run_name": run_name, "metadata": values}
