"""Luna adapter unit tests with mocks (no paid API calls)."""

from __future__ import annotations


import pytest

from app.config import Settings
from app.ingestion.luna import LunaClient, LunaDisabledError, mock_enrichment
from app.schemas import VisualElement
from app.infrastructure.storage import LocalStorage


def _settings(**kwargs) -> Settings:
    base = dict(
        luna_enabled=True,
        allow_external_api=True,
        luna_api_key="test-key",
        luna_model="mock-model",
        luna_max_retries=1,
        luna_request_delay_seconds=0,
    )
    base.update(kwargs)
    return Settings(**base)


def test_disabled_raises():
    client = LunaClient(settings=_settings(luna_enabled=False, allow_external_api=False))
    with pytest.raises(LunaDisabledError):
        client.enrich_element(
            paper_id="p",
            element=VisualElement(element_id="formula_001", type="formula", needs_enrichment=True),
            image_bytes=b"png",
        )


def test_mocked_structured_response(tmp_path, monkeypatch):
    store = LocalStorage(tmp_path)
    client = LunaClient(settings=_settings(), storage=store)

    def fake_call(prompt, image_bytes):
        return mock_enrichment("formula"), {"total_tokens": 10}

    monkeypatch.setattr(client, "_call_provider", fake_call)
    element = VisualElement(
        element_id="formula_001",
        type="formula",
        needs_enrichment=True,
        page=1,
        surrounding_text=["context"],
    )
    result = client.enrich_element(paper_id="p1", element=element, image_bytes=b"image-bytes")
    assert result.status == "completed"
    assert result.result["latex"]
    assert result.cached is False


def test_caching(tmp_path, monkeypatch):
    store = LocalStorage(tmp_path)
    client = LunaClient(settings=_settings(), storage=store)
    calls = {"n": 0}

    def fake_call(prompt, image_bytes):
        calls["n"] += 1
        return mock_enrichment("formula"), {"total_tokens": 3}

    monkeypatch.setattr(client, "_call_provider", fake_call)
    element = VisualElement(element_id="formula_001", type="formula", needs_enrichment=True)
    first = client.enrich_element(paper_id="p1", element=element, image_bytes=b"same")
    second = client.enrich_element(paper_id="p1", element=element, image_bytes=b"same")
    assert first.cached is False
    assert second.cached is True
    assert calls["n"] == 1


def test_invalid_response_handling(tmp_path, monkeypatch):
    store = LocalStorage(tmp_path)
    client = LunaClient(settings=_settings(luna_max_retries=0), storage=store)

    def bad_call(prompt, image_bytes):
        # Wrong types should fail FormulaEnrichmentResult validation.
        return {"latex": 123, "transcription_confidence": "nope"}, None

    monkeypatch.setattr(client, "_call_provider", bad_call)
    element = VisualElement(element_id="formula_001", type="formula", needs_enrichment=True)
    result = client.enrich_element(paper_id="p1", element=element, image_bytes=b"x")
    assert result.status == "failed"
    assert result.error


def test_retry_logic(tmp_path, monkeypatch):
    store = LocalStorage(tmp_path)
    client = LunaClient(settings=_settings(luna_max_retries=2), storage=store)
    attempts = {"n": 0}

    def flaky(prompt, image_bytes):
        attempts["n"] += 1
        if attempts["n"] < 2:
            raise RuntimeError("transient")
        return mock_enrichment("figure"), {"total_tokens": 1}

    monkeypatch.setattr(client, "_call_provider", flaky)
    element = VisualElement(element_id="figure_001", type="figure")
    result = client.enrich_element(paper_id="p1", element=element, image_bytes=b"img")
    assert result.status == "completed"
    assert attempts["n"] == 2


def test_richer_figure_components_are_normalized(tmp_path, monkeypatch):
    client = LunaClient(
        settings=_settings(luna_max_retries=0),
        storage=LocalStorage(tmp_path),
    )

    def rich_call(prompt, image_bytes):
        return {
            "visual_type": "architecture diagram",
            "description": "A model pipeline.",
            "main_message": "Features feed a classifier.",
            "components": [{"name": "Encoder", "description": "Creates features"}],
            "relationships": "Encoder feeds classifier",
            "evidence_from_caption": [],
            "uncertainties": [],
            "confidence": 0.9,
        }, {"total_tokens": 12}

    monkeypatch.setattr(client, "_call_provider", rich_call)
    result = client.enrich_element(
        paper_id="p1",
        element=VisualElement(element_id="figure_001", type="figure"),
        image_bytes=b"img",
    )

    assert result.status == "completed"
    assert result.result["components"] == ["name: Encoder; description: Creates features"]
    assert result.result["relationships"] == ["Encoder feeds classifier"]


def test_no_call_when_disabled(monkeypatch):
    client = LunaClient(settings=_settings(allow_external_api=False, luna_enabled=True))
    called = {"n": 0}

    def boom(prompt, image_bytes):
        called["n"] += 1
        raise AssertionError("should not call provider")

    monkeypatch.setattr(client, "_call_provider", boom)
    with pytest.raises(LunaDisabledError):
        client.enrich_element(
            paper_id="p",
            element=VisualElement(element_id="formula_001", type="formula"),
            image_bytes=b"x",
        )
    assert called["n"] == 0
