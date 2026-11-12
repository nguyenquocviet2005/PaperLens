"""API tests for PaperLens FastAPI app."""

from __future__ import annotations


import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.infrastructure.storage import LocalStorage


@pytest.fixture()
def client(tmp_path, monkeypatch):
    settings = Settings(
        _env_file=None,
        auth_enabled=False,
        app_env="development",
        storage_backend="local",
        local_storage_root=tmp_path,
        luna_enabled=False,
        allow_external_api=False,
        max_pdf_size_mb=1,
        docling_ocr_mode="off",
    )
    get_settings.cache_clear()
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_STORAGE_ROOT", str(tmp_path))
    monkeypatch.setenv("LUNA_ENABLED", "false")
    monkeypatch.setenv("ALLOW_EXTERNAL_API", "false")
    monkeypatch.setenv("MAX_PDF_SIZE_MB", "1")

    def _settings() -> Settings:
        return settings

    app.dependency_overrides[get_settings] = _settings
    # Also patch get_storage used by routes dependency
    from app import routes

    def _storage_dep(settings: Settings = None):
        return LocalStorage(tmp_path)

    app.dependency_overrides[routes.storage_dep] = lambda: LocalStorage(tmp_path)
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
    get_settings.cache_clear()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_capabilities_are_secret_free(client):
    response = client.get("/capabilities")
    assert response.status_code == 200
    body = response.json()
    assert body["reader"] is True
    assert body["visual_enrichment"] is False
    assert "api_key" not in str(body).lower()


def test_unified_agent_greeting_has_no_fake_confidence(client, monkeypatch):
    monkeypatch.setattr(
        "app.routes.run_agent",
        lambda *args, **kwargs: {
            "conversation_id": "conversation-1",
            "answer": "Hi! How can I help with your research?",
            "grounded": False,
            "citations": [],
            "tool_calls": [],
            "artifacts": [],
        },
    )
    response = client.post(
        "/agent",
        json={"message": "Hi!", "selected_papers": []},
    )

    assert response.status_code == 200
    assert response.json()["grounded"] is False
    assert "confidence" not in response.json()


def test_unified_agent_stream_emits_tokens_and_done(client, monkeypatch):
    def fake_stream(*args, **kwargs):
        yield {"type": "start", "conversation_id": "conversation-1"}
        yield {"type": "token", "content": "**Hello**"}
        yield {
            "type": "done",
            "conversation_id": "conversation-1",
            "answer": "**Hello**",
            "grounded": False,
            "citations": [],
            "tool_calls": [],
            "artifacts": [],
        }

    monkeypatch.setattr("app.routes.stream_agent", fake_stream)
    response = client.post(
        "/agent/stream",
        json={"message": "Hi!", "selected_papers": []},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert '"type": "token"' in response.text
    assert "**Hello**" in response.text


def test_cloud_ui_cors_preflight(client):
    response = client.options(
        "/papers",
        headers={
            "Origin": "https://paperlens.ailearn-vaic.cloud",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "https://paperlens.ailearn-vaic.cloud"


def test_foreign_origin_cors_preflight_rejected(client):
    response = client.options(
        "/papers",
        headers={
            "Origin": "https://paperlens-ui-uopctebpeq-as.a.run.app",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert "access-control-allow-origin" not in response.headers


def test_invalid_extension(client):
    response = client.post(
        "/papers",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400
    assert "pdf" in response.json()["detail"].lower()


def test_invalid_pdf_signature(client):
    response = client.post(
        "/papers",
        files={"file": ("fake.pdf", b"not-a-pdf", "application/pdf")},
    )
    assert response.status_code == 400
    assert "signature" in response.json()["detail"].lower()


def test_empty_upload(client):
    response = client.post(
        "/papers",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_oversized_upload(client):
    payload = b"%PDF" + b"x" * (2 * 1024 * 1024)
    response = client.post(
        "/papers",
        files={"file": ("big.pdf", payload, "application/pdf")},
    )
    assert response.status_code == 400
    assert "max_pdf_size" in response.json()["detail"].lower() or "exceeds" in response.json()["detail"].lower()


def test_unknown_paper_id(client):
    response = client.get("/papers/does-not-exist")
    assert response.status_code == 404


def test_luna_endpoint_disabled_by_default(client, tmp_path):
    # Seed minimal meta so enrich route can resolve paper_id
    store = LocalStorage(tmp_path)
    store.save_json(
        "normalized/papers/p1/meta.json",
        {
            "paper_id": "p1",
            "filename": "x.pdf",
            "status": "completed",
            "parse_status": "SUCCESS",
            "artifacts": {},
        },
    )
    store.save_json(
        "normalized/papers/p1/assets_manifest.json",
        {"tables": [], "figures": [], "formulas": []},
    )
    response = client.post("/papers/p1/enrich", json={"element_types": ["formula"]})
    assert response.status_code == 200
    body = response.json()
    assert body["luna_enabled"] is False
    assert body["allow_external_api"] is False
    assert "disabled" in (body.get("message") or "").lower()


def test_document_retrieval(client, tmp_path):
    store = LocalStorage(tmp_path)
    store.save_json(
        "normalized/papers/p2/meta.json",
        {
            "paper_id": "p2",
            "filename": "x.pdf",
            "status": "completed",
            "parse_status": "SUCCESS",
            "pages": 1,
            "artifacts": {},
        },
    )
    store.save_json(
        "normalized/papers/p2/paper_document.json",
        {
            "paper_id": "p2",
            "filename": "x.pdf",
            "page_count": 1,
            "text_elements": [],
            "tables": [],
            "figures": [],
            "formulas": [],
            "sections": [],
        },
    )
    response = client.get("/papers/p2/document")
    assert response.status_code == 200
    assert response.json()["paper_id"] == "p2"


def test_private_asset_content_proxy(client, tmp_path):
    store = LocalStorage(tmp_path)
    store.save_json(
        "normalized/papers/p3/meta.json",
        {
            "paper_id": "p3",
            "filename": "x.pdf",
            "status": "completed",
            "parse_status": "SUCCESS",
            "pages": 1,
            "artifacts": {},
        },
    )
    image_key = "assets/papers/p3/figures/figure_001.png"
    store.save_bytes(image_key, b"\x89PNG\r\n\x1a\nmock", content_type="image/png")
    store.save_json(
        "normalized/papers/p3/assets_manifest.json",
        {
            "tables": [],
            "formulas": [],
            "figures": [
                {
                    "element_id": "figure_001",
                    "type": "figure",
                    "page": 1,
                    "image_uri": image_key,
                }
            ],
        },
    )
    response = client.get("/papers/p3/assets/figure/figure_001/content")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content.startswith(b"\x89PNG")


def test_valid_pdf_upload_mocked(client, monkeypatch, tmp_path):
    from app.ingestion import pipeline
    from app.schemas import ArtifactPaths, IngestionResponse

    def fake_ingest(data, filename, settings=None, storage=None, paper_id=None):
        return IngestionResponse(
            paper_id="mock-id",
            filename=filename,
            status="completed",
            parse_status="SUCCESS",
            pages=1,
            text_elements=1,
            tables=0,
            pictures=0,
            formulas=0,
            artifacts=ArtifactPaths(raw_pdf="raw/papers/mock-id/source.pdf"),
        )

    monkeypatch.setattr(pipeline, "ingest_pdf_bytes", fake_ingest)
    # routes imports ingest_pdf_bytes directly
    monkeypatch.setattr("app.routes.ingest_pdf_bytes", fake_ingest)

    response = client.post(
        "/papers",
        files={"file": ("paper.pdf", b"%PDF-1.4 mock", "application/pdf")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["paper_id"] == "mock-id"
    assert body["status"] == "completed"
