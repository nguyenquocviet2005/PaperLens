"""Characterization tests for the ingestion pipeline's success and rejection paths.

Docling, asset cropping and cleaning are replaced with fakes so the orchestration
(storage layout, report updates, DB state, warnings) is tested without heavy models.
Chunking and the SQLite repository are real.
"""

from __future__ import annotations

import pytest
from sqlalchemy import func, select

from app.config import Settings
from app.db.models import Job, Paper, PaperChunk
from app.db.session import init_db, reset_engine, session_scope
from app.infrastructure.storage import LocalStorage
from app.ingestion import pipeline
from app.ingestion.cleaner import CleaningResult
from app.ingestion.parser import ParseResult, ParseValidation
from app.ingestion.pipeline import IngestionError, ingest_pdf_bytes
from app.schemas import PaperDocument, TextElement

PDF = b"%PDF-1.4\nsynthetic"
PID = "paper-ok"


@pytest.fixture
def env(tmp_path, monkeypatch):
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}",
        storage_backend="local",
        local_storage_root=tmp_path / "objects",
        docling_ocr_mode="off",
        ingest_auto_text_enrich=False,
    )
    reset_engine()
    init_db(settings)
    storage = LocalStorage(tmp_path / "objects")

    state = {"accepted": True, "index_error": None}

    def fake_convert(self, pdf_path):
        assert pdf_path.read_bytes() == PDF
        accepted = state["accepted"]
        return ParseResult(
            conversion=None,
            document=object(),
            validation=ParseValidation(
                accepted=accepted,
                status="SUCCESS" if accepted else "PARTIAL_SUCCESS",
                reason=None if accepted else "pages missing",
            ),
            parse_report={
                "warnings": ["parse_warn"],
                "document_page_count": 2,
                "text_count": 3,
                "table_count": 1,
                "picture_count": 0,
                "formula_count": 0,
            },
            elapsed_seconds=0.1,
            exports={"json": "{}", "md": "# Doc", "html": None},
            ocr_enabled=False,
        )

    def fake_assets(document, *, paper_id, storage, settings):
        return {"tables": [], "figures": [], "formulas": [], "warnings": ["asset_warn"]}

    def fake_clean(document, *, paper_id, filename, parse_report, source_pdf_uri,
                   parse_report_uri, visual_elements):
        words = " ".join(f"word{i}" for i in range(400))
        doc = PaperDocument(
            paper_id=paper_id,
            filename=filename,
            title="A Paper",
            page_count=2,
            text_elements=[
                TextElement(element_id="el_0001", order=1, page=1,
                            section_path=["Intro"], type="TextItem", text=words),
            ],
            warnings=["clean_warn"],
        )
        return CleaningResult(
            audit_records=[],
            cleaned_elements=[{"element_id": "el_0001"}],
            paper_document=doc,
            statistics={"kept": 1},
            cleaned_markdown="# A Paper",
        )

    def fake_index(paper_id, *, settings):
        if state["index_error"]:
            raise state["index_error"]
        return {"embedded": 1}

    monkeypatch.setattr(pipeline.DoclingParser, "convert", fake_convert)
    monkeypatch.setattr(pipeline, "extract_assets", fake_assets)
    monkeypatch.setattr(pipeline, "build_audit_and_clean", fake_clean)
    monkeypatch.setattr(pipeline, "index_paper_chunks", fake_index)

    yield settings, storage, state
    reset_engine()


def _ingest(settings, storage):
    return ingest_pdf_bytes(PDF, filename="My Paper.pdf", settings=settings,
                            storage=storage, paper_id=PID, user_id="u1")


def test_success_writes_artifacts_db_and_report(env):
    settings, storage, _ = env
    result = _ingest(settings, storage)

    assert result.status == "completed"
    assert result.parse_status == "SUCCESS"
    assert result.pages == 2
    assert result.text_elements == 1
    assert result.warnings == ["clean_warn", "parse_warn", "asset_warn"]

    art = result.artifacts
    assert art.docling_json and art.docling_md and art.docling_html is None
    for key in (art.raw_pdf, art.docling_json, art.docling_md, art.parse_report,
                art.element_audit, art.cleaned_text, art.cleaned_document,
                art.paper_document, art.assets_manifest):
        assert storage.exists(key), key

    report = storage.read_json(art.parse_report)
    assert report["cleaning_statistics"] == {"kept": 1}
    assert "chunking_statistics" in report
    assert report["embedding_statistics"] == {"embedded": 1}

    meta = storage.read_json(f"normalized/papers/{PID}/meta.json")
    assert meta["status"] == "completed"
    assert meta["title"] == "A Paper"
    assert meta["counts"]["text_elements"] == 1
    assert storage.exists(f"normalized/papers/{PID}/chunks_report.json")

    with session_scope(settings) as session:
        paper = session.get(Paper, PID)
        assert paper.status == "completed" and paper.parse_status == "SUCCESS"
        assert paper.user_id == "u1"
        job = session.scalar(select(Job).where(Job.paper_id == PID))
        assert job.status == "completed" and job.progress == 1.0
        assert job.result["pages"] == 2
        chunks = session.scalar(
            select(func.count()).select_from(PaperChunk).where(PaperChunk.paper_id == PID)
        )
        assert chunks > 0


def test_rejected_parse_is_recorded_as_incomplete(env):
    settings, storage, state = env
    state["accepted"] = False
    result = _ingest(settings, storage)

    assert result.status == "incomplete"
    assert result.parse_status == "PARTIAL_SUCCESS"
    assert result.error == "pages missing"
    assert result.pages == 2 and result.tables == 1
    assert result.artifacts.cleaned_text is None

    meta = storage.read_json(f"normalized/papers/{PID}/meta.json")
    assert meta["status"] == "incomplete"
    with session_scope(settings) as session:
        assert session.get(Paper, PID).status == "incomplete"
        job = session.scalar(select(Job).where(Job.paper_id == PID))
        assert job.status == "failed" and job.error == "pages missing"


def test_embedding_failure_is_reported_but_ingest_completes(env):
    settings, storage, state = env
    state["index_error"] = RuntimeError("embedder down")
    result = _ingest(settings, storage)

    assert result.status == "completed"
    assert "embedding_failed: embedder down" in result.warnings


@pytest.mark.parametrize(
    ("filename", "data", "message"),
    [
        ("", PDF, "Filename is required"),
        ("paper.txt", PDF, "extension"),
        ("paper.pdf", b"", "Empty"),
        ("paper.pdf", b"not a pdf", "signature"),
    ],
)
def test_invalid_uploads_are_rejected(env, filename, data, message):
    settings, storage, _ = env
    with pytest.raises(IngestionError, match=message):
        ingest_pdf_bytes(data, filename=filename, settings=settings, storage=storage)


def test_luna_runs_in_pipeline_and_reaches_the_index(env, monkeypatch):
    from app.ingestion.chunking import MODEL_TEXT_LABEL
    from app.ingestion.luna import LunaClient
    from app.schemas import VisualElement

    settings, storage, _ = env
    settings = settings.model_copy(update={
        "luna_enabled": True, "allow_external_api": True, "luna_api_key": "k",
        "luna_model": "vlm", "ingest_auto_visual_enrich": True,
        "luna_request_delay_seconds": 0, "luna_max_retries": 0,
    })
    real_clean = pipeline.build_audit_and_clean

    def clean_with_table(document, **kwargs):
        result = real_clean(document, **kwargs)
        image_key = f"assets/papers/{PID}/tables/table_001.png"
        storage.save_bytes(image_key, b"png-bytes", content_type="image/png")
        result.paper_document.tables = [VisualElement(
            element_id="table_001", type="table", page=1, caption="Table 1: Sizes.",
            image_uri=image_key, docling_text="| a | b |\n| --- | --- |\n| 1 | 2 |",
        )]
        return result

    calls = []

    def fake_provider(self, *, prompt, image_bytes):
        calls.append(image_bytes)
        return {"table_purpose": "Compares sizes", "columns": ["a", "b"],
                "main_results": ["a is 1"], "confidence": 0.9}, None

    monkeypatch.setattr(pipeline, "build_audit_and_clean", clean_with_table)
    monkeypatch.setattr(LunaClient, "_call_provider", fake_provider)
    result = _ingest(settings, storage)

    assert result.status == "completed"
    assert calls == [b"png-bytes"]
    report = storage.read_json(result.artifacts.parse_report)
    assert report["visual_enrichment"] == {"requested": 1, "completed": 1, "failed": []}
    manifest = storage.read_json(result.artifacts.assets_manifest)
    assert manifest["tables"][0]["enrichment"]["status"] == "completed"
    with session_scope(settings) as session:
        table_chunk = session.scalar(
            select(PaperChunk).where(PaperChunk.paper_id == PID, PaperChunk.chunk_type == "table")
        )
        assert f"{MODEL_TEXT_LABEL} Compares sizes" in table_chunk.content
        assert "[Table data]" in table_chunk.content


def test_luna_is_not_called_when_auto_enrich_is_off(env, monkeypatch):
    from app.ingestion.luna import LunaClient

    settings, storage, _ = env
    settings = settings.model_copy(update={
        "luna_enabled": True, "allow_external_api": True, "luna_api_key": "k",
        "luna_model": "vlm", "ingest_auto_visual_enrich": False,
    })

    def boom(self, **kwargs):
        raise AssertionError("Luna must not be called")

    monkeypatch.setattr(LunaClient, "_call_provider", boom)
    assert _ingest(settings, storage).status == "completed"


def test_rebuild_paper_index_rechunks_a_stored_paper(env, monkeypatch):
    settings, storage, _ = env
    _ingest(settings, storage)
    indexed = []
    monkeypatch.setattr(pipeline, "index_paper_chunks",
                        lambda paper_id, *, settings, force=False: indexed.append(force) or {"ok": 1})

    summary = pipeline.rebuild_paper_index(PID, settings=settings, storage=storage)

    assert summary["context_refreshed"] == 0  # the fake Docling JSON has no items
    assert summary["chunks"] > 0
    assert indexed == [True]
    with session_scope(settings) as session:
        kinds = set(session.scalars(select(PaperChunk.chunk_type).where(PaperChunk.paper_id == PID)))
    assert kinds == {"parent_passage", "text"}


def test_reindex_cli_reports_each_paper(env, monkeypatch, capsys):
    from app import cli

    settings, storage, _ = env
    calls = []
    monkeypatch.setattr("app.config.get_settings", lambda: settings)
    monkeypatch.setattr("app.ingestion.pipeline.rebuild_paper_index",
                        lambda pid, **kw: calls.append((pid, kw)) or {"paper_id": pid, "chunks": 3})
    assert cli.main(["reindex", "--paper-id", "p1", "--paper-id", "p2", "--enrich"]) == 0
    assert [c[0] for c in calls] == ["p1", "p2"]
    assert calls[0][1]["enrich"] is True and calls[0][1]["refresh_context"] is True
    assert "Re-indexed 2/2 papers" in capsys.readouterr().out
