"""End-to-end paper ingestion pipeline."""

from __future__ import annotations

import json
import logging
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import select

from app.ingestion.assets import extract_assets
from app.ingestion.chunking import ChunkingResult, chunk_paper_document, persist_chunks
from app.ingestion.cleaner import CleaningResult, audit_to_csv, build_audit_and_clean
from app.config import Settings, get_settings
from app.db.models import Job
from app.db.repository import PaperRepository
from app.db.session import session_scope
from app.rag.embeddings import index_paper_chunks
from app.ingestion.parser import DoclingParser, ParseResult, is_pdf_bytes, sanitize_filename
from app.schemas import ArtifactPaths, IngestionResponse, PaperDocument, VisualElement
from app.infrastructure.storage import (
    StorageBackend,
    get_storage,
    paper_meta_key,
    paper_normalized_key,
    paper_parsed_key,
    paper_raw_pdf_key,
)
from app.infrastructure.observability import emit_metrics
from app.ingestion.luna import LunaClient
from app.ingestion.text_enrichment import enrich_cleaned_text

logger = logging.getLogger(__name__)

_VISUAL_KINDS = ("tables", "figures", "formulas")


class IngestionError(Exception):
    """Raised for invalid uploads or failed ingestion."""


def _persist_start(
    cfg: Settings,
    *,
    paper_id: str,
    filename: str,
    storage_uri: str,
    user_id: str,
    existing_job_id: str | None = None,
) -> str | None:
    try:
        with session_scope(cfg) as session:
            repo = PaperRepository(session)
            repo.upsert_pending_paper(
                paper_id=paper_id,
                filename=filename,
                storage_uri=storage_uri,
                status="processing",
                parse_status="running",
                user_id=user_id,
            )
            job = session.get(Job, existing_job_id) if existing_job_id else None
            if job is not None:
                repo.update_job(job, status="running", progress=0.05)
            else:
                job = repo.create_job(paper_id=paper_id, job_type="ingest", status="running")
            return job.id
    except Exception as exc:
        logger.warning("Database unavailable during ingest start: %s", exc)
        return None


def _persist_failure(
    cfg: Settings,
    *,
    paper_id: str,
    filename: str,
    storage_uri: str,
    status: str,
    parse_status: str,
    error: str | None,
    artifacts: ArtifactPaths,
    warnings: list[str],
    page_count: int,
    job_id: str | None,
    user_id: str,
) -> None:
    try:
        with session_scope(cfg) as session:
            repo = PaperRepository(session)
            repo.mark_failed(
                paper_id,
                filename=filename,
                storage_uri=storage_uri,
                status=status,
                parse_status=parse_status,
                error=error,
                artifacts=artifacts,
                warnings=warnings,
                page_count=page_count,
                user_id=user_id,
            )
            job = session.get(Job, job_id) if job_id else session.scalar(
                select(Job)
                .where(Job.paper_id == paper_id, Job.status.in_(["queued", "running"]))
                .order_by(Job.created_at.desc())
                .limit(1)
            )
            if job is not None:
                repo.update_job(job, status="failed", progress=1.0, error=error)
    except Exception as exc:
        logger.warning("Failed to persist incomplete ingest to DB: %s", exc)


def _update_job_progress(cfg: Settings, job_id: str | None, progress: float) -> None:
    if not job_id:
        return
    try:
        with session_scope(cfg) as session:
            job = session.get(Job, job_id)
            if job is not None:
                PaperRepository(session).update_job(
                    job, status="running", progress=progress
                )
    except Exception:
        logger.debug("Could not update ingest progress", exc_info=True)


def _persist_success(
    cfg: Settings,
    *,
    paper_doc: PaperDocument,
    artifacts: ArtifactPaths,
    parse_status: str,
    job_id: str | None,
    user_id: str,
) -> None:
    try:
        with session_scope(cfg) as session:
            repo = PaperRepository(session)
            paper = repo.replace_document_graph(
                paper_doc, artifacts, user_id=user_id
            )
            paper.parse_status = parse_status
            paper.status = "completed"
            if job_id:
                job = session.get(Job, job_id)
                if job is not None:
                    repo.update_job(
                        job,
                        status="completed",
                        progress=1.0,
                        result={
                            "pages": paper_doc.page_count,
                            "tables": len(paper_doc.tables),
                            "figures": len(paper_doc.figures),
                            "formulas": len(paper_doc.formulas),
                        },
                    )
    except Exception as exc:
        logger.warning("Failed to persist completed ingest to DB: %s", exc)


@dataclass
class _IngestRun:
    """State shared by the steps of one ingestion run."""

    cfg: Settings
    store: StorageBackend
    paper_id: str
    filename: str
    raw_key: str
    user_id: str
    job_id: str | None = None
    artifacts: ArtifactPaths = field(default_factory=ArtifactPaths)
    warnings: list[str] = field(default_factory=list)
    report: dict[str, Any] = field(default_factory=dict)
    report_key: str = ""

    def save_report(self) -> None:
        self.store.save_json(self.report_key, self.report)

    def progress(self, value: float) -> None:
        _update_job_progress(self.cfg, self.job_id, value)


def _validate_upload(data: bytes, filename: str, cfg: Settings) -> str:
    """Reject bad uploads before anything is stored; returns the sanitized filename."""
    if not filename:
        raise IngestionError("Filename is required")
    safe_name = sanitize_filename(filename)
    if not safe_name.lower().endswith(".pdf"):
        raise IngestionError("File extension must be .pdf")
    if not data:
        raise IngestionError("Empty file rejected")
    if len(data) > cfg.max_pdf_size_bytes:
        raise IngestionError(f"PDF exceeds MAX_PDF_SIZE_MB={cfg.max_pdf_size_mb}")
    if not is_pdf_bytes(data):
        raise IngestionError("Invalid PDF signature")
    return safe_name


_DOCLING_EXPORTS = {
    "json": ("document.json", "docling_json"),
    "md": ("document.md", "docling_md"),
    "html": ("document.html", "docling_html"),
}


def _save_parse_outputs(run: _IngestRun, parse_result: ParseResult) -> None:
    """Store Docling exports and the parse report."""
    for fmt, (name, field_name) in _DOCLING_EXPORTS.items():
        content = parse_result.exports.get(fmt)
        if content:
            key = paper_parsed_key(run.paper_id, name)
            run.store.save_text(key, content)
            setattr(run.artifacts, field_name, key)

    run.report_key = paper_parsed_key(run.paper_id, "parse_report.json")
    run.save_report()
    run.artifacts.parse_report = run.report_key


def _reject_parse(run: _IngestRun, parse_result: ParseResult) -> IngestionResponse:
    """Record a parse that failed validation as failed/incomplete."""
    report = run.report
    validation = parse_result.validation
    fail_status = "failed" if parse_result.document is None else "incomplete"
    pages = int(report.get("document_page_count") or report.get("total_pdf_pages") or 0)

    run.store.save_json(
        paper_meta_key(run.paper_id),
        {
            "paper_id": run.paper_id,
            "filename": run.filename,
            "status": fail_status,
            "parse_status": validation.status,
            "artifacts": run.artifacts.model_dump(),
            "warnings": run.warnings,
            "error": validation.reason,
        },
    )
    _persist_failure(
        run.cfg,
        paper_id=run.paper_id,
        filename=run.filename,
        storage_uri=run.raw_key,
        status=fail_status,
        parse_status=validation.status,
        error=validation.reason,
        artifacts=run.artifacts,
        warnings=run.warnings,
        page_count=pages,
        job_id=run.job_id,
        user_id=run.user_id,
    )
    return IngestionResponse(
        paper_id=run.paper_id,
        filename=run.filename,
        status=fail_status,  # type: ignore[arg-type]
        parse_status=validation.status,
        pages=pages,
        text_elements=int(report.get("text_count") or 0),
        tables=int(report.get("table_count") or 0),
        pictures=int(report.get("picture_count") or 0),
        formulas=int(report.get("formula_count") or 0),
        artifacts=run.artifacts,
        warnings=run.warnings,
        error=validation.reason,
    )


def _save_cleaning_outputs(run: _IngestRun, cleaning: CleaningResult) -> None:
    """Store the element audit, cleaned JSONL and cleaned Markdown."""
    run.report["cleaning_statistics"] = cleaning.statistics
    run.save_report()

    audit_key = paper_normalized_key(run.paper_id, "element_audit.csv")
    run.store.save_text(audit_key, audit_to_csv(cleaning.audit_records))
    run.artifacts.element_audit = audit_key

    cleaned_lines = [json.dumps(row, ensure_ascii=False) for row in cleaning.cleaned_elements]
    cleaned_key = paper_normalized_key(run.paper_id, "cleaned_text.jsonl")
    run.store.save_text(cleaned_key, "\n".join(cleaned_lines) + ("\n" if cleaned_lines else ""))
    run.artifacts.cleaned_text = cleaned_key

    md_key = paper_normalized_key(run.paper_id, "cleaned_document.md")
    run.store.save_text(md_key, cleaning.cleaned_markdown)
    run.artifacts.cleaned_document = md_key


def _enrich_text(run: _IngestRun, paper_doc: PaperDocument) -> None:
    """Optional LLM summaries of cleaned sections; failures only add a warning."""
    if not (run.cfg.ingest_auto_text_enrich and run.cfg.text_enrichment_enabled):
        return
    try:
        sections = enrich_cleaned_text(paper_doc, settings=run.cfg, storage=run.store)
    except Exception as exc:
        paper_doc.warnings.append(f"text_enrichment_failed:{type(exc).__name__}")
        logger.warning("Optional cleaned text enrichment failed: %s", type(exc).__name__)
        return
    paper_doc.metadata["text_enrichment"] = {
        "source": "cleaned_docling_text",
        "sections": sections,
    }


def enrichment_targets(paper_doc: PaperDocument, *, force: bool = False) -> list[VisualElement]:
    """Visual elements worth a VLM request: everything with an image, except formulas
    Docling already transcribed (unless forced)."""
    targets: list[VisualElement] = []
    for el in [*paper_doc.tables, *paper_doc.figures, *paper_doc.formulas]:
        if not el.image_uri:
            continue
        if not force and el.type == "formula" and el.docling_text and not el.needs_enrichment:
            continue
        targets.append(el)
    return targets


def enrich_visuals(
    paper_doc: PaperDocument,
    *,
    settings: Settings,
    storage: StorageBackend,
    elements: list[VisualElement],
    force: bool = False,
) -> dict[str, str]:
    """Run Luna on `elements` in parallel, attach results in place; returns status per id."""
    from concurrent.futures import ThreadPoolExecutor

    client = LunaClient(settings=settings, storage=storage)

    def one(el: VisualElement) -> str:
        try:
            if not storage.exists(el.image_uri or ""):
                return "skipped:no_image"
            enrichment = client.enrich_element(
                paper_id=paper_doc.paper_id,
                element=el,
                image_bytes=storage.read_bytes(el.image_uri),
                force=force,
            )
        except Exception as exc:
            return f"failed:{type(exc).__name__}"
        el.enrichment = enrichment
        return enrichment.status

    with ThreadPoolExecutor(max(1, settings.luna_max_concurrency)) as pool:
        statuses = list(pool.map(one, elements))
    return {el.element_id: status for el, status in zip(elements, statuses)}


def _enrich_visuals(run: _IngestRun, paper_doc: PaperDocument) -> None:
    """Luna descriptions for tables/figures/formulas, so they reach the chunks and index."""
    if not run.cfg.ingest_auto_visual_enrich or not LunaClient(run.cfg).is_enabled:
        return
    statuses = enrich_visuals(
        paper_doc,
        settings=run.cfg,
        storage=run.store,
        elements=enrichment_targets(paper_doc),
    )
    failed = sorted(eid for eid, status in statuses.items() if status != "completed")
    emit_metrics({
        "VisualEnrichRequested": (len(statuses), "Count"),
        "VisualEnrichFailed": (len(failed), "Count"),
    })
    run.report["visual_enrichment"] = {
        "requested": len(statuses),
        "completed": len(statuses) - len(failed),
        "failed": failed,
    }
    if failed:
        paper_doc.warnings.append(f"visual_enrichment_incomplete:{len(failed)}/{len(statuses)}")


def _save_document_outputs(run: _IngestRun, paper_doc: PaperDocument) -> None:
    """Store the normalized PaperDocument and the visual assets manifest."""
    paper_key = paper_normalized_key(run.paper_id, "paper_document.json")
    run.store.save_json(paper_key, paper_doc.model_dump(mode="json"))
    run.artifacts.paper_document = paper_key

    manifest_key = paper_normalized_key(run.paper_id, "assets_manifest.json")
    run.store.save_json(
        manifest_key,
        {kind: [el.model_dump(mode="json") for el in getattr(paper_doc, kind)] for kind in _VISUAL_KINDS},
    )
    run.artifacts.assets_manifest = manifest_key


def _save_meta(run: _IngestRun, paper_doc: PaperDocument, parse_status: str) -> None:
    run.store.save_json(
        paper_meta_key(run.paper_id),
        {
            "paper_id": run.paper_id,
            "filename": run.filename,
            "status": "completed",
            "parse_status": parse_status,
            "pages": paper_doc.page_count,
            "title": paper_doc.title,
            "created_at": paper_doc.created_at.isoformat(),
            "artifacts": run.artifacts.model_dump(),
            "warnings": paper_doc.warnings,
            "counts": {
                "text_elements": len(paper_doc.text_elements),
                "tables": len(paper_doc.tables),
                "pictures": len(paper_doc.figures),
                "formulas": len(paper_doc.formulas),
            },
        },
    )


def _chunk_and_index(run: _IngestRun, paper_doc: PaperDocument) -> list[str]:
    """Chunk the paper and embed the chunks. Returns warnings instead of raising,
    because the paper itself is already stored and readable without an index."""
    try:
        chunk_result = chunk_paper_document(paper_doc)
        persist_chunks(run.paper_id, chunk_result, settings=run.cfg)
        run.store.save_json(
            paper_normalized_key(run.paper_id, "chunks_report.json"),
            _chunks_report(chunk_result),
        )
        run.report["chunking_statistics"] = chunk_result.metrics
        run.save_report()
    except Exception as exc:
        logger.warning("Chunking failed for %s: %s", run.paper_id, exc)
        return [f"chunking_failed: {exc}"]

    try:
        run.report["embedding_statistics"] = index_paper_chunks(run.paper_id, settings=run.cfg)
        run.save_report()
    except Exception as exc:
        logger.warning("Embedding index failed for %s: %s", run.paper_id, exc)
        return [f"embedding_failed: {exc}"]
    return []


def _chunks_report(chunk_result: ChunkingResult) -> dict[str, Any]:
    child_types = sorted({c.chunk_type for c in chunk_result.child_chunks})
    return {
        "metrics": chunk_result.metrics,
        "parent_preview": [
            {
                "section_path": p.section_path,
                "pages": [p.page_start, p.page_end],
                "tokens": p.token_count,
                "chars": len(p.content),
            }
            for p in chunk_result.parent_chunks[:20]
        ],
        "child_types": {
            t: sum(1 for c in chunk_result.child_chunks if c.chunk_type == t) for t in child_types
        },
    }


def _process_parsed_document(run: _IngestRun, parse_result: ParseResult) -> IngestionResponse:
    """Assets -> cleaning -> enrichment -> persist -> chunk/index for an accepted parse."""
    cfg, pid = run.cfg, run.paper_id
    parse_status = parse_result.validation.status

    assets = extract_assets(parse_result.document, paper_id=pid, storage=run.store, settings=cfg)
    run.warnings.extend(assets.pop("warnings", []))  # type: ignore[arg-type]

    cleaning = build_audit_and_clean(
        parse_result.document,
        paper_id=pid,
        filename=run.filename,
        parse_report=run.report,
        source_pdf_uri=run.raw_key,
        parse_report_uri=run.report_key,
        visual_elements={kind: assets[kind] for kind in _VISUAL_KINDS},
    )
    run.progress(0.75)
    _save_cleaning_outputs(run, cleaning)

    paper_doc: PaperDocument = cleaning.paper_document
    paper_doc.warnings = list(dict.fromkeys([*paper_doc.warnings, *run.warnings]))
    _enrich_visuals(run, paper_doc)
    run.progress(0.85)
    _enrich_text(run, paper_doc)
    _save_document_outputs(run, paper_doc)
    _save_meta(run, paper_doc, parse_status)
    _persist_success(
        cfg,
        paper_doc=paper_doc,
        artifacts=run.artifacts,
        parse_status=parse_status,
        job_id=run.job_id,
        user_id=run.user_id,
    )

    # Chunks reference the paper row, so they are written after _persist_success.
    index_warnings = _chunk_and_index(run, paper_doc)
    if index_warnings:
        paper_doc.warnings.extend(index_warnings)
        _save_meta(run, paper_doc, parse_status)

    logger.info("Ingested paper_id=%s pages=%s", pid, paper_doc.page_count)
    return IngestionResponse(
        paper_id=pid,
        filename=run.filename,
        status="completed",
        parse_status=parse_status,
        pages=paper_doc.page_count,
        text_elements=len(paper_doc.text_elements),
        tables=len(paper_doc.tables),
        pictures=len(paper_doc.figures),
        formulas=len(paper_doc.formulas),
        artifacts=run.artifacts,
        warnings=paper_doc.warnings,
    )


def refresh_visual_context(
    paper_doc: PaperDocument, *, settings: Settings, storage: StorageBackend
) -> int:
    """Recompute surrounding text and table data from the stored Docling JSON.

    Lets papers ingested before the current context selection benefit without a re-parse.
    Returns how many elements were updated (0 when the Docling JSON is missing).
    """
    from docling_core.types.doc import DoclingDocument

    from app.ingestion.assets import _embedder, _table_markdown
    from app.ingestion.surrounding import build_document_index, select_surrounding_text

    key = paper_parsed_key(paper_doc.paper_id, "document.json")
    if not storage.exists(key):
        return 0
    try:
        document = DoclingDocument.model_validate_json(storage.read_text(key))
    except Exception as exc:
        logger.warning("Stored Docling JSON unreadable for %s: %s", paper_doc.paper_id, type(exc).__name__)
        return 0
    index = build_document_index(document)
    embed = _embedder(settings)
    updated = 0
    for el in [*paper_doc.tables, *paper_doc.figures, *paper_doc.formulas]:
        if not el.source_ref or el.source_ref not in index.position:
            continue
        el.surrounding_text = select_surrounding_text(
            index,
            kind=el.type,
            caption=el.caption,
            source_ref=el.source_ref,
            page=el.page,
            budget_tokens=settings.surrounding_text_max_tokens,
            embed=embed,
        )
        if el.type == "table":
            el.docling_text = _table_markdown(index.items[index.position[el.source_ref]], document)
        updated += 1
    return updated


def rebuild_paper_index(
    paper_id: str,
    *,
    settings: Settings | None = None,
    storage: StorageBackend | None = None,
    refresh_context: bool = True,
    enrich: bool = False,
) -> dict[str, Any]:
    """Re-chunk and re-index a stored paper with the current chunking and context rules."""
    cfg = settings or get_settings()
    store = storage or get_storage(cfg)
    paper_key = paper_normalized_key(paper_id, "paper_document.json")
    paper_doc = PaperDocument.model_validate(store.read_json(paper_key))
    summary: dict[str, Any] = {"paper_id": paper_id}
    if refresh_context:
        summary["context_refreshed"] = refresh_visual_context(paper_doc, settings=cfg, storage=store)
    if enrich and LunaClient(cfg).is_enabled:
        statuses = enrich_visuals(
            paper_doc, settings=cfg, storage=store, elements=enrichment_targets(paper_doc)
        )
        summary["enriched"] = sum(1 for s in statuses.values() if s == "completed")
    store.save_json(paper_key, paper_doc.model_dump(mode="json"))
    store.save_json(
        paper_normalized_key(paper_id, "assets_manifest.json"),
        {kind: [el.model_dump(mode="json") for el in getattr(paper_doc, kind)] for kind in _VISUAL_KINDS},
    )
    result = chunk_paper_document(paper_doc)
    persist_chunks(paper_id, result, settings=cfg)
    store.save_json(paper_normalized_key(paper_id, "chunks_report.json"), _chunks_report(result))
    summary["chunks"] = result.metrics["total_chunks"]
    summary["index"] = index_paper_chunks(paper_id, settings=cfg, force=True)
    return summary


def _ingest_pdf_bytes_impl(
    data: bytes,
    *,
    filename: str,
    settings: Settings | None = None,
    storage: StorageBackend | None = None,
    paper_id: str | None = None,
    user_id: str = "local-user",
    job_id: str | None = None,
) -> IngestionResponse:
    """Validate, store, parse, clean, and export a research PDF."""
    cfg = settings or get_settings()
    store = storage or get_storage(cfg)
    safe_name = _validate_upload(data, filename, cfg)

    pid = paper_id or str(uuid.uuid4())
    raw_key = paper_raw_pdf_key(pid)
    store.save_bytes(raw_key, data, content_type="application/pdf")

    run = _IngestRun(
        cfg=cfg,
        store=store,
        paper_id=pid,
        filename=safe_name,
        raw_key=raw_key,
        user_id=user_id,
        artifacts=ArtifactPaths(raw_pdf=raw_key),
    )
    run.job_id = _persist_start(
        cfg,
        paper_id=pid,
        filename=safe_name,
        storage_uri=raw_key,
        user_id=user_id,
        existing_job_id=job_id,
    )

    # Keep the temp PDF alive until processing ends: the Docling document may still read it.
    with tempfile.TemporaryDirectory(prefix="paperlens-") as tmp:
        local_pdf = Path(tmp) / safe_name
        local_pdf.write_bytes(data)

        parse_result = DoclingParser(cfg).convert(local_pdf)
        run.progress(0.45)
        run.report = dict(parse_result.parse_report)
        run.warnings.extend(run.report.get("warnings") or [])
        _save_parse_outputs(run, parse_result)

        if not parse_result.validation.accepted or parse_result.document is None:
            return _reject_parse(run, parse_result)
        return _process_parsed_document(run, parse_result)


def _ingest_metrics(started: float, status: str, *, pages: int, size: int) -> None:
    emit_metrics(
        {
            "Ingests": (1, "Count"),
            "IngestFailures": (0 if status == "completed" else 1, "Count"),
            "IngestDuration": (round(time.perf_counter() - started, 2), "Seconds"),
            "IngestPages": (pages, "Count"),
        },
        dimensions={"Status": status},
        properties={"pdf_bytes": size},
    )


def ingest_pdf_bytes(
    data: bytes,
    *,
    filename: str,
    settings: Settings | None = None,
    storage: StorageBackend | None = None,
    paper_id: str | None = None,
    user_id: str = "local-user",
    job_id: str | None = None,
) -> IngestionResponse:
    """Run ingestion and persist an explicit failure state on unexpected errors."""
    cfg = settings or get_settings()
    store = storage or get_storage(cfg)
    pid = paper_id or str(uuid.uuid4())
    started = time.perf_counter()
    try:
        response = _ingest_pdf_bytes_impl(
            data,
            filename=filename,
            settings=cfg,
            storage=store,
            paper_id=pid,
            user_id=user_id,
            job_id=job_id,
        )
        _ingest_metrics(started, response.status, pages=response.pages, size=len(data))
        return response
    except IngestionError:
        _ingest_metrics(started, "rejected", pages=0, size=len(data))
        raise
    except Exception as exc:
        _ingest_metrics(started, "error", pages=0, size=len(data))
        safe_name = sanitize_filename(filename) if filename else "upload.pdf"
        raw_key = paper_raw_pdf_key(pid)
        artifacts = ArtifactPaths()
        try:
            if store.exists(raw_key):
                artifacts.raw_pdf = raw_key
            store.save_json(
                paper_meta_key(pid),
                {
                    "paper_id": pid,
                    "filename": safe_name,
                    "status": "failed",
                    "parse_status": "error",
                    "artifacts": artifacts.model_dump(),
                    "warnings": [],
                    "error": f"{type(exc).__name__}: processing failed",
                },
            )
        except Exception as storage_exc:
            logger.warning("Could not persist failed ingest metadata: %s", type(storage_exc).__name__)
        _persist_failure(
            cfg,
            paper_id=pid,
            filename=safe_name,
            storage_uri=raw_key,
            status="failed",
            parse_status="error",
            error=f"{type(exc).__name__}: processing failed",
            artifacts=artifacts,
            warnings=[],
            page_count=0,
            job_id=None,
            user_id=user_id,
        )
        raise
