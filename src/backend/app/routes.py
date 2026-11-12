"""FastAPI route handlers for PaperLens."""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Response, UploadFile
from fastapi.responses import StreamingResponse

from app.config import Settings, get_settings
from app.auth import CurrentUser, current_user
from app.db.agent_repository import AgentConversationRepository
from app.db.repository import PaperRepository
from app.db.session import session_scope
from app.guest import consume_guest_quota, create_guest_session, guest_trial_available
from app.ingestion.luna import LunaClient, LunaDisabledError
from app.ingestion.chunking import chunk_paper_document, persist_chunks
from app.ingestion.pipeline import IngestionError, ingest_pdf_bytes
from app.schemas import (
    AgentConversationResponse,
    AgentConversationListResponse,
    AgentConversationSummary,
    AgentConversationTurn,
    AgentRequest,
    AgentResponse,
    ArtifactPaths,
    AssetManifest,
    AuthMeResponse,
    CompareRequest,
    DeletePaperResponse,
    DiscoverRequest,
    EnrichRequest,
    EnrichResponse,
    EnrichResultItem,
    GuestQuota,
    GuestSessionResponse,
    HealthResponse,
    IndexRequest,
    IngestionResponse,
    JobStatusResponse,
    PaperDocument,
    PaperLibraryItem,
    PaperLibraryResponse,
    PaperMetadataResponse,
    QARequest,
    ResearchRequest,
    RetrieveRequest,
    VisualElement,
)
from app.harness.agent import get_agent_conversation, run_agent, stream_agent
from app.research.compare import compare_papers
from app.infrastructure.accelerator import accelerator_status
from app.research.discovery import DiscoveryError, discover_papers, find_library_duplicates
from app.rag.embeddings import index_paper_chunks
from app.harness.guardrails import GuardrailError, validate_agent_input
from app.rag.qa import answer_paper_question
from app.research.research_graph import run_research
from app.rag.retrieval import retrieve
from app.ingestion.text_enrichment import enrich_cleaned_text

from app.infrastructure.storage import (
    ObjectNotFoundError,
    StorageBackend,
    get_storage,
    paper_meta_key,
    paper_normalized_key,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def storage_dep(settings: Settings = Depends(get_settings)) -> StorageBackend:
    return get_storage(settings)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/capabilities")
def capabilities(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Expose non-secret feature flags for clients."""
    return {
        "ingestion": True,
        "reader": True,
        "retrieval": True,
        "qa_mode": (
            "langchain_openai_grounded"
            if settings.llm_enabled and settings.allow_external_api
            else "extractive"
        ),
        "discovery": True,
        "comparison": True,
        "research_workflow": True,
        "research_orchestrator": "langgraph",
        "unified_research_agent": True,
        "agent_reasoning": {
            "enabled": settings.agent_reasoning_enabled,
            "effort": settings.agent_reasoning_effort,
            "transport": "responses_api",
            "private_chain_of_thought_exposed": False,
        },
        "account_auth": settings.auth_enabled,
        "guest_trial": {
            "enabled": guest_trial_available(settings),
            "max_queries": settings.guest_max_queries,
            "max_papers": settings.guest_max_papers,
            "max_images": settings.guest_max_images,
            "session_ttl_hours": settings.guest_session_ttl_hours,
        },
        "visual_enrichment": settings.luna_enabled and settings.allow_external_api,
        "text_enrichment": (
            settings.text_enrichment_enabled and settings.allow_external_api
        ),
        "embedding_provider": settings.embedding_provider,
        "docling_accelerator": accelerator_status(
            settings.docling_accelerator_device
        ),
    }


@router.post("/auth/guest", response_model=GuestSessionResponse)
def create_guest_auth(settings: Settings = Depends(get_settings)) -> GuestSessionResponse:
    """Issue a browser-bound anonymous trial session with hard usage quotas."""
    access_token, user_id, snapshot = create_guest_session(settings)
    return GuestSessionResponse(
        access_token=access_token,
        expires_at=int(snapshot.expires_at.timestamp()),
        user={
            "id": user_id,
            "email": None,
            "is_guest": True,
        },
        guest_quota=GuestQuota.model_validate(snapshot.as_dict()),
    )


@router.get("/auth/me", response_model=AuthMeResponse)
def auth_me(user: CurrentUser = Depends(current_user)) -> AuthMeResponse:
    quota = None
    if user.is_guest and user.guest_quota is not None:
        quota = GuestQuota.model_validate(user.guest_quota.as_dict())
    return AuthMeResponse(
        user_id=user.user_id,
        email=user.email,
        is_guest=user.is_guest,
        guest_quota=quota,
    )


def _consume_guest_if_needed(
    user: CurrentUser,
    settings: Settings,
    *,
    queries: int = 0,
    papers: int = 0,
    images: int = 0,
) -> GuestQuota | None:
    if not user.is_guest:
        return None
    snapshot = consume_guest_quota(
        user.user_id,
        settings=settings,
        queries=queries,
        papers=papers,
        images=images,
    )
    return GuestQuota.model_validate(snapshot.as_dict())


@router.get("/papers", response_model=PaperLibraryResponse)
def list_papers(
    q: str | None = None,
    status: str | None = None,
    author: str | None = None,
    year: int | None = None,
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    settings: Settings = Depends(get_settings),
    user: CurrentUser = Depends(current_user),
) -> PaperLibraryResponse:
    with session_scope(settings) as session:
        repo = PaperRepository(session)
        papers = repo.list_papers(
            q=q,
            status=status,
            author=author,
            year=year,
            limit=limit,
            offset=offset,
            user_id=user.user_id,
        )
        items = [
            PaperLibraryItem(
                paper_id=p.id,
                filename=p.filename,
                title=p.title,
                status=p.status,
                parse_status=p.parse_status,
                page_count=p.page_count,
                publication_year=p.publication_year,
                authors=p.authors,
                created_at=p.created_at,
                updated_at=p.updated_at,
                storage_uri=p.storage_uri,
            )
            for p in papers
        ]
    return PaperLibraryResponse(count=len(items), papers=items)


@router.post("/papers", response_model=IngestionResponse)
async def upload_paper(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    storage: StorageBackend = Depends(storage_dep),
    user: CurrentUser = Depends(current_user),
) -> IngestionResponse:
    filename = file.filename or ""
    data = await file.read()

    if settings.ingest_async:
        # Validate quickly, store raw PDF, return accepted, parse in background.
        from app.ingestion.parser import is_pdf_bytes, sanitize_filename
        from app.db.repository import PaperRepository
        from app.infrastructure.storage import paper_raw_pdf_key

        if not filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        safe_name = sanitize_filename(filename)
        if not safe_name.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="File extension must be .pdf")
        if not data:
            raise HTTPException(status_code=400, detail="Empty file rejected")
        if len(data) > settings.max_pdf_size_bytes:
            raise HTTPException(
                status_code=400, detail=f"PDF exceeds MAX_PDF_SIZE_MB={settings.max_pdf_size_mb}"
            )
        if not is_pdf_bytes(data):
            raise HTTPException(status_code=400, detail="Invalid PDF signature")

        _consume_guest_if_needed(user, settings, papers=1)
        paper_id = str(uuid.uuid4())
        raw_key = paper_raw_pdf_key(paper_id)
        storage.save_bytes(raw_key, data, content_type="application/pdf")
        job_id = None
        try:
            with session_scope(settings) as session:
                repo = PaperRepository(session)
                repo.upsert_pending_paper(
                    paper_id=paper_id,
                    filename=safe_name,
                    storage_uri=raw_key,
                    status="accepted",
                    parse_status="queued",
                    user_id=user.user_id,
                )
                job = repo.create_job(paper_id=paper_id, job_type="ingest", status="queued")
                job_id = job.id
        except Exception as exc:
            logger.warning("DB unavailable for async accept: %s", exc)

        background_tasks.add_task(
            ingest_pdf_bytes,
            data,
            filename=safe_name,
            settings=settings,
            storage=storage,
            paper_id=paper_id,
            user_id=user.user_id,
            job_id=job_id,
        )
        return IngestionResponse(
            paper_id=paper_id,
            filename=safe_name,
            status="accepted",
            parse_status="queued",
            artifacts=ArtifactPaths(raw_pdf=raw_key),
            job_id=job_id,
        )

    try:
        _consume_guest_if_needed(user, settings, papers=1)
        ingest_kwargs = {
            "filename": filename,
            "settings": settings,
            "storage": storage,
        }
        if settings.auth_enabled:
            ingest_kwargs["user_id"] = user.user_id
        return ingest_pdf_bytes(data, **ingest_kwargs)
    except HTTPException:
        raise
    except IngestionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion failed")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    settings: Settings = Depends(get_settings),
    user: CurrentUser = Depends(current_user),
) -> JobStatusResponse:
    from sqlalchemy import select
    from app.db.models import Job, Paper

    with session_scope(settings) as session:
        stmt = (
            select(Job)
            .join(Paper, Paper.id == Job.paper_id)
            .where(Job.id == job_id, Paper.user_id == user.user_id)
        )
        job = session.scalar(stmt)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return JobStatusResponse(
            job_id=job.id,
            paper_id=job.paper_id,
            status=job.status,
            progress=job.progress,
            error=job.error,
            result=job.result,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )


def _load_meta(
    storage: StorageBackend,
    paper_id: str,
    *,
    user_id: str = "local-user",
    settings: Settings | None = None,
) -> dict[str, Any]:
    cfg = settings or get_settings()
    with session_scope(cfg) as session:
        paper = PaperRepository(session).get_paper(paper_id, user_id=user_id)
        if paper is None and cfg.auth_enabled:
            raise HTTPException(status_code=404, detail=f"Unknown paper_id: {paper_id}")
        paper_snapshot = None if paper is None else {
            "paper_id": paper.id,
            "filename": paper.filename,
            "status": paper.status,
            "parse_status": paper.parse_status,
            "pages": paper.page_count,
            "created_at": paper.created_at.isoformat() if paper.created_at else None,
            "artifacts": paper.artifacts or {},
            "warnings": paper.warnings or [],
            "title": paper.title,
        }
    key = paper_meta_key(paper_id)
    if not storage.exists(key):
        # Fallback to DB library record
        if paper_snapshot is None:
            raise HTTPException(status_code=404, detail=f"Unknown paper_id: {paper_id}")
        return paper_snapshot
    try:
        return storage.read_json(key)
    except ObjectNotFoundError as exc:
        raise HTTPException(status_code=404, detail=f"Unknown paper_id: {paper_id}") from exc


@router.get("/papers/{paper_id}", response_model=PaperMetadataResponse)
def get_paper(
    paper_id: str,
    settings: Settings = Depends(get_settings),
    storage: StorageBackend = Depends(storage_dep),
    user: CurrentUser = Depends(current_user),
) -> PaperMetadataResponse:
    meta = _load_meta(storage, paper_id, user_id=user.user_id, settings=settings)
    artifacts = meta.get("artifacts") or {}
    return PaperMetadataResponse(
        paper_id=meta["paper_id"],
        filename=meta.get("filename") or "",
        status=meta.get("status") or "unknown",
        parse_status=meta.get("parse_status") or "unknown",
        pages=int(meta.get("pages") or 0),
        created_at=meta.get("created_at"),
        artifacts=artifacts,
        warnings=list(meta.get("warnings") or []),
        title=meta.get("title"),
    )


@router.delete("/papers/{paper_id}", response_model=DeletePaperResponse)
def delete_paper(
    paper_id: str,
    settings: Settings = Depends(get_settings),
    storage: StorageBackend = Depends(storage_dep),
    user: CurrentUser = Depends(current_user),
) -> DeletePaperResponse:
    _load_meta(storage, paper_id, user_id=user.user_id, settings=settings)
    deleted_objects = 0
    prefixes = [
        f"raw/papers/{paper_id}/",
        f"parsed/papers/{paper_id}/",
        f"normalized/papers/{paper_id}/",
        f"assets/papers/{paper_id}/",
        f"enrichment/papers/{paper_id}/",
    ]
    for prefix in prefixes:
        for key in storage.list_objects(prefix):
            storage.delete_object(key)
            deleted_objects += 1

    with session_scope(settings) as session:
        ok = PaperRepository(session).delete_paper(paper_id, user_id=user.user_id)

    if not ok and deleted_objects == 0:
        raise HTTPException(status_code=404, detail=f"Unknown paper_id: {paper_id}")
    return DeletePaperResponse(
        paper_id=paper_id,
        deleted=True,
        storage_objects_deleted=deleted_objects,
    )


@router.get("/papers/{paper_id}/document", response_model=PaperDocument)
def get_document(
    paper_id: str,
    settings: Settings = Depends(get_settings),
    storage: StorageBackend = Depends(storage_dep),
    user: CurrentUser = Depends(current_user),
) -> PaperDocument:
    _load_meta(storage, paper_id, user_id=user.user_id, settings=settings)
    key = paper_normalized_key(paper_id, "paper_document.json")
    if not storage.exists(key):
        raise HTTPException(status_code=404, detail="paper_document.json not found")
    return PaperDocument.model_validate(storage.read_json(key))


@router.get("/papers/{paper_id}/elements")
def get_elements(
    paper_id: str,
    type: str | None = Query(default=None, alias="type"),
    page: int | None = None,
    section: str | None = None,
    kept_only: bool = True,
    settings: Settings = Depends(get_settings),
    storage: StorageBackend = Depends(storage_dep),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    _load_meta(storage, paper_id, user_id=user.user_id, settings=settings)
    if kept_only:
        key = paper_normalized_key(paper_id, "cleaned_text.jsonl")
        if not storage.exists(key):
            raise HTTPException(status_code=404, detail="cleaned_text.jsonl not found")
        lines = storage.read_text(key).splitlines()
        elements: list[dict[str, Any]] = []
        for line in lines:
            if not line.strip():
                continue
            import json

            elements.append(json.loads(line))
    else:
        doc = get_document(paper_id, settings, storage, user)
        elements = [el.model_dump(mode="json") for el in doc.text_elements]

    filtered: list[dict[str, Any]] = []
    for el in elements:
        if type and str(el.get("element_type") or el.get("type") or "").lower() != type.lower():
            continue
        if page is not None and el.get("page") != page:
            continue
        if section:
            path = el.get("section") or el.get("section_path") or []
            joined = " > ".join(path) if isinstance(path, list) else str(path)
            if section.lower() not in joined.lower():
                continue
        filtered.append(el)
    return {"paper_id": paper_id, "count": len(filtered), "elements": filtered}


@router.get("/papers/{paper_id}/assets", response_model=AssetManifest)
def get_assets(
    paper_id: str,
    settings: Settings = Depends(get_settings),
    storage: StorageBackend = Depends(storage_dep),
    user: CurrentUser = Depends(current_user),
) -> AssetManifest:
    _load_meta(storage, paper_id, user_id=user.user_id, settings=settings)
    key = paper_normalized_key(paper_id, "assets_manifest.json")
    if storage.exists(key):
        return AssetManifest.model_validate(storage.read_json(key))
    doc = get_document(paper_id, settings, storage, user)
    return AssetManifest(tables=doc.tables, figures=doc.figures, formulas=doc.formulas)


@router.get("/papers/{paper_id}/assets/{asset_type}/{element_id}/content")
def get_asset_content(
    paper_id: str,
    asset_type: str,
    element_id: str,
    settings: Settings = Depends(get_settings),
    storage: StorageBackend = Depends(storage_dep),
    user: CurrentUser = Depends(current_user),
) -> Response:
    """Return a private visual asset through the API for browser display."""
    manifest = get_assets(paper_id, settings, storage, user)
    buckets = {
        "table": manifest.tables,
        "figure": manifest.figures,
        "formula": manifest.formulas,
    }
    if asset_type not in buckets:
        raise HTTPException(status_code=400, detail="asset_type must be table, figure, or formula")
    element = next((item for item in buckets[asset_type] if item.element_id == element_id), None)
    if element is None:
        raise HTTPException(status_code=404, detail=f"Unknown {asset_type} element: {element_id}")
    if not element.image_uri or not storage.exists(element.image_uri):
        raise HTTPException(status_code=404, detail="Image asset is unavailable")
    return Response(
        content=storage.read_bytes(element.image_uri),
        media_type="image/png",
        headers={"Cache-Control": "private, max-age=300"},
    )


@router.post("/papers/{paper_id}/index")
def index_paper(
    paper_id: str,
    body: IndexRequest | None = None,
    settings: Settings = Depends(get_settings),
    storage: StorageBackend = Depends(storage_dep),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    _load_meta(storage, paper_id, user_id=user.user_id, settings=settings)
    force = body.force if body else False
    return index_paper_chunks(paper_id, settings=settings, force=force)


@router.post("/retrieve")
def retrieve_endpoint(
    body: RetrieveRequest,
    settings: Settings = Depends(get_settings),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    if body.paper_id:
        with session_scope(settings) as session:
            if PaperRepository(session).get_paper(body.paper_id, user_id=user.user_id) is None:
                raise HTTPException(status_code=404, detail="Paper not found")
    return retrieve(
        body.query,
        paper_id=body.paper_id,
        settings=settings,
        top_k=body.top_k or settings.retrieval_top_k,
        use_mmr=body.use_mmr or settings.retrieval_use_mmr,
        user_id=user.user_id,
    )


@router.get("/papers/{paper_id}/chunks")
def list_chunks(
    paper_id: str,
    settings: Settings = Depends(get_settings),
    storage: StorageBackend = Depends(storage_dep),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    _load_meta(storage, paper_id, user_id=user.user_id, settings=settings)
    from sqlalchemy import select
    from app.db.models import PaperChunk

    with session_scope(settings) as session:
        rows = list(
            session.scalars(select(PaperChunk).where(PaperChunk.paper_id == paper_id))
        )
        return {
            "paper_id": paper_id,
            "count": len(rows),
            "chunks": [
                {
                    "id": r.id,
                    "chunk_type": r.chunk_type,
                    "section_path": r.section_path,
                    "page_start": r.page_start,
                    "page_end": r.page_end,
                    "token_count": r.token_count,
                    "has_embedding": bool(r.embedding_model),
                    "content_preview": (r.content or "")[:240],
                }
                for r in rows
            ],
        }


@router.post("/papers/{paper_id}/qa")
def paper_qa(
    paper_id: str,
    body: QARequest,
    settings: Settings = Depends(get_settings),
    storage: StorageBackend = Depends(storage_dep),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    _load_meta(storage, paper_id, user_id=user.user_id, settings=settings)
    answer, state = answer_paper_question(
        paper_id=paper_id,
        question=body.question,
        settings=settings,
        top_k=body.top_k,
        conversation_id=body.conversation_id,
        persist=True,
        user_id=user.user_id,
    )
    return {
        "paper_id": paper_id,
        "conversation_id": state.conversation_id,
        "answer": answer.model_dump(mode="json"),
        "turn_count": len(state.turns),
    }


@router.post("/discover")
def discover_endpoint(
    body: DiscoverRequest,
    settings: Settings = Depends(get_settings),
    storage: StorageBackend = Depends(storage_dep),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    try:
        result = discover_papers(
            body.query,
            source=body.source,
            year_min=body.year_min,
            year_max=body.year_max,
            limit=body.limit,
            settings=settings,
            storage=storage,
            force_refresh=body.force_refresh,
        )
    except DiscoveryError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    payload = result.model_dump(mode="json")
    for item in payload["results"]:
        from app.research.discovery import DiscoveryPaper

        paper = DiscoveryPaper.model_validate(item)
        item["library_matches"] = find_library_duplicates(
            paper, settings=settings, user_id=user.user_id
        )
    return payload


@router.post("/compare")
def compare_endpoint(
    body: CompareRequest,
    settings: Settings = Depends(get_settings),
    storage: StorageBackend = Depends(storage_dep),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    if len(body.paper_ids) < 2:
        raise HTTPException(status_code=400, detail="At least two paper_ids required")
    titles: dict[str, str | None] = {}
    for pid in body.paper_ids:
        meta = _load_meta(storage, pid, user_id=user.user_id, settings=settings)
        titles[pid] = meta.get("title")
    try:
        result = compare_papers(
            paper_ids=body.paper_ids,
            question=body.question,
            settings=settings,
            top_k_per_paper=body.top_k_per_paper,
            titles=titles,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump(mode="json")


@router.post("/research")
def research_endpoint(
    body: ResearchRequest,
    settings: Settings = Depends(get_settings),
    user: CurrentUser = Depends(current_user),
) -> dict[str, Any]:
    with session_scope(settings) as session:
        repo = PaperRepository(session)
        for paper_id in body.selected_papers or []:
            if repo.get_paper(paper_id, user_id=user.user_id) is None:
                raise HTTPException(status_code=404, detail="Paper not found")
    report = run_research(
        body.research_question,
        selected_papers=body.selected_papers,
        settings=settings,
        enable_external=body.enable_external,
        max_external_searches=body.max_external_searches,
        user_id=user.user_id,
    )
    return report.model_dump(mode="json")


@router.post("/agent", response_model=AgentResponse)
def agent_endpoint(
    body: AgentRequest,
    settings: Settings = Depends(get_settings),
    user: CurrentUser = Depends(current_user),
) -> AgentResponse:
    try:
        validated = validate_agent_input(
            body.message,
            selected_papers=body.selected_papers,
            image_data_url=body.image,
            settings=settings,
        )
        guest_quota = _consume_guest_if_needed(
            user,
            settings,
            queries=1,
            images=1 if validated.image is not None else 0,
        )
        if settings.auth_enabled:
            with session_scope(settings) as session:
                repo = PaperRepository(session)
                for paper_id in validated.selected_papers:
                    if repo.get_paper(paper_id, user_id=user.user_id) is None:
                        raise HTTPException(status_code=404, detail="Paper not found")
                if body.conversation_id:
                    conversation = AgentConversationRepository(session).get_with_messages(
                        body.conversation_id, user_id=user.user_id
                    )
                    if conversation is None:
                        raise HTTPException(status_code=404, detail="Conversation not found")
        agent_kwargs = {
            "selected_papers": validated.selected_papers,
            "conversation_id": body.conversation_id,
            "settings": settings,
            "image": validated.image,
        }
        if settings.auth_enabled:
            agent_kwargs["user_id"] = user.user_id
        result = AgentResponse.model_validate(
            run_agent(validated.message, **agent_kwargs)
        )
        if guest_quota is not None:
            result.guest_quota = guest_quota.model_dump(mode="json")
        return result
    except GuardrailError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": exc.code, "message": exc.safe_message},
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/agent/stream")
def agent_stream_endpoint(
    body: AgentRequest,
    settings: Settings = Depends(get_settings),
    user: CurrentUser = Depends(current_user),
) -> StreamingResponse:
    try:
        validated = validate_agent_input(
            body.message,
            selected_papers=body.selected_papers,
            image_data_url=body.image,
            settings=settings,
        )
        guest_quota = _consume_guest_if_needed(
            user,
            settings,
            queries=1,
            images=1 if validated.image is not None else 0,
        )
        if settings.auth_enabled:
            with session_scope(settings) as session:
                repo = PaperRepository(session)
                for paper_id in validated.selected_papers:
                    if repo.get_paper(paper_id, user_id=user.user_id) is None:
                        raise HTTPException(status_code=404, detail="Paper not found")
                if body.conversation_id:
                    conversation = AgentConversationRepository(session).get_with_messages(
                        body.conversation_id, user_id=user.user_id
                    )
                    if conversation is None:
                        raise HTTPException(status_code=404, detail="Conversation not found")
    except GuardrailError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": exc.code, "message": exc.safe_message},
        ) from exc

    def events():
        try:
            agent_kwargs = {
                "selected_papers": validated.selected_papers,
                "conversation_id": body.conversation_id,
                "settings": settings,
                "image": validated.image,
            }
            if settings.auth_enabled:
                agent_kwargs["user_id"] = user.user_id
            for event in stream_agent(validated.message, **agent_kwargs):
                if (
                    guest_quota is not None
                    and isinstance(event, dict)
                    and event.get("type") == "done"
                ):
                    event = {**event, "guest_quota": guest_quota.model_dump(mode="json")}
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        except GuardrailError as exc:
            error = {
                "type": "error",
                "error": exc.code,
                "message": exc.safe_message,
            }
            yield f"data: {json.dumps(error)}\n\n"
        except Exception as exc:
            logger.exception("Research agent stream failed")
            error = {"type": "error", "message": f"{type(exc).__name__}: request failed"}
            yield f"data: {json.dumps(error)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get(
    "/agent/conversations",
    response_model=AgentConversationListResponse,
)
def list_agent_conversations_endpoint(
    limit: int = Query(default=100, ge=1, le=200),
    settings: Settings = Depends(get_settings),
    user: CurrentUser = Depends(current_user),
) -> AgentConversationListResponse:
    with session_scope(settings) as session:
        rows = AgentConversationRepository(session).list_for_user(
            user.user_id, limit=limit
        )
        conversations = [
            AgentConversationSummary(
                conversation_id=row.id,
                title=row.title or "New conversation",
                selected_papers=list(row.selected_papers or []),
                created_at=row.created_at,
                updated_at=row.updated_at,
            )
            for row in rows
        ]
    return AgentConversationListResponse(conversations=conversations)


@router.get(
    "/agent/conversations/{conversation_id}",
    response_model=AgentConversationResponse,
)
def get_agent_conversation_endpoint(
    conversation_id: str,
    settings: Settings = Depends(get_settings),
    user: CurrentUser = Depends(current_user),
) -> AgentConversationResponse:
    payload = get_agent_conversation(
        conversation_id, settings=settings, user_id=user.user_id
    )
    if payload is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return AgentConversationResponse(
        conversation_id=payload["conversation_id"],
        title=payload.get("title"),
        selected_papers=list(payload.get("selected_papers") or []),
        created_at=payload.get("created_at"),
        updated_at=payload.get("updated_at"),
        turns=[
            AgentConversationTurn.model_validate(turn) for turn in payload.get("turns") or []
        ],
    )


@router.delete("/agent/conversations/{conversation_id}", status_code=204)
def delete_agent_conversation_endpoint(
    conversation_id: str,
    settings: Settings = Depends(get_settings),
    user: CurrentUser = Depends(current_user),
) -> Response:
    with session_scope(settings) as session:
        deleted = AgentConversationRepository(session).delete_for_user(
            conversation_id, user.user_id
        )
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return Response(status_code=204)


@router.post("/papers/{paper_id}/enrich", response_model=EnrichResponse)
def enrich_paper(
    paper_id: str,
    body: EnrichRequest,
    settings: Settings = Depends(get_settings),
    storage: StorageBackend = Depends(storage_dep),
    user: CurrentUser = Depends(current_user),
) -> EnrichResponse:
    _load_meta(storage, paper_id, user_id=user.user_id, settings=settings)
    client = LunaClient(settings=settings, storage=storage)
    if body.scope in {"visual", "both"} and not client.is_enabled and body.scope == "visual":
        return EnrichResponse(
            paper_id=paper_id,
            luna_enabled=settings.luna_enabled,
            allow_external_api=settings.allow_external_api,
            results=[],
            message="Luna enrichment disabled. Set LUNA_ENABLED=true and ALLOW_EXTERNAL_API=true.",
        )

    manifest = get_assets(paper_id, settings, storage, user)
    candidates: list[VisualElement] = []
    if body.scope in {"visual", "both"} and client.is_enabled:
        for kind in body.element_types:
            bucket = getattr(manifest, f"{kind}s")
            for el in bucket:
                if body.element_ids and el.element_id not in body.element_ids:
                    continue
                if not body.force and kind == "formula" and not el.needs_enrichment and el.docling_text:
                    continue
                candidates.append(el)

    results: list[EnrichResultItem] = []
    paper = None
    paper_key = paper_normalized_key(paper_id, "paper_document.json")
    if storage.exists(paper_key):
        paper = PaperDocument.model_validate(storage.read_json(paper_key))
    text_results: list[dict[str, Any]] = []
    if body.scope in {"text", "both"} and paper is not None:
        try:
            text_results = enrich_cleaned_text(
                paper,
                settings=settings,
                storage=storage,
                force=body.force,
            )
            paper.metadata["text_enrichment"] = {
                "source": "cleaned_docling_text",
                "sections": text_results,
            }
        except Exception as exc:
            logger.exception("Cleaned text enrichment failed for %s", paper_id)
            text_results = [{"status": "failed", "error": str(exc)}]

    for el in candidates:
        image_bytes = None
        if el.image_uri and storage.exists(el.image_uri):
            image_bytes = storage.read_bytes(el.image_uri)
        try:
            enrichment = client.enrich_element(
                paper_id=paper_id,
                element=el,
                image_bytes=image_bytes,
                force=body.force,
            )
            el.enrichment = enrichment
            results.append(
                EnrichResultItem(
                    element_id=el.element_id,
                    type=el.type,
                    status=enrichment.status,
                    cached=enrichment.cached,
                    error=enrichment.error,
                    enrichment_uri=None,
                )
            )
            if paper is not None:
                _update_paper_visual(paper, el)
        except LunaDisabledError as exc:
            results.append(
                EnrichResultItem(
                    element_id=el.element_id,
                    type=el.type,
                    status="skipped",
                    error=str(exc),
                )
            )
        except Exception as exc:
            logger.exception("Enrichment failed for %s", el.element_id)
            results.append(
                EnrichResultItem(
                    element_id=el.element_id,
                    type=el.type,
                    status="failed",
                    error=str(exc),
                )
            )

    message = None
    if paper is not None:
        storage.save_json(paper_key, paper.model_dump(mode="json"))
        storage.save_json(
            paper_normalized_key(paper_id, "assets_manifest.json"),
            {
                "tables": [t.model_dump(mode="json") for t in paper.tables],
                "figures": [f.model_dump(mode="json") for f in paper.figures],
                "formulas": [f.model_dump(mode="json") for f in paper.formulas],
            },
        )
        if any(item.status == "completed" for item in results):
            # New descriptions only help retrieval once they are in the chunks.
            try:
                persist_chunks(paper_id, chunk_paper_document(paper), settings=settings)
                index_paper_chunks(paper_id, settings=settings, force=True)
                message = "Enrichment added to the search index."
            except Exception as exc:
                logger.warning("Re-indexing after enrichment failed for %s: %s", paper_id, exc)
                message = "Enrichment saved, but re-indexing failed; search uses the old chunks."

    return EnrichResponse(
        message=message,
        paper_id=paper_id,
        luna_enabled=settings.luna_enabled,
        allow_external_api=settings.allow_external_api,
        results=results,
        text_results=text_results,
    )


def _update_paper_visual(paper: PaperDocument, updated: VisualElement) -> None:
    buckets = {
        "table": paper.tables,
        "figure": paper.figures,
        "formula": paper.formulas,
    }
    bucket = buckets[updated.type]
    for idx, el in enumerate(bucket):
        if el.element_id == updated.element_id:
            bucket[idx] = updated
            return
