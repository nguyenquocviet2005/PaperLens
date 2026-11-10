"""LangChain embedding models and PostgreSQL vector indexing."""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import DeterministicFakeEmbedding, Embeddings
from sqlalchemy import select, text
from sqlalchemy.pool import NullPool

from app.config import Settings, get_settings
from app.db.models import PaperChunk
from app.db.session import session_scope
from app.db.url_utils import classify_database_url, ensure_sslmode

logger = logging.getLogger(__name__)

VECTOR_COLLECTION = "paperlens_chunks"


class EmbeddingError(Exception):
    pass


def get_embeddings(settings: Settings | None = None) -> tuple[Embeddings, str]:
    """Return a LangChain Embeddings implementation and its stable model name."""
    cfg = settings or get_settings()
    provider = (cfg.embedding_provider or "hashing").lower()
    if provider == "hashing":
        return (
            DeterministicFakeEmbedding(size=cfg.embedding_dimensions),
            f"langchain-deterministic-{cfg.embedding_dimensions}",
        )
    if provider == "openai":
        if not cfg.embedding_api_key or not cfg.embedding_model:
            raise EmbeddingError(
                "EMBEDDING_API_KEY and EMBEDDING_MODEL are required for OpenAI embeddings"
            )
        from langchain_openai import OpenAIEmbeddings

        kwargs: dict[str, Any] = {
            "model": cfg.embedding_model,
            "api_key": cfg.embedding_api_key,
        }
        if cfg.embedding_send_dimensions:
            kwargs["dimensions"] = cfg.embedding_dimensions
        if cfg.embedding_batch_size:
            kwargs["chunk_size"] = cfg.embedding_batch_size
        if cfg.embedding_base_url:
            kwargs["base_url"] = cfg.embedding_base_url
            # OpenAI-compatible providers (OpenRouter, TEI, vLLM) expect raw strings, not the
            # tiktoken token arrays LangChain sends to api.openai.com by default.
            kwargs["check_embedding_ctx_length"] = False
        return OpenAIEmbeddings(**kwargs), cfg.embedding_model
    raise EmbeddingError(f"Unknown embedding provider: {provider}")


def get_vector_store(
    settings: Settings | None = None,
    *,
    embeddings: Embeddings | None = None,
):
    """Return LangChain PGVector for PostgreSQL; SQLite intentionally has no PG store."""
    cfg = settings or get_settings()
    info = classify_database_url(cfg.database_url)
    if not info.dialect.startswith("postgres"):
        return None

    from langchain_postgres import PGVector

    model = embeddings or get_embeddings(cfg)[0]
    connection = ensure_sslmode(cfg.database_url, default="require")
    engine_args: dict[str, Any] = {"pool_pre_ping": True}
    if info.connection_mode == "transaction_pooler":
        engine_args = {
            "poolclass": NullPool,
            "connect_args": {"prepare_threshold": None},
        }
    return PGVector(
        embeddings=model,
        connection=connection,
        collection_name=VECTOR_COLLECTION,
        embedding_length=cfg.embedding_dimensions,
        use_jsonb=True,
        create_extension=True,
        engine_args=engine_args,
    )


def _chunk_document(row: PaperChunk) -> Document:
    metadata = {
        "chunk_id": row.id,
        "paper_id": row.paper_id,
        "parent_chunk_id": row.parent_chunk_id,
        "chunk_type": row.chunk_type,
        "section_path": list(row.section_path or []),
        "page_start": row.page_start,
        "page_end": row.page_end,
        **dict(row.chunk_metadata or {}),
    }
    page = row.page_start or row.page_end
    section = " > ".join(row.section_path or [])
    if row.chunk_type in {"table", "figure", "equation"}:
        label = row.chunk_type.title()
        metadata["citation"] = f"[{label}, Page {page}]" if page else f"[{label}]"
    elif page and section:
        metadata["citation"] = f"[Page {page}, Section {section}]"
    elif page:
        metadata["citation"] = f"[Page {page}]"
    else:
        metadata["citation"] = f"[Chunk {row.id[:8]}]"
    return Document(page_content=row.content, metadata=metadata)


def index_paper_chunks(
    paper_id: str,
    *,
    settings: Settings | None = None,
    force: bool = False,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Index paper chunks through LangChain embeddings and PGVector."""
    cfg = settings or get_settings()
    embeddings, model_name = get_embeddings(cfg)
    with session_scope(cfg) as session:
        rows = list(
            session.scalars(
                select(PaperChunk)
                .where(PaperChunk.paper_id == paper_id)
                .order_by(PaperChunk.chunk_type, PaperChunk.id)
            )
        )
        pending = [
            row
            for row in rows
            if force or row.embedding_model != model_name
        ]
        documents = [_chunk_document(row) for row in pending]
        ids = [row.id for row in pending]

        vector_store = get_vector_store(cfg, embeddings=embeddings)
        if vector_store is not None and documents:
            if force:
                existing_ids = list(
                    session.scalars(
                        text(
                            "SELECT id FROM langchain_pg_embedding "
                            "WHERE cmetadata->>'paper_id' = :paper_id"
                        ).bindparams(paper_id=paper_id)
                    )
                )
                if existing_ids:
                    vector_store.delete(ids=[str(value) for value in existing_ids])
            for start in range(0, len(documents), batch_size):
                vector_store.add_documents(
                    documents[start : start + batch_size],
                    ids=ids[start : start + batch_size],
                )
            for row in pending:
                row.embedding = None
                row.embedding_model = model_name
        elif documents:
            # SQLite test/dev fallback still delegates vector creation to LangChain.
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                vectors = embeddings.embed_documents([row.content for row in batch])
                for row, vector in zip(batch, vectors):
                    row.embedding = list(vector)
                    row.embedding_model = model_name
        session.flush()

    result = {
        "paper_id": paper_id,
        "provider": model_name,
        "dimensions": cfg.embedding_dimensions,
        "updated": len(pending),
        "skipped": len(rows) - len(pending),
        "total": len(rows),
        "vector_store": "langchain_pgvector" if vector_store is not None else "sqlite_json_fallback",
    }
    logger.info("Indexed embeddings %s", result)
    return result
