"""Hybrid retrieval composed from LangChain BM25, vector, and ensemble retrievers."""

from __future__ import annotations

import logging
from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sqlalchemy import select

from app.config import Settings, get_settings
from app.db.models import Paper, PaperChunk
from app.db.session import session_scope
from app.rag.embeddings import get_embeddings, get_vector_store

logger = logging.getLogger(__name__)


def normalize_query(query: str) -> str:
    return " ".join(query.strip().split())


def _row_document(row: PaperChunk) -> Document:
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
    metadata["citation"] = _citation_label(metadata)
    return Document(page_content=row.content, metadata=metadata)


class _SparseFallbackRetriever(BaseRetriever):
    """Hybrid retriever that degrades to BM25 when the embedding provider fails at query time.

    Embedding errors (rate limits, exhausted credits, provider outages) surface on
    `invoke`, not when the vector store is constructed, so they need handling here.
    """

    primary: Any
    sparse: Any

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        try:
            return self.primary.invoke(query, config={"callbacks": run_manager.get_child()})
        except Exception as exc:
            logger.warning("Dense retrieval failed; falling back to BM25: %s", type(exc).__name__)
            return self.sparse.invoke(query, config={"callbacks": run_manager.get_child()})


def _build_retriever(
    documents: list[Document],
    *,
    settings: Settings,
    paper_id: str | None,
    candidate_pool: int,
    use_mmr: bool,
):
    from langchain_community.retrievers import BM25Retriever

    sparse = BM25Retriever.from_documents(documents)
    sparse.k = min(candidate_pool, len(documents))
    retrievers: list[Any] = [sparse]
    weights = [1.0]
    source = "langchain_bm25"

    try:
        embeddings, _model_name = get_embeddings(settings)
        vector_store = get_vector_store(settings, embeddings=embeddings)
        if vector_store is None:
            from langchain_core.vectorstores import InMemoryVectorStore

            vector_store = InMemoryVectorStore.from_documents(
                documents,
                embedding=embeddings,
            )
        search_kwargs: dict[str, Any] = {"k": min(candidate_pool, len(documents))}
        if paper_id and settings.database_info.dialect.startswith("postgres"):
            search_kwargs["filter"] = {"paper_id": {"$eq": paper_id}}
        dense = vector_store.as_retriever(
            search_type="mmr" if use_mmr else "similarity",
            search_kwargs=search_kwargs,
        )
        retrievers.append(dense)
        weights = [0.45, 0.55]
        source = "langchain_bm25_pgvector_ensemble"
    except Exception as exc:
        logger.warning("LangChain vector retriever unavailable; using BM25: %s", type(exc).__name__)

    if len(retrievers) == 1:
        return sparse, source

    from langchain_classic.retrievers import EnsembleRetriever

    ensemble = EnsembleRetriever(
        retrievers=retrievers,
        weights=weights,
        id_key="chunk_id",
    )
    return _SparseFallbackRetriever(primary=ensemble, sparse=sparse), source


def get_langchain_retriever(
    *,
    paper_id: str | None,
    settings: Settings | None = None,
    candidate_pool: int = 40,
    use_mmr: bool = False,
    user_id: str | None = None,
):
    """Build the native LangChain retriever consumed by QA chains and LangGraph tools."""
    cfg = settings or get_settings()
    with session_scope(cfg) as session:
        stmt = select(PaperChunk)
        if user_id is not None:
            stmt = stmt.join(Paper, Paper.id == PaperChunk.paper_id).where(
                Paper.user_id == user_id
            )
        if paper_id:
            stmt = stmt.where(PaperChunk.paper_id == paper_id)
        documents = [_row_document(row) for row in session.scalars(stmt)]
    if not documents:
        return None, "none", 0
    retriever, source = _build_retriever(
        documents,
        settings=cfg,
        paper_id=paper_id,
        candidate_pool=candidate_pool,
        use_mmr=use_mmr,
    )
    return retriever, source, len(documents)


def retrieve(
    query: str,
    *,
    paper_id: str | None = None,
    settings: Settings | None = None,
    top_k: int = 8,
    candidate_pool: int = 40,
    use_mmr: bool = False,
    expand_parents: bool = True,
    user_id: str | None = None,
) -> dict[str, Any]:
    cfg = settings or get_settings()
    q = normalize_query(query)
    if not q:
        return {"query": query, "results": [], "diagnostics": {"error": "empty_query"}}

    with session_scope(cfg) as session:
        stmt = select(PaperChunk)
        if user_id is not None:
            stmt = stmt.join(Paper, Paper.id == PaperChunk.paper_id).where(
                Paper.user_id == user_id
            )
        if paper_id:
            stmt = stmt.where(PaperChunk.paper_id == paper_id)
        rows = list(session.scalars(stmt))
        documents = [_row_document(row) for row in rows]
        parent_snapshots = {
            row.id: {
                "content": row.content,
                "page_start": row.page_start,
                "page_end": row.page_end,
            }
            for row in rows
        }

    if not rows:
        return {
            "query": q,
            "results": [],
            "diagnostics": {
                "returned": 0,
                "paper_id": paper_id,
                "retriever": "none",
            },
        }

    retriever, source = _build_retriever(
        documents,
        settings=cfg,
        paper_id=paper_id,
        candidate_pool=candidate_pool,
        use_mmr=use_mmr,
    )
    allowed_chunk_ids = set(parent_snapshots)
    selected = [
        document
        for document in retriever.invoke(q)
        if str(document.metadata.get("chunk_id") or "") in allowed_chunk_ids
    ][:top_k]
    results: list[dict[str, Any]] = []
    for rank, document in enumerate(selected, start=1):
        metadata = dict(document.metadata)
        parent_id = metadata.get("parent_chunk_id")
        if expand_parents and parent_id in parent_snapshots:
            parent = parent_snapshots[parent_id]
            metadata["parent_content_preview"] = str(parent["content"])[:500]
            metadata["parent_pages"] = [parent["page_start"], parent["page_end"]]
        result = {
            "chunk_id": str(metadata.get("chunk_id") or ""),
            "paper_id": str(metadata.get("paper_id") or ""),
            "content": document.page_content,
            "chunk_type": str(metadata.get("chunk_type") or "text"),
            "section_path": list(metadata.get("section_path") or []),
            "page_start": metadata.get("page_start"),
            "page_end": metadata.get("page_end"),
            "score": 1.0 / rank,
            "metadata": metadata,
            "parent_chunk_id": parent_id,
            "sources": [source],
        }
        result["citation"] = _citation_label(result)
        results.append(result)

    return {
        "query": q,
        "results": results,
        "diagnostics": {
            "corpus_size": len(documents),
            "returned": len(results),
            "mmr": use_mmr,
            "paper_id": paper_id,
            "retriever": source,
        },
    }


def _citation_label(item: dict[str, Any]) -> str:
    page = item.get("page_start") or item.get("page_end")
    section_path = item.get("section_path") or []
    section = " > ".join(section_path) if section_path else None
    chunk_type = item.get("chunk_type")
    if chunk_type == "table":
        return f"[Table, Page {page}]" if page else "[Table]"
    if chunk_type == "figure":
        return f"[Figure, Page {page}]" if page else "[Figure]"
    if chunk_type == "equation":
        return f"[Equation, Page {page}]" if page else "[Equation]"
    if page and section:
        return f"[Page {page}, Section {section}]"
    if page:
        return f"[Page {page}]"
    if section:
        return f"[Section {section}]"
    chunk_id = str(item.get("chunk_id") or "")
    return f"[Chunk {chunk_id[:8]}]"


def evaluate_retrieval(
    cases: list[dict[str, Any]],
    *,
    settings: Settings | None = None,
    k_values: Sequence[int] = (5, 10),
) -> dict[str, Any]:
    """Compute Recall@k and MRR over expected chunk IDs or content substrings."""
    recalls: dict[int, list[float]] = {k: [] for k in k_values}
    reciprocal_ranks: list[float] = []
    details = []
    for case in cases:
        output = retrieve(
            case["query"],
            paper_id=case.get("paper_id"),
            settings=settings,
            top_k=max(k_values),
        )
        expected_ids = set(case.get("expected_chunk_ids") or [])
        expected_substrings = [
            value.lower() for value in case.get("expected_substrings") or []
        ]
        hit_rank = None
        for rank, item in enumerate(output["results"], start=1):
            if item["chunk_id"] in expected_ids or any(
                value in item["content"].lower() for value in expected_substrings
            ):
                hit_rank = rank
                break
        reciprocal_ranks.append(0.0 if hit_rank is None else 1.0 / hit_rank)
        for k in k_values:
            recalls[k].append(1.0 if hit_rank is not None and hit_rank <= k else 0.0)
        details.append(
            {
                "query": case["query"],
                "hit_rank": hit_rank,
                "diagnostics": output["diagnostics"],
            }
        )

    def average(values: list[float]) -> float:
        return round(sum(values) / len(values), 4) if values else 0.0

    return {
        "metrics": {
            "n_cases": len(cases),
            "MRR": average(reciprocal_ranks),
            **{f"Recall@{k}": average(values) for k, values in recalls.items()},
        },
        "details": details,
    }
