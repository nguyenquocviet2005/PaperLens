"""Hybrid retrieval composed from LangChain BM25, vector, and ensemble retrievers."""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from sqlalchemy import select

from app.config import Settings, get_settings
from app.db.models import Paper, PaperChunk
from app.db.session import session_scope
from app.infrastructure.observability import emit_metrics
from app.rag.embeddings import PARENT_CHUNK_TYPE, get_embeddings, get_vector_store

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


def _searchable(row: PaperChunk) -> bool:
    return row.chunk_type != PARENT_CHUNK_TYPE


def _chunk_query(paper_id: str | None, user_id: str | None):
    stmt = select(PaperChunk)
    if user_id is not None:
        stmt = stmt.join(Paper, Paper.id == PaperChunk.paper_id).where(Paper.user_id == user_id)
    if paper_id:
        stmt = stmt.where(PaperChunk.paper_id == paper_id)
    return stmt


def get_langchain_retriever(
    *,
    paper_id: str | None,
    settings: Settings | None = None,
    candidate_pool: int = 40,
    use_mmr: bool = False,
    user_id: str | None = None,
):
    """Native LangChain retriever over the searchable (child) chunks."""
    cfg = settings or get_settings()
    with session_scope(cfg) as session:
        documents = [
            _row_document(row)
            for row in session.scalars(_chunk_query(paper_id, user_id))
            if _searchable(row)
        ]
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


def _count_tokens(text: str) -> int:
    from app.ingestion.surrounding import count_tokens

    return count_tokens(text)


def retrieve(
    query: str,
    *,
    paper_id: str | None = None,
    settings: Settings | None = None,
    top_k: int = 8,
    candidate_pool: int = 40,
    use_mmr: bool = False,
    expand_parents: bool | None = None,
    max_evidence_tokens: int | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Hybrid search over child chunks, returned as at most `top_k` evidence units.

    Small-to-big: a matching text child is replaced by its parent passage, and several
    children of one parent collapse into a single unit (pages come from the children that
    matched). Tables, figures and equations are their own unit. Units stop once the
    evidence token budget is used, but the first unit is always returned.
    """
    cfg = settings or get_settings()
    started = time.perf_counter()
    expand = cfg.retrieval_expand_parents if expand_parents is None else expand_parents
    budget = max_evidence_tokens or cfg.retrieval_max_evidence_tokens
    q = normalize_query(query)
    if not q:
        return {"query": query, "results": [], "diagnostics": {"error": "empty_query"}}

    with session_scope(cfg) as session:
        rows = list(session.scalars(_chunk_query(paper_id, user_id)))
        documents = [_row_document(row) for row in rows if _searchable(row)]
        parents = {
            row.id: {
                "content": row.content,
                "section_path": list(row.section_path or []),
                "page_start": row.page_start,
                "page_end": row.page_end,
            }
            for row in rows
            if not _searchable(row)
        }

    if not documents:
        return {
            "query": q,
            "results": [],
            "diagnostics": {"returned": 0, "paper_id": paper_id, "retriever": "none"},
        }

    retriever, source = _build_retriever(
        documents,
        settings=cfg,
        paper_id=paper_id,
        candidate_pool=candidate_pool,
        use_mmr=use_mmr,
    )
    searchable_ids = {str(d.metadata["chunk_id"]) for d in documents}
    hits = [d for d in retriever.invoke(q) if str(d.metadata.get("chunk_id") or "") in searchable_ids]

    units: dict[str, dict[str, Any]] = {}
    used_tokens = 0
    budget_stopped = False
    for document in hits:
        metadata = dict(document.metadata)
        parent_id = metadata.get("parent_chunk_id")
        parent = parents.get(parent_id) if expand and parent_id else None
        unit_id = str(parent_id if parent else metadata["chunk_id"])
        pages = [p for p in (metadata.get("page_start"), metadata.get("page_end")) if p is not None]
        if unit_id in units:
            unit = units[unit_id]
            unit["metadata"]["matched_chunk_ids"].append(metadata["chunk_id"])
            unit["_pages"].update(pages)
            continue
        if len(units) >= top_k:
            continue
        content = str(parent["content"]) if parent else document.page_content
        tokens = _count_tokens(content)
        if units and used_tokens + tokens > budget:
            budget_stopped = True
            continue
        used_tokens += tokens
        metadata["matched_chunk_ids"] = [metadata["chunk_id"]]
        metadata["expanded_to_parent"] = bool(parent)
        units[unit_id] = {
            "chunk_id": unit_id,
            "paper_id": str(metadata.get("paper_id") or ""),
            "content": content,
            "matched_text": document.page_content,
            "chunk_type": str(metadata.get("chunk_type") or "text"),
            "section_path": list((parent or metadata).get("section_path") or []),
            "metadata": metadata,
            "parent_chunk_id": parent_id,
            "sources": [source],
            "_pages": set(pages),
        }

    results: list[dict[str, Any]] = []
    for rank, unit in enumerate(units.values(), start=1):
        pages = sorted(unit.pop("_pages"))
        unit["page_start"] = pages[0] if pages else None
        unit["page_end"] = pages[-1] if pages else None
        unit["score"] = 1.0 / rank
        results.append(unit)
    _assign_citation_labels(results)
    emit_metrics(
        {
            "RetrievalLatency": (round((time.perf_counter() - started) * 1000, 1), "Milliseconds"),
            "EvidenceTokens": (used_tokens, "Count"),
            "RetrievalEmpty": (0 if results else 1, "Count"),
        },
        dimensions={"Retriever": source},
    )

    return {
        "query": q,
        "results": results,
        "diagnostics": {
            "corpus_size": len(documents),
            "candidates": len(hits),
            "returned": len(results),
            "evidence_tokens": used_tokens,
            "budget_stopped": budget_stopped,
            "expand_parents": expand,
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


def _assign_citation_labels(results: list[dict[str, Any]]) -> None:
    """Give each result a label no other result in the list shares.

    Labels are how the LLM points at evidence, so two chunks with the same page and
    section must still be told apart: a short id is appended, lengthened on collision.
    """
    for width in (4, 6, 8, 36):
        labels = [
            _citation_label(item)[:-1] + f" #{str(item['chunk_id']).replace('-', '')[:width]}]"
            for item in results
        ]
        if len(set(labels)) == len(labels):
            break
    for item, label in zip(results, labels):
        item["citation"] = label


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
