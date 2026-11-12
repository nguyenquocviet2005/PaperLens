"""Small-to-big chunking and retrieval (index children, hand the LLM their parents)."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import select

from app.config import Settings, get_settings
from app.db.models import PaperChunk
from app.db.repository import PaperRepository
from app.db.session import init_db, reset_engine, session_scope
from app.ingestion.chunking import (
    MODEL_TEXT_LABEL,
    PARENT_CHUNK_TYPE,
    ChunkingConfig,
    _bpe_tokens,
    chunk_paper_document,
    persist_chunks,
)
from app.rag.embeddings import index_paper_chunks
from app.rag.retrieval import retrieve
from app.schemas import ArtifactPaths, PaperDocument, TextElement, VisualElement, VisualEnrichment

SMALL = ChunkingConfig(
    parent_min_tokens=120,
    parent_max_tokens=300,
    child_min_tokens=30,
    child_max_tokens=80,
    overlap_tokens=10,
    child_hard_max_tokens=100,
)


def _el(i: int, text: str, section: str, page: int) -> TextElement:
    return TextElement(
        element_id=f"el_{i:04d}", order=i, page=page, section_path=[section],
        type="TextItem", text=text,
    )


def _sentences(topic: str, n: int) -> str:
    return " ".join(f"Sentence {i} explains {topic} in detail for the reader." for i in range(n))


def _paper() -> PaperDocument:
    table = VisualElement(
        element_id="table_001", type="table", page=3, caption="Table 1: Model sizes.",
        docling_text="| model | params |\n| --- | --- |\n| big | 213M |",
        surrounding_text=["As shown in Table 1, the big model has more parameters."],
        enrichment=VisualEnrichment(
            provider="openai", model="m", prompt_version="v1", schema_version="v1",
            status="completed", result={"table_purpose": "Compares model sizes",
                                        "main_results": ["big has 213M parameters"]},
        ),
    )
    return PaperDocument(
        paper_id="s2b",
        filename="s.pdf",
        page_count=3,
        text_elements=[
            _el(1, "Short abstract about calibration.", "Abstract", 1),
            _el(2, "A short introduction to calibration methods.", "1 Introduction", 1),
            _el(3, _sentences("temperature scaling", 20), "2 Method", 2),
            _el(4, _sentences("reliability diagrams", 20), "3 Results", 3),
            _el(5, "Tiny conclusion.", "4 Conclusion", 3),
        ],
        tables=[table],
    )


def test_short_sections_are_merged_into_parents_within_limits():
    result = chunk_paper_document(_paper(), config=SMALL)
    parents = result.parent_chunks
    assert parents[0].metadata["sections"][:2] == ["Abstract", "1 Introduction"]
    for parent in parents:
        tokens = parent.metadata["bpe_tokens"]
        assert tokens <= SMALL.parent_max_tokens
    # Every parent but possibly the last reaches the minimum.
    assert all(p.metadata["bpe_tokens"] >= SMALL.parent_min_tokens for p in parents[:-1])
    # The tiny conclusion does not stand alone.
    assert not any(p.content.strip() == "Tiny conclusion." for p in parents)


def test_oversized_section_is_split_on_its_own():
    paper = _paper()
    paper.text_elements[2].text = _sentences("temperature scaling", 80)  # ~1000 tokens
    parents = chunk_paper_document(paper, config=SMALL).parent_chunks
    method_parts = [p for p in parents if p.metadata["sections"] == ["2 Method"]]
    assert len(method_parts) >= 3
    assert all(p.metadata["bpe_tokens"] <= SMALL.parent_max_tokens for p in parents)


def test_children_stay_small_and_point_at_parents():
    result = chunk_paper_document(_paper(), config=SMALL)
    keys = {p.metadata["parent_key"] for p in result.parent_chunks}
    text_children = [c for c in result.child_chunks if c.chunk_type == "text"]
    assert text_children
    for child in text_children:
        assert child.parent_key in keys
        assert _bpe_tokens(child.content) <= SMALL.child_hard_max_tokens
    assert result.metrics["indexed_chunks"] == len(result.child_chunks)


def test_table_chunk_carries_data_and_labelled_model_text():
    result = chunk_paper_document(_paper(), config=SMALL)
    table = next(c for c in result.child_chunks if c.chunk_type == "table")
    content = table.content
    assert content.startswith("Table 1: Model sizes.")
    assert f"{MODEL_TEXT_LABEL} Compares model sizes" in content
    assert "As shown in Table 1" in content
    assert "[Table data]" in content and "213M" in content
    # Order: model text before context, raw data last.
    assert content.index(MODEL_TEXT_LABEL) < content.index("As shown") < content.index("[Table data]")
    assert table.metadata["model_generated_text"] is True


@pytest.fixture()
def s2b_env(tmp_path: Path):
    reset_engine()
    get_settings.cache_clear()
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 's.db').as_posix()}",
        local_storage_root=tmp_path / "store",
        embedding_provider="hashing",
        embedding_dimensions=64,
    )
    init_db(settings)
    paper = _paper()
    with session_scope(settings) as session:
        PaperRepository(session).replace_document_graph(paper, ArtifactPaths(raw_pdf="raw/x.pdf"))
    persist_chunks("s2b", chunk_paper_document(paper, config=SMALL), settings=settings)
    index_paper_chunks("s2b", settings=settings, force=True)
    yield settings
    reset_engine()
    get_settings.cache_clear()


def test_parents_are_not_embedded(s2b_env: Settings):
    with session_scope(s2b_env) as session:
        rows = [(r.chunk_type, r.embedding)
                for r in session.scalars(select(PaperChunk).where(PaperChunk.paper_id == "s2b"))]
    parents = [emb for kind, emb in rows if kind == PARENT_CHUNK_TYPE]
    children = [emb for kind, emb in rows if kind != PARENT_CHUNK_TYPE]
    assert parents and children
    assert all(emb is None for emb in parents)
    assert all(emb is not None for emb in children)


def test_retrieve_returns_parent_passages_deduplicated(s2b_env: Settings):
    out = retrieve("temperature scaling", paper_id="s2b", settings=s2b_env, top_k=3)
    results = out["results"]
    assert results
    with session_scope(s2b_env) as session:
        parent_text = {
            r.id: r.content
            for r in session.scalars(select(PaperChunk).where(PaperChunk.chunk_type == PARENT_CHUNK_TYPE))
        }
    text_units = [r for r in results if r["chunk_type"] == "text"]
    assert text_units
    for unit in text_units:
        assert unit["chunk_id"] in parent_text  # the unit is the parent ...
        assert unit["content"] == parent_text[unit["chunk_id"]]
        assert unit["matched_text"] in unit["content"]  # ... found through a child
    ids = [r["chunk_id"] for r in results]
    assert len(ids) == len(set(ids))


def test_top_k_and_evidence_budget_are_enforced(s2b_env: Settings):
    out = retrieve("calibration", paper_id="s2b", settings=s2b_env, top_k=2)
    assert len(out["results"]) <= 2
    out = retrieve("calibration", paper_id="s2b", settings=s2b_env, top_k=10, max_evidence_tokens=1)
    assert len(out["results"]) == 1  # the first unit always fits
    assert out["diagnostics"]["budget_stopped"] is True


def test_citation_labels_are_unique(s2b_env: Settings):
    out = retrieve("calibration detail reader", paper_id="s2b", settings=s2b_env, top_k=10,
                   expand_parents=False)
    labels = [r["citation"] for r in out["results"]]
    assert len(labels) > 1
    assert len(labels) == len(set(labels))
    assert all("#" in label for label in labels)


def test_child_only_mode_returns_children(s2b_env: Settings):
    out = retrieve("temperature scaling", paper_id="s2b", settings=s2b_env, top_k=3,
                   expand_parents=False)
    assert all(r["chunk_type"] != PARENT_CHUNK_TYPE for r in out["results"])
    assert all(not r["metadata"]["expanded_to_parent"] for r in out["results"])
