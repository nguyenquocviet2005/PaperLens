"""Structure-aware hierarchical chunking for research papers."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.db.models import PaperChunk
from app.db.repository import PaperRepository
from app.db.session import session_scope
from app.config import Settings, get_settings
from app.schemas import PaperDocument, TextElement, VisualElement

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    """Fast approximate token count (whitespace/punctuation aware)."""
    if not text:
        return 0
    return max(1, len(_TOKEN_RE.findall(text)))


@dataclass
class ChunkDraft:
    content: str
    chunk_type: str
    section_path: list[str]
    page_start: int | None
    page_end: int | None
    token_count: int
    parent_key: str | None = None
    element_ids: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkingResult:
    parent_chunks: list[ChunkDraft]
    child_chunks: list[ChunkDraft]
    metrics: dict[str, Any]


@dataclass
class ChunkingConfig:
    parent_min_tokens: int = 800
    parent_max_tokens: int = 1800
    child_min_tokens: int = 250
    child_max_tokens: int = 450
    overlap_tokens: int = 55
    # A short last child is merged into the previous one only if the result stays under
    # this size; bge-small (the production embedder) reads at most 512 tokens.
    child_hard_max_tokens: int = 500
    # Raw table data appended to table chunks (characters).
    table_data_max_chars: int = 4000


PARENT_CHUNK_TYPE = "parent_passage"
VISUAL_CHUNK_TYPES = frozenset({"table", "figure", "equation"})
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
_SPLITTER = "langchain_recursive_tiktoken"


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("gpt2")  # the encoding LangChain's tiktoken splitter uses


def _bpe_tokens(text: str) -> int:
    return len(_encoding().encode(text or ""))


def _group_text_by_section(elements: Iterable[TextElement]) -> list[tuple[tuple[str, ...], list[TextElement]]]:
    groups: list[tuple[tuple[str, ...], list[TextElement]]] = []
    current_path: tuple[str, ...] | None = None
    bucket: list[TextElement] = []
    for el in elements:
        path = tuple(el.section_path)
        if current_path is None:
            current_path = path
            bucket = [el]
            continue
        if path != current_path:
            groups.append((current_path, bucket))
            current_path = path
            bucket = [el]
        else:
            bucket.append(el)
    if current_path is not None and bucket:
        groups.append((current_path, bucket))
    return groups


def _pages_of(elements: list[TextElement]) -> tuple[int | None, int | None]:
    pages = [el.page for el in elements if el.page is not None]
    if not pages:
        return None, None
    return min(pages), max(pages)


@dataclass
class _ParentDraft:
    sections: list[list[str]] = field(default_factory=list)
    elements: list[TextElement] = field(default_factory=list)
    text: str = ""
    tokens: int = 0

    def absorb(self, sections: list[list[str]], elements: list[TextElement], text: str) -> None:
        self.sections.extend(sections)
        self.elements.extend(elements)
        self.text = f"{self.text}\n\n{text}".strip()
        self.tokens = _bpe_tokens(self.text)


def _build_parents(paper: PaperDocument, cfg: ChunkingConfig) -> list[_ParentDraft]:
    """Merge consecutive short sections until a parent reaches parent_min_tokens.

    Never merges past parent_max_tokens; a single section above the maximum is split on
    its own. A short final parent is folded into the previous one when it fits.
    """
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=cfg.parent_max_tokens, chunk_overlap=0, separators=_SEPARATORS
    )
    parents: list[_ParentDraft] = []
    current: _ParentDraft | None = None
    for path, elems in _group_text_by_section(paper.text_elements):
        body = "\n\n".join(el.text for el in elems if el.text).strip()
        if not body:
            continue
        size = _bpe_tokens(body)
        if size > cfg.parent_max_tokens:
            if current:
                parents.append(current)
                current = None
            for piece in splitter.split_text(body):
                part = _ParentDraft()
                part.absorb([list(path)], [el for el in elems if el.text and el.text[:60] in piece] or elems, piece)
                parents.append(part)
            continue
        if current and (
            current.tokens >= cfg.parent_min_tokens
            or current.tokens + size > cfg.parent_max_tokens
        ):
            parents.append(current)
            current = None
        current = current or _ParentDraft()
        current.absorb([list(path)], elems, body)
    if current:
        parents.append(current)
    if (
        len(parents) > 1
        and parents[-1].tokens < cfg.parent_min_tokens
        and parents[-2].tokens + parents[-1].tokens <= cfg.parent_max_tokens
    ):
        tail = parents.pop()
        parents[-1].absorb(tail.sections, tail.elements, tail.text)
    return parents


def _child_pages(text: str, elements: list[TextElement]) -> tuple[int | None, int | None]:
    """Pages of the elements whose text appears in this child (citations stay precise
    even when a merged parent spans several pages)."""
    inside = [el for el in elements if el.text and el.text[:60] in text]
    return _pages_of(inside or elements)


def _split_children(text: str, cfg: ChunkingConfig) -> list[str]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=cfg.child_max_tokens, chunk_overlap=cfg.overlap_tokens, separators=_SEPARATORS
    )
    pieces = splitter.split_text(text)
    if len(pieces) > 1 and _bpe_tokens(pieces[-1]) < cfg.child_min_tokens:
        merged = f"{pieces[-2]} {pieces[-1]}"
        if _bpe_tokens(merged) <= cfg.child_hard_max_tokens:
            pieces[-2:] = [merged]
    return pieces


MODEL_TEXT_LABEL = "[Model-generated description]"


def luna_summary(visual: VisualElement) -> str:
    """Short, labelled text from a completed Luna enrichment (empty when there is none)."""
    enrichment = visual.enrichment
    result = enrichment.result if enrichment and enrichment.status == "completed" else None
    if not result:
        return ""
    if visual.type == "table":
        parts = [result.get("table_purpose", ""), *list(result.get("main_results") or [])[:4]]
    elif visual.type == "figure":
        parts = [result.get("description", ""), result.get("main_message", "")]
    else:
        parts = [result.get("latex", ""), result.get("plain_reading", ""), result.get("explanation", "")]
    text = " ".join(str(p) for p in parts if p)
    return f"{MODEL_TEXT_LABEL} {text}" if text else ""


def visual_chunk_content(visual: VisualElement, cfg: ChunkingConfig) -> str:
    """Caption, transcription, Luna description, context paragraphs, then raw table data.

    The order matters: the embedder reads only the first ~512 tokens, so the most
    descriptive parts come first and long table data last.
    """
    formula_text = visual.docling_text if visual.type == "formula" else ""
    table_data = ""
    if visual.type == "table" and visual.docling_text:
        table_data = "[Table data]\n" + visual.docling_text[: cfg.table_data_max_chars]
    parts = [
        visual.caption or "",
        formula_text or "",
        luna_summary(visual),
        " ".join(visual.surrounding_text or []),
        table_data,
    ]
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


def chunk_paper_document(
    paper: PaperDocument,
    *,
    config: ChunkingConfig | None = None,
) -> ChunkingResult:
    """Small-to-big chunks: parents are stored for the LLM, children are indexed.

    Text: consecutive sections are merged into parents of parent_min..parent_max tokens,
    and each parent is split into children of at most child_max tokens. Tables, figures
    and equations are standalone indexed chunks (their own "parent").
    """
    cfg = config or ChunkingConfig()
    parents: list[ChunkDraft] = []
    children: list[ChunkDraft] = []

    drafts = _build_parents(paper, cfg)
    for parent_idx, draft in enumerate(drafts, start=1):
        parent_key = f"parent_{parent_idx:04d}"
        page_start, page_end = _pages_of(draft.elements)
        element_ids = list(dict.fromkeys(el.element_id for el in draft.elements))
        section_path = draft.sections[0] if draft.sections else []
        parents.append(
            ChunkDraft(
                content=draft.text,
                chunk_type=PARENT_CHUNK_TYPE,
                section_path=section_path,
                page_start=page_start,
                page_end=page_end,
                token_count=estimate_tokens(draft.text),
                element_ids=element_ids,
                metadata={
                    "paper_id": paper.paper_id,
                    "parent_key": parent_key,
                    "sections": [" > ".join(s) for s in draft.sections],
                    "bpe_tokens": draft.tokens,
                    "splitter": _SPLITTER,
                },
            )
        )
        for piece in _split_children(draft.text, cfg):
            child_start, child_end = _child_pages(piece, draft.elements)
            children.append(
                ChunkDraft(
                    content=piece,
                    chunk_type="text",
                    section_path=section_path,
                    page_start=child_start,
                    page_end=child_end,
                    token_count=estimate_tokens(piece),
                    parent_key=parent_key,
                    element_ids=element_ids,
                    metadata={
                        "paper_id": paper.paper_id,
                        "parent_key": parent_key,
                        "splitter": _SPLITTER,
                    },
                )
            )

    def add_visual(visual: VisualElement, chunk_type: str) -> None:
        # Figure-internal OCR is kept in metadata, not mixed unmarked into the body.
        internal = list(visual.internal_text or [])
        content = visual_chunk_content(visual, cfg) or f"{chunk_type} on page {visual.page}"
        meta: dict[str, Any] = {
            "paper_id": paper.paper_id,
            "element_id": visual.element_id,
            "image_uri": visual.image_uri,
            "structured_data_uri": visual.structured_data_uri,
            "needs_enrichment": visual.needs_enrichment,
            "source_ref": visual.source_ref,
            "model_generated_text": bool(luna_summary(visual)),
        }
        if internal:
            meta["figure_internal_text"] = internal
            meta["figure_internal_marked"] = True
        children.append(
            ChunkDraft(
                content=content,
                chunk_type=chunk_type,
                section_path=list(visual.section_path or []),
                page_start=visual.page,
                page_end=visual.page,
                token_count=estimate_tokens(content),
                parent_key=None,
                element_ids=[visual.element_id],
                metadata=meta,
            )
        )

    for table in paper.tables:
        add_visual(table, "table")
    for figure in paper.figures:
        add_visual(figure, "figure")
    for formula in paper.formulas:
        add_visual(formula, "equation")

    covered = {eid for p in parents for eid in p.element_ids}
    orphaned = [el.element_id for el in paper.text_elements if el.text and el.element_id not in covered]
    parent_sizes = [d.tokens for d in drafts] or [0]
    child_sizes = [_bpe_tokens(c.content) for c in children if c.chunk_type == "text"] or [0]
    metrics = {
        "parent_count": len(parents),
        "child_count": len(children),
        "indexed_chunks": len(children),
        "total_chunks": len(parents) + len(children),
        "parent_tokens": {"min": min(parent_sizes), "max": max(parent_sizes),
                          "avg": round(sum(parent_sizes) / len(parent_sizes), 1)},
        "child_tokens": {"min": min(child_sizes), "max": max(child_sizes),
                         "avg": round(sum(child_sizes) / len(child_sizes), 1)},
        "orphaned_text_elements": len(orphaned),
        "orphaned_ids_sample": orphaned[:10],
        "visual_chunks": sum(1 for c in children if c.chunk_type in VISUAL_CHUNK_TYPES),
        "visual_chunks_with_model_text": sum(
            1 for c in children if c.metadata.get("model_generated_text")
        ),
        "section_groups": len(_group_text_by_section(paper.text_elements)),
        "splitter": _SPLITTER,
        "strategy": "small_to_big",
    }
    return ChunkingResult(parent_chunks=parents, child_chunks=children, metrics=metrics)


def persist_chunks(
    paper_id: str,
    result: ChunkingResult,
    *,
    settings: Settings | None = None,
) -> int:
    """Replace stored chunks for a paper. Returns inserted count."""
    cfg = settings or get_settings()
    inserted = 0
    with session_scope(cfg) as session:
        repo = PaperRepository(session)
        paper = repo.get_paper(paper_id)
        if paper is None:
            raise ValueError(f"Unknown paper_id for chunking: {paper_id}")
        paper.chunks.clear()
        session.flush()

        parent_db_ids: dict[str, str] = {}
        for parent in result.parent_chunks:
            key = str(parent.metadata.get("parent_key"))
            row = PaperChunk(
                paper_id=paper_id,
                parent_chunk_id=None,
                section_path=parent.section_path,
                page_start=parent.page_start,
                page_end=parent.page_end,
                content=parent.content,
                chunk_type=parent.chunk_type,
                token_count=parent.token_count,
                chunk_metadata={**parent.metadata, "element_ids": parent.element_ids},
            )
            session.add(row)
            session.flush()
            parent_db_ids[key] = row.id
            inserted += 1

        for child in result.child_chunks:
            parent_key = child.parent_key or child.metadata.get("parent_key")
            parent_id = parent_db_ids.get(str(parent_key)) if parent_key else None
            session.add(
                PaperChunk(
                    paper_id=paper_id,
                    parent_chunk_id=parent_id,
                    section_path=child.section_path,
                    page_start=child.page_start,
                    page_end=child.page_end,
                    content=child.content,
                    chunk_type=child.chunk_type,
                    token_count=child.token_count,
                    chunk_metadata={**child.metadata, "element_ids": child.element_ids},
                )
            )
            inserted += 1
        session.flush()
    logger.info("Persisted %s chunks for paper %s", inserted, paper_id)
    return inserted
