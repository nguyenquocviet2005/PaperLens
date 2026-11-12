"""Context paragraphs for tables, figures and formulas.

Tables and figures are LaTeX floats: they often sit pages away from the paragraph that
discusses them, so "the text items next to the element" is mostly captions, headers and
unrelated prose. Measured on a 84-question benchmark, that approach scored below having no
context at all. The selection here:

1. Every sentence anywhere in the paper that mentions the element ("Table 2", "Fig. 3"),
   plus the sentence after it.
2. The remaining budget is filled with real paragraphs within one page of the element,
   ranked by embedding similarity to the caption (or by reading-order distance when there
   is no caption or embeddings fail).
3. Captions, headers, footnotes and text inside other figures/tables are never used.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from functools import lru_cache
from itertools import zip_longest
from typing import Any, Callable, Sequence

logger = logging.getLogger(__name__)

_PARAGRAPH_LABELS = {"text", "list_item", "paragraph"}
_CAPTION_RE = re.compile(r"^\s*(table|tab\.|figure|fig\.)\s*(\d+)", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")

EmbedFn = Callable[[Sequence[str]], list[list[float]]]


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("gpt2")


def count_tokens(text: str) -> int:
    return len(_encoding().encode(text or ""))


def _trim_to_tokens(text: str, budget: int) -> str:
    enc = _encoding()
    return enc.decode(enc.encode(text)[:budget])


def _label(item: Any) -> str:
    label = getattr(item, "label", None)
    return str(getattr(label, "value", label))


def _page(item: Any) -> int | None:
    prov = getattr(item, "prov", None) or []
    return getattr(prov[0], "page_no", None) if prov else None


def _text(item: Any) -> str:
    return str(getattr(item, "text", "") or "").strip()


@dataclass
class DocumentIndex:
    items: list[Any]
    position: dict[str, int]
    paragraphs: list[int]
    _vectors: dict[int, list[float]] = field(default_factory=dict)


def build_document_index(document: Any) -> DocumentIndex:
    items = [item for item, _level in document.iterate_items()]
    position = {
        str(item.self_ref): i for i, item in enumerate(items) if getattr(item, "self_ref", None)
    }

    def is_paragraph(item: Any) -> bool:
        parent = getattr(getattr(item, "parent", None), "cref", "") or ""
        return (
            _label(item) in _PARAGRAPH_LABELS
            and bool(_text(item))
            and not parent.startswith(("#/pictures", "#/tables"))
        )

    paragraphs = [i for i, item in enumerate(items) if is_paragraph(item)]
    return DocumentIndex(items=items, position=position, paragraphs=paragraphs)


def element_label(
    index: DocumentIndex, *, kind: str, caption: str | None, source_ref: str | None, page: int | None
) -> tuple[str, int] | None:
    """('table'|'figure', number) from the caption.

    Docling sometimes splits one figure into several pictures and attaches the caption to
    the last one, so an uncaptioned figure borrows a caption that follows it on its page.
    """
    match = _CAPTION_RE.match(caption or "")
    if not match and kind == "figure" and source_ref in index.position:
        i = index.position[source_ref]
        for item in index.items[i + 1 : i + 4]:
            if _label(item) == "caption" and _page(item) == page:
                match = _CAPTION_RE.match(_text(item))
                break
    if not match:
        return None
    found_kind = "table" if match.group(1).lower().startswith("tab") else "figure"
    return found_kind, int(match.group(2))


def _mention_regex(kind: str, number: int) -> re.Pattern[str]:
    words = r"(?:tables?|tab\.)" if kind == "table" else r"(?:figures?|figs?\.)"
    return re.compile(rf"\b{words}\s*{number}(?!\d)", re.IGNORECASE)


def _mention_snippet(text: str, pattern: re.Pattern[str]) -> str:
    sentences = _SENTENCE_SPLIT.split(text)
    hits = [i for i, sentence in enumerate(sentences) if pattern.search(sentence)]
    keep = sorted({j for i in hits for j in (i, i + 1) if j < len(sentences)})
    return " ".join(sentences[j] for j in keep)


def _nearest_paragraphs(
    index: DocumentIndex, source_ref: str | None, page: int | None, max_page_gap: int = 1
) -> list[int]:
    """Paragraph positions by reading-order distance, alternating before and after."""
    i = index.position.get(source_ref or "")
    if i is None:
        return []

    def close(j: int) -> bool:
        other = _page(index.items[j])
        return page is None or other is None or abs(other - page) <= max_page_gap

    before = [j for j in reversed(index.paragraphs) if j < i and close(j)]
    after = [j for j in index.paragraphs if j > i and close(j)]
    return [j for pair in zip_longest(before, after) for j in pair if j is not None]


def _fit_budget(snippets: list[str], budget: int) -> list[str]:
    out: list[str] = []
    used = 0
    for snippet in snippets:
        n = count_tokens(snippet)
        if used + n <= budget:
            out.append(snippet)
            used += n
        elif not out:
            out.append(_trim_to_tokens(snippet, budget))
            break
    return out


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _rank_by_similarity(
    index: DocumentIndex, caption: str, candidates: list[int], embed: EmbedFn
) -> list[int]:
    missing = [j for j in candidates if j not in index._vectors]
    texts = [caption, *(_text(index.items[j]) for j in missing)]
    vectors = embed(texts)
    query = vectors[0]
    for j, vector in zip(missing, vectors[1:]):
        index._vectors[j] = vector
    return sorted(candidates, key=lambda j: -_cosine(query, index._vectors[j]))


def select_surrounding_text(
    index: DocumentIndex,
    *,
    kind: str,
    caption: str | None,
    source_ref: str | None,
    page: int | None,
    budget_tokens: int = 200,
    embed: EmbedFn | None = None,
) -> list[str]:
    near = _nearest_paragraphs(index, source_ref, page)
    references: list[tuple[int, str]] = []
    label = element_label(index, kind=kind, caption=caption, source_ref=source_ref, page=page)
    if label:
        pattern = _mention_regex(*label)
        references = [
            (j, _mention_snippet(_text(index.items[j]), pattern))
            for j in index.paragraphs
            if pattern.search(_text(index.items[j]))
        ]
    referenced = {j for j, _ in references}
    fill = [j for j in near if j not in referenced]
    if embed is not None and caption and fill:
        try:
            fill = _rank_by_similarity(index, caption, fill, embed)
        except Exception as exc:  # embedding outage: keep the distance order
            logger.warning("Surrounding-text ranking fell back to distance: %s", type(exc).__name__)
    snippets = [snippet for _, snippet in references] + [_text(index.items[j]) for j in fill]
    return _fit_budget(snippets, budget_tokens)
