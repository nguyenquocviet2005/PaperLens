"""Context selection for tables, figures and formulas (explicit references first)."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

from app.ingestion.surrounding import (
    build_document_index,
    count_tokens,
    element_label,
    select_surrounding_text,
)


@dataclass
class Item:
    self_ref: str
    label: str
    text: str = ""
    page: int = 1
    parent_ref: str = "#/body"
    prov: list = field(default_factory=list)

    def __post_init__(self):
        self.prov = [SimpleNamespace(page_no=self.page)]
        self.parent = SimpleNamespace(cref=self.parent_ref)


class Doc:
    def __init__(self, items):
        self.items = items

    def iterate_items(self):
        return [(item, 0) for item in self.items]


def _doc():
    return Doc([
        Item("#/texts/0", "section_header", "3 Model Architecture", page=3),
        Item("#/texts/1", "text", "The encoder maps symbols to representations.", page=3),
        Item("#/tables/0", "table", "", page=3),
        Item("#/texts/2", "caption", "Table 1: Complexity per layer.", page=3),
        Item("#/texts/3", "text", "Inside-table note", page=3, parent_ref="#/tables/0"),
        Item("#/texts/4", "text", "Positional encodings use sine functions.", page=3),
        Item("#/texts/5", "footnote", "1 A footnote.", page=3),
        Item("#/texts/6", "text", "Unrelated text far away about optimizers.", page=7),
        Item("#/texts/7", "text",
             "Training was fast. As noted in Table 1, self-attention layers are cheaper. "
             "This holds when n is small. Another sentence.", page=6),
        Item("#/pictures/0", "picture", "", page=4),
        Item("#/texts/8", "caption", "Figure 2: Multi-head attention.", page=4),
    ])


def test_reference_sentence_comes_first_even_pages_away():
    index = build_document_index(_doc())
    out = select_surrounding_text(
        index, kind="table", caption="Table 1: Complexity per layer.",
        source_ref="#/tables/0", page=3, budget_tokens=200,
    )
    assert out[0].startswith("As noted in Table 1")
    assert "This holds when n is small." in out[0]  # plus the sentence after
    assert "Training was fast" not in out[0]


def test_only_real_paragraphs_within_one_page_are_used():
    index = build_document_index(_doc())
    out = " ".join(select_surrounding_text(
        index, kind="table", caption="Table 1: Complexity per layer.",
        source_ref="#/tables/0", page=3, budget_tokens=200,
    ))
    assert "Complexity per layer" not in out      # caption
    assert "Model Architecture" not in out         # header
    assert "footnote" not in out                   # footnote
    assert "Inside-table note" not in out          # text inside the table
    assert "optimizers" not in out                 # page 7 is too far
    assert "encoder maps" in out and "Positional encodings" in out


def test_budget_is_respected():
    index = build_document_index(_doc())
    out = select_surrounding_text(
        index, kind="table", caption="Table 1: Complexity per layer.",
        source_ref="#/tables/0", page=3, budget_tokens=25,
    )
    assert sum(count_tokens(t) for t in out) <= 25


def test_embedding_similarity_orders_the_fill():
    index = build_document_index(_doc())

    def embed(texts):  # "similar" means mentions "Positional"
        return [[1.0, 0.0] if ("Complexity" in t or "Positional" in t) else [0.0, 1.0] for t in texts]

    out = select_surrounding_text(
        index, kind="table", caption="Table 1: Complexity per layer.",
        source_ref="#/tables/0", page=3, budget_tokens=200, embed=embed,
    )
    fill = [t for t in out if not t.startswith("As noted")]
    assert fill[0].startswith("Positional encodings")


def test_embedding_failure_falls_back_to_distance():
    index = build_document_index(_doc())

    def broken(texts):
        raise RuntimeError("embedder down")

    out = select_surrounding_text(
        index, kind="table", caption="Table 1: Complexity per layer.",
        source_ref="#/tables/0", page=3, budget_tokens=200, embed=broken,
    )
    assert out  # still returns distance-ordered context


def test_uncaptioned_figure_borrows_following_caption():
    index = build_document_index(_doc())
    assert element_label(index, kind="figure", caption=None, source_ref="#/pictures/0", page=4) == (
        "figure", 2)
    assert element_label(index, kind="table", caption="Table 10: x", source_ref=None, page=1) == (
        "table", 10)
