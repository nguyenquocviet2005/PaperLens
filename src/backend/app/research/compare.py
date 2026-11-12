"""Multi-paper comparison with per-paper attribution."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.rag.qa import answer_paper_question


class PaperFinding(BaseModel):
    paper_id: str
    title: str | None = None
    summary: str
    confidence: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    question: str
    comparison_dimensions: list[str]
    paper_findings: list[PaperFinding]
    agreements: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)


def compare_papers(
    *,
    paper_ids: list[str],
    question: str,
    settings: Settings | None = None,
    top_k_per_paper: int = 4,
    titles: dict[str, str | None] | None = None,
) -> ComparisonResult:
    cfg = settings or get_settings()
    titles = titles or {}
    if len(paper_ids) < 2:
        raise ValueError("At least two paper_ids are required for comparison")

    findings: list[PaperFinding] = []
    evidence: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []

    for pid in paper_ids:
        answer, _state = answer_paper_question(
            paper_id=pid,
            question=question,
            settings=cfg,
            top_k=top_k_per_paper,
        )
        findings.append(
            PaperFinding(
                paper_id=pid,
                title=titles.get(pid),
                summary=answer.answer,
                confidence=answer.confidence,
                citations=[c.model_dump(mode="json") for c in answer.citations],
                pages=list(answer.pages),
                limitations=list(answer.limitations),
            )
        )
        for ev in answer.evidence:
            evidence.append({"paper_id": pid, **ev.model_dump(mode="json")})
        for c in answer.citations:
            citations.append({"paper_id": pid, **c.model_dump(mode="json")})

    # Deterministic structural comparison (no invented cross-paper claims)
    agreements: list[str] = []
    differences: list[str] = []
    contradictions: list[str] = []
    uncertainties: list[str] = []

    sufficient = [f for f in findings if f.confidence != "insufficient"]
    insufficient = [f for f in findings if f.confidence == "insufficient"]
    for f in insufficient:
        uncertainties.append(f"Insufficient evidence in paper {f.paper_id}")

    if len(sufficient) >= 2:
        # Token overlap heuristic for agreement/difference signals only
        def tokens(text: str) -> set[str]:
            return {t.lower() for t in text.split() if len(t) > 4}

        base = tokens(sufficient[0].summary)
        for other in sufficient[1:]:
            overlap = tokens(other.summary) & base
            if len(overlap) >= 5:
                agreements.append(
                    f"Papers {sufficient[0].paper_id} and {other.paper_id} share overlapping retrieved terminology related to the question."
                )
            else:
                differences.append(
                    f"Papers {sufficient[0].paper_id} and {other.paper_id} retrieve largely distinct evidence for this question."
                )
    else:
        uncertainties.append("Not enough papers had grounded answers for a firm comparison.")

    dimensions = ["problem", "method", "datasets", "metrics", "results", "limitations"]
    return ComparisonResult(
        question=question,
        comparison_dimensions=dimensions,
        paper_findings=findings,
        agreements=agreements,
        differences=differences,
        contradictions=contradictions,
        limitations=[
            "Comparison claims are restricted to per-paper retrieved evidence.",
            "Cross-paper agreements/differences are heuristic and may be incomplete.",
        ],
        evidence=evidence,
        citations=citations,
        uncertainties=uncertainties,
    )
