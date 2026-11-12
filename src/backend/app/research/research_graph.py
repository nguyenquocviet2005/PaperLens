"""Bounded LangGraph research workflow over PaperLens tools."""

from __future__ import annotations

import logging
from typing import Any, Literal, TypedDict
from uuid import uuid4

from pydantic import BaseModel, Field

from app.research.compare import compare_papers
from app.config import Settings, get_settings
from app.research.discovery import discover_papers
from app.rag.qa import answer_paper_question
from app.rag.retrieval import retrieve

logger = logging.getLogger(__name__)


class EvidenceItem(BaseModel):
    paper_id: str | None = None
    claim: str
    citation: str | None = None
    chunk_id: str | None = None
    page: int | None = None
    supported: bool = True


class ResearchState(TypedDict, total=False):
    user_id: str
    research_question: str
    clarified_question: str
    search_queries: list[str]
    candidate_papers: list[dict[str, Any]]
    selected_papers: list[str]
    evidence_items: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    critic_feedback: list[str]
    citations: list[dict[str, Any]]
    draft: str
    final_report: str
    errors: list[str]
    cost: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    step: int
    max_steps: int
    max_external_searches: int
    external_searches: int
    status: str


class ResearchReport(BaseModel):
    run_id: str
    status: str
    research_question: str
    clarified_question: str
    selected_papers: list[str] = Field(default_factory=list)
    final_report: str
    claims: list[dict[str, Any]] = Field(default_factory=list)
    critic_feedback: list[str] = Field(default_factory=list)
    citations: list[dict[str, Any]] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    evidence_items: list[dict[str, Any]] = Field(default_factory=list)


def _record_tool(state: ResearchState, name: str, detail: dict[str, Any]) -> None:
    calls = list(state.get("tool_calls") or [])
    calls.append({"name": name, **detail})
    state["tool_calls"] = calls


def node_clarify(state: ResearchState) -> ResearchState:
    q = (state.get("research_question") or "").strip()
    clarified = q
    if len(q.split()) < 4:
        clarified = f"Provide a grounded literature-oriented answer to: {q}"
    state["clarified_question"] = clarified
    state["search_queries"] = [clarified, f"{clarified} limitations", f"{clarified} methods"]
    state["step"] = int(state.get("step") or 0) + 1
    _record_tool(state, "clarify", {"clarified_question": clarified})
    return state


def node_search_library(state: ResearchState, settings: Settings) -> ResearchState:
    from sqlalchemy import select
    from app.db.models import Paper
    from app.db.session import session_scope

    selected = list(state.get("selected_papers") or [])
    with session_scope(settings) as session:
        stmt = select(Paper).where(Paper.status == "completed")
        user_id = state.get("user_id")
        if user_id:
            stmt = stmt.where(Paper.user_id == user_id)
        papers = list(session.scalars(stmt))
        summaries = [
            {"paper_id": p.id, "title": p.title, "filename": p.filename} for p in papers
        ]
        if not selected:
            selected = [row["paper_id"] for row in summaries[:3]]
    state["selected_papers"] = selected
    state["candidate_papers"] = summaries
    state["step"] = int(state.get("step") or 0) + 1
    _record_tool(state, "search_library", {"selected": selected, "library_size": len(summaries)})
    return state


def node_external_discovery(state: ResearchState, settings: Settings) -> ResearchState:
    if int(state.get("external_searches") or 0) >= int(state.get("max_external_searches") or 0):
        _record_tool(state, "external_discovery", {"skipped": True, "reason": "budget"})
        state["step"] = int(state.get("step") or 0) + 1
        return state
    query = (state.get("search_queries") or [state.get("clarified_question") or ""])[0]
    try:
        result = discover_papers(query, source="arxiv", limit=5, settings=settings)
        cands = list(state.get("candidate_papers") or [])
        for item in result.results:
            cands.append(item.model_dump(mode="json"))
        state["candidate_papers"] = cands
        state["external_searches"] = int(state.get("external_searches") or 0) + 1
        _record_tool(state, "external_discovery", {"count": result.count, "cached": result.cached})
    except Exception as exc:
        errors = list(state.get("errors") or [])
        errors.append(f"external_discovery:{exc}")
        state["errors"] = errors
        _record_tool(state, "external_discovery", {"error": str(exc)})
    state["step"] = int(state.get("step") or 0) + 1
    return state


def node_retrieve_evidence(state: ResearchState, settings: Settings) -> ResearchState:
    question = state.get("clarified_question") or state.get("research_question") or ""
    evidence: list[dict[str, Any]] = []
    citations: list[dict[str, Any]] = []
    for paper_id in state.get("selected_papers") or []:
        out = retrieve(
            question,
            paper_id=paper_id,
            settings=settings,
            top_k=4,
            user_id=state.get("user_id"),
        )
        for item in out.get("results") or []:
            evidence.append(
                {
                    "paper_id": paper_id,
                    "claim": (item.get("matched_text") or item.get("content", ""))[:500],
                    "citation": item.get("citation"),
                    "chunk_id": item.get("chunk_id"),
                    "page": item.get("page_start"),
                    "supported": True,
                }
            )
            citations.append(
                {
                    "paper_id": paper_id,
                    "label": item.get("citation"),
                    "chunk_id": item.get("chunk_id"),
                    "page": item.get("page_start"),
                }
            )
        _record_tool(state, "retrieve", {"paper_id": paper_id, "n": len(out.get("results") or [])})
    state["evidence_items"] = evidence
    state["citations"] = citations
    state["step"] = int(state.get("step") or 0) + 1
    return state


def node_analyze(state: ResearchState, settings: Settings) -> ResearchState:
    question = state.get("clarified_question") or ""
    claims: list[dict[str, Any]] = []
    draft_parts: list[str] = [f"# Research note\n\nQuestion: {question}\n"]
    for paper_id in state.get("selected_papers") or []:
        answer, _ = answer_paper_question(
            paper_id=paper_id, question=question, settings=settings, top_k=4
        )
        claims.append(
            {
                "paper_id": paper_id,
                "text": answer.answer,
                "confidence": answer.confidence,
                "chunk_ids": list(answer.used_chunks),
                "pages": list(answer.pages),
            }
        )
        draft_parts.append(f"## Paper `{paper_id}`\n\n{answer.answer}\n")
        _record_tool(state, "analyze_paper", {"paper_id": paper_id, "confidence": answer.confidence})
    state["claims"] = claims
    state["draft"] = "\n".join(draft_parts)
    state["step"] = int(state.get("step") or 0) + 1
    return state


def node_verify_claims(state: ResearchState) -> ResearchState:
    """Evidence critic: reject claims without chunk ids / pages."""
    feedback: list[str] = []
    verified: list[dict[str, Any]] = []
    evidence_chunk_ids = {e.get("chunk_id") for e in state.get("evidence_items") or [] if e.get("chunk_id")}
    for claim in state.get("claims") or []:
        chunk_ids = set(claim.get("chunk_ids") or [])
        if not chunk_ids:
            feedback.append(f"Rejected claim for {claim.get('paper_id')}: no chunk ids")
            continue
        if evidence_chunk_ids and not (chunk_ids & evidence_chunk_ids):
            # still allow if QA used chunks even if retrieve set differs slightly
            feedback.append(
                f"Warning for {claim.get('paper_id')}: claim chunks not in retrieve set"
            )
        if claim.get("confidence") == "insufficient":
            feedback.append(f"Rejected claim for {claim.get('paper_id')}: insufficient evidence")
            continue
        verified.append({**claim, "supported": True})
    state["claims"] = verified
    state["critic_feedback"] = feedback
    state["step"] = int(state.get("step") or 0) + 1
    _record_tool(state, "verify_claims", {"accepted": len(verified), "feedback": feedback})
    return state


def node_synthesize(state: ResearchState, settings: Settings) -> ResearchState:
    selected = state.get("selected_papers") or []
    question = state.get("clarified_question") or ""
    lines = [
        "# Research report",
        "",
        f"**Question:** {question}",
        "",
        "## Supported findings",
    ]
    for claim in state.get("claims") or []:
        pages = claim.get("pages") or []
        page_bit = f" pages={pages}" if pages else ""
        lines.append(
            f"- ({claim.get('paper_id')}){page_bit} {claim.get('text')}"
        )
    if len(selected) >= 2:
        try:
            comparison = compare_papers(
                paper_ids=selected[:3], question=question, settings=settings, top_k_per_paper=3
            )
            lines.append("")
            lines.append("## Comparison notes")
            for item in comparison.agreements:
                lines.append(f"- Agreement: {item}")
            for item in comparison.differences:
                lines.append(f"- Difference: {item}")
            for item in comparison.uncertainties:
                lines.append(f"- Uncertainty: {item}")
            _record_tool(state, "compare", {"papers": selected[:3]})
        except Exception as exc:
            errors = list(state.get("errors") or [])
            errors.append(f"compare:{exc}")
            state["errors"] = errors
    if state.get("critic_feedback"):
        lines.append("")
        lines.append("## Critic feedback")
        for fb in state["critic_feedback"]:
            lines.append(f"- {fb}")
    lines.append("")
    lines.append("## Citations")
    for c in state.get("citations") or []:
        lines.append(
            f"- paper={c.get('paper_id')} {c.get('label')} chunk={c.get('chunk_id')}"
        )
    state["final_report"] = "\n".join(lines)
    state["status"] = "completed"
    state["step"] = int(state.get("step") or 0) + 1
    _record_tool(state, "synthesize", {"chars": len(state["final_report"])})
    return state


def route_after_library(state: ResearchState) -> Literal["external_discovery", "retrieve_evidence"]:
    if (state.get("selected_papers") or []) and int(state.get("external_searches") or 0) >= int(
        state.get("max_external_searches") or 0
    ):
        return "retrieve_evidence"
    if int(state.get("step") or 0) >= int(state.get("max_steps") or 12):
        return "retrieve_evidence"
    # External discovery optional once
    if int(state.get("external_searches") or 0) < int(state.get("max_external_searches") or 0):
        return "external_discovery"
    return "retrieve_evidence"


def build_research_graph(settings: Settings | None = None):
    from langgraph.graph import END, StateGraph

    cfg = settings or get_settings()
    graph = StateGraph(ResearchState)
    graph.add_node("clarify", node_clarify)
    graph.add_node("search_library", lambda s: node_search_library(s, cfg))
    graph.add_node("external_discovery", lambda s: node_external_discovery(s, cfg))
    graph.add_node("retrieve_evidence", lambda s: node_retrieve_evidence(s, cfg))
    graph.add_node("analyze", lambda s: node_analyze(s, cfg))
    graph.add_node("verify_claims", node_verify_claims)
    graph.add_node("synthesize", lambda s: node_synthesize(s, cfg))

    graph.set_entry_point("clarify")
    graph.add_edge("clarify", "search_library")
    graph.add_conditional_edges(
        "search_library",
        route_after_library,
        {
            "external_discovery": "external_discovery",
            "retrieve_evidence": "retrieve_evidence",
        },
    )
    graph.add_edge("external_discovery", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "analyze")
    graph.add_edge("analyze", "verify_claims")
    graph.add_edge("verify_claims", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


def run_research(
    research_question: str,
    *,
    selected_papers: list[str] | None = None,
    settings: Settings | None = None,
    max_steps: int = 12,
    max_external_searches: int = 0,
    enable_external: bool = False,
    user_id: str = "local-user",
) -> ResearchReport:
    cfg = settings or get_settings()
    app = build_research_graph(cfg)
    run_id = str(uuid4())
    initial: ResearchState = {
        "user_id": user_id,
        "research_question": research_question,
        "selected_papers": list(selected_papers or []),
        "candidate_papers": [],
        "evidence_items": [],
        "claims": [],
        "critic_feedback": [],
        "citations": [],
        "errors": [],
        "tool_calls": [],
        "cost": {
            "mode": (
                "langgraph_with_langchain_llm"
                if cfg.llm_enabled and cfg.allow_external_api
                else "langgraph_local_extractive"
            )
        },
        "step": 0,
        "max_steps": max_steps,
        "max_external_searches": max_external_searches if enable_external else 0,
        "external_searches": 0,
        "status": "running",
    }
    final = app.invoke(initial)
    return ResearchReport(
        run_id=run_id,
        status=str(final.get("status") or "completed"),
        research_question=research_question,
        clarified_question=str(final.get("clarified_question") or research_question),
        selected_papers=list(final.get("selected_papers") or []),
        final_report=str(final.get("final_report") or final.get("draft") or ""),
        claims=list(final.get("claims") or []),
        critic_feedback=list(final.get("critic_feedback") or []),
        citations=list(final.get("citations") or []),
        tool_calls=list(final.get("tool_calls") or []),
        errors=list(final.get("errors") or []),
        evidence_items=list(final.get("evidence_items") or []),
    )
