"""Grounded single-paper QA with citation safety."""

from __future__ import annotations

import logging
import re
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from app.config import Settings, get_settings
from app.infrastructure.observability import emit_metrics, llm_callbacks, trace_config
from app.rag.query_rewrite import rewrite_query
from app.rag.retrieval import retrieve

logger = logging.getLogger(__name__)


class Citation(BaseModel):
    label: str
    paper_id: str
    chunk_id: str
    page: int | None = None
    section_path: list[str] = Field(default_factory=list)
    chunk_type: str | None = None


class EvidenceItem(BaseModel):
    chunk_id: str
    text: str
    citation: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    chunk_type: str | None = None
    score: float = 0.0
    element_id: str | None = None


class QAAnswer(BaseModel):
    answer: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    pages: list[int] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low", "insufficient"] = "insufficient"
    limitations: list[str] = Field(default_factory=list)
    used_visual_elements: list[str] = Field(default_factory=list)
    used_chunks: list[str] = Field(default_factory=list)
    mode: str = "extractive"
    diagnostics: dict[str, Any] = Field(default_factory=dict)


class GroundedSynthesis(BaseModel):
    answer: str
    confidence: Literal["high", "medium", "low", "insufficient"]
    used_citation_labels: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class ConversationTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    answer: QAAnswer | None = None


class ConversationState(BaseModel):
    conversation_id: str
    paper_id: str
    turns: list[ConversationTurn] = Field(default_factory=list)


# Conversations of the standalone /papers/{id}/qa chat live in the same tables as the
# agent's (kind="qa"), scoped to the user, instead of process memory.
QA_HISTORY_TURNS = 3


def _load_qa_history(
    conversation_id: str, *, user_id: str, paper_id: str, settings: Settings
) -> list[tuple[str, str]]:
    """(question, answer) pairs of an earlier /qa conversation about the same paper."""
    from app.db.agent_repository import AgentConversationRepository
    from app.db.session import session_scope

    try:
        with session_scope(settings) as session:
            conversation = AgentConversationRepository(session).get_with_messages(
                conversation_id, user_id=user_id
            )
            if conversation is None or conversation.kind != "qa":
                return []
            if list(conversation.selected_papers or []) != [paper_id]:
                return []
            pairs: list[tuple[str, str]] = []
            for message in conversation.messages:
                if message.role == "human":
                    pairs.append((message.content, ""))
                elif message.role == "ai" and pairs and not message.tool_calls:
                    pairs[-1] = (pairs[-1][0], message.content)
            return pairs
    except Exception:
        logger.exception("Could not load QA conversation %s", conversation_id)
        return []


def _save_qa_turn(
    conversation_id: str,
    *,
    user_id: str,
    paper_id: str,
    question: str,
    answer: QAAnswer,
    settings: Settings,
) -> None:
    from app.db.agent_repository import AgentConversationRepository
    from app.db.session import session_scope

    artifact = {
        "kind": "paper_answer",
        "paper_id": paper_id,
        "answer": answer.answer,
        "confidence": answer.confidence,
        "citations": [c.model_dump(mode="json") for c in answer.citations],
    }
    try:
        with session_scope(settings) as session:
            AgentConversationRepository(session).append_messages(
                conversation_id,
                [
                    {"role": "human", "content": question},
                    {"role": "ai", "content": answer.answer, "artifact": artifact},
                ],
                selected_papers=[paper_id],
                user_id=user_id,
                kind="qa",
                max_messages=settings.agent_history_limit,
            )
    except Exception:
        logger.exception("Could not save QA conversation %s", conversation_id)


def _insufficient(reason: str, diagnostics: dict[str, Any] | None = None) -> QAAnswer:
    return QAAnswer(
        answer=(
            "I do not have sufficient grounded evidence in the retrieved passages "
            f"to answer this question. {reason}"
        ),
        confidence="insufficient",
        limitations=[reason],
        diagnostics=diagnostics or {},
    )


def _sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text.strip())
    return [p.strip() for p in parts if p.strip()]


_WORD_RE = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> set[str]:
    return {t.lower() for t in _WORD_RE.findall(text)}


def _overlap_score(query: str, sentence: str) -> float:
    q = _words(query)
    s = _words(sentence)
    if not q or not s:
        return 0.0
    return len(q & s) / len(q)


# Digits only, so "340M", "4x" and "dk=64" still yield 340, 4 and 64.
_NUMBER_RE = re.compile(r"(?<![\d.])\d+(?:[.,]\d+)*(?!\d)")
_LABEL_RE = re.compile(r"\[[^\[\]]*\]")
_LIST_MARKER_RE = re.compile(r"(?m)^\s*\d+[.)]\s+")


def _numbers(text: str) -> list[str]:
    return [n.replace(",", "") for n in _NUMBER_RE.findall(text)]


def unsupported_numbers(answer: str, evidence_texts: list[str], question: str = "") -> list[str]:
    """Numbers stated in the answer that appear in neither the cited evidence nor the question.

    A citation label only proves the model pointed at a real passage, not that the passage
    says what the answer claims ("64 heads" citing a passage that says 8 passes a label
    check). Numbers are the cheapest claims to verify mechanically.
    """
    body = _LIST_MARKER_RE.sub("", _LABEL_RE.sub(" ", answer))
    supported = set(_numbers(" ".join(evidence_texts))) | set(_numbers(question))
    missing = [n for n in _numbers(body) if n not in supported]
    return list(dict.fromkeys(missing))


def _llm_synthesize(
    *,
    question: str,
    evidence: list["EvidenceItem"],
    settings: Settings,
    standalone_query: str | None = None,
) -> GroundedSynthesis:
    """Structured grounded answer over the already-retrieved evidence units."""
    if settings.llm_provider.lower() != "openai":
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
    if not settings.llm_api_key or not settings.llm_model:
        raise ValueError("LLM_API_KEY and LLM_MODEL are required")

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "api_key": settings.llm_api_key,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
    }
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    structured = ChatOpenAI(**kwargs, callbacks=llm_callbacks()).with_structured_output(
        GroundedSynthesis
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer only from the supplied research-paper evidence. Never use outside "
                "knowledge. Every supported statement must cite one or more citation labels "
                "exactly as written. Copy numbers exactly as they appear in the evidence. "
                "Text marked [Model-generated description] was written by a vision model "
                "and may contain mistakes; prefer [Table data] and paper text when they "
                "disagree. If the evidence does not answer the question, return "
                "confidence='insufficient'. Answer in the language of the question.",
            ),
            ("human", "Question:\n{input}{resolved}\n\nEvidence:\n{context}"),
        ]
    )
    context = "\n\n".join(f"{item.citation}\n{item.text}" for item in evidence)
    resolved = ""
    if standalone_query and standalone_query.strip() != question.strip():
        resolved = f"\n(Standalone form, from the conversation: {standalone_query})"
    result = (prompt | structured).invoke(
        {"input": question, "context": context, "resolved": resolved},
        config=trace_config("paperlens-grounded-qa", evidence_units=len(evidence)),
    )
    return GroundedSynthesis.model_validate(result)


def _remember(
    state: ConversationState, answer: QAAnswer, save: Any = None
) -> tuple[QAAnswer, ConversationState]:
    state.turns.append(ConversationTurn(role="assistant", content=answer.answer, answer=answer))
    if save is not None:
        save(answer)
    return answer, state


def _grounded_answer(
    synthesis: GroundedSynthesis,
    *,
    question: str,
    evidence: list[EvidenceItem],
    full_text: dict[str, str],
    citations: list[Citation],
    diagnostics: dict[str, Any],
    settings: Settings,
) -> QAAnswer:
    allowed_by_label = {citation.label: citation for citation in citations}
    used_labels = list(
        dict.fromkeys(label for label in synthesis.used_citation_labels if label in allowed_by_label)
    )
    if synthesis.confidence != "insufficient" and not used_labels:
        raise ValueError("LLM answer did not reference a valid evidence citation")
    safe_citations = [allowed_by_label[label] for label in used_labels]
    answer_text = synthesis.answer.strip()
    for label in used_labels:
        if label not in answer_text:
            answer_text = f"{answer_text} {label}".strip()
    used_chunk_ids = [citation.chunk_id for citation in safe_citations]
    safe_evidence = [item for item in evidence if item.chunk_id in set(used_chunk_ids)]

    confidence = synthesis.confidence
    limitations = list(synthesis.limitations)
    missing_numbers: list[str] = []
    if confidence != "insufficient":
        missing_numbers = unsupported_numbers(
            answer_text, [full_text[cid] for cid in used_chunk_ids], question
        )
        if missing_numbers:
            confidence = "low"
            limitations.append(
                "These numbers do not appear in the cited evidence: "
                + ", ".join(missing_numbers)
            )

    return QAAnswer(
        answer=answer_text,
        evidence=safe_evidence,
        citations=safe_citations,
        pages=sorted({c.page for c in safe_citations if isinstance(c.page, int)}),
        confidence=confidence,
        limitations=limitations,
        used_visual_elements=[
            str(item.element_id)
            for item in safe_evidence
            if item.chunk_type in {"table", "figure", "equation"} and item.element_id
        ],
        used_chunks=used_chunk_ids,
        mode="langchain_openai_grounded",
        diagnostics={
            **diagnostics,
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
            "unsupported_numbers": missing_numbers,
        },
    )


def _extractive_answer(
    *,
    question: str,
    evidence: list[EvidenceItem],
    full_text: dict[str, str],
    citations: list[Citation],
    diagnostics: dict[str, Any],
) -> QAAnswer:
    """Pick the evidence sentences that share the most words with the question."""
    scored: list[tuple[float, str, str]] = []
    seen: set[str] = set()
    for ev in evidence:
        for sentence in _sentence_split(full_text[ev.chunk_id]):
            key = " ".join(sentence.lower().split())
            if key in seen or len(sentence.split()) < 4:
                continue  # repeated sentences and bare headings
            seen.add(key)
            score = _overlap_score(question, sentence) + 0.05 * ev.score
            if score > 0.15:
                scored.append((score, sentence, ev.citation))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return _insufficient(
            "Retrieved chunks did not contain overlapping answer content. Without an LLM, "
            "answers are matched word-for-word, so the question must use the paper's language.",
            diagnostics,
        )

    chosen = scored[:3]
    top_score = chosen[0][0]
    confidence: Literal["high", "medium", "low", "insufficient"]
    if top_score >= 0.55 and len(chosen) >= 2:
        confidence = "high"
    elif top_score >= 0.35:
        confidence = "medium"
    else:
        confidence = "low"
    evidence_pages = {
        p for ev in evidence for p in (ev.page_start, ev.page_end) if isinstance(p, int)
    }
    return QAAnswer(
        answer=" ".join(f"{sentence} {label}" for _score, sentence, label in chosen),
        evidence=evidence,
        citations=citations,
        pages=sorted(evidence_pages),
        confidence=confidence,
        limitations=["Extractive mode: answer sentences are taken from retrieved evidence only."],
        used_visual_elements=[
            str(ev.element_id)
            for ev in evidence
            if ev.chunk_type in {"table", "figure", "equation"} and ev.element_id
        ],
        used_chunks=[ev.chunk_id for ev in evidence],
        mode="extractive",
        diagnostics=diagnostics,
    )


def answer_paper_question(
    *,
    paper_id: str,
    question: str,
    settings: Settings | None = None,
    top_k: int = 6,
    conversation_id: str | None = None,
    persist: bool = False,
    user_id: str = "local-user",
) -> tuple[QAAnswer, ConversationState]:
    """Answer using retrieved evidence only; citations come from retrieval metadata.

    `persist=True` (the /qa endpoint) loads the earlier turns of `conversation_id` to
    resolve follow-ups and stores the new turn. Tool callers (agent, research, compare)
    pass self-contained questions and keep no state here.
    """
    cfg = settings or get_settings()
    cid = conversation_id or str(uuid4())
    history = (
        _load_qa_history(cid, user_id=user_id, paper_id=paper_id, settings=cfg)
        if persist and conversation_id
        else []
    )
    state = ConversationState(conversation_id=cid, paper_id=paper_id)
    for past_question, past_answer in history:
        state.turns.append(ConversationTurn(role="user", content=past_question))
        if past_answer:
            state.turns.append(ConversationTurn(role="assistant", content=past_answer))
    state.turns.append(ConversationTurn(role="user", content=question))
    save = (
        (lambda answer: _save_qa_turn(
            cid, user_id=user_id, paper_id=paper_id, question=question, answer=answer, settings=cfg
        ))
        if persist
        else None
    )

    query, rewrite_method = rewrite_query(
        question, history[-QA_HISTORY_TURNS:], settings=cfg
    )
    retrieval = retrieve(query, paper_id=paper_id, settings=cfg, top_k=top_k)
    results = retrieval.get("results") or []
    diagnostics = dict(retrieval.get("diagnostics") or {})
    diagnostics["query_rewrite"] = rewrite_method
    if query != question:
        diagnostics["retrieval_query"] = query
    if not results:
        return _remember(
            state, _insufficient("No relevant chunks were retrieved.", diagnostics), save
        )

    evidence: list[EvidenceItem] = []
    citations: list[Citation] = []
    full_text: dict[str, str] = {}
    for item in results:
        page = item.get("page_start") or item.get("page_end")
        full_text[item["chunk_id"]] = item["content"]
        evidence.append(
            EvidenceItem(
                chunk_id=item["chunk_id"],
                text=item["content"],
                citation=item["citation"],
                page_start=item.get("page_start"),
                page_end=item.get("page_end"),
                section_path=list(item.get("section_path") or []),
                chunk_type=item.get("chunk_type"),
                score=float(item.get("score") or 0.0),
                element_id=(item.get("metadata") or {}).get("element_id"),
            )
        )
        citations.append(
            Citation(
                label=item["citation"],
                paper_id=item["paper_id"],
                chunk_id=item["chunk_id"],
                page=page if isinstance(page, int) else None,
                section_path=list(item.get("section_path") or []),
                chunk_type=item.get("chunk_type"),
            )
        )

    llm_error: str | None = None
    if cfg.llm_enabled and cfg.allow_external_api:
        try:
            synthesis = _llm_synthesize(
                question=question, evidence=evidence, settings=cfg, standalone_query=query
            )
            answer = _grounded_answer(
                synthesis,
                question=question,
                evidence=evidence,
                full_text=full_text,
                citations=citations,
                diagnostics=diagnostics,
                settings=cfg,
            )
            return _remember(state, _trim_evidence(answer), save)
        except Exception as exc:
            llm_error = type(exc).__name__
            logger.warning("Grounded LLM answer failed; using extractive fallback: %s", llm_error)

    answer = _extractive_answer(
        question=question,
        evidence=evidence,
        full_text=full_text,
        citations=citations,
        diagnostics=diagnostics,
    )
    if llm_error:
        answer.diagnostics["llm_fallback"] = llm_error
    return _remember(state, _trim_evidence(answer), save)


def _trim_evidence(answer: QAAnswer, limit: int = 1500) -> QAAnswer:
    """Parents can be long; the API returns a preview, the LLM saw the full text.

    Also the single exit of answer_paper_question, so the answer metric is emitted here.
    """
    emit_metrics(
        {
            "QAAnswers": (1, "Count"),
            "QAUnsupportedNumbers": (
                1 if answer.diagnostics.get("unsupported_numbers") else 0,
                "Count",
            ),
            "QALLMFallbacks": (1 if answer.diagnostics.get("llm_fallback") else 0, "Count"),
        },
        dimensions={"Confidence": answer.confidence, "Mode": answer.mode},
    )
    for item in answer.evidence:
        item.text = item.text[:limit]
    return answer
