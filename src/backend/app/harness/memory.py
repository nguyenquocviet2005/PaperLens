"""Context engineering for the research agent.

The prompt is assembled from layers, each with its own budget, instead of "the last N
messages verbatim":

    system prompt + tools          fixed, first (stable prefix for provider prompt caching)
    conversation memory            appended to the system prompt: state + rolling summary
                                   (one system message: some models reject a second one)
    recent turns                   whole turns, newest first until the token budget is spent,
                                   with tool results compacted to answer + citations
    current turn                   the new message; tool results in it stay complete

Rules that keep this safe:
- History is cut on turn boundaries (a turn starts at a human message), so a tool result
  is never separated from the tool call that produced it.
- The full history stays in the database; trimming happens only when building a prompt.
- The summary and state help the agent resolve references ("that table"). They are
  never a source: claims about a paper still need a fresh retrieval with citations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from app.config import Settings
from app.harness.context import content_text
from app.rag.query_rewrite import detect_language

logger = logging.getLogger(__name__)

MEMORY_HEADER = (
    "Conversation memory. Use it only to understand references to earlier turns. It is "
    "not evidence: never cite it, and call a tool again for facts about a paper."
)


def count_tokens(text: str) -> int:
    from app.ingestion.surrounding import count_tokens as _count

    return _count(text)


def message_tokens(message: BaseMessage) -> int:
    text = content_text(message.content)
    if isinstance(message, AIMessage) and message.tool_calls:
        text += json.dumps([c.get("args") for c in message.tool_calls], default=str)
    return count_tokens(text) + 4  # role/format overhead


def split_turns(messages: list[BaseMessage]) -> list[list[BaseMessage]]:
    """Group messages into turns; each turn starts at a human message.

    Messages before the first human message (only possible in legacy data trimmed by
    message count) are dropped: they may be tool results without their tool call.
    """
    turns: list[list[BaseMessage]] = []
    for message in messages:
        if isinstance(message, HumanMessage):
            turns.append([message])
        elif turns:
            turns[-1].append(message)
    return [_drop_unanswered_tool_calls(turn) for turn in turns]


def _drop_unanswered_tool_calls(turn: list[BaseMessage]) -> list[BaseMessage]:
    """A turn interrupted mid-tool-call would also be rejected by the provider."""
    answered = {m.tool_call_id for m in turn if isinstance(m, ToolMessage)}
    called: set[str] = set()
    cleaned: list[BaseMessage] = []
    for message in turn:
        if isinstance(message, AIMessage) and message.tool_calls:
            if not all(call.get("id") in answered for call in message.tool_calls):
                break  # the rest of the turn is incomplete
            called.update(call.get("id") for call in message.tool_calls)
        if isinstance(message, ToolMessage) and message.tool_call_id not in called:
            continue
        cleaned.append(message)
    return cleaned


def compact_tool_message(message: ToolMessage, max_chars: int = 600) -> ToolMessage:
    """Keep what later turns need (answer, confidence, citations); drop raw evidence.

    The full artifact stays in the database; if the user asks about the same passage
    again, the agent retrieves it again.
    """
    artifact = message.artifact if isinstance(message.artifact, dict) else None
    if artifact and artifact.get("kind") == "paper_answer":
        compact = {
            "kind": "paper_answer",
            "paper_id": artifact.get("paper_id"),
            "answer": artifact.get("answer"),
            "confidence": artifact.get("confidence"),
            "citations": [c.get("label") for c in artifact.get("citations") or []],
            "note": "evidence omitted from history; call ask_paper again for details",
        }
        content = json.dumps(compact, ensure_ascii=False)
    else:
        content = content_text(message.content)
        if len(content) > max_chars:
            content = content[:max_chars] + " …[truncated from history]"
    return ToolMessage(
        content=content,
        name=message.name,
        tool_call_id=message.tool_call_id,
        status=message.status,
    )


def compact_turn(turn: list[BaseMessage]) -> list[BaseMessage]:
    return [compact_tool_message(m) if isinstance(m, ToolMessage) else m for m in turn]


def turn_brief(turn: list[BaseMessage], max_chars: int = 160) -> str:
    """One line per turn (question -> answer), used when no LLM summary covers it."""
    question = content_text(turn[0].content).strip().replace("\n", " ")
    answer = next(
        (
            content_text(m.content).strip().replace("\n", " ")
            for m in reversed(turn)
            if isinstance(m, AIMessage) and not m.tool_calls and content_text(m.content).strip()
        ),
        "",
    )
    tools = [m.name for m in turn if isinstance(m, ToolMessage)]
    line = f"User: {question[:max_chars]}"
    if tools:
        line += f" | tools: {', '.join(dict.fromkeys(tools))}"
    if answer:
        line += f" | Assistant: {answer[:max_chars]}"
    return line


def update_state(
    state: dict[str, Any] | None,
    turn: list[BaseMessage],
    *,
    selected_papers: list[str],
    paper_titles: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Deterministic, structured facts about the conversation (no model call)."""
    new = dict(state or {})
    titles = dict(new.get("paper_titles") or {})
    titles.update({k: v for k, v in (paper_titles or {}).items() if v})
    discussed = list(new.get("papers_discussed") or [])
    citations: list[str] = []
    for message in turn:
        if isinstance(message, ToolMessage) and isinstance(message.artifact, dict):
            artifact = message.artifact
            paper_id = artifact.get("paper_id")
            if paper_id and paper_id not in discussed:
                discussed.append(paper_id)
            if artifact.get("title") and paper_id:
                titles[paper_id] = str(artifact["title"])
            citations += [c.get("label") for c in artifact.get("citations") or [] if c.get("label")]
    new["selected_papers"] = list(selected_papers)
    new["papers_discussed"] = discussed[-10:]
    new["paper_titles"] = {k: titles[k] for k in list(titles)[-10:]}
    if citations:
        new["last_citations"] = citations[:6]
    new["user_language"] = detect_language(content_text(turn[0].content)) if turn else "en"
    new["turns"] = int(new.get("turns") or 0) + 1
    return new


def render_state(state: dict[str, Any]) -> str:
    if not state:
        return ""
    titles = state.get("paper_titles") or {}

    def name(pid: str) -> str:
        return f"{titles[pid]} ({pid})" if pid in titles else pid

    lines = []
    if state.get("selected_papers"):
        lines.append("Selected papers: " + "; ".join(name(p) for p in state["selected_papers"]))
    if state.get("papers_discussed"):
        lines.append("Papers discussed so far: " + "; ".join(name(p) for p in state["papers_discussed"]))
    if state.get("last_citations"):
        lines.append("Most recent citations: " + ", ".join(state["last_citations"]))
    if state.get("user_language") == "vi":
        lines.append("The user writes in Vietnamese; answer in Vietnamese.")
    return "\n".join(lines)


@dataclass
class ContextPlan:
    messages: list[BaseMessage]
    memory: str
    recent_turns: int
    dropped_turns: int
    first_recent_turn: int
    unsummarized_dropped: list[list[BaseMessage]] = field(default_factory=list)
    tokens: dict[str, int] = field(default_factory=dict)


def build_context(
    history: list[BaseMessage],
    *,
    summary: str | None,
    summarized_turns: int,
    state: dict[str, Any] | None,
    settings: Settings,
) -> ContextPlan:
    """History messages to send before the new human message."""
    turns = [compact_turn(t) for t in split_turns(history)]
    budget = settings.agent_context_recent_tokens
    keep: list[list[BaseMessage]] = []
    used = 0
    for turn in reversed(turns[-settings.agent_context_max_turns :] if settings.agent_context_max_turns else turns):
        size = sum(message_tokens(m) for m in turn)
        if keep and used + size > budget:
            break
        keep.insert(0, turn)
        used += size
    first_recent = len(turns) - len(keep)
    dropped = turns[:first_recent]
    unsummarized = dropped[summarized_turns:]

    memory_parts: list[str] = []
    state_text = render_state(state or {})
    if state_text:
        memory_parts.append("State:\n" + state_text)
    if summary and summarized_turns:
        memory_parts.append(f"Summary of turns 1-{summarized_turns}:\n{summary}")
    if unsummarized:
        memory_parts.append(
            "Earlier turns (brief):\n" + "\n".join(turn_brief(t) for t in unsummarized[-8:])
        )
    memory = MEMORY_HEADER + "\n\n" + "\n\n".join(memory_parts) if memory_parts else ""
    messages: list[BaseMessage] = [m for turn in keep for m in turn]
    return ContextPlan(
        messages=messages,
        memory=memory,
        recent_turns=len(keep),
        dropped_turns=len(dropped),
        first_recent_turn=first_recent,
        unsummarized_dropped=unsummarized,
        tokens={"memory": count_tokens(memory) if memory else 0, "recent": used},
    )


SUMMARY_PROMPT = (
    "You maintain a running summary of a conversation between a user and a research-paper "
    "assistant. Merge the earlier summary with the new turns into at most {words} words. "
    "Keep: which papers were discussed, what the user wanted, conclusions reached, numbers "
    "and citation labels mentioned, open questions, and user preferences. Drop greetings. "
    "Write plain sentences in English."
)


def summarize_turns(
    previous: str | None,
    turns: list[list[BaseMessage]],
    *,
    settings: Settings,
) -> tuple[str, str]:
    """Fold `turns` into the rolling summary. Returns (summary, method)."""
    briefs = "\n".join(turn_brief(t, max_chars=400) for t in turns)
    if settings.agent_summary_use_llm and settings.llm_enabled and settings.allow_external_api:
        try:
            from langchain_openai import ChatOpenAI

            from app.infrastructure.observability import llm_callbacks

            kwargs: dict[str, Any] = {
                "model": settings.agent_summary_model or settings.llm_model,
                "api_key": settings.llm_api_key,
                "timeout": settings.llm_timeout_seconds,
                "max_retries": 0,
            }
            if settings.llm_base_url and not settings.agent_summary_model:
                kwargs["base_url"] = settings.llm_base_url
            if settings.agent_summary_model and settings.agent_summary_api_key:
                kwargs["api_key"] = settings.agent_summary_api_key
            model = ChatOpenAI(**kwargs, callbacks=llm_callbacks())
            reply = model.invoke(
                [
                    SystemMessage(content=SUMMARY_PROMPT.format(words=settings.agent_summary_max_words)),
                    HumanMessage(
                        content=f"Earlier summary:\n{previous or '(none)'}\n\nNew turns:\n{briefs}"
                    ),
                ],
                config={"run_name": "paperlens-history-summary"},
            )
            text = content_text(reply.content).strip()
            if text:
                return text, "llm"
        except Exception as exc:
            logger.warning("History summary fell back to briefs: %s", type(exc).__name__)
    merged = "\n".join(p for p in [previous or "", briefs] if p)
    lines = merged.splitlines()[-12:]  # bounded without a model
    return "\n".join(lines), "extractive"
