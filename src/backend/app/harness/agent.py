"""Unified LangChain/LangGraph research assistant with explicit tools."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings
from app.harness.context import (
    StoredConversation,
    build_human_message,
    content_text,
    get_agent_conversation,
    load_conversation,
    save_history,
    save_memory,
)
from app.harness.memory import (
    ContextPlan,
    build_context,
    split_turns,
    summarize_turns,
    update_state,
)
from app.infrastructure.observability import emit_metrics, llm_callbacks, trace_config
from app.harness.guardrails import (
    ValidatedImage,
    apply_output_guardrails,
    StreamGuard,
)
from app.infrastructure.storage import get_storage, paper_normalized_key
from app.rag.qa import answer_paper_question
from app.research.compare import compare_papers
from app.research.discovery import discover_papers
from app.research.research_graph import run_research
from app.schemas import PaperDocument
from app.tools.code_tools import build_code_tools
from app.tools.math_tools import build_math_tools

logger = logging.getLogger(__name__)

# Re-export conversation helpers used by API and tests.
__all__ = [
    "build_tools",
    "get_agent_conversation",
    "run_agent",
    "stream_agent",
]


def _model(settings: Settings) -> ChatOpenAI:
    if not settings.llm_api_key or not settings.llm_model:
        raise ValueError("The research agent requires LLM_API_KEY and LLM_MODEL")
    kwargs: dict[str, Any] = {
        "model": settings.llm_model,
        "api_key": settings.llm_api_key,
        "timeout": settings.llm_timeout_seconds,
        "max_retries": settings.llm_max_retries,
    }
    if settings.llm_base_url:
        kwargs["base_url"] = settings.llm_base_url
    model_name = settings.llm_model.lower()
    supports_reasoning = any(
        marker in model_name for marker in ("luna", "gpt-5", "o1", "o3", "o4")
    )
    if settings.agent_reasoning_enabled and supports_reasoning:
        # Tool calling with reasoning is supported through the Responses API.
        # Request summaries only; private chain-of-thought is never exposed.
        kwargs["use_responses_api"] = True
        kwargs["reasoning"] = {
            "effort": settings.agent_reasoning_effort,
            "summary": "auto",
        }
    elif "luna" in model_name:
        # Luna Chat Completions rejects function tools with reasoning enabled.
        kwargs["reasoning_effort"] = "none"
    return ChatOpenAI(**kwargs, callbacks=llm_callbacks())


def _resolve_paper_id(requested: str, selected: list[str]) -> str:
    paper_id = requested.strip() if requested else ""
    if paper_id:
        if paper_id not in selected:
            raise ValueError("That paper is not in the selected account context.")
        return paper_id
    if selected:
        return selected[0]
    raise ValueError("No paper is selected. Ask the user to select or upload a paper.")


def build_tools(
    settings: Settings,
    selected_papers: list[str],
    *,
    user_id: str = "local-user",
    model: ChatOpenAI | None = None,
):
    @tool(response_format="content_and_artifact")
    def ask_paper(question: str, paper_id: str = "") -> tuple[str, dict[str, Any]]:
        """Answer a question about one selected library paper with page citations."""
        resolved = _resolve_paper_id(paper_id, selected_papers)
        answer, _state = answer_paper_question(
            paper_id=resolved,
            question=question,
            settings=settings,
            top_k=6,
        )
        artifact = {
            "kind": "paper_answer",
            "paper_id": resolved,
            "answer": answer.answer,
            "confidence": answer.confidence,
            "citations": [item.model_dump(mode="json") for item in answer.citations],
            "evidence": [item.model_dump(mode="json") for item in answer.evidence[:4]],
        }
        return json.dumps(artifact, ensure_ascii=False), artifact

    @tool(response_format="content_and_artifact")
    def read_paper(paper_id: str = "") -> tuple[str, dict[str, Any]]:
        """Inspect a selected paper's title, sections, pages, and visual assets."""
        resolved = _resolve_paper_id(paper_id, selected_papers)
        storage = get_storage(settings)
        document = PaperDocument.model_validate(
            storage.read_json(paper_normalized_key(resolved, "paper_document.json"))
        )
        artifact = {
            "kind": "paper_reader",
            "paper_id": resolved,
            "title": document.title or document.filename,
            "page_count": document.page_count,
            "sections": [
                section.model_dump(mode="json") for section in document.sections[:30]
            ],
            "counts": {
                "text": len(document.text_elements),
                "tables": len(document.tables),
                "figures": len(document.figures),
                "formulas": len(document.formulas),
            },
        }
        return json.dumps(artifact, ensure_ascii=False), artifact

    @tool(response_format="content_and_artifact")
    def discover_research(
        query: str,
        source: str = "auto",
        limit: int = 8,
    ) -> tuple[str, dict[str, Any]]:
        """Search arXiv or OpenAlex for external scholarly papers."""
        selected_source = source if source in {"auto", "arxiv", "openalex"} else "auto"
        result = discover_papers(
            query,
            source=selected_source,  # type: ignore[arg-type]
            limit=max(1, min(limit, 12)),
            settings=settings,
        )
        artifact = {
            "kind": "discovery",
            "query": query,
            "source": result.source,
            "results": [
                item.model_dump(mode="json", exclude={"raw"})
                for item in result.results
            ],
        }
        return json.dumps(artifact, ensure_ascii=False), artifact

    @tool(response_format="content_and_artifact")
    def compare_paper_set(question: str) -> tuple[str, dict[str, Any]]:
        """Compare two or more selected library papers using grounded evidence."""
        if len(selected_papers) < 2:
            raise ValueError("Select at least two papers before requesting a comparison.")
        result = compare_papers(
            paper_ids=selected_papers,
            question=question,
            settings=settings,
        )
        artifact = {
            "kind": "comparison",
            **result.model_dump(mode="json"),
        }
        return json.dumps(artifact, ensure_ascii=False), artifact

    @tool(response_format="content_and_artifact")
    def run_research_workflow(question: str) -> tuple[str, dict[str, Any]]:
        """Run the bounded LangGraph research workflow over selected papers."""
        report = run_research(
            question,
            selected_papers=selected_papers,
            settings=settings,
            enable_external=True,
            max_external_searches=1,
            user_id=user_id,
        )
        artifact = {
            "kind": "research_report",
            **report.model_dump(mode="json"),
        }
        return json.dumps(artifact, ensure_ascii=False), artifact

    return [
        ask_paper,
        read_paper,
        discover_research,
        compare_paper_set,
        run_research_workflow,
        *build_math_tools(),
        *build_code_tools(model or _model(settings)),
    ]


def _create_graph(
    settings: Settings, selected: list[str], *, user_id: str = "local-user", memory: str = ""
):
    selected_context = ", ".join(selected) if selected else "none"
    # Fixed instructions first, per-conversation memory last: a stable prompt prefix
    # is what provider prompt caching reuses.
    memory_block = f"\n\n{memory}" if memory else ""
    model = _model(settings)
    return create_agent(
        model,
        build_tools(settings, selected, user_id=user_id, model=model),
        system_prompt=(
            "You are PaperLens, a friendly research assistant. Decide whether to answer "
            "conversationally or call tools. Greetings, thanks, and UI questions require no "
            "tool and must never be labeled insufficient. For claims about a library paper, "
            "call ask_paper or read_paper. For new literature, call discover_research. For "
            "multi-paper synthesis, call compare_paper_set or run_research_workflow. For "
            "symbolic math, call analyze_math. For plotting y=f(x), call plot_function "
            "(returns points for frontend SVG — do not invent plot files). For code explanation, "
            "debugging, review, or generation, call assist_code; it never executes code. Never invent "
            "citations. Explain when a user must upload or select a paper. Format "
            "substantive answers as clean GitHub-flavored Markdown with short paragraphs, "
            "headings only when useful, and blank lines before lists. Use $...$ for inline "
            "math and $$...$$ for display formulas. Do not repeat a separate citation dump "
            "after already citing claims in the answer. Reason carefully before acting, but "
            "never reveal private chain-of-thought; provide concise conclusions and an "
            "inspectable tool/evidence trace instead. "
            f"Currently selected paper IDs: {selected_context}."
            f"{memory_block}"
        ),
        name="paperlens_research_agent",
    )


def _collect_result(
    messages: list[BaseMessage],
    *,
    history_length: int,
    conversation_id: str,
) -> dict[str, Any]:
    final = next(
        (
            item
            for item in reversed(messages)
            if isinstance(item, AIMessage) and item.content and not item.tool_calls
        ),
        None,
    )
    artifacts: list[dict[str, Any]] = []
    tool_calls: list[dict[str, Any]] = []
    for item in messages[history_length:]:
        if isinstance(item, AIMessage):
            for call in item.tool_calls:
                tool_calls.append(
                    {
                        "name": call.get("name"),
                        "arguments": call.get("args") or {},
                        "status": "requested",
                    }
                )
        elif isinstance(item, ToolMessage):
            artifact = item.artifact if isinstance(item.artifact, dict) else None
            if artifact:
                artifacts.append(artifact)
            tool_calls.append(
                {
                    "name": item.name,
                    "status": item.status,
                }
            )

    citations = []
    for artifact in artifacts:
        citations.extend(artifact.get("citations") or [])
    grounded = any(
        artifact.get("kind") in {"paper_answer", "comparison", "research_report"}
        for artifact in artifacts
    )
    return {
        "conversation_id": conversation_id,
        "answer": (
            content_text(final.content)
            if final
            else "I could not produce a response."
        ),
        "grounded": grounded,
        "citations": citations,
        "tool_calls": tool_calls,
        "artifacts": artifacts,
    }


@dataclass
class _Turn:
    conversation_id: str
    selected: list[str]
    stored: StoredConversation
    plan: ContextPlan
    inputs: list[BaseMessage]


def _start_turn(
    message: str,
    *,
    cid: str,
    selected: list[str],
    cfg: Settings,
    image: ValidatedImage | None,
    user_id: str,
) -> _Turn:
    stored = load_conversation(cid, cfg, user_id=user_id)
    plan = build_context(
        stored.messages,
        summary=stored.summary,
        summarized_turns=stored.summarized_turns,
        state=stored.state,
        settings=cfg,
    )
    emit_metrics(
        {
            "ContextMemoryTokens": (plan.tokens.get("memory", 0), "Count"),
            "ContextHistoryTokens": (plan.tokens.get("recent", 0), "Count"),
            "HistoryTurnsKept": (plan.recent_turns, "Count"),
            "HistoryTurnsDropped": (plan.dropped_turns, "Count"),
        }
    )
    human = build_human_message(message, image=image)
    return _Turn(cid, selected, stored, plan, [*plan.messages, human])


def _graph_for(turn: _Turn, cfg: Settings, user_id: str):
    kwargs: dict[str, Any] = {}
    if user_id != "local-user":
        kwargs["user_id"] = user_id
    if turn.plan.memory:
        kwargs["memory"] = turn.plan.memory
    return _create_graph(cfg, turn.selected, **kwargs)


def _paper_titles(paper_ids: list[str], cfg: Settings, user_id: str) -> dict[str, str]:
    if not paper_ids:
        return {}
    try:
        from app.db.repository import PaperRepository
        from app.db.session import session_scope

        with session_scope(cfg) as session:
            repo = PaperRepository(session)
            titles = {}
            for paper_id in paper_ids:
                paper = repo.get_paper(paper_id)
                if paper is not None and (paper.title or paper.filename):
                    titles[paper_id] = str(paper.title or paper.filename)
            return titles
    except Exception:
        logger.debug("Paper titles unavailable for conversation state", exc_info=True)
        return {}


def _finish_turn(
    turn: _Turn,
    final_messages: list[BaseMessage],
    *,
    cfg: Settings,
    image: ValidatedImage | None,
    user_id: str,
) -> list[BaseMessage]:
    """Append this turn (the human message onwards) to the stored history."""
    new_messages = final_messages[len(turn.inputs) - 1 :]
    save_history(
        turn.conversation_id,
        new_messages,
        selected_papers=turn.selected,
        settings=cfg,
        human_image=image,
        user_id=user_id,
    )
    return new_messages


def _update_memory(
    turn: _Turn, new_messages: list[BaseMessage], *, cfg: Settings, user_id: str
) -> None:
    """Refresh the structured state and fold turns that will no longer fit into the summary.

    Runs after the answer is delivered, so a summary call never delays the user.
    """
    try:
        new_turns = split_turns(new_messages)
        state = update_state(
            turn.stored.state,
            new_turns[0] if new_turns else [],
            selected_papers=turn.selected,
            paper_titles=_paper_titles(turn.selected, cfg, user_id),
        )
        history = [*turn.stored.messages, *new_messages]
        upcoming = build_context(
            history,
            summary=turn.stored.summary,
            summarized_turns=turn.stored.summarized_turns,
            state=state,
            settings=cfg,
        )
        summary, summarized = None, None
        if len(upcoming.unsummarized_dropped) >= max(1, cfg.agent_summary_batch_turns):
            summary, method = summarize_turns(
                turn.stored.summary, upcoming.unsummarized_dropped, settings=cfg
            )
            summarized = upcoming.dropped_turns
            emit_metrics(
                {
                    "HistorySummaries": (1, "Count"),
                    "HistoryTurnsSummarized": (len(upcoming.unsummarized_dropped), "Count"),
                },
                dimensions={"Method": method},
            )
        save_memory(
            turn.conversation_id, cfg, state=state, summary=summary, summarized_turns=summarized
        )
    except Exception:
        logger.exception("Conversation memory update failed for %s", turn.conversation_id)


def run_agent(
    message: str,
    *,
    selected_papers: list[str] | None = None,
    conversation_id: str | None = None,
    settings: Settings | None = None,
    image: ValidatedImage | None = None,
    user_id: str = "local-user",
) -> dict[str, Any]:
    cfg = settings or get_settings()
    selected = list(dict.fromkeys(selected_papers or []))
    cid = conversation_id or str(uuid4())
    turn = _start_turn(message, cid=cid, selected=selected, cfg=cfg, image=image, user_id=user_id)
    output = _graph_for(turn, cfg, user_id).invoke(
        {"messages": turn.inputs},
        config=trace_config("paperlens-agent", conversation_id=cid, papers=len(selected)),
    )
    messages = list(output["messages"])
    new_messages = _finish_turn(turn, messages, cfg=cfg, image=image, user_id=user_id)
    result = _collect_result(messages, history_length=len(turn.inputs) - 1, conversation_id=cid)
    _update_memory(turn, new_messages, cfg=cfg, user_id=user_id)
    return apply_output_guardrails(result, settings=cfg)


def stream_agent(
    message: str,
    *,
    selected_papers: list[str] | None = None,
    conversation_id: str | None = None,
    settings: Settings | None = None,
    image: ValidatedImage | None = None,
    user_id: str = "local-user",
) -> Iterator[dict[str, Any]]:
    """Yield token, tool, and final events from the native LangGraph stream."""
    cfg = settings or get_settings()
    selected = list(dict.fromkeys(selected_papers or []))
    cid = conversation_id or str(uuid4())
    turn = _start_turn(message, cid=cid, selected=selected, cfg=cfg, image=image, user_id=user_id)
    final_messages: list[BaseMessage] = []
    observed_messages = len(turn.inputs)

    yield {"type": "start", "conversation_id": cid}
    guard = StreamGuard(cfg)
    for mode, data in _graph_for(turn, cfg, user_id).stream(
        {"messages": turn.inputs},
        stream_mode=["messages", "values"],
        config=trace_config("paperlens-agent-stream", conversation_id=cid, papers=len(selected)),
    ):
        if mode == "messages":
            chunk, metadata = data
            if isinstance(chunk, AIMessageChunk):
                # Nested LLMs inside tools (for example structured paper QA) also
                # emit message chunks. Only root agent-model tokens belong in chat.
                if str((metadata or {}).get("langgraph_node") or "") != "model":
                    continue
                safe = guard.feed(content_text(chunk.content))
                if safe:
                    yield {"type": "token", "content": safe}
            continue

        final_messages = list(data.get("messages") or [])
        for item in final_messages[observed_messages:]:
            if isinstance(item, AIMessage):
                for call in item.tool_calls:
                    yield {
                        "type": "tool",
                        "name": call.get("name"),
                        "status": "running",
                    }
            elif isinstance(item, ToolMessage):
                yield {
                    "type": "tool",
                    "name": item.name,
                    "status": item.status,
                }
        observed_messages = len(final_messages)

    tail = guard.flush()
    if tail:
        yield {"type": "token", "content": tail}
    new_messages = _finish_turn(turn, final_messages, cfg=cfg, image=image, user_id=user_id)
    result = _collect_result(
        final_messages, history_length=len(turn.inputs) - 1, conversation_id=cid
    )
    result = apply_output_guardrails(result, settings=cfg)
    yield {"type": "done", **result}
    # After "done": the client already has the answer when the summary call runs.
    _update_memory(turn, new_messages, cfg=cfg, user_id=user_id)
