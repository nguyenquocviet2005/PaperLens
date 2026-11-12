"""Context engineering: turn-aligned windows, compaction, summaries, append-only storage."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.config import Settings, get_settings
from app.db.agent_repository import AgentConversationRepository
from app.db.session import init_db, reset_engine, session_scope
from app.harness import agent as agent_module
from app.harness.context import load_conversation, save_history
from app.harness.memory import (
    MEMORY_HEADER,
    build_context,
    compact_tool_message,
    split_turns,
    summarize_turns,
    update_state,
)
from app.rag.query_rewrite import detect_language, rewrite_query


@pytest.fixture()
def settings(tmp_path: Path):
    reset_engine()
    get_settings.cache_clear()
    cfg = Settings(
        _env_file=None,
        database_url=f"sqlite:///{(tmp_path / 'm.db').as_posix()}",
        embedding_provider="hashing",
        agent_context_recent_tokens=400,
        agent_context_max_turns=6,
    )
    init_db(cfg)
    yield cfg
    reset_engine()
    get_settings.cache_clear()


def _call(i: int, name: str = "ask_paper") -> dict:
    return {"name": name, "args": {"question": f"q{i}"}, "id": f"c{i}", "type": "tool_call"}


def _paper_turn(i: int, evidence_chars: int = 2000) -> list:
    artifact = {
        "kind": "paper_answer", "paper_id": "p1", "answer": f"answer {i}", "confidence": "high",
        "citations": [{"label": f"[Page {i} #ab{i:02d}]"}],
        "evidence": [{"text": "x" * evidence_chars}],
    }
    return [
        HumanMessage(content=f"question {i}"),
        AIMessage(content="", tool_calls=[_call(i)]),
        ToolMessage(content=json.dumps(artifact), name="ask_paper", tool_call_id=f"c{i}", artifact=artifact),
        AIMessage(content=f"final answer {i}"),
    ]


def _chat_turn(i: int) -> list:
    return [HumanMessage(content=f"hi {i}"), AIMessage(content=f"hello {i}")]


def _assert_valid_for_provider(messages):
    """Every tool message follows the AI message that called it (OpenAI rejects otherwise)."""
    called: set[str] = set()
    for message in messages:
        if isinstance(message, AIMessage):
            called.update(c["id"] for c in message.tool_calls)
        if isinstance(message, ToolMessage):
            assert message.tool_call_id in called, f"orphan tool message {message.tool_call_id}"
    assert not messages or isinstance(messages[0], HumanMessage)


# --------------------------------------------------------------------- turn handling


def test_legacy_history_starting_with_orphan_tool_message_is_repaired():
    # What the old count-based trim stored: the turn's tool call was cut away.
    legacy = [
        ToolMessage(content="{}", name="ask_paper", tool_call_id="c0"),
        AIMessage(content="a0"),
        *_chat_turn(1),
    ]
    turns = split_turns(legacy)
    assert [type(m).__name__ for m in turns[0]] == ["HumanMessage", "AIMessage"]
    _assert_valid_for_provider([m for t in turns for m in t])


def test_interrupted_tool_call_is_dropped():
    turn = [HumanMessage(content="q"), AIMessage(content="", tool_calls=[_call(1)])]
    assert split_turns(turn) == [[turn[0]]]


def test_window_keeps_whole_recent_turns_within_budget(settings):
    history = [m for i in range(10) for m in (_paper_turn(i) if i % 2 else _chat_turn(i))]
    plan = build_context(history, summary=None, summarized_turns=0, state=None, settings=settings)
    _assert_valid_for_provider(plan.messages)
    assert plan.recent_turns + plan.dropped_turns == 10
    assert 1 <= plan.recent_turns <= settings.agent_context_max_turns
    assert plan.tokens["recent"] <= settings.agent_context_recent_tokens
    assert plan.messages[-1].content == "hello 8" or plan.messages[-1].content == "final answer 9"
    # Dropped turns show up as one-line briefs until a summary covers them.
    assert MEMORY_HEADER in plan.memory and "question 1" in plan.memory


def test_latest_turn_is_kept_even_if_over_budget(settings):
    tiny = settings.model_copy(update={"agent_context_recent_tokens": 5})
    plan = build_context(_paper_turn(1), summary=None, summarized_turns=0, state=None, settings=tiny)
    assert plan.recent_turns == 1


def test_old_tool_results_are_compacted(settings):
    plan = build_context(_paper_turn(1, evidence_chars=6000), summary=None, summarized_turns=0,
                         state=None, settings=settings)
    tool = next(m for m in plan.messages if isinstance(m, ToolMessage))
    payload = json.loads(tool.content)
    assert payload["answer"] == "answer 1" and payload["citations"] == ["[Page 1 #ab01]"]
    assert "evidence" not in payload and len(tool.content) < 400


def test_other_tool_output_is_truncated():
    long = ToolMessage(content="y" * 5000, name="discover_research", tool_call_id="c1")
    assert len(compact_tool_message(long).content) < 700


# --------------------------------------------------------------------- memory


def test_state_tracks_papers_citations_and_language():
    turn = _paper_turn(3)
    turn[0] = HumanMessage(content="Bài báo này dùng bao nhiêu attention head?")
    state = update_state(None, turn, selected_papers=["p1"], paper_titles={"p1": "Attention"})
    assert state["papers_discussed"] == ["p1"]
    assert state["last_citations"] == ["[Page 3 #ab03]"]
    assert state["user_language"] == "vi" and state["turns"] == 1


def test_summary_is_used_and_briefs_cover_only_newer_dropped_turns(settings):
    history = [m for i in range(10) for m in _paper_turn(i)]
    plan = build_context(history, summary="Earlier: the user studied attention heads.",
                         summarized_turns=3, state={"selected_papers": ["p1"]}, settings=settings)
    assert "Summary of turns 1-3" in plan.memory
    assert "question 0" not in plan.memory  # summarized, not repeated as a brief
    assert "question 3" in plan.memory or plan.dropped_turns <= 3


def test_extractive_summary_without_llm_is_bounded(settings):
    turns = [_paper_turn(i) for i in range(30)]
    summary, method = summarize_turns("old", turns, settings=settings)
    assert method == "extractive" and len(summary.splitlines()) <= 12


# --------------------------------------------------------------------- storage


def test_history_is_appended_not_rewritten(settings):
    save_history("conv", _chat_turn(1), selected_papers=[], settings=settings)
    with session_scope(settings) as session:
        first_ids = [m.id for m in AgentConversationRepository(session).get_with_messages("conv").messages]
    save_history("conv", _paper_turn(2), selected_papers=["p1"], settings=settings)
    with session_scope(settings) as session:
        rows = AgentConversationRepository(session).get_with_messages("conv").messages
        assert [m.id for m in rows][:2] == first_ids  # earlier rows untouched
        assert [m.order_index for m in rows] == list(range(6))


def test_storage_cap_prunes_oldest(settings):
    capped = settings.model_copy(update={"agent_history_limit": 5})
    for i in range(4):
        save_history("cap", _chat_turn(i), selected_papers=[], settings=capped)
    stored = load_conversation("cap", capped)
    assert len(stored.messages) == 5
    _assert_valid_for_provider([m for t in split_turns(stored.messages) for m in t])


# --------------------------------------------------------------------- agent wiring


class _RecordingGraph:
    def __init__(self, calls):
        self.calls = calls

    def invoke(self, inputs, config=None):
        self.calls.append(list(inputs["messages"]))
        question = inputs["messages"][-1].content
        return {"messages": [*inputs["messages"], AIMessage(content=f"re: {question}")]}


def test_run_agent_builds_layered_context_across_many_turns(settings, monkeypatch):
    calls, graphs = [], []

    def fake_create_graph(cfg, selected, **kwargs):
        graphs.append(kwargs)
        return _RecordingGraph(calls)

    monkeypatch.setattr(agent_module, "_create_graph", fake_create_graph)
    long_settings = settings.model_copy(update={"agent_context_max_turns": 3})
    for i in range(8):
        agent_module.run_agent(f"message number {i}", conversation_id="long", settings=long_settings)

    _assert_valid_for_provider(calls[-1])
    assert len(calls[-1]) == 3 * 2 + 1  # three whole turns + the new message
    assert calls[1][0].content == "message number 0"  # turn 2 sees turn 1
    # Older turns reach the model through the memory block in the system prompt.
    assert "memory" in graphs[-1] and "message number 0" in graphs[-1]["memory"]
    stored = load_conversation("long", long_settings)
    assert len(stored.messages) == 16  # full history kept in the database
    # 8 turns, 3 kept: 5 dropped; summaries run in batches of 3, the rest stay briefs.
    assert stored.summarized_turns == 3 and "message number 0" in (stored.summary or "")
    # Prompt for turn 8: turns 0-2 summarized, turn 3 dropped but not yet summarized.
    assert "User: message number 3" in graphs[-1]["memory"]
    assert stored.state["turns"] == 8


# --------------------------------------------------------------------- query rewriting


def test_language_detection():
    assert detect_language("How many heads?") == "en"
    assert detect_language("Bao nhiêu attention head?") == "vi"


def test_rewrite_is_skipped_for_standalone_english(settings):
    assert rewrite_query("How many heads?", [], settings=settings) == ("How many heads?", "none")


def test_rewrite_needs_an_llm(settings):
    query, method = rewrite_query("Còn BERT thì sao?", [("q", "a")], settings=settings)
    assert (query, method) == ("Còn BERT thì sao?", "unavailable")


def test_rewrite_uses_history(settings, monkeypatch):
    seen = {}

    class FakeChat:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages, config=None):
            seen["prompt"] = messages[-1].content
            return AIMessage(content='"How many attention heads does BERT-large use?"')

    monkeypatch.setattr("langchain_openai.ChatOpenAI", FakeChat)
    llm = settings.model_copy(update={"llm_enabled": True, "allow_external_api": True,
                                      "llm_api_key": "k", "llm_model": "m"})
    query, method = rewrite_query(
        "Còn BERT-large thì sao?", [("How many heads does the Transformer use?", "8 heads.")],
        settings=llm,
    )
    assert method == "llm" and query == "How many attention heads does BERT-large use?"
    assert "How many heads does the Transformer use?" in seen["prompt"]
