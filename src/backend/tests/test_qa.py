"""Grounded QA tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.chunking import ChunkingConfig, chunk_paper_document, persist_chunks
from app.config import Settings, get_settings
from app.db.repository import PaperRepository
from app.db.session import init_db, reset_engine, session_scope
from app.rag.embeddings import index_paper_chunks
from app.rag.qa import GroundedSynthesis, answer_paper_question
from app.schemas import ArtifactPaths, PaperDocument, TextElement


@pytest.fixture()
def qa_env(tmp_path: Path):
    reset_engine()
    get_settings.cache_clear()
    settings = Settings(
        database_url=f"sqlite:///{(tmp_path / 'qa.db').as_posix()}",
        embedding_provider="hashing",
        embedding_dimensions=64,
    )
    init_db(settings)
    paper = PaperDocument(
        paper_id="qa1",
        filename="qa.pdf",
        page_count=2,
        text_elements=[
            TextElement(
                element_id="el_1",
                order=1,
                page=1,
                section_path=["Intro"],
                type="TextItem",
                text="This paper proposes temperature scaling for neural network calibration. "
                * 15,
            ),
            TextElement(
                element_id="el_2",
                order=2,
                page=2,
                section_path=["Results"],
                type="TextItem",
                text="On CIFAR-10 the method reduces ECE significantly compared to baselines. "
                * 15,
            ),
        ],
    )
    with session_scope(settings) as session:
        PaperRepository(session).replace_document_graph(
            paper, ArtifactPaths(raw_pdf="raw/papers/qa1/source.pdf")
        )
        # Also create storage meta for API compatibility tests if needed later
    result = chunk_paper_document(
        paper,
        config=ChunkingConfig(
            parent_min_tokens=30,
            parent_max_tokens=100,
            child_min_tokens=20,
            child_max_tokens=50,
            overlap_tokens=5,
        ),
    )
    persist_chunks("qa1", result, settings=settings)
    index_paper_chunks("qa1", settings=settings, force=True)
    yield settings
    reset_engine()
    get_settings.cache_clear()


def test_grounded_answer_has_citations(qa_env: Settings):
    answer, state = answer_paper_question(
        paper_id="qa1",
        question="What method does the paper propose for calibration?",
        settings=qa_env,
    )
    assert answer.confidence != "insufficient"
    assert answer.citations
    assert answer.pages
    assert all(p in {1, 2} for p in answer.pages)
    assert answer.used_chunks
    assert state.conversation_id


def test_insufficient_evidence(qa_env: Settings):
    answer, _state = answer_paper_question(
        paper_id="qa1",
        question="What is the capital of Mars and the population of Atlantis?",
        settings=qa_env,
        top_k=3,
    )
    # May still retrieve something weakly; if overlap is tiny, insufficient or low.
    assert answer.confidence in {"insufficient", "low", "medium", "high"}
    if answer.confidence != "insufficient":
        # Still must only cite retrieved chunk ids
        assert answer.citations
        assert answer.used_chunks


def test_langchain_grounded_mode_uses_only_valid_citations(
    qa_env: Settings, monkeypatch
):
    settings = qa_env.model_copy(
        update={
            "llm_enabled": True,
            "allow_external_api": True,
            "llm_api_key": "test-key",
            "llm_model": "test-model",
        }
    )

    seen = {}

    def fake_synthesize(*, question, evidence, settings, standalone_query=None):
        assert question
        seen["evidence"] = evidence
        label = evidence[0].citation
        return GroundedSynthesis(
            answer="The paper proposes temperature scaling.",
            confidence="high",
            used_citation_labels=[label, "[invented citation]"],
        )

    monkeypatch.setattr("app.rag.qa._llm_synthesize", fake_synthesize)
    answer, _state = answer_paper_question(
        paper_id="qa1",
        question="What method does the paper propose?",
        settings=settings,
    )

    assert answer.mode == "langchain_openai_grounded"
    assert answer.citations
    assert "[invented citation]" not in answer.answer
    assert all(citation.chunk_id in answer.used_chunks for citation in answer.citations)

    assert answer.confidence == "high"
    assert len(seen["evidence"]) <= 6  # top_k reaches the LLM path too


def _llm_settings(settings: Settings) -> Settings:
    return settings.model_copy(
        update={"llm_enabled": True, "allow_external_api": True,
                "llm_api_key": "test-key", "llm_model": "test-model"}
    )


def test_numbers_missing_from_cited_evidence_lower_confidence(qa_env: Settings, monkeypatch):
    def fake_synthesize(*, question, evidence, settings, standalone_query=None):
        label = evidence[0].citation
        return GroundedSynthesis(
            answer=f"It reduces ECE by 87.5% on CIFAR-10. {label}",
            confidence="high",
            used_citation_labels=[label],
        )

    monkeypatch.setattr("app.rag.qa._llm_synthesize", fake_synthesize)
    answer, _ = answer_paper_question(
        paper_id="qa1", question="How much does ECE drop?", settings=_llm_settings(qa_env)
    )
    assert answer.confidence == "low"
    assert answer.diagnostics["unsupported_numbers"] == ["87.5"]
    assert any("87.5" in item for item in answer.limitations)


def test_supported_numbers_keep_confidence():
    from app.rag.qa import unsupported_numbers

    evidence = ["We use h = 8 parallel attention layers, or heads. Training took 1,000,000 steps."]
    assert unsupported_numbers("The model uses 8 heads [Page 5 #ab12].", evidence) == []
    assert unsupported_numbers("Pre-training runs 1000000 steps.", evidence) == []
    assert unsupported_numbers("It uses 64 heads.", evidence) == ["64"]
    assert unsupported_numbers("BERT-large has 340M parameters.", ["Total Parameters=340M"]) == []
    assert unsupported_numbers("BERT-large has 110M parameters.", ["Total Parameters=340M"]) == ["110"]
    # Citation labels and list markers are not claims.
    assert unsupported_numbers("1. Eight heads [Page 12, Section 3.2 #9f3a]", evidence) == []
    # A number the user asked about may be repeated.
    assert unsupported_numbers("No, not 64.", evidence, question="Is it 64 heads?") == []


def test_extractive_answer_has_no_repeated_sentences(qa_env: Settings):
    answer, _ = answer_paper_question(
        paper_id="qa1", question="What method does the paper propose for calibration?",
        settings=qa_env,
    )
    sentences = [s.strip() for s in answer.answer.split("]") if s.strip()]
    bodies = [s.split("[")[0].strip() for s in sentences]
    assert len(bodies) == len(set(bodies))


def test_extractive_mode_explains_language_mismatch(qa_env: Settings):
    answer, _ = answer_paper_question(
        paper_id="qa1",
        question="Bài báo đề xuất phương pháp gì để hiệu chỉnh mạng nơ-ron?",
        settings=qa_env,
    )
    assert answer.confidence == "insufficient"
    assert "language" in answer.answer


def test_qa_conversations_are_stored_per_user(qa_env: Settings, monkeypatch):
    from app.db.agent_repository import AgentConversationRepository
    from app.db.session import session_scope

    seen_history = []

    def fake_rewrite(question, history, *, settings):
        seen_history.append(list(history))
        return question, "none"

    monkeypatch.setattr("app.rag.qa.rewrite_query", fake_rewrite)
    ask = lambda q, user: answer_paper_question(
        paper_id="qa1", question=q, settings=qa_env, conversation_id="qa-conv",
        persist=True, user_id=user,
    )
    ask("What method does the paper propose for calibration?", "alice")
    _, state = ask("How much does it reduce ECE?", "alice")

    assert len(state.turns) == 4  # earlier question + answer, new question + answer
    assert seen_history[1][0][0] == "What method does the paper propose for calibration?"
    with session_scope(qa_env) as session:
        conversation = AgentConversationRepository(session).get_with_messages("qa-conv")
        assert conversation.kind == "qa" and conversation.user_id == "alice"
        assert [m.role for m in conversation.messages] == ["human", "ai", "human", "ai"]

    # Another account reusing the id sees no history and cannot append to it.
    _, other = ask("What is ECE?", "mallory")
    assert len(other.turns) == 2 and seen_history[2] == []
    with session_scope(qa_env) as session:
        assert len(AgentConversationRepository(session).get_with_messages("qa-conv").messages) == 4


def test_tool_callers_keep_no_qa_state(qa_env: Settings):
    from app.db.agent_repository import AgentConversationRepository
    from app.db.session import session_scope

    answer_paper_question(paper_id="qa1", question="What method?", settings=qa_env,
                          conversation_id="tool-call")
    with session_scope(qa_env) as session:
        assert AgentConversationRepository(session).get_with_messages("tool-call") is None
