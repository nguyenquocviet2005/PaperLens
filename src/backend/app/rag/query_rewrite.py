"""Standalone retrieval queries from follow-up and non-English questions.

Retrieval sees one string. "And on the other dataset?" or a Vietnamese question against
an English paper retrieves poorly (bge-small-en: 95% hit rate on English questions, 74%
on the same questions in Vietnamese). This step rewrites the question, using recent
turns, into one standalone English query. The answer is still written for the original
question, in the user's language.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from app.config import Settings

logger = logging.getLogger(__name__)

_VI_CHARS = re.compile(
    r"[ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ]", re.I
)
_NON_ASCII_WORD = re.compile(r"[^\x00-\x7f]")

REWRITE_PROMPT = (
    "Rewrite the user's latest question as ONE standalone search query, in English, for "
    "retrieving passages from an English research paper. Resolve pronouns and references "
    "using the conversation. Keep numbers, names, symbols and technical terms exactly. "
    "Do not answer the question. Reply with the query only."
)


def detect_language(text: str) -> str:
    """'vi' for Vietnamese, 'other' for other non-ASCII text, else 'en'."""
    if _VI_CHARS.search(text or ""):
        return "vi"
    letters = [c for c in text or "" if c.isalpha()]
    if letters and sum(1 for c in letters if _NON_ASCII_WORD.match(c)) / len(letters) > 0.3:
        return "other"
    return "en"


def rewrite_query(
    question: str,
    history: list[tuple[str, str]] | None,
    *,
    settings: Settings,
) -> tuple[str, str]:
    """Return (retrieval_query, method). method: none | llm | unavailable | failed."""
    history = [pair for pair in history or [] if pair[0] or pair[1]]
    if not settings.qa_query_rewrite or (not history and detect_language(question) == "en"):
        return question, "none"
    if not (settings.llm_enabled and settings.allow_external_api and settings.llm_api_key):
        return question, "unavailable"
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        from app.infrastructure.observability import llm_callbacks

        kwargs: dict[str, Any] = {
            "model": settings.agent_summary_model or settings.llm_model,
            "api_key": settings.agent_summary_api_key or settings.llm_api_key,
            "timeout": settings.llm_timeout_seconds,
            "max_retries": 0,
        }
        if settings.llm_base_url and not settings.agent_summary_model:
            kwargs["base_url"] = settings.llm_base_url
        conversation = "\n".join(
            f"User: {user[:400]}\nAssistant: {assistant[:400]}" for user, assistant in history[-3:]
        )
        reply = ChatOpenAI(**kwargs, callbacks=llm_callbacks()).invoke(
            [
                SystemMessage(content=REWRITE_PROMPT),
                HumanMessage(
                    content=f"Conversation:\n{conversation or '(none)'}\n\nLatest question:\n{question}"
                ),
            ],
            config={"run_name": "paperlens-query-rewrite"},
        )
        content = reply.content
        if isinstance(content, list):  # Responses-style content blocks
            content = "".join(
                str(b.get("text") or "") for b in content if isinstance(b, dict)
            )
        query = content.strip().strip('"').splitlines()[0].strip() if content.strip() else ""
        if not query or len(query) > 500:
            return question, "failed"
        return query, "llm"
    except Exception as exc:
        logger.warning("Query rewrite failed; using the question as is: %s", type(exc).__name__)
        return question, "failed"
