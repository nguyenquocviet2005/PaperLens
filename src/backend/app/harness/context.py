"""Conversation and paper-context helpers for the research agent harness."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)

from app.config import Settings, get_settings
from app.db.agent_repository import AgentConversationRepository
from app.db.session import session_scope
from app.harness.guardrails import ValidatedImage

logger = logging.getLogger(__name__)


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text") or "")
            for block in content
            if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
        )
    return str(content or "")


def build_human_message(
    message: str,
    *,
    image: ValidatedImage | None = None,
) -> HumanMessage:
    if image is None:
        return HumanMessage(content=message)
    blocks: list[dict[str, Any]] = []
    if message:
        blocks.append({"type": "text", "text": message})
    else:
        blocks.append({"type": "text", "text": "Please analyze the attached image."})
    blocks.append(
        {
            "type": "image_url",
            "image_url": {"url": image.data_url},
        }
    )
    return HumanMessage(content=blocks)


def serialize_message(msg: BaseMessage) -> dict[str, Any]:
    if isinstance(msg, HumanMessage):
        meta: dict[str, Any] = {}
        content = msg.content
        if isinstance(content, list):
            text = content_text(content)
            has_image = any(
                isinstance(block, dict)
                and block.get("type") in {"image_url", "image"}
                for block in content
            )
            if has_image:
                meta["has_image"] = True
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    if block.get("type") == "image_url":
                        url = (block.get("image_url") or {}).get("url") or ""
                        if isinstance(url, str) and url.startswith("data:"):
                            mime = url[5:].split(";", 1)[0].lower()
                            meta["image_mime"] = mime
                        # Intentionally omit base64 payload from persistence.
                        break
            return {
                "role": "human",
                "content": text,
                "message_metadata": meta or None,
            }
        return {"role": "human", "content": content_text(content)}

    if isinstance(msg, AIMessage):
        tool_calls = []
        for call in msg.tool_calls or []:
            tool_calls.append(
                {
                    "name": call.get("name"),
                    "args": call.get("args") or {},
                    "id": call.get("id"),
                    "type": call.get("type") or "tool_call",
                }
            )
        return {
            "role": "ai",
            "content": content_text(msg.content),
            "tool_calls": tool_calls or None,
        }

    if isinstance(msg, ToolMessage):
        artifact = msg.artifact if isinstance(msg.artifact, dict) else None
        return {
            "role": "tool",
            "content": content_text(msg.content),
            "tool_name": msg.name,
            "tool_call_id": msg.tool_call_id,
            "status": msg.status,
            "artifact": artifact,
        }

    return {"role": "human", "content": content_text(getattr(msg, "content", ""))}


def deserialize_message(record: dict[str, Any]) -> BaseMessage:
    role = record.get("role")
    content = record.get("content") or ""
    if role == "human":
        # Restored turns never rehydrate image bytes — text-only for replay.
        return HumanMessage(content=content)
    if role == "ai":
        kwargs: dict[str, Any] = {"content": content}
        tool_calls = record.get("tool_calls") or []
        if tool_calls:
            kwargs["tool_calls"] = tool_calls
        return AIMessage(**kwargs)
    if role == "tool":
        return ToolMessage(
            content=content,
            name=record.get("tool_name") or "",
            tool_call_id=record.get("tool_call_id") or "",
            status=record.get("status") or "success",
            artifact=record.get("artifact"),
        )
    return HumanMessage(content=content)


@dataclass
class StoredConversation:
    messages: list[BaseMessage] = field(default_factory=list)
    summary: str | None = None
    summarized_turns: int = 0
    state: dict[str, Any] | None = None


def load_conversation(
    conversation_id: str, settings: Settings, *, user_id: str = "local-user"
) -> StoredConversation:
    """Full stored history plus the conversation memory (summary and state)."""
    try:
        with session_scope(settings) as session:
            repo = AgentConversationRepository(session)
            conversation = repo.get_with_messages(conversation_id, user_id=user_id)
            if conversation is None:
                return StoredConversation()
            records = [
                {
                    "role": m.role,
                    "content": m.content,
                    "tool_name": m.tool_name,
                    "tool_call_id": m.tool_call_id,
                    "tool_calls": m.tool_calls,
                    "artifact": m.artifact,
                    "status": m.status,
                    "message_metadata": m.message_metadata,
                }
                for m in conversation.messages
            ]
            stored = StoredConversation(
                summary=conversation.context_summary,
                summarized_turns=int(conversation.summarized_turns or 0),
                state=dict(conversation.context_state or {}) or None,
            )
    except Exception:
        logger.exception("Failed to load agent conversation %s", conversation_id)
        return StoredConversation()
    stored.messages = [deserialize_message(item) for item in records]
    return stored


def load_history(
    conversation_id: str, settings: Settings, *, user_id: str = "local-user"
) -> list[BaseMessage]:
    return load_conversation(conversation_id, settings, user_id=user_id).messages


def save_history(
    conversation_id: str,
    messages: list[BaseMessage],
    *,
    selected_papers: list[str],
    settings: Settings,
    human_image: ValidatedImage | None = None,
    user_id: str = "local-user",
    kind: str = "agent",
) -> None:
    """Append the messages of one turn (the new human message and what followed)."""
    records = [serialize_message(m) for m in messages]
    # Attach image metadata to the latest human turn without storing bytes.
    if human_image is not None:
        for item in reversed(records):
            if item.get("role") == "human":
                meta = dict(item.get("message_metadata") or {})
                meta["has_image"] = True
                meta["image_mime"] = human_image.mime
                meta["image_decoded_bytes"] = human_image.decoded_size
                item["message_metadata"] = meta
                break
    try:
        with session_scope(settings) as session:
            AgentConversationRepository(session).append_messages(
                conversation_id,
                records,
                selected_papers=selected_papers,
                user_id=user_id,
                kind=kind,
                max_messages=settings.agent_history_limit,
            )
    except Exception:
        logger.exception("Failed to persist agent conversation %s", conversation_id)


def save_memory(
    conversation_id: str,
    settings: Settings,
    *,
    state: dict[str, Any] | None = None,
    summary: str | None = None,
    summarized_turns: int | None = None,
) -> None:
    try:
        with session_scope(settings) as session:
            AgentConversationRepository(session).update_memory(
                conversation_id, state=state, summary=summary, summarized_turns=summarized_turns
            )
    except Exception:
        logger.exception("Failed to persist conversation memory %s", conversation_id)


def get_agent_conversation(
    conversation_id: str,
    *,
    settings: Settings | None = None,
    user_id: str = "local-user",
) -> dict[str, Any] | None:
    cfg = settings or get_settings()
    with session_scope(cfg) as session:
        repo = AgentConversationRepository(session)
        conversation = repo.get_with_messages(conversation_id, user_id=user_id)
        if conversation is None:
            return None
        turns = repo.chat_turns(conversation_id, user_id=user_id)
        return {
            "conversation_id": conversation.id,
            "title": conversation.title,
            "created_at": conversation.created_at,
            "updated_at": conversation.updated_at,
            "selected_papers": list(conversation.selected_papers or []),
            "turns": turns,
        }
