"""Repository for unified-agent conversation persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, selectinload

from app.db.models import AgentConversation, AgentMessage


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _message_row(conversation_id: str, order_index: int, item: dict[str, Any]) -> AgentMessage:
    return AgentMessage(
        conversation_id=conversation_id,
        order_index=order_index,
        role=str(item.get("role") or "human"),
        content=str(item.get("content") or ""),
        tool_name=item.get("tool_name"),
        tool_call_id=item.get("tool_call_id"),
        tool_calls=item.get("tool_calls"),
        artifact=item.get("artifact"),
        status=item.get("status"),
        message_metadata=item.get("message_metadata"),
    )


class AgentConversationRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def get_or_create(
        self,
        conversation_id: str,
        *,
        selected_papers: list[str] | None = None,
        user_id: str = "local-user",
        title: str | None = None,
    ) -> AgentConversation:
        row = self.session.get(AgentConversation, conversation_id)
        if row is None:
            row = AgentConversation(
                id=conversation_id,
                user_id=user_id,
                title=title,
                selected_papers=list(selected_papers or []),
            )
            self.session.add(row)
            self.session.flush()
            return row
        if row.user_id != user_id:
            raise ValueError("Conversation belongs to another account")
        if title and not row.title:
            row.title = title
        if selected_papers is not None:
            row.selected_papers = list(selected_papers)
            row.updated_at = _utcnow()
            self.session.flush()
        return row

    def get_with_messages(
        self, conversation_id: str, *, user_id: str | None = None
    ) -> AgentConversation | None:
        stmt = (
            select(AgentConversation)
            .where(AgentConversation.id == conversation_id)
            .options(selectinload(AgentConversation.messages))
        )
        if user_id is not None:
            stmt = stmt.where(AgentConversation.user_id == user_id)
        return self.session.scalar(stmt)

    def list_for_user(self, user_id: str, *, limit: int = 100) -> list[AgentConversation]:
        stmt = (
            select(AgentConversation)
            .where(AgentConversation.user_id == user_id)
            .order_by(AgentConversation.updated_at.desc())
            .limit(limit)
        )
        return list(self.session.scalars(stmt))

    def delete_for_user(self, conversation_id: str, user_id: str) -> bool:
        result = self.session.execute(
            delete(AgentConversation).where(
                AgentConversation.id == conversation_id,
                AgentConversation.user_id == user_id,
            )
        )
        self.session.flush()
        return bool(result.rowcount)

    def append_messages(
        self,
        conversation_id: str,
        records: list[dict[str, Any]],
        *,
        selected_papers: list[str] | None = None,
        user_id: str = "local-user",
        kind: str = "agent",
        max_messages: int = 400,
    ) -> AgentConversation:
        """Append one turn's messages (no rewrite of earlier rows).

        Earlier rows are only deleted past `max_messages`, oldest first; prompt-size
        trimming happens when the prompt is built, never here.
        """
        conversation = self.get_or_create(
            conversation_id, selected_papers=selected_papers, user_id=user_id
        )
        last_index = self.session.scalar(
            select(func.max(AgentMessage.order_index)).where(
                AgentMessage.conversation_id == conversation_id
            )
        )
        start = 0 if last_index is None else last_index + 1
        if start == 0:
            conversation.kind = kind
        for offset, item in enumerate(records):
            self.session.add(_message_row(conversation_id, start + offset, item))
        if not conversation.title:
            first_human = next(
                (str(item.get("content") or "").strip() for item in records if item.get("role") == "human"),
                "",
            )
            conversation.title = first_human[:80] or "New conversation"
        conversation.updated_at = _utcnow()
        self.session.flush()
        total = start + len(records)
        if max_messages and total > max_messages:
            self.session.execute(
                delete(AgentMessage).where(
                    AgentMessage.conversation_id == conversation_id,
                    AgentMessage.order_index < total - max_messages,
                )
            )
            self.session.flush()
        return conversation

    def update_memory(
        self,
        conversation_id: str,
        *,
        state: dict[str, Any] | None = None,
        summary: str | None = None,
        summarized_turns: int | None = None,
    ) -> None:
        conversation = self.session.get(AgentConversation, conversation_id)
        if conversation is None:
            return
        if state is not None:
            conversation.context_state = state
        if summary is not None:
            conversation.context_summary = summary
        if summarized_turns is not None:
            conversation.summarized_turns = summarized_turns
        self.session.flush()

    def replace_messages(
        self,
        conversation_id: str,
        records: list[dict[str, Any]],
        *,
        selected_papers: list[str] | None = None,
        keep_last: int = 24,
        user_id: str = "local-user",
    ) -> AgentConversation:
        """Replace conversation message rows with a bounded serializable history."""
        conversation = self.get_or_create(
            conversation_id,
            selected_papers=selected_papers,
            user_id=user_id,
        )
        conversation.messages.clear()
        self.session.flush()

        trimmed = records[-keep_last:] if keep_last > 0 else records
        for idx, item in enumerate(trimmed):
            conversation.messages.append(_message_row(conversation_id, idx, item))
        if not conversation.title:
            first_human = next(
                (str(item.get("content") or "").strip() for item in records if item.get("role") == "human"),
                "",
            )
            conversation.title = first_human[:80] or "New conversation"
        conversation.updated_at = _utcnow()
        self.session.flush()
        return conversation

    def chat_turns(
        self, conversation_id: str, *, user_id: str | None = None
    ) -> list[dict[str, Any]]:
        """User/assistant turns suitable for restoring the chat UI."""
        conversation = self.get_with_messages(conversation_id, user_id=user_id)
        if conversation is None:
            return []
        turns: list[dict[str, Any]] = []
        for msg in conversation.messages:
            if msg.role == "human":
                meta = msg.message_metadata or {}
                turns.append(
                    {
                        "role": "user",
                        "content": msg.content,
                        "has_image": bool(meta.get("has_image")),
                        "image_mime": meta.get("image_mime"),
                    }
                )
            elif msg.role == "ai" and not msg.tool_calls:
                turns.append(
                    {
                        "role": "assistant",
                        "content": msg.content,
                        "artifact": msg.artifact,
                    }
                )
        return turns
