"""SQLAlchemy models for the paper library."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


def _uuid() -> str:
    return str(uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Paper(Base):
    __tablename__ = "papers"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, default="local-user", index=True)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    title: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    authors: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    publication_year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    venue: Mapped[str | None] = mapped_column(String(512), nullable=True)
    doi: Mapped[str | None] = mapped_column(String(256), nullable=True, index=True)
    arxiv_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    source_url: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    storage_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    status: Mapped[str] = mapped_column(String(64), default="pending", index=True)
    parse_status: Mapped[str | None] = mapped_column(String(64), nullable=True)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    title_confidence: Mapped[str | None] = mapped_column(String(32), nullable=True)
    artifacts: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    warnings: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    sections: Mapped[list[PaperSection]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    elements: Mapped[list[PaperElement]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    visual_elements: Mapped[list[VisualElementRow]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    chunks: Mapped[list[PaperChunk]] = relationship(
        back_populates="paper", cascade="all, delete-orphan"
    )
    jobs: Mapped[list[Job]] = relationship(back_populates="paper", cascade="all, delete-orphan")


class PaperSection(Base):
    __tablename__ = "paper_sections"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    heading: Mapped[str] = mapped_column(String(1024), nullable=False)
    section_path: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    level: Mapped[int] = mapped_column(Integer, default=1)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, default=0)

    paper: Mapped[Paper] = relationship(back_populates="sections")


class PaperElement(Base):
    __tablename__ = "paper_elements"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    section_id: Mapped[str | None] = mapped_column(
        ForeignKey("paper_sections.id", ondelete="SET NULL"), nullable=True, index=True
    )
    element_key: Mapped[str] = mapped_column(String(64), nullable=False)
    type: Mapped[str] = mapped_column(String(64), nullable=False)
    label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text: Mapped[str | None] = mapped_column(Text, nullable=True)
    bbox: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    source_ref: Mapped[str | None] = mapped_column(String(256), nullable=True)
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    element_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    paper: Mapped[Paper] = relationship(back_populates="elements")

    __table_args__ = (UniqueConstraint("paper_id", "element_key", name="uq_paper_element_key"),)


class VisualElementRow(Base):
    __tablename__ = "visual_elements"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    element_key: Mapped[str] = mapped_column(String(64), nullable=False)
    type: Mapped[str] = mapped_column(String(32), nullable=False)  # figure|table|formula
    page: Mapped[int | None] = mapped_column(Integer, nullable=True)
    caption: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    structured_data_uri: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    enrichment_status: Mapped[str] = mapped_column(String(32), default="pending")
    enrichment_json: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    needs_enrichment: Mapped[bool] = mapped_column(default=False)
    docling_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    bbox: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    source_ref: Mapped[str | None] = mapped_column(String(256), nullable=True)
    visual_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)

    paper: Mapped[Paper] = relationship(back_populates="visual_elements")

    __table_args__ = (UniqueConstraint("paper_id", "element_key", name="uq_paper_visual_key"),)


class PaperChunk(Base):
    __tablename__ = "paper_chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    paper_id: Mapped[str] = mapped_column(ForeignKey("papers.id", ondelete="CASCADE"), index=True)
    parent_chunk_id: Mapped[str | None] = mapped_column(
        ForeignKey("paper_chunks.id", ondelete="SET NULL"), nullable=True, index=True
    )
    section_path: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_type: Mapped[str] = mapped_column(String(64), default="text")
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    chunk_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # Stored as JSON list of floats for SQLite portability; pgvector can replace later.
    embedding: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)

    paper: Mapped[Paper] = relationship(back_populates="chunks")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    paper_id: Mapped[str | None] = mapped_column(
        ForeignKey("papers.id", ondelete="CASCADE"), nullable=True, index=True
    )
    job_type: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued", index=True)
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    paper: Mapped[Paper | None] = relationship(back_populates="jobs")


class AgentConversation(Base):
    """Persisted unified-agent conversation (SQLite / Postgres)."""

    __tablename__ = "agent_conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, default="local-user", index=True)
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    selected_papers: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    # "agent" (main chat) or "qa" (single-paper /papers/{id}/qa chat).
    kind: Mapped[str] = mapped_column(String(32), nullable=False, default="agent")
    # Rolling summary of the oldest `summarized_turns` turns and structured state
    # (app/harness/memory.py). Context for the model, never a citation source.
    context_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    summarized_turns: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    context_state: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )

    messages: Mapped[list["AgentMessage"]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="AgentMessage.order_index",
    )


class GuestSession(Base):
    """Anonymous trial session with hard usage quotas."""

    __tablename__ = "guest_sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    # Stable ownership key used on papers / conversations: guest-<id>
    user_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    queries_used: Mapped[int] = mapped_column(Integer, default=0)
    papers_used: Mapped[int] = mapped_column(Integer, default=0)
    images_used: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class AgentMessage(Base):
    """Serializable agent turn for history restore and LangChain replay."""

    __tablename__ = "agent_messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("agent_conversations.id", ondelete="CASCADE"), index=True
    )
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # human|ai|tool
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    tool_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tool_call_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tool_calls: Mapped[list[Any] | None] = mapped_column(JSON, nullable=True)
    artifact: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    status: Mapped[str | None] = mapped_column(String(32), nullable=True)
    # Image metadata only — never store raw base64 image bytes.
    message_metadata: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    conversation: Mapped["AgentConversation"] = relationship(back_populates="messages")
