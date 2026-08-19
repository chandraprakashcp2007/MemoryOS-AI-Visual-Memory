"""Durable, user-scoped records used only by the cloud API.

The legacy tables are deliberately left untouched so local installations can
migrate explicitly.  These tables work on PostgreSQL and SQLite (for tests),
but production validation requires PostgreSQL.
"""
from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import Boolean, DateTime, ForeignKey, Index, Integer, JSON, LargeBinary, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base


def now() -> datetime:
    return datetime.now(timezone.utc)


class CloudUser(Base):
    __tablename__ = "cloud_users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)


class CloudMemory(Base):
    __tablename__ = "cloud_memories"
    memory_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("cloud_users.id", ondelete="CASCADE"), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    mime_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(Integer)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    storage_key: Mapped[str] = mapped_column(Text)
    thumbnail_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    processing_status: Mapped[str] = mapped_column(String(40), default="uploaded", index=True)
    ocr_status: Mapped[str] = mapped_column(String(30), default="pending")
    visual_status: Mapped[str] = mapped_column(String(30), default="pending")
    embedding_status: Mapped[str] = mapped_column(String(30), default="pending")
    ocr_text: Mapped[str] = mapped_column(Text, default="")
    ocr_confidence: Mapped[float | None] = mapped_column(nullable=True)
    visual_analysis: Mapped[dict] = mapped_column(JSON, default=dict)
    search_text: Mapped[str] = mapped_column(Text, default="")
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    processing_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
    __table_args__ = (
        UniqueConstraint("user_id", "content_hash", name="uq_cloud_memory_user_hash"),
        Index("ix_cloud_memories_user_created", "user_id", "created_at"),
    )


class CloudJob(Base):
    __tablename__ = "cloud_memory_jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    memory_id: Mapped[str] = mapped_column(ForeignKey("cloud_memories.memory_id", ondelete="CASCADE"), unique=True)
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    next_attempt_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True, index=True)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)


class CloudWorkerHeartbeat(Base):
    """Single durable heartbeat written by the independently deployed worker."""
    __tablename__ = "cloud_worker_heartbeat"
    id: Mapped[str] = mapped_column(String(40), primary_key=True, default="primary")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=now, onupdate=now)
