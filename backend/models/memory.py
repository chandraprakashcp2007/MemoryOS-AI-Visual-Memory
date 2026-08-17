"""
MemoryOS - Memory SQLAlchemy Model
==================================

Core persistent model for MemoryOS.

A Memory is the logical container for one captured item.

Lifecycle:

    Upload
      ↓
    Memory
      ↓
    ImageAsset
      ↓
    ProcessingRun
      ↓
    OCR
      ↓
    AI Analysis
      ↓
    Entity Extraction
      ↓
    Embedding
      ↓
    FAISS
      ↓
    Semantic Search
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base


class Memory(Base):
    """
    Primary persistent MemoryOS memory record.

    One Memory represents one logical captured item.

    Physical uploaded files:
        ImageAsset

    Extracted entities:
        Entity

    Processing history:
        ProcessingRun
    """

    __tablename__ = "memories"

    # ========================================================================
    # TABLE CONFIGURATION
    # ========================================================================
    #
    # All custom indexes are defined here.
    #
    # IMPORTANT:
    # The corresponding mapped_column() declarations below intentionally
    # DO NOT use index=True for these same fields.
    #
    # This prevents duplicate index creation errors such as:
    #
    #     ix_memories_duplicate_of_id already exists
    #
    # ========================================================================

    __table_args__ = (
        UniqueConstraint(
            "memory_id",
            name="uq_memories_memory_id",
        ),
        Index(
            "ix_memories_category",
            "category",
        ),
        Index(
            "ix_memories_intent",
            "intent",
        ),
        Index(
            "ix_memories_processing_status",
            "processing_status",
        ),
        Index(
            "ix_memories_indexed",
            "indexed",
        ),
        Index(
            "ix_memories_is_duplicate",
            "is_duplicate",
        ),
        Index(
            "ix_memories_duplicate_of_id",
            "duplicate_of_id",
        ),
        Index(
            "ix_memories_created_at",
            "created_at",
        ),
        Index(
            "ix_memories_updated_at",
            "updated_at",
        ),
        Index(
            "ix_memories_confidence_score",
            "confidence_score",
        ),
    )

    # ========================================================================
    # IDENTITY
    # ========================================================================

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    memory_id: Mapped[str] = mapped_column(
        String(80),
        nullable=False,
        unique=True,
        doc="Public MemoryOS identifier.",
    )

    # ========================================================================
    # USER-FACING CONTENT
    # ========================================================================

    title: Mapped[str] = mapped_column(
        String(160),
        nullable=False,
        default="Untitled Memory",
        server_default="Untitled Memory",
    )

    summary: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        server_default="",
    )

    # ========================================================================
    # SEARCHABLE CONTENT
    # ========================================================================

    normalized_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        server_default="",
        doc="Normalized searchable text.",
    )

    raw_ocr_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        server_default="",
        doc="Raw OCR text extracted from the image.",
    )

    # ========================================================================
    # CLASSIFICATION
    # ========================================================================

    category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="unknown",
        server_default="unknown",
    )

    intent: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="unknown",
        server_default="unknown",
    )

    confidence_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0",
    )

    classification_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0",
    )

    extraction_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0",
    )

    # ========================================================================
    # AI EXPLANATION
    # ========================================================================

    ai_explanation: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        server_default="",
    )

    why_this_matched_template: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        server_default="",
    )

    # ========================================================================
    # PROCESSING STATE
    # ========================================================================

    processing_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="pending",
        server_default="pending",
    )

    processing_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    processing_duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    # ========================================================================
    # VECTOR / SEARCH STATE
    # ========================================================================

    indexed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="0",
    )

    embedding_model: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
    )

    embedding_dimension: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    faiss_vector_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        unique=True,
    )

    indexed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ========================================================================
    # AI RAW DATA
    # ========================================================================

    ai_analysis: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    extracted_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # ========================================================================
    # DUPLICATE DETECTION
    # ========================================================================

    duplicate_of_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    is_duplicate: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="0",
    )

    # ========================================================================
    # TIMESTAMPS
    # ========================================================================

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # ========================================================================
    # RELATIONSHIPS
    # ========================================================================

    images: Mapped[list["ImageAsset"]] = relationship(
        "ImageAsset",
        back_populates="memory",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ImageAsset.sequence_number",
    )

    entities: Mapped[list["Entity"]] = relationship(
        "Entity",
        back_populates="memory",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    processing_runs: Mapped[list["ProcessingRun"]] = relationship(
        "ProcessingRun",
        back_populates="memory",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ProcessingRun.started_at.desc()",
    )

    # ========================================================================
    # COMPUTED PROPERTIES
    # ========================================================================

    @property
    def is_processed(self) -> bool:
        """Return True when processing completed successfully."""
        return self.processing_status == "completed"

    @property
    def is_processing(self) -> bool:
        """Return True while processing is active."""
        return self.processing_status == "processing"

    @property
    def has_processing_error(self) -> bool:
        """Return True when a processing error exists."""
        return bool(self.processing_error)

    @property
    def has_embedding(self) -> bool:
        """Return True when embedding metadata exists."""
        return self.embedding_model is not None and self.embedding_dimension is not None

    @property
    def is_searchable(self) -> bool:
        """Return True when the memory has an active FAISS vector."""
        return self.indexed and self.faiss_vector_id is not None

    @property
    def primary_image(self) -> "ImageAsset | None":
        """
        Return the explicitly primary image.

        Falls back to the first image when no image is marked primary.
        """

        for image in self.images:
            if image.is_primary:
                return image

        return self.images[0] if self.images else None

    # ========================================================================
    # REPRESENTATION
    # ========================================================================

    def __repr__(self) -> str:
        return (
            "<Memory("
            f"id={self.id!r}, "
            f"memory_id={self.memory_id!r}, "
            f"title={self.title!r}, "
            f"category={self.category!r}, "
            f"status={self.processing_status!r}"
            ")>"
        )


# ============================================================================
# TYPE CHECKING ONLY
# ============================================================================

if TYPE_CHECKING:
    from backend.models.entity import Entity
    from backend.models.image import ImageAsset
    from backend.models.processing_run import ProcessingRun


__all__ = [
    "Memory",
]
