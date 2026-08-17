"""
MemoryOS - Database Models

SQLAlchemy ORM models for the complete MemoryOS MVP.

Core entities:

    Memory
        ↓
    ImageAsset
        ↓
    Entity
        ↓
    SearchHistory

The database stores metadata and structured memory information.

FAISS remains responsible for vector retrieval.

SQLite remains the source of truth for persistent metadata.

Architecture:

    Uploaded Image
          │
          ▼
      ImageAsset
          │
          ▼
        Memory
       /  |  \
      /   |   \
     ▼    ▼    ▼
  Entity AI   Search
           Analysis
             │
             ▼
          FAISS ID

Important design principle:

    Database ≠ Vector Database

SQLite stores:
    - memory metadata
    - image paths
    - OCR text
    - AI results
    - entities
    - confidence
    - processing status
    - timestamps

FAISS stores:
    - embedding vectors
    - vector IDs

This separation allows the FAISS index to be rebuilt without losing memories.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import (
    Mapped,
    mapped_column,
    relationship,
)

from backend.database import Base

# ============================================================================
# ENUMS
# ============================================================================


class ProcessingStatus(str, Enum):
    """
    Processing lifecycle of an uploaded memory.
    """

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class MemoryCategory(str, Enum):
    """
    High-level memory classification.

    Gemini/local AI can later map images into these categories.
    """

    PRODUCT = "product"
    RECEIPT = "receipt"
    DOCUMENT = "document"
    SCREENSHOT = "screenshot"
    TICKET = "ticket"
    NOTE = "note"
    SOCIAL = "social"
    TRAVEL = "travel"
    FOOD = "food"
    FINANCE = "finance"
    SHOPPING = "shopping"
    EDUCATION = "education"
    WORK = "work"
    PERSONAL = "personal"
    OTHER = "other"
    UNKNOWN = "unknown"


class MemoryIntent(str, Enum):
    """
    Detected purpose/intent of a memory.
    """

    REFERENCE = "reference"
    PURCHASE = "purchase"
    PAYMENT = "payment"
    TRAVEL = "travel"
    COMMUNICATION = "communication"
    LEARNING = "learning"
    TASK = "task"
    INFORMATION = "information"
    REMINDER = "reminder"
    TRACKING = "tracking"
    UNKNOWN = "unknown"


class EntityType(str, Enum):
    """
    Entity types extracted from memories.
    """

    PERSON = "person"
    ORGANIZATION = "organization"
    PRODUCT = "product"
    BRAND = "brand"
    LOCATION = "location"
    DATE = "date"
    TIME = "time"
    PRICE = "price"
    CURRENCY = "currency"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    ORDER_ID = "order_id"
    TRANSACTION_ID = "transaction_id"
    INVOICE_ID = "invoice_id"
    EVENT = "event"
    CATEGORY = "category"
    OTHER = "other"


# ============================================================================
# DATETIME HELPERS
# ============================================================================


def utc_now() -> datetime:
    """
    Return a timezone-aware UTC datetime.

    Keeping timestamps in UTC makes future cloud deployment and analytics
    significantly easier.
    """

    return datetime.now(timezone.utc)


# ============================================================================
# MEMORY
# ============================================================================


class Memory(Base):
    """
    Central MemoryOS memory record.

    A Memory represents the semantic understanding of an image or group
    of related images.

    Example:

        Image:
            screenshot of Adidas shoes

        Memory:
            title = "Blue Adidas shoes"
            category = "product"
            intent = "shopping"
            text = "Blue Adidas running shoes..."
            confidence = 0.94
    """

    __tablename__ = "memories"

    # ------------------------------------------------------------------------
    # PRIMARY KEY
    # ------------------------------------------------------------------------

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # ------------------------------------------------------------------------
    # CORE CONTENT
    # ------------------------------------------------------------------------

    title: Mapped[str] = mapped_column(
        String(160),
        nullable=False,
        default="Untitled Memory",
    )

    normalized_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
    )

    raw_ocr_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
    )

    summary: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
    )

    # ------------------------------------------------------------------------
    # CLASSIFICATION
    # ------------------------------------------------------------------------

    category: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=MemoryCategory.UNKNOWN.value,
        index=True,
    )

    intent: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default=MemoryIntent.UNKNOWN.value,
        index=True,
    )

    # ------------------------------------------------------------------------
    # AI CONFIDENCE
    # ------------------------------------------------------------------------

    confidence_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    classification_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    extraction_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    # ------------------------------------------------------------------------
    # EXPLAINABILITY
    # ------------------------------------------------------------------------

    why_this_matched_template: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
    )

    ai_explanation: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
    )

    # ------------------------------------------------------------------------
    # AI STRUCTURED DATA
    # ------------------------------------------------------------------------

    ai_analysis: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    extracted_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # ------------------------------------------------------------------------
    # VECTOR SEARCH
    # ------------------------------------------------------------------------

    embedding_model: Mapped[str | None] = mapped_column(
        String(255),
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
        index=True,
    )

    embedding_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    indexed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        index=True,
    )

    # ------------------------------------------------------------------------
    # PROCESSING
    # ------------------------------------------------------------------------

    processing_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default=ProcessingStatus.PENDING.value,
        index=True,
    )

    processing_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    processing_duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    # ------------------------------------------------------------------------
    # DUPLICATE DETECTION
    # ------------------------------------------------------------------------

    content_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        unique=True,
        index=True,
    )

    perceptual_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    duplicate_of_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "memories.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # ------------------------------------------------------------------------
    # TIMESTAMPS
    # ------------------------------------------------------------------------

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    indexed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ------------------------------------------------------------------------
    # RELATIONSHIPS
    # ------------------------------------------------------------------------

    images: Mapped[list["ImageAsset"]] = relationship(
        "ImageAsset",
        back_populates="memory",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    entities: Mapped[list["Entity"]] = relationship(
        "Entity",
        back_populates="memory",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    duplicate_of: Mapped["Memory | None"] = relationship(
        "Memory",
        remote_side=[id],
        foreign_keys=[duplicate_of_id],
    )

    # ------------------------------------------------------------------------
    # TABLE INDEXES
    # ------------------------------------------------------------------------

    __table_args__ = (
        Index(
            "ix_memories_category_intent",
            "category",
            "intent",
        ),
        Index(
            "ix_memories_status_created",
            "processing_status",
            "created_at",
        ),
        Index(
            "ix_memories_indexed_status",
            "indexed",
            "processing_status",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Memory("
            f"id={self.id}, "
            f"title={self.title!r}, "
            f"category={self.category!r}, "
            f"status={self.processing_status!r}"
            f")>"
        )


# ============================================================================
# IMAGE ASSET
# ============================================================================


class ImageAsset(Base):
    """
    Physical image associated with a Memory.

    One Memory can contain multiple images.

    Example:

        Memory
        ├── screenshot_1.png
        ├── screenshot_2.png
        └── screenshot_3.png

    This supports the required multi-image upload capability.
    """

    __tablename__ = "image_assets"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # ------------------------------------------------------------------------
    # MEMORY RELATION
    # ------------------------------------------------------------------------

    memory_id: Mapped[int] = mapped_column(
        ForeignKey(
            "memories.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    # ------------------------------------------------------------------------
    # FILE INFORMATION
    # ------------------------------------------------------------------------

    original_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    stored_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    original_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    thumbnail_path: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    mime_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    extension: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    file_size_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    # ------------------------------------------------------------------------
    # IMAGE DIMENSIONS
    # ------------------------------------------------------------------------

    width: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    height: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    color_mode: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    image_format: Mapped[str | None] = mapped_column(
        String(30),
        nullable=True,
    )

    # ------------------------------------------------------------------------
    # HASHES
    # ------------------------------------------------------------------------

    file_hash: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    image_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        index=True,
    )

    # ------------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------------

    ocr_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
    )

    ocr_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    ocr_language: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
    )

    ocr_processed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ------------------------------------------------------------------------
    # PROCESSING
    # ------------------------------------------------------------------------

    processing_status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default=ProcessingStatus.PENDING.value,
        index=True,
    )

    processing_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------------
    # ORDER / GROUPING
    # ------------------------------------------------------------------------

    sequence_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    is_primary: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    # ------------------------------------------------------------------------
    # TIMESTAMPS
    # ------------------------------------------------------------------------

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ------------------------------------------------------------------------
    # RELATIONSHIP
    # ------------------------------------------------------------------------

    memory: Mapped["Memory"] = relationship(
        "Memory",
        back_populates="images",
    )

    # ------------------------------------------------------------------------
    # INDEXES
    # ------------------------------------------------------------------------

    __table_args__ = (
        UniqueConstraint(
            "memory_id",
            "sequence_number",
            name="uq_memory_image_sequence",
        ),
        Index(
            "ix_image_assets_hash",
            "file_hash",
            "image_hash",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<ImageAsset("
            f"id={self.id}, "
            f"filename={self.original_filename!r}, "
            f"memory_id={self.memory_id}"
            f")>"
        )


# ============================================================================
# ENTITY
# ============================================================================


class Entity(Base):
    """
    Structured entity extracted from a Memory.

    Example:

        Memory:
            "Bought blue Adidas shoes for ₹14,999"

        Entities:
            Adidas      → BRAND
            blue shoes  → PRODUCT
            ₹14,999     → PRICE
    """

    __tablename__ = "entities"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # ------------------------------------------------------------------------
    # RELATION
    # ------------------------------------------------------------------------

    memory_id: Mapped[int] = mapped_column(
        ForeignKey(
            "memories.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    # ------------------------------------------------------------------------
    # ENTITY CONTENT
    # ------------------------------------------------------------------------

    entity_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    value: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
    )

    normalized_value: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
    )

    context: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    # ------------------------------------------------------------------------
    # SOURCE
    # ------------------------------------------------------------------------

    source: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="ai",
    )

    source_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------------
    # TIMESTAMP
    # ------------------------------------------------------------------------

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    # ------------------------------------------------------------------------
    # RELATIONSHIP
    # ------------------------------------------------------------------------

    memory: Mapped["Memory"] = relationship(
        "Memory",
        back_populates="entities",
    )

    # ------------------------------------------------------------------------
    # INDEXES
    # ------------------------------------------------------------------------

    __table_args__ = (
        Index(
            "ix_entities_type_value",
            "entity_type",
            "normalized_value",
        ),
        Index(
            "ix_entities_memory_type",
            "memory_id",
            "entity_type",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<Entity("
            f"id={self.id}, "
            f"type={self.entity_type!r}, "
            f"value={self.value!r}"
            f")>"
        )


# ============================================================================
# SEARCH HISTORY
# ============================================================================


class SearchHistory(Base):
    """
    Stores search activity for analytics and debugging.

    Example:

        Query:
            "Where did I see Adidas shoes?"

        Stored:
            query
            result count
            search duration
            top result
            filters
    """

    __tablename__ = "search_history"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # ------------------------------------------------------------------------
    # QUERY
    # ------------------------------------------------------------------------

    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    normalized_query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # ------------------------------------------------------------------------
    # SEARCH CONFIGURATION
    # ------------------------------------------------------------------------

    search_mode: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="hybrid",
    )

    filters: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    top_k: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=10,
    )

    # ------------------------------------------------------------------------
    # RESULTS
    # ------------------------------------------------------------------------

    result_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    top_memory_id: Mapped[int | None] = mapped_column(
        ForeignKey(
            "memories.id",
            ondelete="SET NULL",
        ),
        nullable=True,
        index=True,
    )

    # ------------------------------------------------------------------------
    # PERFORMANCE
    # ------------------------------------------------------------------------

    search_duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    embedding_duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    reranking_duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    # ------------------------------------------------------------------------
    # TIMESTAMP
    # ------------------------------------------------------------------------

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    # ------------------------------------------------------------------------
    # INDEXES
    # ------------------------------------------------------------------------

    __table_args__ = (
        Index(
            "ix_search_history_created_query",
            "created_at",
            "query",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<SearchHistory("
            f"id={self.id}, "
            f"query={self.query!r}, "
            f"results={self.result_count}"
            f")>"
        )


# ============================================================================
# PROCESSING RUN
# ============================================================================


class ProcessingRun(Base):
    """
    Tracks processing execution for batch indexing.

    This is useful for the hackathon demo because we can show:

        25 images
        ├── 23 completed
        ├── 1 partial
        └── 1 failed

    without losing the individual image errors.
    """

    __tablename__ = "processing_runs"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # ------------------------------------------------------------------------
    # RUN IDENTIFICATION
    # ------------------------------------------------------------------------

    run_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        unique=True,
        index=True,
    )

    run_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="batch",
    )

    # ------------------------------------------------------------------------
    # COUNTS
    # ------------------------------------------------------------------------

    total_items: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    successful_items: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    partial_items: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    failed_items: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
    )

    # ------------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------------

    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default=ProcessingStatus.PENDING.value,
        index=True,
    )

    error_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # ------------------------------------------------------------------------
    # PERFORMANCE
    # ------------------------------------------------------------------------

    duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    # ------------------------------------------------------------------------
    # EXTRA INFORMATION
    # ------------------------------------------------------------------------

    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # ------------------------------------------------------------------------
    # TIMESTAMPS
    # ------------------------------------------------------------------------

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ------------------------------------------------------------------------
    # INDEX
    # ------------------------------------------------------------------------

    __table_args__ = (
        Index(
            "ix_processing_runs_status_started",
            "status",
            "started_at",
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<ProcessingRun("
            f"id={self.id}, "
            f"run_id={self.run_id!r}, "
            f"status={self.status!r}"
            f")>"
        )


# ============================================================================
# MODEL EXPORTS
# ============================================================================


__all__ = [
    "ProcessingStatus",
    "MemoryCategory",
    "MemoryIntent",
    "EntityType",
    "Memory",
    "ImageAsset",
    "Entity",
    "SearchHistory",
    "ProcessingRun",
]
