"""
MemoryOS - Entity Database Model
================================

SQLAlchemy model for entities extracted from memories.

Responsibilities
----------------
- Persist AI/OCR extracted entities
- Associate entities with a Memory
- Store normalized entity values
- Store entity type and confidence
- Preserve extraction source
- Store source text/context
- Support efficient entity search
- Prevent accidental duplicate entities
- Maintain timestamps
- Support future entity linking

Architecture
------------

Memory
   │
   ├── Entity
   │      ├── person
   │      ├── organization
   │      ├── product
   │      ├── brand
   │      ├── location
   │      ├── date
   │      ├── price
   │      └── ...
   │
   └── Images

AI / OCR
    ↓
Entity extraction
    ↓
Entity model
    ↓
Database
    ↓
Search / filtering / explanation
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import (
    CheckConstraint,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database import Base

if TYPE_CHECKING:
    from backend.models.memory import Memory


# ============================================================================
# CONSTANTS
# ============================================================================

ENTITY_TYPE_MAX_LENGTH = 50
ENTITY_VALUE_MAX_LENGTH = 1000
SOURCE_MAX_LENGTH = 50

MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0


# ============================================================================
# MODEL
# ============================================================================


class Entity(Base):
    """
    Persistent entity extracted from a Memory.

    Examples
    --------
    A screenshot might produce:

        person       -> "John Smith"
        organization -> "Amazon"
        product      -> "AirPods Pro"
        price        -> "$249.99"
        order_id     -> "ORD-12345"

    The raw value is preserved in ``value`` while a normalized version is
    stored in ``normalized_value`` for searching and duplicate detection.
    """

    __tablename__ = "entities"

    # ========================================================================
    # PRIMARY KEY
    # ========================================================================

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # ========================================================================
    # MEMORY RELATIONSHIP
    # ========================================================================

    memory_id: Mapped[int] = mapped_column(
        ForeignKey(
            "memories.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    # ========================================================================
    # ENTITY INFORMATION
    # ========================================================================

    entity_type: Mapped[str] = mapped_column(
        String(ENTITY_TYPE_MAX_LENGTH),
        nullable=False,
    )

    value: Mapped[str] = mapped_column(
        String(ENTITY_VALUE_MAX_LENGTH),
        nullable=False,
    )

    normalized_value: Mapped[str] = mapped_column(
        String(ENTITY_VALUE_MAX_LENGTH),
        nullable=False,
    )

    # ========================================================================
    # CONTEXT
    # ========================================================================

    context: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    source_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # ========================================================================
    # EXTRACTION INFORMATION
    # ========================================================================

    confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
    )

    source: Mapped[str] = mapped_column(
        String(SOURCE_MAX_LENGTH),
        nullable=False,
        default="ai",
    )

    # ========================================================================
    # TIMESTAMPS
    # ========================================================================

    created_at: Mapped[datetime] = mapped_column(
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    updated_at: Mapped[datetime] = mapped_column(
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # ========================================================================
    # RELATIONSHIPS
    # ========================================================================

    memory: Mapped["Memory"] = relationship(
        "Memory",
        back_populates="entities",
    )

    # ========================================================================
    # TABLE CONSTRAINTS
    # ========================================================================

    __table_args__ = (
        CheckConstraint(
            "confidence >= 0.0 AND confidence <= 1.0",
            name="ck_entities_confidence_range",
        ),
        CheckConstraint(
            "length(entity_type) > 0",
            name="ck_entities_entity_type_not_empty",
        ),
        CheckConstraint(
            "length(value) > 0",
            name="ck_entities_value_not_empty",
        ),
        CheckConstraint(
            "length(normalized_value) > 0",
            name="ck_entities_normalized_value_not_empty",
        ),
        # Prevent the exact same normalized entity from being inserted
        # repeatedly for the same memory and entity type.
        UniqueConstraint(
            "memory_id",
            "entity_type",
            "normalized_value",
            name="uq_entities_memory_type_normalized",
        ),
        # Fast lookup by memory.
        Index(
            "ix_entities_memory_type",
            "memory_id",
            "entity_type",
        ),
        # Fast normalized entity search.
        Index(
            "ix_entities_normalized_value",
            "normalized_value",
        ),
        # Useful for filtering high-confidence entities.
        Index(
            "ix_entities_type_confidence",
            "entity_type",
            "confidence",
        ),
        # Useful for reverse entity lookup.
        Index(
            "ix_entities_type_normalized",
            "entity_type",
            "normalized_value",
        ),
    )

    # ========================================================================
    # REPRESENTATION
    # ========================================================================

    def __repr__(self) -> str:
        """
        Developer-friendly representation.

        Avoids printing potentially large context/source text.
        """

        return (
            f"<Entity("
            f"id={self.id!r}, "
            f"memory_id={self.memory_id!r}, "
            f"type={self.entity_type!r}, "
            f"value={self.value!r}"
            f")>"
        )

    # ========================================================================
    # NORMALIZATION HELPERS
    # ========================================================================

    @staticmethod
    def normalize_value(value: str) -> str:
        """
        Normalize an entity value for matching.

        This intentionally stays conservative.

        Example
        -------
            "  Amazon  " -> "amazon"

        More sophisticated normalization can later be handled by the
        entity-extraction service.
        """

        return " ".join(value.strip().lower().split())

    # ========================================================================
    # FACTORY
    # ========================================================================

    @classmethod
    def create(
        cls,
        *,
        memory_id: int,
        entity_type: str,
        value: str,
        normalized_value: str | None = None,
        context: str | None = None,
        source_text: str | None = None,
        confidence: float = 0.0,
        source: str = "ai",
    ) -> "Entity":
        """
        Create a validated Entity instance.

        This does not commit anything to the database.

        The service layer remains responsible for persistence.
        """

        clean_value = value.strip()

        if not clean_value:
            raise ValueError("Entity value cannot be empty.")

        clean_type = entity_type.strip().lower()

        if not clean_type:
            raise ValueError("Entity type cannot be empty.")

        clean_source = source.strip().lower()

        if not clean_source:
            raise ValueError("Entity source cannot be empty.")

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Entity confidence must be between 0.0 and 1.0.")

        if normalized_value is None:
            normalized_value = cls.normalize_value(clean_value)
        else:
            normalized_value = normalized_value.strip().lower()

        if not normalized_value:
            raise ValueError("Normalized entity value cannot be empty.")

        return cls(
            memory_id=memory_id,
            entity_type=clean_type,
            value=clean_value,
            normalized_value=normalized_value,
            context=context,
            source_text=source_text,
            confidence=confidence,
            source=clean_source,
        )

    # ========================================================================
    # UPDATE HELPER
    # ========================================================================

    def update_confidence(
        self,
        confidence: float,
    ) -> None:
        """
        Update extraction confidence safely.
        """

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("Entity confidence must be between 0.0 and 1.0.")

        self.confidence = confidence

    # ========================================================================
    # SERIALIZATION HELPER
    # ========================================================================

    def to_dict(self) -> dict[str, object]:
        """
        Return a safe dictionary representation.

        Useful for debugging, logging, or service-layer processing.
        """

        return {
            "id": self.id,
            "memory_id": self.memory_id,
            "entity_type": self.entity_type,
            "value": self.value,
            "normalized_value": self.normalized_value,
            "context": self.context,
            "confidence": self.confidence,
            "source": self.source,
            "source_text": self.source_text,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "Entity",
]
