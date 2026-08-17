"""
MemoryOS - Image Asset Database Model
=====================================

Persistent database representation of an image belonging to a MemoryOS
memory.

The physical image file is stored by the upload/storage layer.

This model stores image metadata, processing state, OCR state, vision state,
hashing information, and its relationship to a Memory.

Architecture
------------

                Upload API
                    │
                    ▼
              Physical File
              data/uploads/
                    │
                    ▼
              ImageAsset
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
         OCR      Vision    Hashing
          │         │         │
          └─────────┼─────────┘
                    ▼
                Processing
                    │
                    ▼
                Embedding
                    │
                    ▼
                  FAISS


Responsibilities
----------------
- Store uploaded image metadata
- Link image to a Memory
- Track original and stored filenames
- Track MIME type and image format
- Track dimensions
- Track file and image hashes
- Track OCR processing state
- Track processing confidence
- Track sequence/order within a memory
- Identify primary images
- Track processing timestamps
- Support duplicate detection
- Support future thumbnails/derivatives
- Provide safe model-level helpers

Important
---------
This model does NOT:

- receive HTTP uploads
- read UploadFile objects
- perform OCR
- call Gemini
- generate embeddings
- write directly to FAISS

Those responsibilities belong to the appropriate service layers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
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

MAX_FILENAME_LENGTH = 255
MAX_MIME_TYPE_LENGTH = 100
MAX_EXTENSION_LENGTH = 20
MAX_FORMAT_LENGTH = 30
MAX_COLOR_MODE_LENGTH = 30
MAX_HASH_LENGTH = 128
MAX_STATUS_LENGTH = 30
MAX_ERROR_LENGTH = 4000

MIN_OCR_CONFIDENCE = 0.0
MAX_OCR_CONFIDENCE = 1.0


# ============================================================================
# PROCESSING STATUS
# ============================================================================

IMAGE_PROCESSING_STATUSES = frozenset(
    {
        "pending",
        "processing",
        "completed",
        "partial",
        "failed",
    }
)


# ============================================================================
# MODEL
# ============================================================================


class ImageAsset(Base):
    """
    Database record representing one image associated with a Memory.

    One Memory can contain multiple ImageAsset records.

    Example:

        Memory #42
            ├── ImageAsset #1
            ├── ImageAsset #2
            └── ImageAsset #3

    The actual image bytes are stored outside the database.
    """

    __tablename__ = "image_assets"

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
    )

    # ========================================================================
    # FILE INFORMATION
    # ========================================================================

    original_filename: Mapped[str] = mapped_column(
        String(MAX_FILENAME_LENGTH),
        nullable=False,
    )

    stored_filename: Mapped[str] = mapped_column(
        String(MAX_FILENAME_LENGTH),
        nullable=False,
    )

    file_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    thumbnail_path: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    thumbnail_url: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # ========================================================================
    # FILE TYPE
    # ========================================================================

    mime_type: Mapped[str] = mapped_column(
        String(MAX_MIME_TYPE_LENGTH),
        nullable=False,
    )

    extension: Mapped[str] = mapped_column(
        String(MAX_EXTENSION_LENGTH),
        nullable=False,
    )

    image_format: Mapped[str] = mapped_column(
        String(MAX_FORMAT_LENGTH),
        nullable=False,
    )

    # ========================================================================
    # IMAGE DIMENSIONS
    # ========================================================================

    width: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    height: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    color_mode: Mapped[str | None] = mapped_column(
        String(MAX_COLOR_MODE_LENGTH),
        nullable=True,
    )

    # ========================================================================
    # FILE SIZE
    # ========================================================================

    file_size_bytes: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    # ========================================================================
    # HASHING / DUPLICATE DETECTION
    #
    # IMPORTANT:
    #
    # Do NOT put index=True here.
    #
    # The indexes are explicitly defined once inside __table_args__ below.
    # This prevents:
    #
    #     index ix_image_assets_file_hash already exists
    #
    # ========================================================================

    file_hash: Mapped[str] = mapped_column(
        String(MAX_HASH_LENGTH),
        nullable=False,
    )

    image_hash: Mapped[str | None] = mapped_column(
        String(MAX_HASH_LENGTH),
        nullable=True,
    )

    # ========================================================================
    # OCR STATE
    # ========================================================================

    ocr_processed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="0",
    )

    ocr_confidence: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        server_default="0",
    )

    ocr_language: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
    )

    ocr_text: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # ========================================================================
    # PROCESSING STATE
    # ========================================================================

    processing_status: Mapped[str] = mapped_column(
        String(MAX_STATUS_LENGTH),
        nullable=False,
        default="pending",
        server_default="pending",
    )

    processing_error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    processing_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    # ========================================================================
    # ORDER / PRIMARY IMAGE
    # ========================================================================

    sequence_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    is_primary: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="0",
    )

    # ========================================================================
    # AI / VISION STATE
    # ========================================================================

    vision_processed: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="0",
    )

    vision_confidence: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    # ========================================================================
    # TIMESTAMPS
    # ========================================================================

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        server_default="CURRENT_TIMESTAMP",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        server_default="CURRENT_TIMESTAMP",
    )

    processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    ocr_processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    vision_processed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    # ========================================================================
    # RELATIONSHIPS
    # ========================================================================

    memory: Mapped["Memory"] = relationship(
        "Memory",
        back_populates="images",
    )

    # ========================================================================
    # TABLE CONSTRAINTS / INDEXES
    #
    # IMPORTANT:
    #
    # All custom indexes are defined HERE.
    #
    # Do NOT add index=True to the corresponding mapped_column() fields.
    # ========================================================================

    __table_args__ = (
        # --------------------------------------------------------------------
        # BASIC VALIDITY
        # --------------------------------------------------------------------
        CheckConstraint(
            "width > 0",
            name="ck_image_assets_width_positive",
        ),
        CheckConstraint(
            "height > 0",
            name="ck_image_assets_height_positive",
        ),
        CheckConstraint(
            "file_size_bytes >= 0",
            name="ck_image_assets_file_size_nonnegative",
        ),
        CheckConstraint(
            "sequence_number >= 0",
            name="ck_image_assets_sequence_nonnegative",
        ),
        CheckConstraint(
            "processing_attempts >= 0",
            name="ck_image_assets_attempts_nonnegative",
        ),
        # --------------------------------------------------------------------
        # OCR CONFIDENCE
        # --------------------------------------------------------------------
        CheckConstraint(
            "ocr_confidence >= 0.0 AND ocr_confidence <= 1.0",
            name="ck_image_assets_ocr_confidence_range",
        ),
        # --------------------------------------------------------------------
        # VISION CONFIDENCE
        # --------------------------------------------------------------------
        CheckConstraint(
            """
            vision_confidence IS NULL
            OR (
                vision_confidence >= 0.0
                AND vision_confidence <= 1.0
            )
            """,
            name="ck_image_assets_vision_confidence_range",
        ),
        # --------------------------------------------------------------------
        # PROCESSING STATUS
        # --------------------------------------------------------------------
        CheckConstraint(
            """
            processing_status IN (
                'pending',
                'processing',
                'completed',
                'partial',
                'failed'
            )
            """,
            name="ck_image_assets_processing_status",
        ),
        # --------------------------------------------------------------------
        # UNIQUE STORED FILE
        # --------------------------------------------------------------------
        UniqueConstraint(
            "stored_filename",
            name="uq_image_assets_stored_filename",
        ),
        # --------------------------------------------------------------------
        # INDEXES
        #
        # Each index exists exactly ONCE.
        # --------------------------------------------------------------------
        Index(
            "ix_image_assets_memory_id",
            "memory_id",
        ),
        Index(
            "ix_image_assets_memory_sequence",
            "memory_id",
            "sequence_number",
        ),
        Index(
            "ix_image_assets_memory_primary",
            "memory_id",
            "is_primary",
        ),
        Index(
            "ix_image_assets_memory_status",
            "memory_id",
            "processing_status",
        ),
        Index(
            "ix_image_assets_processing_status",
            "processing_status",
        ),
        Index(
            "ix_image_assets_file_hash",
            "file_hash",
        ),
        Index(
            "ix_image_assets_image_hash",
            "image_hash",
        ),
        Index(
            "ix_image_assets_created_at",
            "created_at",
        ),
        Index(
            "ix_image_assets_updated_at",
            "updated_at",
        ),
    )

    # ========================================================================
    # REPRESENTATION
    # ========================================================================

    def __repr__(self) -> str:
        """
        Safe developer representation.

        Large OCR text and paths are intentionally excluded.
        """

        return (
            "<ImageAsset("
            f"id={self.id!r}, "
            f"memory_id={self.memory_id!r}, "
            f"filename={self.original_filename!r}, "
            f"status={self.processing_status!r}"
            ")>"
        )

    # ========================================================================
    # STATUS HELPERS
    # ========================================================================

    def mark_processing(self) -> None:
        """
        Mark this image as currently being processed.
        """

        self.processing_status = "processing"
        self.processing_attempts += 1
        self.processing_error = None

    def mark_completed(self) -> None:
        """
        Mark image processing as successfully completed.
        """

        now = datetime.now(timezone.utc)

        self.processing_status = "completed"
        self.processed_at = now
        self.processing_error = None

    def mark_partial(
        self,
        error: str | None = None,
    ) -> None:
        """
        Mark image processing as partially successful.
        """

        self.processing_status = "partial"
        self.processing_error = error[:MAX_ERROR_LENGTH] if error else None

    def mark_failed(
        self,
        error: str,
    ) -> None:
        """
        Mark image processing as failed.
        """

        self.processing_status = "failed"
        self.processing_error = error[:MAX_ERROR_LENGTH]

    def reset_processing(self) -> None:
        """
        Reset processing state so the asset can be retried.
        """

        self.processing_status = "pending"
        self.processing_error = None
        self.processed_at = None

    # ========================================================================
    # OCR HELPERS
    # ========================================================================

    def mark_ocr_completed(
        self,
        *,
        text: str,
        confidence: float,
        language: str | None = None,
    ) -> None:
        """
        Store successful OCR output.
        """

        if not MIN_OCR_CONFIDENCE <= confidence <= MAX_OCR_CONFIDENCE:
            raise ValueError("OCR confidence must be between 0.0 and 1.0.")

        now = datetime.now(timezone.utc)

        self.ocr_processed = True
        self.ocr_text = text
        self.ocr_confidence = confidence
        self.ocr_language = language
        self.ocr_processed_at = now

    def mark_ocr_failed(
        self,
        error: str,
    ) -> None:
        """
        Record an OCR failure without deleting the image.
        """

        self.ocr_processed = False
        self.processing_error = error[:MAX_ERROR_LENGTH]

    # ========================================================================
    # VISION HELPERS
    # ========================================================================

    def mark_vision_completed(
        self,
        confidence: float | None = None,
    ) -> None:
        """
        Mark vision analysis as completed.
        """

        if confidence is not None:
            if not 0.0 <= confidence <= 1.0:
                raise ValueError("Vision confidence must be between 0.0 and 1.0.")

        self.vision_processed = True
        self.vision_confidence = confidence
        self.vision_processed_at = datetime.now(timezone.utc)

    # ========================================================================
    # PRIMARY IMAGE
    # ========================================================================

    def set_primary(self) -> None:
        """
        Mark this image as the primary image for its memory.

        The service layer should ensure other images belonging to the same
        memory are unmarked when necessary.
        """

        self.is_primary = True

    def unset_primary(self) -> None:
        """
        Remove primary-image status.
        """

        self.is_primary = False

    # ========================================================================
    # DUPLICATE HELPERS
    # ========================================================================

    def has_same_file_hash(
        self,
        file_hash: str,
    ) -> bool:
        """
        Check exact file-level duplication.
        """

        return bool(file_hash and self.file_hash and self.file_hash == file_hash)

    def has_same_image_hash(
        self,
        image_hash: str,
    ) -> bool:
        """
        Check image-content hash equality.

        Useful when the same image is saved with different file metadata.
        """

        if not image_hash or not self.image_hash:
            return False

        return self.image_hash == image_hash

    # ========================================================================
    # DISPLAY HELPERS
    # ========================================================================

    @property
    def aspect_ratio(self) -> float:
        """
        Return the image aspect ratio.
        """

        if self.height <= 0:
            return 0.0

        return self.width / self.height

    @property
    def is_landscape(self) -> bool:
        """
        Return True if the image is wider than it is tall.
        """

        return self.width > self.height

    @property
    def is_portrait(self) -> bool:
        """
        Return True if the image is taller than it is wide.
        """

        return self.height > self.width

    @property
    def is_square(self) -> bool:
        """
        Return True if the image has equal width and height.
        """

        return self.width == self.height

    @property
    def has_ocr(self) -> bool:
        """
        Return True when OCR has successfully produced text.
        """

        return self.ocr_processed and bool(self.ocr_text)

    @property
    def has_vision(self) -> bool:
        """
        Return True when vision processing completed.
        """

        return self.vision_processed

    @property
    def is_processed(self) -> bool:
        """
        Return True when image processing completed.
        """

        return self.processing_status == "completed"

    @property
    def is_processing(self) -> bool:
        """
        Return True while image processing is active.
        """

        return self.processing_status == "processing"

    @property
    def has_processing_error(self) -> bool:
        """
        Return True when an image processing error exists.
        """

        return bool(self.processing_error)

    # ========================================================================
    # SERIALIZATION
    # ========================================================================

    def to_dict(self) -> dict[str, object]:
        """
        Return a safe metadata representation.

        Raw binary image content is never returned.
        """

        return {
            "id": self.id,
            "memory_id": self.memory_id,
            "original_filename": self.original_filename,
            "stored_filename": self.stored_filename,
            "file_path": self.file_path,
            "thumbnail_path": self.thumbnail_path,
            "thumbnail_url": self.thumbnail_url,
            "mime_type": self.mime_type,
            "extension": self.extension,
            "image_format": self.image_format,
            "width": self.width,
            "height": self.height,
            "color_mode": self.color_mode,
            "file_size_bytes": self.file_size_bytes,
            "file_hash": self.file_hash,
            "image_hash": self.image_hash,
            "ocr_processed": self.ocr_processed,
            "ocr_confidence": self.ocr_confidence,
            "ocr_language": self.ocr_language,
            "processing_status": self.processing_status,
            "processing_error": self.processing_error,
            "processing_attempts": self.processing_attempts,
            "sequence_number": self.sequence_number,
            "is_primary": self.is_primary,
            "vision_processed": self.vision_processed,
            "vision_confidence": self.vision_confidence,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "processed_at": self.processed_at,
            "ocr_processed_at": self.ocr_processed_at,
            "vision_processed_at": self.vision_processed_at,
        }


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "ImageAsset",
    "IMAGE_PROCESSING_STATUSES",
]
