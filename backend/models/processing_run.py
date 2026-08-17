"""
MemoryOS - Processing Run Model
===============================

Premium SQLAlchemy model for tracking background and synchronous
MemoryOS processing runs.

Responsibilities
----------------
- Track OCR / vision / AI / embedding / indexing runs
- Track batch progress
- Track success / partial / failure state
- Track processing duration
- Track retry attempts
- Store structured error information
- Store extensible JSON metadata
- Support observability and dashboard statistics
- Provide safe database constraints and indexes

Architecture
------------

Upload
   │
   ▼
Memory
   │
   ├── OCR
   ├── Vision
   ├── AI analysis
   ├── Embedding
   └── FAISS indexing
            │
            ▼
      ProcessingRun
            │
            ├── status
            ├── progress
            ├── errors
            ├── timing
            └── metadata

This model intentionally contains processing-run information only.
Actual Memory/Image/Entity data belongs to their respective models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base

# ============================================================================
# CONSTANTS
# ============================================================================

MAX_RUN_ID_LENGTH = 100
MAX_RUN_TYPE_LENGTH = 50
MAX_STATUS_LENGTH = 20
MAX_ERROR_LENGTH = 5000


# ============================================================================
# ENUM-LIKE VALUES
# ============================================================================

PROCESSING_RUN_TYPES = frozenset(
    {
        "upload",
        "ocr",
        "vision",
        "ai_analysis",
        "embedding",
        "indexing",
        "search",
        "batch",
        "reprocessing",
        "maintenance",
        "cleanup",
        "other",
    }
)

PROCESSING_STATUSES = frozenset(
    {
        "pending",
        "processing",
        "completed",
        "partial",
        "failed",
        "cancelled",
    }
)


# ============================================================================
# TIME HELPER
# ============================================================================


def utc_now() -> datetime:
    """
    Return a timezone-aware UTC datetime.

    Keeping this helper local makes timestamp creation consistent
    throughout the model.
    """

    return datetime.now(timezone.utc)


# ============================================================================
# MODEL
# ============================================================================


class ProcessingRun(Base):
    """
    Persistent processing-run record.

    A single row represents one logical processing operation.

    Examples
    --------

    OCR batch:

        run_type = "ocr"

    Gemini analysis:

        run_type = "ai_analysis"

    Embedding generation:

        run_type = "embedding"

    FAISS rebuild:

        run_type = "indexing"

    Batch upload:

        run_type = "upload"
    """

    __tablename__ = "processing_runs"

    # ========================================================================
    # PRIMARY KEY
    # ========================================================================

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # ========================================================================
    # PUBLIC RUN IDENTIFIER
    # ========================================================================

    run_id: Mapped[str] = mapped_column(
        String(MAX_RUN_ID_LENGTH),
        unique=True,
        nullable=False,
        index=True,
    )

    # ========================================================================
    # RUN TYPE
    # ========================================================================

    run_type: Mapped[str] = mapped_column(
        String(MAX_RUN_TYPE_LENGTH),
        nullable=False,
        index=True,
    )

    # ========================================================================
    # STATUS
    # ========================================================================

    status: Mapped[str] = mapped_column(
        String(MAX_STATUS_LENGTH),
        nullable=False,
        default="pending",
        server_default="pending",
        index=True,
    )

    # ========================================================================
    # PROGRESS
    # ========================================================================

    total_items: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    successful_items: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    partial_items: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    failed_items: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    # ========================================================================
    # RETRY / ATTEMPT INFORMATION
    # ========================================================================

    attempt_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default="1",
    )

    max_attempts: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=3,
        server_default="3",
    )

    # ========================================================================
    # ERROR INFORMATION
    # ========================================================================

    error_summary: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    error_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    # ========================================================================
    # TIMING
    # ========================================================================

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    completed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )

    duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    # ========================================================================
    # EXTENSIBLE METADATA
    # ========================================================================

    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # ========================================================================
    # CREATED / UPDATED
    # ========================================================================

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        onupdate=utc_now,
    )

    # ========================================================================
    # DATABASE CONSTRAINTS
    # ========================================================================

    __table_args__ = (
        CheckConstraint(
            "total_items >= 0",
            name="ck_processing_runs_total_items_nonnegative",
        ),
        CheckConstraint(
            "successful_items >= 0",
            name="ck_processing_runs_successful_items_nonnegative",
        ),
        CheckConstraint(
            "partial_items >= 0",
            name="ck_processing_runs_partial_items_nonnegative",
        ),
        CheckConstraint(
            "failed_items >= 0",
            name="ck_processing_runs_failed_items_nonnegative",
        ),
        CheckConstraint(
            "attempt_number >= 1",
            name="ck_processing_runs_attempt_positive",
        ),
        CheckConstraint(
            "max_attempts >= 1",
            name="ck_processing_runs_max_attempts_positive",
        ),
        CheckConstraint(
            "error_count >= 0",
            name="ck_processing_runs_error_count_nonnegative",
        ),
        CheckConstraint(
            "duration_ms IS NULL OR duration_ms >= 0",
            name="ck_processing_runs_duration_nonnegative",
        ),
        CheckConstraint(
            """
            successful_items
            + partial_items
            + failed_items
            <= total_items
            """,
            name="ck_processing_runs_progress_valid",
        ),
        Index(
            "ix_processing_runs_status_created",
            "status",
            "created_at",
        ),
        Index(
            "ix_processing_runs_type_created",
            "run_type",
            "created_at",
        ),
        Index(
            "ix_processing_runs_active",
            "status",
        ),
    )

    # ========================================================================
    # REPRESENTATION
    # ========================================================================

    def __repr__(self) -> str:
        return (
            f"<ProcessingRun("
            f"id={self.id!r}, "
            f"run_id={self.run_id!r}, "
            f"run_type={self.run_type!r}, "
            f"status={self.status!r}, "
            f"total_items={self.total_items!r}"
            f")>"
        )

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def processed_items(self) -> int:
        """
        Return the number of items that have finished processing.
        """

        return self.successful_items + self.partial_items + self.failed_items

    @property
    def remaining_items(self) -> int:
        """
        Return the number of items still waiting for processing.
        """

        return max(
            self.total_items - self.processed_items,
            0,
        )

    @property
    def progress_percent(self) -> float:
        """
        Return processing progress as a percentage.
        """

        if self.total_items <= 0:
            return 0.0

        return round(
            min(
                self.processed_items / self.total_items * 100,
                100.0,
            ),
            2,
        )

    @property
    def is_finished(self) -> bool:
        """
        Return True when the run reached a terminal state.
        """

        return self.status in {
            "completed",
            "partial",
            "failed",
            "cancelled",
        }

    @property
    def is_successful(self) -> bool:
        """
        Return True when processing completed without failures.
        """

        return self.status == "completed"

    @property
    def can_retry(self) -> bool:
        """
        Return whether another processing attempt is allowed.
        """

        return (
            self.status in {"failed", "partial"}
            and self.attempt_number < self.max_attempts
        )

    # ========================================================================
    # LIFECYCLE METHODS
    # ========================================================================

    def start(self) -> None:
        """
        Mark the processing run as active.
        """

        self.status = "processing"

        if self.started_at is None:
            self.started_at = utc_now()

        self.updated_at = utc_now()

    def complete(self) -> None:
        """
        Mark the run as successfully completed.
        """

        self.status = "completed"

        self.successful_items = max(
            self.successful_items,
            self.total_items,
        )

        self.partial_items = 0
        self.failed_items = 0

        self.completed_at = utc_now()

        self._calculate_duration()

    def mark_partial(
        self,
        error_summary: str | None = None,
    ) -> None:
        """
        Mark the run as partially successful.
        """

        self.status = "partial"

        if error_summary:
            self.error_summary = error_summary

        self.completed_at = utc_now()

        self._calculate_duration()

    def fail(
        self,
        error_summary: str | None = None,
    ) -> None:
        """
        Mark the entire run as failed.
        """

        self.status = "failed"

        if error_summary:
            self.error_summary = error_summary

        self.completed_at = utc_now()

        self._calculate_duration()

    def cancel(
        self,
        reason: str | None = None,
    ) -> None:
        """
        Mark the run as cancelled.
        """

        self.status = "cancelled"

        if reason:
            self.error_summary = reason

        self.completed_at = utc_now()

        self._calculate_duration()

    def increment_success(
        self,
    ) -> None:
        """
        Register one successful item.
        """

        self.successful_items += 1
        self._update_progress_status()

    def increment_partial(
        self,
    ) -> None:
        """
        Register one partially successful item.
        """

        self.partial_items += 1
        self._update_progress_status()

    def increment_failure(
        self,
        error: str | None = None,
    ) -> None:
        """
        Register one failed item.
        """

        self.failed_items += 1
        self.error_count += 1

        if error:
            self.error_summary = error[:MAX_ERROR_LENGTH]

        self._update_progress_status()

    # ========================================================================
    # RETRY
    # ========================================================================

    def prepare_retry(self) -> None:
        """
        Prepare this run for another processing attempt.

        Raises
        ------
        RuntimeError
            If retrying is no longer allowed.
        """

        if not self.can_retry:
            raise RuntimeError("Processing run cannot be retried.")

        self.attempt_number += 1
        self.status = "pending"

        self.started_at = utc_now()
        self.completed_at = None
        self.duration_ms = None

        self.successful_items = 0
        self.partial_items = 0
        self.failed_items = 0
        self.error_count = 0
        self.error_summary = None

        self.updated_at = utc_now()

    # ========================================================================
    # METADATA
    # ========================================================================

    def set_metadata(
        self,
        values: dict[str, Any],
    ) -> None:
        """
        Merge metadata into the existing metadata dictionary.
        """

        if not isinstance(values, dict):
            raise TypeError("Processing run metadata must be a dictionary.")

        current = self.metadata_json or {}

        current.update(values)

        self.metadata_json = current

        self.updated_at = utc_now()

    def get_metadata(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Read one metadata value safely.
        """

        if not self.metadata_json:
            return default

        return self.metadata_json.get(
            key,
            default,
        )

    # ========================================================================
    # INTERNAL HELPERS
    # ========================================================================

    def _calculate_duration(self) -> None:
        """
        Calculate total processing duration.
        """

        if not self.started_at or not self.completed_at:
            return

        started = self.started_at

        completed = self.completed_at

        # SQLite may return naive datetimes depending on configuration.
        if started.tzinfo is None:
            started = started.replace(
                tzinfo=timezone.utc,
            )

        if completed.tzinfo is None:
            completed = completed.replace(
                tzinfo=timezone.utc,
            )

        duration = (completed - started).total_seconds() * 1000

        self.duration_ms = max(
            int(duration),
            0,
        )

    def _update_progress_status(self) -> None:
        """
        Update run status based on current item counters.
        """

        processed = self.processed_items

        if self.total_items <= 0:
            return

        if processed < self.total_items:
            self.status = "processing"
            self.updated_at = utc_now()
            return

        self.completed_at = utc_now()

        if self.failed_items == self.total_items:
            self.status = "failed"

        elif self.failed_items > 0 or self.partial_items > 0:
            self.status = "partial"

        else:
            self.status = "completed"

        self._calculate_duration()

        self.updated_at = utc_now()


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "ProcessingRun",
    "PROCESSING_RUN_TYPES",
    "PROCESSING_STATUSES",
]
