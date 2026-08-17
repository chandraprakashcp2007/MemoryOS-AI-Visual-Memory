"""
MemoryOS - Search History Model
===============================

Premium persistent search-history model.

Responsibilities
----------------
- Store every user search
- Store normalized search queries
- Store search mode
- Store semantic / keyword configuration
- Store filters used by the search
- Store result statistics
- Store top matching memory
- Store performance metrics
- Store embedding / reranking metrics
- Store explainability information
- Support dashboard analytics
- Support future search personalization
- Support safe cleanup and retention

Architecture
------------

Frontend
    │
    ▼
SearchRequest
    │
    ▼
Search Service
    │
    ├── Semantic Search
    ├── Keyword Search
    ├── Reranking
    └── Filtering
            │
            ▼
      SearchHistory
            │
            ├── query
            ├── filters
            ├── scores
            ├── timings
            └── top result

Important
---------
Search history is an analytics/audit record.

It should NOT contain the full search result payload.
The actual memories remain in the Memory table.
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
    Float,
)
from sqlalchemy.dialects.sqlite import JSON
from sqlalchemy.orm import Mapped, mapped_column

from backend.database import Base

# ============================================================================
# CONSTANTS
# ============================================================================

MAX_QUERY_LENGTH = 1000
MAX_NORMALIZED_QUERY_LENGTH = 1000
MAX_SEARCH_MODE_LENGTH = 50
MAX_EMBEDDING_MODEL_LENGTH = 200
MAX_ERROR_LENGTH = 5000


# ============================================================================
# SEARCH MODES
# ============================================================================

SEARCH_MODES = frozenset(
    {
        "semantic",
        "keyword",
        "hybrid",
        "filtered",
        "reranked",
    }
)


# ============================================================================
# TIME HELPER
# ============================================================================


def utc_now() -> datetime:
    """
    Return the current timezone-aware UTC timestamp.
    """

    return datetime.now(timezone.utc)


# ============================================================================
# MODEL
# ============================================================================


class SearchHistory(Base):
    """
    Persistent record of a MemoryOS search operation.

    One row represents one completed or attempted search.

    Example
    -------

    User searches:

        "Where did I see my Adidas shoes?"

    The row can store:

        query = "Where did I see my Adidas shoes?"
        normalized_query = "where did i see my adidas shoes"
        search_mode = "hybrid"
        top_k = 10
        result_count = 6
        top_memory_id = 42
    """

    __tablename__ = "search_history"

    # ========================================================================
    # PRIMARY KEY
    # ========================================================================

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    # ========================================================================
    # QUERY
    # ========================================================================

    query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    normalized_query: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    # ========================================================================
    # SEARCH CONFIGURATION
    # ========================================================================

    search_mode: Mapped[str] = mapped_column(
        String(MAX_SEARCH_MODE_LENGTH),
        nullable=False,
        default="hybrid",
        server_default="hybrid",
        index=True,
    )

    top_k: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=10,
        server_default="10",
    )

    semantic_weight: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.70,
        server_default="0.70",
    )

    keyword_weight: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.30,
        server_default="0.30",
    )

    reranking_enabled: Mapped[bool] = mapped_column(
        nullable=False,
        default=True,
        server_default="1",
    )

    # ========================================================================
    # SEARCH FLAGS
    # ========================================================================

    semantic_search_used: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default="0",
    )

    keyword_search_used: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default="0",
    )

    reranking_used: Mapped[bool] = mapped_column(
        nullable=False,
        default=False,
        server_default="0",
    )

    # ========================================================================
    # FILTERS
    # ========================================================================

    filters: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # ========================================================================
    # RESULT INFORMATION
    # ========================================================================

    result_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )

    top_memory_id: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        index=True,
    )

    top_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    average_score: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
    )

    # ========================================================================
    # PERFORMANCE
    # ========================================================================

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

    database_duration_ms: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    # ========================================================================
    # EMBEDDING INFORMATION
    # ========================================================================

    query_embedding_model: Mapped[str | None] = mapped_column(
        String(MAX_EMBEDDING_MODEL_LENGTH),
        nullable=True,
    )

    query_embedding_dimension: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    # ========================================================================
    # RESULT / EXPLANATION INFORMATION
    # ========================================================================

    explanation_enabled: Mapped[bool] = mapped_column(
        nullable=False,
        default=True,
        server_default="1",
    )

    entities_enabled: Mapped[bool] = mapped_column(
        nullable=False,
        default=True,
        server_default="1",
    )

    images_enabled: Mapped[bool] = mapped_column(
        nullable=False,
        default=True,
        server_default="1",
    )

    # ========================================================================
    # EXTENSIBLE SEARCH METADATA
    # ========================================================================

    metadata_json: Mapped[dict[str, Any] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # ========================================================================
    # ERROR INFORMATION
    # ========================================================================

    success: Mapped[bool] = mapped_column(
        nullable=False,
        default=True,
        server_default="1",
    )

    error_message: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    # ========================================================================
    # TIMESTAMP
    # ========================================================================

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=utc_now,
        index=True,
    )

    # ========================================================================
    # DATABASE CONSTRAINTS
    # ========================================================================

    __table_args__ = (
        CheckConstraint(
            "length(query) >= 1",
            name="ck_search_history_query_not_empty",
        ),
        CheckConstraint(
            "length(normalized_query) >= 1",
            name="ck_search_history_normalized_query_not_empty",
        ),
        CheckConstraint(
            "top_k >= 1",
            name="ck_search_history_top_k_positive",
        ),
        CheckConstraint(
            "result_count >= 0",
            name="ck_search_history_result_count_nonnegative",
        ),
        CheckConstraint(
            "semantic_weight >= 0.0",
            name="ck_search_history_semantic_weight_nonnegative",
        ),
        CheckConstraint(
            "semantic_weight <= 1.0",
            name="ck_search_history_semantic_weight_max",
        ),
        CheckConstraint(
            "keyword_weight >= 0.0",
            name="ck_search_history_keyword_weight_nonnegative",
        ),
        CheckConstraint(
            "keyword_weight <= 1.0",
            name="ck_search_history_keyword_weight_max",
        ),
        CheckConstraint(
            """
            semantic_weight + keyword_weight <= 1.000001
            """,
            name="ck_search_history_weights_valid",
        ),
        CheckConstraint(
            "search_duration_ms IS NULL OR search_duration_ms >= 0",
            name="ck_search_history_search_duration_nonnegative",
        ),
        CheckConstraint(
            "embedding_duration_ms IS NULL OR embedding_duration_ms >= 0",
            name="ck_search_history_embedding_duration_nonnegative",
        ),
        CheckConstraint(
            "reranking_duration_ms IS NULL OR reranking_duration_ms >= 0",
            name="ck_search_history_reranking_duration_nonnegative",
        ),
        CheckConstraint(
            "database_duration_ms IS NULL OR database_duration_ms >= 0",
            name="ck_search_history_database_duration_nonnegative",
        ),
        CheckConstraint(
            """
            query_embedding_dimension IS NULL
            OR query_embedding_dimension > 0
            """,
            name="ck_search_history_embedding_dimension_positive",
        ),
        Index(
            "ix_search_history_created_mode",
            "created_at",
            "search_mode",
        ),
        Index(
            "ix_search_history_normalized_query",
            "normalized_query",
        ),
        Index(
            "ix_search_history_top_memory_created",
            "top_memory_id",
            "created_at",
        ),
        Index(
            "ix_search_history_success_created",
            "success",
            "created_at",
        ),
    )

    # ========================================================================
    # REPRESENTATION
    # ========================================================================

    def __repr__(self) -> str:
        return (
            f"<SearchHistory("
            f"id={self.id!r}, "
            f"query={self.query[:40]!r}, "
            f"mode={self.search_mode!r}, "
            f"results={self.result_count!r}"
            f")>"
        )

    # ========================================================================
    # PROPERTIES
    # ========================================================================

    @property
    def has_results(self) -> bool:
        """
        Return whether the search produced at least one result.
        """

        return self.result_count > 0

    @property
    def is_empty(self) -> bool:
        """
        Return whether the search returned no results.
        """

        return self.result_count == 0

    @property
    def total_weight(self) -> float:
        """
        Return combined semantic + keyword weight.
        """

        return round(
            self.semantic_weight + self.keyword_weight,
            4,
        )

    @property
    def is_hybrid(self) -> bool:
        """
        Return whether both semantic and keyword search were used.
        """

        return self.semantic_search_used and self.keyword_search_used

    @property
    def performance_total_ms(self) -> int:
        """
        Return the sum of known internal search timings.

        This is useful for diagnostics when the overall duration
        is available but individual stages are also recorded.
        """

        values = (
            self.embedding_duration_ms,
            self.reranking_duration_ms,
            self.database_duration_ms,
        )

        return sum(value for value in values if value is not None)

    # ========================================================================
    # FACTORY
    # ========================================================================

    @classmethod
    def create(
        cls,
        *,
        query: str,
        normalized_query: str | None = None,
        search_mode: str = "hybrid",
        top_k: int = 10,
        semantic_weight: float = 0.70,
        keyword_weight: float = 0.30,
        filters: dict[str, Any] | None = None,
        reranking_enabled: bool = True,
        explanation_enabled: bool = True,
        entities_enabled: bool = True,
        images_enabled: bool = True,
    ) -> SearchHistory:
        """
        Create a validated search-history record.

        This is the preferred factory for the search service.
        """

        clean_query = query.strip()

        if not clean_query:
            raise ValueError("Search query cannot be empty.")

        if len(clean_query) > MAX_QUERY_LENGTH:
            raise ValueError("Search query exceeds maximum length.")

        if search_mode not in SEARCH_MODES:
            raise ValueError(f"Unsupported search mode: {search_mode}")

        if top_k < 1:
            raise ValueError("top_k must be greater than zero.")

        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be between 0 and 1.")

        if not 0.0 <= keyword_weight <= 1.0:
            raise ValueError("keyword_weight must be between 0 and 1.")

        if semantic_weight + keyword_weight > 1.000001:
            raise ValueError("semantic_weight + keyword_weight cannot exceed 1.")

        normalized = (
            normalized_query.strip()
            if normalized_query
            else cls.normalize_query(clean_query)
        )

        return cls(
            query=clean_query,
            normalized_query=normalized,
            search_mode=search_mode,
            top_k=top_k,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            filters=filters,
            reranking_enabled=reranking_enabled,
            explanation_enabled=explanation_enabled,
            entities_enabled=entities_enabled,
            images_enabled=images_enabled,
        )

    # ========================================================================
    # QUERY NORMALIZATION
    # ========================================================================

    @staticmethod
    def normalize_query(
        query: str,
    ) -> str:
        """
        Perform lightweight deterministic query normalization.

        Heavy NLP normalization belongs in the search service.
        """

        return " ".join(query.strip().lower().split())

    # ========================================================================
    # RESULT RECORDING
    # ========================================================================

    def record_results(
        self,
        *,
        result_count: int,
        top_memory_id: int | None = None,
        top_score: float | None = None,
        average_score: float | None = None,
    ) -> None:
        """
        Record final search result information.
        """

        if result_count < 0:
            raise ValueError("result_count cannot be negative.")

        self.result_count = result_count
        self.top_memory_id = top_memory_id
        self.top_score = top_score
        self.average_score = average_score

    # ========================================================================
    # PERFORMANCE RECORDING
    # ========================================================================

    def record_timing(
        self,
        *,
        search_duration_ms: int | None = None,
        embedding_duration_ms: int | None = None,
        reranking_duration_ms: int | None = None,
        database_duration_ms: int | None = None,
    ) -> None:
        """
        Record search pipeline timing information.
        """

        values = {
            "search_duration_ms": search_duration_ms,
            "embedding_duration_ms": embedding_duration_ms,
            "reranking_duration_ms": reranking_duration_ms,
            "database_duration_ms": database_duration_ms,
        }

        for field_name, value in values.items():
            if value is not None and value < 0:
                raise ValueError(f"{field_name} cannot be negative.")

        if search_duration_ms is not None:
            self.search_duration_ms = search_duration_ms

        if embedding_duration_ms is not None:
            self.embedding_duration_ms = embedding_duration_ms

        if reranking_duration_ms is not None:
            self.reranking_duration_ms = reranking_duration_ms

        if database_duration_ms is not None:
            self.database_duration_ms = database_duration_ms

    # ========================================================================
    # EMBEDDING INFORMATION
    # ========================================================================

    def record_embedding(
        self,
        *,
        model: str | None,
        dimension: int | None,
        duration_ms: int | None = None,
    ) -> None:
        """
        Record query-embedding information.
        """

        if dimension is not None and dimension <= 0:
            raise ValueError("Embedding dimension must be greater than zero.")

        if duration_ms is not None and duration_ms < 0:
            raise ValueError("Embedding duration cannot be negative.")

        self.query_embedding_model = model
        self.query_embedding_dimension = dimension
        self.embedding_duration_ms = duration_ms

        self.semantic_search_used = True

    # ========================================================================
    # SEARCH ENGINE FLAGS
    # ========================================================================

    def mark_semantic_search_used(self) -> None:
        """
        Mark semantic search as used.
        """

        self.semantic_search_used = True

    def mark_keyword_search_used(self) -> None:
        """
        Mark keyword search as used.
        """

        self.keyword_search_used = True

    def mark_reranking_used(
        self,
    ) -> None:
        """
        Mark reranking as used.
        """

        self.reranking_used = True
        self.reranking_enabled = True

    # ========================================================================
    # SUCCESS / FAILURE
    # ========================================================================

    def mark_success(
        self,
    ) -> None:
        """
        Mark the search as successful.
        """

        self.success = True
        self.error_message = None

    def mark_failed(
        self,
        error_message: str,
    ) -> None:
        """
        Mark the search as failed.
        """

        self.success = False

        self.error_message = error_message[:MAX_ERROR_LENGTH]

    # ========================================================================
    # METADATA
    # ========================================================================

    def set_metadata(
        self,
        values: dict[str, Any],
    ) -> None:
        """
        Merge additional search metadata.

        Useful for:

        - client information
        - search engine version
        - reranker version
        - feature flags
        - experimental ranking information
        """

        if not isinstance(values, dict):
            raise TypeError("Search metadata must be a dictionary.")

        current = self.metadata_json or {}

        current.update(values)

        self.metadata_json = current

    def get_metadata(
        self,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Safely retrieve one metadata value.
        """

        if not self.metadata_json:
            return default

        return self.metadata_json.get(
            key,
            default,
        )


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "SearchHistory",
    "SEARCH_MODES",
]
