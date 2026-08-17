"""
MemoryOS - Statistics API
=========================

Provides aggregated system statistics for the MemoryOS dashboard.

Responsibilities
----------------
- Expose system-level memory statistics.
- Validate/normalize repository output.
- Keep database logic outside the API layer.
- Return stable API contracts.
- Gracefully handle unavailable statistics.

The router does NOT calculate statistics from individual records.
That responsibility belongs to the repository/service layer.
"""

from __future__ import annotations

import logging
import json
import time
from pathlib import Path
from typing import Annotated, Any, Protocol

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    prefix="/stats",
    tags=["Statistics"],
)


# ============================================================================
# CONSTANTS
# ============================================================================

MAX_TOP_CATEGORIES = 20
MAX_RECENT_SEARCHES = 10
MEMORY_INDEX_FILE = Path(__file__).resolve().parents[2] / "data" / "memory_index" / "memories.json"
DATA_DIR = MEMORY_INDEX_FILE.parents[1]


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class CategoryStat(BaseModel):
    """Statistics for one memory category."""

    model_config = ConfigDict(
        extra="allow",
    )

    category: str

    count: int = Field(ge=0)


class SearchStat(BaseModel):
    """Aggregated search statistics."""

    total_searches: int = Field(
        default=0,
        ge=0,
    )

    successful_searches: int = Field(
        default=0,
        ge=0,
    )

    empty_searches: int = Field(
        default=0,
        ge=0,
    )

    average_processing_time_ms: float = Field(
        default=0.0,
        ge=0.0,
    )


class StorageStat(BaseModel):
    """Storage-related statistics."""

    upload_bytes: int = Field(
        default=0,
        ge=0,
    )

    thumbnail_bytes: int = Field(
        default=0,
        ge=0,
    )

    database_bytes: int = Field(
        default=0,
        ge=0,
    )

    vector_index_bytes: int = Field(
        default=0,
        ge=0,
    )

    total_bytes: int = Field(
        default=0,
        ge=0,
    )


class MemoryStats(BaseModel):
    """Core MemoryOS statistics."""

    total_memories: int = Field(
        default=0,
        ge=0,
    )

    processed_memories: int = Field(
        default=0,
        ge=0,
    )

    processing_memories: int = Field(
        default=0,
        ge=0,
    )

    failed_memories: int = Field(
        default=0,
        ge=0,
    )

    indexed_memories: int = Field(
        default=0,
        ge=0,
    )

    duplicate_memories: int = Field(
        default=0,
        ge=0,
    )


class StatsResponse(BaseModel):
    """
    Complete dashboard statistics response.
    """

    success: bool

    generated_at: str

    memories: MemoryStats

    searches: SearchStat

    categories: list[CategoryStat]

    storage: StorageStat

    processing_success_rate: float = Field(
        ge=0.0,
        le=100.0,
    )

    indexing_rate: float = Field(
        ge=0.0,
        le=100.0,
    )

    degraded: bool = False

    message: str | None = None

    processing_time_ms: float = Field(ge=0.0)


# ============================================================================
# REPOSITORY CONTRACT
# ============================================================================


class StatsRepository(Protocol):
    """
    Contract required by the statistics API.

    The real SQLAlchemy repository will implement this interface.
    """

    async def get_stats(self) -> dict[str, Any]:
        """
        Return aggregated MemoryOS statistics.
        """
        ...


class StatsRepositoryUnavailableError(RuntimeError):
    """Raised when statistics persistence is unavailable."""


class JsonStatsRepository:
    """Statistics derived from the durable JSON memory index.

    The upload and backfill pipelines already use this index as their
    authoritative compatibility store.  Reading it here keeps the dashboard
    count consistent with ``/memories`` without inventing demo values.
    """

    @staticmethod
    def _load_memories() -> dict[str, dict[str, Any]]:
        try:
            payload = json.loads(MEMORY_INDEX_FILE.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    async def get_stats(self) -> dict[str, Any]:
        memories = self._load_memories()
        records = [item for item in memories.values() if isinstance(item, dict)]
        categories: dict[str, int] = {}
        indexed = 0
        failed = 0
        processing = 0

        for record in records:
            category = str(record.get("category") or "other").strip() or "other"
            categories[category] = categories.get(category, 0) + 1
            status = str(record.get("status") or record.get("processing_status") or "").lower()
            if record.get("indexed") is True or status in {"processed", "completed", "partial"}:
                indexed += 1
            if "fail" in status:
                failed += 1
            elif status in {"processing", "uploaded", "pending"}:
                processing += 1

        def directory_size(path: Path) -> int:
            try:
                return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())
            except OSError:
                return 0

        return {
            "memories": {
                "total_memories": len(records),
                "processed_memories": max(0, len(records) - failed - processing),
                "processing_memories": processing,
                "failed_memories": failed,
                "indexed_memories": indexed,
                "duplicate_memories": 0,
            },
            "categories": [{"category": key, "count": value} for key, value in categories.items()],
            "storage": {
                "upload_bytes": directory_size(DATA_DIR / "uploads"),
                "thumbnail_bytes": directory_size(DATA_DIR / "thumbnails"),
                "database_bytes": directory_size(DATA_DIR / "database"),
                "vector_index_bytes": directory_size(DATA_DIR / "vector_store"),
            },
            "degraded": False,
        }


def get_stats_repository() -> StatsRepository:
    """
    Dependency boundary for the statistics repository.

    The actual database implementation will be connected here once
    the SQLAlchemy repository layer is finalized.
    """

    return JsonStatsRepository()


# ============================================================================
# NORMALIZATION HELPERS
# ============================================================================


def _safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """Safely convert a value to a non-negative integer."""
    if value is None:
        return default

    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return default


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """Safely convert a value to a non-negative float."""
    if value is None:
        return default

    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


def _safe_percentage(
    value: Any,
) -> float:
    """
    Normalize a percentage to the range 0-100.
    """
    return min(
        100.0,
        max(
            0.0,
            _safe_float(value),
        ),
    )


def _normalize_categories(
    value: Any,
) -> list[CategoryStat]:
    """
    Normalize category statistics.
    """
    if not isinstance(
        value,
        (list, tuple),
    ):
        return []

    categories: list[CategoryStat] = []

    for item in value:
        if isinstance(item, dict):
            category = item.get("category") or item.get("name")

            count = item.get(
                "count",
                0,
            )

        elif isinstance(item, (list, tuple)):
            if len(item) < 2:
                continue

            category = item[0]
            count = item[1]

        else:
            continue

        if category is None:
            continue

        category_text = str(category).strip()

        if not category_text:
            continue

        categories.append(
            CategoryStat(
                category=category_text[:100],
                count=_safe_int(count),
            )
        )

    categories.sort(
        key=lambda item: item.count,
        reverse=True,
    )

    return categories[:MAX_TOP_CATEGORIES]


def _normalize_memory_stats(
    data: dict[str, Any],
) -> MemoryStats:
    """
    Normalize memory statistics from repository output.
    """
    return MemoryStats(
        total_memories=_safe_int(
            data.get(
                "total_memories",
                data.get("total", 0),
            )
        ),
        processed_memories=_safe_int(
            data.get(
                "processed_memories",
                data.get("processed", 0),
            )
        ),
        processing_memories=_safe_int(
            data.get(
                "processing_memories",
                data.get("processing", 0),
            )
        ),
        failed_memories=_safe_int(
            data.get(
                "failed_memories",
                data.get("failed", 0),
            )
        ),
        indexed_memories=_safe_int(
            data.get(
                "indexed_memories",
                data.get("indexed", 0),
            )
        ),
        duplicate_memories=_safe_int(
            data.get(
                "duplicate_memories",
                data.get("duplicates", 0),
            )
        ),
    )


def _normalize_search_stats(
    data: dict[str, Any],
) -> SearchStat:
    """
    Normalize search statistics.
    """
    return SearchStat(
        total_searches=_safe_int(
            data.get(
                "total_searches",
                data.get("searches", 0),
            )
        ),
        successful_searches=_safe_int(
            data.get(
                "successful_searches",
                data.get("successful", 0),
            )
        ),
        empty_searches=_safe_int(
            data.get(
                "empty_searches",
                data.get("empty", 0),
            )
        ),
        average_processing_time_ms=_safe_float(
            data.get(
                "average_processing_time_ms",
                data.get(
                    "average_latency_ms",
                    0.0,
                ),
            )
        ),
    )


def _normalize_storage_stats(
    data: dict[str, Any],
) -> StorageStat:
    """
    Normalize storage statistics.

    If total_bytes is not provided, calculate it from the individual
    components.
    """
    upload_bytes = _safe_int(data.get("upload_bytes"))

    thumbnail_bytes = _safe_int(data.get("thumbnail_bytes"))

    database_bytes = _safe_int(data.get("database_bytes"))

    vector_index_bytes = _safe_int(data.get("vector_index_bytes"))

    supplied_total = data.get("total_bytes")

    if supplied_total is None:
        total_bytes = (
            upload_bytes + thumbnail_bytes + database_bytes + vector_index_bytes
        )
    else:
        total_bytes = _safe_int(supplied_total)

    return StorageStat(
        upload_bytes=upload_bytes,
        thumbnail_bytes=thumbnail_bytes,
        database_bytes=database_bytes,
        vector_index_bytes=vector_index_bytes,
        total_bytes=total_bytes,
    )


def _calculate_success_rate(
    memories: MemoryStats,
) -> float:
    """
    Calculate processing success percentage.
    """
    completed = memories.processed_memories + memories.failed_memories

    if completed <= 0:
        return 0.0

    return min(
        100.0,
        max(
            0.0,
            (memories.processed_memories / completed) * 100.0,
        ),
    )


def _calculate_indexing_rate(
    memories: MemoryStats,
) -> float:
    """
    Calculate percentage of memories represented in the vector index.
    """
    if memories.total_memories <= 0:
        return 0.0

    return min(
        100.0,
        max(
            0.0,
            (memories.indexed_memories / memories.total_memories) * 100.0,
        ),
    )


def _iso_timestamp() -> str:
    """
    Generate a UTC ISO-8601 timestamp.

    Uses the standard library only.
    """
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


# ============================================================================
# STATS ENDPOINT
# ============================================================================


@router.get(
    "",
    response_model=StatsResponse,
    status_code=status.HTTP_200_OK,
    summary="Get MemoryOS statistics",
    description=("Return aggregated memory, search, category and storage statistics."),
)
async def get_stats(
    repository: Annotated[
        StatsRepository,
        Depends(get_stats_repository),
    ] = None,  # type: ignore[assignment]
) -> StatsResponse:
    """
    Return dashboard statistics.

    The route performs only normalization and presentation logic.
    """

    started_at = time.perf_counter()

    try:
        raw_stats = await repository.get_stats()

    except StatsRepositoryUnavailableError as exc:
        logger.error("Statistics repository is unavailable.")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "stats_unavailable",
                "message": ("Statistics storage is not configured."),
            },
        ) from exc

    except Exception as exc:
        logger.exception("Failed to generate MemoryOS statistics.")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "stats_generation_failed",
                "message": ("Unable to generate system statistics."),
            },
        ) from exc

    if not isinstance(raw_stats, dict):
        logger.error("Statistics repository returned invalid data.")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "invalid_stats_data",
                "message": ("Statistics data is invalid."),
            },
        )

    memories_data = raw_stats.get(
        "memories",
        raw_stats,
    )

    searches_data = raw_stats.get(
        "searches",
        {},
    )

    storage_data = raw_stats.get(
        "storage",
        {},
    )

    categories_data = raw_stats.get(
        "categories",
        [],
    )

    if not isinstance(
        memories_data,
        dict,
    ):
        memories_data = {}

    if not isinstance(
        searches_data,
        dict,
    ):
        searches_data = {}

    if not isinstance(
        storage_data,
        dict,
    ):
        storage_data = {}

    memories = _normalize_memory_stats(memories_data)

    searches = _normalize_search_stats(searches_data)

    storage = _normalize_storage_stats(storage_data)

    categories = _normalize_categories(categories_data)

    processing_success_rate = _calculate_success_rate(memories)

    indexing_rate = _calculate_indexing_rate(memories)

    degraded = bool(
        raw_stats.get(
            "degraded",
            False,
        )
    )

    message = raw_stats.get("message")

    elapsed_ms = (time.perf_counter() - started_at) * 1000

    return StatsResponse(
        success=True,
        generated_at=_iso_timestamp(),
        memories=memories,
        searches=searches,
        categories=categories,
        storage=storage,
        processing_success_rate=round(
            processing_success_rate,
            2,
        ),
        indexing_rate=round(
            indexing_rate,
            2,
        ),
        degraded=degraded,
        message=(str(message) if message is not None else None),
        processing_time_ms=round(
            elapsed_ms,
            2,
        ),
    )


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "router",
    "CategoryStat",
    "SearchStat",
    "StorageStat",
    "MemoryStats",
    "StatsResponse",
    "StatsRepository",
    "JsonStatsRepository",
    "get_stats_repository",
]
