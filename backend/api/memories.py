"""
MemoryOS - Memories API
=======================

REST API endpoints for retrieving and managing indexed memories.

Responsibilities
----------------
- Validate memory identifiers.
- Validate pagination.
- Delegate persistence operations.
- Return stable API responses.
- Keep database/business logic outside the router.

Endpoints
---------
GET    /memories
GET    /memories/{memory_id}
DELETE /memories/{memory_id}

The router intentionally does NOT implement SQLAlchemy queries directly.
"""

from __future__ import annotations

import logging
import json
import re
import time
from pathlib import Path as FilePath
from typing import Annotated, Any, Protocol, Sequence

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
    status,
)
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    prefix="/memories",
    tags=["Memories"],
)


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_PAGE = 1
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100

MEMORY_INDEX_FILE = FilePath(__file__).resolve().parents[2] / "data" / "memory_index" / "memories.json"

MEMORY_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class MemorySummary(BaseModel):
    """
    Lightweight representation used by the memory listing endpoint.
    """

    model_config = ConfigDict(
        extra="allow",
    )

    memory_id: str

    filename: str | None = None

    original_image_url: str | None = None

    image_hash: str | None = None

    title: str | None = None

    summary: str | None = None

    category: str | None = None

    intent: str | None = None

    thumbnail_url: str | None = None

    created_at: str | None = None

    updated_at: str | None = None

    processing_status: str | None = None


class MemoryDetail(MemorySummary):
    """
    Full memory representation.

    Contains the structured information produced by the MemoryOS
    ingestion pipeline.
    """

    original_filename: str | None = None

    ocr_text: str | None = None

    gemini_analysis: Any | None = None

    entities: list[str] = Field(default_factory=list)

    semantic_text: str | None = None

    file_hash: str | None = None

    image_hash: str | None = None

    processing_error: str | None = None

    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryListResponse(BaseModel):
    """
    Paginated memory response.
    """

    success: bool

    page: int

    page_size: int

    total: int

    total_pages: int

    has_next: bool

    has_previous: bool

    memories: list[MemorySummary]

    processing_time_ms: float


class MemoryDetailResponse(BaseModel):
    """
    Single memory response.
    """

    success: bool

    memory: MemoryDetail

    processing_time_ms: float


class MemoryDeleteResponse(BaseModel):
    """
    Delete operation response.
    """

    success: bool

    memory_id: str

    deleted: bool

    message: str


# ============================================================================
# REPOSITORY CONTRACT
# ============================================================================


class MemoryRepository(Protocol):
    """
    Repository contract required by the API layer.

    The concrete SQLAlchemy implementation will satisfy this protocol.

    Keeping this contract here prevents the router from depending directly
    on a particular database implementation.
    """

    async def list_memories(
        self,
        *,
        offset: int,
        limit: int,
        category: str | None = None,
        sort: str = "newest",
    ) -> tuple[Sequence[Any], int]:
        """
        Return memories and total count.
        """
        ...

    async def get_memory(
        self,
        memory_id: str,
    ) -> Any | None:
        """
        Return one memory or None.
        """
        ...

    async def delete_memory(
        self,
        memory_id: str,
    ) -> bool:
        """
        Delete one memory.

        Returns True if something was deleted.
        """
        ...


# ============================================================================
# TEMPORARY DEPENDENCY BOUNDARY
# ============================================================================


class RepositoryUnavailableError(RuntimeError):
    """
    Raised when the database repository is not yet configured.
    """


class JsonMemoryRepository:
    """Read the existing durable memory index when no DB repository exists."""

    @staticmethod
    def _load() -> dict[str, dict[str, Any]]:
        try:
            data = json.loads(MEMORY_INDEX_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    async def list_memories(self, *, offset: int, limit: int, category: str | None = None, sort: str = "newest") -> tuple[Sequence[Any], int]:
        records = self._load()
        ordered = list(records.values())
        if category:
            expected = category.strip().casefold()
            ordered = [item for item in ordered if str(item.get("category") or "other").casefold() == expected]
        ordered.sort(key=lambda item: str(item.get("indexed_at") or item.get("created_at") or ""), reverse=sort != "oldest")
        return ordered[offset : offset + limit], len(ordered)

    async def get_memory(self, memory_id: str) -> Any | None:
        return self._load().get(memory_id)

    async def delete_memory(self, memory_id: str) -> bool:
        # Deletion remains unsupported without the authoritative DB layer.
        return False


def get_memory_repository() -> MemoryRepository:
    """
    FastAPI dependency for the memory repository.

    This is intentionally a dependency boundary.

    Once the SQLAlchemy repository exists, this function should return the
    shared repository/session-backed implementation.

    We do NOT silently create fake in-memory data here.
    """

    return JsonMemoryRepository()


# ============================================================================
# HELPERS
# ============================================================================


def _validate_memory_id(
    memory_id: str,
) -> str:
    """
    Validate and normalize a memory identifier.
    """
    normalized = memory_id.strip()

    if not normalized:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_memory_id",
                "message": "Memory ID cannot be empty.",
            },
        )

    if len(normalized) > 128:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_memory_id",
                "message": ("Memory ID cannot exceed 128 characters."),
            },
        )

    if not MEMORY_ID_PATTERN.fullmatch(normalized):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "invalid_memory_id",
                "message": "Memory ID contains invalid characters.",
            },
        )

    return normalized


def _to_string(
    value: Any,
) -> str | None:
    """
    Safely convert a value to a trimmed string.
    """
    if value is None:
        return None

    text = str(value).strip()

    return text or None


def _first(
    data: dict[str, Any],
    *keys: str,
) -> Any:
    """
    Return the first available value.
    """
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]

    return None


def _entity_list(
    value: Any,
) -> list[str]:
    """
    Normalize entities into a unique list of strings.
    """
    if value is None:
        return []

    if isinstance(value, str):
        value = [value]

    if not isinstance(
        value,
        (list, tuple, set),
    ):
        return []

    result: list[str] = []

    for item in value:
        text = _to_string(item)

        if text and text not in result:
            result.append(text)

    return result[:100]


def _object_to_dict(
    value: Any,
) -> dict[str, Any]:
    """
    Convert ORM-like objects or dictionaries into a dictionary.

    Supports:
    - dict
    - Pydantic models
    - SQLAlchemy-style objects
    """
    if isinstance(value, dict):
        return dict(value)

    model_dump = getattr(
        value,
        "model_dump",
        None,
    )

    if callable(model_dump):
        try:
            result = model_dump()

            if isinstance(result, dict):
                return result

        except Exception:
            logger.debug(
                "model_dump() failed while normalizing memory.",
                exc_info=True,
            )

    try:
        return dict(vars(value))
    except (TypeError, ValueError):
        return {}


def _normalize_memory_summary(
    memory: Any,
) -> MemorySummary | None:
    """
    Convert a database/domain object to the API summary model.
    """
    data = _object_to_dict(memory)

    memory_id = _first(
        data,
        "memory_id",
        "id",
    )

    memory_id = _to_string(memory_id)

    if not memory_id:
        return None

    return MemorySummary(
        memory_id=memory_id,
        filename=_to_string(_first(data, "filename", "original_filename", "source_name")),
        original_image_url=_to_string(_first(data, "original_image_url", "image_url")),
        image_hash=_to_string(_first(data, "image_hash", "sha256", "file_hash")),
        title=_to_string(
            _first(
                data,
                "title",
                "name",
            )
        ),
        summary=_to_string(
            _first(
                data,
                "summary",
                "description",
            )
        ),
        category=_to_string(
            _first(
                data,
                "category",
            )
        ),
        intent=_to_string(
            _first(
                data,
                "intent",
            )
        ),
        thumbnail_url=_to_string(
            _first(
                data,
                "thumbnail_url",
                "thumbnail_path",
            )
        ),
        created_at=_to_string(
            _first(
                data,
                "created_at",
            )
        ),
        updated_at=_to_string(
            _first(
                data,
                "updated_at",
            )
        ),
        processing_status=_to_string(
            _first(
                data,
                "processing_status",
                "status",
            )
        ),
    )


def _normalize_memory_detail(
    memory: Any,
) -> MemoryDetail | None:
    """
    Convert a database/domain object to the complete API model.
    """
    data = _object_to_dict(memory)

    memory_id = _to_string(
        _first(
            data,
            "memory_id",
            "id",
        )
    )

    if not memory_id:
        return None

    metadata = _first(
        data,
        "metadata",
        "extra_metadata",
    )

    if not isinstance(metadata, dict):
        metadata = {}

    return MemoryDetail(
        memory_id=memory_id,
        title=_to_string(
            _first(
                data,
                "title",
                "name",
            )
        ),
        summary=_to_string(
            _first(
                data,
                "summary",
                "description",
            )
        ),
        category=_to_string(
            _first(
                data,
                "category",
            )
        ),
        intent=_to_string(
            _first(
                data,
                "intent",
            )
        ),
        thumbnail_url=_to_string(
            _first(
                data,
                "thumbnail_url",
                "thumbnail_path",
            )
        ),
        created_at=_to_string(
            _first(
                data,
                "created_at",
            )
        ),
        updated_at=_to_string(
            _first(
                data,
                "updated_at",
            )
        ),
        processing_status=_to_string(
            _first(
                data,
                "processing_status",
                "status",
            )
        ),
        original_filename=_to_string(
            _first(
                data,
                "original_filename",
                "filename",
            )
        ),
        ocr_text=_to_string(
            _first(
                data,
                "ocr_text",
            )
        ),
        gemini_analysis=_first(
            data,
            "gemini_analysis",
            "ai_analysis",
        ),
        entities=_entity_list(
            _first(
                data,
                "entities",
                "entity_names",
            )
        ),
        semantic_text=_to_string(
            _first(
                data,
                "semantic_text",
            )
        ),
        file_hash=_to_string(
            _first(
                data,
                "file_hash",
            )
        ),
        image_hash=_to_string(
            _first(
                data,
                "image_hash",
            )
        ),
        processing_error=_to_string(
            _first(
                data,
                "processing_error",
            )
        ),
        metadata=metadata,
    )


# ============================================================================
# LIST MEMORIES
# ============================================================================


@router.get(
    "",
    response_model=MemoryListResponse,
    summary="List stored memories",
    description=("Return indexed MemoryOS memories using pagination."),
)
async def list_memories(
    page: Annotated[
        int,
        Query(
            ge=DEFAULT_PAGE,
            description="1-based page number.",
        ),
    ] = DEFAULT_PAGE,
    page_size: Annotated[
        int,
        Query(
            ge=1,
            le=MAX_PAGE_SIZE,
            description="Number of memories per page.",
        ),
    ] = DEFAULT_PAGE_SIZE,
    limit: Annotated[
        int | None,
        Query(ge=1, le=MAX_PAGE_SIZE, description="Compatibility alias for page_size."),
    ] = None,
    category: Annotated[str | None, Query(max_length=100, description="Optional AI-derived category filter.")] = None,
    sort: Annotated[str, Query(pattern="^(newest|oldest)$", description="Chronological sort order.")] = "newest",
    repository: Annotated[
        MemoryRepository,
        Depends(get_memory_repository),
    ] = None,  # type: ignore[assignment]
) -> MemoryListResponse:
    """
    List stored memories.

    The API uses offset/limit internally while exposing page/page_size
    externally.
    """
    started_at = time.perf_counter()

    if limit is not None:
        page_size = limit

    offset = (page - 1) * page_size

    try:
        rows, total = await repository.list_memories(
            offset=offset,
            limit=page_size,
            category=category,
            sort=sort,
        )

    except RepositoryUnavailableError as exc:
        logger.error("Memory repository unavailable.")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "database_unavailable",
                "message": ("Memory storage is not configured."),
            },
        ) from exc

    except Exception as exc:
        logger.exception("Failed to list memories.")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "memory_list_failed",
                "message": ("Unable to retrieve memories."),
            },
        ) from exc

    normalized: list[MemorySummary] = []

    for row in rows:
        memory = _normalize_memory_summary(row)

        if memory is not None:
            normalized.append(memory)

    total_pages = (total + page_size - 1) // page_size if total > 0 else 0

    elapsed_ms = (time.perf_counter() - started_at) * 1000

    return MemoryListResponse(
        success=True,
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_previous=page > 1 and total > 0,
        memories=normalized,
        processing_time_ms=round(
            elapsed_ms,
            2,
        ),
    )


# ============================================================================
# GET ONE MEMORY
# ============================================================================


@router.get(
    "/{memory_id}",
    response_model=MemoryDetailResponse,
    summary="Get one memory",
    description=("Return the complete structured representation of a memory."),
)
async def get_memory(
    memory_id: Annotated[
        str,
        Path(
            min_length=1,
            max_length=128,
            description="Unique MemoryOS memory identifier.",
        ),
    ],
    repository: Annotated[
        MemoryRepository,
        Depends(get_memory_repository),
    ] = None,  # type: ignore[assignment]
) -> MemoryDetailResponse:
    """
    Retrieve a single memory by ID.
    """
    started_at = time.perf_counter()

    memory_id = _validate_memory_id(memory_id)

    try:
        memory = await repository.get_memory(memory_id)

    except RepositoryUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "database_unavailable",
                "message": ("Memory storage is not configured."),
            },
        ) from exc

    except Exception as exc:
        logger.exception(
            "Failed to retrieve memory %s.",
            memory_id,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "memory_retrieval_failed",
                "message": ("Unable to retrieve the requested memory."),
            },
        ) from exc

    if memory is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "memory_not_found",
                "message": (f"Memory '{memory_id}' was not found."),
                "memory_id": memory_id,
            },
        )

    normalized = _normalize_memory_detail(memory)

    if normalized is None:
        logger.error(
            "Memory repository returned invalid memory %s.",
            memory_id,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "invalid_memory_data",
                "message": ("Stored memory data is invalid."),
            },
        )

    elapsed_ms = (time.perf_counter() - started_at) * 1000

    return MemoryDetailResponse(
        success=True,
        memory=normalized,
        processing_time_ms=round(
            elapsed_ms,
            2,
        ),
    )


# ============================================================================
# DELETE MEMORY
# ============================================================================


@router.delete(
    "/{memory_id}",
    response_model=MemoryDeleteResponse,
    summary="Delete one memory",
    description=("Delete a memory from persistent storage."),
)
async def delete_memory(
    memory_id: Annotated[
        str,
        Path(
            min_length=1,
            max_length=128,
            description="Unique MemoryOS memory identifier.",
        ),
    ],
    repository: Annotated[
        MemoryRepository,
        Depends(get_memory_repository),
    ] = None,  # type: ignore[assignment]
) -> MemoryDeleteResponse:
    """
    Delete a memory.

    Important:
    The repository layer is responsible for coordinating persistent
    metadata deletion. Once the complete vector lifecycle is connected,
    deletion must also remove or rebuild the corresponding FAISS entry.
    """
    memory_id = _validate_memory_id(memory_id)

    try:
        deleted = await repository.delete_memory(memory_id)

    except RepositoryUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "database_unavailable",
                "message": ("Memory storage is not configured."),
            },
        ) from exc

    except Exception as exc:
        logger.exception(
            "Failed to delete memory %s.",
            memory_id,
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "memory_delete_failed",
                "message": ("Unable to delete the requested memory."),
            },
        ) from exc

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "memory_not_found",
                "message": (f"Memory '{memory_id}' was not found."),
                "memory_id": memory_id,
            },
        )

    return MemoryDeleteResponse(
        success=True,
        memory_id=memory_id,
        deleted=True,
        message="Memory deleted successfully.",
    )


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "router",
    "MemoryRepository",
    "MemorySummary",
    "MemoryDetail",
    "MemoryListResponse",
    "MemoryDetailResponse",
    "MemoryDeleteResponse",
    "get_memory_repository",
]
