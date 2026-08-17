"""
MemoryOS - API Schemas

Central Pydantic schema definitions for the MemoryOS backend.

Responsibilities:
    - Validate API input
    - Standardize API output
    - Protect internal database models
    - Define upload responses
    - Define OCR responses
    - Define AI analysis responses
    - Define memory responses
    - Define search responses
    - Define statistics responses
    - Define error responses

Important architecture rule:

    SQLAlchemy Models
            ↓
       Service Layer
            ↓
      Pydantic Schemas
            ↓
         FastAPI
            ↓
        Frontend

The frontend should never depend directly on SQLAlchemy models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Generic, Literal, TypeVar

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

# ============================================================================
# GENERIC TYPES
# ============================================================================

T = TypeVar("T")


# ============================================================================
# BASE CONFIGURATION
# ============================================================================


class MemoryOSBaseSchema(BaseModel):
    """
    Base schema shared by all MemoryOS API schemas.
    """

    model_config = ConfigDict(
        from_attributes=True,
        populate_by_name=True,
        str_strip_whitespace=True,
        extra="ignore",
    )


# ============================================================================
# ENUM-LIKE LITERALS
# ============================================================================

ProcessingStatusValue = Literal[
    "pending",
    "processing",
    "completed",
    "partial",
    "failed",
]

MemoryCategoryValue = Literal[
    "product",
    "receipt",
    "document",
    "screenshot",
    "ticket",
    "note",
    "social",
    "travel",
    "food",
    "finance",
    "shopping",
    "education",
    "work",
    "personal",
    "other",
    "unknown",
]

MemoryIntentValue = Literal[
    "reference",
    "purchase",
    "payment",
    "travel",
    "communication",
    "learning",
    "task",
    "information",
    "reminder",
    "tracking",
    "unknown",
]

EntityTypeValue = Literal[
    "person",
    "organization",
    "product",
    "brand",
    "location",
    "date",
    "time",
    "price",
    "currency",
    "email",
    "phone",
    "url",
    "order_id",
    "transaction_id",
    "invoice_id",
    "event",
    "category",
    "other",
]


# ============================================================================
# COMMON RESPONSE SCHEMAS
# ============================================================================


class MessageResponse(MemoryOSBaseSchema):
    """
    Simple success/info response.
    """

    success: bool = True

    message: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )


class HealthResponse(MemoryOSBaseSchema):
    """
    Backend health response.
    """

    status: Literal[
        "healthy",
        "degraded",
        "unhealthy",
    ]

    service: str = "MemoryOS"

    version: str

    database: bool

    vector_index: bool

    ocr_available: bool

    vision_available: bool

    timestamp: datetime


class ErrorDetail(MemoryOSBaseSchema):
    """
    Structured error information.
    """

    code: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    field: str | None = Field(
        default=None,
        max_length=200,
    )

    details: dict[str, Any] | None = None


class ErrorResponse(MemoryOSBaseSchema):
    """
    Standard API error response.

    Example:

        {
            "success": false,
            "error": {
                "code": "INVALID_IMAGE",
                "message": "Unsupported image format"
            }
        }
    """

    success: bool = False

    error: ErrorDetail


# ============================================================================
# PAGINATION
# ============================================================================


class PaginationRequest(MemoryOSBaseSchema):
    """
    Generic pagination parameters.
    """

    page: int = Field(
        default=1,
        ge=1,
        le=100000,
    )

    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
    )


class PaginationMeta(MemoryOSBaseSchema):
    """
    Pagination information returned to the frontend.
    """

    page: int = Field(
        ...,
        ge=1,
    )

    page_size: int = Field(
        ...,
        ge=1,
    )

    total_items: int = Field(
        ...,
        ge=0,
    )

    total_pages: int = Field(
        ...,
        ge=0,
    )

    has_next: bool

    has_previous: bool


# ============================================================================
# IMAGE SCHEMAS
# ============================================================================


class ImageMetadataResponse(MemoryOSBaseSchema):
    """
    Technical metadata of an image.
    """

    width: int = Field(
        ...,
        gt=0,
    )

    height: int = Field(
        ...,
        gt=0,
    )

    mode: str

    format: str

    mime_type: str

    has_alpha: bool

    file_size_bytes: int = Field(
        ...,
        ge=0,
    )


class ImageAssetResponse(MemoryOSBaseSchema):
    """
    Public representation of a stored image.
    """

    id: int

    memory_id: int

    original_filename: str

    stored_filename: str

    thumbnail_url: str | None = None

    mime_type: str

    extension: str

    file_size_bytes: int = Field(
        ...,
        ge=0,
    )

    width: int = Field(
        ...,
        gt=0,
    )

    height: int = Field(
        ...,
        gt=0,
    )

    color_mode: str | None = None

    image_format: str | None = None

    file_hash: str

    image_hash: str | None = None

    ocr_processed: bool

    ocr_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    processing_status: ProcessingStatusValue

    sequence_number: int = Field(
        default=0,
        ge=0,
    )

    is_primary: bool

    created_at: datetime

    processed_at: datetime | None = None


# ============================================================================
# UPLOAD SCHEMAS
# ============================================================================


class UploadOptions(MemoryOSBaseSchema):
    """
    Optional upload processing configuration.
    """

    run_ocr: bool = True

    run_vision: bool = True

    create_thumbnail: bool = True

    generate_embedding: bool = True

    detect_duplicates: bool = True

    continue_on_error: bool = True


class UploadItemResponse(MemoryOSBaseSchema):
    """
    Result for one uploaded image.

    Batch uploads should isolate errors per image.
    """

    filename: str

    success: bool

    memory_id: int | None = None

    image_id: int | None = None

    status: ProcessingStatusValue

    message: str | None = None

    error: ErrorDetail | None = None


class BatchUploadResponse(MemoryOSBaseSchema):
    """
    Complete batch upload result.
    """

    success: bool

    run_id: str

    total_files: int = Field(
        ...,
        ge=0,
    )

    successful_files: int = Field(
        ...,
        ge=0,
    )

    partial_files: int = Field(
        ...,
        ge=0,
    )

    failed_files: int = Field(
        ...,
        ge=0,
    )

    items: list[UploadItemResponse]

    duration_ms: int | None = Field(
        default=None,
        ge=0,
    )


# ============================================================================
# OCR SCHEMAS
# ============================================================================


class OCRRequest(MemoryOSBaseSchema):
    """
    OCR processing configuration.
    """

    language: str = Field(
        default="eng",
        min_length=2,
        max_length=50,
    )

    preprocess: bool = True

    detect_orientation: bool = True


class OCRResult(MemoryOSBaseSchema):
    """
    OCR output.
    """

    text: str = ""

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    language: str = "eng"

    word_count: int = Field(
        default=0,
        ge=0,
    )

    character_count: int = Field(
        default=0,
        ge=0,
    )

    processing_time_ms: int | None = Field(
        default=None,
        ge=0,
    )

    success: bool = True

    error: str | None = None


# ============================================================================
# AI ENTITY SCHEMAS
# ============================================================================


class EntitySchema(MemoryOSBaseSchema):
    """
    Public entity representation.
    """

    id: int | None = None

    entity_type: EntityTypeValue

    value: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )

    normalized_value: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )

    context: str | None = None

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    source: str = Field(
        default="ai",
        min_length=1,
        max_length=50,
    )

    source_text: str | None = None


class EntityCreate(EntitySchema):
    """
    Entity creation schema.
    """

    id: None = None


# ============================================================================
# AI ANALYSIS
# ============================================================================


class AIClassification(MemoryOSBaseSchema):
    """
    AI classification output.
    """

    category: MemoryCategoryValue

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )


class AIIntent(MemoryOSBaseSchema):
    """
    AI intent detection output.
    """

    intent: MemoryIntentValue

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )


class AIAnalysisRequest(MemoryOSBaseSchema):
    """
    Configuration for AI analysis.
    """

    use_vision: bool = True

    use_ocr_context: bool = True

    extract_entities: bool = True

    detect_intent: bool = True

    generate_summary: bool = True


class AIAnalysisResponse(MemoryOSBaseSchema):
    """
    Unified AI analysis result.
    """

    title: str = Field(
        default="Untitled Memory",
        max_length=160,
    )

    summary: str = ""

    normalized_text: str = ""

    classification: AIClassification

    intent: AIIntent

    entities: list[EntitySchema] = Field(
        default_factory=list,
    )

    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    explanation: str = ""

    raw_analysis: dict[str, Any] | None = None

    model: str | None = None

    success: bool = True

    fallback_used: bool = False

    processing_time_ms: int | None = Field(
        default=None,
        ge=0,
    )


# ============================================================================
# MEMORY SCHEMAS
# ============================================================================


class MemorySummaryResponse(MemoryOSBaseSchema):
    """
    Lightweight memory representation.

    Used for:
        - lists
        - search results
        - dashboard
        - autocomplete
    """

    id: int

    title: str

    summary: str

    category: MemoryCategoryValue

    intent: MemoryIntentValue

    confidence_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    processing_status: ProcessingStatusValue

    indexed: bool

    primary_thumbnail_url: str | None = None

    created_at: datetime

    updated_at: datetime


class MemoryDetailResponse(MemorySummaryResponse):
    """
    Full memory representation.
    """

    normalized_text: str

    raw_ocr_text: str

    ai_explanation: str

    why_this_matched_template: str

    classification_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    extraction_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    embedding_model: str | None = None

    embedding_dimension: int | None = None

    faiss_vector_id: int | None = None

    images: list[ImageAssetResponse] = Field(
        default_factory=list,
    )

    entities: list[EntitySchema] = Field(
        default_factory=list,
    )

    ai_analysis: dict[str, Any] | None = None

    extracted_metadata: dict[str, Any] | None = None

    processing_error: str | None = None

    processing_duration_ms: int | None = None

    indexed_at: datetime | None = None


# ============================================================================
# MEMORY LIST
# ============================================================================


class MemoryListResponse(MemoryOSBaseSchema):
    """
    Paginated memory list.
    """

    success: bool = True

    items: list[MemorySummaryResponse]

    pagination: PaginationMeta


# ============================================================================
# MEMORY FILTERS
# ============================================================================


class MemoryFilters(MemoryOSBaseSchema):
    """
    Filters used by memory listing/search.
    """

    category: MemoryCategoryValue | None = None

    intent: MemoryIntentValue | None = None

    processing_status: ProcessingStatusValue | None = None

    indexed: bool | None = None

    created_after: datetime | None = None

    created_before: datetime | None = None

    min_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    max_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
    )

    @field_validator(
        "created_before",
    )
    @classmethod
    def validate_date_range(
        cls,
        value: datetime | None,
    ) -> datetime | None:
        return value


# ============================================================================
# SEARCH REQUEST
# ============================================================================


class SearchRequest(MemoryOSBaseSchema):
    """
    Main semantic search request.

    Example:

        {
            "query": "Where did I see my Adidas shoes?",
            "top_k": 10,
            "hybrid": true
        }
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
    )

    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
    )

    hybrid: bool = True

    semantic_weight: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
    )

    keyword_weight: float = Field(
        default=0.30,
        ge=0.0,
        le=1.0,
    )

    rerank: bool = True

    filters: MemoryFilters = Field(
        default_factory=MemoryFilters,
    )

    include_explanation: bool = True

    include_entities: bool = True

    include_images: bool = True

    @field_validator(
        "query",
    )
    @classmethod
    def validate_query(
        cls,
        value: str,
    ) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Search query cannot be empty.")

        return value


# ============================================================================
# SEARCH RESULT
# ============================================================================


class SearchScoreBreakdown(MemoryOSBaseSchema):
    """
    Explainable score components.

    This is critical for the MemoryOS demo.
    """

    semantic_score: float = Field(
        default=0.0,
        ge=0.0,
    )

    keyword_score: float = Field(
        default=0.0,
        ge=0.0,
    )

    rerank_score: float = Field(
        default=0.0,
        ge=0.0,
    )

    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
    )

    final_score: float = Field(
        default=0.0,
        ge=0.0,
    )


class MatchExplanation(MemoryOSBaseSchema):
    """
    Human-readable explanation for why a memory matched.
    """

    summary: str

    matched_terms: list[str] = Field(
        default_factory=list,
    )

    matched_entities: list[str] = Field(
        default_factory=list,
    )

    category_match: bool = False

    intent_match: bool = False

    explanation: str


class SearchResult(MemoryOSBaseSchema):
    """
    Individual search result.
    """

    rank: int = Field(
        ...,
        ge=1,
    )

    memory: MemorySummaryResponse

    score: float = Field(
        ...,
        ge=0.0,
    )

    scores: SearchScoreBreakdown

    explanation: MatchExplanation | None = None

    entities: list[EntitySchema] = Field(
        default_factory=list,
    )

    image_url: str | None = None

    preview_url: str | None = None

    thumbnail_url: str | None = None

    original_image_url: str | None = None


class SearchResponse(MemoryOSBaseSchema):
    """
    Complete search response.
    """

    success: bool = True

    query: str

    normalized_query: str

    results: list[SearchResult]

    total_results: int = Field(
        ...,
        ge=0,
    )

    search_mode: str = "hybrid"

    semantic_search_used: bool

    keyword_search_used: bool

    reranking_used: bool

    search_duration_ms: int | None = Field(
        default=None,
        ge=0,
    )

    embedding_duration_ms: int | None = Field(
        default=None,
        ge=0,
    )

    reranking_duration_ms: int | None = Field(
        default=None,
        ge=0,
    )

    query_embedding_model: str | None = None


# ============================================================================
# SEARCH HISTORY
# ============================================================================


class SearchHistoryResponse(MemoryOSBaseSchema):
    """
    Search history item.
    """

    id: int

    query: str

    normalized_query: str

    search_mode: str

    filters: dict[str, Any] | None = None

    top_k: int

    result_count: int

    top_memory_id: int | None = None

    search_duration_ms: int | None = None

    embedding_duration_ms: int | None = None

    reranking_duration_ms: int | None = None

    created_at: datetime


class SearchHistoryListResponse(MemoryOSBaseSchema):
    """
    Paginated search history.
    """

    success: bool = True

    items: list[SearchHistoryResponse]

    pagination: PaginationMeta


# ============================================================================
# STATISTICS
# ============================================================================


class CategoryStatistic(MemoryOSBaseSchema):
    """
    Number of memories grouped by category.
    """

    category: str

    count: int = Field(
        ...,
        ge=0,
    )

    percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )


class IntentStatistic(MemoryOSBaseSchema):
    """
    Number of memories grouped by intent.
    """

    intent: str

    count: int = Field(
        ...,
        ge=0,
    )

    percentage: float = Field(
        default=0.0,
        ge=0.0,
        le=100.0,
    )


class ProcessingStatistics(MemoryOSBaseSchema):
    """
    Processing pipeline statistics.
    """

    pending: int = Field(
        default=0,
        ge=0,
    )

    processing: int = Field(
        default=0,
        ge=0,
    )

    completed: int = Field(
        default=0,
        ge=0,
    )

    partial: int = Field(
        default=0,
        ge=0,
    )

    failed: int = Field(
        default=0,
        ge=0,
    )


class SearchStatistics(MemoryOSBaseSchema):
    """
    Search performance statistics.
    """

    total_searches: int = Field(
        default=0,
        ge=0,
    )

    average_search_duration_ms: float = Field(
        default=0.0,
        ge=0.0,
    )

    average_result_count: float = Field(
        default=0.0,
        ge=0.0,
    )


class SystemStatistics(MemoryOSBaseSchema):
    """
    Complete dashboard statistics.
    """

    total_memories: int = Field(
        default=0,
        ge=0,
    )

    total_images: int = Field(
        default=0,
        ge=0,
    )

    total_entities: int = Field(
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

    category_distribution: list[CategoryStatistic] = Field(
        default_factory=list,
    )

    intent_distribution: list[IntentStatistic] = Field(
        default_factory=list,
    )

    processing: ProcessingStatistics = Field(
        default_factory=ProcessingStatistics,
    )

    search: SearchStatistics = Field(
        default_factory=SearchStatistics,
    )

    generated_at: datetime


# ============================================================================
# PROCESSING RUN
# ============================================================================


class ProcessingRunResponse(MemoryOSBaseSchema):
    """
    Batch processing status.
    """

    id: int

    run_id: str

    run_type: str

    total_items: int = Field(
        ...,
        ge=0,
    )

    successful_items: int = Field(
        ...,
        ge=0,
    )

    partial_items: int = Field(
        ...,
        ge=0,
    )

    failed_items: int = Field(
        ...,
        ge=0,
    )

    status: ProcessingStatusValue

    error_summary: str | None = None

    duration_ms: int | None = None

    metadata_json: dict[str, Any] | None = None

    started_at: datetime

    completed_at: datetime | None = None


# ============================================================================
# EMBEDDING SCHEMAS
# ============================================================================


class EmbeddingRequest(MemoryOSBaseSchema):
    """
    Embedding generation configuration.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
    )

    normalize: bool = True


class EmbeddingResponse(MemoryOSBaseSchema):
    """
    Embedding metadata.

    We intentionally do NOT return the complete vector in normal API
    responses because embedding vectors are large and unnecessary for
    the frontend.
    """

    success: bool = True

    model: str

    dimension: int = Field(
        ...,
        gt=0,
    )

    normalized: bool

    vector_id: int | None = None

    processing_time_ms: int | None = None


# ============================================================================
# FAISS INDEX STATUS
# ============================================================================


class VectorIndexStatus(MemoryOSBaseSchema):
    """
    FAISS index health/status.
    """

    available: bool

    initialized: bool

    index_type: str

    dimension: int | None = None

    total_vectors: int = Field(
        default=0,
        ge=0,
    )

    index_path: str | None = None

    last_updated: datetime | None = None

    needs_rebuild: bool = False


# ============================================================================
# MEMORY UPDATE
# ============================================================================


class MemoryUpdateRequest(MemoryOSBaseSchema):
    """
    User-editable memory fields.

    AI-generated fields are intentionally excluded here.
    """

    title: str | None = Field(
        default=None,
        min_length=1,
        max_length=160,
    )

    summary: str | None = Field(
        default=None,
        max_length=5000,
    )


# ============================================================================
# DELETE RESPONSE
# ============================================================================


class DeleteMemoryResponse(MemoryOSBaseSchema):
    """
    Memory deletion response.
    """

    success: bool

    memory_id: int

    deleted_images: int = Field(
        default=0,
        ge=0,
    )

    removed_from_vector_index: bool = False

    message: str


# ============================================================================
# GENERIC PAGINATED RESPONSE
# ============================================================================


class PaginatedResponse(
    MemoryOSBaseSchema,
    Generic[T],
):
    """
    Generic reusable paginated response.

    Useful for future endpoints without duplicating pagination logic.
    """

    success: bool = True

    items: list[T]

    pagination: PaginationMeta


# ============================================================================
# API RESPONSE WRAPPER
# ============================================================================


class APIResponse(
    MemoryOSBaseSchema,
    Generic[T],
):
    """
    Generic API response wrapper.

    Example:

        {
            "success": true,
            "data": {...},
            "message": "Success"
        }
    """

    success: bool = True

    data: T | None = None

    message: str | None = None


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    # Base
    "MemoryOSBaseSchema",
    # Common
    "MessageResponse",
    "HealthResponse",
    "ErrorDetail",
    "ErrorResponse",
    # Pagination
    "PaginationRequest",
    "PaginationMeta",
    # Image
    "ImageMetadataResponse",
    "ImageAssetResponse",
    # Upload
    "UploadOptions",
    "UploadItemResponse",
    "BatchUploadResponse",
    # OCR
    "OCRRequest",
    "OCRResult",
    # Entities
    "EntitySchema",
    "EntityCreate",
    # AI
    "AIClassification",
    "AIIntent",
    "AIAnalysisRequest",
    "AIAnalysisResponse",
    # Memory
    "MemorySummaryResponse",
    "MemoryDetailResponse",
    "MemoryListResponse",
    "MemoryFilters",
    "MemoryUpdateRequest",
    # Search
    "SearchRequest",
    "SearchScoreBreakdown",
    "MatchExplanation",
    "SearchResult",
    "SearchResponse",
    # Search history
    "SearchHistoryResponse",
    "SearchHistoryListResponse",
    # Statistics
    "CategoryStatistic",
    "IntentStatistic",
    "ProcessingStatistics",
    "SearchStatistics",
    "SystemStatistics",
    # Processing
    "ProcessingRunResponse",
    # Embeddings
    "EmbeddingRequest",
    "EmbeddingResponse",
    # FAISS
    "VectorIndexStatus",
    # Delete
    "DeleteMemoryResponse",
    # Generic
    "PaginatedResponse",
    "APIResponse",
]
