"""
MemoryOS - Application Configuration

Centralized configuration for the complete MemoryOS backend.

Responsibilities:
    - Environment variable loading
    - Application metadata
    - Directory management
    - Upload configuration
    - Image configuration
    - OCR configuration
    - Gemini configuration
    - Embedding configuration
    - FAISS configuration
    - Search configuration
    - API configuration
    - Security-related limits
    - Development/testing configuration

Design principle:

    Environment variables
            ↓
       Settings object
            ↓
       Entire backend
            ↓
    services / APIs / AI

No service should read os.getenv() directly.

That keeps configuration centralized, testable and predictable.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ============================================================================
# PROJECT PATHS
# ============================================================================

# backend/config.py
#      ↑
# backend/
#      ↑
# project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

BACKEND_DIR = PROJECT_ROOT / "backend"

DATA_DIR = PROJECT_ROOT / "data"

UPLOADS_DIR = DATA_DIR / "uploads"

ORIGINALS_DIR = UPLOADS_DIR / "originals"

THUMBNAILS_DIR = UPLOADS_DIR / "thumbnails"

PROCESSED_DIR = DATA_DIR / "processed"

OCR_DIR = PROCESSED_DIR / "ocr"

AI_DIR = PROCESSED_DIR / "ai"

INDEX_DIR = DATA_DIR / "index"

FAISS_DIR = INDEX_DIR / "faiss"

DATABASE_DIR = DATA_DIR / "database"

LOGS_DIR = PROJECT_ROOT / "logs"

TESTS_DIR = PROJECT_ROOT / "tests"


# ============================================================================
# ENVIRONMENT FILE
# ============================================================================

ENV_FILE = PROJECT_ROOT / ".env"

load_dotenv(
    ENV_FILE,
    override=False,
)


# ============================================================================
# SETTINGS
# ============================================================================


class Settings(BaseSettings):
    """
    Central MemoryOS application settings.

    Values can be supplied through:

        1. Environment variables
        2. .env file
        3. Defaults defined here

    Environment variables take priority over defaults.
    """

    # ========================================================================
    # APPLICATION
    # ========================================================================

    app_name: str = Field(
        default="MemoryOS",
        description="Application name.",
    )

    app_version: str = Field(
        default="1.0.0",
        description="Application version.",
    )

    app_environment: Literal[
        "development",
        "testing",
        "production",
    ] = Field(
        default="development",
        description="Current application environment.",
    )

    debug: bool = Field(
        default=True,
        validation_alias="MEMORYOS_DEBUG",
        description="Enable development debugging.",
    )

    timezone: str = Field(
        default="Asia/Kolkata",
        description="Application timezone.",
    )

    # ========================================================================
    # API
    # ========================================================================

    api_prefix: str = Field(
        default="/api",
        description="Prefix used by all API routes.",
    )

    host: str = Field(
        default="127.0.0.1",
        description="FastAPI host.",
    )

    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="FastAPI server port.",
    )

    cors_origins: str = Field(
        default="http://127.0.0.1:8000,http://localhost:8000",
        description="Comma-separated CORS origins.",
    )

    # ========================================================================
    # DATABASE
    # ========================================================================

    database_url: str = Field(
        default="sqlite:///./data/database/memoryos.db",
        description="SQLAlchemy database URL.",
    )

    database_echo: bool = Field(
        default=False,
        description="Enable SQLAlchemy SQL logging.",
    )

    # ========================================================================
    # CLOUD / IDENTITY
    # ========================================================================

    cloud_mode: bool = Field(default=False, validation_alias="MEMORYOS_CLOUD_MODE")
    storage_backend: str = Field(default="local", validation_alias="MEMORYOS_STORAGE_BACKEND")
    supabase_url: str = Field(default="", validation_alias="SUPABASE_URL")
    supabase_service_role_key: str = Field(default="", validation_alias="SUPABASE_SERVICE_ROLE_KEY")
    supabase_storage_bucket: str = Field(default="memoryos-private", validation_alias="SUPABASE_STORAGE_BUCKET")
    auth_secret: str = Field(default="", validation_alias="MEMORYOS_AUTH_SECRET")
    auth_token_ttl_hours: int = Field(default=168, validation_alias="MEMORYOS_AUTH_TOKEN_TTL_HOURS", ge=1, le=24 * 90)
    vector_backend: str = Field(default="pgvector", validation_alias="MEMORYOS_VECTOR_BACKEND")
    vector_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", validation_alias="MEMORYOS_VECTOR_MODEL")
    vector_dimension: int = Field(default=384, validation_alias="MEMORYOS_VECTOR_DIMENSION", ge=1, le=4096)
    worker_max_attempts: int = Field(default=5, validation_alias="MEMORYOS_WORKER_MAX_ATTEMPTS", ge=1, le=20)

    # ========================================================================
    # UPLOADS
    # ========================================================================

    max_upload_size_mb: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum upload size per image in MB.",
    )

    max_batch_images: int = Field(
        default=25,
        ge=1,
        le=500,
        description="Maximum number of images per batch.",
    )

    max_total_batch_size_mb: int = Field(
        default=100,
        ge=1,
        le=1000,
        description="Maximum total batch upload size in MB.",
    )

    allowed_image_extensions: str = Field(
        default=".jpg,.jpeg,.png,.webp,.gif,.bmp,.tiff,.tif",
        description="Allowed image extensions.",
    )

    allowed_image_mime_types: str = Field(
        default=("image/jpeg," "image/png," "image/webp," "image/gif," "image/bmp," "image/tiff"),
        description="Allowed image MIME types.",
    )

    # ========================================================================
    # IMAGE PROCESSING
    # ========================================================================

    max_image_dimension: int = Field(
        default=8192,
        ge=512,
        le=20000,
        description="Maximum width or height accepted by image utilities.",
    )

    image_thumbnail_width: int = Field(
        default=480,
        ge=64,
        le=2000,
        description="Thumbnail width.",
    )

    image_thumbnail_height: int = Field(
        default=480,
        ge=64,
        le=2000,
        description="Thumbnail height.",
    )

    image_jpeg_quality: int = Field(
        default=92,
        ge=50,
        le=100,
        description="JPEG processing quality.",
    )

    # ========================================================================
    # OCR
    # ========================================================================

    tesseract_cmd: str = Field(
        default="",
        description=(
            "Optional path to the Tesseract executable. " "Empty means use system PATH."
        ),
    )

    ocr_enabled: bool = Field(
        default=True,
        description="Enable local OCR.",
    )

    ocr_language: str = Field(
        default="eng",
        description="Tesseract OCR language configuration.",
    )

    ocr_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Maximum OCR execution time.",
    )

    ocr_psm: int = Field(
        default=6,
        ge=0,
        le=13,
        description="Tesseract page segmentation mode.",
    )

    # ========================================================================
    # GEMINI
    # ========================================================================

    gemini_api_key: str = Field(
        default="",
        description="Google Gemini API key.",
    )

    gemini_model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini vision model.",
    )

    gemini_enabled: bool = Field(
        default=True,
        description="Enable Gemini Vision processing.",
    )

    gemini_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=2.0,
        description="Gemini generation temperature.",
    )

    gemini_max_output_tokens: int = Field(
        default=4096,
        ge=256,
        le=32768,
        description="Maximum Gemini response tokens.",
    )

    gemini_timeout_seconds: int = Field(
        default=60,
        ge=5,
        le=300,
        description="Gemini request timeout.",
    )

    gemini_max_retries: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Maximum Gemini retry count.",
    )

    # ========================================================================
    # EMBEDDINGS
    # ========================================================================

    embedding_enabled: bool = Field(
        default=True,
        description="Enable local embedding generation.",
    )

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence Transformer embedding model.",
    )

    embedding_dimension: int = Field(
        default=384,
        ge=1,
        le=4096,
        description="Expected embedding vector dimension.",
    )

    embedding_batch_size: int = Field(
        default=32,
        ge=1,
        le=512,
        description="Embedding batch size.",
    )

    embedding_normalize: bool = Field(
        default=True,
        description="Normalize embedding vectors.",
    )

    # ========================================================================
    # FAISS
    # ========================================================================

    faiss_enabled: bool = Field(
        default=True,
        description="Enable FAISS vector search.",
    )

    faiss_index_type: Literal[
        "flat",
        "hnsw",
    ] = Field(
        default="flat",
        description="FAISS index implementation.",
    )

    faiss_metric: Literal[
        "cosine",
        "l2",
        "inner_product",
    ] = Field(
        default="cosine",
        description="FAISS similarity metric.",
    )

    faiss_top_k: int = Field(
        default=20,
        ge=1,
        le=500,
        description="Default number of vector candidates.",
    )

    faiss_hnsw_m: int = Field(
        default=32,
        ge=4,
        le=128,
        description="HNSW graph connectivity.",
    )

    # ========================================================================
    # SEARCH
    # ========================================================================

    search_top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Default number of final search results.",
    )

    search_min_score: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Minimum acceptable search score.",
    )

    semantic_search_weight: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description="Semantic/vector search contribution.",
    )

    keyword_search_weight: float = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Keyword search contribution.",
    )

    metadata_search_weight: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="Metadata search contribution.",
    )

    reranking_enabled: bool = Field(
        default=True,
        description="Enable result reranking.",
    )

    rerank_top_k: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Number of candidates sent to reranking.",
    )

    # ========================================================================
    # MEMORY PROCESSING
    # ========================================================================

    memory_title_max_length: int = Field(
        default=160,
        ge=20,
        le=1000,
        description="Maximum generated memory title length.",
    )

    memory_text_max_length: int = Field(
        default=50_000,
        ge=1000,
        le=1_000_000,
        description="Maximum normalized memory text length.",
    )

    entity_max_count: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum entities extracted from one memory.",
    )

    # ========================================================================
    # SECURITY
    # ========================================================================

    max_filename_length: int = Field(
        default=255,
        ge=32,
        le=512,
        description="Maximum sanitized filename length.",
    )

    allow_symlinks: bool = Field(
        default=False,
        description="Whether uploaded files may follow symlinks.",
    )

    # ========================================================================
    # LOGGING
    # ========================================================================

    log_level: Literal[
        "DEBUG",
        "INFO",
        "WARNING",
        "ERROR",
        "CRITICAL",
    ] = Field(
        default="INFO",
        description="Application logging level.",
    )

    # ========================================================================
    # TESTING
    # ========================================================================

    testing: bool = Field(
        default=False,
        description="Enable test mode.",
    )

    # ========================================================================
    # PYDANTIC SETTINGS CONFIG
    # ========================================================================

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ========================================================================
    # VALIDATORS
    # ========================================================================

    @field_validator(
        "max_upload_size_mb",
        "max_total_batch_size_mb",
        "max_image_dimension",
        "embedding_dimension",
        "search_top_k",
        "faiss_top_k",
    )
    @classmethod
    def validate_positive_integer(
        cls,
        value: int,
    ) -> int:
        """
        Ensure important numeric settings are positive.
        """

        if value <= 0:
            raise ValueError("Configuration value must be greater than zero.")

        return value

    @field_validator(
        "semantic_search_weight",
        "keyword_search_weight",
        "metadata_search_weight",
    )
    @classmethod
    def validate_search_weight(
        cls,
        value: float,
    ) -> float:
        """
        Validate individual search weights.
        """

        if not 0.0 <= value <= 1.0:
            raise ValueError("Search weights must be between 0 and 1.")

        return value

    # ========================================================================
    # COMPUTED PROPERTIES
    # ========================================================================

    @property
    def upload_max_bytes(self) -> int:
        """
        Maximum individual upload size in bytes.
        """

        return self.max_upload_size_mb * 1024 * 1024

    @property
    def total_batch_max_bytes(self) -> int:
        """
        Maximum batch upload size in bytes.
        """

        return self.max_total_batch_size_mb * 1024 * 1024

    @property
    def cors_origin_list(self) -> list[str]:
        """
        Convert comma-separated CORS configuration into a list.
        """

        return [
            origin.strip() for origin in self.cors_origins.split(",") if origin.strip()
        ]

    @property
    def allowed_extensions(self) -> set[str]:
        """
        Return normalized allowed image extensions.
        """

        return {
            (
                extension.strip().lower()
                if extension.strip().startswith(".")
                else f".{extension.strip().lower()}"
            )
            for extension in self.allowed_image_extensions.split(",")
            if extension.strip()
        }

    @property
    def allowed_mime_types(self) -> set[str]:
        """
        Return normalized allowed MIME types.
        """

        return {
            mime.strip().lower()
            for mime in self.allowed_image_mime_types.split(",")
            if mime.strip()
        }

    @property
    def gemini_available(self) -> bool:
        """
        Determine whether Gemini can actually be used.

        Gemini is considered available only when:
            - Gemini is enabled
            - an API key exists
        """

        return self.gemini_enabled and bool(self.gemini_api_key.strip())

    @property
    def tesseract_available(self) -> bool:
        """
        Determine whether an explicit Tesseract executable was configured.

        An empty value means Tesseract may still be available through PATH.
        """

        return bool(self.tesseract_cmd.strip())

    @property
    def search_weights(self) -> dict[str, float]:
        """
        Return hybrid search weights.
        """

        return {
            "semantic": self.semantic_search_weight,
            "keyword": self.keyword_search_weight,
            "metadata": self.metadata_search_weight,
        }

    @property
    def database_path(self) -> Path:
        """
        Return the SQLite database path when SQLite is being used.
        """

        prefix = "sqlite:///"

        if not self.database_url.startswith(prefix):
            return DATABASE_DIR / "memoryos.db"

        relative_path = self.database_url[len(prefix) :]

        path = Path(relative_path)

        if path.is_absolute():
            return path

        return PROJECT_ROOT / path

    # ========================================================================
    # DIRECTORY INITIALIZATION
    # ========================================================================

    def ensure_directories(self) -> None:
        """
        Create all directories required by MemoryOS.
        """

        directories = (
            DATA_DIR,
            UPLOADS_DIR,
            ORIGINALS_DIR,
            THUMBNAILS_DIR,
            PROCESSED_DIR,
            OCR_DIR,
            AI_DIR,
            INDEX_DIR,
            FAISS_DIR,
            DATABASE_DIR,
            LOGS_DIR,
        )

        for directory in directories:
            directory.mkdir(
                parents=True,
                exist_ok=True,
            )

    # ========================================================================
    # SAFE DEBUG REPRESENTATION
    # ========================================================================

    def safe_dict(self) -> dict[str, object]:
        """
        Return configuration without exposing secrets.

        API keys must never appear in logs or debug output.
        """

        data = self.model_dump()

        secret_fields = {
            "gemini_api_key",
            "supabase_service_role_key",
            "auth_secret",
        }

        for field_name in secret_fields:
            if field_name in data:
                value = data[field_name]

                if value:
                    data[field_name] = "***configured***"
                else:
                    data[field_name] = "***not configured***"

        return data


# ============================================================================
# GLOBAL SETTINGS INSTANCE
# ============================================================================

settings = Settings()


# ============================================================================
# DEVELOPMENT DIRECTORY INITIALIZATION
# ============================================================================

# Creating these directories is intentionally lightweight and safe.
#
# It means services can immediately use:
#
#     settings.ensure_directories()
#
# without failing because data directories do not exist.
#
# We do NOT automatically create directories during import because importing
# configuration should ideally remain side-effect-light.
# ============================================================================


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "PROJECT_ROOT",
    "BACKEND_DIR",
    "DATA_DIR",
    "UPLOADS_DIR",
    "ORIGINALS_DIR",
    "THUMBNAILS_DIR",
    "PROCESSED_DIR",
    "OCR_DIR",
    "AI_DIR",
    "INDEX_DIR",
    "FAISS_DIR",
    "DATABASE_DIR",
    "LOGS_DIR",
    "TESTS_DIR",
    "ENV_FILE",
    "Settings",
    "settings",
]
