"""
MemoryOS — Production Processing Service
========================================

Canonical ingestion pipeline:

    IMAGE
      │
      ├── validate
      ├── read bytes
      ├── sha256
      ├── stable memory identity
      │
      ├── OCR
      ├── Vision / Gemini
      ├── semantic classification
      ├── entities / keywords / intent
      ├── embedding (384D)
      ├── MemoryService persistence
      └── FAISS indexing

Important:
------------
This service NEVER invents a second identity when the upload layer already
provided a memory_id.

Expected upload metadata may contain:

    memory_id
    stored_path
    file_path
    source_path
    upload_id
    preview_url
    image_url
    thumbnail_url

This is important because the upload API and image-serving endpoint must use
the SAME memory identifier.
"""

from __future__ import annotations

import hashlib
import logging
import mimetypes
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger("memoryos.processing")


# ============================================================================
# CONSTANTS
# ============================================================================

EMBEDDING_DIMENSION = 384

DEFAULT_MAX_OCR_TEXT = 20_000
DEFAULT_MAX_ERRORS = 30

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}

SUPPORTED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/gif",
    "image/tiff",
}


# ============================================================================
# EXCEPTIONS
# ============================================================================


class ProcessingError(RuntimeError):
    """Base MemoryOS processing exception."""


class ProcessingValidationError(ProcessingError):
    """Raised when an input image is invalid."""


class ProcessingDependencyError(ProcessingError):
    """Raised when a required processing dependency is unavailable."""


class EmbeddingDimensionError(ProcessingError):
    """Raised when the embedding is not exactly 384 dimensions."""


# ============================================================================
# DATACLASSES
# ============================================================================


@dataclass
class ProcessingDiagnostic:
    stage: str
    message: str
    level: str = "warning"
    exception_type: str | None = None
    recoverable: bool = True
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProcessingResult:
    success: bool

    memory_id: str | None = None

    source_name: str | None = None
    source_type: str = "image"
    source_path: str | None = None

    sha256: str | None = None
    mime_type: str | None = None
    file_size: int | None = None

    # Image-serving information.
    image_url: str | None = None
    preview_url: str | None = None
    thumbnail_url: str | None = None

    ocr_text: str = ""

    description: str = ""
    category: str = "other"
    confidence: float | None = None

    entities: list[Any] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    intent: str | None = None

    embedding_dimension: int | None = None

    indexed: bool = False
    duplicate: bool = False

    processing_time_ms: float = 0.0

    diagnostics: list[ProcessingDiagnostic] = field(default_factory=list)

    memory: Any | None = None
    vision: Any | None = None

    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def add_diagnostic(
        self,
        stage: str,
        message: str,
        *,
        level: str = "warning",
        exception: Exception | None = None,
        recoverable: bool = True,
    ) -> None:
        self.diagnostics.append(
            ProcessingDiagnostic(
                stage=stage,
                message=message,
                level=level,
                exception_type=(
                    type(exception).__name__ if exception is not None else None
                ),
                recoverable=recoverable,
            )
        )

    @property
    def warnings(self) -> list[str]:
        return [
            diagnostic.message
            for diagnostic in self.diagnostics
            if diagnostic.level == "warning"
        ]

    @property
    def errors(self) -> list[str]:
        return [
            diagnostic.message
            for diagnostic in self.diagnostics
            if diagnostic.level == "error"
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "memory_id": self.memory_id,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "source_path": self.source_path,
            "sha256": self.sha256,
            "mime_type": self.mime_type,
            "file_size": self.file_size,
            # Critical for frontend image rendering.
            "image_url": self.image_url,
            "preview_url": self.preview_url,
            "thumbnail_url": self.thumbnail_url,
            "ocr_text": self.ocr_text,
            "description": self.description,
            "category": self.category,
            "confidence": self.confidence,
            "entities": self.entities,
            "keywords": self.keywords,
            "intent": self.intent,
            "vision": _serialize_safe(self.vision),
            "embedding_dimension": self.embedding_dimension,
            "indexed": self.indexed,
            "duplicate": self.duplicate,
            "processing_time_ms": self.processing_time_ms,
            "diagnostics": [diagnostic.to_dict() for diagnostic in self.diagnostics],
            "raw_metadata": self.raw_metadata,
        }


@dataclass
class BatchProcessingResult:
    success: bool

    total: int = 0
    successful: int = 0
    failed: int = 0
    duplicates: int = 0
    indexed: int = 0

    processing_time_ms: float = 0.0

    results: list[ProcessingResult] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "total": self.total,
            "successful": self.successful,
            "failed": self.failed,
            "duplicates": self.duplicates,
            "indexed": self.indexed,
            "processing_time_ms": self.processing_time_ms,
            "results": [result.to_dict() for result in self.results],
            "errors": self.errors,
        }


# ============================================================================
# GENERAL HELPERS
# ============================================================================


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_now() -> str:
    return _utc_now().isoformat()


def _safe_string(
    value: Any,
    default: str = "",
) -> str:
    if value is None:
        return default

    if isinstance(value, str):
        return value.strip()

    try:
        return str(value).strip()
    except Exception:
        return default


def _get_value(
    obj: Any,
    *names: str,
    default: Any = None,
) -> Any:
    """
    Safely retrieve a value from:

    - dict
    - Pydantic model
    - dataclass
    - normal object
    """

    if obj is None:
        return default

    for name in names:

        if isinstance(obj, Mapping):
            if name in obj:
                value = obj[name]

                if value is not None:
                    return value

        try:
            value = getattr(obj, name)

            if value is not None:
                return value

        except Exception:
            continue

    return default


def _serialize_safe(value: Any) -> Any:
    """
    Convert service/model values to JSON-compatible structures.
    """

    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, Mapping):
        return {str(key): _serialize_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_serialize_safe(item) for item in value]

    if hasattr(value, "model_dump"):
        try:
            return _serialize_safe(value.model_dump())
        except Exception:
            pass

    if hasattr(value, "dict"):
        try:
            return _serialize_safe(value.dict())
        except Exception:
            pass

    if hasattr(value, "__dict__"):
        try:
            return {
                str(key): _serialize_safe(item)
                for key, item in vars(value).items()
                if not key.startswith("_")
            }
        except Exception:
            pass

    return str(value)


# ============================================================================
# SOURCE HELPERS
# ============================================================================


def _source_name(
    source: Any,
    fallback: str,
) -> str:

    if isinstance(source, (str, Path)):
        return Path(source).name

    name = _get_value(
        source,
        "filename",
        "name",
        default=None,
    )

    if name:
        return _safe_string(
            name,
            fallback,
        )

    return fallback


def _detect_mime_type(
    source: Any,
) -> str | None:

    if isinstance(source, (str, Path)):

        mime, _ = mimetypes.guess_type(str(source))

        if mime:
            return mime

        suffix = Path(source).suffix.lower()

        mapping = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".webp": "image/webp",
            ".bmp": "image/bmp",
            ".gif": "image/gif",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
        }

        return mapping.get(suffix)

    return None


def _read_source_bytes(
    source: Any,
) -> bytes:

    if isinstance(source, bytes):
        return source

    if isinstance(source, bytearray):
        return bytes(source)

    if isinstance(source, memoryview):
        return source.tobytes()

    if isinstance(source, (str, Path)):

        path = Path(source)

        if not path.exists():
            raise ProcessingValidationError(f"Source file does not exist: {path}")

        if not path.is_file():
            raise ProcessingValidationError(f"Source is not a file: {path}")

        return path.read_bytes()

    read_method = getattr(
        source,
        "read",
        None,
    )

    if callable(read_method):

        try:
            result = read_method()

            if hasattr(result, "__await__"):
                raise ProcessingValidationError(
                    "Async UploadFile must be read before "
                    "calling ProcessingService.process()."
                )

            if isinstance(result, bytes):
                return result

            if isinstance(result, bytearray):
                return bytes(result)

        except ProcessingValidationError:
            raise

        except Exception as exc:
            raise ProcessingValidationError(f"Unable to read source: {exc}") from exc

    raise ProcessingValidationError(
        f"Unsupported source type: " f"{type(source).__name__}"
    )


def _calculate_sha256(
    data: bytes,
) -> str:
    return hashlib.sha256(data).hexdigest()


def _validate_image_source(
    source: Any,
    *,
    source_name: str,
    mime_type: str | None,
) -> None:

    if not source_name:
        raise ProcessingValidationError("Source name cannot be empty.")

    suffix = Path(source_name).suffix.lower()

    if mime_type:

        if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            raise ProcessingValidationError(f"Unsupported image MIME type: {mime_type}")

    elif suffix:

        if suffix not in SUPPORTED_IMAGE_EXTENSIONS:
            raise ProcessingValidationError(f"Unsupported image extension: {suffix}")


# ============================================================================
# NORMALIZATION
# ============================================================================


def _normalize_list(
    value: Any,
) -> list[Any]:

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, set):
        return list(value)

    if isinstance(value, str):

        value = value.strip()

        if not value:
            return []

        return [value]

    return [value]


def _normalize_keywords(
    value: Any,
) -> list[str]:

    values = _normalize_list(value)

    result: list[str] = []

    for item in values:

        text = _safe_string(item)

        if not text:
            continue

        if text.lower() not in {value.lower() for value in result}:
            result.append(text)

    return result


def _clamp_confidence(
    value: Any,
) -> float | None:

    if value is None:
        return None

    try:
        confidence = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return None

    if not np.isfinite(confidence):
        return None

    return max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )


# ============================================================================
# EMBEDDING
# ============================================================================


def _extract_embedding_array(
    value: Any,
) -> np.ndarray:

    if value is None:
        raise ProcessingDependencyError("Embedding service returned no embedding.")

    if isinstance(value, Mapping):

        for key in (
            "embedding",
            "vector",
            "values",
            "data",
        ):

            if key in value:
                return _extract_embedding_array(value[key])

    for attr in (
        "embedding",
        "vector",
        "values",
    ):

        nested = getattr(
            value,
            attr,
            None,
        )

        if nested is not None:
            return _extract_embedding_array(nested)

    array = np.asarray(
        value,
        dtype=np.float32,
    )

    if array.ndim == 2:

        if array.shape[0] == 1:
            array = array[0]

        elif array.shape[1] == 1:
            array = array[:, 0]

    array = np.asarray(
        array,
        dtype=np.float32,
    ).reshape(-1)

    if array.size == 0:
        raise ProcessingDependencyError("Embedding service returned an empty vector.")

    if not np.all(np.isfinite(array)):
        raise ProcessingDependencyError("Embedding contains NaN or infinite values.")

    if array.shape[0] != EMBEDDING_DIMENSION:
        raise EmbeddingDimensionError(
            "Invalid embedding dimension: "
            f"expected {EMBEDDING_DIMENSION}, "
            f"received {array.shape[0]}"
        )

    norm = float(np.linalg.norm(array))

    if norm <= 0:
        raise ProcessingDependencyError("Embedding vector has zero magnitude.")

    array = array / norm

    return np.ascontiguousarray(
        array,
        dtype=np.float32,
    )


# ============================================================================
# PROCESSING SERVICE
# ============================================================================


class ProcessingService:

    def __init__(
        self,
        *,
        memory_service_instance: Any | None = None,
        vision_service_instance: Any | None = None,
        ocr_service_instance: Any | None = None,
        embedding_service_instance: Any | None = None,
        vector_store_instance: Any | None = None,
        max_ocr_text: int = DEFAULT_MAX_OCR_TEXT,
        max_errors: int = DEFAULT_MAX_ERRORS,
    ) -> None:

        self.max_ocr_text = max(
            1,
            int(max_ocr_text),
        )

        self.max_errors = max(
            1,
            int(max_errors),
        )

        self.memory_service = (
            memory_service_instance
            if memory_service_instance is not None
            else self._load_memory_service()
        )

        self.vision_service = (
            vision_service_instance
            if vision_service_instance is not None
            else self._load_vision_service()
        )

        self.ocr_service = (
            ocr_service_instance
            if ocr_service_instance is not None
            else self._load_ocr_service()
        )

        self.embedding_service = (
            embedding_service_instance
            if embedding_service_instance is not None
            else self._load_embedding_service()
        )

        self.vector_store = (
            vector_store_instance
            if vector_store_instance is not None
            else self._load_vector_store()
        )

        logger.info(
            "MemoryOS ProcessingService initialized " "(embedding_dim=%s)",
            EMBEDDING_DIMENSION,
        )

    # ========================================================================
    # DEPENDENCY LOADERS
    # ========================================================================

    @staticmethod
    def _load_memory_service() -> Any | None:

        try:

            from backend.services.memory_service import (
                memory_service,
            )

            return memory_service

        except Exception as exc:

            logger.warning(
                "MemoryService unavailable: %s",
                exc,
            )

            return None

    @staticmethod
    def _load_vision_service() -> Any | None:

        try:

            from backend.services.vision_service import (
                VisionService,
            )

            return VisionService()

        except Exception as exc:

            logger.warning(
                "VisionService unavailable: %s",
                exc,
            )

            return None

    @staticmethod
    def _load_ocr_service() -> Any | None:

        try:

            from backend.services.ocr_service import (
                OCRService,
            )

            return OCRService()

        except Exception as exc:

            logger.warning(
                "OCRService unavailable: %s",
                exc,
            )

            return None

    @staticmethod
    def _load_embedding_service() -> Any | None:

        try:

            from backend.services.semantic_embedding_service import (
                get_embedding_service,
            )

            return get_embedding_service()

        except Exception as exc:

            logger.warning(
                "EmbeddingService unavailable: %s",
                exc,
            )

            return None

    @staticmethod
    def _load_vector_store() -> Any | None:

        try:

            from backend.services.vector_store import (
                create_vector_store,
            )

            return create_vector_store()

        except Exception as exc:

            logger.warning(
                "VectorStore unavailable: %s",
                exc,
            )

            return None

    # ========================================================================
    # MEMORY IDENTITY
    # ========================================================================

    @staticmethod
    def _resolve_memory_id(
        *,
        metadata: Mapping[str, Any] | None,
        sha256: str,
        source_name: str,
    ) -> str:

        metadata = metadata or {}

        # IMPORTANT:
        # Prefer the ID created by the upload service.
        for key in (
            "memory_id",
            "id",
            "upload_id",
        ):

            value = metadata.get(key)

            if value:

                value = _safe_string(value)

                if value:
                    return value

        # Fallback only when upload layer did not provide one.
        digest = hashlib.sha256(f"{sha256}:{source_name}".encode("utf-8")).hexdigest()[
            :24
        ]

        return f"mem_{digest}"

    # ========================================================================
    # IMAGE URLS
    # ========================================================================

    @staticmethod
    def _resolve_image_urls(
        *,
        memory_id: str,
        metadata: Mapping[str, Any] | None,
    ) -> tuple[
        str | None,
        str | None,
        str | None,
    ]:

        metadata = metadata or {}

        image_url = _get_value(
            metadata,
            "image_url",
            "imageUrl",
            default=None,
        )

        preview_url = _get_value(
            metadata,
            "preview_url",
            "previewUrl",
            default=None,
        )

        thumbnail_url = _get_value(
            metadata,
            "thumbnail_url",
            "thumbnailUrl",
            default=None,
        )

        # If upload service already supplied URLs, preserve them.
        if image_url or preview_url or thumbnail_url:

            return (
                _safe_string(image_url) or None,
                _safe_string(preview_url) or None,
                _safe_string(thumbnail_url) or None,
            )

        # Do not fabricate a frontend URL.
        #
        # The backend upload router should normally expose:
        #
        #     /upload/file/{memory_id}
        #
        # Returning the relative API route here gives the frontend/API
        # layer a canonical value without hardcoding localhost.
        canonical = f"/upload/file/{memory_id}"

        return (
            canonical,
            canonical,
            canonical,
        )

    # ========================================================================
    # PROCESS
    # ========================================================================

    def process(
        self,
        source: Any,
        *,
        source_name: str = "upload",
        metadata: Mapping[str, Any] | None = None,
        index: bool = True,
        allow_partial_success: bool = True,
    ) -> ProcessingResult:

        started = time.perf_counter()

        metadata_dict = dict(metadata or {})

        result = ProcessingResult(
            success=False,
            source_name=source_name,
            source_type="image",
            raw_metadata=metadata_dict,
        )

        try:

            # ----------------------------------------------------------------
            # 1. Source metadata
            # ----------------------------------------------------------------

            actual_source_name = _source_name(
                source,
                source_name,
            )

            result.source_name = actual_source_name

            mime_type = (
                _get_value(
                    source,
                    "content_type",
                    "mime_type",
                    default=None,
                )
                or _get_value(
                    metadata_dict,
                    "mime_type",
                    "content_type",
                    default=None,
                )
                or _detect_mime_type(source)
            )

            result.mime_type = mime_type

            # IMPORTANT:
            # Preserve the path generated by upload_service.
            source_path = _get_value(
                metadata_dict,
                "stored_path",
                "file_path",
                "source_path",
                "path",
                default=None,
            )

            if not source_path:

                if isinstance(
                    source,
                    (str, Path),
                ):

                    source_path = str(Path(source).resolve())

                else:

                    source_path = _get_value(
                        source,
                        "path",
                        "file_path",
                        "source_path",
                        default=None,
                    )

            if source_path:
                source_path = str(source_path)

            result.source_path = source_path

            # ----------------------------------------------------------------
            # 2. Validate
            # ----------------------------------------------------------------

            _validate_image_source(
                source,
                source_name=actual_source_name,
                mime_type=mime_type,
            )

            source_bytes = _read_source_bytes(source)

            result.file_size = len(source_bytes)

            if result.file_size <= 0:
                raise ProcessingValidationError("Image file is empty.")

            result.sha256 = _calculate_sha256(source_bytes)

            # ----------------------------------------------------------------
            # 3. Stable memory identity
            # ----------------------------------------------------------------

            result.memory_id = self._resolve_memory_id(
                metadata=metadata_dict,
                sha256=result.sha256,
                source_name=actual_source_name,
            )

            logger.info(
                "Processing memory_id=%s source=%s",
                result.memory_id,
                actual_source_name,
            )

            # ----------------------------------------------------------------
            # 4. Resolve image URLs
            # ----------------------------------------------------------------

            (
                result.image_url,
                result.preview_url,
                result.thumbnail_url,
            ) = self._resolve_image_urls(
                memory_id=result.memory_id,
                metadata=metadata_dict,
            )

            # Store canonical URLs in metadata too.
            metadata_dict.setdefault(
                "memory_id",
                result.memory_id,
            )

            if result.source_path:
                metadata_dict.setdefault(
                    "stored_path",
                    result.source_path,
                )

            if result.image_url:
                metadata_dict.setdefault(
                    "image_url",
                    result.image_url,
                )

            if result.preview_url:
                metadata_dict.setdefault(
                    "preview_url",
                    result.preview_url,
                )

            if result.thumbnail_url:
                metadata_dict.setdefault(
                    "thumbnail_url",
                    result.thumbnail_url,
                )

            result.raw_metadata = metadata_dict

            # ----------------------------------------------------------------
            # 5. Duplicate detection
            # ----------------------------------------------------------------

            if self._is_duplicate(
                result.memory_id,
                result.sha256,
                metadata_dict,
            ):

                result.duplicate = True

                result.add_diagnostic(
                    "duplicate",
                    "Duplicate memory detected.",
                    level="info",
                    recoverable=True,
                )

                existing_memory = self._find_existing_memory(
                    result.memory_id,
                    result.sha256,
                )

                result.memory = existing_memory

                result.success = True

                return result

            # ----------------------------------------------------------------
            # 6. OCR
            # ----------------------------------------------------------------

            ocr_text = ""

            if self.ocr_service is None:

                result.add_diagnostic(
                    "ocr",
                    "OCR service unavailable. " "Continuing without OCR.",
                    level="warning",
                )

            else:

                try:

                    ocr_result = self.ocr_service.extract_text(source)

                    ocr_text = _safe_string(
                        _get_value(
                            ocr_result,
                            "text",
                            "ocr_text",
                            "content",
                            default=ocr_result,
                        )
                    )

                    if len(ocr_text) > self.max_ocr_text:

                        ocr_text = ocr_text[: self.max_ocr_text] + "..."

                except Exception as exc:

                    result.add_diagnostic(
                        "ocr",
                        f"OCR failed: {exc}",
                        exception=exc,
                        recoverable=allow_partial_success,
                    )

                    if not allow_partial_success:
                        raise

            result.ocr_text = ocr_text

            # ----------------------------------------------------------------
            # 7. Vision / Gemini
            # ----------------------------------------------------------------

            vision_result = None

            if self.vision_service is None:

                result.add_diagnostic(
                    "vision",
                    "Vision service unavailable. " "Continuing with OCR/local signals.",
                    level="warning",
                )

            else:

                try:

                    vision_result = self.vision_service.analyze_with_ocr(
                        source,
                        ocr_text=ocr_text,
                    )

                except Exception as exc:

                    result.add_diagnostic(
                        "vision",
                        f"Vision analysis failed: {exc}",
                        exception=exc,
                        recoverable=allow_partial_success,
                    )

                    if not allow_partial_success:
                        raise

            result.vision = vision_result

            # ----------------------------------------------------------------
            # 8. Semantic extraction
            # ----------------------------------------------------------------

            description = _safe_string(
                _get_value(
                    vision_result,
                    "description",
                    "summary",
                    "visual_description",
                    "caption",
                    "text_description",
                    default="",
                )
            )

            category = _safe_string(
                _get_value(
                    vision_result,
                    "category",
                    "classification",
                    "type",
                    "memory_type",
                    default="other",
                ),
                default="other",
            )

            confidence = _clamp_confidence(
                _get_value(
                    vision_result,
                    "confidence",
                    "score",
                    "classification_confidence",
                    default=None,
                )
            )

            entities = _normalize_list(
                _get_value(
                    vision_result,
                    "entities",
                    "extracted_entities",
                    default=[],
                )
            )

            keywords = _normalize_keywords(
                _get_value(
                    vision_result,
                    "keywords",
                    "tags",
                    default=[],
                )
            )

            keywords = self._merge_keywords(
                keywords,
                _normalize_keywords(_get_value(vision_result, "products", default=[])),
            )

            intent = (
                _safe_string(
                    _get_value(
                        vision_result,
                        "intent",
                        "user_intent",
                        default=None,
                    )
                )
                or None
            )

            # ----------------------------------------------------------------
            # 9. Fallback description
            # ----------------------------------------------------------------

            if not description:

                description = self._build_fallback_description(
                    actual_source_name,
                    ocr_text,
                    category,
                )

            # ----------------------------------------------------------------
            # 10. Normalize category
            # ----------------------------------------------------------------

            category = self._normalize_category(
                category,
                ocr_text,
                description,
            )

            # ----------------------------------------------------------------
            # 11. Local keywords
            # ----------------------------------------------------------------

            keywords = self._merge_keywords(
                keywords,
                self._keywords_from_text(ocr_text),
            )

            # ----------------------------------------------------------------
            # 12. Build embedding text
            # ----------------------------------------------------------------

            embedding_text = self._build_embedding_text(
                source_name=actual_source_name,
                category=category,
                description=description,
                ocr_text=ocr_text,
                entities=entities,
                keywords=keywords,
                intent=intent,
            )

            # ----------------------------------------------------------------
            # 13. Embedding
            # ----------------------------------------------------------------

            embedding = None

            if self.embedding_service is None:

                result.add_diagnostic(
                    "embedding",
                    "Embedding service unavailable.",
                    level="error",
                    recoverable=False,
                )

                if index:

                    raise ProcessingDependencyError(
                        "Embedding service is required " "when index=True."
                    )

            else:

                try:

                    embedding_result = self.embedding_service.embed_text(embedding_text)

                    embedding = _extract_embedding_array(embedding_result)

                    result.embedding_dimension = int(embedding.shape[0])

                except Exception as exc:

                    result.add_diagnostic(
                        "embedding",
                        f"Embedding failed: {exc}",
                        level="error",
                        exception=exc,
                        recoverable=False,
                    )

                    if index:
                        raise

            # ----------------------------------------------------------------
            # 14. Build memory
            # ----------------------------------------------------------------

            memory = self._build_memory(
                memory_id=result.memory_id,
                source_name=actual_source_name,
                source_path=source_path,
                sha256=result.sha256,
                mime_type=mime_type,
                file_size=result.file_size,
                ocr_text=ocr_text,
                description=description,
                category=category,
                confidence=confidence,
                entities=entities,
                keywords=keywords,
                intent=intent,
                metadata=metadata_dict,
            )

            result.memory = memory

            # ----------------------------------------------------------------
            # 15. Persist memory
            # ----------------------------------------------------------------

            self._persist_memory(
                memory=memory,
                result=result,
            )

            # ----------------------------------------------------------------
            # 16. FAISS
            # ----------------------------------------------------------------

            if index:

                if embedding is None:
                    raise ProcessingDependencyError(
                        "Cannot index memory without embedding."
                    )

                self._index_vector(
                    memory_id=result.memory_id,
                    embedding=embedding,
                )

                result.indexed = True

            # ----------------------------------------------------------------
            # 17. Final result
            # ----------------------------------------------------------------

            result.description = description
            result.category = category
            result.confidence = confidence
            result.entities = entities
            result.keywords = keywords
            result.intent = intent

            result.success = True

            return result

        except Exception as exc:

            logger.exception(
                "MemoryOS processing failed " "for %s",
                result.source_name,
            )

            result.success = False

            result.add_diagnostic(
                "pipeline",
                f"Processing failed: {exc}",
                level="error",
                exception=exc,
                recoverable=False,
            )

            return result

        finally:

            result.processing_time_ms = (time.perf_counter() - started) * 1000.0

            logger.info(
                "Processing finished: "
                "source=%s "
                "memory_id=%s "
                "success=%s "
                "indexed=%s "
                "time_ms=%.2f",
                result.source_name,
                result.memory_id,
                result.success,
                result.indexed,
                result.processing_time_ms,
            )

    # ========================================================================
    # BATCH
    # ========================================================================

    def process_batch(
        self,
        sources: Iterable[Any],
        *,
        source_names: Sequence[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        index: bool = True,
        allow_partial_success: bool = True,
    ) -> BatchProcessingResult:

        started = time.perf_counter()

        source_list = list(sources)

        batch_result = BatchProcessingResult(
            success=False,
            total=len(source_list),
        )

        for index_number, source in enumerate(source_list):

            if source_names is not None and index_number < len(source_names):

                current_name = source_names[index_number]

            else:

                current_name = _source_name(
                    source,
                    f"upload_{index_number + 1}",
                )

            result = self.process(
                source,
                source_name=current_name,
                metadata=metadata,
                index=index,
                allow_partial_success=(allow_partial_success),
            )

            batch_result.results.append(result)

            if result.success:
                batch_result.successful += 1
            else:
                batch_result.failed += 1
                batch_result.errors.extend(result.errors)

            if result.duplicate:
                batch_result.duplicates += 1

            if result.indexed:
                batch_result.indexed += 1

        batch_result.processing_time_ms = (time.perf_counter() - started) * 1000.0

        batch_result.success = batch_result.failed == 0 or (
            allow_partial_success and batch_result.successful > 0
        )

        logger.info(
            "Batch processing completed: "
            "total=%s "
            "successful=%s "
            "failed=%s "
            "duplicates=%s "
            "indexed=%s",
            batch_result.total,
            batch_result.successful,
            batch_result.failed,
            batch_result.duplicates,
            batch_result.indexed,
        )

        return batch_result

    # ========================================================================
    # DUPLICATES
    # ========================================================================

    def _is_duplicate(
        self,
        memory_id: str,
        sha256: str | None,
        metadata: Mapping[str, Any] | None,
    ) -> bool:

        if metadata:

            if metadata.get("duplicate") is True:

                return True

        vector_store = self.vector_store

        if vector_store is None:
            return False

        try:

            contains = getattr(
                vector_store,
                "contains",
                None,
            )

            if callable(contains):

                return bool(contains(memory_id))

        except Exception as exc:

            logger.warning(
                "FAISS duplicate check failed: %s",
                exc,
            )

        return False

    def _find_existing_memory(
        self,
        memory_id: str,
        sha256: str | None,
    ) -> Any | None:

        service = self.memory_service

        if service is None:
            return None

        for method_name in (
            "get",
            "get_memory",
            "find_by_id",
            "find",
        ):

            method = getattr(
                service,
                method_name,
                None,
            )

            if not callable(method):
                continue

            try:

                return method(memory_id)

            except Exception:
                continue

        return None

    # ========================================================================
    # DESCRIPTION
    # ========================================================================

    @staticmethod
    def _build_fallback_description(
        source_name: str,
        ocr_text: str,
        category: str,
    ) -> str:

        if ocr_text:

            clean_text = " ".join(ocr_text.split())

            if len(clean_text) > 500:
                clean_text = clean_text[:500] + "..."

            return (
                f"{category.title()} screenshot " f"containing text: " f"{clean_text}"
            )

        return f"{category.title()} image " f"named {source_name}"

    # ========================================================================
    # CATEGORY
    # ========================================================================

    @staticmethod
    def _normalize_category(
        category: str,
        ocr_text: str,
        description: str,
    ) -> str:

        normalized = _safe_string(
            category,
            default="other",
        ).lower()

        valid_categories = {
            "receipt",
            "recipe",
            "food",
            "address",
            "product",
            "shopping",
            "document",
            "ticket",
            "travel",
            "finance",
            "message",
            "social",
            "code",
            "education",
            "note",
            "screenshot",
            "other",
        }

        if normalized not in valid_categories:
            normalized = "other"

        if normalized != "other":
            return normalized

        combined = (f"{ocr_text} {description}").lower()

        keyword_categories = {
            "receipt": (
                "total",
                "subtotal",
                "invoice",
                "tax",
                "receipt",
            ),
            "recipe": (
                "ingredients",
                "recipe",
                "instructions",
                "preheat",
            ),
            "address": (
                "street",
                "road",
                "avenue",
                "pincode",
                "pin code",
                "address",
            ),
            "product": (
                "buy now",
                "add to cart",
                "price",
                "₹",
                "$",
                "product",
            ),
            "education": (
                "assignment",
                "exam",
                "question",
                "lecture",
                "college",
            ),
            "code": (
                "python",
                "javascript",
                "typescript",
                "java",
                "function",
                "class ",
                "import ",
            ),
        }

        for candidate, words in keyword_categories.items():

            if any(word in combined for word in words):

                return candidate

        return "other"

    # ========================================================================
    # KEYWORDS
    # ========================================================================

    @staticmethod
    def _keywords_from_text(
        text: str,
    ) -> list[str]:

        if not text:
            return []

        words: list[str] = []

        for token in text.split():

            token = token.strip(".,!?;:\"'()[]{}<>").lower()

            if len(token) < 4:
                continue

            if token.isdigit():
                continue

            if token not in words:
                words.append(token)

            if len(words) >= 30:
                break

        return words

    @staticmethod
    def _merge_keywords(
        primary: Sequence[str],
        secondary: Sequence[str],
    ) -> list[str]:

        result: list[str] = []

        seen: set[str] = set()

        for value in [
            *primary,
            *secondary,
        ]:

            text = _safe_string(value)

            if not text:
                continue

            normalized = text.lower()

            if normalized in seen:
                continue

            seen.add(normalized)

            result.append(text)

        return result[:50]

    # ========================================================================
    # EMBEDDING TEXT
    # ========================================================================

    def _build_embedding_text(
        self,
        *,
        source_name: str,
        category: str,
        description: str,
        ocr_text: str,
        entities: Sequence[Any],
        keywords: Sequence[str],
        intent: str | None,
    ) -> str:

        entity_text: list[str] = []

        for entity in entities:

            if isinstance(
                entity,
                Mapping,
            ):

                value = (
                    entity.get("text")
                    or entity.get("value")
                    or entity.get("name")
                    or entity.get("entity")
                )

                if value:
                    entity_text.append(_safe_string(value))

            else:

                text = _safe_string(entity)

                if text:
                    entity_text.append(text)

        parts = [
            f"File: {source_name}",
            f"Category: {category}",
            f"Description: {description}",
        ]

        if ocr_text:
            parts.append(f"OCR Text: {ocr_text}")

        if entity_text:
            parts.append("Entities: " + ", ".join(entity_text))

        if keywords:
            parts.append("Keywords: " + ", ".join(keywords))

        if intent:
            parts.append(f"Intent: {intent}")

        return "\n".join(parts)[:30_000]

    # ========================================================================
    # MEMORY BUILD
    # ========================================================================

    def _build_memory(
        self,
        *,
        memory_id: str,
        source_name: str,
        source_path: str | None,
        sha256: str | None,
        mime_type: str | None,
        file_size: int | None,
        ocr_text: str,
        description: str,
        category: str,
        confidence: float | None,
        entities: Sequence[Any],
        keywords: Sequence[str],
        intent: str | None,
        metadata: Mapping[str, Any],
    ) -> Any:

        metadata_dict = dict(metadata)

        image_url = _get_value(
            metadata_dict,
            "image_url",
            default=f"/upload/file/{memory_id}",
        )

        preview_url = _get_value(
            metadata_dict,
            "preview_url",
            default=image_url,
        )

        thumbnail_url = _get_value(
            metadata_dict,
            "thumbnail_url",
            default=preview_url,
        )

        payload = {
            "memory_id": memory_id,
            "source_name": source_name,
            "source_path": source_path,
            "sha256": sha256,
            "mime_type": mime_type,
            "file_size": file_size,
            "ocr_text": ocr_text,
            "description": description,
            "category": category,
            "confidence": confidence,
            "entities": _serialize_safe(list(entities)),
            "keywords": list(keywords),
            "intent": intent,
            # Critical metadata for retrieval/UI.
            "metadata": _serialize_safe(
                {
                    **metadata_dict,
                    "memory_id": memory_id,
                    "source_name": source_name,
                    "source_path": source_path,
                    "image_url": image_url,
                    "preview_url": preview_url,
                    "thumbnail_url": thumbnail_url,
                }
            ),
            "image_url": image_url,
            "preview_url": preview_url,
            "thumbnail_url": thumbnail_url,
            "created_at": _iso_now(),
        }

        memory_service = self.memory_service

        if memory_service is not None:

            for method_name in (
                "create_memory",
                "create",
                "save_memory",
                "save",
                "upsert",
            ):

                method = getattr(
                    memory_service,
                    method_name,
                    None,
                )

                if not callable(method):
                    continue

                try:

                    memory = method(payload)

                    if memory is not None:
                        return memory

                except TypeError:

                    try:

                        memory = method(**payload)

                        if memory is not None:
                            return memory

                    except Exception as exc:

                        logger.debug(
                            "MemoryService keyword " "save failed: %s",
                            exc,
                        )

                except Exception as exc:

                    logger.warning(
                        "MemoryService %s failed: %s",
                        method_name,
                        exc,
                    )

        return payload

    # ========================================================================
    # PERSIST
    # ========================================================================

    def _persist_memory(
        self,
        *,
        memory: Any,
        result: ProcessingResult,
    ) -> None:

        if memory is None:
            return

        if self.memory_service is None:

            result.add_diagnostic(
                "memory",
                "MemoryService unavailable; " "memory exists only in memory.",
                level="warning",
            )

            return

        # _build_memory() already calls the service.
        # Avoid duplicate DB writes.
        return

    # ========================================================================
    # FAISS
    # ========================================================================

    def _index_vector(
        self,
        *,
        memory_id: str | None,
        embedding: np.ndarray,
    ) -> None:

        if not memory_id:
            raise ProcessingValidationError("Cannot index memory without memory_id.")

        if self.vector_store is None:
            raise ProcessingDependencyError("VectorStore is unavailable.")

        if embedding.shape != (EMBEDDING_DIMENSION,):

            raise EmbeddingDimensionError(
                "FAISS vector shape mismatch: "
                f"expected "
                f"({EMBEDDING_DIMENSION},), "
                f"received "
                f"{embedding.shape}"
            )

        add_method = getattr(
            self.vector_store,
            "add",
            None,
        )

        if not callable(add_method):

            raise ProcessingDependencyError(
                "VectorStore does not expose " "the required add() method."
            )

        try:

            # A backfill updates an existing memory's representation. Replace
            # its vector under the same ID rather than adding a duplicate.
            contains_method = getattr(self.vector_store, "contains", None)
            remove_method = getattr(self.vector_store, "remove", None)
            if callable(contains_method) and callable(remove_method) and contains_method(memory_id):
                remove_method(memory_id, persist=False)

            add_method(
                memory_id,
                embedding,
                persist=True,
            )

        except TypeError:

            add_method(
                memory_id,
                embedding,
            )

        logger.debug(
            "FAISS indexed memory %s",
            memory_id,
        )

    # ========================================================================
    # HEALTH
    # ========================================================================

    def health(self) -> dict[str, Any]:

        vector_count = None

        if self.vector_store is not None:

            try:

                vector_count = getattr(
                    self.vector_store,
                    "count",
                    None,
                )

                if callable(vector_count):
                    vector_count = vector_count()

            except Exception:
                vector_count = None

        return {
            "service": "processing",
            "status": "healthy",
            "embedding_dimension": (EMBEDDING_DIMENSION),
            "ocr_available": (self.ocr_service is not None),
            "vision_available": (self.vision_service is not None),
            "embedding_available": (self.embedding_service is not None),
            "vector_store_available": (self.vector_store is not None),
            "memory_service_available": (self.memory_service is not None),
            "vector_count": vector_count,
        }


# ============================================================================
# SINGLETON
# ============================================================================

processing_service = ProcessingService()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def process_image(
    source: Any,
    *,
    source_name: str = "upload",
    metadata: Mapping[str, Any] | None = None,
    index: bool = True,
    allow_partial_success: bool = True,
) -> ProcessingResult:

    return processing_service.process(
        source,
        source_name=source_name,
        metadata=metadata,
        index=index,
        allow_partial_success=allow_partial_success,
    )


def process_images(
    sources: Iterable[Any],
    *,
    source_names: Sequence[str] | None = None,
    metadata: Mapping[str, Any] | None = None,
    index: bool = True,
    allow_partial_success: bool = True,
) -> BatchProcessingResult:

    return processing_service.process_batch(
        sources,
        source_names=source_names,
        metadata=metadata,
        index=index,
        allow_partial_success=allow_partial_success,
    )


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "ProcessingError",
    "ProcessingValidationError",
    "ProcessingDependencyError",
    "EmbeddingDimensionError",
    "ProcessingDiagnostic",
    "ProcessingResult",
    "BatchProcessingResult",
    "ProcessingService",
    "processing_service",
    "process_image",
    "process_images",
    "EMBEDDING_DIMENSION",
]
