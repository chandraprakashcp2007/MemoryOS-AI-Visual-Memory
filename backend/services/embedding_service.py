"""
MemoryOS — Processing Service
=============================

Single production-grade ingestion orchestrator.

Pipeline
--------
IMAGE/PATH
   |
   +--> OCR
   |
   +--> Vision / Gemini
   |
   +--> Memory normalization
   |
   +--> Embedding
   |
   +--> FAISS vector index
   |
   +--> Optional persistence

Design goals
------------
- Never corrupt the pipeline because one optional service fails.
- Accept str, Path, bytes and file-like image sources.
- Always normalize filesystem paths to pathlib.Path.
- Never assume optional service methods exist.
- Never assume vector_store.count is callable.
- Never import a non-existent singleton from vector_store.py.
- Work with the existing MemoryOS service implementations.
- Support partial-success processing.
- Produce deterministic memory IDs.
- Return structured diagnostics.
- Keep compatibility with the current MemoryOS codebase.

Schema version: 5.0.0
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import mimetypes
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "5.0.0"


# ============================================================================
# Generic helpers
# ============================================================================

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso() -> str:
    return _utc_now().isoformat()


def _safe_str(value: Any, default: str = "") -> str:
    if value is None:
        return default

    try:
        return str(value)
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _is_path_like(value: Any) -> bool:
    return isinstance(value, (str, Path))


def _normalize_source(source: Any) -> Any:
    """
    Convert filesystem strings into Path objects.

    This is critical because several MemoryOS image utilities intentionally
    accept Path but do not accept raw strings.
    """
    if isinstance(source, Path):
        return source

    if isinstance(source, str):
        return Path(source)

    return source


def _source_name(source: Any, explicit: str | None = None) -> str:
    if explicit:
        return explicit

    if isinstance(source, Path):
        return source.name

    if isinstance(source, str):
        return Path(source).name

    name = getattr(source, "name", None)

    if name:
        try:
            return Path(str(name)).name
        except Exception:
            return str(name)

    return "memory"


def _source_type(source: Any) -> str:
    if isinstance(source, Path):
        return "path"

    if isinstance(source, str):
        return "path"

    if isinstance(source, (bytes, bytearray, memoryview)):
        return "bytes"

    if hasattr(source, "read"):
        return "file"

    return type(source).__name__


def _read_source_bytes(source: Any) -> bytes:
    """
    Read an image source without changing the original stream unexpectedly.
    """
    if isinstance(source, Path):
        return source.read_bytes()

    if isinstance(source, str):
        return Path(source).read_bytes()

    if isinstance(source, bytes):
        return source

    if isinstance(source, bytearray):
        return bytes(source)

    if isinstance(source, memoryview):
        return source.tobytes()

    if hasattr(source, "read"):
        stream = source

        position = None

        try:
            position = stream.tell()
        except Exception:
            position = None

        try:
            data = stream.read()
        finally:
            if position is not None:
                try:
                    stream.seek(position)
                except Exception:
                    pass

        if isinstance(data, bytes):
            return data

        if isinstance(data, bytearray):
            return bytes(data)

        if isinstance(data, memoryview):
            return data.tobytes()

        raise TypeError("File-like image source did not return bytes.")

    raise TypeError(
        "Unsupported image source. "
        "Expected bytes, file-like object, str, or Path."
    )


def _sha256(source: Any) -> str:
    data = _read_source_bytes(source)
    return hashlib.sha256(data).hexdigest()


def _mime_type(source: Any, source_name: str) -> str:
    if isinstance(source, Path):
        guessed, _ = mimetypes.guess_type(str(source))
    else:
        guessed, _ = mimetypes.guess_type(source_name)

    if guessed:
        return guessed

    return "application/octet-stream"


def _object_to_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return dict(value)

    if hasattr(value, "to_dict"):
        try:
            result = value.to_dict()
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    if hasattr(value, "model_dump"):
        try:
            result = value.model_dump()
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    if hasattr(value, "dict"):
        try:
            result = value.dict()
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    if hasattr(value, "__dataclass_fields__"):
        try:
            result = asdict(value)
            if isinstance(result, dict):
                return result
        except Exception:
            pass

    result: dict[str, Any] = {}

    for name in (
        "text",
        "ocr_text",
        "description",
        "title",
        "summary",
        "category",
        "subcategory",
        "intent",
        "keywords",
        "entities",
        "people",
        "organizations",
        "locations",
        "products",
        "dates",
        "prices",
        "emails",
        "phone_numbers",
        "urls",
        "importance",
        "confidence",
        "model",
        "fallback",
        "error",
    ):
        if hasattr(value, name):
            try:
                result[name] = getattr(value, name)
            except Exception:
                pass

    return result


def _extract_text(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, dict):
        for key in (
            "text",
            "ocr_text",
            "content",
            "extracted_text",
            "full_text",
        ):
            candidate = value.get(key)

            if candidate:
                return _safe_str(candidate).strip()

        return ""

    for key in (
        "text",
        "ocr_text",
        "content",
        "extracted_text",
        "full_text",
    ):
        if hasattr(value, key):
            try:
                candidate = getattr(value, key)
                if candidate:
                    return _safe_str(candidate).strip()
            except Exception:
                pass

    return ""


def _call_safely(
    obj: Any,
    method_name: str,
    *args: Any,
    **kwargs: Any,
) -> tuple[bool, Any, str | None]:
    """
    Call an optional service method safely.

    Returns:
        (available_and_called, result, error)
    """
    method = getattr(obj, method_name, None)

    if method is None or not callable(method):
        return False, None, None

    try:
        return True, method(*args, **kwargs), None
    except TypeError as exc:
        return True, None, f"TypeError: {exc}"
    except Exception as exc:
        logger.exception(
            "Processing service call failed: %s.%s",
            type(obj).__name__,
            method_name,
        )
        return True, None, f"{type(exc).__name__}: {exc}"


def _first_available_method(obj: Any, names: tuple[str, ...]) -> str | None:
    for name in names:
        method = getattr(obj, name, None)
        if callable(method):
            return name

    return None


# ============================================================================
# Result models
# ============================================================================

@dataclass
class ProcessingDiagnostics:
    processing_id: str
    schema_version: str
    started_at: str
    completed_at: str | None = None

    source_name: str = ""
    source_type: str = ""

    memory_id: str = ""

    success: bool = False
    partial_success: bool = False

    ocr_attempted: bool = False
    ocr_success: bool = False

    vision_attempted: bool = False
    vision_success: bool = False

    embedding_attempted: bool = False
    embedding_success: bool = False

    vector_index_attempted: bool = False
    vector_index_success: bool = False

    memory_persist_attempted: bool = False
    memory_persist_success: bool = False

    embedding_dimension: int = 0
    ocr_character_count: int = 0

    processing_time_ms: float = 0.0

    stages: dict[str, float] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessingResult:
    success: bool
    memory_id: str

    ocr_text: str = ""
    indexed: bool = False
    persisted: bool = False

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    vision: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    embedding: dict[str, Any] = field(default_factory=dict)

    diagnostics: ProcessingDiagnostics | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "memory_id": self.memory_id,
            "ocr_text": self.ocr_text,
            "indexed": self.indexed,
            "persisted": self.persisted,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "vision": self.vision,
            "memory": self.memory,
            "embedding": self.embedding,
            "diagnostics": (
                asdict(self.diagnostics)
                if self.diagnostics is not None
                else {}
            ),
        }


# ============================================================================
# Vector adapter
# ============================================================================

class _VectorStoreAdapter:
    """
    Compatibility adapter around the existing MemoryOS VectorStore.

    Important:
    Some versions expose:
        store.count()

    Others expose:
        store.count

    This adapter supports both.
    """

    def __init__(self, store: Any = None):
        self.store = store

    def _resolve_count(self) -> int:
        if self.store is None:
            return 0

        value = getattr(self.store, "count", None)

        if value is None:
            value = getattr(self.store, "vector_count", 0)

        if callable(value):
            try:
                value = value()
            except TypeError:
                return 0
            except Exception:
                logger.exception("Vector-store count failed.")
                return 0

        return _safe_int(value, 0)

    def count(self) -> int:
        return self._resolve_count()

    @property
    def vector_count(self) -> int:
        return self._resolve_count()

    def add(
        self,
        embedding: Any,
        memory_id: str,
        **kwargs: Any,
    ) -> Any:
        if self.store is None:
            raise RuntimeError("Vector store is unavailable.")

        # Most common method names first.
        candidates = (
            "add",
            "add_vector",
            "add_embedding",
            "index",
            "upsert",
        )

        for name in candidates:
            method = getattr(self.store, name, None)

            if not callable(method):
                continue

            attempts = (
                lambda: method(embedding, memory_id, **kwargs),
                lambda: method(
                    embedding=embedding,
                    memory_id=memory_id,
                    **kwargs,
                ),
                lambda: method(
                    vector=embedding,
                    memory_id=memory_id,
                    **kwargs,
                ),
                lambda: method(
                    embedding,
                    memory_id,
                ),
                lambda: method(
                    embedding=embedding,
                    memory_id=memory_id,
                ),
            )

            last_error: Exception | None = None

            for attempt in attempts:
                try:
                    return attempt()
                except TypeError as exc:
                    last_error = exc
                    continue

            if last_error is not None:
                logger.debug(
                    "Vector method %s signature mismatch: %s",
                    name,
                    last_error,
                )

        raise RuntimeError(
            "No compatible vector-store insertion method found."
        )

    def search(
        self,
        embedding: Any,
        top_k: int = 10,
        **kwargs: Any,
    ) -> Any:
        if self.store is None:
            return []

        for name in (
            "search",
            "similarity_search",
            "query",
        ):
            method = getattr(self.store, name, None)

            if not callable(method):
                continue

            attempts = (
                lambda: method(embedding, top_k=top_k, **kwargs),
                lambda: method(
                    embedding=embedding,
                    top_k=top_k,
                    **kwargs,
                ),
                lambda: method(
                    vector=embedding,
                    top_k=top_k,
                    **kwargs,
                ),
                lambda: method(embedding, top_k),
            )

            for attempt in attempts:
                try:
                    return attempt()
                except TypeError:
                    continue
                except Exception:
                    logger.exception(
                        "Vector search failed using %s",
                        name,
                    )
                    return []

        return []

    def health(self) -> dict[str, Any]:
        if self.store is None:
            return {
                "healthy": False,
                "available": False,
                "provider": None,
                "vector_count": 0,
            }

        stats = None

        for name in ("stats", "get_stats", "statistics"):
            attr = getattr(self.store, name, None)

            if attr is None:
                continue

            try:
                stats = attr() if callable(attr) else attr
                break
            except Exception:
                continue

        return {
            "healthy": True,
            "available": True,
            "provider": type(self.store).__name__,
            "stats": _safe_str(stats) if stats is not None else None,
            "vector_count": self.count(),
        }


# ============================================================================
# Processing Service
# ============================================================================

class ProcessingService:
    """
    Main MemoryOS ingestion service.
    """

    def __init__(
        self,
        ocr_service: Any = None,
        vision_service: Any = None,
        embedding_service: Any = None,
        vector_store: Any = None,
        memory_service: Any = None,
    ):
        self.ocr = ocr_service or self._load_ocr()
        self.vision = vision_service or self._load_vision()
        self.embedding = embedding_service or self._load_embedding()
        self.memory_service = memory_service or self._load_memory()

        underlying_vector_store = (
            vector_store
            if vector_store is not None
            else self._load_vector_store()
        )

        self.vector_store = _VectorStoreAdapter(
            underlying_vector_store
        )

    # ---------------------------------------------------------------------
    # Service discovery
    # ---------------------------------------------------------------------

    @staticmethod
    def _load_ocr() -> Any:
        try:
            from backend.services.ocr_service import ocr_service

            return ocr_service
        except Exception:
            logger.exception("Unable to load OCR service.")
            return None

    @staticmethod
    def _load_vision() -> Any:
        try:
            from backend.services.vision_service import vision_service

            return vision_service
        except Exception:
            logger.exception("Unable to load vision service.")
            return None

    @staticmethod
    def _load_embedding() -> Any:
        try:
            from backend.services.embedding_service import (
                embedding_service,
            )

            return embedding_service
        except Exception:
            logger.exception("Unable to load embedding service.")
            return None

    @staticmethod
    def _load_vector_store() -> Any:
        """
        Import VectorStore class safely.

        DO NOT import `vector_store` singleton because some project versions
        expose only the class.
        """
        try:
            from backend.services.vector_store import VectorStore

            # First try normal construction.
            try:
                return VectorStore()
            except TypeError:
                pass

            # Some implementations may expose a class-level singleton.
            candidate = getattr(VectorStore, "instance", None)

            if candidate is not None:
                return candidate

            raise RuntimeError(
                "VectorStore could not be initialized."
            )

        except Exception:
            logger.exception("Unable to load VectorStore.")
            return None

    @staticmethod
    def _load_memory() -> Any:
        try:
            from backend.services.memory_service import memory_service

            return memory_service
        except Exception:
            logger.exception("Unable to load MemoryService.")
            return None

    # ---------------------------------------------------------------------
    # Health
    # ---------------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        embedding_available = self.embedding is not None
        ocr_available = self.ocr is not None
        vision_available = self.vision is not None

        embedding_healthy = embedding_available
        ocr_healthy = ocr_available

        if self.ocr is not None:
            available_attr = getattr(self.ocr, "available", None)

            try:
                if callable(available_attr):
                    ocr_healthy = bool(available_attr())
                elif available_attr is not None:
                    ocr_healthy = bool(available_attr)
            except Exception:
                ocr_healthy = False

        vector_health = self.vector_store.health()

        overall = bool(
            embedding_available
            and vector_health.get("available", False)
        )

        return {
            "service": "MemoryOS ProcessingService",
            "healthy": overall,
            "schema_version": SCHEMA_VERSION,
            "embedding": {
                "healthy": embedding_healthy,
                "available": embedding_available,
                "provider": (
                    type(self.embedding).__name__
                    if self.embedding
                    else None
                ),
            },
            "ocr": {
                "healthy": ocr_healthy,
                "available": ocr_available,
                "provider": (
                    type(self.ocr).__name__
                    if self.ocr
                    else None
                ),
            },
            "vision": {
                "available": vision_available,
                "provider": (
                    type(self.vision).__name__
                    if self.vision
                    else None
                ),
            },
            "vector_store": vector_health,
            "vector_count": self.vector_store.count(),
        }

    # ---------------------------------------------------------------------
    # Memory ID
    # ---------------------------------------------------------------------

    @staticmethod
    def _memory_id(
        source: Any,
        source_name: str,
        source_hash: str | None = None,
    ) -> str:
        if source_hash:
            digest = source_hash
        else:
            try:
                digest = _sha256(source)
            except Exception:
                digest = hashlib.sha256(
                    (
                        f"{source_name}|"
                        f"{_source_type(source)}"
                    ).encode("utf-8")
                ).hexdigest()

        return f"memory_{digest[:32]}"

    # ---------------------------------------------------------------------
    # OCR
    # ---------------------------------------------------------------------

    def _run_ocr(
        self,
        source: Any,
        diagnostics: ProcessingDiagnostics,
    ) -> tuple[str, dict[str, Any]]:
        diagnostics.ocr_attempted = True

        if self.ocr is None:
            diagnostics.errors.append(
                "OCR service is unavailable."
            )
            return "", {}

        # IMPORTANT:
        # OCRService.extract_text accepts the image source. Pass a Path,
        # never the raw path string.
        normalized_source = _normalize_source(source)

        method = getattr(self.ocr, "extract_text", None)

        if not callable(method):
            diagnostics.errors.append(
                "OCR service has no compatible extract_text method."
            )
            return "", {}

        started = time.perf_counter()

        attempts = (
            lambda: method(normalized_source),
            lambda: method(source=normalized_source),
            lambda: method(image=normalized_source),
            lambda: method(path=normalized_source),
        )

        last_error: str | None = None
        result: Any = None

        for attempt in attempts:
            try:
                result = attempt()
                last_error = None
                break
            except TypeError as exc:
                last_error = f"TypeError: {exc}"
                continue
            except Exception as exc:
                last_error = (
                    f"{type(exc).__name__}: {exc}"
                )
                logger.exception("OCR extraction failed.")
                break

        diagnostics.stages["ocr"] = (
            time.perf_counter() - started
        ) * 1000.0

        if last_error is not None:
            diagnostics.ocr_success = False
            diagnostics.errors.append(
                f"OCR extraction failed: {last_error}"
            )
            return "", {}

        text = _extract_text(result)

        diagnostics.ocr_success = True
        diagnostics.ocr_character_count = len(text)

        return text, _object_to_dict(result)

    # ---------------------------------------------------------------------
    # Vision
    # ---------------------------------------------------------------------

    def _run_vision(
        self,
        source: Any,
        ocr_text: str,
        diagnostics: ProcessingDiagnostics,
    ) -> tuple[dict[str, Any], Any]:
        diagnostics.vision_attempted = True

        if self.vision is None:
            diagnostics.errors.append(
                "Vision service is unavailable."
            )
            return {}, None

        normalized_source = _normalize_source(source)

        started = time.perf_counter()

        result: Any = None
        last_error: str | None = None

        # Prefer OCR-aware analysis.
        methods = (
            "analyze_with_ocr",
            "analyze",
            "analyze_image",
        )

        for method_name in methods:
            method = getattr(self.vision, method_name, None)

            if not callable(method):
                continue

            attempts = []

            if method_name == "analyze_with_ocr":
                attempts = [
                    lambda: method(
                        normalized_source,
                        ocr_text=ocr_text,
                    ),
                    lambda: method(
                        source=normalized_source,
                        ocr_text=ocr_text,
                    ),
                    lambda: method(
                        normalized_source,
                        ocr_text,
                    ),
                ]
            else:
                attempts = [
                    lambda: method(normalized_source),
                    lambda: method(source=normalized_source),
                    lambda: method(
                        normalized_source,
                        ocr_text=ocr_text,
                    ),
                    lambda: method(
                        source=normalized_source,
                        ocr_text=ocr_text,
                    ),
                ]

            for attempt in attempts:
                try:
                    result = attempt()
                    last_error = None
                    break
                except TypeError as exc:
                    last_error = f"TypeError: {exc}"
                    continue
                except Exception as exc:
                    last_error = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    logger.exception(
                        "Vision analysis failed."
                    )
                    break

            if result is not None:
                break

        diagnostics.stages["vision"] = (
            time.perf_counter() - started
        ) * 1000.0

        if result is None:
            diagnostics.vision_success = False

            if last_error:
                diagnostics.errors.append(
                    f"Vision analysis failed: {last_error}"
                )

            return {}, None

        diagnostics.vision_success = True

        return _object_to_dict(result), result

    # ---------------------------------------------------------------------
    # Memory construction
    # ---------------------------------------------------------------------

    def _build_memory(
        self,
        source: Any,
        source_name: str,
        memory_id: str,
        ocr_text: str,
        vision_dict: dict[str, Any],
        vision_object: Any,
        source_hash: str | None,
    ) -> dict[str, Any]:
        if self.memory_service is None:
            return {
                "memory_id": memory_id,
                "source_name": source_name,
                "ocr_text": ocr_text,
                "vision": vision_dict,
                "sha256": source_hash or "",
            }

        service = self.memory_service

        method = getattr(service, "build_memory", None)

        if not callable(method):
            return {
                "memory_id": memory_id,
                "source_name": source_name,
                "ocr_text": ocr_text,
                "vision": vision_dict,
                "sha256": source_hash or "",
            }

        metadata = {
            "memory_id": memory_id,
            "source_name": source_name,
            "source_type": _source_type(source),
            "sha256": source_hash or "",
            "mime_type": _mime_type(
                source,
                source_name,
            ),
        }

        # Build a broad normalized context object.
        context = {
            "memory_id": memory_id,
            "source_name": source_name,
            "ocr_text": ocr_text,
            "ocr": ocr_text,
            "vision": vision_object,
            "vision_result": vision_object,
            "vision_data": vision_dict,
            "metadata": metadata,
        }

        attempts = [
            lambda: method(**context),
            lambda: method(
                memory_id=memory_id,
                source_name=source_name,
                ocr_text=ocr_text,
                vision=vision_object,
                metadata=metadata,
            ),
            lambda: method(
                memory_id=memory_id,
                ocr_text=ocr_text,
                vision=vision_object,
            ),
        ]

        for attempt in attempts:
            try:
                result = attempt()
                return _object_to_dict(result)
            except TypeError:
                continue
            except Exception as exc:
                logger.exception(
                    "Memory construction failed."
                )
                raise RuntimeError(
                    f"Memory construction failed: {exc}"
                ) from exc

        # If build_memory exists but signatures are incompatible,
        # return a safe fallback instead of crashing the entire pipeline.
        return {
            "memory_id": memory_id,
            "source_name": source_name,
            "ocr_text": ocr_text,
            "vision": vision_dict,
            "metadata": metadata,
        }

    # ---------------------------------------------------------------------
    # Embedding text
    # ---------------------------------------------------------------------

    def _embedding_text(
        self,
        memory: dict[str, Any],
        ocr_text: str,
        vision: dict[str, Any],
        source_name: str,
    ) -> str:
        if self.memory_service is not None:
            builder = getattr(
                self.memory_service,
                "build_embedding_text",
                None,
            )

            if callable(builder):
                attempts = [
                    lambda: builder(memory),
                    lambda: builder(
                        memory=memory,
                    ),
                ]

                for attempt in attempts:
                    try:
                        value = attempt()

                        if value:
                            return _safe_str(value).strip()

                    except TypeError:
                        continue
                    except Exception:
                        logger.exception(
                            "Memory embedding-text builder failed."
                        )
                        break

        pieces: list[str] = []

        pieces.append(source_name)

        if ocr_text:
            pieces.append(ocr_text)

        for key in (
            "title",
            "summary",
            "description",
            "visual_description",
            "category",
            "subcategory",
            "intent",
        ):
            value = vision.get(key)

            if value:
                pieces.append(_safe_str(value))

        for key in (
            "keywords",
            "entities",
            "people",
            "organizations",
            "locations",
            "products",
            "dates",
            "prices",
        ):
            value = vision.get(key)

            if not value:
                continue

            if isinstance(value, (list, tuple, set)):
                pieces.extend(
                    _safe_str(item)
                    for item in value
                    if item
                )
            else:
                pieces.append(_safe_str(value))

        # Avoid creating an empty embedding input.
        text = " ".join(
            piece.strip()
            for piece in pieces
            if piece and piece.strip()
        )

        return text[:20000]

    # ---------------------------------------------------------------------
    # Embedding
    # ---------------------------------------------------------------------

    def _run_embedding(
        self,
        text: str,
        diagnostics: ProcessingDiagnostics,
    ) -> tuple[Any, dict[str, Any]]:
        diagnostics.embedding_attempted = True

        if self.embedding is None:
            diagnostics.errors.append(
                "Embedding service is unavailable."
            )
            return None, {}

        if not text.strip():
            diagnostics.errors.append(
                "Cannot create embedding from empty text."
            )
            return None, {}

        started = time.perf_counter()

        candidates = (
            "encode",
            "embed",
            "embed_text",
            "create_embedding",
        )

        result: Any = None
        last_error: str | None = None

        for method_name in candidates:
            method = getattr(
                self.embedding,
                method_name,
                None,
            )

            if not callable(method):
                continue

            attempts = [
                lambda: method(text),
                lambda: method(texts=[text]),
                lambda: method([text]),
                lambda: method(text=text),
            ]

            for attempt in attempts:
                try:
                    result = attempt()
                    last_error = None
                    break
                except TypeError as exc:
                    last_error = f"TypeError: {exc}"
                    continue
                except Exception as exc:
                    last_error = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    logger.exception(
                        "Embedding generation failed."
                    )
                    break

            if result is not None:
                break

        diagnostics.stages["embedding"] = (
            time.perf_counter() - started
        ) * 1000.0

        if result is None:
            diagnostics.embedding_success = False

            diagnostics.errors.append(
                "Embedding generation failed"
                + (
                    f": {last_error}"
                    if last_error
                    else "."
                )
            )

            return None, {}

        # Normalize numpy/list result.
        embedding_vector = result

        try:
            # sentence-transformers usually returns numpy.ndarray.
            if hasattr(result, "tolist"):
                converted = result.tolist()

                # Single vector.
                if (
                    isinstance(converted, list)
                    and converted
                    and isinstance(converted[0], (int, float))
                ):
                    embedding_vector = converted

                # Batch of one.
                elif (
                    isinstance(converted, list)
                    and len(converted) == 1
                    and isinstance(converted[0], list)
                ):
                    embedding_vector = converted[0]
        except Exception:
            pass

        if (
            isinstance(embedding_vector, tuple)
        ):
            embedding_vector = list(embedding_vector)

        if (
            isinstance(embedding_vector, list)
            and embedding_vector
            and isinstance(
                embedding_vector[0],
                (list, tuple),
            )
        ):
            embedding_vector = list(
                embedding_vector[0]
            )

        try:
            dimension = len(embedding_vector)
        except Exception:
            dimension = 0

        diagnostics.embedding_dimension = dimension
        diagnostics.embedding_success = dimension > 0

        if dimension <= 0:
            diagnostics.errors.append(
                "Embedding service returned an empty vector."
            )
            return None, {}

        return embedding_vector, {
            "dimension": dimension,
            "provider": type(self.embedding).__name__,
        }

    # ---------------------------------------------------------------------
    # Vector indexing
    # ---------------------------------------------------------------------

    def _index_vector(
        self,
        embedding: Any,
        memory_id: str,
        diagnostics: ProcessingDiagnostics,
    ) -> bool:
        diagnostics.vector_index_attempted = True

        if embedding is None:
            diagnostics.errors.append(
                "Vector indexing skipped because embedding is missing."
            )
            return False

        if self.vector_store.store is None:
            diagnostics.errors.append(
                "Vector store is unavailable."
            )
            return False

        started = time.perf_counter()

        try:
            self.vector_store.add(
                embedding,
                memory_id,
            )

            diagnostics.vector_index_success = True

            return True

        except Exception as exc:
            diagnostics.vector_index_success = False

            diagnostics.errors.append(
                f"Vector indexing failed: "
                f"{type(exc).__name__}: {exc}"
            )

            logger.exception(
                "Vector indexing failed for %s",
                memory_id,
            )

            return False

        finally:
            diagnostics.stages["vector_index"] = (
                time.perf_counter() - started
            ) * 1000.0

    # ---------------------------------------------------------------------
    # Persistence
    # ---------------------------------------------------------------------

    def _persist_memory(
        self,
        memory: dict[str, Any],
        diagnostics: ProcessingDiagnostics,
    ) -> bool:
        diagnostics.memory_persist_attempted = True

        if self.memory_service is None:
            diagnostics.errors.append(
                "MemoryService is unavailable."
            )
            return False

        service = self.memory_service

        # Current MemoryService may intentionally be a pure builder.
        # Support persistence if a persistence method exists.
        candidates = (
            "create",
            "save",
            "persist",
            "store",
            "save_memory",
            "create_memory",
            "persist_memory",
            "insert",
        )

        started = time.perf_counter()

        found_method = False

        for method_name in candidates:
            method = getattr(service, method_name, None)

            if not callable(method):
                continue

            found_method = True

            attempts = [
                lambda: method(memory),
                lambda: method(**memory),
                lambda: method(
                    memory=memory,
                ),
            ]

            for attempt in attempts:
                try:
                    attempt()

                    diagnostics.memory_persist_success = True

                    return True

                except TypeError:
                    continue

                except Exception as exc:
                    logger.exception(
                        "Memory persistence failed."
                    )

                    diagnostics.errors.append(
                        f"Memory persistence failed: "
                        f"{type(exc).__name__}: {exc}"
                    )

                    return False

        diagnostics.stages["memory_persistence"] = (
            time.perf_counter() - started
        ) * 1000.0

        if not found_method:
            # This is NOT fatal for indexing.
            diagnostics.warnings.append(
                "MemoryService has no persistence method; "
                "memory remains available to the processing result."
            )

        return False

    # ---------------------------------------------------------------------
    # Main processing pipeline
    # ---------------------------------------------------------------------

    def process(
        self,
        source: Any,
        *,
        source_name: str | None = None,
        index: bool = True,
        persist: bool = True,
        allow_partial_success: bool = True,
    ) -> ProcessingResult:
        processing_id = uuid.uuid4().hex
        started_at = _utc_now()
        timer = time.perf_counter()

        normalized_source = _normalize_source(source)
        name = _source_name(
            normalized_source,
            source_name,
        )

        diagnostics = ProcessingDiagnostics(
            processing_id=processing_id,
            schema_version=SCHEMA_VERSION,
            started_at=started_at.isoformat(),
            source_name=name,
            source_type=_source_type(normalized_source),
        )

        errors: list[str] = []
        warnings: list[str] = []

        # Validate path early.
        if isinstance(normalized_source, Path):
            if not normalized_source.exists():
                message = (
                    f"Image source does not exist: "
                    f"{normalized_source}"
                )

                diagnostics.errors.append(message)
                diagnostics.completed_at = _utc_iso()
                diagnostics.processing_time_ms = (
                    time.perf_counter() - timer
                ) * 1000.0

                return ProcessingResult(
                    success=False,
                    memory_id="",
                    errors=[message],
                    diagnostics=diagnostics,
                )

            if not normalized_source.is_file():
                message = (
                    f"Image source is not a file: "
                    f"{normalized_source}"
                )

                diagnostics.errors.append(message)
                diagnostics.completed_at = _utc_iso()
                diagnostics.processing_time_ms = (
                    time.perf_counter() - timer
                ) * 1000.0

                return ProcessingResult(
                    success=False,
                    memory_id="",
                    errors=[message],
                    diagnostics=diagnostics,
                )

        # -----------------------------------------------------------------
        # Hash
        # -----------------------------------------------------------------

        source_hash: str | None = None

        try:
            source_hash = _sha256(normalized_source)
        except Exception as exc:
            warnings.append(
                f"Unable to calculate SHA-256: "
                f"{type(exc).__name__}: {exc}"
            )

        memory_id = self._memory_id(
            normalized_source,
            name,
            source_hash,
        )

        diagnostics.memory_id = memory_id

        # -----------------------------------------------------------------
        # OCR
        # -----------------------------------------------------------------

        ocr_text, ocr_data = self._run_ocr(
            normalized_source,
            diagnostics,
        )

        # -----------------------------------------------------------------
        # Vision
        # -----------------------------------------------------------------

        vision_data, vision_object = self._run_vision(
            normalized_source,
            ocr_text,
            diagnostics,
        )

        # -----------------------------------------------------------------
        # Memory
        # -----------------------------------------------------------------

        try:
            memory = self._build_memory(
                normalized_source,
                name,
                memory_id,
                ocr_text,
                vision_data,
                vision_object,
                source_hash,
            )
        except Exception as exc:
            message = (
                f"Memory construction failed: "
                f"{type(exc).__name__}: {exc}"
            )

            errors.append(message)
            diagnostics.errors.append(message)

            memory = {
                "memory_id": memory_id,
                "source_name": name,
                "ocr_text": ocr_text,
                "vision": vision_data,
                "sha256": source_hash or "",
            }

        # Always ensure critical fields exist.
        memory.setdefault(
            "memory_id",
            memory_id,
        )
        memory.setdefault(
            "source_name",
            name,
        )
        memory.setdefault(
            "ocr_text",
            ocr_text,
        )

        # -----------------------------------------------------------------
        # Embedding text
        # -----------------------------------------------------------------

        embedding_text_started = time.perf_counter()

        embedding_text = self._embedding_text(
            memory,
            ocr_text,
            vision_data,
            name,
        )

        diagnostics.stages["embedding_text"] = (
            time.perf_counter()
            - embedding_text_started
        ) * 1000.0

        # -----------------------------------------------------------------
        # Embedding
        # -----------------------------------------------------------------

        embedding, embedding_info = self._run_embedding(
            embedding_text,
            diagnostics,
        )

        # -----------------------------------------------------------------
        # Index
        # -----------------------------------------------------------------

        indexed = False

        if index and embedding is not None:
            indexed = self._index_vector(
                embedding,
                memory_id,
                diagnostics,
            )

        # -----------------------------------------------------------------
        # Persistence
        # -----------------------------------------------------------------

        persisted = False

        if persist:
            persisted = self._persist_memory(
                memory,
                diagnostics,
            )

        # -----------------------------------------------------------------
        # Final status
        # -----------------------------------------------------------------

        fatal_stage_failure = (
            not diagnostics.embedding_success
            or (
                index
                and not diagnostics.vector_index_success
            )
        )

        if allow_partial_success:
            success = bool(
                diagnostics.embedding_success
                or diagnostics.ocr_success
                or diagnostics.vision_success
            )

        else:
            success = not fatal_stage_failure

        # OCR and persistence are optional in this architecture.
        if not diagnostics.ocr_success:
            warnings.append(
                "OCR did not produce text."
            )

        if diagnostics.vision_attempted and not diagnostics.vision_success:
            warnings.append(
                "Vision analysis did not produce a successful result."
            )

        if persist and not persisted:
            warnings.append(
                "Memory persistence is unavailable or was skipped."
            )

        # Combine unique errors.
        combined_errors: list[str] = []

        for item in (
            diagnostics.errors
            + errors
        ):
            if item and item not in combined_errors:
                combined_errors.append(item)

        combined_warnings: list[str] = []

        for item in (
            diagnostics.warnings
            + warnings
        ):
            if item and item not in combined_warnings:
                combined_warnings.append(item)

        diagnostics.errors = combined_errors
        diagnostics.warnings = combined_warnings
        diagnostics.success = success
        diagnostics.partial_success = (
            success
            and bool(
                combined_errors
                or combined_warnings
                or not diagnostics.ocr_success
                or (persist and not persisted)
            )
        )
        diagnostics.completed_at = _utc_iso()
        diagnostics.processing_time_ms = (
            time.perf_counter() - timer
        ) * 1000.0

        vision_public = dict(vision_data)

        if not vision_public:
            vision_public = {
                "description": "",
                "category": "other",
                "intent": "unknown",
                "title": "",
                "entities": [],
                "keywords": [],
                "raw": {},
            }

        return ProcessingResult(
            success=success,
            memory_id=memory_id,
            ocr_text=ocr_text,
            indexed=indexed,
            persisted=persisted,
            errors=combined_errors,
            warnings=combined_warnings,
            vision=vision_public,
            memory=memory,
            embedding=embedding_info,
            diagnostics=diagnostics,
        )

    # ---------------------------------------------------------------------
    # Batch
    # ---------------------------------------------------------------------

    def process_batch(
        self,
        sources: list[Any] | tuple[Any, ...],
        *,
        index: bool = True,
        persist: bool = True,
        allow_partial_success: bool = True,
    ) -> list[ProcessingResult]:
        results: list[ProcessingResult] = []

        for source in sources:
            try:
                result = self.process(
                    source,
                    index=index,
                    persist=persist,
                    allow_partial_success=allow_partial_success,
                )
            except Exception as exc:
                logger.exception(
                    "Unexpected batch processing failure."
                )

                diagnostics = ProcessingDiagnostics(
                    processing_id=uuid.uuid4().hex,
                    schema_version=SCHEMA_VERSION,
                    started_at=_utc_iso(),
                    source_name=_source_name(source),
                    source_type=_source_type(source),
                    success=False,
                )

                diagnostics.errors.append(
                    f"Unexpected processing failure: "
                    f"{type(exc).__name__}: {exc}"
                )

                diagnostics.completed_at = _utc_iso()

                result = ProcessingResult(
                    success=False,
                    memory_id="",
                    errors=list(diagnostics.errors),
                    diagnostics=diagnostics,
                )

            results.append(result)

        return results

    # ---------------------------------------------------------------------
    # Convenience
    # ---------------------------------------------------------------------

    def count(self) -> int:
        return self.vector_store.count()


# ============================================================================
# Singleton
# ============================================================================

processing_service = ProcessingService()


__all__ = [
    "ProcessingService",
    "ProcessingResult",
    "ProcessingDiagnostics",
    "processing_service",
]