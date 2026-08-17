"""
MemoryOS — Premium OCR API
===========================

Production-grade REST API for MemoryOS local OCR.

Architecture
------------

    Uploaded Image
          │
          ▼
    /ocr/extract/{memory_id}
          │
          ▼
    Uploaded Image Resolver
          │
          ▼
    OCRService
          │
          ├── Image preprocessing
          ├── Tesseract OCR
          ├── Multiple OCR strategies
          ├── Confidence estimation
          ├── Word extraction
          └── OCR text cleaning
          │
          ▼
    Structured OCR Response


Supported OCR strategies
------------------------

    auto
    single_block
    sparse
    single_line


Endpoints
---------

GET  /ocr/status
GET  /ocr/strategies
GET  /ocr/extract/{memory_id}
POST /ocr/extract/{memory_id}
"""

from __future__ import annotations

import logging
import mimetypes
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

from backend.services.ocr_service import (
    OCRProcessingError,
    OCRService,
    OCRServiceError,
    OCRTimeoutError,
    OCRUnavailableError,
)

# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger("memoryos.ocr.api")


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    prefix="/ocr",
    tags=["OCR"],
)


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

UPLOAD_DIR = DATA_DIR / "uploads"

# Compatibility with the current upload API.
# Your upload.py stores images here.
LEGACY_UPLOAD_DIR = DATA_DIR / "uploads"


# ============================================================================
# OCR SERVICE
# ============================================================================

ocr_service = OCRService()


# ============================================================================
# CONSTANTS
# ============================================================================

SUPPORTED_STRATEGIES = (
    "auto",
    "single_block",
    "sparse",
    "single_line",
)

SUPPORTED_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
)

SUPPORTED_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


# ============================================================================
# RESPONSE HELPERS
# ============================================================================


def _success_response(
    *,
    data: dict[str, Any],
    processing_time_ms: float,
) -> dict[str, Any]:
    """
    Build a consistent successful API response.
    """

    return {
        "success": True,
        "service": "memoryos-ocr",
        "version": "1.0",
        "processing_time_ms": round(
            processing_time_ms,
            2,
        ),
        "data": data,
    }


def _error_response(
    *,
    code: str,
    message: str,
    memory_id: str | None = None,
    details: Any = None,
) -> dict[str, Any]:
    """
    Build a consistent OCR error response.
    """

    response: dict[str, Any] = {
        "success": False,
        "service": "memoryos-ocr",
        "version": "1.0",
        "error": {
            "code": code,
            "message": message,
        },
    }

    if memory_id is not None:
        response["error"]["memory_id"] = memory_id

    if details is not None:
        response["error"]["details"] = details

    return response


# ============================================================================
# MEMORY ID VALIDATION
# ============================================================================


def _validate_memory_id(memory_id: str) -> str:
    """
    Validate a MemoryOS memory ID.

    Prevents path traversal such as:

        ../../secret.txt
        C:\\Windows\\...

    while allowing normal MemoryOS IDs such as:

        mem_0a2461c7...
    """

    if not memory_id:
        raise HTTPException(
            status_code=422,
            detail=_error_response(
                code="invalid_memory_id",
                message="Memory ID cannot be empty.",
            ),
        )

    memory_id = memory_id.strip()

    if len(memory_id) > 200:
        raise HTTPException(
            status_code=422,
            detail=_error_response(
                code="invalid_memory_id",
                message="Memory ID is too long.",
            ),
        )

    # Path traversal protection.
    safe_name = Path(memory_id).name

    if safe_name != memory_id:
        raise HTTPException(
            status_code=400,
            detail=_error_response(
                code="invalid_memory_id",
                message="Invalid memory ID.",
            ),
        )

    # Extra protection against Windows path separators.
    if "\\" in memory_id or "/" in memory_id:
        raise HTTPException(
            status_code=400,
            detail=_error_response(
                code="invalid_memory_id",
                message="Invalid memory ID.",
            ),
        )

    return memory_id


# ============================================================================
# IMAGE RESOLUTION
# ============================================================================


def _find_uploaded_image(
    memory_id: str,
) -> Path | None:
    """
    Locate an uploaded MemoryOS image.

    Current upload system stores:

        data/uploads/{memory_id}.png
        data/uploads/{memory_id}.jpg
        data/uploads/{memory_id}.jpeg
        data/uploads/{memory_id}.webp
    """

    memory_id = _validate_memory_id(memory_id)

    candidate_dirs = (
        UPLOAD_DIR,
        LEGACY_UPLOAD_DIR,
    )

    checked: set[Path] = set()

    for directory in candidate_dirs:

        if directory in checked:
            continue

        checked.add(directory)

        if not directory.exists():
            continue

        for extension in SUPPORTED_EXTENSIONS:

            candidate = directory / (f"{memory_id}{extension}")

            if candidate.is_file():

                try:
                    resolved = candidate.resolve()
                    root = directory.resolve()

                    if not resolved.is_relative_to(root):
                        logger.warning(
                            "Blocked unsafe OCR path: %s",
                            candidate,
                        )
                        continue

                except Exception:
                    logger.exception("Failed to validate OCR image path.")
                    continue

                return candidate

    return None


# ============================================================================
# IMAGE METADATA
# ============================================================================


def _image_metadata(
    image_path: Path,
) -> dict[str, Any]:
    """
    Return safe metadata about the source image.
    """

    try:
        size = image_path.stat().st_size

    except OSError:
        size = 0

    extension = image_path.suffix.lower()

    return {
        "filename": image_path.name,
        "extension": extension,
        "mime_type": SUPPORTED_MIME_TYPES.get(
            extension,
            mimetypes.guess_type(
                image_path.name,
            )[0],
        ),
        "size_bytes": size,
    }


# ============================================================================
# OCR RESULT SERIALIZATION
# ============================================================================


def _serialize_result(
    *,
    result: Any,
    memory_id: str,
    image_path: Path,
    strategy: str,
) -> dict[str, Any]:
    """
    Convert OCRService result into a stable API response.

    This keeps the API independent from the internal OCR model.
    """

    text = (
        getattr(
            result,
            "text",
            "",
        )
        or ""
    )

    raw_text = (
        getattr(
            result,
            "raw_text",
            "",
        )
        or ""
    )

    confidence = getattr(
        result,
        "confidence",
        0.0,
    )

    word_count = getattr(
        result,
        "word_count",
        0,
    )

    character_count = getattr(
        result,
        "character_count",
        len(text),
    )

    language = getattr(
        result,
        "language",
        None,
    )

    engine = getattr(
        result,
        "engine",
        "tesseract",
    )

    processing_strategy = getattr(
        result,
        "processing_strategy",
        strategy,
    )

    available = getattr(
        result,
        "available",
        False,
    )

    error = getattr(
        result,
        "error",
        None,
    )

    # Defensive normalization.
    try:
        confidence = float(confidence or 0.0)

    except (TypeError, ValueError):
        confidence = 0.0

    confidence = max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )

    try:
        word_count = int(
            word_count or 0,
        )

    except (TypeError, ValueError):
        word_count = 0

    try:
        character_count = int(
            character_count or 0,
        )

    except (TypeError, ValueError):
        character_count = len(text)

    return {
        "memory_id": memory_id,
        "source": {
            "filename": image_path.name,
            "path": str(
                image_path.relative_to(
                    PROJECT_ROOT,
                )
            ),
            "mime_type": _image_metadata(
                image_path,
            )["mime_type"],
            "size_bytes": _image_metadata(
                image_path,
            )["size_bytes"],
        },
        "ocr": {
            "success": bool(
                getattr(
                    result,
                    "success",
                    False,
                )
            ),
            "text": text,
            "raw_text": raw_text,
            "confidence": round(
                confidence,
                4,
            ),
            "confidence_percent": round(
                confidence * 100,
                2,
            ),
            "word_count": word_count,
            "character_count": character_count,
            "language": language,
            "engine": engine,
            "processing_strategy": (processing_strategy),
            "available": bool(available),
            "has_text": bool(text.strip()),
            "error": error,
        },
        "strategies": {
            "requested": strategy,
            "supported": list(SUPPORTED_STRATEGIES),
        },
    }


# ============================================================================
# OCR STATUS
# ============================================================================


@router.get(
    "/status",
    summary="OCR engine health",
    description=(
        "Check the availability of the local Tesseract OCR engine " "used by MemoryOS."
    ),
)
async def get_ocr_status() -> dict[str, Any]:
    """
    Check OCR engine availability.
    """

    started = time.perf_counter()

    try:

        available = bool(ocr_service.available())

        processing_time_ms = (time.perf_counter() - started) * 1000

        return _success_response(
            processing_time_ms=processing_time_ms,
            data={
                "status": ("ready" if available else "degraded"),
                "available": available,
                "engine": "tesseract",
                "service": "local-ocr",
                "strategies": list(SUPPORTED_STRATEGIES),
                "upload_directory": str(
                    UPLOAD_DIR.relative_to(
                        PROJECT_ROOT,
                    )
                ),
                "message": (
                    "OCR engine is ready."
                    if available
                    else "OCR engine is unavailable. " "Check Tesseract installation."
                ),
            },
        )

    except Exception as exc:

        logger.exception("OCR health check failed.")

        processing_time_ms = (time.perf_counter() - started) * 1000

        return JSONResponse(
            status_code=503,
            content=_error_response(
                code="ocr_health_check_failed",
                message=str(exc),
            ),
        )


# ============================================================================
# OCR STRATEGIES
# ============================================================================


@router.get(
    "/strategies",
    summary="List OCR strategies",
    description=("Return all OCR processing strategies supported " "by MemoryOS."),
)
async def get_ocr_strategies() -> dict[str, Any]:
    """
    Return supported OCR strategies.
    """

    return {
        "success": True,
        "service": "memoryos-ocr",
        "strategies": [
            {
                "name": "auto",
                "recommended": True,
                "description": (
                    "Automatically test multiple OCR "
                    "strategies and choose the best result."
                ),
            },
            {
                "name": "single_block",
                "recommended": False,
                "description": (
                    "Best for a normal paragraph or " "single block of text."
                ),
            },
            {
                "name": "sparse",
                "recommended": False,
                "description": (
                    "Best for screenshots or images " "with scattered text."
                ),
            },
            {
                "name": "single_line",
                "recommended": False,
                "description": ("Best when the image contains " "one line of text."),
            },
        ],
    }


# ============================================================================
# INTERNAL OCR EXECUTION
# ============================================================================


async def _run_ocr(
    *,
    memory_id: str,
    strategy: str,
) -> dict[str, Any]:
    """
    Shared OCR execution pipeline used by GET and POST.
    """

    started = time.perf_counter()

    memory_id = _validate_memory_id(memory_id)

    # --------------------------------------------------------------
    # Validate strategy
    # --------------------------------------------------------------

    strategy = strategy.strip().lower()

    if strategy not in SUPPORTED_STRATEGIES:

        raise HTTPException(
            status_code=400,
            detail=_error_response(
                code="invalid_strategy",
                message=(f"Unsupported OCR strategy: " f"{strategy}"),
                details={"supported": list(SUPPORTED_STRATEGIES)},
            ),
        )

    # --------------------------------------------------------------
    # Locate image
    # --------------------------------------------------------------

    image_path = _find_uploaded_image(memory_id)

    if image_path is None:

        raise HTTPException(
            status_code=404,
            detail=_error_response(
                code="memory_image_not_found",
                message=("No uploaded image was found " "for this memory ID."),
                memory_id=memory_id,
                details={
                    "searched_directory": str(UPLOAD_DIR),
                    "supported_extensions": list(SUPPORTED_EXTENSIONS),
                },
            ),
        )

    logger.info(
        "Starting OCR | memory_id=%s | strategy=%s | file=%s",
        memory_id,
        strategy,
        image_path.name,
    )

    # --------------------------------------------------------------
    # OCR ENGINE
    # --------------------------------------------------------------

    try:

        result = ocr_service.extract_text(
            image_path,
            strategy=strategy,
        )

    except OCRUnavailableError as exc:

        logger.error(
            "Tesseract unavailable: %s",
            exc,
        )

        raise HTTPException(
            status_code=503,
            detail=_error_response(
                code="ocr_engine_unavailable",
                message=str(exc),
                memory_id=memory_id,
            ),
        ) from exc

    except OCRTimeoutError as exc:

        logger.error(
            "OCR timeout | memory_id=%s",
            memory_id,
        )

        raise HTTPException(
            status_code=504,
            detail=_error_response(
                code="ocr_timeout",
                message=str(exc),
                memory_id=memory_id,
            ),
        ) from exc

    except OCRProcessingError as exc:

        logger.exception(
            "OCR processing error | memory_id=%s",
            memory_id,
        )

        raise HTTPException(
            status_code=500,
            detail=_error_response(
                code="ocr_processing_failed",
                message=str(exc),
                memory_id=memory_id,
            ),
        ) from exc

    except OCRServiceError as exc:

        logger.exception(
            "OCR service error | memory_id=%s",
            memory_id,
        )

        raise HTTPException(
            status_code=500,
            detail=_error_response(
                code="ocr_service_error",
                message=str(exc),
                memory_id=memory_id,
            ),
        ) from exc

    except Exception as exc:

        logger.exception(
            "Unexpected OCR error | memory_id=%s",
            memory_id,
        )

        raise HTTPException(
            status_code=500,
            detail=_error_response(
                code="ocr_internal_error",
                message=("Unexpected OCR processing error."),
                memory_id=memory_id,
                details=str(exc),
            ),
        ) from exc

    # --------------------------------------------------------------
    # RESPONSE
    # --------------------------------------------------------------

    processing_time_ms = (time.perf_counter() - started) * 1000

    data = _serialize_result(
        result=result,
        memory_id=memory_id,
        image_path=image_path,
        strategy=strategy,
    )

    logger.info(
        (
            "OCR completed | memory_id=%s | "
            "success=%s | confidence=%s | "
            "words=%s | time=%.2fms"
        ),
        memory_id,
        getattr(
            result,
            "success",
            False,
        ),
        getattr(
            result,
            "confidence",
            0.0,
        ),
        getattr(
            result,
            "word_count",
            0,
        ),
        processing_time_ms,
    )

    return _success_response(
        data=data,
        processing_time_ms=processing_time_ms,
    )


# ============================================================================
# POST OCR
# ============================================================================


@router.post(
    "/extract/{memory_id}",
    summary="Extract text from a memory image",
    description=(
        "Run premium local OCR on an already uploaded MemoryOS image. "
        "The default 'auto' strategy tests multiple OCR configurations "
        "and selects the strongest result."
    ),
)
async def extract_memory_ocr(
    memory_id: str,
    strategy: str = Query(
        default="auto",
        description=("OCR strategy: auto, single_block, " "sparse, or single_line."),
    ),
) -> dict[str, Any]:
    """
    POST endpoint for production OCR processing.
    """

    return await _run_ocr(
        memory_id=memory_id,
        strategy=strategy,
    )


# ============================================================================
# GET OCR
# ============================================================================


@router.get(
    "/extract/{memory_id}",
    summary="Get OCR text from a memory image",
    description=(
        "Convenience GET endpoint that runs OCR using " "the automatic strategy."
    ),
)
async def get_memory_ocr(
    memory_id: str,
) -> dict[str, Any]:
    """
    GET endpoint for quick OCR testing.
    """

    return await _run_ocr(
        memory_id=memory_id,
        strategy="auto",
    )


# ============================================================================
# SIMPLE TEXT ENDPOINT
# ============================================================================


@router.get(
    "/text/{memory_id}",
    summary="Get extracted OCR text only",
    description=(
        "Return only the cleaned OCR text for a memory image. "
        "Useful for frontend search and AI pipelines."
    ),
)
async def get_ocr_text(
    memory_id: str,
) -> dict[str, Any]:
    """
    Lightweight OCR text endpoint.
    """

    result = await _run_ocr(
        memory_id=memory_id,
        strategy="auto",
    )

    data = result.get(
        "data",
        {},
    )

    ocr = data.get(
        "ocr",
        {},
    )

    return {
        "success": True,
        "memory_id": memory_id,
        "text": ocr.get(
            "text",
            "",
        ),
        "confidence": ocr.get(
            "confidence",
            0.0,
        ),
        "word_count": ocr.get(
            "word_count",
            0,
        ),
        "has_text": ocr.get(
            "has_text",
            False,
        ),
    }


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "router",
]
