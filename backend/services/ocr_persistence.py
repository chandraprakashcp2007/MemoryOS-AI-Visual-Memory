"""
MemoryOS - OCR Persistence Service

Responsible for permanently storing OCR results
for each MemoryOS memory.

This service does NOT perform OCR.
It only persists and retrieves OCR results.
"""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OCR_DATABASE_DIR = PROJECT_ROOT / "data" / "database" / "ocr"


# ---------------------------------------------------------------------------
# Directory initialization
# ---------------------------------------------------------------------------


def ensure_ocr_database() -> Path:
    """
    Ensure the OCR persistence directory exists.
    """
    OCR_DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    return OCR_DATABASE_DIR


# ---------------------------------------------------------------------------
# Memory-safe filename
# ---------------------------------------------------------------------------


def _ocr_file(memory_id: str) -> Path:
    """
    Return the JSON persistence path for a memory.

    Example:
        mem_123 -> data/database/ocr/mem_123.json
    """

    if not memory_id:
        raise ValueError("memory_id is required")

    # Prevent path traversal.
    safe_memory_id = Path(memory_id).name

    if safe_memory_id != memory_id:
        raise ValueError("Invalid memory_id")

    return ensure_ocr_database() / f"{safe_memory_id}.json"


# ---------------------------------------------------------------------------
# Text hashing
# ---------------------------------------------------------------------------


def calculate_text_hash(text: str) -> str:
    """
    Calculate a stable SHA-256 hash for OCR text.
    """

    normalized = (text or "").strip()

    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    """
    Convert common Python objects into JSON-safe values.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    if hasattr(value, "model_dump"):
        return _json_safe(value.model_dump())

    if hasattr(value, "dict"):
        return _json_safe(value.dict())

    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))

    return str(value)


# ---------------------------------------------------------------------------
# Save OCR result
# ---------------------------------------------------------------------------


def save_ocr_result(
    memory_id: str,
    ocr_result: Any,
    source: Optional[Dict[str, Any]] = None,
    processing_time_ms: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Persist OCR information for a MemoryOS memory.

    The write is atomic:
        temporary file -> final JSON file

    This prevents partially-written OCR records.
    """

    if not memory_id:
        raise ValueError("memory_id is required")

    ensure_ocr_database()

    # Convert OCRResult / Pydantic model / dict into dictionary.
    ocr_data = _json_safe(ocr_result)

    if not isinstance(ocr_data, dict):
        ocr_data = {"text": str(ocr_data)}

    text = str(ocr_data.get("text") or ocr_data.get("raw_text") or "")

    raw_text = str(ocr_data.get("raw_text") or text)

    # Preserve existing hash if available.
    text_hash = ocr_data.get("text_hash") or calculate_text_hash(text)

    # Build premium persistent record.
    record: Dict[str, Any] = {
        "memory_id": memory_id,
        "source": _json_safe(source or {}),
        "ocr": {
            "success": bool(ocr_data.get("success", False)),
            "text": text,
            "raw_text": raw_text,
            "confidence": ocr_data.get("confidence"),
            "confidence_percent": ocr_data.get("confidence_percent"),
            "word_count": ocr_data.get(
                "word_count",
                0,
            ),
            "character_count": ocr_data.get(
                "character_count",
                len(text),
            ),
            "language": ocr_data.get(
                "language",
                "eng",
            ),
            "engine": ocr_data.get(
                "engine",
                "tesseract",
            ),
            "processing_strategy": ocr_data.get("processing_strategy"),
            "available": ocr_data.get(
                "available",
                True,
            ),
            "has_text": bool(
                ocr_data.get(
                    "has_text",
                    bool(text.strip()),
                )
            ),
            "error": ocr_data.get("error"),
            "text_hash": text_hash,
            "words": _json_safe(ocr_data.get("words", [])),
        },
        "metadata": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": (
                round(
                    float(processing_time_ms),
                    2,
                )
                if processing_time_ms is not None
                else None
            ),
            "text_hash": text_hash,
            "schema_version": "1.0",
        },
    }

    record = _json_safe(record)

    destination = _ocr_file(memory_id)

    # Atomic write.
    fd, temporary_path = tempfile.mkstemp(
        prefix=f".{memory_id}.",
        suffix=".tmp",
        dir=str(OCR_DATABASE_DIR),
        text=True,
    )

    try:
        with os.fdopen(
            fd,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                record,
                file,
                indent=2,
                ensure_ascii=False,
            )

            file.write("\n")

        os.replace(
            temporary_path,
            destination,
        )

    except Exception:

        try:
            os.unlink(temporary_path)
        except OSError:
            pass

        raise

    return record


# ---------------------------------------------------------------------------
# Load OCR result
# ---------------------------------------------------------------------------


def load_ocr_result(
    memory_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Load a persisted OCR record.

    Returns:
        dict if found
        None if not found
    """

    path = _ocr_file(memory_id)

    if not path.exists():
        return None

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ---------------------------------------------------------------------------
# Check existence
# ---------------------------------------------------------------------------


def ocr_result_exists(
    memory_id: str,
) -> bool:
    """
    Check whether OCR persistence exists.
    """

    return _ocr_file(memory_id).exists()


# ---------------------------------------------------------------------------
# Delete OCR result
# ---------------------------------------------------------------------------


def delete_ocr_result(
    memory_id: str,
) -> bool:
    """
    Delete persisted OCR information.

    Returns True if deleted.
    """

    path = _ocr_file(memory_id)

    if not path.exists():
        return False

    path.unlink()

    return True
