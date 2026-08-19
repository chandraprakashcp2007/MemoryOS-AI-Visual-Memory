"""
MemoryOS — Upload API
=====================

Robust upload, storage, processing and image-serving API.

Supports:
    POST /upload
    POST /upload/batch
    GET  /upload/file/{memory_id}
    GET  /upload/status
    GET  /upload/metadata/{memory_id}

Image resolution supports BOTH:

1. New uploaded memories
2. Existing screenshot-scanner memories

This is important because older MemoryOS memories may have IDs such as:

    mem_db75c87a61a473cd

while newly uploaded images may use a different deterministic ID format.

The image resolver therefore does NOT assume that every memory ID
can be reconstructed from the image SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps, UnidentifiedImageError

# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger("memoryos.upload")


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    prefix="/upload",
    tags=["Upload"],
)


# ============================================================================
# PROJECT PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"

UPLOAD_DIR = DATA_DIR / "uploads"

THUMBNAIL_DIR = DATA_DIR / "thumbnails"

METADATA_DIR = DATA_DIR / "database" / "uploads"

DATABASE_DIR = DATA_DIR / "database"

MEMORY_INDEX_FILE = DATA_DIR / "memory_index" / "memories.json"


for directory in (
    DATA_DIR,
    UPLOAD_DIR,
    THUMBNAIL_DIR,
    METADATA_DIR,
    MEMORY_INDEX_FILE.parent,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# LIMITS
# ============================================================================

MAX_FILE_SIZE = 10 * 1024 * 1024

MAX_BATCH_FILES = 50

MIN_IMAGE_WIDTH = 16

MIN_IMAGE_HEIGHT = 16

THUMBNAIL_SIZE = (720, 720)

# Image byte size alone does not prevent decompression bombs. These limits are
# deliberately generous for screenshots while bounding memory use.
MAX_IMAGE_DIMENSION = 10_000
MAX_IMAGE_PIXELS = 40_000_000


# ============================================================================
# SUPPORTED IMAGE TYPES
# ============================================================================

SUPPORTED_EXTENSIONS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

SUPPORTED_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/gif",
    "image/bmp",
}


# ============================================================================
# EXCEPTIONS
# ============================================================================


class UploadServiceError(Exception):
    """Base upload service exception."""


class UploadValidationError(UploadServiceError):
    """Invalid uploaded image."""


class DuplicateUploadError(UploadServiceError):
    """Duplicate image."""


class UploadStorageError(UploadServiceError):
    """Storage failure."""


# ============================================================================
# TIME
# ============================================================================


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


# ============================================================================
# STRING HELPERS
# ============================================================================


def clean_string(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, bytes):
        value = value.decode(
            "utf-8",
            errors="replace",
        )

    value = str(value)

    value = value.replace(
        "\x00",
        " ",
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


def safe_filename(filename: str | None) -> str:
    value = clean_string(filename)

    if not value:
        value = "upload"

    value = Path(value).name

    value = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "_",
        value,
    )

    value = value.strip("._")

    if not value:
        value = "upload"

    return value[:180]


# ============================================================================
# JSON SERIALIZATION
# ============================================================================


def make_json_safe(value: Any) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}

    if isinstance(
        value,
        (list, tuple, set),
    ):
        return [make_json_safe(item) for item in value]

    if hasattr(
        value,
        "to_dict",
    ):
        try:
            return make_json_safe(value.to_dict())
        except Exception:
            pass

    if hasattr(
        value,
        "__dict__",
    ):
        try:
            return make_json_safe(vars(value))
        except Exception:
            pass

    return str(value)


# ============================================================================
# HASH
# ============================================================================


def calculate_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def calculate_file_sha256(path: Path) -> str | None:

    try:

        sha256 = hashlib.sha256()

        with path.open("rb") as handle:

            while True:

                chunk = handle.read(1024 * 1024)

                if not chunk:
                    break

                sha256.update(chunk)

        return sha256.hexdigest()

    except Exception:

        return None


# ============================================================================
# MEMORY ID
# ============================================================================


def generate_memory_id(image_hash: str) -> str:

    image_hash = clean_string(image_hash).lower()

    if not image_hash:
        return "mem_" + uuid.uuid4().hex[:24]

    digest = hashlib.sha256(image_hash.encode("utf-8")).hexdigest()

    return "mem_" + digest[:32]


# ============================================================================
# IMAGE TYPE
# ============================================================================


def detect_extension(
    filename: str,
    content_type: str | None,
) -> str:

    suffix = Path(filename).suffix.lower()

    if suffix in SUPPORTED_EXTENSIONS:
        return suffix

    mime = clean_string(content_type).lower()

    for extension, supported_mime in SUPPORTED_EXTENSIONS.items():

        if mime == supported_mime:
            return extension

    guessed = mimetypes.guess_extension(mime)

    if guessed:

        guessed = guessed.lower()

        if guessed in SUPPORTED_EXTENSIONS:
            return guessed

    raise UploadValidationError(
        "Unsupported image format. " "Supported formats: JPEG, PNG and WEBP."
    )


# ============================================================================
# IMAGE VALIDATION
# ============================================================================


def validate_image_bytes(
    data: bytes,
    *,
    filename: str,
    content_type: str | None,
) -> tuple[str, int, int]:

    if not data:

        raise UploadValidationError("Uploaded file is empty.")

    if len(data) > MAX_FILE_SIZE:

        raise UploadValidationError(
            f"File exceeds " f"{MAX_FILE_SIZE // (1024 * 1024)} MB."
        )

    extension = detect_extension(
        filename,
        content_type,
    )

    try:

        with Image.open(BytesIO(data)) as image:

            image.verify()

        with Image.open(BytesIO(data)) as image:

            width, height = image.size

            actual_format = (image.format or "").upper()

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:

        raise UploadValidationError("The uploaded file is not a valid image.") from exc

    if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:

        raise UploadValidationError("Image dimensions are too small.")

    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        raise UploadValidationError("Image dimensions exceed the supported limit.")

    if width * height > MAX_IMAGE_PIXELS:
        raise UploadValidationError("Image contains too many pixels to process safely.")

    if actual_format not in {
        "JPEG",
        "JPG",
        "PNG",
        "WEBP",
    }:

        raise UploadValidationError("Unsupported image format.")

    return (
        extension,
        width,
        height,
    )


# ============================================================================
# THUMBNAIL
# ============================================================================


def thumbnail_file(
    memory_id: str,
) -> Path:

    return THUMBNAIL_DIR / f"{memory_id}.jpg"


def create_thumbnail(
    data: bytes,
    output_path: Path,
) -> None:

    try:

        with Image.open(BytesIO(data)) as image:

            image = ImageOps.exif_transpose(image)

            if image.mode not in (
                "RGB",
                "RGBA",
            ):

                image = image.convert("RGB")

            image.thumbnail(
                THUMBNAIL_SIZE,
                Image.Resampling.LANCZOS,
            )

            if image.mode == "RGBA":

                background = Image.new(
                    "RGB",
                    image.size,
                    "white",
                )

                background.paste(
                    image,
                    mask=image.getchannel("A"),
                )

                image = background

            elif image.mode != "RGB":

                image = image.convert("RGB")

            image.save(
                output_path,
                format="JPEG",
                quality=90,
                optimize=True,
            )

    except Exception as exc:

        raise UploadStorageError("Unable to generate thumbnail.") from exc


# ============================================================================
# METADATA
# ============================================================================


def metadata_path(
    memory_id: str,
) -> Path:

    return METADATA_DIR / f"{memory_id}.json"


def save_metadata(
    memory_id: str,
    metadata: dict[str, Any],
) -> None:

    target = metadata_path(memory_id)

    temp = target.with_suffix(".json.tmp")

    try:

        with temp.open(
            "w",
            encoding="utf-8",
        ) as handle:

            json.dump(
                metadata,
                handle,
                ensure_ascii=False,
                indent=2,
                default=str,
            )

        temp.replace(target)

    except OSError as exc:

        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass

        raise UploadStorageError("Unable to save metadata.") from exc


def load_metadata(
    memory_id: str,
) -> dict[str, Any] | None:

    target = metadata_path(memory_id)

    if not target.is_file():
        return None

    try:

        with target.open(
            "r",
            encoding="utf-8",
        ) as handle:

            data = json.load(handle)

        if isinstance(data, dict):
            return data

    except Exception as exc:

        logger.warning(
            "Metadata read failed for %s: %s",
            memory_id,
            exc,
        )

    return None


# ============================================================================
# DUPLICATE DETECTION
# ============================================================================


def find_duplicate_by_hash(
    image_hash: str,
) -> dict[str, Any] | None:

    normalized = clean_string(image_hash).lower()

    if not normalized:
        return None

    try:

        for path in METADATA_DIR.glob("*.json"):

            try:

                with path.open(
                    "r",
                    encoding="utf-8",
                ) as handle:

                    metadata = json.load(handle)

            except Exception:
                continue

            if not isinstance(
                metadata,
                dict,
            ):
                continue

            stored_hash = clean_string(metadata.get("image_hash")).lower()

            if stored_hash == normalized:
                return metadata

    except Exception as exc:

        logger.warning(
            "Duplicate scan failed: %s",
            exc,
        )

    return None


# ============================================================================
# ORIGINAL STORAGE
# ============================================================================


def store_original(
    data: bytes,
    *,
    memory_id: str,
    extension: str,
) -> Path:

    target = UPLOAD_DIR / f"{memory_id}{extension}"

    temp = UPLOAD_DIR / (f".{memory_id}." f"{uuid.uuid4().hex}.tmp")

    try:

        with temp.open("wb") as handle:
            handle.write(data)

        temp.replace(target)

        return target

    except OSError as exc:

        try:
            temp.unlink(missing_ok=True)
        except OSError:
            pass

        raise UploadStorageError("Unable to store uploaded image.") from exc


def find_uploaded_original(
    memory_id: str,
) -> Path | None:

    for extension in SUPPORTED_EXTENSIONS:

        candidate = UPLOAD_DIR / f"{memory_id}{extension}"

        if candidate.is_file():
            return candidate

    return None


# ============================================================================
# PATH VALIDATION
# ============================================================================


def is_image_file(
    path: Path,
) -> bool:

    try:

        return (
            path.is_file()
            and path.stat().st_size > 0
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    except OSError:

        return False


def is_original_image_file(path: Path) -> bool:
    """Return true only for a source image, never a generated thumbnail.

    Legacy metadata may contain a thumbnail path. Serving it from the original
    endpoint would violate memory identity, so unresolved originals are
    deliberately reported as unavailable instead.
    """
    if not is_image_file(path):
        return False
    try:
        return not path.resolve().is_relative_to(THUMBNAIL_DIR.resolve())
    except OSError:
        return False


def resolve_path_value(
    value: Any,
) -> Path | None:

    raw = clean_string(value)

    if not raw:
        return None

    if raw.lower().startswith("file:///"):

        raw = raw[8:]

    raw = raw.replace(
        "\\\\",
        "\\",
    )

    candidates: list[Path] = []

    candidate = Path(raw)

    if candidate.is_absolute():

        candidates.append(candidate)

    else:

        candidates.extend(
            [
                BASE_DIR / candidate,
                DATA_DIR / candidate,
                BASE_DIR / "data" / candidate,
                Path.cwd() / candidate,
            ]
        )

    for path in candidates:

        try:

            resolved = path.resolve()

            if is_image_file(resolved):
                return resolved

        except Exception:
            continue

    return None


# ============================================================================
# RECURSIVE JSON PATH EXTRACTION
# ============================================================================


PATH_KEYS = {
    "path",
    "file",
    "filepath",
    "file_path",
    "source",
    "source_path",
    "image_path",
    "stored_path",
    "original_path",
    "thumbnail_path",
    "image",
    "filename",
    "file_name",
}


def extract_paths_from_object(
    obj: Any,
) -> list[Path]:

    found: list[Path] = []

    def walk(
        value: Any,
    ) -> None:

        if isinstance(
            value,
            dict,
        ):

            for key, item in value.items():

                key_normalized = (
                    clean_string(key).lower().replace("-", "_").replace(" ", "_")
                )

                if key_normalized in PATH_KEYS or "path" in key_normalized:

                    path = resolve_path_value(item)

                    if path and path not in found:

                        found.append(path)

                walk(item)

        elif isinstance(
            value,
            (list, tuple),
        ):

            for item in value:
                walk(item)

    walk(obj)

    return found


# ============================================================================
# METADATA PATH RESOLUTION
# ============================================================================


def paths_from_metadata(
    memory_id: str,
) -> list[Path]:

    found: list[Path] = []

    metadata = load_metadata(memory_id)

    if metadata:

        found.extend(extract_paths_from_object(metadata))

    return found


# ============================================================================
# ⭐ GLOBAL JSON MEMORY SEARCH
# ============================================================================


def search_all_json_metadata(
    memory_id: str,
) -> list[Path]:
    """
    Search every JSON file under data/.

    This is specifically for old screenshot-scanner
    memories that were indexed before upload metadata existed.

    Example:

        data/vector_store/metadata.json

    containing:

        {
            "memory_id": "mem_db75c87a61a473cd",
            "file_path": "C:\\Users\\...\\Screenshot.png"
        }

    will now be resolved correctly.
    """

    found: list[Path] = []

    memory_id = clean_string(memory_id)

    if not memory_id:
        return found

    try:

        json_files = list(DATA_DIR.rglob("*.json"))

    except Exception:

        return found

    for json_file in json_files:

        # Avoid temporary metadata files.
        if json_file.name.endswith(".json.tmp"):
            continue

        try:

            # Do not read enormous files.
            if json_file.stat().st_size > 50 * 1024 * 1024:
                continue

        except OSError:
            continue

        try:

            with json_file.open(
                "r",
                encoding="utf-8",
                errors="ignore",
            ) as handle:

                text = handle.read()

        except Exception:

            continue

        if memory_id not in text:
            continue

        try:

            parsed = json.loads(text)

            # Try to extract paths only from the specific memory entry.
            # This prevents all files from being returned when searching
            # a shared index file like scanner_state.json or memories.json.

            paths = []

            # Case 1: Direct memory ID key (memories.json structure)
            if isinstance(parsed, dict) and memory_id in parsed:
                paths = extract_paths_from_object(parsed[memory_id])

            # Case 2: Nested "files" dict (scanner_state.json structure)
            elif isinstance(parsed, dict) and "files" in parsed:
                files_dict = parsed.get("files", {})
                if isinstance(files_dict, dict):
                    for filename, entry in files_dict.items():
                        if isinstance(entry, dict) and entry.get("id") == memory_id:
                            # Found the memory entry - extract paths
                            entry_paths = extract_paths_from_object(entry)
                            if entry_paths:
                                paths = entry_paths
                            else:
                                # If no paths extracted (relative paths not resolved),
                                # try to find the file using the filename
                                if filename:
                                    for screenshot_dir in screenshot_directories():
                                        candidate = screenshot_dir / filename
                                        if is_image_file(candidate):
                                            paths = [candidate]
                                            break
                            break

            # Case 3: Unknown structure - extract all paths
            if not paths:
                paths = extract_paths_from_object(parsed)

            for path in paths:

                if path not in found:
                    found.append(path)

        except Exception:

            # If the JSON contains the memory ID
            # but is not valid JSON, inspect lines.
            for line in text.splitlines():

                if memory_id not in line:
                    continue

                quoted_paths = re.findall(
                    r'["\']([^"\']+\.(?:png|jpg|jpeg|webp))["\']',
                    line,
                    flags=re.IGNORECASE,
                )

                for raw_path in quoted_paths:

                    path = resolve_path_value(raw_path)

                    if path and path not in found:

                        found.append(path)

    return found


# ============================================================================
# SQLITE DISCOVERY
# ============================================================================


def database_files() -> list[Path]:

    found: list[Path] = []

    locations = [
        DATA_DIR,
        DATABASE_DIR,
        BASE_DIR,
    ]

    for location in locations:

        if not location.exists():
            continue

        for pattern in (
            "*.db",
            "*.sqlite",
            "*.sqlite3",
        ):

            try:

                for path in location.rglob(pattern):

                    if path.is_file() and path not in found:

                        found.append(path)

            except Exception:
                continue

    return found


def sqlite_tables(
    connection: sqlite3.Connection,
) -> list[str]:

    try:

        rows = connection.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            """).fetchall()

        return [str(row[0]) for row in rows if row and row[0]]

    except Exception:

        return []


def sqlite_columns(
    connection: sqlite3.Connection,
    table: str,
) -> list[str]:

    try:

        rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()

        return [str(row[1]) for row in rows if len(row) > 1]

    except Exception:

        return []


def paths_from_database(
    memory_id: str,
) -> list[Path]:

    found: list[Path] = []

    for database in database_files():

        connection = None

        try:

            connection = sqlite3.connect(
                str(database),
                timeout=1,
            )

            connection.row_factory = sqlite3.Row

            for table in sqlite_tables(connection):

                columns = sqlite_columns(
                    connection,
                    table,
                )

                if not columns:
                    continue

                id_columns = [
                    column
                    for column in columns
                    if column.lower()
                    in {
                        "memory_id",
                        "memoryid",
                        "id",
                        "uuid",
                    }
                ]

                path_columns = [
                    column
                    for column in columns
                    if (
                        "path" in column.lower()
                        or "filename" in column.lower()
                        or "file_name" in column.lower()
                    )
                ]

                if not id_columns or not path_columns:
                    continue

                for id_column in id_columns:

                    try:

                        rows = connection.execute(
                            f"""
                            SELECT *
                            FROM "{table}"
                            WHERE "{id_column}" = ?
                            LIMIT 20
                            """,
                            (memory_id,),
                        ).fetchall()

                    except Exception:

                        continue

                    for row in rows:

                        for column in path_columns:

                            try:

                                value = row[column]

                            except Exception:

                                continue

                            path = resolve_path_value(value)

                            if path and path not in found:

                                found.append(path)

        except Exception as exc:

            logger.debug(
                "SQLite search skipped: %s",
                exc,
            )

        finally:

            if connection:

                try:
                    connection.close()
                except Exception:
                    pass

    return found


# ============================================================================
# SCREENSHOT DIRECTORIES
# ============================================================================


def screenshot_directories() -> list[Path]:

    home = Path.home()

    candidates = [
        home / "Pictures" / "Screenshots",
        home / "OneDrive" / "Pictures" / "Screenshots",
        home / "OneDrive" / "Pictures",
        BASE_DIR / "data" / "screenshots",
        BASE_DIR / "screenshots",
    ]

    result: list[Path] = []

    for path in candidates:

        try:

            if path.is_dir():

                if path not in result:
                    result.append(path)

        except Exception:
            continue

    return result


# ============================================================================
# FILENAME MEMORY-ID SEARCH
# ============================================================================


def find_image_by_filename(
    memory_id: str,
) -> Path | None:
    """
    Some older systems store the memory ID directly
    in the image filename.

    Example:

        mem_db75c87a61a473cd.png
    """

    memory_id_lower = memory_id.lower()

    for directory in screenshot_directories():

        try:

            for path in directory.rglob("*"):

                if not is_image_file(path):
                    continue

                if memory_id_lower in path.stem.lower():

                    return path

        except Exception:
            continue

    return None


# ============================================================================
# SCREENSHOT HASH SEARCH
# ============================================================================


def find_screenshot_by_hash(
    memory_id: str,
) -> Path | None:
    """
    Final fallback for scanner memories.

    This supports memory IDs generated directly from
    screenshot content.
    """

    for directory in screenshot_directories():

        try:

            for path in directory.rglob("*"):

                if not is_image_file(path):
                    continue

                file_hash = calculate_file_sha256(path)

                if not file_hash:
                    continue

                generated_id = generate_memory_id(file_hash)

                if generated_id == memory_id:

                    return path

        except Exception as exc:

            logger.debug(
                "Screenshot hash search failed: %s",
                exc,
            )

    return None


# ============================================================================
# ⭐ MASTER IMAGE RESOLVER
# ============================================================================


def resolve_memory_image(
    memory_id: str,
) -> Path | None:
    """
    Resolve any MemoryOS memory to its real image.

    Priority:

        1. direct uploaded image
        2. upload thumbnail
        3. upload metadata
        4. global JSON/index metadata
        5. SQLite
        6. filename search
        7. screenshot hash search
    """

    memory_id = clean_string(memory_id)

    if not memory_id:
        return None

    logger.info(
        "Resolving MemoryOS image: %s",
        memory_id,
    )

    # ------------------------------------------------------------------
    # 1. Direct upload
    # ------------------------------------------------------------------

    path = find_uploaded_original(memory_id)

    if path:

        logger.info(
            "IMAGE RESOLVED [UPLOAD]: %s",
            path,
        )

        return path

    # ------------------------------------------------------------------
    # 2. Metadata
    # ------------------------------------------------------------------

    for path in paths_from_metadata(memory_id):

        if is_original_image_file(path):

            logger.info(
                "IMAGE RESOLVED [METADATA]: %s",
                path,
            )

            return path

    # ------------------------------------------------------------------
    # 3. Global JSON metadata
    # ------------------------------------------------------------------

    for path in search_all_json_metadata(memory_id):

        if is_original_image_file(path):

            logger.info(
                "IMAGE RESOLVED [JSON INDEX]: %s",
                path,
            )

            return path

    # ------------------------------------------------------------------
    # 4. SQLite
    # ------------------------------------------------------------------

    for path in paths_from_database(memory_id):

        if is_original_image_file(path):

            logger.info(
                "IMAGE RESOLVED [SQLITE]: %s",
                path,
            )

            return path

    # ------------------------------------------------------------------
    # 5. Filename
    # ------------------------------------------------------------------

    path = find_image_by_filename(memory_id)

    if path:

        logger.info(
            "IMAGE RESOLVED [FILENAME]: %s",
            path,
        )

        return path

    # ------------------------------------------------------------------
    # 6. Screenshot hash
    # ------------------------------------------------------------------

    path = find_screenshot_by_hash(memory_id)

    if path:

        logger.info(
            "IMAGE RESOLVED [HASH]: %s",
            path,
        )

        return path

    logger.warning(
        "IMAGE RESOLUTION FAILED: %s",
        memory_id,
    )

    return None


# ============================================================================
# PROCESSING
# ============================================================================


def normalize_processing_result(
    result: Any,
) -> dict[str, Any]:

    if result is None:

        return {
            "success": True,
            "processed": True,
            "status": "processed",
        }

    if isinstance(
        result,
        dict,
    ):

        data = dict(result)

    elif hasattr(
        result,
        "to_dict",
    ):

        try:
            data = result.to_dict()
        except Exception:
            data = {"result": str(result)}

    elif hasattr(
        result,
        "__dict__",
    ):

        try:
            data = dict(vars(result))
        except Exception:
            data = {"result": str(result)}

    else:

        data = {"result": str(result)}

    if not isinstance(
        data,
        dict,
    ):

        data = {"result": str(data)}

    data.setdefault(
        "success",
        True,
    )

    data.setdefault(
        "processed",
        True,
    )

    data.setdefault(
        "status",
        "processed",
    )

    return make_json_safe(data)


def persist_search_memory(
    *,
    memory_id: str,
    metadata: dict[str, Any],
    processing: dict[str, Any],
) -> None:
    """Persist the upload's own OCR and visual analysis under its memory ID."""
    try:
        memories: dict[str, Any] = {}
        if MEMORY_INDEX_FILE.exists():
            with MEMORY_INDEX_FILE.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
                if isinstance(data, dict):
                    memories = data

        vision = processing.get("vision")
        vision = vision if isinstance(vision, dict) else {}
        summary = str(processing.get("description") or vision.get("summary") or "")
        visual_description = str(vision.get("visual_description") or "")
        ocr_text = str(processing.get("ocr_text") or "")
        keywords = processing.get("keywords") or vision.get("keywords") or []
        entities = processing.get("entities") or vision.get("entities") or []
        keywords = keywords if isinstance(keywords, list) else [keywords]
        entities = entities if isinstance(entities, list) else [entities]

        # Backfill updates analysis in place.  Preserve any existing durable
        # fields that this compatibility index does not itself own.
        existing = memories.get(memory_id)
        record = dict(existing) if isinstance(existing, dict) else {}
        def concept_list(value: Any) -> list[Any]:
            return value if isinstance(value, list) else ([value] if value else [])

        concept_field_names = (
            "visual_concepts", "objects", "actions", "relationships",
            "food_concepts", "product_concepts", "document_concepts",
            "technology_concepts", "coding_concepts", "ui_concepts",
            "location_concepts", "activity_concepts", "topic_concepts",
            "general_concepts",
        )
        concept_fields = {name: concept_list(vision.get(name)) for name in concept_field_names}
        for value in (*concept_fields.values(), concept_list(vision.get("scene")), concept_list(vision.get("environment"))):
            keywords.extend(str(item) for item in value if str(item).strip())
        keywords = list(dict.fromkeys(str(item) for item in keywords if str(item).strip()))

        record.update({
            "id": memory_id,
            "memory_id": memory_id,
            "filename": metadata["filename"],
            "path": metadata["original_path"],
            "image_hash": metadata.get("image_hash", ""),
            "sha256": processing.get("sha256") or metadata.get("image_hash", ""),
            "image_url": f"/upload/file/{memory_id}",
            "original_image_url": f"/upload/file/{memory_id}",
            "preview_url": f"/upload/file/{memory_id}?preview=true",
            "thumbnail_url": f"/upload/file/{memory_id}?thumbnail=true",
            "category": str(processing.get("category") or vision.get("category") or "other"),
            "summary": summary,
            "visual_description": visual_description,
            **{name: [str(item) for item in values] for name, values in concept_fields.items()},
            "scene": str(vision.get("scene") or ""),
            "environment": str(vision.get("environment") or ""),
            "ocr_text": ocr_text,
            "entities": [str(item) for item in entities],
            "keywords": [str(item) for item in keywords],
            "vision_analysis": vision,
            "search_document": "\n".join(
                part for part in (
                    summary, visual_description, ocr_text,
                    " ".join(map(str, entities)), " ".join(map(str, keywords)),
                    *(" ".join(map(str, values)) for values in concept_fields.values()),
                    str(vision.get("scene") or ""), str(vision.get("environment") or ""),
                ) if part
            ),
            "status": str(processing.get("status") or "processed"),
            "indexed_at": utc_now_iso(),
        })
        memories[memory_id] = record
        temporary = MEMORY_INDEX_FILE.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(memories, handle, ensure_ascii=False, indent=2)
        temporary.replace(MEMORY_INDEX_FILE)
    except Exception:
        logger.exception("Could not persist searchable memory: %s", memory_id)


def process_uploaded_memory(
    *,
    source: Path,
    memory_id: str,
    metadata: dict[str, Any],
) -> dict[str, Any]:

    logger.info(
        "Starting processing: %s",
        memory_id,
    )

    try:

        from backend.services.processing_service import (
            process_image,
        )

    except Exception as exc:

        logger.exception("Unable to import processing_service.")

        return {
            "success": False,
            "processed": False,
            "status": "processing_failed",
            "error": ("Unable to import processing service: " f"{exc}"),
        }

    # ------------------------------------------------------------------
    # Current processing_service API
    # ------------------------------------------------------------------

    try:

        result = process_image(
            source,
            source_name="upload",
            metadata={
                **metadata,
                "memory_id": memory_id,
                "source": "upload",
            },
            index=True,
            allow_partial_success=True,
        )

        normalized = normalize_processing_result(result)

        logger.info(
            "Processing completed: %s -> %s",
            memory_id,
            normalized.get("status"),
        )

        return normalized

    except TypeError as first_error:

        logger.warning(
            "Using legacy processing_service signature " "for %s: %s",
            memory_id,
            first_error,
        )

        try:

            result = process_image(source)

            return normalize_processing_result(result)

        except Exception as exc:

            logger.exception("Legacy processing failed.")

            return {
                "success": False,
                "processed": False,
                "status": "processing_failed",
                "error": str(exc),
            }

    except Exception as exc:

        logger.exception(
            "MemoryOS processing failed: %s",
            memory_id,
        )

        return {
            "success": False,
            "processed": False,
            "status": "processing_failed",
            "error": str(exc),
        }


# ============================================================================
# URLS
# ============================================================================


def build_file_urls(
    memory_id: str,
) -> dict[str, str]:

    return {
        "original_url": f"/upload/file/{memory_id}",
        "preview_url": f"/upload/file/{memory_id}?preview=true",
        "thumbnail_url": f"/upload/file/{memory_id}?thumbnail=true",
    }


# ============================================================================
# RESPONSE
# ============================================================================


def build_upload_response(
    *,
    memory_id: str,
    filename: str,
    image_hash: str,
    width: int,
    height: int,
    file_size: int,
    duplicate: bool,
    status: str,
    processing: dict[str, Any] | None = None,
) -> dict[str, Any]:

    response = {
        "success": True,
        "memory_id": memory_id,
        "id": memory_id,
        "filename": filename,
        "image_hash": image_hash,
        "duplicate": duplicate,
        "status": status,
        "file": {
            "filename": filename,
            "size": file_size,
            "width": width,
            "height": height,
        },
        "urls": build_file_urls(memory_id),
        "created_at": utc_now_iso(),
    }

    if processing is not None:

        response["processing"] = processing

    return response


# ============================================================================
# INTERNAL UPLOAD
# ============================================================================


async def _handle_upload(
    file: UploadFile,
) -> dict[str, Any]:

    if file is None:

        raise UploadValidationError("No file was provided.")

    filename = safe_filename(file.filename)

    content_type = clean_string(file.content_type).lower()

    try:

        data = await file.read()

    except Exception as exc:

        raise UploadStorageError("Unable to read uploaded file.") from exc

    if not data:

        raise UploadValidationError("Uploaded file is empty.")

    image_hash = calculate_sha256(data)

    # ------------------------------------------------------------------
    # Duplicate
    # ------------------------------------------------------------------

    duplicate = find_duplicate_by_hash(image_hash)

    if duplicate:

        existing_id = clean_string(duplicate.get("memory_id") or duplicate.get("id"))

        if existing_id:

            return build_upload_response(
                memory_id=existing_id,
                filename=clean_string(
                    duplicate.get(
                        "filename",
                        filename,
                    )
                ),
                image_hash=image_hash,
                width=int(
                    duplicate.get(
                        "width",
                        0,
                    )
                    or 0
                ),
                height=int(
                    duplicate.get(
                        "height",
                        0,
                    )
                    or 0
                ),
                file_size=int(
                    duplicate.get(
                        "file_size",
                        len(data),
                    )
                    or len(data)
                ),
                duplicate=True,
                status="duplicate",
                processing={
                    "success": True,
                    "processed": bool(
                        duplicate.get(
                            "processed",
                            False,
                        )
                    ),
                    "status": "duplicate",
                },
            )

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    extension, width, height = validate_image_bytes(
        data,
        filename=filename,
        content_type=content_type,
    )

    # ------------------------------------------------------------------
    # Memory ID
    # ------------------------------------------------------------------

    memory_id = generate_memory_id(image_hash)

    # ------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------

    original_path = store_original(
        data,
        memory_id=memory_id,
        extension=extension,
    )

    preview_path = thumbnail_file(memory_id)

    try:

        create_thumbnail(
            data,
            preview_path,
        )

    except Exception:

        logger.exception(
            "Thumbnail creation failed: %s",
            memory_id,
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    metadata = {
        "memory_id": memory_id,
        "id": memory_id,
        "filename": filename,
        "original_filename": filename,
        "mime_type": SUPPORTED_EXTENSIONS.get(
            extension,
            content_type,
        ),
        "extension": extension,
        "file_size": len(data),
        "file_size_bytes": len(data),
        "width": width,
        "height": height,
        "image_hash": image_hash,
        "original_path": str(original_path.resolve()),
        "file_path": str(original_path.resolve()),
        "source_path": str(original_path.resolve()),
        "thumbnail_path": (
            str(preview_path.resolve()) if preview_path.exists() else None
        ),
        "created_at": utc_now_iso(),
        "updated_at": utc_now_iso(),
        "processed": False,
        "status": "uploaded",
        "source": "upload",
    }

    save_metadata(
        memory_id,
        metadata,
    )

    # ------------------------------------------------------------------
    # REAL PROCESSING PIPELINE
    # ------------------------------------------------------------------

    processing = process_uploaded_memory(
        source=original_path,
        memory_id=memory_id,
        metadata=metadata,
    )

    persist_search_memory(
        memory_id=memory_id,
        metadata=metadata,
        processing=processing,
    )

    processed = bool(
        processing.get(
            "processed",
            False,
        )
    )

    status = clean_string(
        processing.get(
            "status",
            "processed" if processed else "processing_failed",
        )
    )

    if not status:

        status = "processed" if processed else "processing_failed"

    metadata.update(
        {
            "processed": processed,
            "status": status,
            "processing_status": status,
            "processing_result": processing,
            "updated_at": utc_now_iso(),
        }
    )

    save_metadata(
        memory_id,
        metadata,
    )

    return build_upload_response(
        memory_id=memory_id,
        filename=filename,
        image_hash=image_hash,
        width=width,
        height=height,
        file_size=len(data),
        duplicate=False,
        status=status,
        processing=processing,
    )


# ============================================================================
# POST /upload
# ============================================================================


@router.post(
    "",
    summary="Upload image",
)
async def upload_image(
    file: UploadFile = File(
        ...,
        description=("JPEG, PNG or WEBP image. " "Maximum 10 MB."),
    ),
):

    try:

        return await _handle_upload(file)

    except UploadValidationError as exc:

        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "invalid_upload",
                "message": str(exc),
            },
        ) from exc

    except UploadStorageError as exc:

        logger.exception("Upload storage error.")

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "storage_error",
                "message": str(exc),
            },
        ) from exc

    except Exception:

        logger.exception("Unexpected upload error.")

        raise HTTPException(
            status_code=500,
            detail={
                "success": False,
                "error": "internal_server_error",
                "message": ("MemoryOS could not process " "the upload."),
            },
        )


# ============================================================================
# POST /upload/batch
# ============================================================================


@router.post(
    "/batch",
    summary="Upload multiple images",
)
async def upload_batch(
    files: list[UploadFile] = File(
        ...,
        description=("Multiple JPEG, PNG or WEBP images. " "Maximum 50."),
    ),
):

    if not files:

        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "empty_batch",
                "message": "No files provided.",
            },
        )

    if len(files) > MAX_BATCH_FILES:

        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "batch_limit_exceeded",
                "message": f"Maximum {MAX_BATCH_FILES} " "files allowed.",
            },
        )

    results: list[dict[str, Any]] = []

    successful = 0
    failed = 0
    duplicates = 0
    processed = 0

    for file in files:

        try:

            result = await _handle_upload(file)

            results.append(result)

            successful += 1

            if result.get("duplicate"):

                duplicates += 1

            processing = result.get("processing")

            if isinstance(
                processing,
                dict,
            ) and processing.get("processed"):

                processed += 1

        except UploadServiceError as exc:

            failed += 1

            results.append(
                {
                    "success": False,
                    "filename": safe_filename(file.filename),
                    "error": "upload_failed",
                    "message": str(exc),
                }
            )

        except Exception as exc:

            failed += 1

            logger.exception("Batch upload failed.")

            results.append(
                {
                    "success": False,
                    "filename": safe_filename(file.filename),
                    "error": "internal_server_error",
                    "message": str(exc),
                }
            )

    return {
        "success": failed == 0,
        "total": len(files),
        "successful": successful,
        "failed": failed,
        "duplicates": duplicates,
        "processed": processed,
        "results": results,
        "created_at": utc_now_iso(),
    }


# ============================================================================
# GET /upload/file/{memory_id}
# ============================================================================


@router.get(
    "/file/{memory_id}",
    summary="Retrieve MemoryOS image",
)
async def get_uploaded_file(
    memory_id: str,
    preview: bool = False,
    thumbnail: bool = False,
):

    memory_id = clean_string(memory_id)

    if not memory_id:

        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "invalid_memory_id",
                "message": "Memory ID is required.",
            },
        )

    if not re.fullmatch(
        r"[A-Za-z0-9_-]+",
        memory_id,
    ):

        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "invalid_memory_id",
                "message": "Invalid memory ID.",
            },
        )

    # ------------------------------------------------------------------
    # Resolve image
    # ------------------------------------------------------------------

    image_path = resolve_memory_image(memory_id)

    if image_path is None:

        logger.warning(
            "IMAGE 404: %s",
            memory_id,
        )

        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": "file_not_found",
                "message": "Image could not be " "resolved for this memory.",
                "memory_id": memory_id,
            },
        )

    # A thumbnail is valid only when it was generated from this exact source.
    # Older thumbnail files did not retain that association, so regenerate
    # them once from the resolved original instead of risking a wrong card.
    if thumbnail or preview:
        source_hash = calculate_file_sha256(image_path)
        thumb = thumbnail_file(memory_id)
        metadata = load_metadata(memory_id) or {}
        if metadata.get("thumbnail_source_sha256") != source_hash or not thumb.is_file():
            create_thumbnail(image_path.read_bytes(), thumb)
            metadata["thumbnail_source_sha256"] = source_hash
            metadata["thumbnail_path"] = str(thumb.resolve())
            metadata["updated_at"] = utc_now_iso()
            save_metadata(memory_id, metadata)

        return FileResponse(
            path=thumb,
            media_type="image/jpeg",
            filename=(f"{memory_id}.jpg"),
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )

    # ------------------------------------------------------------------
    # Serve
    # ------------------------------------------------------------------

    mime_type = mimetypes.guess_type(image_path.name)[0] or SUPPORTED_EXTENSIONS.get(
        image_path.suffix.lower(),
        "image/jpeg",
    )

    logger.info(
        "Serving image: %s -> %s",
        memory_id,
        image_path,
    )

    return FileResponse(
        path=image_path,
        media_type=mime_type,
        filename=image_path.name,
        headers={
            "Cache-Control": "no-cache, must-revalidate",
        },
    )


# ============================================================================
# GET /upload/status
# ============================================================================


@router.get(
    "/status",
    summary="Upload storage status",
)
async def upload_status():

    try:

        original_files = sum(
            1
            for extension in SUPPORTED_EXTENSIONS
            for _ in UPLOAD_DIR.glob(f"*{extension}")
        )

        thumbnails = sum(1 for _ in THUMBNAIL_DIR.glob("*.jpg"))

        metadata_files = sum(1 for _ in METADATA_DIR.glob("*.json"))

        upload_bytes = 0

        for extension in SUPPORTED_EXTENSIONS:

            for path in UPLOAD_DIR.glob(f"*{extension}"):

                try:

                    upload_bytes += path.stat().st_size

                except OSError:
                    pass

        return {
            "success": True,
            "status": "ready",
            "storage": {
                "upload_directory": str(UPLOAD_DIR),
                "thumbnail_directory": str(THUMBNAIL_DIR),
                "metadata_directory": str(METADATA_DIR),
                "original_files": original_files,
                "thumbnails": thumbnails,
                "metadata_files": metadata_files,
                "storage_bytes": upload_bytes,
                "storage_mb": round(
                    upload_bytes / (1024 * 1024),
                    2,
                ),
            },
            "limits": {
                "max_file_size_mb": MAX_FILE_SIZE // (1024 * 1024),
                "max_batch_files": MAX_BATCH_FILES,
                "supported_types": sorted(SUPPORTED_MIME_TYPES),
            },
            "timestamp": utc_now_iso(),
        }

    except Exception as exc:

        logger.exception("Upload status failed.")

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "status": "error",
                "error": "status_unavailable",
                "message": str(exc),
            },
        )


# ============================================================================
# GET /upload/metadata/{memory_id}
# ============================================================================


@router.get(
    "/metadata/{memory_id}",
    summary="Get upload metadata",
)
async def get_upload_metadata(
    memory_id: str,
):

    memory_id = clean_string(memory_id)

    if not re.fullmatch(
        r"[A-Za-z0-9_-]+",
        memory_id,
    ):

        raise HTTPException(
            status_code=400,
            detail={
                "success": False,
                "error": "invalid_memory_id",
                "message": "Invalid memory ID.",
            },
        )

    metadata = load_metadata(memory_id)

    if metadata is None:

        raise HTTPException(
            status_code=404,
            detail={
                "success": False,
                "error": "metadata_not_found",
                "message": "Upload metadata was " "not found.",
            },
        )

    return {
        "success": True,
        "memory_id": memory_id,
        "metadata": metadata,
        "urls": build_file_urls(memory_id),
    }


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "router",
    "UploadServiceError",
    "UploadValidationError",
    "DuplicateUploadError",
    "UploadStorageError",
    "upload_image",
    "upload_batch",
    "get_uploaded_file",
    "upload_status",
    "get_upload_metadata",
    "resolve_memory_image",
]
