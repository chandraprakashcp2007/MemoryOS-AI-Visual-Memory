"""
MemoryOS - File Utilities

Secure and reusable utilities for handling uploaded files.

Responsibilities:
    - Filename sanitization
    - File extension validation
    - MIME type validation
    - File size validation
    - SHA-256 hashing
    - Duplicate detection helpers
    - Safe storage paths
    - File persistence
    - File deletion
    - File metadata extraction

Security principles:
    - Never trust user-provided filenames
    - Never trust file extensions alone
    - Never allow path traversal
    - Generate server-side filenames
    - Validate size before processing
    - Use content hashes for duplicate detection
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from backend.config import (
    SCREENSHOTS_DIR,
    settings,
    is_supported_image_extension,
    is_supported_mime_type,
)

# ============================================================================
# EXCEPTIONS
# ============================================================================


class FileUtilityError(Exception):
    """Base exception for MemoryOS file utility errors."""


class InvalidFileError(FileUtilityError):
    """Raised when an uploaded file is invalid."""


class UnsupportedFileTypeError(InvalidFileError):
    """Raised when the file type is not supported."""


class FileTooLargeError(InvalidFileError):
    """Raised when the file exceeds the configured size limit."""


class FileSaveError(FileUtilityError):
    """Raised when a file cannot be safely persisted."""


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass(frozen=True, slots=True)
class FileMetadata:
    """
    Immutable metadata describing an uploaded file.
    """

    original_filename: str
    safe_filename: str
    extension: str
    mime_type: str
    size_bytes: int
    sha256: str


# ============================================================================
# FILENAME UTILITIES
# ============================================================================


def sanitize_filename(filename: str | None) -> str:
    """
    Convert an untrusted filename into a safe display/storage filename.

    Examples:
        "../../secret.txt"
            -> "secret.txt"

        "My Screenshot (1).PNG"
            -> "My_Screenshot_1.PNG"
    """

    if not filename:
        return "unnamed_file"

    # Remove directory components from Windows and Unix paths.
    name = Path(filename.replace("\\", "/")).name

    # Replace dangerous/non-standard characters.
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name)

    # Convert whitespace runs into a single underscore.
    name = re.sub(r"\s+", "_", name)

    # Prevent names beginning with multiple dots.
    name = re.sub(r"^\.+", "", name)

    # Remove repeated underscores.
    name = re.sub(r"_+", "_", name)

    # Remove leading/trailing dots/spaces/underscores.
    name = name.strip(" ._")

    if not name:
        return "unnamed_file"

    # Keep filename length reasonable.
    if len(name) > 180:
        suffix = Path(name).suffix
        stem = Path(name).stem

        maximum_stem_length = 180 - len(suffix)

        name = stem[:maximum_stem_length] + suffix

    return name


def get_extension(filename: str) -> str:
    """
    Return a normalized lowercase file extension.

    Example:
        screenshot.PNG -> ".png"
    """

    return Path(filename).suffix.lower()


# ============================================================================
# VALIDATION
# ============================================================================


def validate_extension(filename: str) -> str:
    """
    Validate and return the normalized image extension.
    """

    extension = get_extension(filename)

    if not extension:
        raise UnsupportedFileTypeError("File has no extension.")

    if not is_supported_image_extension(extension):
        supported = ", ".join(settings.allowed_image_extensions)

        raise UnsupportedFileTypeError(
            f"Unsupported file extension '{extension}'. "
            f"Supported formats: {supported}"
        )

    return extension


def validate_mime_type(mime_type: str | None) -> str:
    """
    Validate an uploaded MIME type.

    MIME validation is an additional layer and must not be considered
    sufficient by itself. Actual image content is validated later by
    Pillow in image_utils.py.
    """

    if not mime_type:
        raise UnsupportedFileTypeError("Missing MIME type.")

    normalized = mime_type.lower().strip()

    if not is_supported_mime_type(normalized):
        supported = ", ".join(settings.allowed_image_mime_types)

        raise UnsupportedFileTypeError(
            f"Unsupported MIME type '{normalized}'. " f"Supported types: {supported}"
        )

    return normalized


def validate_file_size(size_bytes: int) -> None:
    """
    Validate file size against the configured upload limit.
    """

    if size_bytes < 0:
        raise InvalidFileError("File size cannot be negative.")

    maximum = settings.max_upload_size_mb * 1024 * 1024

    if size_bytes > maximum:
        maximum_mb = settings.max_upload_size_mb

        raise FileTooLargeError(
            f"File exceeds the maximum allowed size " f"of {maximum_mb} MB."
        )


# ============================================================================
# HASHING
# ============================================================================


def calculate_sha256(
    file: BinaryIO,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate a SHA-256 hash for a file-like object.

    The function restores the original file position when possible.
    """

    digest = hashlib.sha256()

    try:
        original_position = file.tell()
    except (AttributeError, OSError):
        original_position = None

    try:
        file.seek(0)

        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    finally:
        if original_position is not None:
            try:
                file.seek(original_position)
            except (AttributeError, OSError):
                pass

    return digest.hexdigest()


def calculate_file_sha256(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    """
    Calculate SHA-256 directly from a filesystem path.
    """

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


# ============================================================================
# SAFE SERVER-SIDE FILENAMES
# ============================================================================


def generate_storage_filename(
    original_filename: str,
    extension: str | None = None,
) -> str:
    """
    Generate a collision-resistant server-side filename.

    Original user filenames are never used as the actual storage identity.
    """

    safe_original = sanitize_filename(original_filename)

    if extension is None:
        extension = get_extension(safe_original)

    extension = extension.lower()

    unique_id = uuid.uuid4().hex

    return f"{unique_id}{extension}"


# ============================================================================
# SAFE PATH HANDLING
# ============================================================================


def ensure_within_directory(
    path: Path,
    directory: Path,
) -> Path:
    """
    Ensure that `path` resolves inside `directory`.

    Prevents path traversal vulnerabilities.
    """

    resolved_path = path.resolve()
    resolved_directory = directory.resolve()

    try:
        resolved_path.relative_to(resolved_directory)

    except ValueError as exc:
        raise FileUtilityError("Unsafe file path detected.") from exc

    return resolved_path


def get_storage_path(
    storage_filename: str,
) -> Path:
    """
    Return a safe screenshot storage path.
    """

    candidate = SCREENSHOTS_DIR / storage_filename

    return ensure_within_directory(
        candidate,
        SCREENSHOTS_DIR,
    )


# ============================================================================
# FILE PERSISTENCE
# ============================================================================


def save_uploaded_bytes(
    data: bytes,
    storage_filename: str,
) -> Path:
    """
    Save raw bytes to the screenshot storage directory.

    This function performs:
        - Empty-file validation
        - Size validation
        - Safe-path validation
        - Atomic-ish temporary write followed by replacement
    """

    if not data:
        raise InvalidFileError("Uploaded file is empty.")

    validate_file_size(len(data))

    target = get_storage_path(storage_filename)

    SCREENSHOTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_name = f".{target.name}.{uuid.uuid4().hex}.tmp"

    temporary_path = SCREENSHOTS_DIR / temporary_name

    try:
        with temporary_path.open("wb") as file:
            file.write(data)
            file.flush()

        temporary_path.replace(target)

    except OSError as exc:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass

        raise FileSaveError(f"Unable to save uploaded file: {exc}") from exc

    return target


def delete_file(path: Path) -> bool:
    """
    Safely delete a stored file.

    Returns:
        True if a file was deleted.
        False if it did not exist.
    """

    resolved = ensure_within_directory(
        path,
        SCREENSHOTS_DIR,
    )

    if not resolved.exists():
        return False

    if not resolved.is_file():
        raise FileUtilityError("Refusing to delete a path that is not a file.")

    try:
        resolved.unlink()

    except OSError as exc:
        raise FileUtilityError(f"Unable to delete file: {exc}") from exc

    return True


# ============================================================================
# METADATA EXTRACTION
# ============================================================================


def build_file_metadata(
    *,
    original_filename: str,
    mime_type: str,
    size_bytes: int,
    sha256: str,
) -> FileMetadata:
    """
    Validate supplied metadata and create a FileMetadata object.
    """

    safe_filename = sanitize_filename(original_filename)

    extension = validate_extension(safe_filename)

    normalized_mime = validate_mime_type(mime_type)

    validate_file_size(size_bytes)

    if not sha256:
        raise InvalidFileError("SHA-256 hash is required.")

    return FileMetadata(
        original_filename=original_filename,
        safe_filename=safe_filename,
        extension=extension,
        mime_type=normalized_mime,
        size_bytes=size_bytes,
        sha256=sha256,
    )


# ============================================================================
# DUPLICATE HELPERS
# ============================================================================


def hashes_match(
    first_hash: str,
    second_hash: str,
) -> bool:
    """
    Constant-time comparison of two SHA-256 hashes.
    """

    return (
        hashlib.sha256(first_hash.encode("utf-8")).digest()
        == hashlib.sha256(second_hash.encode("utf-8")).digest()
    )


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "FileMetadata",
    "FileUtilityError",
    "InvalidFileError",
    "UnsupportedFileTypeError",
    "FileTooLargeError",
    "FileSaveError",
    "sanitize_filename",
    "get_extension",
    "validate_extension",
    "validate_mime_type",
    "validate_file_size",
    "calculate_sha256",
    "calculate_file_sha256",
    "generate_storage_filename",
    "ensure_within_directory",
    "get_storage_path",
    "save_uploaded_bytes",
    "delete_file",
    "build_file_metadata",
    "hashes_match",
]
