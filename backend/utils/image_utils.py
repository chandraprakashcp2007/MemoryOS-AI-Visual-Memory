"""
MemoryOS - Image Utilities

Production-grade image processing utilities.

Responsibilities:
    - Validate image files
    - Safely open images
    - Verify actual image content
    - Extract image metadata
    - Correct EXIF orientation
    - Resize oversized images
    - Generate thumbnails
    - Convert images to RGB
    - Prepare images for OCR / Vision AI
    - Calculate perceptual image information

Security:
    - Never trust file extensions alone
    - Never trust MIME types alone
    - Pillow verifies actual image content
    - Decompression-bomb protection
    - Controlled image dimensions
    - Controlled output formats

The module is intentionally independent from OCR and Gemini.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageFile, ImageOps, UnidentifiedImageError

from backend.config import settings

# ============================================================================
# PILLOW SECURITY CONFIGURATION
# ============================================================================

# Allow Pillow to process slightly damaged images when possible.
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Pillow's decompression-bomb protection.
#
# A malicious image can claim extremely large dimensions and consume huge
# amounts of memory when decoded. We keep Pillow's protection enabled.
Image.MAX_IMAGE_PIXELS = 50_000_000


# ============================================================================
# EXCEPTIONS
# ============================================================================


class ImageUtilityError(Exception):
    """Base exception for image processing errors."""


class InvalidImageError(ImageUtilityError):
    """Raised when the supplied file is not a valid image."""


class ImageTooLargeError(ImageUtilityError):
    """Raised when an image exceeds configured dimension limits."""


class ImageProcessingError(ImageUtilityError):
    """Raised when an image cannot be processed."""


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """
    Basic technical metadata for an image.
    """

    width: int
    height: int
    mode: str
    format: str
    mime_type: str
    has_alpha: bool
    file_size_bytes: int


@dataclass(frozen=True, slots=True)
class PreparedImage:
    """
    Result returned after preparing an image for AI/OCR processing.
    """

    image: Image.Image
    width: int
    height: int
    original_width: int
    original_height: int
    resized: bool
    format: str


# ============================================================================
# INTERNAL HELPERS
# ============================================================================


def _read_bytes(
    source: bytes | bytearray | memoryview | BinaryIO | Path,
) -> bytes:
    """
    Convert a supported image source into bytes.
    """

    if isinstance(source, bytes):
        return source

    if isinstance(source, bytearray):
        return bytes(source)

    if isinstance(source, memoryview):
        return source.tobytes()

    if isinstance(source, Path):
        try:
            return source.read_bytes()
        except OSError as exc:
            raise ImageProcessingError(f"Unable to read image file: {source}") from exc

    if hasattr(source, "read"):
        try:
            current_position = source.tell()
        except (AttributeError, OSError):
            current_position = None

        try:
            source.seek(0)
            data = source.read()
        except (AttributeError, OSError) as exc:
            raise ImageProcessingError("Unable to read image stream.") from exc
        finally:
            if current_position is not None:
                try:
                    source.seek(current_position)
                except (AttributeError, OSError):
                    pass

        if not isinstance(data, bytes):
            data = bytes(data)

        return data

    raise TypeError(
        "Unsupported image source. " "Expected bytes, file-like object, or Path."
    )


def _validate_dimensions(
    width: int,
    height: int,
) -> None:
    """
    Validate image dimensions against MemoryOS limits.
    """

    if width <= 0 or height <= 0:
        raise InvalidImageError("Image dimensions must be greater than zero.")

    maximum_dimension = getattr(
        settings,
        "max_image_dimension",
        8192,
    )

    if width > maximum_dimension or height > maximum_dimension:
        raise ImageTooLargeError(
            f"Image dimensions {width}x{height} exceed "
            f"the maximum allowed dimension of "
            f"{maximum_dimension}px."
        )


def _detect_mime_type(
    image_format: str,
) -> str:
    """
    Convert Pillow format names into MIME types.
    """

    mapping = {
        "JPEG": "image/jpeg",
        "JPG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "GIF": "image/gif",
        "BMP": "image/bmp",
        "TIFF": "image/tiff",
        "TIF": "image/tiff",
    }

    return mapping.get(
        image_format.upper(),
        "application/octet-stream",
    )


# ============================================================================
# IMAGE VALIDATION
# ============================================================================


def verify_image(
    source: bytes | bytearray | memoryview | BinaryIO | Path,
) -> ImageMetadata:
    """
    Verify that the supplied data is a real readable image.

    This is stronger than checking:
        - filename extension
        - MIME type

    Pillow actually parses the image structure.
    """

    data = _read_bytes(source)

    if not data:
        raise InvalidImageError("Image data is empty.")

    try:
        with Image.open(io.BytesIO(data)) as image:
            image.verify()

        # Re-open after verify() because Pillow invalidates the image
        # object after verification.
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            image_format = image.format or "UNKNOWN"
            mode = image.mode

            _validate_dimensions(
                width,
                height,
            )

            return ImageMetadata(
                width=width,
                height=height,
                mode=mode,
                format=image_format,
                mime_type=_detect_mime_type(image_format),
                has_alpha=("A" in image.getbands() or image_format.upper() == "PNG"),
                file_size_bytes=len(data),
            )

    except Image.DecompressionBombError as exc:
        raise InvalidImageError("Image exceeds Pillow's safe pixel limit.") from exc

    except Image.DecompressionBombWarning:
        raise InvalidImageError("Image dimensions are unsafe.")

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise InvalidImageError(
            "The supplied file is not a valid readable image."
        ) from exc


# ============================================================================
# IMAGE OPENING
# ============================================================================


def open_image(
    source: bytes | bytearray | memoryview | BinaryIO | Path,
) -> Image.Image:
    """
    Safely open an image and return a detached Pillow image.

    The returned image is fully loaded into memory so the underlying
    source can safely be closed.
    """

    data = _read_bytes(source)

    if not data:
        raise InvalidImageError("Image data is empty.")

    try:
        image = Image.open(io.BytesIO(data))

        image.load()

        _validate_dimensions(
            image.width,
            image.height,
        )

        return image.copy()

    except Image.DecompressionBombError as exc:
        raise InvalidImageError("Image exceeds the safe pixel limit.") from exc

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise InvalidImageError("Unable to decode image.") from exc


# ============================================================================
# EXIF ORIENTATION
# ============================================================================


def correct_orientation(
    image: Image.Image,
) -> Image.Image:
    """
    Apply EXIF orientation information.

    Smartphone screenshots/photos can contain orientation metadata.
    OCR and vision models work better when pixels are physically oriented.
    """

    try:
        corrected = ImageOps.exif_transpose(image)

        if corrected is None:
            return image.copy()

        return corrected

    except (
        AttributeError,
        ValueError,
        OSError,
    ) as exc:
        raise ImageProcessingError("Unable to correct image orientation.") from exc


# ============================================================================
# COLOR NORMALIZATION
# ============================================================================


def convert_to_rgb(
    image: Image.Image,
    background: tuple[int, int, int] = (255, 255, 255),
) -> Image.Image:
    """
    Convert an image into RGB.

    Transparent images are composited against a white background.

    This produces a consistent format for:
        - OCR
        - computer vision
        - embeddings
        - Gemini Vision
    """

    if image.mode == "RGB":
        return image.copy()

    if image.mode in ("RGBA", "LA"):
        rgba = image.convert("RGBA")

        background_image = Image.new(
            "RGB",
            rgba.size,
            background,
        )

        background_image.paste(
            rgba,
            mask=rgba.getchannel("A"),
        )

        return background_image

    if image.mode == "P":
        if "transparency" in image.info:
            rgba = image.convert("RGBA")

            background_image = Image.new(
                "RGB",
                rgba.size,
                background,
            )

            background_image.paste(
                rgba,
                mask=rgba.getchannel("A"),
            )

            return background_image

        return image.convert("RGB")

    return image.convert("RGB")


# ============================================================================
# RESIZING
# ============================================================================


def calculate_resize_dimensions(
    width: int,
    height: int,
    max_dimension: int,
) -> tuple[int, int]:
    """
    Calculate aspect-ratio-preserving dimensions.
    """

    if width <= 0 or height <= 0:
        raise ValueError("Width and height must be positive.")

    if max_dimension <= 0:
        raise ValueError("max_dimension must be positive.")

    largest_side = max(
        width,
        height,
    )

    if largest_side <= max_dimension:
        return width, height

    scale = max_dimension / largest_side

    new_width = max(
        1,
        round(width * scale),
    )

    new_height = max(
        1,
        round(height * scale),
    )

    return new_width, new_height


def resize_image(
    image: Image.Image,
    max_dimension: int | None = None,
) -> tuple[Image.Image, bool]:
    """
    Resize an image while preserving its aspect ratio.

    Returns:
        (image, resized)
    """

    if max_dimension is None:
        max_dimension = getattr(
            settings,
            "max_image_dimension",
            8192,
        )

    new_width, new_height = calculate_resize_dimensions(
        image.width,
        image.height,
        max_dimension,
    )

    if new_width == image.width and new_height == image.height:
        return image.copy(), False

    resized = image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )

    return resized, True


# ============================================================================
# THUMBNAILS
# ============================================================================


def create_thumbnail(
    image: Image.Image,
    size: tuple[int, int] = (480, 480),
) -> Image.Image:
    """
    Create a high-quality thumbnail while preserving aspect ratio.
    """

    if size[0] <= 0 or size[1] <= 0:
        raise ValueError("Thumbnail dimensions must be positive.")

    thumbnail = image.copy()

    thumbnail.thumbnail(
        size,
        Image.Resampling.LANCZOS,
    )

    return thumbnail


def save_thumbnail(
    image: Image.Image,
    destination: Path,
    size: tuple[int, int] = (480, 480),
    quality: int = 88,
) -> Path:
    """
    Generate and save a JPEG thumbnail.

    JPEG is used for predictable browser/API compatibility.
    """

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    thumbnail = create_thumbnail(
        convert_to_rgb(image),
        size,
    )

    try:
        thumbnail.save(
            destination,
            format="JPEG",
            quality=quality,
            optimize=True,
        )

    except OSError as exc:
        raise ImageProcessingError(f"Unable to save thumbnail: {destination}") from exc

    finally:
        thumbnail.close()

    return destination


# ============================================================================
# IMAGE PREPARATION
# ============================================================================


def prepare_for_ai(
    source: bytes | bytearray | memoryview | BinaryIO | Path,
    max_dimension: int | None = None,
) -> PreparedImage:
    """
    Prepare an uploaded image for OCR and Vision AI.

    Pipeline:

        raw source
             ↓
        decode image
             ↓
        EXIF orientation
             ↓
        RGB conversion
             ↓
        dimension normalization
             ↓
        ready for AI
    """

    image = open_image(source)

    original_width = image.width
    original_height = image.height

    try:
        oriented = correct_orientation(image)

        if oriented is not image:
            image.close()

        rgb = convert_to_rgb(oriented)

        if rgb is not oriented:
            oriented.close()

        prepared, resized = resize_image(
            rgb,
            max_dimension,
        )

        if prepared is not rgb:
            rgb.close()

        return PreparedImage(
            image=prepared,
            width=prepared.width,
            height=prepared.height,
            original_width=original_width,
            original_height=original_height,
            resized=resized,
            format="JPEG-compatible RGB",
        )

    except Exception:
        image.close()
        raise


# ============================================================================
# ENCODING
# ============================================================================


def image_to_bytes(
    image: Image.Image,
    format: str = "JPEG",
    quality: int = 92,
) -> bytes:
    """
    Encode a Pillow image into bytes.

    JPEG output is the default because it is compact and widely supported.
    """

    output = io.BytesIO()

    normalized_format = format.upper()

    if normalized_format in {
        "JPEG",
        "JPG",
    }:
        image_to_save = convert_to_rgb(image)

        image_to_save.save(
            output,
            format="JPEG",
            quality=quality,
            optimize=True,
        )

        image_to_save.close()

    elif normalized_format == "PNG":
        image.save(
            output,
            format="PNG",
            optimize=True,
        )

    elif normalized_format == "WEBP":
        image.save(
            output,
            format="WEBP",
            quality=quality,
        )

    else:
        raise ValueError(f"Unsupported output format: {format}")

    return output.getvalue()


# ============================================================================
# IMAGE HASH
# ============================================================================


def calculate_image_hash(
    image: Image.Image,
) -> str:
    """
    Calculate a deterministic SHA-256 hash from normalized image pixels.

    Unlike a file hash, this can identify visually identical images that
    were saved using different filenames or metadata.
    """

    normalized = convert_to_rgb(image)

    try:
        pixel_data = normalized.tobytes()

        metadata = (
            f"{normalized.width}x" f"{normalized.height}|" f"{normalized.mode}|"
        ).encode("utf-8")

        return hashlib.sha256(metadata + pixel_data).hexdigest()

    finally:
        normalized.close()


# ============================================================================
# IMAGE INFORMATION
# ============================================================================


def get_image_metadata(
    source: bytes | bytearray | memoryview | BinaryIO | Path,
) -> ImageMetadata:
    """
    Public convenience wrapper around image verification.
    """

    return verify_image(source)


# ============================================================================
# RESOURCE MANAGEMENT
# ============================================================================


def close_image(
    image: Image.Image | None,
) -> None:
    """
    Safely close a Pillow image.
    """

    if image is None:
        return

    try:
        image.close()
    except (
        AttributeError,
        OSError,
    ):
        pass


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "ImageMetadata",
    "PreparedImage",
    "ImageUtilityError",
    "InvalidImageError",
    "ImageTooLargeError",
    "ImageProcessingError",
    "verify_image",
    "open_image",
    "correct_orientation",
    "convert_to_rgb",
    "calculate_resize_dimensions",
    "resize_image",
    "create_thumbnail",
    "save_thumbnail",
    "prepare_for_ai",
    "image_to_bytes",
    "calculate_image_hash",
    "get_image_metadata",
    "close_image",
]
