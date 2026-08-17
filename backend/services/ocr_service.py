"""
MemoryOS - OCR Service

Production-grade local OCR service built around Tesseract.

Responsibilities
----------------
- Detect Tesseract installation
- Process uploaded images
- Apply OCR-friendly preprocessing
- Run multiple OCR strategies
- Extract text
- Estimate OCR confidence
- Clean OCR output
- Preserve raw OCR output
- Gracefully handle missing Tesseract
- Never crash the complete indexing pipeline because one image fails

Architecture
------------

Image
  ↓
Image validation
  ↓
Orientation correction
  ↓
RGB conversion
  ↓
OCR preprocessing
  ↓
Tesseract
  ↓
Raw OCR
  ↓
Text normalization
  ↓
Confidence calculation
  ↓
OCRResult
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from pytesseract import Output
from pytesseract.pytesseract import TesseractNotFoundError

from backend.utils.image_utils import (
    ImageProcessingError,
    InvalidImageError,
    close_image,
    convert_to_rgb,
    correct_orientation,
    open_image,
)
from backend.utils.text_utils import (
    clean_ocr_text,
    is_meaningful_text,
    text_hash,
)

# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger("memoryos.ocr")


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_LANGUAGE = "eng"

DEFAULT_OCR_TIMEOUT = 30

# Tesseract Page Segmentation Modes
PSM_AUTO = 3
PSM_SINGLE_BLOCK = 6
PSM_SINGLE_LINE = 7
PSM_SPARSE_TEXT = 11


# ============================================================================
# EXCEPTIONS
# ============================================================================


class OCRServiceError(Exception):
    """Base OCR service exception."""


class OCRUnavailableError(OCRServiceError):
    """
    Raised when Tesseract is not installed or cannot be executed.
    """


class OCRProcessingError(OCRServiceError):
    """
    Raised when OCR processing fails.
    """


class OCRTimeoutError(OCRProcessingError):
    """
    Raised when Tesseract exceeds the configured timeout.
    """


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass(frozen=True, slots=True)
class OCRWord:
    """
    Represents one recognized OCR word.
    """

    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class OCRResult:
    """
    Complete OCR result for one image.
    """

    success: bool

    text: str

    raw_text: str

    confidence: float

    word_count: int

    character_count: int

    language: str

    engine: str

    processing_strategy: str

    available: bool

    error: str | None = None

    text_hash: str | None = None

    words: tuple[OCRWord, ...] = field(default_factory=tuple)

    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# TESSERACT DETECTION
# ============================================================================


def find_tesseract() -> str | None:
    """
    Find the Tesseract executable.

    Search order:

    1. TESSERACT_CMD environment variable
    2. PATH
    3. Common Windows installation locations
    """

    environment_path = os.getenv("TESSERACT_CMD")

    if environment_path:
        path = Path(environment_path)

        if path.exists():
            return str(path)

    executable = shutil.which("tesseract")

    if executable:
        return executable

    if os.name == "nt":
        common_paths = [
            Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
            Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
            Path(
                os.getenv(
                    "LOCALAPPDATA",
                    "",
                )
            )
            / "Programs"
            / "Tesseract-OCR"
            / "tesseract.exe",
        ]

        for path in common_paths:
            if path.exists():
                return str(path)

    return None


def configure_tesseract(
    command: str | None = None,
) -> str:
    """
    Configure pytesseract to use a specific Tesseract executable.
    """

    executable = command or find_tesseract()

    if not executable:
        raise OCRUnavailableError(
            "Tesseract OCR was not found. "
            "Install Tesseract OCR or set the "
            "TESSERACT_CMD environment variable."
        )

    pytesseract.pytesseract.tesseract_cmd = executable

    return executable


def is_tesseract_available() -> bool:
    """
    Return True when Tesseract can be located.
    """

    return find_tesseract() is not None


def get_tesseract_version() -> str | None:
    """
    Return the installed Tesseract version.

    Returns None when Tesseract is unavailable.
    """

    try:
        configure_tesseract()

        version = pytesseract.get_tesseract_version()

        return str(version)

    except (
        OCRUnavailableError,
        TesseractNotFoundError,
        OSError,
        RuntimeError,
    ):
        return None


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================


def _grayscale(
    image: Image.Image,
) -> Image.Image:
    """
    Convert image to grayscale.
    """

    return ImageOps.grayscale(image)


def _auto_contrast(
    image: Image.Image,
) -> Image.Image:
    """
    Improve contrast for OCR.
    """

    return ImageOps.autocontrast(image)


def _sharpen(
    image: Image.Image,
) -> Image.Image:
    """
    Apply light sharpening.
    """

    return image.filter(ImageFilter.SHARPEN)


def _upscale_for_ocr(
    image: Image.Image,
    minimum_width: int = 1200,
) -> Image.Image:
    """
    Upscale small images.

    Small screenshots frequently produce poor OCR.
    """

    if image.width >= minimum_width:
        return image.copy()

    scale = minimum_width / image.width

    new_width = minimum_width
    new_height = max(
        1,
        round(image.height * scale),
    )

    return image.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS,
    )


def preprocess_for_ocr(
    image: Image.Image,
) -> Image.Image:
    """
    Prepare an image for OCR.

    Pipeline:

        orientation
             ↓
          RGB
             ↓
        grayscale
             ↓
        autocontrast
             ↓
         sharpen
             ↓
         optional upscale
    """

    oriented = correct_orientation(image)

    try:
        rgb = convert_to_rgb(oriented)

        gray = _grayscale(rgb)

        contrast = _auto_contrast(gray)

        sharpened = _sharpen(contrast)

        upscaled = _upscale_for_ocr(sharpened)

        return upscaled

    finally:
        if oriented is not image:
            close_image(oriented)

        # Intermediate objects may share underlying resources, but
        # explicitly closing them is safe after the final copy is created.
        if "rgb" in locals():
            close_image(rgb)

        if "gray" in locals():
            close_image(gray)

        if "contrast" in locals():
            close_image(contrast)

        if "sharpened" in locals():
            close_image(sharpened)


# ============================================================================
# OCR WORD EXTRACTION
# ============================================================================


def _extract_words(
    image: Image.Image,
    language: str,
    psm: int,
    timeout: int,
) -> tuple[OCRWord, ...]:
    """
    Extract word-level OCR information.
    """

    try:
        data = pytesseract.image_to_data(
            image,
            lang=language,
            config=f"--psm {psm}",
            output_type=Output.DICT,
            timeout=timeout,
        )

    except RuntimeError as exc:
        message = str(exc).lower()

        if "timeout" in message:
            raise OCRTimeoutError("Tesseract OCR timed out.") from exc

        raise OCRProcessingError(f"Tesseract word extraction failed: {exc}") from exc

    words: list[OCRWord] = []

    texts = data.get(
        "text",
        [],
    )

    confidences = data.get(
        "conf",
        [],
    )

    lefts = data.get(
        "left",
        [],
    )

    tops = data.get(
        "top",
        [],
    )

    widths = data.get(
        "width",
        [],
    )

    heights = data.get(
        "height",
        [],
    )

    for index, value in enumerate(texts):
        text = str(value).strip()

        if not text:
            continue

        try:
            confidence = float(confidences[index])
        except (
            ValueError,
            TypeError,
            IndexError,
        ):
            confidence = 0.0

        if confidence < 0:
            confidence = 0.0

        try:
            left = int(lefts[index])
            top = int(tops[index])
            width = int(widths[index])
            height = int(heights[index])
        except (
            ValueError,
            TypeError,
            IndexError,
        ):
            left = 0
            top = 0
            width = 0
            height = 0

        words.append(
            OCRWord(
                text=text,
                confidence=round(
                    confidence,
                    2,
                ),
                left=left,
                top=top,
                width=width,
                height=height,
            )
        )

    return tuple(words)


# ============================================================================
# OCR EXECUTION
# ============================================================================


def _run_tesseract(
    image: Image.Image,
    language: str,
    psm: int,
    timeout: int,
) -> tuple[str, tuple[OCRWord, ...]]:
    """
    Execute Tesseract and return text + word metadata.
    """

    try:
        raw_text = pytesseract.image_to_string(
            image,
            lang=language,
            config=f"--psm {psm}",
            timeout=timeout,
        )

    except TesseractNotFoundError as exc:
        raise OCRUnavailableError("Tesseract executable could not be found.") from exc

    except RuntimeError as exc:
        message = str(exc).lower()

        if "timeout" in message:
            raise OCRTimeoutError("Tesseract OCR timed out.") from exc

        raise OCRProcessingError(f"Tesseract OCR failed: {exc}") from exc

    except OSError as exc:
        raise OCRProcessingError(
            f"Tesseract could not process the image: {exc}"
        ) from exc

    words = _extract_words(
        image=image,
        language=language,
        psm=psm,
        timeout=timeout,
    )

    return (
        raw_text,
        words,
    )


# ============================================================================
# CONFIDENCE
# ============================================================================


def calculate_confidence(
    words: tuple[OCRWord, ...],
) -> float:
    """
    Calculate average confidence from recognized words.
    """

    valid_confidences = [word.confidence for word in words if word.confidence >= 0]

    if not valid_confidences:
        return 0.0

    return round(
        sum(valid_confidences) / len(valid_confidences),
        2,
    )


# ============================================================================
# STRATEGY SELECTION
# ============================================================================


def _run_strategy(
    image: Image.Image,
    language: str,
    psm: int,
    timeout: int,
) -> OCRResult:
    """
    Run one OCR strategy.
    """

    raw_text, words = _run_tesseract(
        image=image,
        language=language,
        psm=psm,
        timeout=timeout,
    )

    cleaned_text = clean_ocr_text(raw_text)

    confidence = calculate_confidence(words)

    return OCRResult(
        success=is_meaningful_text(cleaned_text),
        text=cleaned_text,
        raw_text=raw_text.strip(),
        confidence=confidence,
        word_count=len(words),
        character_count=len(cleaned_text),
        language=language,
        engine="tesseract",
        processing_strategy=f"psm_{psm}",
        available=True,
        error=None,
        text_hash=(text_hash(cleaned_text) if cleaned_text else None),
        words=words,
    )


def _score_result(
    result: OCRResult,
) -> float:
    """
    Score an OCR result for choosing the best strategy.

    Confidence alone is not enough.

    We also consider:
        - meaningful text
        - word count
        - character count
    """

    if not result.success:
        return 0.0

    confidence_score = result.confidence / 100.0

    word_score = min(
        result.word_count / 30.0,
        1.0,
    )

    character_score = min(
        result.character_count / 300.0,
        1.0,
    )

    return confidence_score * 0.60 + word_score * 0.20 + character_score * 0.20


# ============================================================================
# MAIN OCR SERVICE
# ============================================================================


class OCRService:
    """
    Production OCR service.

    The service is intentionally stateless.

    That makes it:
        - easy to test
        - safe to reuse
        - suitable for FastAPI
        - easy to replace later
    """

    def __init__(
        self,
        language: str = DEFAULT_LANGUAGE,
        timeout: int = DEFAULT_OCR_TIMEOUT,
        tesseract_command: str | None = None,
    ) -> None:
        if not language:
            raise ValueError("OCR language cannot be empty.")

        if timeout <= 0:
            raise ValueError("OCR timeout must be greater than zero.")

        self.language = language
        self.timeout = timeout
        self.tesseract_command = tesseract_command

    # ---------------------------------------------------------------------
    # Availability
    # ---------------------------------------------------------------------

    def available(self) -> bool:
        """
        Check whether Tesseract is available.
        """

        return find_tesseract() is not None

    # ---------------------------------------------------------------------
    # Version
    # ---------------------------------------------------------------------

    def version(self) -> str | None:
        """
        Return Tesseract version.
        """

        try:
            configure_tesseract(self.tesseract_command)

            return get_tesseract_version()

        except OCRUnavailableError:
            return None

    # ---------------------------------------------------------------------
    # Single OCR
    # ---------------------------------------------------------------------

    def extract_text(
        self,
        source: bytes | bytearray | memoryview | Any,
        *,
        strategy: str = "auto",
    ) -> OCRResult:
        """
        Extract text from one image.

        strategy:
            auto
            single_block
            sparse
            single_line
        """

        try:
            configure_tesseract(self.tesseract_command)

        except OCRUnavailableError as exc:
            logger.warning(
                "OCR unavailable: %s",
                exc,
            )

            return OCRResult(
                success=False,
                text="",
                raw_text="",
                confidence=0.0,
                word_count=0,
                character_count=0,
                language=self.language,
                engine="tesseract",
                processing_strategy="unavailable",
                available=False,
                error=str(exc),
            )

        image: Image.Image | None = None
        prepared: Image.Image | None = None

        try:
            image = open_image(source)

            prepared = preprocess_for_ocr(image)

            if strategy == "auto":
                return self._extract_auto(prepared)

            psm = self._strategy_to_psm(strategy)

            return _run_strategy(
                image=prepared,
                language=self.language,
                psm=psm,
                timeout=self.timeout,
            )

        except (
            InvalidImageError,
            ImageProcessingError,
            OCRProcessingError,
            OCRUnavailableError,
        ) as exc:
            logger.exception("OCR processing failed.")

            return OCRResult(
                success=False,
                text="",
                raw_text="",
                confidence=0.0,
                word_count=0,
                character_count=0,
                language=self.language,
                engine="tesseract",
                processing_strategy=strategy,
                available=self.available(),
                error=str(exc),
            )

        finally:
            close_image(prepared)

            close_image(image)

    # ---------------------------------------------------------------------
    # Automatic OCR
    # ---------------------------------------------------------------------

    def _extract_auto(
        self,
        image: Image.Image,
    ) -> OCRResult:
        """
        Run multiple OCR strategies and choose the best result.
        """

        strategies = [
            (
                "auto",
                PSM_AUTO,
            ),
            (
                "single_block",
                PSM_SINGLE_BLOCK,
            ),
            (
                "sparse",
                PSM_SPARSE_TEXT,
            ),
        ]

        results: list[OCRResult] = []

        for name, psm in strategies:
            try:
                result = _run_strategy(
                    image=image,
                    language=self.language,
                    psm=psm,
                    timeout=self.timeout,
                )

                # Replace generic strategy label with useful information.
                result = OCRResult(
                    success=result.success,
                    text=result.text,
                    raw_text=result.raw_text,
                    confidence=result.confidence,
                    word_count=result.word_count,
                    character_count=result.character_count,
                    language=result.language,
                    engine=result.engine,
                    processing_strategy=name,
                    available=result.available,
                    error=result.error,
                    text_hash=result.text_hash,
                    words=result.words,
                    metadata=result.metadata,
                )

                results.append(result)

            except OCRProcessingError as exc:
                logger.warning(
                    "OCR strategy %s failed: %s",
                    name,
                    exc,
                )

        if not results:
            raise OCRProcessingError("All OCR strategies failed.")

        return max(
            results,
            key=_score_result,
        )

    # ---------------------------------------------------------------------
    # Strategy conversion
    # ---------------------------------------------------------------------

    @staticmethod
    def _strategy_to_psm(
        strategy: str,
    ) -> int:
        mapping = {
            "auto": PSM_AUTO,
            "single_block": PSM_SINGLE_BLOCK,
            "single_line": PSM_SINGLE_LINE,
            "sparse": PSM_SPARSE_TEXT,
        }

        normalized = strategy.strip().lower()

        if normalized not in mapping:
            raise ValueError(
                f"Unknown OCR strategy: {strategy}. "
                f"Supported strategies: "
                f"{', '.join(mapping)}"
            )

        return mapping[normalized]


# ============================================================================
# DEFAULT SERVICE
# ============================================================================


ocr_service = OCRService()


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def extract_text(
    source: bytes | bytearray | memoryview | Any,
    *,
    language: str = DEFAULT_LANGUAGE,
    timeout: int = DEFAULT_OCR_TIMEOUT,
) -> OCRResult:
    """
    Convenience wrapper around OCRService.
    """

    service = OCRService(
        language=language,
        timeout=timeout,
    )

    return service.extract_text(source)


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "OCRWord",
    "OCRResult",
    "OCRServiceError",
    "OCRUnavailableError",
    "OCRProcessingError",
    "OCRTimeoutError",
    "find_tesseract",
    "configure_tesseract",
    "is_tesseract_available",
    "get_tesseract_version",
    "preprocess_for_ocr",
    "calculate_confidence",
    "OCRService",
    "ocr_service",
    "extract_text",
]
