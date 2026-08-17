"""
MemoryOS - Text Utilities

Production-grade text processing utilities used throughout MemoryOS.

Responsibilities
----------------
- Normalize OCR text
- Remove invisible/control characters
- Normalize whitespace
- Preserve meaningful punctuation
- Clean repeated OCR artifacts
- Normalize Unicode
- Extract useful text statistics
- Generate searchable text
- Create deterministic text hashes
- Detect whether text is meaningful
- Split long text into safe chunks
- Prepare text for embeddings

Design principle
----------------
This module does NOT perform AI reasoning.

It only prepares text so that downstream systems receive
clean, consistent, deterministic input.

Pipeline:

    OCR / Gemini
          ↓
    text_utils.py
          ↓
    normalized text
          ↓
    classification
          ↓
    entity extraction
          ↓
    intent detection
          ↓
    embedding
          ↓
    FAISS
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_MAX_TEXT_LENGTH = 100_000
DEFAULT_CHUNK_SIZE = 1_500
DEFAULT_CHUNK_OVERLAP = 150

# Common Unicode characters that are useful in human text.
# We deliberately do not remove normal punctuation because dates,
# prices, emails, URLs and product names may depend on punctuation.
ZERO_WIDTH_PATTERN = re.compile(r"[\u200b\u200c\u200d\ufeff]")

CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

MULTI_SPACE_PATTERN = re.compile(r"[ \t]+")

MULTI_NEWLINE_PATTERN = re.compile(r"\n{3,}")

OCR_BULLET_PATTERN = re.compile(
    r"^[\s]*[•·▪▫◦●○■□◆◇►▶➤➢]\s*",
    re.MULTILINE,
)

OCR_LINE_ARTIFACT_PATTERN = re.compile(
    r"^[\s]*[-_=]{4,}[\s]*$",
    re.MULTILINE,
)

REPEATED_CHARACTER_PATTERN = re.compile(r"(.)\1{5,}")

EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")

URL_PATTERN = re.compile(
    r"\b(?:https?://|www\.)[^\s<>]+",
    re.IGNORECASE,
)

PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")

MONEY_PATTERN = re.compile(
    r"""
    (?:
        ₹\s?\d[\d,]*(?:\.\d+)?
        |
        Rs\.?\s?\d[\d,]*(?:\.\d+)?
        |
        INR\s?\d[\d,]*(?:\.\d+)?
        |
        \$\s?\d[\d,]*(?:\.\d+)?
        |
        €\s?\d[\d,]*(?:\.\d+)?
        |
        £\s?\d[\d,]*(?:\.\d+)?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass(frozen=True, slots=True)
class TextStatistics:
    """
    Useful statistics about a processed text document.
    """

    characters: int
    characters_no_spaces: int
    words: int
    lines: int
    sentences: int
    paragraphs: int
    unique_words: int
    average_word_length: float
    has_numbers: bool
    has_email: bool
    has_url: bool
    has_phone: bool
    has_money: bool


@dataclass(frozen=True, slots=True)
class TextChunk:
    """
    Represents a searchable chunk of a larger text.
    """

    text: str
    index: int
    start: int
    end: int


# ============================================================================
# BASIC VALIDATION
# ============================================================================


def ensure_text(
    value: object,
) -> str:
    """
    Convert a supported value into a string.

    None becomes an empty string.

    This function intentionally avoids surprising conversions for complex
    objects such as lists and dictionaries.
    """

    if value is None:
        return ""

    if isinstance(value, str):
        return value

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        )

    return str(value)


def is_meaningful_text(
    text: str | None,
    minimum_characters: int = 2,
    minimum_alphanumeric: int = 1,
) -> bool:
    """
    Determine whether text contains enough meaningful information.

    Useful for rejecting OCR results such as:

        "..."
        "---"
        "   "
        "###"
    """

    if not text:
        return False

    normalized = normalize_text(text)

    if len(normalized) < minimum_characters:
        return False

    alphanumeric_count = sum(1 for character in normalized if character.isalnum())

    return alphanumeric_count >= minimum_alphanumeric


# ============================================================================
# UNICODE NORMALIZATION
# ============================================================================


def normalize_unicode(
    text: str,
    form: str = "NFKC",
) -> str:
    """
    Normalize Unicode characters.

    NFKC is useful for OCR because visually similar representations
    can otherwise become different strings.

    Example:

        full-width characters
                ↓
        normalized characters
    """

    value = ensure_text(text)

    try:
        normalized = unicodedata.normalize(
            form,
            value,
        )
    except ValueError as exc:
        raise ValueError(f"Invalid Unicode normalization form: {form}") from exc

    return normalized


def remove_invisible_characters(
    text: str,
) -> str:
    """
    Remove zero-width and unsafe control characters while preserving
    normal whitespace and punctuation.
    """

    value = ensure_text(text)

    value = ZERO_WIDTH_PATTERN.sub(
        "",
        value,
    )

    value = CONTROL_CHARACTER_PATTERN.sub(
        "",
        value,
    )

    return value


# ============================================================================
# WHITESPACE NORMALIZATION
# ============================================================================


def normalize_whitespace(
    text: str,
    preserve_line_breaks: bool = True,
) -> str:
    """
    Normalize unnecessary spaces and line breaks.

    When preserve_line_breaks=True:

        "hello      world"

    becomes:

        "hello world"

    while meaningful line structure is retained.
    """

    value = ensure_text(text)

    if preserve_line_breaks:
        lines = value.splitlines()

        cleaned_lines: list[str] = []

        for line in lines:
            cleaned = MULTI_SPACE_PATTERN.sub(
                " ",
                line.strip(),
            )

            cleaned_lines.append(cleaned)

        value = "\n".join(cleaned_lines)

        value = MULTI_NEWLINE_PATTERN.sub(
            "\n\n",
            value,
        )

    else:
        value = re.sub(
            r"\s+",
            " ",
            value,
        )

    return value.strip()


# ============================================================================
# OCR CLEANING
# ============================================================================


def remove_ocr_artifacts(
    text: str,
) -> str:
    """
    Remove common visual artifacts produced by OCR.

    This is deliberately conservative.

    We should never aggressively "correct" text here because
    downstream AI should receive as much original information
    as possible.
    """

    value = ensure_text(text)

    value = OCR_BULLET_PATTERN.sub(
        "",
        value,
    )

    value = OCR_LINE_ARTIFACT_PATTERN.sub(
        "",
        value,
    )

    # Extremely long repeated characters are usually OCR noise.
    value = REPEATED_CHARACTER_PATTERN.sub(
        r"\1\1\1",
        value,
    )

    return value


def clean_ocr_text(
    text: str,
) -> str:
    """
    Main OCR-cleaning pipeline.
    """

    value = ensure_text(text)

    if not value:
        return ""

    value = normalize_unicode(value)

    value = remove_invisible_characters(value)

    value = remove_ocr_artifacts(value)

    value = normalize_whitespace(
        value,
        preserve_line_breaks=True,
    )

    return value


# ============================================================================
# GENERAL TEXT NORMALIZATION
# ============================================================================


def normalize_text(
    text: str,
    preserve_line_breaks: bool = True,
) -> str:
    """
    Complete general-purpose text normalization pipeline.

    This is the primary function that downstream services should use.
    """

    value = ensure_text(text)

    if not value:
        return ""

    value = normalize_unicode(value)

    value = remove_invisible_characters(value)

    value = remove_ocr_artifacts(value)

    value = normalize_whitespace(
        value,
        preserve_line_breaks=preserve_line_breaks,
    )

    return value


# ============================================================================
# SEARCH NORMALIZATION
# ============================================================================


def normalize_for_search(
    text: str,
) -> str:
    """
    Produce a deterministic representation optimized for search.

    Important:
        This does NOT replace the original text.

    Original text should always be preserved separately.

    Search normalization:
        - Unicode normalization
        - lowercase
        - whitespace normalization
        - harmless punctuation cleanup
    """

    value = normalize_text(
        text,
        preserve_line_breaks=False,
    )

    value = value.casefold()

    # Preserve characters useful for:
    # emails, prices, versions, product codes and dates.
    value = re.sub(
        r"[^\w\s@.+#%₹$€£:/_-]",
        " ",
        value,
        flags=re.UNICODE,
    )

    value = normalize_whitespace(
        value,
        preserve_line_breaks=False,
    )

    return value


# ============================================================================
# EMBEDDING PREPARATION
# ============================================================================


def prepare_for_embedding(
    text: str,
    max_length: int = DEFAULT_MAX_TEXT_LENGTH,
) -> str:
    """
    Prepare text for a sentence-transformer embedding model.

    We retain semantic information while removing unnecessary formatting.
    """

    if max_length <= 0:
        raise ValueError("max_length must be greater than zero.")

    value = normalize_text(
        text,
        preserve_line_breaks=False,
    )

    if len(value) > max_length:
        value = value[:max_length]

    return value


# ============================================================================
# WORD PROCESSING
# ============================================================================


def tokenize_words(
    text: str,
) -> list[str]:
    """
    Extract Unicode-aware word-like tokens.

    This intentionally avoids external NLP dependencies.
    """

    normalized = normalize_text(
        text,
        preserve_line_breaks=False,
    )

    if not normalized:
        return []

    return re.findall(
        r"\b[\w'-]+\b",
        normalized,
        flags=re.UNICODE,
    )


def unique_words(
    text: str,
) -> set[str]:
    """
    Return unique case-insensitive words.
    """

    return {word.casefold() for word in tokenize_words(text)}


# ============================================================================
# SENTENCE / PARAGRAPH HELPERS
# ============================================================================


def split_sentences(
    text: str,
) -> list[str]:
    """
    Lightweight sentence splitter.

    This is intentionally dependency-free.
    Advanced NLP sentence segmentation can be added later.
    """

    normalized = normalize_text(
        text,
        preserve_line_breaks=True,
    )

    if not normalized:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        normalized,
    )

    return [sentence.strip() for sentence in sentences if sentence.strip()]


def split_paragraphs(
    text: str,
) -> list[str]:
    """
    Split text into meaningful paragraphs.
    """

    normalized = normalize_text(
        text,
        preserve_line_breaks=True,
    )

    if not normalized:
        return []

    paragraphs = re.split(
        r"\n\s*\n",
        normalized,
    )

    return [paragraph.strip() for paragraph in paragraphs if paragraph.strip()]


# ============================================================================
# ENTITY-LIKE EXTRACTION
# ============================================================================


def extract_emails(
    text: str,
) -> list[str]:
    """
    Extract email addresses.
    """

    return list(dict.fromkeys(EMAIL_PATTERN.findall(ensure_text(text))))


def extract_urls(
    text: str,
) -> list[str]:
    """
    Extract HTTP/HTTPS/www URLs.
    """

    return list(dict.fromkeys(URL_PATTERN.findall(ensure_text(text))))


def extract_phone_numbers(
    text: str,
) -> list[str]:
    """
    Extract phone-like numeric sequences.

    This is intentionally conservative and does not claim that
    every numeric sequence is a phone number.
    """

    matches = PHONE_PATTERN.findall(ensure_text(text))

    return list(dict.fromkeys(match.strip() for match in matches))


def extract_money_values(
    text: str,
) -> list[str]:
    """
    Extract common currency expressions.
    """

    matches = MONEY_PATTERN.findall(ensure_text(text))

    return list(dict.fromkeys(match.strip() for match in matches))


# ============================================================================
# TEXT STATISTICS
# ============================================================================


def count_sentences(
    text: str,
) -> int:
    """
    Count detected sentences.
    """

    return len(split_sentences(text))


def count_paragraphs(
    text: str,
) -> int:
    """
    Count detected paragraphs.
    """

    return len(split_paragraphs(text))


def get_text_statistics(
    text: str,
) -> TextStatistics:
    """
    Calculate useful statistics for a memory.
    """

    normalized = normalize_text(
        text,
        preserve_line_breaks=True,
    )

    words = tokenize_words(normalized)

    total_word_characters = sum(len(word) for word in words)

    average_word_length = total_word_characters / len(words) if words else 0.0

    return TextStatistics(
        characters=len(normalized),
        characters_no_spaces=len(
            re.sub(
                r"\s+",
                "",
                normalized,
            )
        ),
        words=len(words),
        lines=(len(normalized.splitlines()) if normalized else 0),
        sentences=count_sentences(normalized),
        paragraphs=count_paragraphs(normalized),
        unique_words=len(unique_words(normalized)),
        average_word_length=round(
            average_word_length,
            2,
        ),
        has_numbers=bool(
            re.search(
                r"\d",
                normalized,
            )
        ),
        has_email=bool(extract_emails(normalized)),
        has_url=bool(extract_urls(normalized)),
        has_phone=bool(extract_phone_numbers(normalized)),
        has_money=bool(extract_money_values(normalized)),
    )


# ============================================================================
# CHUNKING
# ============================================================================


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """
    Split long text into overlapping chunks.

    Why?

    Embedding models have practical input limits.

    Example:

        Large memory
             ↓
        ┌───────────────┐
        │ Chunk 0       │
        ├───────────────┤
        │ Chunk 1       │
        ├───────────────┤
        │ Chunk 2       │
        └───────────────┘

    Overlap preserves context between neighboring chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0:
        raise ValueError("overlap cannot be negative.")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    normalized = normalize_text(
        text,
        preserve_line_breaks=True,
    )

    if not normalized:
        return []

    if len(normalized) <= chunk_size:
        return [
            TextChunk(
                text=normalized,
                index=0,
                start=0,
                end=len(normalized),
            )
        ]

    chunks: list[TextChunk] = []

    start = 0
    index = 0

    while start < len(normalized):
        proposed_end = min(
            start + chunk_size,
            len(normalized),
        )

        end = proposed_end

        # Prefer a natural boundary.
        if proposed_end < len(normalized):
            boundary_candidates = [
                normalized.rfind(
                    "\n",
                    start,
                    proposed_end,
                ),
                normalized.rfind(
                    ". ",
                    start,
                    proposed_end,
                ),
                normalized.rfind(
                    " ",
                    start,
                    proposed_end,
                ),
            ]

            best_boundary = max(boundary_candidates)

            minimum_reasonable_end = start + int(chunk_size * 0.55)

            if best_boundary >= minimum_reasonable_end:
                if normalized[best_boundary : best_boundary + 2] == ". ":
                    end = best_boundary + 1
                else:
                    end = best_boundary

        chunk = normalized[start:end].strip()

        if chunk:
            actual_start = normalized.find(
                chunk,
                start,
                end,
            )

            actual_end = actual_start + len(chunk)

            chunks.append(
                TextChunk(
                    text=chunk,
                    index=index,
                    start=actual_start,
                    end=actual_end,
                )
            )

            index += 1

        if end >= len(normalized):
            break

        next_start = end - overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


# ============================================================================
# TEXT HASHING
# ============================================================================


def text_hash(
    text: str,
    normalize: bool = True,
) -> str:
    """
    Generate a deterministic SHA-256 hash.

    Useful for:
        - duplicate detection
        - cache keys
        - change detection
        - deterministic indexing
    """

    value = normalize_text(text) if normalize else ensure_text(text)

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


# ============================================================================
# TEXT MERGING
# ============================================================================


def join_text_parts(
    parts: Iterable[str | None],
    separator: str = "\n",
) -> str:
    """
    Safely combine multiple text fragments.

    Empty fragments are ignored.
    """

    cleaned_parts: list[str] = []

    for part in parts:
        if part is None:
            continue

        normalized = normalize_text(
            part,
            preserve_line_breaks=True,
        )

        if normalized:
            cleaned_parts.append(normalized)

    return separator.join(cleaned_parts)


# ============================================================================
# SAFE TRUNCATION
# ============================================================================


def truncate_text(
    text: str,
    max_length: int,
    suffix: str = "…",
) -> str:
    """
    Truncate text without exceeding max_length.

    Attempts to break at a natural whitespace boundary.
    """

    if max_length <= 0:
        raise ValueError("max_length must be greater than zero.")

    normalized = normalize_text(
        text,
        preserve_line_breaks=False,
    )

    if len(normalized) <= max_length:
        return normalized

    if not suffix:
        return normalized[:max_length]

    if len(suffix) >= max_length:
        return suffix[:max_length]

    available = max_length - len(suffix)

    candidate = normalized[:available]

    boundary = candidate.rfind(" ")

    if boundary >= int(available * 0.6):
        candidate = candidate[:boundary]

    return candidate.rstrip() + suffix


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "TextStatistics",
    "TextChunk",
    "ensure_text",
    "is_meaningful_text",
    "normalize_unicode",
    "remove_invisible_characters",
    "normalize_whitespace",
    "remove_ocr_artifacts",
    "clean_ocr_text",
    "normalize_text",
    "normalize_for_search",
    "prepare_for_embedding",
    "tokenize_words",
    "unique_words",
    "split_sentences",
    "split_paragraphs",
    "extract_emails",
    "extract_urls",
    "extract_phone_numbers",
    "extract_money_values",
    "count_sentences",
    "count_paragraphs",
    "get_text_statistics",
    "chunk_text",
    "text_hash",
    "join_text_parts",
    "truncate_text",
]
