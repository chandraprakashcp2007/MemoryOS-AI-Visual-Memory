"""
MemoryOS - Unified AI Parser

This module converts OCR output + Gemini Vision analysis into one
normalized memory representation.

Pipeline
--------

Image
  |
  +--> OCR
  |
  +--> Gemini Vision
           |
           v
      AI Parser
           |
           v
   UnifiedMemory
           |
     +-----+------+----------+
     |            |          |
 Classification  Entities   Intent
     |            |          |
     +------------+----------+
                  |
                  v
             Search Text
                  |
                  v
              Embeddings
                  |
                  v
                FAISS


Design goals
------------

1. Deterministic normalization
2. No hallucination when Gemini is unavailable
3. OCR and Vision evidence remain distinguishable
4. Search-ready text is generated consistently
5. Structured fields are strongly typed
6. Downstream services do not need to understand Gemini
7. Easy serialization into SQLite / API responses
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

from backend.services.vision_service import (
    VisionAnalysis,
    VisionService,
    vision_service,
)

# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger("memoryos.ai_parser")


# ============================================================================
# CONSTANTS
# ============================================================================

PARSER_VERSION = "1.0.0"

MAX_TEXT_LENGTH = 50_000

MAX_FIELD_LENGTH = 5_000

MAX_LIST_ITEMS = 100

DEFAULT_CATEGORY = "other"

DEFAULT_INTENT = "unknown"

MIN_IMPORTANCE = 1

MAX_IMPORTANCE = 5


SUPPORTED_CATEGORIES = {
    "shopping",
    "finance",
    "education",
    "travel",
    "food",
    "work",
    "personal",
    "social",
    "communication",
    "technology",
    "entertainment",
    "health",
    "documents",
    "other",
}


SUPPORTED_INTENTS = {
    "buy",
    "sell",
    "save",
    "remember",
    "learn",
    "compare",
    "contact",
    "travel",
    "track",
    "schedule",
    "read",
    "reference",
    "unknown",
}


# ============================================================================
# EXCEPTIONS
# ============================================================================


class AIParserError(Exception):
    """Base AI parser exception."""


class MemoryNormalizationError(AIParserError):
    """Raised when memory data cannot be normalized."""


class MemorySerializationError(AIParserError):
    """Raised when memory data cannot be serialized."""


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass(frozen=True, slots=True)
class MemoryEntity:
    """
    A normalized named entity.

    Example:

        {
            "text": "Adidas",
            "type": "organization"
        }
    """

    text: str
    type: str
    confidence: float = 0.5


@dataclass(frozen=True, slots=True)
class MemorySource:
    """
    Describes where information came from.
    """

    ocr: bool = False
    vision: bool = False
    manual: bool = False


@dataclass(frozen=True, slots=True)
class UnifiedMemory:
    """
    Canonical MemoryOS representation.

    This is the central contract shared by:

        AI parser
        classification
        entity extraction
        intent detection
        embedding service
        SQLite
        FAISS
        search
        API
        frontend
    """

    memory_id: str

    title: str

    summary: str

    category: str

    subcategory: str

    intent: str

    importance: int

    ocr_text: str

    visual_description: str

    search_text: str

    entities: tuple[MemoryEntity, ...]

    people: tuple[str, ...]

    organizations: tuple[str, ...]

    locations: tuple[str, ...]

    products: tuple[str, ...]

    dates: tuple[str, ...]

    prices: tuple[str, ...]

    emails: tuple[str, ...]

    phone_numbers: tuple[str, ...]

    urls: tuple[str, ...]

    keywords: tuple[str, ...]

    actionable_items: tuple[str, ...]

    source: MemorySource

    parser_version: str

    created_at: str

    confidence: float

    fallback: bool

    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================


def normalize_whitespace(
    text: str | None,
) -> str:
    """
    Normalize repeated whitespace without destroying useful line breaks.
    """

    if not text:
        return ""

    value = str(text)

    value = value.replace(
        "\r\n",
        "\n",
    )

    value = value.replace(
        "\r",
        "\n",
    )

    # Remove trailing whitespace from each line.
    lines = [line.strip() for line in value.split("\n")]

    # Collapse excessive blank lines.
    cleaned_lines: list[str] = []

    blank_count = 0

    for line in lines:
        if not line:
            blank_count += 1

            if blank_count <= 1:
                cleaned_lines.append("")

            continue

        blank_count = 0

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def clean_text(
    text: str | None,
    *,
    max_length: int = MAX_TEXT_LENGTH,
) -> str:
    """
    Clean and safely truncate text.
    """

    value = normalize_whitespace(text)

    if len(value) <= max_length:
        return value

    return value[: max_length - 3] + "..."


def clean_field(
    value: Any,
) -> str:
    """
    Convert a value into a safe normalized field.
    """

    if value is None:
        return ""

    return clean_text(
        str(value),
        max_length=MAX_FIELD_LENGTH,
    )


# ============================================================================
# LIST NORMALIZATION
# ============================================================================


def normalize_list(
    values: Iterable[Any] | Any,
    *,
    max_items: int = MAX_LIST_ITEMS,
) -> tuple[str, ...]:
    """
    Normalize arbitrary list-like data into a deterministic tuple.
    """

    if values is None:
        return ()

    if isinstance(
        values,
        str,
    ):
        values = [values]

    if not isinstance(
        values,
        Iterable,
    ):
        values = [values]

    result: list[str] = []

    for value in values:

        cleaned = clean_field(value)

        if not cleaned:
            continue

        # Prevent pathological duplicates.
        if cleaned.lower() in {item.lower() for item in result}:
            continue

        result.append(cleaned)

        if len(result) >= max_items:
            break

    return tuple(result)


# ============================================================================
# CATEGORY / INTENT
# ============================================================================


def normalize_category(
    category: str | None,
) -> str:
    """
    Normalize category to a supported MemoryOS value.
    """

    value = clean_field(category).lower()

    value = value.replace(
        "-",
        "_",
    )

    if value in SUPPORTED_CATEGORIES:
        return value

    aliases = {
        "shopping_products": "shopping",
        "product": "shopping",
        "products": "shopping",
        "money": "finance",
        "banking": "finance",
        "school": "education",
        "college": "education",
        "university": "education",
        "job": "work",
        "career": "work",
        "restaurant": "food",
        "food_order": "food",
        "places": "travel",
        "trip": "travel",
        "medical": "health",
        "document": "documents",
        "tech": "technology",
    }

    return aliases.get(
        value,
        DEFAULT_CATEGORY,
    )


def normalize_intent(
    intent: str | None,
) -> str:
    """
    Normalize intent.
    """

    value = clean_field(intent).lower()

    value = value.replace(
        "-",
        "_",
    )

    if value in SUPPORTED_INTENTS:
        return value

    aliases = {
        "purchase": "buy",
        "shopping": "buy",
        "save_for_later": "save",
        "bookmark": "save",
        "study": "learn",
        "learning": "learn",
        "read_later": "read",
        "reminder": "remember",
        "reference_material": "reference",
        "unknown_intent": "unknown",
    }

    return aliases.get(
        value,
        DEFAULT_INTENT,
    )


# ============================================================================
# IMPORTANCE
# ============================================================================


def normalize_importance(
    value: Any,
) -> int:
    """
    Normalize importance into 1-5.
    """

    try:
        importance = int(float(value))
    except (
        TypeError,
        ValueError,
    ):
        importance = MIN_IMPORTANCE

    return max(
        MIN_IMPORTANCE,
        min(
            importance,
            MAX_IMPORTANCE,
        ),
    )


# ============================================================================
# CONFIDENCE
# ============================================================================


def calculate_confidence(
    *,
    vision_success: bool,
    has_ocr: bool,
    has_title: bool,
    has_summary: bool,
    has_category: bool,
    entity_count: int,
    has_search_text: bool,
) -> float:
    """
    Calculate deterministic confidence.

    This is NOT a probability.

    It is a quality score describing how much usable information
    the unified memory contains.
    """

    score = 0.0

    if vision_success:
        score += 0.35

    if has_ocr:
        score += 0.15

    if has_title:
        score += 0.10

    if has_summary:
        score += 0.10

    if has_category:
        score += 0.10

    if entity_count > 0:
        score += 0.10

    if has_search_text:
        score += 0.10

    return round(
        max(
            0.0,
            min(
                score,
                1.0,
            ),
        ),
        4,
    )


# ============================================================================
# ENTITY CREATION
# ============================================================================


def _entity(
    text: str,
    entity_type: str,
    confidence: float = 0.5,
) -> MemoryEntity:
    """
    Safely create a MemoryEntity.
    """

    return MemoryEntity(
        text=clean_field(text),
        type=clean_field(entity_type).lower() or "unknown",
        confidence=max(
            0.0,
            min(
                float(confidence),
                1.0,
            ),
        ),
    )


def build_entities(
    analysis: VisionAnalysis,
) -> tuple[MemoryEntity, ...]:
    """
    Convert Gemini's separate entity lists into one normalized entity list.
    """

    result: list[MemoryEntity] = []

    groups = (
        (
            analysis.people,
            "person",
        ),
        (
            analysis.organizations,
            "organization",
        ),
        (
            analysis.locations,
            "location",
        ),
        (
            analysis.products,
            "product",
        ),
        (
            analysis.emails,
            "email",
        ),
        (
            analysis.phone_numbers,
            "phone",
        ),
        (
            analysis.urls,
            "url",
        ),
    )

    for values, entity_type in groups:

        for value in values:

            if not value:
                continue

            candidate = _entity(
                value,
                entity_type,
                confidence=0.75,
            )

            if not candidate.text:
                continue

            duplicate = any(
                existing.text.lower() == candidate.text.lower()
                and existing.type == candidate.type
                for existing in result
            )

            if not duplicate:
                result.append(candidate)

    # Also include generic entities returned by Gemini.
    for value in analysis.entities:

        if not value:
            continue

        candidate = _entity(
            value,
            "entity",
            confidence=0.65,
        )

        duplicate = any(
            existing.text.lower() == candidate.text.lower() for existing in result
        )

        if not duplicate:
            result.append(candidate)

    return tuple(result[:MAX_LIST_ITEMS])


# ============================================================================
# KEYWORD EXTRACTION
# ============================================================================


_STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "this",
    "that",
    "from",
    "your",
    "you",
    "are",
    "was",
    "were",
    "have",
    "has",
    "had",
    "into",
    "about",
    "then",
    "than",
    "there",
    "their",
    "they",
    "will",
    "would",
    "could",
    "should",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "at",
    "is",
    "it",
    "be",
    "as",
    "or",
    "by",
    "we",
    "i",
}


def extract_keywords_from_text(
    text: str,
    *,
    limit: int = 25,
) -> tuple[str, ...]:
    """
    Extract lightweight deterministic keywords from OCR/summary text.

    This is deliberately not an ML keyword extractor.
    A later NLP layer can replace this without changing the
    UnifiedMemory contract.
    """

    if not text:
        return ()

    words = re.findall(
        r"[A-Za-z0-9][A-Za-z0-9_+#.-]{2,}",
        text.lower(),
    )

    result: list[str] = []

    for word in words:

        cleaned = word.strip(".,!?;:()[]{}<>\"'`")

        if not cleaned:
            continue

        if cleaned in _STOP_WORDS:
            continue

        if len(cleaned) < 3:
            continue

        if cleaned not in result:
            result.append(cleaned)

        if len(result) >= limit:
            break

    return tuple(result)


# ============================================================================
# SEARCH TEXT
# ============================================================================


def build_search_text(
    *,
    title: str,
    summary: str,
    category: str,
    subcategory: str,
    intent: str,
    ocr_text: str,
    visual_description: str,
    people: Iterable[str],
    organizations: Iterable[str],
    locations: Iterable[str],
    products: Iterable[str],
    dates: Iterable[str],
    prices: Iterable[str],
    keywords: Iterable[str],
    actionable_items: Iterable[str],
) -> str:
    """
    Build the canonical text used later for embeddings.

    Repetition is intentional.

    Important semantic fields are explicitly included so that
    embedding models receive the memory's context.
    """

    sections = [
        title,
        summary,
        f"category: {category}",
        f"subcategory: {subcategory}",
        f"intent: {intent}",
        visual_description,
        ocr_text,
        "people: " + ", ".join(people),
        "organizations: " + ", ".join(organizations),
        "locations: " + ", ".join(locations),
        "products: " + ", ".join(products),
        "dates: " + ", ".join(dates),
        "prices: " + ", ".join(prices),
        "keywords: " + ", ".join(keywords),
        "actions: " + ", ".join(actionable_items),
    ]

    return clean_text("\n".join(section for section in sections if section))


# ============================================================================
# MEMORY ID
# ============================================================================


def generate_memory_id(
    *,
    title: str,
    summary: str,
    ocr_text: str,
    search_text: str,
) -> str:
    """
    Generate deterministic memory ID.

    Same normalized content produces the same ID.

    This helps with duplicate detection.
    """

    canonical = "|".join(
        (
            title.strip().lower(),
            summary.strip().lower(),
            ocr_text.strip().lower(),
            search_text.strip().lower(),
        )
    )

    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    return "mem_" + digest[:32]


# ============================================================================
# SERIALIZATION
# ============================================================================


def memory_to_dict(
    memory: UnifiedMemory,
) -> dict[str, Any]:
    """
    Convert UnifiedMemory into JSON-compatible dictionary.
    """

    try:
        data = asdict(memory)

        data["entities"] = [asdict(entity) for entity in memory.entities]

        return data

    except Exception as exc:
        raise MemorySerializationError("Unable to serialize memory.") from exc


def memory_to_json(
    memory: UnifiedMemory,
    *,
    indent: int | None = None,
) -> str:
    """
    Serialize UnifiedMemory to JSON.
    """

    try:
        return json.dumps(
            memory_to_dict(memory),
            ensure_ascii=False,
            indent=indent,
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise MemorySerializationError("Unable to serialize memory to JSON.") from exc


# ============================================================================
# MAIN PARSER
# ============================================================================


class AIParser:
    """
    Central MemoryOS parser.

    It accepts:
        - OCR text
        - Gemini VisionAnalysis

    and produces:
        - UnifiedMemory
    """

    def __init__(
        self,
        *,
        vision: VisionService | None = None,
    ) -> None:

        self.vision = vision or vision_service

    # ---------------------------------------------------------------------
    # Parse Vision result
    # ---------------------------------------------------------------------

    def parse(
        self,
        *,
        vision_analysis: VisionAnalysis,
        ocr_text: str = "",
    ) -> UnifiedMemory:
        """
        Convert OCR + VisionAnalysis into UnifiedMemory.
        """

        normalized_ocr = clean_text(ocr_text)

        title = clean_field(vision_analysis.title)

        summary = clean_text(
            vision_analysis.summary,
            max_length=MAX_FIELD_LENGTH,
        )

        category = normalize_category(vision_analysis.category)

        subcategory = clean_field(vision_analysis.subcategory)

        intent = normalize_intent(vision_analysis.intent)

        importance = normalize_importance(vision_analysis.importance)

        visual_description = clean_text(
            vision_analysis.visual_description,
            max_length=MAX_FIELD_LENGTH,
        )

        people = normalize_list(vision_analysis.people)

        organizations = normalize_list(vision_analysis.organizations)

        locations = normalize_list(vision_analysis.locations)

        products = normalize_list(vision_analysis.products)

        dates = normalize_list(vision_analysis.dates)

        prices = normalize_list(vision_analysis.prices)

        emails = normalize_list(vision_analysis.emails)

        phone_numbers = normalize_list(vision_analysis.phone_numbers)

        urls = normalize_list(vision_analysis.urls)

        actionable_items = normalize_list(vision_analysis.actionable_items)

        model_keywords = normalize_list(vision_analysis.keywords)

        generated_keywords = extract_keywords_from_text(
            " ".join(
                (
                    title,
                    summary,
                    normalized_ocr,
                    visual_description,
                )
            )
        )

        keywords = normalize_list(
            (
                *model_keywords,
                *generated_keywords,
            ),
            max_items=50,
        )

        entities = build_entities(vision_analysis)

        search_text = build_search_text(
            title=title,
            summary=summary,
            category=category,
            subcategory=subcategory,
            intent=intent,
            ocr_text=normalized_ocr,
            visual_description=(visual_description),
            people=people,
            organizations=organizations,
            locations=locations,
            products=products,
            dates=dates,
            prices=prices,
            keywords=keywords,
            actionable_items=(actionable_items),
        )

        memory_id = generate_memory_id(
            title=title,
            summary=summary,
            ocr_text=normalized_ocr,
            search_text=search_text,
        )

        source = MemorySource(
            ocr=bool(normalized_ocr),
            vision=(vision_analysis.success and not vision_analysis.fallback),
            manual=False,
        )

        confidence = calculate_confidence(
            vision_success=(vision_analysis.success and not vision_analysis.fallback),
            has_ocr=bool(normalized_ocr),
            has_title=bool(title),
            has_summary=bool(summary),
            has_category=(category != DEFAULT_CATEGORY),
            entity_count=len(entities),
            has_search_text=bool(search_text),
        )

        metadata = {
            "vision_model": (vision_analysis.model),
            "vision_fallback": (vision_analysis.fallback),
            "vision_error": (vision_analysis.error),
            "parser_version": (PARSER_VERSION),
        }

        return UnifiedMemory(
            memory_id=memory_id,
            title=title,
            summary=summary,
            category=category,
            subcategory=subcategory,
            intent=intent,
            importance=importance,
            ocr_text=normalized_ocr,
            visual_description=(visual_description),
            search_text=search_text,
            entities=entities,
            people=people,
            organizations=organizations,
            locations=locations,
            products=products,
            dates=dates,
            prices=prices,
            emails=emails,
            phone_numbers=phone_numbers,
            urls=urls,
            keywords=keywords,
            actionable_items=(actionable_items),
            source=source,
            parser_version=PARSER_VERSION,
            created_at=datetime.now(timezone.utc).isoformat(),
            confidence=confidence,
            fallback=vision_analysis.fallback,
            metadata=metadata,
        )

    # ---------------------------------------------------------------------
    # Parse image directly
    # ---------------------------------------------------------------------

    def parse_image(
        self,
        source: Any,
        *,
        ocr_text: str = "",
        allow_fallback: bool = True,
    ) -> UnifiedMemory:
        """
        Analyze an image with Gemini and immediately create UnifiedMemory.
        """

        vision_analysis = self.vision.analyze_with_ocr(
            source,
            ocr_text=ocr_text,
            allow_fallback=allow_fallback,
        )

        return self.parse(
            vision_analysis=vision_analysis,
            ocr_text=ocr_text,
        )


# ============================================================================
# DEFAULT PARSER
# ============================================================================


ai_parser = AIParser()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def parse_analysis(
    *,
    vision_analysis: VisionAnalysis,
    ocr_text: str = "",
) -> UnifiedMemory:
    """
    Convenience wrapper for parsing an existing VisionAnalysis.
    """

    return ai_parser.parse(
        vision_analysis=vision_analysis,
        ocr_text=ocr_text,
    )


def parse_image(
    source: Any,
    *,
    ocr_text: str = "",
    allow_fallback: bool = True,
) -> UnifiedMemory:
    """
    Convenience wrapper for direct image parsing.
    """

    return ai_parser.parse_image(
        source,
        ocr_text=ocr_text,
        allow_fallback=allow_fallback,
    )


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "PARSER_VERSION",
    "MemoryEntity",
    "MemorySource",
    "UnifiedMemory",
    "AIParserError",
    "MemoryNormalizationError",
    "MemorySerializationError",
    "normalize_whitespace",
    "clean_text",
    "clean_field",
    "normalize_list",
    "normalize_category",
    "normalize_intent",
    "normalize_importance",
    "calculate_confidence",
    "build_entities",
    "extract_keywords_from_text",
    "build_search_text",
    "generate_memory_id",
    "memory_to_dict",
    "memory_to_json",
    "AIParser",
    "ai_parser",
    "parse_analysis",
    "parse_image",
]
