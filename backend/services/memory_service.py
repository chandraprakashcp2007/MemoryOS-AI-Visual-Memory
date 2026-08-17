"""
MemoryOS - Memory Service

Central orchestration layer for converting extracted information into a
unified Memory representation.

Responsibilities
----------------
- Normalize incoming analysis data
- Combine OCR + AI output
- Preserve raw text
- Store cleaned/searchable text
- Attach classification
- Attach entities
- Attach intent
- Calculate deterministic metadata
- Generate a stable memory identifier
- Build embedding-ready text
- Produce explainable memory records

Important architecture rule
----------------------------
This service DOES NOT directly depend on:
    - Gemini
    - FAISS
    - SQLite
    - FastAPI

Those systems consume the unified representation produced here.

Pipeline
--------
Image
  ↓
OCR / Vision
  ↓
MemoryService
  ↓
Unified Memory
  ├── raw text
  ├── cleaned text
  ├── title
  ├── category
  ├── intent
  ├── entities
  ├── confidence
  ├── timestamps
  ├── image metadata
  └── embedding text
       ↓
Embedding Service
       ↓
FAISS
       ↓
Search
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

MEMORY_SCHEMA_VERSION = "1.0"

DEFAULT_CATEGORY = "other"

MAX_TITLE_LENGTH = 160
MAX_TEXT_LENGTH = 50_000

SUPPORTED_CATEGORIES = (
    "shopping",
    "travel",
    "food",
    "finance",
    "learning",
    "work",
    "communication",
    "social",
    "entertainment",
    "health",
    "document",
    "technology",
    "event",
    "location",
    "reminder",
    "reference",
    "payment",
    "booking",
    "tracking",
    "other",
)


# ============================================================================
# EXCEPTIONS
# ============================================================================


class MemoryServiceError(Exception):
    """Base exception for memory-service errors."""


class InvalidMemoryInputError(MemoryServiceError):
    """Raised when required memory input is invalid."""


class MemorySerializationError(MemoryServiceError):
    """Raised when a memory cannot be serialized safely."""


# ============================================================================
# HELPERS
# ============================================================================


def utc_now() -> datetime:
    """
    Return a timezone-aware UTC timestamp.
    """

    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """
    Return current UTC timestamp in ISO-8601 format.
    """

    return utc_now().isoformat()


def clean_string(
    value: Any,
) -> str:
    """
    Safely convert arbitrary values into normalized text.
    """

    if value is None:
        return ""

    if isinstance(
        value,
        bytes,
    ):
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


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """
    Clamp a number into a safe range.
    """

    return max(
        minimum,
        min(
            maximum,
            float(value),
        ),
    )


def safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Safely convert a value to float.
    """

    try:
        return float(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


def safe_int(
    value: Any,
    default: int = 0,
) -> int:
    """
    Safely convert a value to integer.
    """

    try:
        return int(value)
    except (
        TypeError,
        ValueError,
    ):
        return default


# ============================================================================
# ENTITY REPRESENTATION
# ============================================================================


@dataclass(slots=True)
class MemoryEntity:
    """
    Serializable entity representation.

    We intentionally do not depend directly on a specific Entity class.
    This keeps MemoryService compatible with:
        - local NER
        - Gemini
        - spaCy
        - future extraction models
    """

    text: str

    entity_type: str

    confidence: float = 0.0

    start: int | None = None

    end: int | None = None

    source: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "entity_type": self.entity_type,
            "confidence": round(
                clamp(self.confidence),
                4,
            ),
            "start": self.start,
            "end": self.end,
            "source": self.source,
        }


# ============================================================================
# IMAGE METADATA
# ============================================================================


@dataclass(slots=True)
class MemoryImageMetadata:
    """
    Image metadata stored alongside a memory.
    """

    filename: str = ""

    mime_type: str = ""

    width: int | None = None

    height: int | None = None

    file_size_bytes: int = 0

    image_hash: str = ""

    thumbnail_path: str | None = None

    original_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ============================================================================
# MEMORY REPRESENTATION
# ============================================================================


@dataclass(slots=True)
class Memory:
    """
    Canonical MemoryOS memory object.

    This is the central representation shared between:

        AI
        database
        embeddings
        FAISS
        search
        API
        frontend
    """

    id: str

    schema_version: str

    raw_text: str

    cleaned_text: str

    title: str

    category: str

    intent: str

    entities: list[MemoryEntity]

    confidence: float

    source: str

    created_at: str

    updated_at: str

    embedding_text: str

    image: MemoryImageMetadata | None = None

    ai_analysis: dict[str, Any] = field(default_factory=dict)

    metadata: dict[str, Any] = field(default_factory=dict)

    tags: list[str] = field(default_factory=list)

    processing_errors: list[str] = field(default_factory=list)

    is_deleted: bool = False

    version: int = 1

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """
        Convert memory into JSON-compatible dictionary.
        """

        return {
            "id": self.id,
            "schema_version": self.schema_version,
            "raw_text": self.raw_text,
            "cleaned_text": self.cleaned_text,
            "title": self.title,
            "category": self.category,
            "intent": self.intent,
            "entities": [entity.to_dict() for entity in self.entities],
            "confidence": round(
                clamp(self.confidence),
                4,
            ),
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "embedding_text": self.embedding_text,
            "image": (self.image.to_dict() if self.image else None),
            "ai_analysis": self.ai_analysis,
            "metadata": self.metadata,
            "tags": self.tags,
            "processing_errors": self.processing_errors,
            "is_deleted": self.is_deleted,
            "version": self.version,
        }

    def to_json(
        self,
    ) -> str:
        """
        Serialize the memory to JSON.
        """

        try:
            return json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise MemorySerializationError("Unable to serialize memory.") from exc


# ============================================================================
# MEMORY SERVICE
# ============================================================================


class MemoryService:
    """
    Builds and manages canonical Memory objects.
    """

    # ------------------------------------------------------------------------
    # TEXT CLEANING
    # ------------------------------------------------------------------------

    @staticmethod
    def clean_text(
        text: Any,
    ) -> str:
        """
        Clean OCR / AI text while preserving useful semantic information.
        """

        value = clean_string(text)

        if not value:
            return ""

        # Normalize common OCR artifacts.
        value = value.replace(
            "\r\n",
            "\n",
        )

        value = value.replace(
            "\r",
            "\n",
        )

        # Collapse excessive spaces.
        value = re.sub(
            r"[ \t]+",
            " ",
            value,
        )

        # Remove excessive blank lines.
        value = re.sub(
            r"\n{3,}",
            "\n\n",
            value,
        )

        # Strip each line.
        lines = [line.strip() for line in value.splitlines()]

        value = "\n".join(lines).strip()

        return value[:MAX_TEXT_LENGTH]

    # ------------------------------------------------------------------------
    # TITLE GENERATION
    # ------------------------------------------------------------------------

    @staticmethod
    def generate_title(
        text: str,
        category: str = DEFAULT_CATEGORY,
        ai_title: Any = None,
    ) -> str:
        """
        Generate a human-readable memory title.

        AI-generated title is preferred when available.
        Otherwise deterministic extraction is used.
        """

        title = clean_string(ai_title)

        if title:
            return title[:MAX_TITLE_LENGTH]

        cleaned = MemoryService.clean_text(text)

        if not cleaned:
            return "Untitled Memory"

        # Use the first meaningful line.
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]

        candidate = lines[0] if lines else cleaned

        # Avoid extremely long titles.
        candidate = re.sub(
            r"\s+",
            " ",
            candidate,
        ).strip()

        if len(candidate) > MAX_TITLE_LENGTH:
            candidate = candidate[: MAX_TITLE_LENGTH - 1].rstrip() + "…"

        return candidate

    # ------------------------------------------------------------------------
    # CATEGORY NORMALIZATION
    # ------------------------------------------------------------------------

    @staticmethod
    def normalize_category(
        category: Any,
    ) -> str:
        """
        Normalize category labels from different AI/model providers.
        """

        value = clean_string(category).lower()

        value = value.replace(
            "-",
            "_",
        )

        aliases = {
            "shopping": "shopping",
            "shop": "shopping",
            "purchase": "shopping",
            "product": "shopping",
            "buy": "shopping",
            "buying": "shopping",
            "travel": "travel",
            "trip": "travel",
            "tourism": "travel",
            "food": "food",
            "restaurant": "food",
            "meal": "food",
            "payment": "payment",
            "payments": "payment",
            "finance": "finance",
            "financial": "finance",
            "banking": "finance",
            "study": "learning",
            "education": "learning",
            "learning": "learning",
            "tech": "technology",
            "technical": "technology",
            "technology": "technology",
            "document": "document",
            "documents": "document",
            "pdf": "document",
            "communication": "communication",
            "message": "communication",
            "email": "communication",
            "social": "social",
            "entertainment": "entertainment",
            "media": "entertainment",
            "work": "work",
            "business": "work",
            "health": "health",
            "fitness": "health",
            "event": "event",
            "events": "event",
            "location": "location",
            "place": "location",
            "reminder": "reminder",
            "reference": "reference",
            "booking": "booking",
            "reservation": "booking",
            "tracking": "tracking",
            "delivery": "tracking",
        }

        normalized = aliases.get(
            value,
            value,
        )

        if normalized not in SUPPORTED_CATEGORIES:
            return DEFAULT_CATEGORY

        return normalized

    # ------------------------------------------------------------------------
    # INTENT NORMALIZATION
    # ------------------------------------------------------------------------

    @staticmethod
    def normalize_intent(
        intent: Any,
    ) -> str:
        """
        Normalize intent labels.

        Kept here as a lightweight normalization layer so MemoryService
        remains robust even if an upstream AI service changes its label.
        """

        value = clean_string(intent).upper()

        aliases = {
            "PURCHASE": "BUY",
            "BUYING": "BUY",
            "SHOP": "SHOPPING",
            "PRODUCT_SEARCH": "SHOPPING",
            "PRODUCT": "SHOPPING",
            "TRIP": "TRAVEL",
            "FLIGHTS": "TRAVEL",
            "RESERVE": "BOOKING",
            "RESERVATION": "BOOKING",
            "FOOD_ORDER": "FOOD",
            "TRANSACTION": "PAYMENT",
            "DELIVERY": "TRACKING",
            "STUDY": "LEARNING",
            "EDUCATION": "LEARNING",
            "TECH": "TECHNOLOGY",
            "TECHNICAL": "TECHNOLOGY",
            "CONTACT": "COMMUNICATION",
            "MESSAGE": "COMMUNICATION",
            "DOC": "DOCUMENT",
            "INFO": "INFORMATION",
            "REMIND": "REMINDER",
            "PLAN": "PLANNING",
        }

        return aliases.get(
            value,
            value or "OTHER",
        )

    # ------------------------------------------------------------------------
    # ENTITY NORMALIZATION
    # ------------------------------------------------------------------------

    @staticmethod
    def normalize_entity(
        entity: Any,
    ) -> MemoryEntity | None:
        """
        Convert arbitrary entity representations into MemoryEntity.
        """

        if isinstance(
            entity,
            MemoryEntity,
        ):
            return entity

        # ------------------------------------------------------------
        # Mapping/dictionary
        # ------------------------------------------------------------

        if isinstance(
            entity,
            Mapping,
        ):

            text = entity.get("text") or entity.get("value") or entity.get("entity")

            entity_type = (
                entity.get("entity_type") or entity.get("type") or entity.get("label")
            )

            if not text or not entity_type:
                return None

            return MemoryEntity(
                text=clean_string(text),
                entity_type=clean_string(entity_type).upper(),
                confidence=clamp(
                    safe_float(
                        entity.get(
                            "confidence",
                            0.0,
                        )
                    )
                ),
                start=(
                    safe_int(entity.get("start"))
                    if entity.get("start") is not None
                    else None
                ),
                end=(
                    safe_int(entity.get("end"))
                    if entity.get("end") is not None
                    else None
                ),
                source=clean_string(
                    entity.get(
                        "source",
                        "unknown",
                    )
                )
                or "unknown",
            )

        # ------------------------------------------------------------
        # Object with attributes
        # ------------------------------------------------------------

        text = getattr(
            entity,
            "text",
            None,
        )

        entity_type = (
            getattr(
                entity,
                "entity_type",
                None,
            )
            or getattr(
                entity,
                "type",
                None,
            )
            or getattr(
                entity,
                "label",
                None,
            )
        )

        if not text or not entity_type:
            return None

        return MemoryEntity(
            text=clean_string(text),
            entity_type=clean_string(entity_type).upper(),
            confidence=clamp(
                safe_float(
                    getattr(
                        entity,
                        "confidence",
                        0.0,
                    )
                )
            ),
            start=getattr(
                entity,
                "start",
                None,
            ),
            end=getattr(
                entity,
                "end",
                None,
            ),
            source=clean_string(
                getattr(
                    entity,
                    "source",
                    "unknown",
                )
            )
            or "unknown",
        )

    # ------------------------------------------------------------------------
    # ENTITY DEDUPLICATION
    # ------------------------------------------------------------------------

    @staticmethod
    def deduplicate_entities(
        entities: Iterable[Any],
    ) -> list[MemoryEntity]:
        """
        Remove duplicate entities while preserving the strongest occurrence.
        """

        unique: dict[
            tuple[str, str],
            MemoryEntity,
        ] = {}

        for raw_entity in entities:

            entity = MemoryService.normalize_entity(raw_entity)

            if entity is None:
                continue

            if not entity.text:
                continue

            key = (
                entity.entity_type,
                entity.text.lower(),
            )

            existing = unique.get(key)

            if existing is None or entity.confidence > existing.confidence:
                unique[key] = entity

        return list(unique.values())

    # ------------------------------------------------------------------------
    # TAG GENERATION
    # ------------------------------------------------------------------------

    @staticmethod
    def generate_tags(
        category: str,
        intent: str,
        entities: Sequence[MemoryEntity],
        provided_tags: Iterable[Any] | None = None,
    ) -> list[str]:
        """
        Generate searchable semantic tags.
        """

        tags: list[str] = []

        def add_tag(
            value: Any,
        ) -> None:

            value = clean_string(value).lower()

            if not value:
                return

            value = re.sub(
                r"\s+",
                "_",
                value,
            )

            if value not in tags:
                tags.append(value)

        add_tag(category)

        add_tag(intent)

        for entity in entities:

            add_tag(entity.entity_type)

            # Only use short entity values as tags.
            if len(entity.text) <= 50:
                add_tag(entity.text)

        if provided_tags:

            for tag in provided_tags:
                add_tag(tag)

        # Avoid uncontrolled tag growth.
        return tags[:50]

    # ------------------------------------------------------------------------
    # CONFIDENCE
    # ------------------------------------------------------------------------

    @staticmethod
    def calculate_confidence(
        *,
        classification_confidence: Any = 0.0,
        intent_confidence: Any = 0.0,
        entity_confidences: Iterable[Any] = (),
        vision_confidence: Any = None,
        ocr_quality: Any = None,
    ) -> float:
        """
        Calculate an overall memory confidence score.

        Weighted model:

            classification  → 20%
            intent          → 20%
            entities        → 20%
            vision          → 25%
            OCR quality     → 15%

        Missing signals are excluded and the remaining weights are
        automatically normalized.
        """

        signals: list[tuple[float, float]] = []

        classification = safe_float(
            classification_confidence,
            -1,
        )

        if classification >= 0:
            signals.append(
                (
                    clamp(classification),
                    0.20,
                )
            )

        intent = safe_float(
            intent_confidence,
            -1,
        )

        if intent >= 0:
            signals.append(
                (
                    clamp(intent),
                    0.20,
                )
            )

        entity_values = [clamp(safe_float(value)) for value in entity_confidences]

        if entity_values:
            signals.append(
                (
                    sum(entity_values) / len(entity_values),
                    0.20,
                )
            )

        if vision_confidence is not None:
            signals.append(
                (
                    clamp(safe_float(vision_confidence)),
                    0.25,
                )
            )

        if ocr_quality is not None:
            signals.append(
                (
                    clamp(safe_float(ocr_quality)),
                    0.15,
                )
            )

        if not signals:
            return 0.0

        weighted_sum = sum(value * weight for value, weight in signals)

        total_weight = sum(weight for _, weight in signals)

        if total_weight <= 0:
            return 0.0

        return clamp(weighted_sum / total_weight)

    # ------------------------------------------------------------------------
    # MEMORY ID
    # ------------------------------------------------------------------------

    @staticmethod
    def generate_memory_id(
        *,
        image_hash: str = "",
        cleaned_text: str = "",
        supplied_id: Any = None,
    ) -> str:
        """
        Generate a stable memory identifier.

        If an ID already exists, preserve it.

        Otherwise:
            image_hash + normalized text
                ↓
              SHA-256
                ↓
          mem_<first 32 chars>
        """

        if supplied_id:

            supplied = clean_string(supplied_id)

            if supplied:
                return supplied

        normalized_text = re.sub(
            r"\s+",
            " ",
            cleaned_text.lower(),
        ).strip()

        seed = image_hash.strip() + "|" + normalized_text

        if not seed.strip("|"):
            return "mem_" + uuid.uuid4().hex

        digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()

        return "mem_" + digest[:32]

    # ------------------------------------------------------------------------
    # EMBEDDING TEXT
    # ------------------------------------------------------------------------

    @staticmethod
    def build_embedding_text(
        *,
        title: str,
        cleaned_text: str,
        category: str,
        intent: str,
        entities: Sequence[MemoryEntity],
        tags: Sequence[str],
    ) -> str:
        """
        Build the canonical text that will later be embedded.

        Important:
        The embedding should represent semantic memory, not just raw OCR.
        """

        parts: list[str] = []

        if title:
            parts.append(f"Title: {title}")

        if category:
            parts.append(f"Category: {category}")

        if intent:
            parts.append(f"Intent: {intent}")

        if entities:

            entity_text = ", ".join(
                f"{entity.entity_type}: " f"{entity.text}" for entity in entities
            )

            parts.append(f"Entities: {entity_text}")

        if tags:

            parts.append("Tags: " + ", ".join(tags))

        if cleaned_text:
            parts.append(f"Content: {cleaned_text}")

        return "\n".join(parts)[:MAX_TEXT_LENGTH]

    # ------------------------------------------------------------------------
    # PROCESSING ERRORS
    # ------------------------------------------------------------------------

    @staticmethod
    def collect_errors(
        errors: Any,
    ) -> list[str]:
        """
        Normalize processing errors.
        """

        if errors is None:
            return []

        if isinstance(
            errors,
            str,
        ):
            errors = [errors]

        if not isinstance(
            errors,
            Iterable,
        ):
            return [clean_string(errors)]

        result: list[str] = []

        for error in errors:

            value = clean_string(error)

            if value and value not in result:
                result.append(value)

        return result[:20]

    # ------------------------------------------------------------------------
    # IMAGE METADATA
    # ------------------------------------------------------------------------

    @staticmethod
    def build_image_metadata(
        image: Any,
    ) -> MemoryImageMetadata | None:
        """
        Convert arbitrary image metadata into MemoryImageMetadata.
        """

        if image is None:
            return None

        if isinstance(
            image,
            MemoryImageMetadata,
        ):
            return image

        if isinstance(
            image,
            Mapping,
        ):
            data = image

            return MemoryImageMetadata(
                filename=clean_string(
                    data.get(
                        "filename",
                        "",
                    )
                ),
                mime_type=clean_string(
                    data.get(
                        "mime_type",
                        "",
                    )
                ),
                width=(
                    safe_int(data.get("width"))
                    if data.get("width") is not None
                    else None
                ),
                height=(
                    safe_int(data.get("height"))
                    if data.get("height") is not None
                    else None
                ),
                file_size_bytes=safe_int(
                    data.get(
                        "file_size_bytes",
                        data.get(
                            "size",
                            0,
                        ),
                    )
                ),
                image_hash=clean_string(
                    data.get(
                        "image_hash",
                        data.get(
                            "hash",
                            "",
                        ),
                    )
                ),
                thumbnail_path=(clean_string(data.get("thumbnail_path")) or None),
                original_path=(clean_string(data.get("original_path")) or None),
            )

        return MemoryImageMetadata(
            filename=clean_string(
                getattr(
                    image,
                    "filename",
                    "",
                )
            ),
            mime_type=clean_string(
                getattr(
                    image,
                    "mime_type",
                    "",
                )
            ),
            width=getattr(
                image,
                "width",
                None,
            ),
            height=getattr(
                image,
                "height",
                None,
            ),
            file_size_bytes=safe_int(
                getattr(
                    image,
                    "file_size_bytes",
                    0,
                )
            ),
            image_hash=clean_string(
                getattr(
                    image,
                    "image_hash",
                    "",
                )
            ),
            thumbnail_path=(
                clean_string(
                    getattr(
                        image,
                        "thumbnail_path",
                        "",
                    )
                )
                or None
            ),
            original_path=(
                clean_string(
                    getattr(
                        image,
                        "original_path",
                        "",
                    )
                )
                or None
            ),
        )

    # =========================================================================
    # BUILD MEMORY
    # =========================================================================

    def build_memory(
        self,
        *,
        raw_text: Any = "",
        cleaned_text: Any = None,
        title: Any = None,
        category: Any = None,
        intent: Any = None,
        entities: Iterable[Any] | None = None,
        confidence: Any = None,
        classification_confidence: Any = 0.0,
        intent_confidence: Any = 0.0,
        vision_confidence: Any = None,
        ocr_quality: Any = None,
        source: Any = "upload",
        image: Any = None,
        image_hash: Any = "",
        ai_analysis: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        tags: Iterable[Any] | None = None,
        processing_errors: Any = None,
        memory_id: Any = None,
        created_at: Any = None,
    ) -> Memory:
        """
        Build a complete canonical Memory object.
        """

        # ---------------------------------------------------------------------
        # Text
        # ---------------------------------------------------------------------

        raw = clean_string(raw_text)

        if cleaned_text is None:
            cleaned = self.clean_text(raw)
        else:
            cleaned = self.clean_text(cleaned_text)

        if not cleaned and raw:
            cleaned = raw[:MAX_TEXT_LENGTH]

        # ---------------------------------------------------------------------
        # Entities
        # ---------------------------------------------------------------------

        normalized_entities = self.deduplicate_entities(entities or [])

        # ---------------------------------------------------------------------
        # Category / Intent
        # ---------------------------------------------------------------------

        normalized_category = self.normalize_category(category)

        normalized_intent = self.normalize_intent(intent)

        # ---------------------------------------------------------------------
        # Title
        # ---------------------------------------------------------------------

        ai_title = None

        if ai_analysis:

            ai_title = ai_analysis.get("title") or ai_analysis.get("summary")

        final_title = self.generate_title(
            cleaned,
            normalized_category,
            title or ai_title,
        )

        # ---------------------------------------------------------------------
        # Image metadata
        # ---------------------------------------------------------------------

        image_metadata = self.build_image_metadata(image)

        final_image_hash = clean_string(image_hash)

        if not final_image_hash and image_metadata:
            final_image_hash = image_metadata.image_hash

        if image_metadata:
            image_metadata.image_hash = final_image_hash

        # ---------------------------------------------------------------------
        # Tags
        # ---------------------------------------------------------------------

        final_tags = self.generate_tags(
            category=normalized_category,
            intent=normalized_intent,
            entities=normalized_entities,
            provided_tags=tags,
        )

        # ---------------------------------------------------------------------
        # Confidence
        # ---------------------------------------------------------------------

        entity_confidences = [entity.confidence for entity in normalized_entities]

        calculated_confidence = self.calculate_confidence(
            classification_confidence=(classification_confidence),
            intent_confidence=(intent_confidence),
            entity_confidences=(entity_confidences),
            vision_confidence=(vision_confidence),
            ocr_quality=(ocr_quality),
        )

        if confidence is not None:
            final_confidence = clamp(safe_float(confidence))
        else:
            final_confidence = calculated_confidence

        # ---------------------------------------------------------------------
        # Embedding text
        # ---------------------------------------------------------------------

        embedding_text = self.build_embedding_text(
            title=final_title,
            cleaned_text=cleaned,
            category=normalized_category,
            intent=normalized_intent,
            entities=normalized_entities,
            tags=final_tags,
        )

        # ---------------------------------------------------------------------
        # ID
        # ---------------------------------------------------------------------

        final_id = self.generate_memory_id(
            image_hash=final_image_hash,
            cleaned_text=cleaned,
            supplied_id=memory_id,
        )

        # ---------------------------------------------------------------------
        # Dates
        # ---------------------------------------------------------------------

        current_timestamp = utc_now_iso()

        final_created_at = clean_string(created_at) if created_at else current_timestamp

        # ---------------------------------------------------------------------
        # Source
        # ---------------------------------------------------------------------

        final_source = clean_string(source) or "unknown"

        # ---------------------------------------------------------------------
        # Metadata
        # ---------------------------------------------------------------------

        final_metadata = dict(metadata or {})

        final_metadata.setdefault(
            "schema_version",
            MEMORY_SCHEMA_VERSION,
        )

        final_metadata.setdefault(
            "embedding_ready",
            bool(embedding_text),
        )

        final_metadata.setdefault(
            "entity_count",
            len(normalized_entities),
        )

        final_metadata.setdefault(
            "has_ocr_text",
            bool(raw),
        )

        final_metadata.setdefault(
            "processing_timestamp",
            current_timestamp,
        )

        # ---------------------------------------------------------------------
        # Errors
        # ---------------------------------------------------------------------

        final_errors = self.collect_errors(processing_errors)

        # ---------------------------------------------------------------------
        # Build
        # ---------------------------------------------------------------------

        memory = Memory(
            id=final_id,
            schema_version=(MEMORY_SCHEMA_VERSION),
            raw_text=raw[:MAX_TEXT_LENGTH],
            cleaned_text=cleaned,
            title=final_title,
            category=normalized_category,
            intent=normalized_intent,
            entities=normalized_entities,
            confidence=final_confidence,
            source=final_source,
            created_at=final_created_at,
            updated_at=current_timestamp,
            embedding_text=embedding_text,
            image=image_metadata,
            ai_analysis=dict(ai_analysis or {}),
            metadata=final_metadata,
            tags=final_tags,
            processing_errors=final_errors,
        )

        logger.debug(
            "Memory created: id=%s category=%s intent=%s confidence=%.3f",
            memory.id,
            memory.category,
            memory.intent,
            memory.confidence,
        )

        return memory

    # =========================================================================
    # UPDATE MEMORY
    # =========================================================================

    def update_memory(
        self,
        memory: Memory,
        **updates: Any,
    ) -> Memory:
        """
        Safely update selected memory fields.

        This creates a new version logically while preserving the same ID.
        """

        if not isinstance(
            memory,
            Memory,
        ):
            raise InvalidMemoryInputError("Expected a Memory object.")

        if "raw_text" in updates:
            memory.raw_text = self.clean_text(updates["raw_text"])

        if "cleaned_text" in updates:
            memory.cleaned_text = self.clean_text(updates["cleaned_text"])

        if "title" in updates:
            memory.title = self.generate_title(
                memory.cleaned_text,
                memory.category,
                updates["title"],
            )

        if "category" in updates:
            memory.category = self.normalize_category(updates["category"])

        if "intent" in updates:
            memory.intent = self.normalize_intent(updates["intent"])

        if "entities" in updates:
            memory.entities = self.deduplicate_entities(updates["entities"])

        if "tags" in updates:
            memory.tags = self.generate_tags(
                category=memory.category,
                intent=memory.intent,
                entities=memory.entities,
                provided_tags=updates["tags"],
            )

        if "confidence" in updates:
            memory.confidence = clamp(safe_float(updates["confidence"]))

        if "processing_errors" in updates:
            memory.processing_errors = self.collect_errors(updates["processing_errors"])

        memory.embedding_text = self.build_embedding_text(
            title=memory.title,
            cleaned_text=memory.cleaned_text,
            category=memory.category,
            intent=memory.intent,
            entities=memory.entities,
            tags=memory.tags,
        )

        memory.updated_at = utc_now_iso()

        memory.version += 1

        return memory

    # =========================================================================
    # SOFT DELETE
    # =========================================================================

    @staticmethod
    def soft_delete(
        memory: Memory,
    ) -> Memory:
        """
        Mark a memory as deleted without destroying its data.
        """

        memory.is_deleted = True

        memory.updated_at = utc_now_iso()

        memory.version += 1

        return memory


# ============================================================================
# DEFAULT SERVICE
# ============================================================================

memory_service = MemoryService()


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def create_memory(
    **kwargs: Any,
) -> Memory:
    """
    Convenience wrapper around MemoryService.build_memory().
    """

    return memory_service.build_memory(**kwargs)


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "MEMORY_SCHEMA_VERSION",
    "SUPPORTED_CATEGORIES",
    "MemoryServiceError",
    "InvalidMemoryInputError",
    "MemorySerializationError",
    "MemoryEntity",
    "MemoryImageMetadata",
    "Memory",
    "MemoryService",
    "memory_service",
    "create_memory",
]
