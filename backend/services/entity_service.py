"""
MemoryOS - Entity Extraction & Normalization Service

Responsibilities
----------------
Extract, normalize, deduplicate, score and classify entities from:

    - OCR text
    - Gemini Vision output
    - AI-generated descriptions
    - user-provided keywords
    - structured metadata

Supported entity types
----------------------
    PERSON
    ORGANIZATION
    COMPANY
    BRAND
    PRODUCT
    LOCATION
    ADDRESS
    EMAIL
    PHONE
    URL
    MONEY
    DATE
    TIME
    DATETIME
    ORDER_ID
    INVOICE_ID
    USERNAME
    TECHNOLOGY
    DOCUMENT
    QUANTITY
    UNKNOWN

Design goals
------------
- deterministic
- explainable
- lightweight
- offline-capable
- Gemini-independent
- safe for OCR noise
- easy to extend with NER models later
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterable

logger = logging.getLogger(__name__)


# ============================================================================
# ENTITY TYPES
# ============================================================================

ENTITY_TYPES = (
    "PERSON",
    "ORGANIZATION",
    "COMPANY",
    "BRAND",
    "PRODUCT",
    "LOCATION",
    "ADDRESS",
    "EMAIL",
    "PHONE",
    "URL",
    "MONEY",
    "DATE",
    "TIME",
    "DATETIME",
    "ORDER_ID",
    "INVOICE_ID",
    "USERNAME",
    "TECHNOLOGY",
    "DOCUMENT",
    "QUANTITY",
    "UNKNOWN",
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass(slots=True)
class Entity:
    """
    Canonical entity representation.
    """

    text: str
    entity_type: str

    normalized: str

    confidence: float = 0.5

    source: str = "rule"

    start: int | None = None
    end: int | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert entity into JSON-compatible structure.
        """

        return {
            "text": self.text,
            "type": self.entity_type,
            "normalized": self.normalized,
            "confidence": round(
                self.confidence,
                4,
            ),
            "source": self.source,
            "start": self.start,
            "end": self.end,
            "metadata": self.metadata,
        }


# ============================================================================
# NORMALIZATION SERVICE
# ============================================================================


class EntityService:
    """
    Extract and normalize entities from text and AI output.
    """

    # ------------------------------------------------------------------------
    # REGEX PATTERNS
    # ------------------------------------------------------------------------

    EMAIL_PATTERN = re.compile(
        r"\b[A-Z0-9._%+-]+@" r"[A-Z0-9.-]+\.[A-Z]{2,}\b",
        re.IGNORECASE,
    )

    URL_PATTERN = re.compile(
        r"\b(?:https?://|www\.)" r"[^\s<>()]+",
        re.IGNORECASE,
    )

    PHONE_PATTERN = re.compile(
        r"(?<!\d)" r"(?:\+91[\s.-]?)?" r"(?:\d[\s.-]?){10}" r"(?!\d)"
    )

    MONEY_PATTERN = re.compile(
        r"(?<!\w)"
        r"(?:₹|rs\.?|inr|\$|usd|€|eur|£|gbp)"
        r"\s*"
        r"\d[\d,]*(?:\.\d{1,2})?"
        r"(?!\w)",
        re.IGNORECASE,
    )

    DATE_PATTERN = re.compile(
        r"\b"
        r"(?:"
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r"|"
        r"\d{4}[/-]\d{1,2}[/-]\d{1,2}"
        r"|"
        r"\d{1,2}\s+"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[a-z]*"
        r"(?:\s+\d{2,4})?"
        r"|"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
        r"[a-z]*\s+\d{1,2}"
        r"(?:,\s*|\s+)\d{2,4}"
        r")"
        r"\b",
        re.IGNORECASE,
    )

    TIME_PATTERN = re.compile(
        r"\b"
        r"\d{1,2}"
        r":"
        r"\d{2}"
        r"(?:"
        r":\d{2}"
        r")?"
        r"\s*"
        r"(?:AM|PM|am|pm)?"
        r"\b"
    )

    ORDER_ID_PATTERN = re.compile(
        r"\b" r"(?:order|ord|order-id)" r"[\s:#-]*" r"[A-Z0-9][A-Z0-9_-]{4,}" r"\b",
        re.IGNORECASE,
    )

    INVOICE_PATTERN = re.compile(
        r"\b" r"(?:invoice|inv)" r"[\s:#-]*" r"[A-Z0-9][A-Z0-9_-]{2,}" r"\b",
        re.IGNORECASE,
    )

    USERNAME_PATTERN = re.compile(r"(?<![\w@])" r"@[A-Za-z0-9_.]{2,30}" r"\b")

    # ------------------------------------------------------------------------
    # KNOWN TECHNOLOGIES
    # ------------------------------------------------------------------------

    TECHNOLOGIES = {
        "python",
        "java",
        "javascript",
        "typescript",
        "kotlin",
        "swift",
        "dart",
        "flutter",
        "android",
        "ios",
        "react",
        "react native",
        "node.js",
        "nodejs",
        "fastapi",
        "django",
        "flask",
        "tensorflow",
        "pytorch",
        "numpy",
        "pandas",
        "scikit-learn",
        "sklearn",
        "faiss",
        "sql",
        "sqlite",
        "postgresql",
        "mysql",
        "mongodb",
        "firebase",
        "docker",
        "kubernetes",
        "git",
        "github",
        "linux",
        "aws",
        "azure",
        "gcp",
        "gemini",
        "openai",
    }

    # ------------------------------------------------------------------------
    # KNOWN BRANDS / COMPANIES
    # ------------------------------------------------------------------------

    KNOWN_BRANDS = {
        "adidas",
        "nike",
        "puma",
        "apple",
        "samsung",
        "oneplus",
        "xiaomi",
        "realme",
        "boat",
        "sony",
        "hp",
        "dell",
        "lenovo",
        "asus",
        "acer",
        "amazon",
        "flipkart",
        "myntra",
        "meesho",
        "zomato",
        "swiggy",
        "netflix",
        "spotify",
        "youtube",
        "instagram",
        "facebook",
        "whatsapp",
        "telegram",
        "google",
        "microsoft",
        "github",
        "openai",
    }

    KNOWN_COMPANIES = {
        "google",
        "microsoft",
        "amazon",
        "apple",
        "meta",
        "openai",
        "infosys",
        "tcs",
        "wipro",
        "accenture",
        "zoho",
        "freshworks",
        "github",
    }

    # ------------------------------------------------------------------------
    # KNOWN LOCATIONS
    # ------------------------------------------------------------------------

    KNOWN_LOCATIONS = {
        "india",
        "chennai",
        "salem",
        "coimbatore",
        "madurai",
        "bangalore",
        "bengaluru",
        "hyderabad",
        "mumbai",
        "delhi",
        "new delhi",
        "kolkata",
        "pune",
        "kerala",
        "tamil nadu",
        "karnataka",
        "andhra pradesh",
        "telangana",
        "new york",
        "london",
        "paris",
        "tokyo",
        "dubai",
        "singapore",
    }

    # =========================================================================
    # BASIC NORMALIZATION
    # =========================================================================

    @staticmethod
    def clean_text(value: Any) -> str:
        """
        Safely convert arbitrary input to clean text.
        """

        if value is None:
            return ""

        if isinstance(value, bytes):
            value = value.decode(
                "utf-8",
                errors="replace",
            )

        if not isinstance(value, str):
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

    @staticmethod
    def normalize_entity_text(
        value: Any,
    ) -> str:
        """
        Normalize an entity for deduplication/search.
        """

        value = EntityService.clean_text(value)

        value = value.lower()

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        value = value.strip(" \t\n\r.,;:!?()[]{}<>\"'")

        return value

    # =========================================================================
    # ENTITY CREATION
    # =========================================================================

    @classmethod
    def create_entity(
        cls,
        *,
        text: str,
        entity_type: str,
        confidence: float,
        source: str,
        start: int | None = None,
        end: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Entity:
        """
        Create a validated canonical Entity.
        """

        clean = cls.clean_text(text)

        normalized_type = cls.normalize_entity_type(entity_type)

        confidence = max(
            0.0,
            min(
                1.0,
                float(confidence),
            ),
        )

        return Entity(
            text=clean,
            entity_type=normalized_type,
            normalized=cls.normalize_entity_text(clean),
            confidence=confidence,
            source=source,
            start=start,
            end=end,
            metadata=metadata or {},
        )

    # =========================================================================
    # ENTITY TYPE NORMALIZATION
    # =========================================================================

    @staticmethod
    def normalize_entity_type(
        entity_type: Any,
    ) -> str:
        """
        Normalize common NER labels into MemoryOS labels.
        """

        if not entity_type:
            return "UNKNOWN"

        value = str(entity_type).strip().upper()

        aliases = {
            "PER": "PERSON",
            "PERSON_NAME": "PERSON",
            "ORG": "ORGANIZATION",
            "ORGANISATION": "ORGANIZATION",
            "COMP": "COMPANY",
            "LOC": "LOCATION",
            "GPE": "LOCATION",
            "PLACE": "LOCATION",
            "PRODUCT_NAME": "PRODUCT",
            "MONEY_AMOUNT": "MONEY",
            "CURRENCY": "MONEY",
            "PHONE_NUMBER": "PHONE",
            "TELEPHONE": "PHONE",
            "MAIL": "EMAIL",
            "LINK": "URL",
            "WEB": "URL",
            "DATE_TIME": "DATETIME",
            "ORDER": "ORDER_ID",
            "INVOICE": "INVOICE_ID",
            "USER": "USERNAME",
            "TECH": "TECHNOLOGY",
            "FILE": "DOCUMENT",
        }

        value = aliases.get(
            value,
            value,
        )

        if value not in ENTITY_TYPES:
            return "UNKNOWN"

        return value

    # =========================================================================
    # REGEX EXTRACTION
    # =========================================================================

    def extract_pattern_entities(
        self,
        text: str,
    ) -> list[Entity]:
        """
        Extract highly reliable structured entities.
        """

        entities: list[Entity] = []

        patterns = (
            (
                self.EMAIL_PATTERN,
                "EMAIL",
                0.99,
            ),
            (
                self.URL_PATTERN,
                "URL",
                0.98,
            ),
            (
                self.MONEY_PATTERN,
                "MONEY",
                0.96,
            ),
            (
                self.PHONE_PATTERN,
                "PHONE",
                0.94,
            ),
            (
                self.ORDER_ID_PATTERN,
                "ORDER_ID",
                0.94,
            ),
            (
                self.INVOICE_PATTERN,
                "INVOICE_ID",
                0.94,
            ),
            (
                self.USERNAME_PATTERN,
                "USERNAME",
                0.90,
            ),
            (
                self.DATE_PATTERN,
                "DATE",
                0.90,
            ),
            (
                self.TIME_PATTERN,
                "TIME",
                0.90,
            ),
        )

        for pattern, entity_type, confidence in patterns:

            for match in pattern.finditer(text):
                value = match.group(0).strip()

                if not value:
                    continue

                entities.append(
                    self.create_entity(
                        text=value,
                        entity_type=entity_type,
                        confidence=confidence,
                        source="regex",
                        start=match.start(),
                        end=match.end(),
                    )
                )

        return entities

    # =========================================================================
    # TECHNOLOGY EXTRACTION
    # =========================================================================

    def extract_technology_entities(
        self,
        text: str,
    ) -> list[Entity]:
        """
        Detect known technologies.
        """

        entities: list[Entity] = []

        normalized_text = text.lower()

        # Longest first prevents:
        #
        # "react native"
        #
        # from becoming:
        #
        # "react"
        #
        # and "native".
        technologies = sorted(
            self.TECHNOLOGIES,
            key=len,
            reverse=True,
        )

        for technology in technologies:

            pattern = re.compile(
                rf"(?<![\w])" rf"{re.escape(technology)}" rf"(?![\w])",
                re.IGNORECASE,
            )

            for match in pattern.finditer(normalized_text):

                actual = text[match.start() : match.end()]

                entities.append(
                    self.create_entity(
                        text=actual,
                        entity_type="TECHNOLOGY",
                        confidence=0.91,
                        source="technology_dictionary",
                        start=match.start(),
                        end=match.end(),
                    )
                )

        return entities

    # =========================================================================
    # BRAND / COMPANY EXTRACTION
    # =========================================================================

    def extract_known_entities(
        self,
        text: str,
    ) -> list[Entity]:
        """
        Detect known brands, companies and locations.
        """

        entities: list[Entity] = []

        dictionaries = (
            (
                self.KNOWN_BRANDS,
                "BRAND",
                0.88,
            ),
            (
                self.KNOWN_COMPANIES,
                "COMPANY",
                0.90,
            ),
            (
                self.KNOWN_LOCATIONS,
                "LOCATION",
                0.89,
            ),
        )

        for values, entity_type, confidence in dictionaries:

            for value in sorted(
                values,
                key=len,
                reverse=True,
            ):

                pattern = re.compile(
                    rf"(?<![\w])" rf"{re.escape(value)}" rf"(?![\w])",
                    re.IGNORECASE,
                )

                for match in pattern.finditer(text):

                    actual = text[match.start() : match.end()]

                    entities.append(
                        self.create_entity(
                            text=actual,
                            entity_type=entity_type,
                            confidence=confidence,
                            source="dictionary",
                            start=match.start(),
                            end=match.end(),
                        )
                    )

        return entities

    # =========================================================================
    # GEMINI ENTITY PARSING
    # =========================================================================

    def parse_ai_entities(
        self,
        entities: Any,
    ) -> list[Entity]:
        """
        Convert Gemini/AI structured entities into canonical entities.

        Supported input examples:

            [
                {
                    "text": "Adidas",
                    "type": "BRAND",
                    "confidence": 0.96
                }
            ]

        Also supports:

            [
                {
                    "name": "Adidas",
                    "entity_type": "brand"
                }
            ]
        """

        if not isinstance(
            entities,
            (list, tuple),
        ):
            return []

        result: list[Entity] = []

        for item in entities:

            if isinstance(
                item,
                str,
            ):
                result.append(
                    self.create_entity(
                        text=item,
                        entity_type="UNKNOWN",
                        confidence=0.60,
                        source="ai",
                    )
                )

                continue

            if not isinstance(
                item,
                dict,
            ):
                continue

            text = (
                item.get("text")
                or item.get("name")
                or item.get("value")
                or item.get("entity")
            )

            if not text:
                continue

            entity_type = (
                item.get("type")
                or item.get("entity_type")
                or item.get("label")
                or "UNKNOWN"
            )

            confidence = item.get(
                "confidence",
                item.get(
                    "score",
                    0.80,
                ),
            )

            try:
                confidence = float(confidence)
            except (
                TypeError,
                ValueError,
            ):
                confidence = 0.80

            result.append(
                self.create_entity(
                    text=str(text),
                    entity_type=str(entity_type),
                    confidence=confidence,
                    source="ai",
                    metadata={
                        key: value
                        for key, value in item.items()
                        if key
                        not in {
                            "text",
                            "name",
                            "value",
                            "entity",
                            "type",
                            "entity_type",
                            "label",
                            "confidence",
                            "score",
                        }
                    },
                )
            )

        return result

    # =========================================================================
    # DEDUPLICATION
    # =========================================================================

    @staticmethod
    def deduplicate(
        entities: Iterable[Entity],
    ) -> list[Entity]:
        """
        Deduplicate entities.

        If the same normalized entity appears multiple times,
        retain the strongest confidence result.
        """

        best: dict[
            tuple[str, str],
            Entity,
        ] = {}

        for entity in entities:

            key = (
                entity.entity_type,
                entity.normalized,
            )

            existing = best.get(key)

            if existing is None:
                best[key] = entity
                continue

            if entity.confidence > existing.confidence:

                # Preserve useful metadata from both.
                merged_metadata = {
                    **existing.metadata,
                    **entity.metadata,
                }

                entity.metadata = merged_metadata

                best[key] = entity

            else:

                existing.metadata.update(entity.metadata)

        return list(best.values())

    # =========================================================================
    # SORTING
    # =========================================================================

    @staticmethod
    def sort_entities(
        entities: Iterable[Entity],
    ) -> list[Entity]:
        """
        Sort entities by:
            1. confidence
            2. entity type
            3. text
        """

        return sorted(
            entities,
            key=lambda entity: (
                -entity.confidence,
                entity.entity_type,
                entity.normalized,
            ),
        )

    # =========================================================================
    # MAIN EXTRACTION
    # =========================================================================

    def extract(
        self,
        *,
        text: Any = None,
        ai_entities: Any = None,
        include_dictionary_entities: bool = True,
        include_technologies: bool = True,
    ) -> list[Entity]:
        """
        Extract entities from all available sources.
        """

        text_value = self.clean_text(text)

        entities: list[Entity] = []

        # ---------------------------------------------------------------------
        # Structured AI entities
        # ---------------------------------------------------------------------

        entities.extend(self.parse_ai_entities(ai_entities))

        # ---------------------------------------------------------------------
        # Rule-based extraction
        # ---------------------------------------------------------------------

        if text_value:

            entities.extend(self.extract_pattern_entities(text_value))

            if include_dictionary_entities:

                entities.extend(self.extract_known_entities(text_value))

            if include_technologies:

                entities.extend(self.extract_technology_entities(text_value))

        # ---------------------------------------------------------------------
        # Deduplicate
        # ---------------------------------------------------------------------

        entities = self.deduplicate(entities)

        # ---------------------------------------------------------------------
        # Sort
        # ---------------------------------------------------------------------

        entities = self.sort_entities(entities)

        return entities

    # =========================================================================
    # SEARCHABLE ENTITY TEXT
    # =========================================================================

    @staticmethod
    def build_entity_text(
        entities: Iterable[Entity],
    ) -> str:
        """
        Build a compact entity representation for embeddings/search.
        """

        parts: list[str] = []

        for entity in entities:

            if not entity.normalized:
                continue

            parts.append(f"{entity.entity_type}: " f"{entity.normalized}")

        return " | ".join(parts)

    # =========================================================================
    # ENTITY SUMMARY
    # =========================================================================

    @staticmethod
    def summarize(
        entities: Iterable[Entity],
    ) -> dict[str, list[str]]:
        """
        Group entities by type.
        """

        summary: dict[
            str,
            list[str],
        ] = {}

        for entity in entities:

            summary.setdefault(
                entity.entity_type,
                [],
            ).append(entity.text)

        return summary

    # =========================================================================
    # HIGH-VALUE ENTITIES
    # =========================================================================

    @staticmethod
    def high_value_entities(
        entities: Iterable[Entity],
        minimum_confidence: float = 0.80,
    ) -> list[Entity]:
        """
        Return high-confidence entities useful for search.
        """

        return [
            entity for entity in entities if entity.confidence >= minimum_confidence
        ]

    # =========================================================================
    # SERIALIZATION
    # =========================================================================

    @staticmethod
    def to_dict_list(
        entities: Iterable[Entity],
    ) -> list[dict[str, Any]]:
        """
        Convert entities into API-safe dictionaries.
        """

        return [entity.to_dict() for entity in entities]


# ============================================================================
# DEFAULT SERVICE
# ============================================================================

entity_service = EntityService()


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def extract_entities(
    text: Any = None,
    ai_entities: Any = None,
) -> list[Entity]:
    """
    Convenience wrapper.
    """

    return entity_service.extract(
        text=text,
        ai_entities=ai_entities,
    )


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "ENTITY_TYPES",
    "Entity",
    "EntityService",
    "entity_service",
    "extract_entities",
]
