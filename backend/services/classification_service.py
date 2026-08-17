"""
MemoryOS - Classification Service

Multi-signal memory classification engine.

Purpose
-------
Determine the most likely category of a captured memory using:

    1. Gemini/Vision category
    2. OCR text
    3. AI-generated description
    4. Extracted entities
    5. Keywords
    6. Intent
    7. Explicit keyword rules

The classifier is intentionally independent from Gemini.

If Gemini fails, local signals can still classify a memory.

Architecture
------------

                    ┌────────────────────┐
                    │   Gemini Category   │
                    └──────────┬─────────┘
                               │
     OCR ──────────────────────┤
     AI description ───────────┤
     Entities ─────────────────┤
     Keywords ─────────────────┤
     Intent ───────────────────┘
                               ↓
                    ┌────────────────────┐
                    │  Signal Scoring    │
                    └──────────┬─────────┘
                               ↓
                    ┌────────────────────┐
                    │ Category Ranking   │
                    └──────────┬─────────┘
                               ↓
                    Category + Confidence

Design goals
------------
- deterministic
- explainable
- lightweight
- testable
- Gemini-independent
- extensible
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# CATEGORIES
# ============================================================================

CATEGORIES = (
    "shopping",
    "finance",
    "education",
    "work",
    "travel",
    "food",
    "entertainment",
    "social",
    "communication",
    "technology",
    "health",
    "documents",
    "productivity",
    "personal",
    "other",
)


# ============================================================================
# CATEGORY ALIASES
# ============================================================================

CATEGORY_ALIASES: dict[str, str] = {
    "purchase": "shopping",
    "purchases": "shopping",
    "ecommerce": "shopping",
    "e-commerce": "shopping",
    "product": "shopping",
    "products": "shopping",
    "store": "shopping",
    "stores": "shopping",
    "retail": "shopping",
    "bank": "finance",
    "banking": "finance",
    "money": "finance",
    "payment": "finance",
    "payments": "finance",
    "invoice": "finance",
    "billing": "finance",
    "bill": "finance",
    "transaction": "finance",
    "college": "education",
    "school": "education",
    "study": "education",
    "studies": "education",
    "learning": "education",
    "course": "education",
    "exam": "education",
    "assignment": "education",
    "class": "education",
    "job": "work",
    "office": "work",
    "professional": "work",
    "career": "work",
    "meeting": "work",
    "project": "work",
    "employee": "work",
    "vacation": "travel",
    "trip": "travel",
    "tour": "travel",
    "hotel": "travel",
    "flight": "travel",
    "airport": "travel",
    "train": "travel",
    "bus": "travel",
    "booking": "travel",
    "restaurant": "food",
    "recipe": "food",
    "recipes": "food",
    "menu": "food",
    "cafe": "food",
    "food-order": "food",
    "food-ordering": "food",
    "movie": "entertainment",
    "movies": "entertainment",
    "music": "entertainment",
    "game": "entertainment",
    "gaming": "entertainment",
    "series": "entertainment",
    "netflix": "entertainment",
    "message": "communication",
    "messages": "communication",
    "chat": "communication",
    "email": "communication",
    "mail": "communication",
    "software": "technology",
    "computer": "technology",
    "programming": "technology",
    "coding": "technology",
    "python": "technology",
    "android": "technology",
    "flutter": "technology",
    "github": "technology",
    "medical": "health",
    "medicine": "health",
    "doctor": "health",
    "hospital": "health",
    "healthcare": "health",
    "pdf": "documents",
    "document": "documents",
    "documents": "documents",
    "report": "documents",
    "certificate": "documents",
    "todo": "productivity",
    "task": "productivity",
    "tasks": "productivity",
    "calendar": "productivity",
    "reminder": "productivity",
}


# ============================================================================
# KEYWORD SIGNALS
# ============================================================================

CATEGORY_KEYWORDS: dict[str, dict[str, float]] = {
    "shopping": {
        "buy": 2.0,
        "bought": 2.0,
        "purchase": 2.5,
        "price": 2.0,
        "discount": 2.5,
        "offer": 2.0,
        "sale": 2.0,
        "cart": 2.5,
        "checkout": 3.0,
        "product": 1.5,
        "amazon": 2.5,
        "flipkart": 2.5,
        "myntra": 2.5,
        "meesho": 2.5,
        "adidas": 1.5,
        "nike": 1.5,
        "shoes": 1.5,
        "shirt": 1.5,
        "dress": 1.5,
        "order": 1.0,
    },
    "finance": {
        "bank": 2.5,
        "account": 1.5,
        "balance": 3.0,
        "transaction": 3.0,
        "payment": 2.5,
        "paid": 2.0,
        "amount": 2.0,
        "invoice": 3.0,
        "bill": 2.0,
        "upi": 3.0,
        "gpay": 2.5,
        "phonepe": 2.5,
        "paytm": 2.5,
        "credit": 2.0,
        "debit": 2.0,
        "salary": 2.5,
        "₹": 1.5,
        "usd": 1.5,
        "refund": 2.5,
    },
    "education": {
        "student": 2.0,
        "college": 3.0,
        "school": 3.0,
        "university": 3.0,
        "assignment": 3.0,
        "exam": 3.0,
        "semester": 2.5,
        "subject": 2.0,
        "question": 1.0,
        "questionnaire": 1.0,
        "marks": 2.0,
        "grade": 2.0,
        "cgpa": 3.0,
        "course": 2.0,
        "lecture": 2.0,
        "notes": 2.0,
        "homework": 3.0,
        "professor": 2.5,
        "teacher": 2.5,
        "class": 2.0,
    },
    "work": {
        "office": 2.5,
        "work": 2.0,
        "meeting": 3.0,
        "deadline": 3.0,
        "employee": 2.5,
        "manager": 2.5,
        "client": 2.5,
        "company": 1.5,
        "project": 2.0,
        "presentation": 2.0,
        "interview": 2.5,
        "resume": 2.5,
        "salary": 1.5,
        "task": 1.5,
    },
    "travel": {
        "flight": 3.5,
        "airport": 3.0,
        "hotel": 3.0,
        "booking": 2.5,
        "ticket": 2.5,
        "train": 2.5,
        "bus": 2.5,
        "travel": 3.0,
        "trip": 3.0,
        "tour": 2.5,
        "vacation": 3.0,
        "passport": 3.0,
        "boarding": 3.0,
        "departure": 2.5,
        "arrival": 2.5,
    },
    "food": {
        "restaurant": 3.0,
        "menu": 3.0,
        "food": 3.0,
        "recipe": 3.0,
        "cooking": 2.5,
        "cafe": 2.5,
        "coffee": 2.0,
        "pizza": 2.0,
        "burger": 2.0,
        "dosa": 2.0,
        "biryani": 2.0,
        "lunch": 2.0,
        "dinner": 2.0,
        "breakfast": 2.0,
        "zomato": 3.0,
        "swiggy": 3.0,
    },
    "entertainment": {
        "movie": 3.0,
        "cinema": 3.0,
        "music": 2.5,
        "song": 2.0,
        "game": 2.5,
        "gaming": 3.0,
        "netflix": 3.0,
        "youtube": 1.5,
        "series": 2.5,
        "episode": 2.5,
        "concert": 3.0,
    },
    "social": {
        "instagram": 3.0,
        "facebook": 3.0,
        "social": 2.5,
        "post": 1.5,
        "story": 2.0,
        "followers": 2.5,
        "follow": 2.0,
        "like": 1.5,
        "comment": 1.5,
        "profile": 1.5,
    },
    "communication": {
        "message": 3.0,
        "messages": 3.0,
        "chat": 3.0,
        "whatsapp": 3.5,
        "telegram": 3.0,
        "email": 3.0,
        "gmail": 3.0,
        "sms": 3.0,
        "call": 2.0,
        "conversation": 2.5,
    },
    "technology": {
        "python": 3.0,
        "javascript": 3.0,
        "typescript": 3.0,
        "flutter": 3.0,
        "android": 3.0,
        "kotlin": 3.0,
        "github": 2.5,
        "git": 2.0,
        "code": 2.0,
        "coding": 3.0,
        "programming": 3.0,
        "software": 2.5,
        "api": 2.5,
        "database": 2.5,
        "ai": 2.5,
        "machine learning": 3.0,
        "artificial intelligence": 3.0,
        "terminal": 2.0,
    },
    "health": {
        "doctor": 3.0,
        "hospital": 3.0,
        "medicine": 3.0,
        "medical": 3.0,
        "health": 3.0,
        "tablet": 2.0,
        "prescription": 3.0,
        "symptom": 3.0,
        "appointment": 1.5,
        "clinic": 3.0,
        "blood": 2.5,
        "diagnosis": 3.0,
    },
    "documents": {
        "document": 3.0,
        "pdf": 3.0,
        "certificate": 3.0,
        "report": 2.5,
        "form": 2.0,
        "application form": 2.5,
        "receipt": 2.5,
        "letter": 2.0,
        "official": 1.5,
    },
    "productivity": {
        "todo": 3.0,
        "task": 2.5,
        "reminder": 3.0,
        "calendar": 3.0,
        "schedule": 3.0,
        "deadline": 2.0,
        "notes": 1.5,
        "checklist": 3.0,
        "plan": 2.0,
    },
    "personal": {
        "birthday": 3.0,
        "anniversary": 3.0,
        "family": 2.5,
        "friend": 2.0,
        "personal": 2.5,
        "memories": 2.5,
        "love": 2.0,
        "relationship": 2.5,
    },
}


# ============================================================================
# RESULT TYPES
# ============================================================================


@dataclass(slots=True)
class ClassificationSignal:
    """
    One explainable classification signal.
    """

    category: str

    source: str

    value: str

    score: float

    reason: str


@dataclass(slots=True)
class ClassificationResult:
    """
    Final classification output.
    """

    category: str

    confidence: float

    scores: dict[str, float]

    signals: list[ClassificationSignal] = field(default_factory=list)

    alternatives: list[tuple[str, float]] = field(default_factory=list)

    source: str = "hybrid"

    def to_dict(self) -> dict[str, Any]:
        """
        Convert result to JSON-compatible structure.
        """

        return {
            "category": self.category,
            "confidence": self.confidence,
            "scores": self.scores,
            "signals": [
                {
                    "category": signal.category,
                    "source": signal.source,
                    "value": signal.value,
                    "score": signal.score,
                    "reason": signal.reason,
                }
                for signal in self.signals
            ],
            "alternatives": [
                {
                    "category": category,
                    "confidence": confidence,
                }
                for category, confidence in self.alternatives
            ],
            "source": self.source,
        }


# ============================================================================
# CLASSIFICATION SERVICE
# ============================================================================


class ClassificationService:
    """
    Hybrid multi-signal classification engine.
    """

    # ------------------------------------------------------------------------
    # WEIGHTS
    # ------------------------------------------------------------------------

    GEMINI_WEIGHT = 4.0

    INTENT_WEIGHT = 2.5

    KEYWORD_WEIGHT = 1.0

    ENTITY_WEIGHT = 1.5

    OCR_WEIGHT = 1.0

    AI_TEXT_WEIGHT = 1.0

    # ------------------------------------------------------------------------
    # NORMALIZATION
    # ------------------------------------------------------------------------

    @staticmethod
    def normalize_text(
        value: Any,
    ) -> str:
        """
        Normalize arbitrary input into searchable lowercase text.
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

        if not isinstance(
            value,
            str,
        ):
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

        return value.strip().lower()

    # ------------------------------------------------------------------------
    # CATEGORY NORMALIZATION
    # ------------------------------------------------------------------------

    @classmethod
    def normalize_category(
        cls,
        value: Any,
    ) -> str | None:
        """
        Convert arbitrary category names into canonical categories.
        """

        normalized = cls.normalize_text(value)

        normalized = normalized.replace(
            "_",
            "-",
        )

        if normalized in CATEGORIES:
            return normalized

        return CATEGORY_ALIASES.get(normalized)

    # ------------------------------------------------------------------------
    # ENTITY EXTRACTION
    # ------------------------------------------------------------------------

    @classmethod
    def extract_entity_texts(
        cls,
        entities: Any,
    ) -> list[str]:
        """
        Extract searchable entity names from dictionaries or strings.
        """

        if not isinstance(
            entities,
            (list, tuple),
        ):
            return []

        result: list[str] = []

        for entity in entities:

            if isinstance(
                entity,
                dict,
            ):
                value = entity.get(
                    "text",
                    entity.get(
                        "name",
                        "",
                    ),
                )

            elif isinstance(
                entity,
                str,
            ):
                value = entity

            else:
                continue

            value = cls.normalize_text(value)

            if value:
                result.append(value)

        return result

    # ------------------------------------------------------------------------
    # KEYWORD MATCHING
    # ------------------------------------------------------------------------

    @classmethod
    def keyword_score(
        cls,
        text: str,
        category: str,
    ) -> tuple[float, list[ClassificationSignal]]:
        """
        Calculate keyword-based category score.
        """

        text = cls.normalize_text(text)

        score = 0.0

        signals: list[ClassificationSignal] = []

        keyword_map = CATEGORY_KEYWORDS.get(
            category,
            {},
        )

        for keyword, weight in keyword_map.items():

            normalized_keyword = cls.normalize_text(keyword)

            if not normalized_keyword:
                continue

            # Multi-word keywords need direct substring matching.
            if " " in normalized_keyword:

                matched = normalized_keyword in text

            else:

                # Word boundary prevents:
                #
                # "class" matching "classification"
                #
                # accidentally.
                matched = bool(
                    re.search(
                        rf"\b{re.escape(normalized_keyword)}\b",
                        text,
                    )
                )

                # Currency symbols aren't word characters.
                if normalized_keyword in {
                    "₹",
                }:
                    matched = normalized_keyword in text

            if matched:

                contribution = weight * cls.KEYWORD_WEIGHT

                score += contribution

                signals.append(
                    ClassificationSignal(
                        category=category,
                        source="keyword",
                        value=keyword,
                        score=contribution,
                        reason=(f"Detected category keyword " f"'{keyword}'."),
                    )
                )

        return score, signals

    # ------------------------------------------------------------------------
    # GEMINI CATEGORY
    # ------------------------------------------------------------------------

    @classmethod
    def gemini_category_score(
        cls,
        category: Any,
    ) -> tuple[str | None, float]:
        """
        Convert Gemini category into a weighted signal.
        """

        normalized = cls.normalize_category(category)

        if normalized is None:
            return None, 0.0

        return (
            normalized,
            cls.GEMINI_WEIGHT,
        )

    # ------------------------------------------------------------------------
    # INTENT SCORING
    # ------------------------------------------------------------------------

    @classmethod
    def intent_score(
        cls,
        intent: Any,
        category: str,
    ) -> tuple[float, list[ClassificationSignal]]:
        """
        Score classification based on user intent.
        """

        text = cls.normalize_text(intent)

        if not text:
            return 0.0, []

        score, signals = cls.keyword_score(
            text,
            category,
        )

        adjusted = score * cls.INTENT_WEIGHT

        adjusted_signals: list[ClassificationSignal] = []

        for signal in signals:

            adjusted_signals.append(
                ClassificationSignal(
                    category=signal.category,
                    source="intent",
                    value=signal.value,
                    score=(signal.score * cls.INTENT_WEIGHT),
                    reason=(f"Intent contains " f"'{signal.value}'."),
                )
            )

        return adjusted, adjusted_signals

    # ------------------------------------------------------------------------
    # ENTITY SCORING
    # ------------------------------------------------------------------------

    @classmethod
    def entity_score(
        cls,
        entities: list[str],
        category: str,
    ) -> tuple[float, list[ClassificationSignal]]:
        """
        Score entities against category keywords.
        """

        if not entities:
            return 0.0, []

        combined = " ".join(entities)

        score, signals = cls.keyword_score(
            combined,
            category,
        )

        adjusted_signals = [
            ClassificationSignal(
                category=signal.category,
                source="entity",
                value=signal.value,
                score=(signal.score * cls.ENTITY_WEIGHT),
                reason=(f"Entity information contains " f"'{signal.value}'."),
            )
            for signal in signals
        ]

        return (
            score * cls.ENTITY_WEIGHT,
            adjusted_signals,
        )

    # ------------------------------------------------------------------------
    # MAIN CLASSIFICATION
    # ------------------------------------------------------------------------

    def classify(
        self,
        *,
        gemini_category: Any = None,
        ocr_text: Any = None,
        ai_text: Any = None,
        entities: Any = None,
        keywords: Any = None,
        intent: Any = None,
    ) -> ClassificationResult:
        """
        Perform hybrid classification.

        All inputs are optional.

        This is important because MemoryOS must remain functional if:
            - OCR fails
            - Gemini fails
            - entity extraction fails
        """

        scores: dict[str, float] = {category: 0.0 for category in CATEGORIES}

        signals: list[ClassificationSignal] = []

        # ====================================================================
        # NORMALIZE INPUTS
        # ====================================================================

        ocr = self.normalize_text(ocr_text)

        ai = self.normalize_text(ai_text)

        intent_text = self.normalize_text(intent)

        if isinstance(
            keywords,
            str,
        ):
            keyword_values = [keywords]

        elif isinstance(
            keywords,
            (list, tuple, set),
        ):
            keyword_values = [self.normalize_text(value) for value in keywords]

        else:
            keyword_values = []

        keyword_values = [value for value in keyword_values if value]

        entity_values = self.extract_entity_texts(entities)

        # ====================================================================
        # GEMINI SIGNAL
        # ====================================================================

        gemini_category_normalized, gemini_score = self.gemini_category_score(
            gemini_category
        )

        if gemini_category_normalized and gemini_score > 0:

            scores[gemini_category_normalized] += gemini_score

            signals.append(
                ClassificationSignal(
                    category=(gemini_category_normalized),
                    source="gemini",
                    value=(gemini_category_normalized),
                    score=gemini_score,
                    reason=("Gemini Vision classified " "the memory in this category."),
                )
            )

        # ====================================================================
        # COMBINED TEXT
        # ====================================================================

        combined_text = " ".join(
            part
            for part in (
                ocr,
                ai,
                " ".join(keyword_values),
            )
            if part
        )

        # ====================================================================
        # CATEGORY KEYWORD SIGNALS
        # ====================================================================

        for category in CATEGORIES:

            if category == "other":
                continue

            score, category_signals = self.keyword_score(
                combined_text,
                category,
            )

            if score > 0:

                scores[category] += score

                signals.extend(category_signals)

        # ====================================================================
        # OCR-SPECIFIC SIGNAL
        # ====================================================================

        if ocr:

            for category in CATEGORIES:

                if category == "other":
                    continue

                score, keyword_signals = self.keyword_score(
                    ocr,
                    category,
                )

                if score <= 0:
                    continue

                contribution = score * self.OCR_WEIGHT

                scores[category] += contribution

                for signal in keyword_signals:

                    signals.append(
                        ClassificationSignal(
                            category=category,
                            source="ocr",
                            value=signal.value,
                            score=(signal.score * self.OCR_WEIGHT),
                            reason=(f"OCR text contains " f"'{signal.value}'."),
                        )
                    )

        # ====================================================================
        # AI DESCRIPTION SIGNAL
        # ====================================================================

        if ai:

            for category in CATEGORIES:

                if category == "other":
                    continue

                score, keyword_signals = self.keyword_score(
                    ai,
                    category,
                )

                if score <= 0:
                    continue

                contribution = score * self.AI_TEXT_WEIGHT

                scores[category] += contribution

                for signal in keyword_signals:

                    signals.append(
                        ClassificationSignal(
                            category=category,
                            source="vision_text",
                            value=signal.value,
                            score=(signal.score * self.AI_TEXT_WEIGHT),
                            reason=(
                                f"Vision description " f"contains " f"'{signal.value}'."
                            ),
                        )
                    )

        # ====================================================================
        # KEYWORD INPUT SIGNAL
        # ====================================================================

        if keyword_values:

            keyword_text = " ".join(keyword_values)

            for category in CATEGORIES:

                if category == "other":
                    continue

                score, keyword_signals = self.keyword_score(
                    keyword_text,
                    category,
                )

                if score <= 0:
                    continue

                scores[category] += score

                signals.extend(keyword_signals)

        # ====================================================================
        # ENTITY SIGNAL
        # ====================================================================

        if entity_values:

            for category in CATEGORIES:

                if category == "other":
                    continue

                score, entity_signals = self.entity_score(
                    entity_values,
                    category,
                )

                if score <= 0:
                    continue

                scores[category] += score

                signals.extend(entity_signals)

        # ====================================================================
        # INTENT SIGNAL
        # ====================================================================

        if intent_text:

            for category in CATEGORIES:

                if category == "other":
                    continue

                score, intent_signals = self.intent_score(
                    intent_text,
                    category,
                )

                if score <= 0:
                    continue

                scores[category] += score

                signals.extend(intent_signals)

        # ====================================================================
        # DETERMINE WINNER
        # ====================================================================

        ranked = sorted(
            scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_category, top_score = ranked[0]

        second_score = ranked[1][1] if len(ranked) > 1 else 0.0

        # ====================================================================
        # NO SIGNAL FALLBACK
        # ====================================================================

        if top_score <= 0:

            return ClassificationResult(
                category="other",
                confidence=0.20,
                scores=scores,
                signals=[],
                alternatives=[],
                source="fallback",
            )

        # ====================================================================
        # CONFIDENCE CALCULATION
        # ====================================================================

        total_score = sum(
            max(
                score,
                0.0,
            )
            for score in scores.values()
        )

        if total_score <= 0:

            confidence = 0.20

        else:

            probability = top_score / total_score

            # Margin rewards cases where the winner is clearly ahead.
            margin = top_score - second_score

            margin_factor = min(
                1.0,
                margin
                / max(
                    top_score,
                    1.0,
                ),
            )

            confidence = (probability * 0.70) + (margin_factor * 0.30)

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        # ====================================================================
        # ALTERNATIVE CATEGORIES
        # ====================================================================

        alternatives: list[tuple[str, float]] = []

        for category, score in ranked[1:4]:

            if score <= 0:
                continue

            alternative_confidence = score / total_score if total_score > 0 else 0.0

            alternatives.append(
                (
                    category,
                    alternative_confidence,
                )
            )

        # ====================================================================
        # LIMIT SIGNALS
        # ====================================================================

        signals.sort(
            key=lambda signal: signal.score,
            reverse=True,
        )

        signals = signals[:50]

        # ====================================================================
        # RESULT
        # ====================================================================

        return ClassificationResult(
            category=top_category,
            confidence=confidence,
            scores=scores,
            signals=signals,
            alternatives=alternatives,
            source="hybrid",
        )


# ============================================================================
# DEFAULT SERVICE
# ============================================================================

classification_service = ClassificationService()


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================


def classify_memory(
    *,
    gemini_category: Any = None,
    ocr_text: Any = None,
    ai_text: Any = None,
    entities: Any = None,
    keywords: Any = None,
    intent: Any = None,
) -> ClassificationResult:
    """
    Convenience wrapper around ClassificationService.
    """

    return classification_service.classify(
        gemini_category=gemini_category,
        ocr_text=ocr_text,
        ai_text=ai_text,
        entities=entities,
        keywords=keywords,
        intent=intent,
    )


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "CATEGORIES",
    "ClassificationSignal",
    "ClassificationResult",
    "ClassificationService",
    "classification_service",
    "classify_memory",
]
