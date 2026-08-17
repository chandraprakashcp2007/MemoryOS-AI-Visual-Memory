"""
MemoryOS - Intent Detection Service

Determines the semantic intent of a memory.

Intent answers:
    "What is this memory mainly about?"

The service combines:

    1. Explicit Gemini/AI intent
    2. Keyword signals
    3. Entity signals
    4. Contextual scoring
    5. Confidence normalization
    6. Intent ranking

Design principles
-----------------
- deterministic fallback
- AI-independent
- explainable
- confidence-aware
- easy to extend
- safe for noisy OCR
- optimized for screenshot memories

Example
-------
Input:
    "Adidas running shoes ₹14,999"

Output:
    SHOPPING

Input:
    "Chennai to Delhi flight 20 August"

Output:
    TRAVEL

Input:
    "Python FastAPI tutorial"

Output:
    LEARNING
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

from backend.services.entity_service import Entity

logger = logging.getLogger(__name__)


# ============================================================================
# INTENT DEFINITIONS
# ============================================================================

INTENTS = (
    "SHOPPING",
    "BUY",
    "COMPARE",
    "TRAVEL",
    "BOOKING",
    "FOOD",
    "PAYMENT",
    "TRACKING",
    "LEARNING",
    "REFERENCE",
    "DOCUMENT",
    "COMMUNICATION",
    "SOCIAL",
    "ENTERTAINMENT",
    "WORK",
    "FINANCE",
    "HEALTH",
    "EVENT",
    "LOCATION",
    "TECHNOLOGY",
    "REMINDER",
    "INFORMATION",
    "PLANNING",
    "OTHER",
)


# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass(slots=True)
class IntentCandidate:
    """
    One possible intent with supporting evidence.
    """

    intent: str

    confidence: float

    score: float

    evidence: list[str] = field(default_factory=list)

    source: str = "rules"

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "confidence": round(
                self.confidence,
                4,
            ),
            "score": round(
                self.score,
                4,
            ),
            "evidence": self.evidence,
            "source": self.source,
        }


@dataclass(slots=True)
class IntentResult:
    """
    Final intent analysis result.
    """

    primary_intent: str

    confidence: float

    candidates: list[IntentCandidate]

    explanation: str

    source: str

    signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_intent": self.primary_intent,
            "confidence": round(
                self.confidence,
                4,
            ),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "explanation": self.explanation,
            "source": self.source,
            "signals": self.signals,
        }


# ============================================================================
# INTENT SERVICE
# ============================================================================


class IntentService:
    """
    Rule-based + AI-compatible intent classifier.
    """

    # ------------------------------------------------------------------------
    # INTENT KEYWORDS
    # ------------------------------------------------------------------------

    INTENT_KEYWORDS: dict[
        str,
        tuple[str, ...],
    ] = {
        "SHOPPING": (
            "shop",
            "shopping",
            "product",
            "products",
            "item",
            "items",
            "store",
            "catalog",
            "cart",
            "wishlist",
            "price",
            "deal",
            "discount",
            "offer",
            "sale",
            "buy",
            "purchase",
        ),
        "BUY": (
            "buy",
            "purchase",
            "order",
            "checkout",
            "add to cart",
            "place order",
            "purchase now",
            "get this",
        ),
        "COMPARE": (
            "compare",
            "comparison",
            "vs",
            "versus",
            "difference",
            "better",
            "best",
            "cheaper",
            "alternative",
            "alternatives",
            "which one",
        ),
        "TRAVEL": (
            "travel",
            "trip",
            "journey",
            "flight",
            "train",
            "bus",
            "airport",
            "railway",
            "hotel",
            "tour",
            "destination",
            "vacation",
            "holiday",
            "tourism",
        ),
        "BOOKING": (
            "book",
            "booking",
            "reserve",
            "reservation",
            "ticket",
            "appointment",
            "schedule",
            "confirm booking",
        ),
        "FOOD": (
            "food",
            "restaurant",
            "menu",
            "dish",
            "meal",
            "lunch",
            "dinner",
            "breakfast",
            "snack",
            "recipe",
            "delivery",
            "pizza",
            "burger",
            "biryani",
        ),
        "PAYMENT": (
            "payment",
            "paid",
            "pay",
            "transaction",
            "upi",
            "card",
            "credit",
            "debit",
            "receipt",
            "payment successful",
            "payment failed",
            "amount paid",
        ),
        "TRACKING": (
            "track",
            "tracking",
            "delivery",
            "shipped",
            "shipment",
            "dispatch",
            "delivered",
            "order status",
            "tracking id",
        ),
        "LEARNING": (
            "learn",
            "learning",
            "tutorial",
            "course",
            "lesson",
            "lecture",
            "study",
            "education",
            "exam",
            "question",
            "solution",
            "notes",
            "documentation",
            "guide",
            "how to",
            "explain",
        ),
        "REFERENCE": (
            "reference",
            "documentation",
            "manual",
            "specification",
            "spec",
            "cheatsheet",
            "reference material",
        ),
        "DOCUMENT": (
            "document",
            "pdf",
            "invoice",
            "receipt",
            "certificate",
            "report",
            "form",
            "letter",
            "application",
            "statement",
        ),
        "COMMUNICATION": (
            "message",
            "email",
            "mail",
            "chat",
            "conversation",
            "reply",
            "contact",
            "call",
            "phone",
        ),
        "SOCIAL": (
            "instagram",
            "facebook",
            "twitter",
            "x.com",
            "social",
            "post",
            "story",
            "reel",
            "profile",
            "followers",
            "likes",
            "comment",
        ),
        "ENTERTAINMENT": (
            "movie",
            "film",
            "series",
            "episode",
            "song",
            "music",
            "video",
            "youtube",
            "netflix",
            "spotify",
            "game",
            "gaming",
        ),
        "WORK": (
            "work",
            "office",
            "meeting",
            "project",
            "task",
            "deadline",
            "employee",
            "client",
            "business",
            "presentation",
            "resume",
            "interview",
        ),
        "FINANCE": (
            "bank",
            "banking",
            "salary",
            "investment",
            "stock",
            "shares",
            "loan",
            "emi",
            "tax",
            "finance",
            "account balance",
            "expense",
            "budget",
        ),
        "HEALTH": (
            "health",
            "medicine",
            "doctor",
            "hospital",
            "clinic",
            "symptom",
            "prescription",
            "treatment",
            "exercise",
            "workout",
            "fitness",
        ),
        "EVENT": (
            "event",
            "birthday",
            "anniversary",
            "wedding",
            "conference",
            "festival",
            "concert",
            "function",
            "ceremony",
        ),
        "LOCATION": (
            "location",
            "address",
            "map",
            "place",
            "nearby",
            "directions",
            "route",
            "where",
        ),
        "TECHNOLOGY": (
            "python",
            "programming",
            "coding",
            "software",
            "developer",
            "development",
            "api",
            "database",
            "android",
            "flutter",
            "machine learning",
            "artificial intelligence",
            "ai",
            "github",
            "cloud",
        ),
        "REMINDER": (
            "remember",
            "reminder",
            "remind",
            "don't forget",
            "deadline",
            "due",
            "tomorrow",
            "later",
        ),
        "INFORMATION": (
            "information",
            "info",
            "details",
            "fact",
            "facts",
            "news",
            "update",
            "announcement",
        ),
        "PLANNING": (
            "plan",
            "planning",
            "schedule",
            "roadmap",
            "strategy",
            "itinerary",
            "checklist",
        ),
    }

    # ------------------------------------------------------------------------
    # ENTITY → INTENT SIGNALS
    # ------------------------------------------------------------------------

    ENTITY_SIGNALS: dict[
        str,
        dict[str, float],
    ] = {
        "MONEY": {
            "SHOPPING": 0.8,
            "BUY": 0.7,
            "PAYMENT": 0.9,
            "FINANCE": 0.6,
        },
        "BRAND": {
            "SHOPPING": 0.7,
            "BUY": 0.4,
        },
        "PRODUCT": {
            "SHOPPING": 0.9,
            "BUY": 0.7,
            "COMPARE": 0.5,
        },
        "ORDER_ID": {
            "BUY": 0.7,
            "TRACKING": 1.0,
            "PAYMENT": 0.5,
        },
        "INVOICE_ID": {
            "PAYMENT": 0.9,
            "DOCUMENT": 0.8,
            "FINANCE": 0.6,
        },
        "LOCATION": {
            "TRAVEL": 0.6,
            "LOCATION": 0.8,
        },
        "ADDRESS": {
            "LOCATION": 1.0,
            "TRAVEL": 0.5,
        },
        "DATE": {
            "EVENT": 0.5,
            "BOOKING": 0.5,
            "TRAVEL": 0.4,
            "REMINDER": 0.4,
            "PLANNING": 0.3,
        },
        "DATETIME": {
            "BOOKING": 0.7,
            "EVENT": 0.7,
            "REMINDER": 0.6,
            "PLANNING": 0.5,
        },
        "TIME": {
            "BOOKING": 0.5,
            "EVENT": 0.4,
            "REMINDER": 0.4,
        },
        "EMAIL": {
            "COMMUNICATION": 0.8,
        },
        "PHONE": {
            "COMMUNICATION": 0.8,
        },
        "URL": {
            "REFERENCE": 0.4,
            "INFORMATION": 0.3,
        },
        "TECHNOLOGY": {
            "TECHNOLOGY": 1.0,
            "LEARNING": 0.5,
        },
        "DOCUMENT": {
            "DOCUMENT": 1.0,
            "REFERENCE": 0.6,
        },
    }

    # =========================================================================
    # TEXT NORMALIZATION
    # =========================================================================

    @staticmethod
    def clean_text(
        value: Any,
    ) -> str:
        """
        Normalize text before intent analysis.
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

        return value.strip()

    @staticmethod
    def normalize_text(
        value: str,
    ) -> str:
        """
        Normalize text for matching.
        """

        value = value.lower()

        value = re.sub(
            r"\s+",
            " ",
            value,
        )

        return value.strip()

    # =========================================================================
    # KEYWORD SCORING
    # =========================================================================

    def score_keywords(
        self,
        text: str,
    ) -> tuple[
        dict[str, float],
        dict[str, list[str]],
    ]:
        """
        Score intents using keyword evidence.
        """

        scores: dict[str, float] = {}

        evidence: dict[
            str,
            list[str],
        ] = {}

        normalized = self.normalize_text(text)

        for intent, keywords in self.INTENT_KEYWORDS.items():

            for keyword in keywords:

                keyword_normalized = keyword.lower()

                if keyword_normalized in normalized:

                    # Longer phrases are stronger signals.
                    weight = 1.0 if " " in keyword else 0.7

                    scores[intent] = (
                        scores.get(
                            intent,
                            0.0,
                        )
                        + weight
                    )

                    evidence.setdefault(
                        intent,
                        [],
                    ).append(keyword)

        return scores, evidence

    # =========================================================================
    # ENTITY SCORING
    # =========================================================================

    def score_entities(
        self,
        entities: Iterable[Entity],
    ) -> tuple[
        dict[str, float],
        dict[str, list[str]],
    ]:
        """
        Convert extracted entities into intent signals.
        """

        scores: dict[str, float] = {}

        evidence: dict[
            str,
            list[str],
        ] = {}

        for entity in entities:

            entity_type = entity.entity_type

            mappings = self.ENTITY_SIGNALS.get(
                entity_type,
                {},
            )

            for intent, weight in mappings.items():

                contribution = weight * max(
                    0.0,
                    min(
                        1.0,
                        entity.confidence,
                    ),
                )

                scores[intent] = (
                    scores.get(
                        intent,
                        0.0,
                    )
                    + contribution
                )

                evidence.setdefault(
                    intent,
                    [],
                ).append(f"{entity_type}:" f"{entity.text}")

        return scores, evidence

    # =========================================================================
    # AI INTENT PARSING
    # =========================================================================

    def parse_ai_intent(
        self,
        ai_result: Any,
    ) -> IntentCandidate | None:
        """
        Parse structured AI intent output.

        Supported:

            {
                "intent": "SHOPPING",
                "confidence": 0.94
            }

        or:

            {
                "primary_intent": "SHOPPING",
                "confidence": 0.94
            }
        """

        if not isinstance(
            ai_result,
            Mapping,
        ):
            return None

        intent = (
            ai_result.get("intent")
            or ai_result.get("primary_intent")
            or ai_result.get("label")
        )

        if not intent:
            return None

        intent = self.normalize_intent(intent)

        if intent == "OTHER":
            return None

        confidence = ai_result.get(
            "confidence",
            ai_result.get(
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

        confidence = max(
            0.0,
            min(
                1.0,
                confidence,
            ),
        )

        evidence = ai_result.get(
            "evidence",
            [],
        )

        if isinstance(
            evidence,
            str,
        ):
            evidence = [evidence]

        if not isinstance(
            evidence,
            list,
        ):
            evidence = []

        return IntentCandidate(
            intent=intent,
            confidence=confidence,
            score=confidence,
            evidence=[str(item) for item in evidence],
            source="ai",
        )

    # =========================================================================
    # INTENT NORMALIZATION
    # =========================================================================

    @staticmethod
    def normalize_intent(
        value: Any,
    ) -> str:
        """
        Normalize AI/rule labels.
        """

        if not value:
            return "OTHER"

        value = str(value).strip().upper()

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

        value = aliases.get(
            value,
            value,
        )

        if value not in INTENTS:
            return "OTHER"

        return value

    # =========================================================================
    # CONFIDENCE NORMALIZATION
    # =========================================================================

    @staticmethod
    def normalize_scores(
        scores: Mapping[str, float],
    ) -> dict[str, float]:
        """
        Convert arbitrary intent scores into 0..1 confidence values.
        """

        if not scores:
            return {}

        maximum = max(scores.values())

        if maximum <= 0:
            return {key: 0.0 for key in scores}

        return {
            key: min(
                1.0,
                max(
                    0.0,
                    value / maximum,
                ),
            )
            for key, value in scores.items()
        }

    # =========================================================================
    # EXPLANATION
    # =========================================================================

    @staticmethod
    def build_explanation(
        intent: str,
        evidence: list[str],
    ) -> str:
        """
        Produce human-readable explanation.
        """

        if not evidence:
            return (
                f"MemoryOS classified this memory "
                f"as {intent.lower()} based on "
                f"available contextual signals."
            )

        unique_evidence = list(dict.fromkeys(evidence))

        display = ", ".join(unique_evidence[:5])

        return f"Detected {intent.lower()} signals " f"from: {display}."

    # =========================================================================
    # MAIN CLASSIFICATION
    # =========================================================================

    def classify(
        self,
        *,
        text: Any = None,
        entities: Iterable[Entity] | None = None,
        ai_result: Any = None,
    ) -> IntentResult:
        """
        Determine the primary intent.

        Priority:

            AI signal
                ↓
            keyword signal
                ↓
            entity signal
                ↓
            fallback
        """

        text_value = self.clean_text(text)

        entity_list = list(entities or [])

        # ---------------------------------------------------------------------
        # Rule scores
        # ---------------------------------------------------------------------

        keyword_scores, keyword_evidence = self.score_keywords(text_value)

        entity_scores, entity_evidence = self.score_entities(entity_list)

        combined_scores: dict[
            str,
            float,
        ] = {}

        evidence: dict[
            str,
            list[str],
        ] = {}

        # Keyword contribution.
        for intent, score in keyword_scores.items():

            combined_scores[intent] = (
                combined_scores.get(
                    intent,
                    0.0,
                )
                + score
            )

            evidence.setdefault(
                intent,
                [],
            ).extend(
                keyword_evidence.get(
                    intent,
                    [],
                )
            )

        # Entity contribution.
        for intent, score in entity_scores.items():

            combined_scores[intent] = (
                combined_scores.get(
                    intent,
                    0.0,
                )
                + score
            )

            evidence.setdefault(
                intent,
                [],
            ).extend(
                entity_evidence.get(
                    intent,
                    [],
                )
            )

        # ---------------------------------------------------------------------
        # AI signal
        # ---------------------------------------------------------------------

        ai_candidate = self.parse_ai_intent(ai_result)

        if ai_candidate:

            # AI gets a strong but NOT absolute influence.
            #
            # This means a bad/uncertain AI result cannot completely destroy
            # deterministic evidence.
            combined_scores[ai_candidate.intent] = combined_scores.get(
                ai_candidate.intent,
                0.0,
            ) + (ai_candidate.confidence * 2.5)

            evidence.setdefault(
                ai_candidate.intent,
                [],
            ).extend(ai_candidate.evidence)

        # ---------------------------------------------------------------------
        # No signals
        # ---------------------------------------------------------------------

        if not combined_scores:

            candidate = IntentCandidate(
                intent="OTHER",
                confidence=0.35,
                score=0.35,
                evidence=[],
                source="fallback",
            )

            return IntentResult(
                primary_intent="OTHER",
                confidence=0.35,
                candidates=[candidate],
                explanation=("No strong intent signals " "were detected."),
                source="fallback",
                signals=[],
            )

        # ---------------------------------------------------------------------
        # Normalize
        # ---------------------------------------------------------------------

        normalized_scores = self.normalize_scores(combined_scores)

        # ---------------------------------------------------------------------
        # Candidates
        # ---------------------------------------------------------------------

        candidates: list[IntentCandidate] = []

        for intent, score in combined_scores.items():

            confidence = normalized_scores.get(
                intent,
                0.0,
            )

            candidates.append(
                IntentCandidate(
                    intent=intent,
                    confidence=confidence,
                    score=score,
                    evidence=list(
                        dict.fromkeys(
                            evidence.get(
                                intent,
                                [],
                            )
                        )
                    ),
                    source=("hybrid" if ai_candidate else "rules"),
                )
            )

        candidates.sort(
            key=lambda candidate: (
                -candidate.confidence,
                -candidate.score,
            )
        )

        # Keep useful candidates only.
        candidates = candidates[:5]

        primary = candidates[0]

        # ---------------------------------------------------------------------
        # Confidence calibration
        # ---------------------------------------------------------------------

        confidence = primary.confidence

        # If AI agrees with the strongest rule result,
        # increase confidence slightly.
        if ai_candidate and ai_candidate.intent == primary.intent:
            confidence = min(
                1.0,
                confidence + 0.05,
            )

        # If only weak signals exist, cap confidence.
        if primary.score < 0.8 and not ai_candidate:
            confidence = min(
                confidence,
                0.72,
            )

        explanation = self.build_explanation(
            primary.intent,
            primary.evidence,
        )

        all_signals = list(dict.fromkeys(primary.evidence))

        return IntentResult(
            primary_intent=primary.intent,
            confidence=confidence,
            candidates=candidates,
            explanation=explanation,
            source=("hybrid" if ai_candidate else "rules"),
            signals=all_signals,
        )


# ============================================================================
# DEFAULT SERVICE
# ============================================================================

intent_service = IntentService()


# ============================================================================
# CONVENIENCE API
# ============================================================================


def detect_intent(
    text: Any = None,
    entities: Iterable[Entity] | None = None,
    ai_result: Any = None,
) -> IntentResult:
    """
    Convenience wrapper for intent detection.
    """

    return intent_service.classify(
        text=text,
        entities=entities,
        ai_result=ai_result,
    )


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "INTENTS",
    "IntentCandidate",
    "IntentResult",
    "IntentService",
    "intent_service",
    "detect_intent",
]
