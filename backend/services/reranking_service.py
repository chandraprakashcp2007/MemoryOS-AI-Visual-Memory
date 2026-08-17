"""
MemoryOS - Reranking Service
============================

Final relevance-ranking layer for MemoryOS search.

Purpose
-------

Initial retrieval systems such as FAISS and keyword search are designed
to retrieve a broad set of potentially relevant memories.

They are NOT necessarily responsible for deciding the final ordering.

Reranking takes those candidates and evaluates additional signals:

    Semantic similarity
    Keyword overlap
    Entity overlap
    Category match
    Intent match
    Metadata relevance
    Exact phrase match
    Recency
    Retrieval agreement

The result is a more useful and explainable ranking.

Architecture
------------

    Search Query
         |
         v
    Embedding Service
         |
         v
       FAISS
         |
         v
    Initial Candidates
         |
         v
    RerankingService
         |
         +--> semantic score
         +--> keyword score
         +--> entity score
         +--> metadata score
         +--> recency score
         +--> exact match
         |
         v
    Final Score
         |
         v
    Confidence
         |
         v
    Explanation

Important
---------

This service does NOT load the embedding model.

It also does NOT directly access FAISS.

Those are separate responsibilities.

This keeps the system modular and testable.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger("memoryos.reranking")


# ============================================================================
# VERSION
# ============================================================================

RERANKING_SERVICE_VERSION = "1.0.0"


# ============================================================================
# DEFAULT WEIGHTS
# ============================================================================

DEFAULT_WEIGHTS = {
    "semantic": 0.40,
    "keyword": 0.18,
    "entity": 0.12,
    "metadata": 0.10,
    "exact_match": 0.08,
    "recency": 0.05,
    "retrieval_agreement": 0.07,
}


# ============================================================================
# LIMITS
# ============================================================================

MAX_CANDIDATES = 200

MAX_TEXT_LENGTH = 50_000

MAX_EXPLANATION_TERMS = 8


# ============================================================================
# EXCEPTIONS
# ============================================================================


class RerankingError(Exception):
    """Base exception for reranking."""


class InvalidRerankingInputError(RerankingError):
    """Raised when reranking input is invalid."""


class RerankingConfigurationError(RerankingError):
    """Raised when reranking configuration is invalid."""


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True, slots=True)
class RerankingWeights:
    """
    Configurable scoring weights.

    All weights are normalized before use.
    """

    semantic: float = DEFAULT_WEIGHTS["semantic"]

    keyword: float = DEFAULT_WEIGHTS["keyword"]

    entity: float = DEFAULT_WEIGHTS["entity"]

    metadata: float = DEFAULT_WEIGHTS["metadata"]

    exact_match: float = DEFAULT_WEIGHTS["exact_match"]

    recency: float = DEFAULT_WEIGHTS["recency"]

    retrieval_agreement: float = DEFAULT_WEIGHTS["retrieval_agreement"]

    def normalized(
        self,
    ) -> "RerankingWeights":
        """
        Normalize all weights so their sum equals 1.
        """

        values = (
            self.semantic,
            self.keyword,
            self.entity,
            self.metadata,
            self.exact_match,
            self.recency,
            self.retrieval_agreement,
        )

        if any(value < 0 for value in values):
            raise RerankingConfigurationError("Reranking weights cannot be negative.")

        total = sum(values)

        if total <= 0:
            raise RerankingConfigurationError(
                "At least one reranking weight " "must be greater than zero."
            )

        return RerankingWeights(
            semantic=self.semantic / total,
            keyword=self.keyword / total,
            entity=self.entity / total,
            metadata=self.metadata / total,
            exact_match=(self.exact_match / total),
            recency=self.recency / total,
            retrieval_agreement=(self.retrieval_agreement / total),
        )


@dataclass(frozen=True, slots=True)
class RerankCandidate:
    """
    Input candidate for reranking.

    This object is intentionally independent from the database model.
    """

    memory_id: str

    semantic_score: float = 0.0

    keyword_score: float = 0.0

    metadata_score: float = 0.0

    retrieval_sources: tuple[
        str,
        ...,
    ] = ()

    memory: Mapping[
        str,
        Any,
    ] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ScoreBreakdown:
    """
    Individual scoring components.
    """

    semantic: float

    keyword: float

    entity: float

    metadata: float

    exact_match: float

    recency: float

    retrieval_agreement: float

    final_score: float


@dataclass(frozen=True, slots=True)
class RerankExplanation:
    """
    Human-readable explanation of the ranking.
    """

    reasons: tuple[str, ...]

    matched_terms: tuple[str, ...]

    matched_entities: tuple[str, ...]

    score_breakdown: ScoreBreakdown

    confidence: float


@dataclass(frozen=True, slots=True)
class RerankedResult:
    """
    Final reranked memory.
    """

    memory_id: str

    rank: int

    score: float

    confidence: float

    memory: Mapping[
        str,
        Any,
    ]

    explanation: RerankExplanation


# ============================================================================
# NORMALIZATION HELPERS
# ============================================================================


def clamp(
    value: float,
    minimum: float = 0.0,
    maximum: float = 1.0,
) -> float:
    """
    Clamp a numeric value into [minimum, maximum].
    """

    return max(
        minimum,
        min(
            float(value),
            maximum,
        ),
    )


def normalize_score(
    value: Any,
) -> float:
    """
    Safely normalize arbitrary score input.
    """

    try:
        numeric = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    if not math.isfinite(numeric):
        return 0.0

    return clamp(numeric)


# ============================================================================
# TEXT PROCESSING
# ============================================================================


def normalize_text(
    value: Any,
) -> str:
    """
    Normalize text for matching.
    """

    if value is None:
        return ""

    text = str(value)

    text = text[:MAX_TEXT_LENGTH]

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize(
    text: str,
) -> tuple[str, ...]:
    """
    Tokenize text while preserving useful technical tokens.
    """

    normalized = normalize_text(text)

    tokens = re.findall(
        r"[a-z0-9₹$€£]+(?:[._+#@:/-][a-z0-9₹$€£]+)*",
        normalized,
    )

    return tuple(dict.fromkeys(tokens))


# ============================================================================
# MEMORY TEXT EXTRACTION
# ============================================================================


SEARCHABLE_FIELDS = (
    "title",
    "summary",
    "description",
    "ocr_text",
    "visual_description",
    "search_text",
    "category",
    "subcategory",
    "intent",
    "keywords",
    "tags",
    "entities",
    "people",
    "organizations",
    "locations",
    "products",
)


def _flatten_value(
    value: Any,
) -> list[str]:
    """
    Convert nested metadata values into searchable text.
    """

    if value is None:
        return []

    if isinstance(
        value,
        str,
    ):
        return [value]

    if isinstance(
        value,
        Mapping,
    ):
        result: list[str] = []

        for key, item in value.items():

            result.append(str(key))

            result.extend(_flatten_value(item))

        return result

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        result = []

        for item in value:
            result.extend(_flatten_value(item))

        return result

    return [str(value)]


def memory_to_search_text(
    memory: Mapping[
        str,
        Any,
    ],
) -> str:
    """
    Build a unified searchable representation.
    """

    parts: list[str] = []

    for field_name in SEARCHABLE_FIELDS:

        if field_name not in memory:
            continue

        parts.extend(_flatten_value(memory[field_name]))

    return normalize_text(" ".join(parts))


# ============================================================================
# KEYWORD SCORE
# ============================================================================


def calculate_keyword_score(
    query: str,
    memory: Mapping[
        str,
        Any,
    ],
) -> tuple[
    float,
    tuple[str, ...],
]:
    """
    Calculate token overlap between query and memory.
    """

    query_tokens = set(tokenize(query))

    if not query_tokens:
        return (
            0.0,
            (),
        )

    memory_tokens = set(tokenize(memory_to_search_text(memory)))

    if not memory_tokens:
        return (
            0.0,
            (),
        )

    matched = sorted(query_tokens & memory_tokens)

    score = len(matched) / len(query_tokens)

    return (
        clamp(score),
        tuple(matched[:MAX_EXPLANATION_TERMS]),
    )


# ============================================================================
# EXACT MATCH SCORE
# ============================================================================


def calculate_exact_match_score(
    query: str,
    memory: Mapping[
        str,
        Any,
    ],
) -> float:
    """
    Detect strong exact phrase matches.
    """

    normalized_query = normalize_text(query)

    if not normalized_query:
        return 0.0

    searchable = memory_to_search_text(memory)

    if not searchable:
        return 0.0

    if normalized_query in searchable:
        return 1.0

    query_tokens = tokenize(normalized_query)

    if len(query_tokens) >= 2:

        phrase = " ".join(query_tokens)

        if phrase in searchable:
            return 0.85

    return 0.0


# ============================================================================
# ENTITY SCORE
# ============================================================================


ENTITY_FIELDS = (
    "entities",
    "people",
    "organizations",
    "locations",
    "products",
)


def extract_memory_entities(
    memory: Mapping[
        str,
        Any,
    ],
) -> tuple[str, ...]:
    """
    Extract entities from memory metadata.
    """

    entities: list[str] = []

    for field_name in ENTITY_FIELDS:

        value = memory.get(field_name)

        if value is None:
            continue

        for item in _flatten_value(value):

            normalized = normalize_text(item)

            if normalized:
                entities.append(normalized)

    return tuple(dict.fromkeys(entities))


def calculate_entity_score(
    query: str,
    memory: Mapping[
        str,
        Any,
    ],
) -> tuple[
    float,
    tuple[str, ...],
]:
    """
    Calculate query/entity overlap.
    """

    query_text = normalize_text(query)

    if not query_text:
        return (
            0.0,
            (),
        )

    entities = extract_memory_entities(memory)

    if not entities:
        return (
            0.0,
            (),
        )

    matched = [entity for entity in entities if entity in query_text]

    if not matched:
        return (
            0.0,
            (),
        )

    score = min(
        1.0,
        len(matched)
        / max(
            1,
            len(entities),
        ),
    )

    return (
        score,
        tuple(matched[:MAX_EXPLANATION_TERMS]),
    )


# ============================================================================
# CATEGORY / INTENT / METADATA SCORE
# ============================================================================


def calculate_metadata_score(
    query: str,
    memory: Mapping[
        str,
        Any,
    ],
) -> float:
    """
    Calculate broad metadata relevance.

    Exact metadata terms receive a stronger signal than arbitrary text.
    """

    query_tokens = set(tokenize(query))

    if not query_tokens:
        return 0.0

    metadata_values: list[str] = []

    for field_name in (
        "category",
        "subcategory",
        "intent",
        "tags",
        "keywords",
    ):

        metadata_values.extend(_flatten_value(memory.get(field_name)))

    metadata_text = normalize_text(" ".join(metadata_values))

    if not metadata_text:
        return 0.0

    metadata_tokens = set(tokenize(metadata_text))

    overlap = query_tokens & metadata_tokens

    if not overlap:
        return 0.0

    return clamp(len(overlap) / len(query_tokens))


# ============================================================================
# RECENCY SCORE
# ============================================================================


DATE_FIELDS = (
    "created_at",
    "captured_at",
    "indexed_at",
    "updated_at",
)


def _parse_datetime(
    value: Any,
) -> datetime | None:
    """
    Parse common ISO-style timestamps.
    """

    if isinstance(
        value,
        datetime,
    ):
        dt = value

    elif isinstance(
        value,
        str,
    ):

        text = value.strip()

        if not text:
            return None

        try:
            dt = datetime.fromisoformat(
                text.replace(
                    "Z",
                    "+00:00",
                )
            )

        except ValueError:
            return None

    else:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return dt.astimezone(timezone.utc)


def calculate_recency_score(
    memory: Mapping[
        str,
        Any,
    ],
) -> float:
    """
    Convert memory age into a soft recency score.

    This intentionally decays gradually rather than completely
    eliminating older memories.

    Approximation:

        0 days   → 1.00
        30 days  → ~0.70
        90 days  → ~0.37
        365 days → ~0.05
    """

    timestamp: datetime | None = None

    for field_name in DATE_FIELDS:

        timestamp = _parse_datetime(memory.get(field_name))

        if timestamp:
            break

    if timestamp is None:
        return 0.0

    now = datetime.now(timezone.utc)

    age_days = max(
        0.0,
        (now - timestamp).total_seconds() / 86_400,
    )

    decay = math.exp(-age_days / 120.0)

    return clamp(decay)


# ============================================================================
# RETRIEVAL AGREEMENT
# ============================================================================


def calculate_retrieval_agreement(
    candidate: RerankCandidate,
) -> float:
    """
    Reward memories discovered by multiple retrieval strategies.

    Example:

        FAISS only:
            0.50

        Keyword only:
            0.50

        FAISS + keyword:
            1.00

    Agreement is a useful signal because independent retrieval
    methods supporting the same memory increases confidence.
    """

    sources = {
        normalize_text(source) for source in candidate.retrieval_sources if source
    }

    if len(sources) >= 2:
        return 1.0

    if len(sources) == 1:
        return 0.5

    return 0.0


# ============================================================================
# CONFIDENCE
# ============================================================================


def calculate_confidence(
    score: float,
    *,
    agreement: float,
    exact_match: float,
) -> float:
    """
    Convert ranking score into an interpretable confidence value.

    Confidence is NOT a probability.

    It represents how strongly the available retrieval signals
    support the ranking.
    """

    base = clamp(score)

    agreement_bonus = 0.08 * clamp(agreement)

    exact_bonus = 0.07 * clamp(exact_match)

    confidence = min(
        1.0,
        base + agreement_bonus + exact_bonus,
    )

    return round(
        confidence,
        4,
    )


# ============================================================================
# EXPLANATION GENERATION
# ============================================================================


def build_explanation(
    *,
    query: str,
    candidate: RerankCandidate,
    breakdown: ScoreBreakdown,
    matched_terms: tuple[
        str,
        ...,
    ],
    matched_entities: tuple[
        str,
        ...,
    ],
) -> RerankExplanation:
    """
    Build human-readable ranking explanation.
    """

    reasons: list[str] = []

    if breakdown.semantic >= 0.80:
        reasons.append("strong semantic similarity")

    elif breakdown.semantic >= 0.60:
        reasons.append("good semantic similarity")

    if breakdown.keyword >= 0.70:
        reasons.append("multiple keyword matches")

    elif breakdown.keyword >= 0.30:
        reasons.append("keyword overlap")

    if matched_terms:
        reasons.append(
            "matched terms: " + ", ".join(matched_terms[:MAX_EXPLANATION_TERMS])
        )

    if matched_entities:
        reasons.append(
            "matched entities: " + ", ".join(matched_entities[:MAX_EXPLANATION_TERMS])
        )

    if breakdown.exact_match >= 0.80:
        reasons.append("exact phrase match")

    if breakdown.metadata >= 0.70:
        reasons.append("metadata supports the query")

    if breakdown.retrieval_agreement >= 1.0:
        reasons.append("multiple retrieval methods " "found this memory")

    if breakdown.recency >= 0.70:
        reasons.append("recent memory")

    if not reasons:
        reasons.append("combined retrieval signals " "indicate relevance")

    return RerankExplanation(
        reasons=tuple(reasons),
        matched_terms=matched_terms,
        matched_entities=matched_entities,
        score_breakdown=breakdown,
        confidence=calculate_confidence(
            breakdown.final_score,
            agreement=(breakdown.retrieval_agreement),
            exact_match=(breakdown.exact_match),
        ),
    )


# ============================================================================
# FINAL SCORE
# ============================================================================


def calculate_final_score(
    *,
    semantic: float,
    keyword: float,
    entity: float,
    metadata: float,
    exact_match: float,
    recency: float,
    retrieval_agreement: float,
    weights: RerankingWeights,
) -> float:
    """
    Calculate weighted final relevance score.
    """

    normalized = weights.normalized()

    score = (
        semantic * normalized.semantic
        + keyword * normalized.keyword
        + entity * normalized.entity
        + metadata * normalized.metadata
        + exact_match * normalized.exact_match
        + recency * normalized.recency
        + retrieval_agreement * normalized.retrieval_agreement
    )

    return round(
        clamp(score),
        6,
    )


# ============================================================================
# RERANKING SERVICE
# ============================================================================


class RerankingService:
    """
    Production-grade MemoryOS reranking engine.
    """

    def __init__(
        self,
        weights: RerankingWeights | None = None,
    ) -> None:

        self.weights = (weights or RerankingWeights()).normalized()

    # ------------------------------------------------------------------
    # Score one candidate
    # ------------------------------------------------------------------

    def score_candidate(
        self,
        query: str,
        candidate: RerankCandidate,
    ) -> RerankedResult:
        """
        Score a single candidate.
        """

        if not candidate.memory_id:
            raise InvalidRerankingInputError("Candidate memory_id cannot be empty.")

        memory = candidate.memory

        semantic = normalize_score(candidate.semantic_score)

        keyword, matched_terms = calculate_keyword_score(
            query,
            memory,
        )

        entity, matched_entities = calculate_entity_score(
            query,
            memory,
        )

        metadata = calculate_metadata_score(
            query,
            memory,
        )

        exact_match = calculate_exact_match_score(
            query,
            memory,
        )

        recency = calculate_recency_score(memory)

        retrieval_agreement = calculate_retrieval_agreement(candidate)

        final_score = calculate_final_score(
            semantic=semantic,
            keyword=keyword,
            entity=entity,
            metadata=metadata,
            exact_match=exact_match,
            recency=recency,
            retrieval_agreement=(retrieval_agreement),
            weights=self.weights,
        )

        breakdown = ScoreBreakdown(
            semantic=round(
                semantic,
                4,
            ),
            keyword=round(
                keyword,
                4,
            ),
            entity=round(
                entity,
                4,
            ),
            metadata=round(
                metadata,
                4,
            ),
            exact_match=round(
                exact_match,
                4,
            ),
            recency=round(
                recency,
                4,
            ),
            retrieval_agreement=round(
                retrieval_agreement,
                4,
            ),
            final_score=final_score,
        )

        explanation = build_explanation(
            query=query,
            candidate=candidate,
            breakdown=breakdown,
            matched_terms=matched_terms,
            matched_entities=(matched_entities),
        )

        return RerankedResult(
            memory_id=candidate.memory_id,
            rank=0,
            score=final_score,
            confidence=(explanation.confidence),
            memory=memory,
            explanation=explanation,
        )

    # ------------------------------------------------------------------
    # Rerank candidates
    # ------------------------------------------------------------------

    def rerank(
        self,
        query: str,
        candidates: Sequence[RerankCandidate],
    ) -> tuple[
        RerankedResult,
        ...,
    ]:
        """
        Rerank a collection of candidates.
        """

        if not query or not str(query).strip():
            raise InvalidRerankingInputError("Reranking query cannot be empty.")

        if not candidates:
            return ()

        candidates = candidates[:MAX_CANDIDATES]

        scored: list[RerankedResult] = []

        for candidate in candidates:

            try:

                result = self.score_candidate(
                    query,
                    candidate,
                )

                scored.append(result)

            except Exception as exc:

                logger.exception(
                    "Failed to rerank memory %s: %s",
                    candidate.memory_id,
                    exc,
                )

        scored.sort(
            key=lambda item: (
                item.score,
                item.confidence,
            ),
            reverse=True,
        )

        ranked: list[RerankedResult] = []

        for rank, result in enumerate(
            scored,
            start=1,
        ):

            ranked.append(
                RerankedResult(
                    memory_id=(result.memory_id),
                    rank=rank,
                    score=result.score,
                    confidence=(result.confidence),
                    memory=result.memory,
                    explanation=(result.explanation),
                )
            )

        return tuple(ranked)


# ============================================================================
# DEFAULT SERVICE
# ============================================================================


reranking_service = RerankingService()


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "RERANKING_SERVICE_VERSION",
    "RerankingError",
    "InvalidRerankingInputError",
    "RerankingConfigurationError",
    "RerankingWeights",
    "RerankCandidate",
    "ScoreBreakdown",
    "RerankExplanation",
    "RerankedResult",
    "normalize_score",
    "normalize_text",
    "tokenize",
    "memory_to_search_text",
    "calculate_keyword_score",
    "calculate_exact_match_score",
    "calculate_entity_score",
    "calculate_metadata_score",
    "calculate_recency_score",
    "calculate_retrieval_agreement",
    "calculate_confidence",
    "calculate_final_score",
    "build_explanation",
    "RerankingService",
    "reranking_service",
]
