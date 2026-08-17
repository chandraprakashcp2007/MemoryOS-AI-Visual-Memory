"""
MemoryOS - Search Service
=========================

Central orchestration layer for MemoryOS search.

Responsibilities
----------------

The SearchService coordinates:

    User Query
        ↓
    Query Normalization
        ↓
    Semantic Retrieval
        ↓
    Keyword Retrieval
        ↓
    Candidate Merging
        ↓
    Reranking
        ↓
    Explainable Results

Important architectural rule
-----------------------------

This module does NOT own:

    - embedding model loading
    - FAISS index implementation
    - database implementation
    - reranking algorithms

Those responsibilities belong to dedicated services.

This keeps MemoryOS modular and allows each component to be
tested and replaced independently.

Future services:

    embedding_service.py
    faiss_service.py
    reranking_service.py
    memory_service.py
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Protocol, Sequence

# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger("memoryos.search")


# ============================================================================
# CONSTANTS
# ============================================================================

SEARCH_SERVICE_VERSION = "1.0.0"

DEFAULT_TOP_K = 10

MAX_TOP_K = 100

MIN_QUERY_LENGTH = 1

MAX_QUERY_LENGTH = 1_000

DEFAULT_SEMANTIC_WEIGHT = 0.65

DEFAULT_KEYWORD_WEIGHT = 0.35


# ============================================================================
# EXCEPTIONS
# ============================================================================


class SearchError(Exception):
    """Base exception for MemoryOS search."""


class InvalidSearchQueryError(SearchError):
    """Raised when a search query is invalid."""


class SearchConfigurationError(SearchError):
    """Raised when search configuration is invalid."""


class SearchExecutionError(SearchError):
    """Raised when search execution fails."""


# ============================================================================
# DATA MODELS
# ============================================================================


@dataclass(frozen=True, slots=True)
class SearchQuery:
    """
    Normalized search request.

    Example:

        SearchQuery(
            text="shoes I wanted to buy",
            top_k=10,
            category="shopping",
        )
    """

    text: str

    top_k: int = DEFAULT_TOP_K

    category: str | None = None

    intent: str | None = None

    entity: str | None = None

    date_from: str | None = None

    date_to: str | None = None

    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT

    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT


@dataclass(frozen=True, slots=True)
class SearchCandidate:
    """
    Candidate returned by an individual retrieval mechanism.

    A candidate can originate from:

        semantic
        keyword
        metadata
        future retrieval methods
    """

    memory_id: str

    score: float

    source: str

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchExplanation:
    """
    Explainable explanation for why a memory matched.
    """

    matched_terms: tuple[str, ...]

    matched_entities: tuple[str, ...]

    semantic_score: float

    keyword_score: float

    metadata_score: float

    final_score: float

    reason: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    """
    Final user-facing search result.

    This is intentionally independent from the database model.
    """

    memory_id: str

    rank: int

    score: float

    title: str

    summary: str

    category: str

    intent: str

    thumbnail_path: str | None

    created_at: str | None

    image_url: str | None = None

    preview_url: str | None = None

    thumbnail_url: str | None = None

    original_image_url: str | None = None

    explanation: SearchExplanation | None = None

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """
    Complete search response.
    """

    query: str

    normalized_query: str

    results: tuple[SearchResult, ...]

    total_results: int

    took_ms: float

    service_version: str

    searched_at: str

    metadata: dict[str, Any] = field(default_factory=dict)


# ============================================================================
# SERVICE PROTOCOLS
# ============================================================================


class SemanticRetriever(Protocol):
    """
    Contract expected from the future FAISS/embedding retrieval layer.
    """

    def search(
        self,
        query: str,
        *,
        top_k: int,
    ) -> Sequence[SearchCandidate]: ...


class KeywordRetriever(Protocol):
    """
    Contract expected from the future SQLite/text retrieval layer.
    """

    def search(
        self,
        query: str,
        *,
        top_k: int,
    ) -> Sequence[SearchCandidate]: ...


class MemoryProvider(Protocol):
    """
    Contract for retrieving complete memory metadata.
    """

    def get_memory(
        self,
        memory_id: str,
    ) -> dict[str, Any] | None: ...


class Reranker(Protocol):
    """
    Contract for the future reranking service.
    """

    def rerank(
        self,
        query: SearchQuery,
        candidates: Sequence[SearchCandidate],
    ) -> Sequence[SearchCandidate]: ...


# ============================================================================
# QUERY NORMALIZATION
# ============================================================================


def normalize_query(
    query: str,
) -> str:
    """
    Normalize a user's search query.

    We intentionally preserve meaningful punctuation such as:

        ₹
        #
        +
        -

    because screenshot memories may contain products,
    prices, technical terms, and identifiers.
    """

    if query is None:
        raise InvalidSearchQueryError("Search query cannot be None.")

    value = str(query)

    value = value.replace(
        "\r\n",
        " ",
    )

    value = value.replace(
        "\n",
        " ",
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    value = value.strip()

    if not value:
        raise InvalidSearchQueryError("Search query cannot be empty.")

    if len(value) > MAX_QUERY_LENGTH:
        value = value[:MAX_QUERY_LENGTH].rstrip()

    if len(value) < MIN_QUERY_LENGTH:
        raise InvalidSearchQueryError("Search query is too short.")

    return value


# ============================================================================
# QUERY TOKENIZATION
# ============================================================================


def tokenize_query(
    query: str,
) -> tuple[str, ...]:
    """
    Create deterministic query tokens.

    Used later by keyword matching and explanation generation.
    """

    normalized = normalize_query(query)

    tokens = re.findall(
        r"[A-Za-z0-9₹$€£]+(?:[._+#-][A-Za-z0-9₹$€£]+)*",
        normalized.lower(),
    )

    unique: list[str] = []

    for token in tokens:

        if token not in unique:
            unique.append(token)

    return tuple(unique)


# ============================================================================
# SEARCH CONFIGURATION
# ============================================================================


def validate_weights(
    semantic_weight: float,
    keyword_weight: float,
) -> tuple[float, float]:
    """
    Validate and normalize search weights.
    """

    try:
        semantic = float(semantic_weight)

        keyword = float(keyword_weight)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise SearchConfigurationError("Search weights must be numeric.") from exc

    if semantic < 0 or keyword < 0:
        raise SearchConfigurationError("Search weights cannot be negative.")

    total = semantic + keyword

    if total <= 0:
        raise SearchConfigurationError(
            "At least one search weight must be greater than zero."
        )

    return (
        semantic / total,
        keyword / total,
    )


def normalize_top_k(
    top_k: int,
) -> int:
    """
    Normalize requested result count.
    """

    try:
        value = int(top_k)

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise SearchConfigurationError("top_k must be an integer.") from exc

    if value <= 0:
        raise SearchConfigurationError("top_k must be greater than zero.")

    return min(
        value,
        MAX_TOP_K,
    )


# ============================================================================
# CANDIDATE MERGING
# ============================================================================


def merge_candidates(
    semantic_candidates: Iterable[SearchCandidate],
    keyword_candidates: Iterable[SearchCandidate],
) -> dict[str, dict[str, Any]]:
    """
    Merge semantic and keyword candidates.

    A memory may appear in both retrieval systems.

    Instead of returning duplicates, we maintain one candidate record.

    Example:

        semantic:
            mem_123 -> 0.91

        keyword:
            mem_123 -> 0.80

    becomes:

        mem_123:
            semantic_score = 0.91
            keyword_score = 0.80
    """

    merged: dict[
        str,
        dict[str, Any],
    ] = {}

    for candidate in semantic_candidates:

        if not candidate.memory_id:
            continue

        entry = merged.setdefault(
            candidate.memory_id,
            {
                "memory_id": (candidate.memory_id),
                "semantic_score": 0.0,
                "keyword_score": 0.0,
                "metadata_score": 0.0,
                "metadata": {},
            },
        )

        entry["semantic_score"] = max(
            entry["semantic_score"],
            float(candidate.score),
        )

        entry["metadata"].update(candidate.metadata)

    for candidate in keyword_candidates:

        if not candidate.memory_id:
            continue

        entry = merged.setdefault(
            candidate.memory_id,
            {
                "memory_id": (candidate.memory_id),
                "semantic_score": 0.0,
                "keyword_score": 0.0,
                "metadata_score": 0.0,
                "metadata": {},
            },
        )

        entry["keyword_score"] = max(
            entry["keyword_score"],
            float(candidate.score),
        )

        entry["metadata"].update(candidate.metadata)

    return merged


# ============================================================================
# HYBRID SCORING
# ============================================================================


def calculate_hybrid_score(
    *,
    semantic_score: float,
    keyword_score: float,
    semantic_weight: float,
    keyword_weight: float,
) -> float:
    """
    Calculate normalized hybrid relevance score.

    Formula:

        score =
            semantic_score × semantic_weight
            +
            keyword_score × keyword_weight
    """

    semantic = max(
        0.0,
        min(
            float(semantic_score),
            1.0,
        ),
    )

    keyword = max(
        0.0,
        min(
            float(keyword_score),
            1.0,
        ),
    )

    score = semantic * semantic_weight + keyword * keyword_weight

    return round(
        max(
            0.0,
            min(
                score,
                1.0,
            ),
        ),
        6,
    )


# ============================================================================
# MEMORY MATCHING
# ============================================================================


def _first_non_empty(*values: Any) -> Any:
    """Return the first non-empty value from a sequence."""
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            value = value.strip()
            if value:
                return value
            continue
        if value != "":
            return value
    return None


def _extract_image_urls(
    memory: dict[str, Any],
) -> tuple[str | None, str | None, str | None, str | None]:
    """Collect the best available original and derivative image references."""
    metadata = (
        memory.get("metadata") if isinstance(memory.get("metadata"), dict) else {}
    )
    nested_memory = (
        memory.get("memory") if isinstance(memory.get("memory"), dict) else {}
    )
    nested_metadata = (
        nested_memory.get("metadata")
        if isinstance(nested_memory.get("metadata"), dict)
        else {}
    )

    image_url = _first_non_empty(
        memory.get("image_url"),
        nested_memory.get("image_url"),
        metadata.get("image_url"),
        nested_metadata.get("image_url"),
        memory.get("original_image_url"),
        nested_memory.get("original_image_url"),
        metadata.get("original_image_url"),
        nested_metadata.get("original_image_url"),
    )
    preview_url = _first_non_empty(
        memory.get("preview_url"),
        nested_memory.get("preview_url"),
        metadata.get("preview_url"),
        nested_metadata.get("preview_url"),
        image_url,
    )
    thumbnail_url = _first_non_empty(
        memory.get("thumbnail_url"),
        nested_memory.get("thumbnail_url"),
        metadata.get("thumbnail_url"),
        nested_metadata.get("thumbnail_url"),
        preview_url,
        memory.get("thumbnail_path"),
        nested_memory.get("thumbnail_path"),
        metadata.get("thumbnail_path"),
        nested_metadata.get("thumbnail_path"),
    )
    original_image_url = _first_non_empty(
        memory.get("original_image_url"),
        nested_memory.get("original_image_url"),
        metadata.get("original_image_url"),
        nested_metadata.get("original_image_url"),
        image_url,
    )
    return image_url, preview_url, thumbnail_url, original_image_url


def find_matching_terms(
    query_tokens: Iterable[str],
    memory: dict[str, Any],
) -> tuple[str, ...]:
    """
    Find query tokens occurring in searchable memory content.
    """

    searchable_fields = (
        "title",
        "summary",
        "ocr_text",
        "visual_description",
        "search_text",
        "category",
        "subcategory",
        "intent",
        "keywords",
        "products",
        "organizations",
        "locations",
        "people",
    )

    combined_parts: list[str] = []

    for field_name in searchable_fields:

        value = memory.get(field_name)

        if value is None:
            continue

        if isinstance(
            value,
            (list, tuple, set),
        ):
            combined_parts.extend(str(item) for item in value)

        else:
            combined_parts.append(str(value))

    searchable_text = " ".join(combined_parts).lower()

    matched: list[str] = []

    for token in query_tokens:

        if token in searchable_text:
            matched.append(token)

    return tuple(matched)


def find_matching_entities(
    query: str,
    memory: dict[str, Any],
) -> tuple[str, ...]:
    """
    Find entities mentioned by the query.
    """

    query_lower = query.lower()

    entity_fields = (
        "people",
        "organizations",
        "locations",
        "products",
        "entities",
    )

    matches: list[str] = []

    for field_name in entity_fields:

        values = memory.get(field_name)

        if not values:
            continue

        if isinstance(
            values,
            dict,
        ):
            values = values.values()

        if isinstance(
            values,
            str,
        ):
            values = [values]

        for value in values:

            if isinstance(
                value,
                dict,
            ):
                value = value.get(
                    "text",
                    "",
                )

            value = str(value).strip()

            if not value:
                continue

            if value.lower() in query_lower:
                matches.append(value)

    return tuple(dict.fromkeys(matches))


# ============================================================================
# EXPLANATION
# ============================================================================


def build_explanation(
    *,
    query: str,
    memory: dict[str, Any],
    semantic_score: float,
    keyword_score: float,
    metadata_score: float,
    final_score: float,
) -> SearchExplanation:
    """
    Generate an explainable search result.
    """

    tokens = tokenize_query(query)

    matched_terms = find_matching_terms(
        tokens,
        memory,
    )

    matched_entities = find_matching_entities(
        query,
        memory,
    )

    reasons: list[str] = []

    if semantic_score >= 0.80:
        reasons.append("strong semantic similarity")

    elif semantic_score >= 0.60:
        reasons.append("good semantic similarity")

    if matched_terms:
        reasons.append("keyword match: " + ", ".join(matched_terms[:5]))

    if matched_entities:
        reasons.append("entity match: " + ", ".join(matched_entities[:5]))

    if metadata_score >= 0.80:
        reasons.append("metadata strongly matches")

    if not reasons:
        reasons.append("the memory is relevant " "based on the combined search signals")

    reason = "Matched because of " + ", ".join(reasons) + "."

    return SearchExplanation(
        matched_terms=matched_terms,
        matched_entities=matched_entities,
        semantic_score=round(
            semantic_score,
            4,
        ),
        keyword_score=round(
            keyword_score,
            4,
        ),
        metadata_score=round(
            metadata_score,
            4,
        ),
        final_score=round(
            final_score,
            4,
        ),
        reason=reason,
    )


# ============================================================================
# SEARCH SERVICE
# ============================================================================


class SearchService:
    """
    Central MemoryOS search orchestration service.

    Dependencies are injected rather than imported globally.

    This allows us to test the search layer without requiring:

        - Gemini
        - FAISS
        - sentence-transformers
        - SQLite
    """

    def __init__(
        self,
        *,
        semantic_retriever: SemanticRetriever | None = None,
        keyword_retriever: KeywordRetriever | None = None,
        memory_provider: MemoryProvider | None = None,
        reranker: Reranker | None = None,
    ) -> None:

        self.semantic_retriever = semantic_retriever

        self.keyword_retriever = keyword_retriever

        self.memory_provider = memory_provider

        self.reranker = reranker

    # ------------------------------------------------------------------
    # Build SearchQuery
    # ------------------------------------------------------------------

    def create_query(
        self,
        text: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        category: str | None = None,
        intent: str | None = None,
        entity: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        semantic_weight: float = (DEFAULT_SEMANTIC_WEIGHT),
        keyword_weight: float = (DEFAULT_KEYWORD_WEIGHT),
    ) -> SearchQuery:
        """
        Validate and create a normalized SearchQuery.
        """

        normalized = normalize_query(text)

        normalized_top_k = normalize_top_k(top_k)

        (
            semantic,
            keyword,
        ) = validate_weights(
            semantic_weight,
            keyword_weight,
        )

        return SearchQuery(
            text=normalized,
            top_k=normalized_top_k,
            category=(category.strip() if category else None),
            intent=(intent.strip() if intent else None),
            entity=(entity.strip() if entity else None),
            date_from=date_from,
            date_to=date_to,
            semantic_weight=semantic,
            keyword_weight=keyword,
        )

    # ------------------------------------------------------------------
    # Candidate retrieval
    # ------------------------------------------------------------------

    def retrieve_candidates(
        self,
        query: SearchQuery,
    ) -> dict[
        str,
        dict[str, Any],
    ]:
        """
        Retrieve and merge semantic + keyword candidates.
        """

        semantic_candidates: Sequence[SearchCandidate] = ()

        keyword_candidates: Sequence[SearchCandidate] = ()

        retrieval_top_k = min(
            query.top_k * 3,
            MAX_TOP_K,
        )

        if self.semantic_retriever:

            try:
                semantic_candidates = self.semantic_retriever.search(
                    query.text,
                    top_k=retrieval_top_k,
                )

            except Exception as exc:

                logger.exception(
                    "Semantic retrieval failed: %s",
                    exc,
                )

        if self.keyword_retriever:

            try:
                keyword_candidates = self.keyword_retriever.search(
                    query.text,
                    top_k=retrieval_top_k,
                )

            except Exception as exc:

                logger.exception(
                    "Keyword retrieval failed: %s",
                    exc,
                )

        return merge_candidates(
            semantic_candidates,
            keyword_candidates,
        )

    # ------------------------------------------------------------------
    # Metadata filtering
    # ------------------------------------------------------------------

    def _passes_filters(
        self,
        memory: dict[str, Any],
        query: SearchQuery,
    ) -> bool:
        """
        Apply optional structured filters.
        """

        if query.category:

            memory_category = str(
                memory.get(
                    "category",
                    "",
                )
            ).lower()

            if memory_category != query.category.lower():
                return False

        if query.intent:

            memory_intent = str(
                memory.get(
                    "intent",
                    "",
                )
            ).lower()

            if memory_intent != query.intent.lower():
                return False

        if query.entity:

            entity_matches = find_matching_entities(
                query.entity,
                memory,
            )

            if not entity_matches:
                return False

        return True

    # ------------------------------------------------------------------
    # Build results
    # ------------------------------------------------------------------

    def build_results(
        self,
        *,
        query: SearchQuery,
        candidates: dict[
            str,
            dict[str, Any],
        ],
    ) -> list[SearchResult]:
        """
        Convert merged candidates into SearchResult objects.
        """

        results: list[SearchResult] = []

        for memory_id, candidate in candidates.items():

            memory: dict[str, Any] = {}

            if self.memory_provider:

                try:

                    loaded = self.memory_provider.get_memory(memory_id)

                    if loaded:
                        memory = loaded

                except Exception as exc:

                    logger.exception(
                        "Failed loading memory %s: %s",
                        memory_id,
                        exc,
                    )

            if not self._passes_filters(
                memory,
                query,
            ):
                continue

            semantic_score = float(
                candidate.get(
                    "semantic_score",
                    0.0,
                )
            )

            keyword_score = float(
                candidate.get(
                    "keyword_score",
                    0.0,
                )
            )

            metadata_score = float(
                candidate.get(
                    "metadata_score",
                    0.0,
                )
            )

            final_score = calculate_hybrid_score(
                semantic_score=semantic_score,
                keyword_score=keyword_score,
                semantic_weight=(query.semantic_weight),
                keyword_weight=(query.keyword_weight),
            )

            explanation = build_explanation(
                query=query.text,
                memory=memory,
                semantic_score=(semantic_score),
                keyword_score=(keyword_score),
                metadata_score=(metadata_score),
                final_score=(final_score),
            )

            image_url, preview_url, thumbnail_url, original_image_url = (
                _extract_image_urls(memory)
            )

            results.append(
                SearchResult(
                    memory_id=memory_id,
                    rank=0,
                    score=final_score,
                    title=str(
                        memory.get(
                            "title",
                            "Untitled Memory",
                        )
                    ),
                    summary=str(
                        memory.get(
                            "summary",
                            "",
                        )
                    ),
                    category=str(
                        memory.get(
                            "category",
                            "other",
                        )
                    ),
                    intent=str(
                        memory.get(
                            "intent",
                            "unknown",
                        )
                    ),
                    thumbnail_path=(memory.get("thumbnail_path")),
                    created_at=(memory.get("created_at")),
                    image_url=image_url,
                    preview_url=preview_url,
                    thumbnail_url=thumbnail_url,
                    original_image_url=original_image_url,
                    explanation=(explanation),
                    metadata=(memory),
                )
            )

        results.sort(
            key=lambda result: result.score,
            reverse=True,
        )

        ranked: list[SearchResult] = []

        for index, result in enumerate(
            results[: query.top_k],
            start=1,
        ):

            ranked.append(
                SearchResult(
                    memory_id=result.memory_id,
                    rank=index,
                    score=result.score,
                    title=result.title,
                    summary=result.summary,
                    category=result.category,
                    intent=result.intent,
                    thumbnail_path=(result.thumbnail_path),
                    created_at=(result.created_at),
                    image_url=result.image_url,
                    preview_url=result.preview_url,
                    thumbnail_url=result.thumbnail_url,
                    original_image_url=result.original_image_url,
                    explanation=(result.explanation),
                    metadata=result.metadata,
                )
            )

        return ranked

    # ------------------------------------------------------------------
    # Main search
    # ------------------------------------------------------------------

    def search(
        self,
        text: str,
        *,
        top_k: int = DEFAULT_TOP_K,
        category: str | None = None,
        intent: str | None = None,
        entity: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        semantic_weight: float = (DEFAULT_SEMANTIC_WEIGHT),
        keyword_weight: float = (DEFAULT_KEYWORD_WEIGHT),
    ) -> SearchResponse:
        """
        Execute a complete MemoryOS search.

        The service gracefully supports partial retrieval infrastructure.

        This is useful during development because:

            semantic retrieval can be added later
            keyword retrieval can be added later
            reranking can be added later

        without rewriting the API contract.
        """

        import time

        start = time.perf_counter()

        query = self.create_query(
            text,
            top_k=top_k,
            category=category,
            intent=intent,
            entity=entity,
            date_from=date_from,
            date_to=date_to,
            semantic_weight=(semantic_weight),
            keyword_weight=(keyword_weight),
        )

        candidates = self.retrieve_candidates(query)

        # Optional future reranking layer.
        if self.reranker and candidates:

            initial_candidates = [
                SearchCandidate(
                    memory_id=memory_id,
                    score=calculate_hybrid_score(
                        semantic_score=float(
                            candidate.get(
                                "semantic_score",
                                0.0,
                            )
                        ),
                        keyword_score=float(
                            candidate.get(
                                "keyword_score",
                                0.0,
                            )
                        ),
                        semantic_weight=(query.semantic_weight),
                        keyword_weight=(query.keyword_weight),
                    ),
                    source="hybrid",
                    metadata=candidate.get(
                        "metadata",
                        {},
                    ),
                )
                for memory_id, candidate in candidates.items()
            ]

            try:

                reranked = self.reranker.rerank(
                    query,
                    initial_candidates,
                )

                reranked_map: dict[
                    str,
                    dict[str, Any],
                ] = {}

                for candidate in reranked:

                    existing = candidates.get(
                        candidate.memory_id,
                        {},
                    )

                    existing["metadata_score"] = float(candidate.score)

                    reranked_map[candidate.memory_id] = existing

                if reranked_map:
                    candidates = reranked_map

            except Exception as exc:

                logger.exception(
                    "Reranking failed. " "Continuing with hybrid scores: %s",
                    exc,
                )

        results = self.build_results(
            query=query,
            candidates=candidates,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000

        return SearchResponse(
            query=text,
            normalized_query=query.text,
            results=tuple(results),
            total_results=len(results),
            took_ms=round(
                elapsed_ms,
                3,
            ),
            service_version=(SEARCH_SERVICE_VERSION),
            searched_at=datetime.now(timezone.utc).isoformat(),
            metadata={
                "semantic_weight": (query.semantic_weight),
                "keyword_weight": (query.keyword_weight),
                "candidate_count": (len(candidates)),
            },
        )


# ============================================================================
# DEFAULT SERVICE
# ============================================================================


search_service = SearchService()


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "SEARCH_SERVICE_VERSION",
    "DEFAULT_TOP_K",
    "MAX_TOP_K",
    "SearchError",
    "InvalidSearchQueryError",
    "SearchConfigurationError",
    "SearchExecutionError",
    "SearchQuery",
    "SearchCandidate",
    "SearchExplanation",
    "SearchResult",
    "SearchResponse",
    "SemanticRetriever",
    "KeywordRetriever",
    "MemoryProvider",
    "Reranker",
    "normalize_query",
    "tokenize_query",
    "validate_weights",
    "normalize_top_k",
    "merge_candidates",
    "calculate_hybrid_score",
    "find_matching_terms",
    "find_matching_entities",
    "build_explanation",
    "SearchService",
    "search_service",
]
