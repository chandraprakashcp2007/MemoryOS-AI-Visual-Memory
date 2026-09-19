"""
MemoryOS - Semantic Screenshot Search API
=========================================

Problem statement:

    "I have 4,000 screenshots. Find the one I mean."

Example queries:

    Find my Python error
    Find the college timetable
    Show me that chicken recipe
    Find the screenshot with the address
    Find the receipt I saved
    Show screenshots about FastAPI

The API searches the indexed screenshot memories using:
    - OCR text
    - summary
    - category
    - entities
    - keywords
    - semantic similarity when available

This is intentionally simple for the hackathon MVP.
"""

from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MEMORY_FILE = PROJECT_ROOT / "data" / "memory_index" / "memories.json"
FEEDBACK_FILE = PROJECT_ROOT / "data" / "search_feedback.json"


# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger("memoryos.search")


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    prefix="/search",
    tags=["Search"],
)


# ============================================================================
# REQUEST / RESPONSE MODELS
# ============================================================================


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description=("Natural language search query."),
    )

    limit: int = Field(
        default=10,
        ge=1,
        le=50,
    )


class SearchResult(BaseModel):
    memory_id: str

    filename: str

    path: str

    # These are API routes, not filesystem paths.  Keeping both references in
    # the result prevents the UI from having to infer an image from unrelated
    # metadata and lets a thumbnail always resolve back to the original file.
    original_image_url: str

    thumbnail_url: str

    score: float

    category: str = ""

    summary: str = ""

    ocr_text: str = ""

    entities: List[str] = []

    keywords: List[str] = []

    why_matched: List[str] = []

    # These fields already come from image ingestion. Keeping them in the
    # search payload lets clients separate visual evidence from OCR.
    visual_description: str = ""
    visual_concepts: List[str] = []
    food_concepts: List[str] = []
    objects: List[str] = []
    general_concepts: List[str] = []
    vision_analysis: Dict[str, Any] = {}


class SearchResponse(BaseModel):
    query: str

    total: int

    results: List[SearchResult]


class SearchFeedbackRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    memory_id: str = Field(..., min_length=1, max_length=128)
    relevant: bool


# ============================================================================
# MEMORY LOADER
# ============================================================================


def load_memories() -> Dict[str, Dict[str, Any]]:
    """
    Load indexed screenshot memories.
    """

    if not MEMORY_FILE.exists():
        return {}

    try:

        with MEMORY_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if isinstance(
            data,
            dict,
        ):
            return data

    except Exception as exc:

        logger.error(
            "Could not load memories: %s",
            exc,
        )

    return {}


# ============================================================================
# TEXT NORMALIZATION
# ============================================================================


def normalize(
    value: Any,
) -> str:

    if value is None:
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value).lower(),
    ).strip()


def tokenize(
    value: str,
) -> List[str]:

    return re.findall(
        r"[a-zA-Z0-9₹$@._-]+",
        normalize(value),
    )


# ============================================================================
# QUERY UNDERSTANDING
# ============================================================================


def meaningful_tokens(
    query: str,
) -> List[str]:

    stop_words = {
        "find",
        "show",
        "me",
        "the",
        "a",
        "an",
        "my",
        "that",
        "this",
        "one",
        "screenshot",
        "screenshots",
        "image",
        "images",
        "picture",
        "pictures",
        "where",
        "i",
        "saw",
        "saved",
        "with",
        "about",
        "containing",
        "which",
        "was",
        "is",
        "for",
        "of",
        "on",
        "in",
        "to",
        "from",
    }

    return [
        token
        for token in tokenize(query)
        if (len(token) >= 2 and token not in stop_words)
    ]


def _token_overlap(query_tokens: set[str], text: Any) -> tuple[float, set[str]]:
    """Return exact-or-close token overlap without query-specific rules."""
    document_tokens = set(tokenize(str(text or "")))
    matches = query_tokens & document_tokens
    if not matches:
        # Handles small typos and singular/plural forms while keeping a high
        # threshold so unrelated short words do not become matches.
        for query_token in query_tokens:
            if len(query_token) < 4:
                continue
            if any(SequenceMatcher(None, query_token, candidate).ratio() >= 0.84 for candidate in document_tokens):
                matches.add(query_token)
    return (len(matches) / len(query_tokens) if query_tokens else 0.0), matches


def _visual_text(memory: Dict[str, Any]) -> str:
    vision = memory.get("vision_analysis")
    vision = vision if isinstance(vision, dict) else {}
    concept_fields = (
        "visual_concepts", "objects", "actions", "relationships",
        "food_concepts", "product_concepts", "document_concepts",
        "technology_concepts", "coding_concepts", "ui_concepts",
        "location_concepts", "activity_concepts", "topic_concepts",
        "general_concepts", "scene", "environment",
    )
    values = [
        memory.get("visual_description", ""),
        vision.get("visual_description", ""),
        vision.get("summary", ""),
        vision.get("subcategory", ""),
        vision.get("products", []),
        vision.get("keywords", []),
        *(memory.get(field, []) for field in concept_fields),
        *(vision.get(field, []) for field in concept_fields),
    ]
    return " ".join(" ".join(map(str, value)) if isinstance(value, list) else str(value or "") for value in values)


def _load_feedback() -> dict[str, dict[str, int]]:
    try:
        data = json.loads(FEEDBACK_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _feedback_adjustment(
    query: str,
    memory_id: str,
) -> float:

    value = (
        _load_feedback()
        .get(
            normalize(query),
            {},
        )
        .get(
            memory_id,
            0,
        )
    )

    if value > 0:
        return 14.0

    if value < 0:
        return -60.0

    return 0.0



# ============================================================================
# CATEGORY UNDERSTANDING
# ============================================================================


def semantic_scores(query: str, limit: int) -> dict[str, float]:
    """Retrieve image/OCR meaning matches from the persistent FAISS index."""
    try:
        from backend.services.semantic_embedding_service import get_embedding_service
        from backend.services.vector_store import create_vector_store

        query_vector = get_embedding_service().embed_text(query)
        store = create_vector_store()
        return {
            result.memory_id: max(0.0, min(1.0, result.score))
            for result in store.search(query_vector, top_k=max(limit, 50), min_score=0.15)
        }
    except Exception as exc:
        # Lexical OCR search remains available if the optional local model is
        # unavailable; never pretend this is a visual result.
        logger.warning("Semantic retrieval unavailable: %s", exc)
        return {}



def visual_scores(
    query: str,
    limit: int,
) -> dict[str, float]:
    """
    Retrieve memories using the actual image pixels.

    Query text is embedded into CLIP's shared image/text space and searched
    against the separate 512-dimensional visual FAISS index.
    """

    try:

        from backend.services.image_embedding_service import (
            get_image_embedding_service,
        )

        from backend.services.visual_vector_store import (
            get_visual_vector_store,
        )

        service = get_image_embedding_service()

        store = get_visual_vector_store()

        if store.count <= 0:
            return {}

        query_vector = service.embed_text(
            query
        )

        results = store.search(
            query_vector,
            top_k=max(
                int(limit),
                80,
            ),
        )

        scores: dict[str, float] = {}

        for result in results:

            memory_id = str(
                result.get(
                    "memory_id",
                    "",
                )
            ).strip()

            if not memory_id:
                continue

            raw_score = float(
                result.get(
                    "score",
                    0.0,
                )
            )

            # CLIP cosine similarity is not a calibrated probability.
            # Preserve the useful positive similarity range and perform
            # weighting later inside the hybrid ranker.
            scores[memory_id] = max(
                0.0,
                min(
                    1.0,
                    raw_score,
                ),
            )

        return scores

    except Exception as exc:

        logger.warning(
            "Visual CLIP retrieval unavailable: %s",
            exc,
        )

        # OCR/MiniLM search remains available.
        return {}



# ============================================================================
# SCORING
# ============================================================================


def score_memory(
    query: str,
    memory: Dict[str, Any],
    semantic_score: float = 0.0,
    visual_score: float = 0.0,
) -> tuple[float, List[str]]:

    query_normalized = normalize(query)

    query_tokens = set(meaningful_tokens(query))

    if not query_tokens:
        return 0.0, []

    score = 0.0

    reasons: List[str] = []

    # --------------------------------------------------------------
    # TRUE VISUAL IMAGE SIMILARITY ? CLIP
    # --------------------------------------------------------------

    if visual_score > 0:

        # CLIP similarities for correct open-vocabulary matches commonly
        # occupy a much smaller numeric range than normalized text scores.
        # Scale them without pretending they are probabilities.
        score += visual_score * 70

        reasons.append(
            "Matched visually using image content"
        )

    if semantic_score > 0:
        score += semantic_score * 45
        reasons.append("Matched by semantic meaning")

    # ------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------

    ocr = normalize(
        memory.get(
            "ocr_text",
            "",
        )
    )

    ocr_ratio, ocr_matches = _token_overlap(query_tokens, ocr)

    if ocr_matches:

        score += ocr_ratio * 30
        reasons.append("Matched by text: " + ", ".join(sorted(ocr_matches)[:6]))

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    summary = normalize(
        memory.get(
            "summary",
            "",
        )
    )

    summary_ratio, summary_matches = _token_overlap(query_tokens, summary)

    if summary_matches:

        score += summary_ratio * 24
        reasons.append("Matched by memory description")

    visual_ratio, visual_matches = _token_overlap(query_tokens, _visual_text(memory))
    if visual_matches:
        score += visual_ratio * 32
        reasons.append("Strong visual concept match: " + ", ".join(sorted(visual_matches)[:6]))

    # ------------------------------------------------------------------
    # ENTITIES
    # ------------------------------------------------------------------

    entities = memory.get(
        "entities",
        [],
    )

    entity_text = normalize(
        " ".join(
            map(
                str,
                entities,
            )
        )
    )

    entity_ratio, entity_matches = _token_overlap(query_tokens, entity_text)

    if entity_matches:

        score += entity_ratio * 20

        reasons.append("Entity matched: " + ", ".join(sorted(entity_matches)[:6]))

    # ------------------------------------------------------------------
    # KEYWORDS
    # ------------------------------------------------------------------

    keywords = memory.get(
        "keywords",
        [],
    )

    keyword_text = normalize(
        " ".join(
            map(
                str,
                keywords,
            )
        )
    )

    keyword_ratio, keyword_matches = _token_overlap(query_tokens, keyword_text)

    if keyword_matches:

        score += keyword_ratio * 26

        reasons.append("Keyword matched: " + ", ".join(sorted(keyword_matches)[:6]))

    # ------------------------------------------------------------------
    # FILENAME
    # ------------------------------------------------------------------

    filename = normalize(
        memory.get(
            "filename",
            "",
        )
    )

    filename_ratio, filename_matches = _token_overlap(query_tokens, filename)

    if filename_matches:

        score += filename_ratio * 12

        reasons.append("Filename matched")

    # ------------------------------------------------------------------
    # EXACT PHRASE
    # ------------------------------------------------------------------

    search_document = normalize(
        memory.get(
            "search_document",
            "",
        )
    )

    if len(query_normalized) >= 4 and query_normalized in search_document:

        score += 40

        reasons.append("Exact phrase matched")

    category_ratio, category_matches = _token_overlap(query_tokens, memory.get("category", ""))
    if category_matches:
        score += category_ratio * 18
        reasons.append("Category matched")

    adjustment = _feedback_adjustment(query, str(memory.get("memory_id") or memory.get("id") or ""))
    if adjustment:
        score += adjustment
        reasons.append("Adjusted from your relevance feedback")

    # ------------------------------------------------------------------
    # NORMALIZE SCORE
    # ------------------------------------------------------------------

    score = max(0.0, min(100.0, score))

    # Remove duplicate reasons.
    reasons = list(dict.fromkeys(reasons))

    return score, reasons


# ============================================================================
# RESULT BUILDER
# ============================================================================


def build_result(
    memory_id: str,
    memory: Dict[str, Any],
    score: float,
    reasons: List[str],
) -> SearchResult:

    image_url = memory.get("image_url") or f"/upload/file/{memory_id}"
    preview_url = memory.get("preview_url") or image_url
    thumbnail_url = (
        memory.get("thumbnail_url") or f"/upload/file/{memory_id}?thumbnail=true"
    )
    original_image_url = memory.get("original_image_url") or image_url

    return SearchResult(
        memory_id=memory_id,
        filename=memory.get(
            "filename",
            "",
        ),
        path=memory.get(
            "path",
            "",
        ),
        image_url=image_url,
        preview_url=preview_url,
        thumbnail_url=thumbnail_url,
        original_image_url=original_image_url,
        score=round(
            score,
            2,
        ),
        category=memory.get(
            "category",
            "",
        ),
        summary=memory.get(
            "summary",
            "",
        ),
        ocr_text=memory.get(
            "ocr_text",
            "",
        )[:2000],
        entities=[
            str(item)
            for item in memory.get(
                "entities",
                [],
            )
        ][:20],
        keywords=[
            str(item)
            for item in memory.get(
                "keywords",
                [],
            )
        ][:30],
        why_matched=reasons,
        visual_description=str(memory.get("visual_description", "") or ""),
        visual_concepts=[str(item) for item in memory.get("visual_concepts", [])][:20],
        food_concepts=[str(item) for item in memory.get("food_concepts", [])][:20],
        objects=[str(item) for item in memory.get("objects", [])][:20],
        general_concepts=[str(item) for item in memory.get("general_concepts", [])][:20],
        vision_analysis=memory.get("vision_analysis") if isinstance(memory.get("vision_analysis"), dict) else {},
    )



def _memory_in_date_window(
    memory: Dict[str, Any],
    date_from: str | None,
    date_to: str | None,
) -> bool:

    if not date_from or not date_to:
        return True

    try:
        from datetime import datetime

        start = datetime.fromisoformat(date_from)
        end = datetime.fromisoformat(date_to)

    except Exception:
        return True

    candidates = (
        memory.get("captured_at"),
        memory.get("created_at"),
        memory.get("indexed_at"),
    )

    for value in candidates:

        if not value:
            continue

        try:

            parsed = datetime.fromisoformat(
                str(value).replace(
                    "Z",
                    "+00:00",
                )
            )

            if parsed.tzinfo is not None:
                parsed = parsed.replace(
                    tzinfo=None
                )

            if start <= parsed < end:
                return True

        except Exception:
            continue

    return False




def _non_ocr_text(
    memory: Dict[str, Any],
) -> str:

    values = [
        memory.get("summary", ""),
        memory.get("category", ""),
        memory.get("filename", ""),
        memory.get("visual_description", ""),
        _visual_text(memory),
        memory.get("entities", []),
        memory.get("keywords", []),
    ]

    parts = []

    for value in values:

        if isinstance(
            value,
            list,
        ):

            parts.extend(
                str(item)
                for item
                in value
            )

        else:

            parts.append(
                str(
                    value
                    or ""
                )
            )

    return " ".join(
        parts
    )


def _reliable_match(
    query: str,
    memory: Dict[str, Any],
    *,
    semantic_score: float,
    visual_score: float,
    visual_floor: float,
    semantic_floor: float,
) -> tuple[bool, list[str]]:

    query_tokens = set(
        meaningful_tokens(
            query
        )
    )

    if not query_tokens:

        return False, []

    non_ocr_text = normalize(
        _non_ocr_text(
            memory
        )
    )

    ocr_text = normalize(
        memory.get(
            "ocr_text",
            "",
        )
    )

    non_ocr_tokens = set(
        tokenize(
            non_ocr_text
        )
    )

    ocr_tokens = set(
        tokenize(
            ocr_text
        )
    )

    non_ocr_matches = (
        query_tokens
        & non_ocr_tokens
    )

    ocr_matches = (
        query_tokens
        & ocr_tokens
    )

    query_normalized = normalize(
        query
    )

    exact_non_ocr = (
        len(
            query_normalized
        ) >= 3
        and query_normalized
        in non_ocr_text
    )

    exact_ocr = (
        len(
            query_normalized
        ) >= 3
        and query_normalized
        in ocr_text
    )

    visual_ok = (
        visual_score
        >= visual_floor
    )

    semantic_ok = (
        semantic_score
        >= semantic_floor
    )

    evidence = []

    # ----------------------------------------------------------
    # 1. Visual evidence
    # ----------------------------------------------------------

    if visual_ok:

        evidence.append(
            "visual"
        )

    # ----------------------------------------------------------
    # 2. AI / stored visual metadata evidence
    # ----------------------------------------------------------

    if (
        non_ocr_matches
        or exact_non_ocr
    ):

        evidence.append(
            "concept"
        )

    # ----------------------------------------------------------
    # 3. Semantic evidence
    #
    # Semantic alone is useful for descriptive multi-word
    # queries, but a one-word query should not return dozens
    # of unrelated memories merely because embedding similarity
    # is weakly positive.
    # ----------------------------------------------------------

    if semantic_ok:

        if (
            len(
                query_tokens
            ) >= 2
            or non_ocr_matches
            or visual_ok
        ):

            evidence.append(
                "semantic"
            )

    # ----------------------------------------------------------
    # 4. OCR evidence
    #
    # For one generic word, OCR-only is deliberately NOT enough.
    # That prevents a corrupted / stale OCR index from producing
    # rows of unrelated screenshots.
    #
    # Multi-word phrases, amounts, long IDs, emails etc. can use
    # OCR directly.
    # ----------------------------------------------------------

    if len(
        query_tokens
    ) >= 2:

        ratio = (
            len(
                ocr_matches
            )
            / max(
                1,
                len(
                    query_tokens
                ),
            )
        )

        if (
            ratio >= 0.5
            or exact_ocr
        ):

            evidence.append(
                "ocr"
            )

    else:

        token = next(
            iter(
                query_tokens
            )
        )

        structured = (
            bool(
                re.search(
                    r"[0-9?$@]",
                    query,
                )
            )
            or "." in token
            or len(
                token
            ) >= 10
        )

        if (
            structured
            and token
            in ocr_tokens
        ):

            evidence.append(
                "ocr"
            )

    return (
        bool(
            evidence
        ),
        list(
            dict.fromkeys(
                evidence
            )
        ),
    )



# ============================================================================
# SEARCH ENGINE
# ============================================================================


def search_memories(
    query: str,
    limit: int = 10,
) -> List[SearchResult]:

    memories = load_memories()

    if not memories:
        return []

    try:

        from backend.services.query_understanding_service import (
            understand_query,
        )

        understanding = (
            understand_query(
                query
            )
        )

        retrieval_query = (
            understanding.search_text
            or query
        )

    except Exception:

        understanding = None
        retrieval_query = query

    # ----------------------------------------------------------
    # Candidate retrieval
    # ----------------------------------------------------------

    semantic = semantic_scores(
        retrieval_query,
        max(
            limit,
            50,
        ),
    )

    visual = visual_scores(
        retrieval_query,
        max(
            limit,
            80,
        ),
    )

    # ----------------------------------------------------------
    # Query-relative thresholds
    #
    # CLIP cosine is NOT probability.
    # We use both an absolute floor and closeness to the best
    # candidate for this specific query.
    # ----------------------------------------------------------

    if visual:

        top_visual = max(
            visual.values()
        )

        visual_floor = max(
            0.235,
            top_visual - 0.055,
        )

    else:

        visual_floor = 999.0

    if semantic:

        top_semantic = max(
            semantic.values()
        )

        semantic_floor = max(
            0.50,
            top_semantic - 0.10,
        )

    else:

        semantic_floor = 999.0

    ranked = []

    for (
        memory_id,
        memory,
    ) in memories.items():

        # ------------------------------------------------------
        # Time filtering
        # ------------------------------------------------------

        if (
            understanding is not None
            and not _memory_in_date_window(
                memory,
                understanding.date_from,
                understanding.date_to,
            )
        ):

            continue

        raw_visual = float(
            visual.get(
                memory_id,
                0.0,
            )
        )

        raw_semantic = float(
            semantic.get(
                memory_id,
                0.0,
            )
        )

        # ------------------------------------------------------
        # User feedback = hard learning signal
        # ------------------------------------------------------

        feedback = _feedback_adjustment(
            retrieval_query,
            memory_id,
        )

        if feedback <= -40:

            # Exact query + Not relevant:
            # do not keep showing the same wrong result.
            continue

        # ------------------------------------------------------
        # Universal evidence gate
        # ------------------------------------------------------

        accepted, evidence = (
            _reliable_match(
                retrieval_query,
                memory,
                semantic_score=
                    raw_semantic,
                visual_score=
                    raw_visual,
                visual_floor=
                    visual_floor,
                semantic_floor=
                    semantic_floor,
            )
        )

        if not accepted:

            continue

        # Weak candidate similarities do not enter score_memory.
        strong_visual = (
            raw_visual
            if raw_visual
            >= visual_floor
            else 0.0
        )

        strong_semantic = (
            raw_semantic
            if raw_semantic
            >= semantic_floor
            else 0.0
        )

        score, reasons = (
            score_memory(
                retrieval_query,
                memory,
                strong_semantic,
                strong_visual,
            )
        )

        # ------------------------------------------------------
        # Evidence quality bonuses
        # ------------------------------------------------------

        if "visual" in evidence:

            score += 12.0

        if "concept" in evidence:

            score += 14.0

        if "semantic" in evidence:

            score += 8.0

        if "ocr" in evidence:

            score += 12.0

        if feedback > 0:

            score += feedback

        # ------------------------------------------------------
        # Date evidence
        # ------------------------------------------------------

        if (
            understanding is not None
            and understanding.date_from
            and understanding.date_to
        ):

            score += 20.0

            reasons.append(
                "Matched by capture date"
            )

        score = max(
            0.0,
            min(
                100.0,
                score,
            ),
        )

        if score <= 0:

            continue

        reasons = list(
            dict.fromkeys(
                reasons
            )
        )

        ranked.append(
            (
                score,
                memory_id,
                memory,
                reasons,
            )
        )

    ranked.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    return [
        build_result(
            memory_id=memory_id,
            memory=memory,
            score=score,
            reasons=reasons,
        )
        for (
            score,
            memory_id,
            memory,
            reasons,
        )
        in ranked[
            :limit
        ]
    ]


def related_memories(memory_id: str, limit: int = 6) -> List[SearchResult]:
    memories = load_memories()
    source = memories.get(memory_id)
    if not source:
        return []
    query = str(source.get("search_document") or _visual_text(source) or source.get("summary") or "").strip()
    if not query:
        return []
    return [result for result in search_memories(query, limit=limit + 1) if result.memory_id != memory_id][:limit]


def search_suggestions(prefix: str, limit: int = 6) -> list[str]:
    needle = normalize(prefix)
    if not needle:
        return []
    counts: dict[str, int] = {}
    for memory in load_memories().values():
        candidates = [memory.get("category", ""), *(memory.get("keywords") or [])]
        vision = memory.get("vision_analysis")
        if isinstance(vision, dict):
            candidates.extend(vision.get("products") or [])
            candidates.extend(vision.get("keywords") or [])
        for value in candidates:
            text = str(value).strip()
            if text and normalize(text).startswith(needle):
                counts[text] = counts.get(text, 0) + 1
    return [value for value, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0].lower()))[:limit]]


# ============================================================================
# API ENDPOINT
# ============================================================================


@router.post(
    "",
    response_model=SearchResponse,
)
async def search(
    request: SearchRequest,
) -> SearchResponse:
    """
    Search screenshot memories.

    Example:

        POST /search

        {
            "query": "find my Python error",
            "limit": 5
        }
    """

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Search query cannot be empty.",
        )

    logger.info(
        "Memory search: %s",
        query,
    )

    results = search_memories(
        query=query,
        limit=request.limit,
    )

    return SearchResponse(
        query=query,
        total=len(results),
        results=results,
    )


# ============================================================================
# SIMPLE GET SEARCH
# ============================================================================


@router.get(
    "",
)
async def search_get(
    q: str,
    limit: int = 10,
):
    """
    Browser-friendly search.

    Example:

        /search?q=python%20error
    """

    if not q.strip():
        raise HTTPException(
            status_code=400,
            detail="Query is required.",
        )

    results = search_memories(
        query=q,
        limit=max(
            1,
            min(
                limit,
                50,
            ),
        ),
    )

    return {
        "query": q,
        "total": len(results),
        "results": [result.model_dump() for result in results],
    }


@router.get("/suggestions")
async def suggestions(q: str, limit: int = 6):
    return {"suggestions": search_suggestions(q, max(1, min(limit, 12)))}


@router.get("/related/{memory_id}")
async def related(memory_id: str, limit: int = 6):
    return {"results": [result.model_dump() for result in related_memories(memory_id, max(1, min(limit, 12)))]}


@router.post("/feedback")
async def feedback(request: SearchFeedbackRequest):
    """Store only a query/result relevance signal; no personal profile data."""
    feedback_data = _load_feedback()
    key = normalize(request.query)
    bucket = feedback_data.setdefault(key, {})
    bucket[request.memory_id] = 1 if request.relevant else -1
    FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary = FEEDBACK_FILE.with_suffix(".tmp")
    temporary.write_text(json.dumps(feedback_data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(FEEDBACK_FILE)
    return {"success": True}


# ============================================================================
# HEALTH / INDEX INFORMATION
# ============================================================================


@router.get(
    "/status",
)
async def search_status():
    """
    Show how many memories are currently searchable.
    """

    memories = load_memories()

    categories = {}

    for memory in memories.values():

        category = memory.get(
            "category",
            "unclassified",
        )

        categories[category] = (
            categories.get(
                category,
                0,
            )
            + 1
        )

    return {
        "status": "ready",
        "searchable_memories": len(memories),
        "categories": categories,
        "memory_file": str(MEMORY_FILE),
    }
