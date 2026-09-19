"""
MemoryOS multilingual query understanding.

Local:
- English
- Tamil
- Tanglish
- today / yesterday

Optional Gemini:
- normalizes other languages to concise English search meaning
- failure never blocks local search
"""

from __future__ import annotations

import json
import os
import re
import unicodedata

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache


@dataclass(frozen=True, slots=True)
class QueryUnderstanding:

    original_query: str
    search_text: str
    language: str

    date_from: str | None = None
    date_to: str | None = None

    used_ai: bool = False


LOCAL_TERMS = {

    # Tamil: yesterday
    "\u0ba8\u0bc7\u0bb1\u0bcd\u0bb1\u0bc1": "yesterday",
    "\u0ba8\u0bc7\u0ba4\u0bcd\u0ba4\u0bc1": "yesterday",
    "\u0ba8\u0bc7\u0bb1\u0bcd\u0bb1\u0bc8\u0baf": "yesterday",

    # Tamil: today
    "\u0b87\u0ba9\u0bcd\u0bb1\u0bc1": "today",
    "\u0b87\u0ba9\u0bcd\u0ba9\u0bc8\u0b95\u0bcd\u0b95\u0bc1": "today",

    # Tamil: outside
    "\u0bb5\u0bc6\u0bb3\u0bbf\u0baf\u0bc7": "outside",
    "\u0bb5\u0bc6\u0bb3\u0bbf\u0baf": "outside",

    # Tamil: food
    "\u0b9a\u0bbe\u0baa\u0bcd\u0baa\u0bbe\u0b9f\u0bc1": "food",
    "\u0b89\u0ba3\u0bb5\u0bc1": "food",

    # Tamil: bike / vehicle
    "\u0baa\u0bc8\u0b95\u0bcd": "bike",
    "\u0bb5\u0ba3\u0bcd\u0b9f\u0bbf": "vehicle",

    # Tamil: bill / receipt
    "\u0baa\u0bbf\u0bb2\u0bcd": "bill",
    "\u0bb0\u0b9a\u0bc0\u0ba4\u0bc1": "receipt",

    # Tamil: photo / photos
    "\u0baa\u0b9f\u0bae\u0bcd": "photo",
    "\u0baa\u0bcb\u0b9f\u0bcd\u0b9f\u0bcb": "photo",
    "\u0baa\u0bc1\u0b95\u0bc8\u0baa\u0bcd\u0baa\u0b9f\u0bae\u0bcd": "photo",
    "\u0baa\u0bc1\u0b95\u0bc8\u0baa\u0bcd\u0baa\u0b9f\u0b99\u0bcd\u0b95\u0bb3\u0bcd": "photos",

    # Tamil: taken
    "\u0b8e\u0b9f\u0bc1\u0ba4\u0bcd\u0ba4": "taken",
    "\u0b8e\u0b9f\u0bc1\u0ba4\u0bcd\u0ba4\u0ba4\u0bc1": "taken",

    # Tamil: light
    "\u0bb5\u0bbf\u0bb3\u0b95\u0bcd\u0b95\u0bc1": "light",
    "\u0bb2\u0bc8\u0b9f\u0bcd": "light",

    # Tanglish
    "nethu": "yesterday",
    "netru": "yesterday",
    "innaiku": "today",
    "inniku": "today",
    "velila": "outside",
    "veliye": "outside",
    "saapadu": "food",
    "sapadu": "food",
    "vandi": "vehicle",
    "edutha": "taken",
    "eduthathu": "taken",
    "rasidhu": "receipt",

    # Common conversational fillers
    "naan": "",
    "nan": "",
    "enoda": "",
}


def clean(value: str) -> str:

    value = unicodedata.normalize(
        "NFKC",
        str(value or ""),
    )

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def local_translate(
    value: str,
) -> str:

    result = value

    # Longest phrases first.
    for source in sorted(
        LOCAL_TERMS,
        key=len,
        reverse=True,
    ):

        result = re.sub(
            re.escape(source),
            LOCAL_TERMS[source],
            result,
            flags=re.IGNORECASE,
        )

    return clean(result)


def date_window(
    value: str,
):

    lowered = value.casefold()

    today = date.today()

    if re.search(
        r"\byesterday\b",
        lowered,
    ):

        target = (
            today
            - timedelta(days=1)
        )

    elif re.search(
        r"\btoday\b",
        lowered,
    ):

        target = today

    else:

        return None, None

    start = datetime.combine(
        target,
        datetime.min.time(),
    )

    end = (
        start
        + timedelta(days=1)
    )

    return (
        start.isoformat(),
        end.isoformat(),
    )


def gemini_normalize(
    query: str,
):

    api_key = (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or ""
    ).strip()

    if not api_key:
        return None

    enabled = (
        os.getenv(
            "MEMORYOS_QUERY_AI_ENABLED",
            "true",
        ).strip().lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        }
    )

    if not enabled:
        return None

    try:

        from google import genai
        from google.genai import types

        client = genai.Client(
            api_key=api_key
        )

        response = (
            client.models.generate_content(
                model=os.getenv(
                    "MEMORYOS_QUERY_MODEL",
                    "gemini-2.5-flash",
                ),

                contents=(
                    "Convert this visual-memory search request "
                    "into concise English visual search meaning. "
                    "Preserve names, prices, amounts, brands and "
                    "technical terms. Do not answer the user. "
                    "Return JSON only with keys search_text and language. "
                    f"Query: {query}"
                ),

                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=160,
                    response_mime_type="application/json",
                ),
            )
        )

        data = json.loads(
            str(
                response.text
                or "{}"
            )
        )

        translated = clean(
            data.get(
                "search_text"
            )
        )

        language = clean(
            data.get(
                "language"
            )
            or "unknown"
        )

        if translated:

            return (
                translated,
                language,
            )

    except Exception:
        return None

    return None


@lru_cache(maxsize=512)
def understand_query(
    query: str,
) -> QueryUnderstanding:

    original = clean(query)

    if not original:

        return QueryUnderstanding(
            "",
            "",
            "unknown",
        )

    locally_translated = (
        local_translate(
            original
        )
    )

    search_text = (
        locally_translated
    )

    language = (
        "mixed"
        if locally_translated
        != original
        else "unknown"
    )

    used_ai = False

    # Any remaining non-ASCII language can optionally be normalized
    # by Gemini. Search still works if Gemini is unavailable.
    if any(
        ord(character) > 127
        for character
        in search_text
    ):

        ai = gemini_normalize(
            original
        )

        if ai:

            search_text, language = ai

            used_ai = True

    date_from, date_to = (
        date_window(
            search_text
        )
    )

    semantic_text = re.sub(
        r"\b(today|yesterday)\b",
        " ",
        search_text,
        flags=re.IGNORECASE,
    )

    semantic_text = clean(
        semantic_text
    )

    if not semantic_text:

        semantic_text = "photo"

    return QueryUnderstanding(
        original_query=original,
        search_text=semantic_text,
        language=language,
        date_from=date_from,
        date_to=date_to,
        used_ai=used_ai,
    )
