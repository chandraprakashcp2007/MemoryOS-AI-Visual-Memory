"""
MemoryOS multilingual query understanding.

Local:
- English
- Tamil / Tanglish
- major Indian scripts and common transliterations
- common search concepts in selected world languages
- today / yesterday

Optional Gemini:
- normalizes likely non-English queries to concise English search meaning
- preserves the original language label
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


# MEMORYOS_MULTILINGUAL_QUERY_V2
LOCAL_TERMS.update({
    "आज": "today", "बाहर": "outside", "खाना": "food", "तस्वीर": "photo",
    "फ़ोटो": "photo", "फोटो": "photo", "बाइक": "bike", "रसीद": "receipt",

    "నిన్న": "yesterday", "ఈరోజు": "today", "బయట": "outside", "ఆహారం": "food",
    "బైక్": "bike", "ఫోటో": "photo", "బిల్లు": "bill", "రసీదు": "receipt",

    "ഇന്നലെ": "yesterday", "ഇന്ന്": "today", "പുറത്ത്": "outside", "ഭക്ഷണം": "food",
    "ബൈക്ക്": "bike", "ഫോട്ടോ": "photo", "ബിൽ": "bill", "രസീത്": "receipt",

    "ನಿನ್ನೆ": "yesterday", "ಇಂದು": "today", "ಹೊರಗೆ": "outside", "ಊಟ": "food",
    "ಬೈಕ್": "bike", "ಫೋಟೋ": "photo", "ಬಿಲ್": "bill", "ರಸೀದಿ": "receipt",

    "গতকাল": "yesterday", "আজ": "today", "বাইরে": "outside", "খাবার": "food",
    "বাইক": "bike", "ছবি": "photo", "বিল": "bill", "রসিদ": "receipt",

    "आज": "today", "बाहेर": "outside", "जेवण": "food", "बाईक": "bike", "पावती": "receipt",

    "ગઈકાલે": "yesterday", "આજે": "today", "બહાર": "outside", "ખોરાક": "food",
    "બાઈક": "bike", "ફોટો": "photo", "બિલ": "bill", "રસીદ": "receipt",

    "ਅੱਜ": "today", "ਬਾਹਰ": "outside", "ਖਾਣਾ": "food", "ਬਾਈਕ": "bike",
    "ਫੋਟੋ": "photo", "ਬਿੱਲ": "bill", "ਰਸੀਦ": "receipt",

    "آج": "today", "باہر": "outside", "کھانا": "food", "بائیک": "bike",
    "تصویر": "photo", "بل": "bill", "رسید": "receipt",

    "ayer": "yesterday", "hoy": "today", "afuera": "outside", "comida": "food",
    "bicicleta": "bike", "moto": "bike", "foto": "photo", "factura": "bill", "recibo": "receipt",

    "hier": "yesterday", "aujourd'hui": "today", "dehors": "outside", "nourriture": "food",
    "vélo": "bike", "photo": "photo", "facture": "bill", "reçu": "receipt",

    "gestern": "yesterday", "heute": "today", "draußen": "outside", "essen": "food",
    "fahrrad": "bike", "motorrad": "bike", "rechnung": "bill", "beleg": "receipt",

    "ontem": "yesterday", "hoje": "today", "fora": "outside", "fatura": "bill",

    "أمس": "yesterday", "اليوم": "today", "خارج": "outside", "طعام": "food",
    "دراجة": "bike", "صورة": "photo", "فاتورة": "bill", "إيصال": "receipt",

    "昨日": "yesterday", "今日": "today", "外": "outside", "食べ物": "food",
    "バイク": "bike", "自転車": "bike", "写真": "photo", "請求書": "bill", "レシート": "receipt",

    "어제": "yesterday", "오늘": "today", "밖": "outside", "음식": "food",
    "오토바이": "bike", "자전거": "bike", "사진": "photo", "청구서": "bill", "영수증": "receipt",

    "昨天": "yesterday", "今天": "today", "外面": "outside", "食物": "food",
    "摩托车": "bike", "自行车": "bike", "照片": "photo", "发票": "bill", "收据": "receipt",

    "aaj": "today", "bahar": "outside", "tasveer": "photo", "rasid": "receipt",
    "ninna": "yesterday", "ivala": "today", "bayata": "outside", "tindi": "food",
    "billu": "bill", "raseedu": "receipt", "innale": "yesterday", "innu": "today",
    "purathu": "outside", "bhakshanam": "food", "ninne": "yesterday", "horage": "outside",
    "oota": "food", "rasidi": "receipt", "gotokal": "yesterday", "ajke": "today",
    "baire": "outside", "baher": "outside", "jevan": "food", "pavati": "receipt",
    "gaikale": "yesterday", "aaje": "today",
})

ROMANIZED_AI_HINTS = {
    "mujhe", "dikhao", "wala", "wali", "mere", "mera",
    "naaku", "chupinchu", "unnadi", "photo kavali",
    "enikku", "kanikku", "ente", "venam",
    "nanage", "torisu", "nanna", "beku",
    "amar", "dekhao", "lagbe",
}

def should_use_ai_normalizer(original: str, locally_translated: str) -> bool:
    if any(ord(character) > 127 for character in locally_translated):
        return True

    lowered = original.casefold()
    return any(hint in lowered for hint in ROMANIZED_AI_HINTS)


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

    # Normalize likely non-English / transliterated queries with Gemini
    # when configured. Deterministic local mappings remain the offline fallback.
    if should_use_ai_normalizer(original, search_text):

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
