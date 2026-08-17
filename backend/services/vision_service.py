"""
MemoryOS - Gemini Vision Service
================================

Multimodal visual intelligence layer for MemoryOS.

Responsibilities
----------------
- Send the ACTUAL image to Gemini
- Perform visual-first understanding
- Use OCR only as supporting evidence
- Recognize food, products, screenshots, documents and scenes
- Extract structured searchable information
- Normalize Gemini output
- Retry transient Gemini failures
- Safely parse JSON
- Provide deterministic failure fallback
- Never expose API credentials

Architecture
------------

                    ACTUAL IMAGE
                         │
                         ▼
                  Gemini Vision
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Visual understanding    OCR context
              │                     │
              └──────────┬──────────┘
                         ▼
                 VisionAnalysis
                         │
                         ▼
                  MemoryService
                         │
              ┌──────────┼──────────┐
              ▼          ▼          ▼
           Search     Embedding    Storage

Important
---------
Gemini Vision is responsible for visual understanding.

OCR is optional supporting evidence.

An image with zero OCR text must still be visually analyzed.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

from backend.utils.image_utils import (
    ImageProcessingError,
    close_image,
    convert_to_rgb,
    image_to_bytes,
    open_image,
)

# ============================================================================
# LOGGER
# ============================================================================

logger = logging.getLogger("memoryos.vision")


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_MODEL = "gemini-2.5-flash"

DEFAULT_TEMPERATURE = 0.1

DEFAULT_MAX_OUTPUT_TOKENS = 4096

DEFAULT_TIMEOUT_SECONDS = 60

DEFAULT_MAX_RETRIES = 2

MAX_OCR_CONTEXT = 12000

MAX_RESPONSE_TEXT = 100000

MAX_KEYWORDS = 50

MAX_PRODUCTS = 30

MAX_ENTITIES = 50


# ============================================================================
# CANONICAL MEMORYOS CATEGORIES
# ============================================================================

SUPPORTED_CATEGORIES = {
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
}


# ============================================================================
# GEMINI VISION PROMPT
# ============================================================================

VISION_SYSTEM_PROMPT = """
You are the visual intelligence engine of MemoryOS.

MemoryOS is an AI-powered visual memory system.

Your job is to inspect the ACTUAL IMAGE and convert what is
visually present into structured, searchable information.

============================================================
1. PRIMARY RULE
============================================================

THE IMAGE IS THE PRIMARY SOURCE OF TRUTH.

You MUST inspect the supplied image.

Do not behave like an OCR-only system.

OCR text, when supplied, is supporting evidence only.

If OCR is empty, incomplete, noisy, or incorrect:

CONTINUE VISUAL ANALYSIS.

Never classify an image as "other" merely because OCR is empty.

============================================================
2. VISUAL UNDERSTANDING
============================================================

Identify visible or strongly supported:

- food
- dishes
- beverages
- products
- shopping items
- electronics
- computers
- phones
- vehicles
- animals
- people
- buildings
- locations
- documents
- receipts
- tickets
- maps
- classroom material
- timetables
- exam papers
- application interfaces
- websites
- social media
- chats
- movie posters
- payment screens
- booking screens
- tracking screens
- logos
- recognizable objects
- scenes

Do NOT invent information that cannot reasonably be supported
by the image.

============================================================
3. CANONICAL CATEGORIES
============================================================

You MUST use one of these categories:

shopping
travel
food
finance
learning
work
communication
social
entertainment
health
document
technology
event
location
reminder
reference
payment
booking
tracking
other

Examples:

Dosa photo
→ food

Masala dosa photo
→ food

Idli photo
→ food

Biryani photo
→ food

Pizza photo
→ food

Laptop photo
→ technology

Phone photo
→ technology

Keyboard photo
→ technology

Shoes photo
→ shopping

Clothing photo
→ shopping

Product advertisement
→ shopping

Receipt
→ finance

Bank transaction
→ finance

UPI payment
→ payment

Classroom
→ learning

Timetable
→ learning

Exam paper
→ learning

Flight ticket
→ travel

Train ticket
→ travel

Map
→ location or travel

Hotel booking
→ booking

Delivery tracking
→ tracking

Chat
→ communication

Social media post
→ social

Movie poster
→ entertainment

Work dashboard
→ work

Medical prescription/document
→ health or document

Generic document
→ document

============================================================
4. FOOD RECOGNITION
============================================================

Food recognition is HIGH PRIORITY.

When food is visible, identify it as specifically as the image
allows.

Examples:

Dosa:

category:
"food"

subcategory:
"south indian food"

products:
["dosa"]

keywords:
["dosa", "food", "south indian", "breakfast", "indian food"]

Masala dosa:

products:
["masala dosa", "dosa"]

keywords:
[
  "masala dosa",
  "dosa",
  "food",
  "south indian",
  "breakfast",
  "indian food"
]

Idli:

products:
["idli"]

keywords:
[
  "idli",
  "food",
  "south indian",
  "breakfast",
  "indian food"
]

Biryani:

products:
["biryani"]

keywords:
[
  "biryani",
  "food",
  "indian food",
  "rice"
]

Pizza:

products:
["pizza"]

keywords:
[
  "pizza",
  "food",
  "italian food"
]

If several foods are visible, include the important recognizable
foods.

Do not invent exact ingredients that cannot be seen.

============================================================
5. PRODUCT RECOGNITION
============================================================

When a recognizable product is visible:

- identify the product
- identify the product type
- identify brand ONLY if clearly visible
- add it to products
- add useful search keywords

Examples:

Laptop:

category:
"technology"

products:
["laptop"]

keywords:
["laptop", "computer", "technology", "device"]

Shoes:

category:
"shopping"

products:
["shoes"]

keywords:
["shoes", "footwear", "shopping", "product"]

============================================================
6. SCREENSHOT UNDERSTANDING
============================================================

Screenshots are NOT just text containers.

Inspect the visual interface.

Identify:

- application
- website
- dashboard
- mobile app
- chat interface
- payment interface
- booking interface
- shopping interface
- social media interface
- code editor
- terminal
- document viewer
- map
- timetable
- educational interface

Use visual structure together with OCR.

============================================================
7. DOCUMENT UNDERSTANDING
============================================================

For documents identify:

- document type
- visible title
- important fields
- structure
- dates
- names
- amounts
- institutions
- visible document purpose

Do not invent unreadable information.

============================================================
8. VISUAL DESCRIPTION
============================================================

Always provide a useful visual_description when meaningful
visual information exists.

Describe:

- main subject
- important visible objects
- environment
- visual structure
- presentation

For food include:

- food type
- presentation
- visible accompaniments
- general appearance

For products include:

- product type
- visible design
- brand if clearly visible

For screenshots include:

- application/interface
- major visible elements
- important content

For documents include:

- document type
- visible structure

Also provide structured concepts whenever they are visibly supported:

- objects and actions
- scene and environment
- relationships between visible subjects (for example, an object in an
  interface or food served with an accompaniment)
- domain concepts for food, products, documents, technology, coding, UI,
  locations, activities, and topics

Leave a field empty when the image does not support it. These concept fields
must be discovered from the image, not selected from a fixed vocabulary.

============================================================
9. SEARCHABILITY
============================================================

MemoryOS must later retrieve memories using natural language.

Keywords MUST be useful for semantic search.

Include:

- object names
- food names
- product names
- category
- useful synonyms
- visible application names
- relevant concepts
- subcategory

Avoid unrelated keywords.

============================================================
10. ENTITIES
============================================================

Extract only entities supported by the image.

Possible entities include:

- people
- organizations
- locations
- products
- dates
- prices
- emails
- phone numbers
- URLs

Do not hallucinate.

============================================================
11. INTENT
============================================================

Use one of:

buy
sell
save
remember
learn
compare
contact
travel
track
schedule
read
reference
unknown

============================================================
12. IMPORTANCE
============================================================

Return an integer:

1 = low importance
2 = somewhat useful
3 = normal
4 = important
5 = highly important

============================================================
13. ANTI-HALLUCINATION
============================================================

Never invent:

- people
- brands
- prices
- locations
- dates
- products
- phone numbers
- emails
- URLs
- exact text

If uncertain:

return null or [].

But if a visual subject is clearly recognizable,
DO provide the appropriate visual information.

============================================================
14. OUTPUT
============================================================

Return ONLY valid JSON.

Do not use Markdown.

Use exactly this structure:

{
  "title": "",
  "summary": "",
  "category": "",
  "subcategory": "",
  "entities": [],
  "people": [],
  "organizations": [],
  "locations": [],
  "products": [],
  "dates": [],
  "prices": [],
  "emails": [],
  "phone_numbers": [],
  "urls": [],
  "intent": "unknown",
  "keywords": [],
  "importance": 1,
  "visual_description": "",
  "visual_concepts": [],
  "objects": [],
  "actions": [],
  "relationships": [],
  "scene": "",
  "environment": "",
  "food_concepts": [],
  "product_concepts": [],
  "document_concepts": [],
  "technology_concepts": [],
  "coding_concepts": [],
  "ui_concepts": [],
  "location_concepts": [],
  "activity_concepts": [],
  "topic_concepts": [],
  "general_concepts": [],
  "actionable_items": []
}
"""


# ============================================================================
# EXCEPTIONS
# ============================================================================


class VisionServiceError(Exception):
    """Base exception for Vision Service."""


class VisionConfigurationError(VisionServiceError):
    """Gemini configuration problem."""


class VisionUnavailableError(VisionServiceError):
    """Gemini service unavailable."""


class VisionAuthenticationError(VisionServiceError):
    """Gemini authentication failure."""


class VisionRateLimitError(VisionServiceError):
    """Gemini rate limit or quota failure."""


class VisionResponseError(VisionServiceError):
    """Gemini returned invalid or unusable output."""


# ============================================================================
# DATA MODEL
# ============================================================================


@dataclass(frozen=True, slots=True)
class VisionAnalysis:
    """
    Canonical structured result from Gemini Vision.
    """

    success: bool

    title: str
    summary: str

    category: str
    subcategory: str

    entities: tuple[str, ...]
    people: tuple[str, ...]
    organizations: tuple[str, ...]
    locations: tuple[str, ...]
    products: tuple[str, ...]

    dates: tuple[str, ...]
    prices: tuple[str, ...]

    emails: tuple[str, ...]
    phone_numbers: tuple[str, ...]
    urls: tuple[str, ...]

    intent: str

    keywords: tuple[str, ...]

    importance: int

    visual_description: str

    # Concepts remain separate from free-form description so retrieval can
    # distinguish direct visual evidence from OCR and filename text.
    visual_concepts: tuple[str, ...]
    objects: tuple[str, ...]
    actions: tuple[str, ...]
    relationships: tuple[str, ...]
    scene: str
    environment: str
    food_concepts: tuple[str, ...]
    product_concepts: tuple[str, ...]
    document_concepts: tuple[str, ...]
    technology_concepts: tuple[str, ...]
    coding_concepts: tuple[str, ...]
    ui_concepts: tuple[str, ...]
    location_concepts: tuple[str, ...]
    activity_concepts: tuple[str, ...]
    topic_concepts: tuple[str, ...]
    general_concepts: tuple[str, ...]

    actionable_items: tuple[str, ...]

    model: str

    fallback: bool

    raw_response: str | None = None

    error: str | None = None

    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert analysis into a JSON-compatible dictionary.
        """

        return {
            "success": self.success,
            "title": self.title,
            "summary": self.summary,
            "category": self.category,
            "subcategory": self.subcategory,
            "entities": list(self.entities),
            "people": list(self.people),
            "organizations": list(self.organizations),
            "locations": list(self.locations),
            "products": list(self.products),
            "dates": list(self.dates),
            "prices": list(self.prices),
            "emails": list(self.emails),
            "phone_numbers": list(self.phone_numbers),
            "urls": list(self.urls),
            "intent": self.intent,
            "keywords": list(self.keywords),
            "importance": self.importance,
            "visual_description": self.visual_description,
            "visual_concepts": list(self.visual_concepts),
            "objects": list(self.objects),
            "actions": list(self.actions),
            "relationships": list(self.relationships),
            "scene": self.scene,
            "environment": self.environment,
            "food_concepts": list(self.food_concepts),
            "product_concepts": list(self.product_concepts),
            "document_concepts": list(self.document_concepts),
            "technology_concepts": list(self.technology_concepts),
            "coding_concepts": list(self.coding_concepts),
            "ui_concepts": list(self.ui_concepts),
            "location_concepts": list(self.location_concepts),
            "activity_concepts": list(self.activity_concepts),
            "topic_concepts": list(self.topic_concepts),
            "general_concepts": list(self.general_concepts),
            "actionable_items": list(self.actionable_items),
            "model": self.model,
            "fallback": self.fallback,
            "error": self.error,
            "metadata": self.metadata,
        }


# ============================================================================
# ENVIRONMENT
# ============================================================================


def get_gemini_api_key() -> str | None:
    """
    Read Gemini API key from environment.

    Supported environment variables:

        GEMINI_API_KEY
        GOOGLE_API_KEY
    """

    key = os.getenv("GEMINI_API_KEY")

    if not key:
        key = os.getenv("GOOGLE_API_KEY")

    if not key:
        return None

    key = key.strip()

    return key or None


def is_gemini_configured() -> bool:
    """
    Return True when a Gemini API key exists.
    """

    return bool(get_gemini_api_key())


# ============================================================================
# STRING HELPERS
# ============================================================================


def _safe_string(value: Any) -> str:
    """
    Safely convert arbitrary values into strings.
    """

    if value is None:
        return ""

    if isinstance(value, bytes):
        return value.decode(
            "utf-8",
            errors="replace",
        ).strip()

    return str(value).strip()


def _safe_string_list(
    value: Any,
    *,
    limit: int = 50,
) -> tuple[str, ...]:
    """
    Normalize Gemini list-like values.
    """

    if value is None:
        return ()

    if isinstance(value, str):
        value = [value]

    if not isinstance(value, (list, tuple, set)):
        return ()

    result: list[str] = []

    for item in value:

        cleaned = _safe_string(item)

        if not cleaned:
            continue

        if cleaned not in result:
            result.append(cleaned)

        if len(result) >= limit:
            break

    return tuple(result)


def _safe_importance(value: Any) -> int:
    """
    Normalize importance into 1-5.
    """

    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = 1

    return max(
        1,
        min(
            5,
            number,
        ),
    )


# ============================================================================
# CATEGORY NORMALIZATION
# ============================================================================


def _normalize_category(value: Any) -> str:
    """
    Normalize Gemini category into MemoryOS canonical categories.
    """

    category = _safe_string(value).lower()

    category = category.replace(
        "_",
        " ",
    ).replace(
        "-",
        " ",
    )

    aliases = {
        "food image": "food",
        "food photo": "food",
        "foods": "food",
        "meal": "food",
        "restaurant": "food",
        "product": "shopping",
        "products": "shopping",
        "shopping item": "shopping",
        "shopping product": "shopping",
        "purchase": "shopping",
        "tech": "technology",
        "electronics": "technology",
        "electronic": "technology",
        "device": "technology",
        "education": "learning",
        "school": "learning",
        "academic": "learning",
        "study": "learning",
        "document": "document",
        "documents": "document",
        "paper": "document",
        "pdf": "document",
        "payment": "payment",
        "payments": "payment",
        "upi": "payment",
        "banking": "finance",
        "financial": "finance",
        "chat": "communication",
        "messaging": "communication",
        "message": "communication",
        "social media": "social",
        "reservation": "booking",
        "hotel booking": "booking",
        "delivery": "tracking",
        "delivery tracking": "tracking",
        "place": "location",
        "map": "location",
        "event": "event",
        "events": "event",
    }

    normalized = aliases.get(
        category,
        category,
    )

    if normalized not in SUPPORTED_CATEGORIES:
        return "other"

    return normalized


# ============================================================================
# SEARCH TERM REINFORCEMENT
# ============================================================================


def _reinforce_search_terms(
    data: dict[str, Any],
) -> dict[str, Any]:
    """
    Add deterministic search terms without inventing visual information.
    """

    category = _normalize_category(data.get("category"))

    data["category"] = category

    keywords = list(
        _safe_string_list(
            data.get("keywords"),
            limit=MAX_KEYWORDS,
        )
    )

    products = list(
        _safe_string_list(
            data.get("products"),
            limit=MAX_PRODUCTS,
        )
    )

    subcategory = _safe_string(data.get("subcategory"))

    title = _safe_string(data.get("title"))

    visual_description = _safe_string(data.get("visual_description"))

    def add(value: Any) -> None:

        value = _safe_string(value)

        if not value:
            return

        if value not in keywords:
            keywords.append(value)

    # Canonical category
    if category != "other":
        add(category)

    # Subcategory
    if subcategory:
        add(subcategory)

    # Every concept comes from Gemini's image-grounded response.  Retaining
    # these terms in keywords makes the existing embedding pipeline extensible
    # without query-specific rules or a second index.
    for product in products:
        add(product)

    for field_name in (
        "visual_concepts", "objects", "actions", "relationships",
        "food_concepts", "product_concepts", "document_concepts",
        "technology_concepts", "coding_concepts", "ui_concepts",
        "location_concepts", "activity_concepts", "topic_concepts",
        "general_concepts",
    ):
        for concept in _safe_string_list(data.get(field_name), limit=MAX_KEYWORDS):
            add(concept)

    for field_name in ("scene", "environment"):
        add(data.get(field_name))

    # Title
    if title:
        add(title)

    # Food reinforcement
    if category == "food":
        add("food")
        add("food image")

    # Visual description is intentionally NOT copied wholesale
    # into keywords because that creates noisy search vectors.

    data["keywords"] = keywords[:MAX_KEYWORDS]

    data["products"] = products[:MAX_PRODUCTS]

    # Keep the description intact.
    data["visual_description"] = visual_description

    return data


# ============================================================================
# JSON PARSING
# ============================================================================


def _remove_markdown_fences(
    text: str,
) -> str:
    """
    Remove accidental Markdown JSON fences.
    """

    value = text.strip()

    value = re.sub(
        r"^```(?:json)?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = re.sub(
        r"\s*```$",
        "",
        value,
    )

    return value.strip()


def _extract_json_object(
    text: str,
) -> dict[str, Any]:
    """
    Safely extract the first valid JSON object.
    """

    cleaned = _remove_markdown_fences(text[:MAX_RESPONSE_TEXT])

    try:

        parsed = json.loads(cleaned)

        if not isinstance(parsed, dict):
            raise VisionResponseError("Gemini response is not a JSON object.")

        return parsed

    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")

    if start < 0:
        raise VisionResponseError("Gemini response contains no JSON object.")

    depth = 0
    in_string = False
    escaped = False

    for index in range(
        start,
        len(cleaned),
    ):

        char = cleaned[index]

        if escaped:
            escaped = False
            continue

        if char == "\\" and in_string:
            escaped = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1

        elif char == "}":

            depth -= 1

            if depth == 0:

                candidate = cleaned[start : index + 1]

                try:

                    parsed = json.loads(candidate)

                    if isinstance(parsed, dict):
                        return parsed

                except json.JSONDecodeError:
                    break

    raise VisionResponseError("Gemini returned invalid JSON.")


# ============================================================================
# RESPONSE CONVERSION
# ============================================================================


def _analysis_from_dict(
    data: dict[str, Any],
    *,
    model: str,
    raw_response: str | None = None,
    fallback: bool = False,
    error: str | None = None,
) -> VisionAnalysis:
    """
    Convert untrusted Gemini JSON into canonical VisionAnalysis.
    """

    data = _reinforce_search_terms(data)

    return VisionAnalysis(
        success=error is None,
        title=_safe_string(data.get("title")),
        summary=_safe_string(data.get("summary")),
        category=_normalize_category(data.get("category")),
        subcategory=_safe_string(data.get("subcategory")),
        entities=_safe_string_list(
            data.get("entities"),
            limit=MAX_ENTITIES,
        ),
        people=_safe_string_list(
            data.get("people"),
            limit=MAX_ENTITIES,
        ),
        organizations=_safe_string_list(
            data.get("organizations"),
            limit=MAX_ENTITIES,
        ),
        locations=_safe_string_list(
            data.get("locations"),
            limit=MAX_ENTITIES,
        ),
        products=_safe_string_list(
            data.get("products"),
            limit=MAX_PRODUCTS,
        ),
        dates=_safe_string_list(
            data.get("dates"),
            limit=30,
        ),
        prices=_safe_string_list(
            data.get("prices"),
            limit=30,
        ),
        emails=_safe_string_list(
            data.get("emails"),
            limit=30,
        ),
        phone_numbers=_safe_string_list(
            data.get("phone_numbers"),
            limit=30,
        ),
        urls=_safe_string_list(
            data.get("urls"),
            limit=30,
        ),
        intent=_safe_string(data.get("intent")) or "unknown",
        keywords=_safe_string_list(
            data.get("keywords"),
            limit=MAX_KEYWORDS,
        ),
        importance=_safe_importance(data.get("importance")),
        visual_description=_safe_string(data.get("visual_description")),
        visual_concepts=_safe_string_list(data.get("visual_concepts"), limit=MAX_KEYWORDS),
        objects=_safe_string_list(data.get("objects"), limit=MAX_ENTITIES),
        actions=_safe_string_list(data.get("actions"), limit=MAX_KEYWORDS),
        relationships=_safe_string_list(data.get("relationships"), limit=MAX_KEYWORDS),
        scene=_safe_string(data.get("scene")),
        environment=_safe_string(data.get("environment")),
        food_concepts=_safe_string_list(data.get("food_concepts"), limit=MAX_KEYWORDS),
        product_concepts=_safe_string_list(data.get("product_concepts"), limit=MAX_KEYWORDS),
        document_concepts=_safe_string_list(data.get("document_concepts"), limit=MAX_KEYWORDS),
        technology_concepts=_safe_string_list(data.get("technology_concepts"), limit=MAX_KEYWORDS),
        coding_concepts=_safe_string_list(data.get("coding_concepts"), limit=MAX_KEYWORDS),
        ui_concepts=_safe_string_list(data.get("ui_concepts"), limit=MAX_KEYWORDS),
        location_concepts=_safe_string_list(data.get("location_concepts"), limit=MAX_KEYWORDS),
        activity_concepts=_safe_string_list(data.get("activity_concepts"), limit=MAX_KEYWORDS),
        topic_concepts=_safe_string_list(data.get("topic_concepts"), limit=MAX_KEYWORDS),
        general_concepts=_safe_string_list(data.get("general_concepts"), limit=MAX_KEYWORDS),
        actionable_items=_safe_string_list(
            data.get("actionable_items"),
            limit=30,
        ),
        model=model,
        fallback=fallback,
        raw_response=raw_response,
        error=error,
        metadata={
            "visual_analysis": not fallback,
            "category_source": ("local_fallback" if fallback else "gemini_vision"),
            "ocr_required": False,
        },
    )


# ============================================================================
# GEMINI CLIENT
# ============================================================================


def _create_client(
    api_key: str,
):
    """
    Lazily create Gemini client.
    """

    try:

        from google import genai

    except ImportError as exc:

        raise VisionConfigurationError("google-genai is not installed.") from exc

    try:

        return genai.Client(api_key=api_key)

    except Exception as exc:

        raise VisionConfigurationError("Unable to initialize Gemini client.") from exc


# ============================================================================
# VISION SERVICE
# ============================================================================


class VisionService:
    """
    Gemini-powered multimodal visual intelligence service.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:

        if not model:
            raise ValueError("Gemini model cannot be empty.")

        if temperature < 0:
            raise ValueError("Temperature cannot be negative.")

        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive.")

        if timeout <= 0:
            raise ValueError("timeout must be positive.")

        if max_retries < 0:
            raise ValueError("max_retries cannot be negative.")

        self.api_key = api_key or get_gemini_api_key()

        self.model = model

        self.temperature = temperature

        self.max_output_tokens = max_output_tokens

        self.timeout = timeout

        self.max_retries = max_retries

        self._client = None

    # ========================================================================
    # CONFIGURATION
    # ========================================================================

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    # ========================================================================
    # CLIENT
    # ========================================================================

    def _get_client(self):

        if not self.api_key:

            raise VisionConfigurationError("Gemini API key is not configured.")

        if self._client is None:

            self._client = _create_client(self.api_key)

        return self._client

    # ========================================================================
    # IMAGE PREPARATION
    # ========================================================================

    def _prepare_image(
        self,
        source: Any,
    ) -> tuple[bytes, str]:

        image: Image.Image | None = None

        try:

            image = open_image(source)

            image = convert_to_rgb(image)

            image_bytes = image_to_bytes(
                image,
                format="JPEG",
                quality=92,
            )

            return (
                image_bytes,
                "image/jpeg",
            )

        except (
            ImageProcessingError,
            OSError,
            ValueError,
        ) as exc:

            raise VisionServiceError(f"Unable to prepare image: {exc}") from exc

        finally:

            close_image(image)

    # ========================================================================
    # GEMINI REQUEST
    # ========================================================================

    def _generate(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> str:

        client = self._get_client()

        try:

            from google.genai import types

            image_part = types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type,
            )

            response = client.models.generate_content(
                model=self.model,
                contents=[
                    image_part,
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_output_tokens,
                    response_mime_type="application/json",
                ),
            )

            text = getattr(
                response,
                "text",
                None,
            )

            if not text:

                raise VisionResponseError("Gemini returned an empty response.")

            return str(text)

        except VisionResponseError:
            raise

        except Exception as exc:

            message = str(exc).lower()

            if any(
                token in message
                for token in (
                    "401",
                    "403",
                    "api key",
                    "authentication",
                    "permission",
                    "unauthorized",
                )
            ):

                raise VisionAuthenticationError(
                    "Gemini authentication failed."
                ) from exc

            if any(
                token in message
                for token in (
                    "429",
                    "rate limit",
                    "resource exhausted",
                    "quota",
                )
            ):

                raise VisionRateLimitError("Gemini API rate limit reached.") from exc

            raise VisionUnavailableError(f"Gemini request failed: {exc}") from exc

    # ========================================================================
    # RETRY
    # ========================================================================

    def _call_with_retry(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> str:

        attempts = self.max_retries + 1

        last_error: Exception | None = None

        for attempt in range(attempts):

            try:

                return self._generate(
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    prompt=prompt,
                )

            except (
                VisionRateLimitError,
                VisionUnavailableError,
            ) as exc:

                last_error = exc

                if attempt >= attempts - 1:
                    break

                delay = min(
                    2**attempt,
                    8,
                )

                logger.warning(
                    "Gemini request failed; " "retrying in %ss.",
                    delay,
                )

                time.sleep(delay)

            except (
                VisionAuthenticationError,
                VisionConfigurationError,
                VisionResponseError,
            ):

                raise

        raise VisionUnavailableError(str(last_error or "Gemini request failed."))

    # ========================================================================
    # PROMPT BUILDER
    # ========================================================================

    @staticmethod
    def _build_prompt(
        ocr_text: str | None = None,
    ) -> str:
        """
        Build the final multimodal prompt.
        """

        prompt = VISION_SYSTEM_PROMPT

        if ocr_text:

            cleaned_ocr = ocr_text.strip()

            if cleaned_ocr:

                prompt += f"""

============================================================
LOCAL OCR SUPPORTING CONTEXT
============================================================

The following text was extracted by the local OCR engine.

IMPORTANT:

- OCR is supporting evidence only.
- Inspect the actual image.
- Do NOT classify only from OCR.
- Correct obvious OCR mistakes using visual evidence.
- Continue visual analysis even when OCR is incomplete.

OCR TEXT:

{cleaned_ocr[:MAX_OCR_CONTEXT]}
"""

                return prompt

        prompt += """

============================================================
NO USEFUL OCR AVAILABLE
============================================================

No useful OCR text was extracted.

This does NOT mean the image is empty.

You MUST inspect the actual image and identify:

- food
- products
- objects
- devices
- documents
- screenshots
- interfaces
- vehicles
- locations
- scenes
- recognizable visual subjects

Perform visual classification independently of OCR.
"""

        return prompt

    # ========================================================================
    # MAIN ANALYSIS
    # ========================================================================

    def analyze(
        self,
        source: Any,
        *,
        allow_fallback: bool = True,
    ) -> VisionAnalysis:
        """
        Perform pure visual analysis.
        """

        if not self.configured:

            error = "Gemini API key is not configured."

            logger.warning(error)

            if allow_fallback:
                return self._fallback_analysis(error)

            raise VisionConfigurationError(error)

        try:

            image_bytes, mime_type = self._prepare_image(source)

            response_text = self._call_with_retry(
                image_bytes=image_bytes,
                mime_type=mime_type,
                prompt=self._build_prompt(),
            )

            parsed = _extract_json_object(response_text)

            return _analysis_from_dict(
                parsed,
                model=self.model,
                raw_response=response_text,
            )

        except VisionServiceError as exc:

            logger.error(
                "Gemini Vision failed: %s",
                exc,
            )

            if allow_fallback:
                return self._fallback_analysis(str(exc))

            raise

        except Exception as exc:

            logger.exception("Unexpected Gemini Vision error.")

            if allow_fallback:

                return self._fallback_analysis(f"Unexpected Gemini error: {exc}")

            raise VisionServiceError(f"Unexpected Gemini error: {exc}") from exc

    # ========================================================================
    # OCR + VISUAL ANALYSIS
    # ========================================================================

    def analyze_with_ocr(
        self,
        source: Any,
        ocr_text: str | None = None,
        *,
        allow_fallback: bool = True,
    ) -> VisionAnalysis:
        """
        Analyze actual image with optional OCR supporting context.
        """

        if not self.configured:

            error = "Gemini API key is not configured."

            logger.warning(error)

            if allow_fallback:
                return self._fallback_analysis(error)

            raise VisionConfigurationError(error)

        try:

            image_bytes, mime_type = self._prepare_image(source)

            prompt = self._build_prompt(ocr_text)

            response_text = self._call_with_retry(
                image_bytes=image_bytes,
                mime_type=mime_type,
                prompt=prompt,
            )

            parsed = _extract_json_object(response_text)

            return _analysis_from_dict(
                parsed,
                model=self.model,
                raw_response=response_text,
            )

        except VisionServiceError as exc:

            logger.error(
                "Gemini OCR-assisted Vision failed: %s",
                exc,
            )

            if allow_fallback:
                return self._fallback_analysis(str(exc))

            raise

        except Exception as exc:

            logger.exception("Unexpected Gemini OCR-assisted error.")

            if allow_fallback:

                return self._fallback_analysis(f"Unexpected Gemini error: {exc}")

            raise VisionServiceError(f"Unexpected Gemini error: {exc}") from exc

    # ========================================================================
    # FALLBACK
    # ========================================================================

    @staticmethod
    def _fallback_analysis(
        error: str,
    ) -> VisionAnalysis:
        """
        Deterministic failure result.

        This explicitly marks Gemini as unavailable.
        It does NOT pretend that visual analysis succeeded.
        """

        return VisionAnalysis(
            success=False,
            title="",
            summary="",
            category="other",
            subcategory="",
            entities=(),
            people=(),
            organizations=(),
            locations=(),
            products=(),
            dates=(),
            prices=(),
            emails=(),
            phone_numbers=(),
            urls=(),
            intent="unknown",
            keywords=(),
            importance=1,
            visual_description="",
            visual_concepts=(),
            objects=(),
            actions=(),
            relationships=(),
            scene="",
            environment="",
            food_concepts=(),
            product_concepts=(),
            document_concepts=(),
            technology_concepts=(),
            coding_concepts=(),
            ui_concepts=(),
            location_concepts=(),
            activity_concepts=(),
            topic_concepts=(),
            general_concepts=(),
            actionable_items=(),
            model="local-fallback",
            fallback=True,
            raw_response=None,
            error=error,
            metadata={
                "visual_analysis": False,
                "category_source": "local_fallback",
                "ocr_required": False,
                "reason": "gemini_unavailable",
            },
        )


# ============================================================================
# DEFAULT SERVICE
# ============================================================================

vision_service = VisionService()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================


def analyze_image(
    source: Any,
    *,
    allow_fallback: bool = True,
) -> VisionAnalysis:
    """
    Analyze an image using Gemini Vision.
    """

    return vision_service.analyze(
        source,
        allow_fallback=allow_fallback,
    )


def analyze_image_with_ocr(
    source: Any,
    ocr_text: str | None = None,
    *,
    allow_fallback: bool = True,
) -> VisionAnalysis:
    """
    Analyze an image with optional OCR supporting context.
    """

    return vision_service.analyze_with_ocr(
        source,
        ocr_text,
        allow_fallback=allow_fallback,
    )


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "VisionAnalysis",
    "VisionServiceError",
    "VisionConfigurationError",
    "VisionUnavailableError",
    "VisionAuthenticationError",
    "VisionRateLimitError",
    "VisionResponseError",
    "get_gemini_api_key",
    "is_gemini_configured",
    "VisionService",
    "vision_service",
    "analyze_image",
    "analyze_image_with_ocr",
]
