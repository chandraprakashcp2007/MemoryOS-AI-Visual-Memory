"""
MemoryOS - Automatic Screenshot Indexer
=======================================

Purpose
-------
Automatically turn a large folder of screenshots into searchable memories.

Pipeline:

    screenshots/
        ↓
    image discovery
        ↓
    validation
        ↓
    OCR
        ↓
    text cleaning
        ↓
    lightweight classification
        ↓
    searchable document
        ↓
    semantic embedding
        ↓
    FAISS index
        ↓
    metadata.json

Designed for the MemoryOS hackathon MVP.

This file intentionally does NOT handle:
- user uploads
- authentication
- frontend rendering
- database users
- cloud deployment

The challenge is simple:

    4,000 screenshots
          ↓
    automatically understand/index them
          ↓
    search by meaning
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependencies
# ---------------------------------------------------------------------------

try:
    from PIL import Image, ImageOps, ImageFilter
except ImportError:
    Image = None
    ImageOps = None
    ImageFilter = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCREENSHOT_DIR = PROJECT_ROOT / "data" / "screenshots"

INDEX_DIR = PROJECT_ROOT / "data" / "faiss"

METADATA_FILE = INDEX_DIR / "screenshot_metadata.json"

INDEX_FILE = INDEX_DIR / "screenshots.index"

STATE_FILE = INDEX_DIR / "index_state.json"

LOG_DIR = PROJECT_ROOT / "logs"

LOG_FILE = LOG_DIR / "screenshot_indexer.log"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}

# Sentence Transformers model.
#
# This is intentionally a small general-purpose semantic model suitable
# for a hackathon prototype.
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "MEMORYOS_EMBEDDING_MODEL",
    "all-MiniLM-L6-v2",
)

BATCH_SIZE = int(os.getenv("MEMORYOS_INDEX_BATCH_SIZE", "32"))

MAX_OCR_CHARS = int(os.getenv("MEMORYOS_MAX_OCR_CHARS", "12000"))

MIN_TEXT_LENGTH = int(os.getenv("MEMORYOS_MIN_TEXT_LENGTH", "2"))

TOP_CATEGORIES = [
    "recipe",
    "receipt",
    "shopping",
    "address",
    "document",
    "education",
    "coding",
    "travel",
    "social",
    "finance",
    "entertainment",
    "other",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("memoryos.screenshot_indexer")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class ScreenshotMemory:
    """
    Metadata stored for every indexed screenshot.

    The embedding itself is stored inside FAISS.
    """

    id: str
    filename: str
    path: str

    category: str

    ocr_text: str
    summary: str

    entities: List[str]

    file_size: int
    width: int
    height: int

    created_at: str
    indexed_at: str

    content_hash: str

    embedding_index: int

    status: str = "indexed"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def ensure_directories() -> None:
    """Create required runtime directories."""

    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def normalize_text(text: str) -> str:
    """
    Clean OCR output while preserving useful searchable information.
    """

    if not text:
        return ""

    text = text.replace("\x00", " ")

    # Normalize whitespace.
    text = re.sub(r"[ \t]+", " ", text)

    # Normalize excessive blank lines.
    text = re.sub(r"\n\s*\n+", "\n", text)

    # Remove obvious OCR noise.
    text = re.sub(r"[|]{3,}", " ", text)
    text = re.sub(r"[_]{4,}", " ", text)

    text = text.strip()

    if len(text) > MAX_OCR_CHARS:
        text = text[:MAX_OCR_CHARS]

    return text


def file_hash(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return a stable SHA-256 hash for an image."""

    digest = hashlib.sha256()

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def make_memory_id(content_hash: str) -> str:
    """Create a short stable MemoryOS ID."""

    return content_hash[:16]


def discover_images(folder: Path) -> List[Path]:
    """
    Recursively find supported images.

    Recursive scanning means users can organize screenshots into folders
    without changing the indexer.
    """

    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)

    images = []

    for path in folder.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() in SUPPORTED_EXTENSIONS:
            images.append(path)

    images.sort(key=lambda item: item.stat().st_mtime if item.exists() else 0)

    return images


# ---------------------------------------------------------------------------
# OCR
# ---------------------------------------------------------------------------


class OCRProcessor:
    """
    OCR processor.

    Uses the existing project's OCR service when it exposes a compatible
    callable. Otherwise falls back to pytesseract.

    This makes the indexer tolerant of the current project architecture.
    """

    def __init__(self) -> None:
        self.external_service = self._load_existing_ocr_service()

    def _load_existing_ocr_service(self) -> Any:
        """
        Try to reuse MemoryOS's existing OCR service.

        We intentionally keep this defensive because the existing project
        contains more than one service layout.
        """

        candidates = [
            ("backend.services.ocr_service", "OCRService"),
            ("backend.services.ocr_service", "ocr_service"),
            ("backend.app.services.ocr.ocr_service", "OCRService"),
            ("backend.app.services.ocr.ocr_service", "ocr_service"),
        ]

        for module_name, attribute_name in candidates:
            try:
                module = __import__(
                    module_name,
                    fromlist=[attribute_name],
                )

                attribute = getattr(
                    module,
                    attribute_name,
                    None,
                )

                if attribute is None:
                    continue

                if callable(attribute):
                    try:
                        instance = attribute()
                        logger.info(
                            "Using existing OCR service: %s.%s",
                            module_name,
                            attribute_name,
                        )
                        return instance
                    except TypeError:
                        return attribute

                return attribute

            except Exception:
                continue

        logger.info(
            "Existing OCR service not automatically detected; "
            "using local OCR fallback."
        )

        return None

    def _try_external(self, image_path: Path) -> Optional[str]:
        """Try common OCR service method names."""

        if self.external_service is None:
            return None

        method_names = [
            "extract_text",
            "extract",
            "process",
            "ocr",
            "recognize",
            "run",
        ]

        for method_name in method_names:
            method = getattr(
                self.external_service,
                method_name,
                None,
            )

            if not callable(method):
                continue

            attempts = [
                lambda: method(str(image_path)),
                lambda: method(image_path),
            ]

            for attempt in attempts:
                try:
                    result = attempt()

                    if isinstance(result, str):
                        return normalize_text(result)

                    if isinstance(result, dict):
                        for key in (
                            "text",
                            "ocr_text",
                            "content",
                            "raw_text",
                        ):
                            value = result.get(key)

                            if isinstance(value, str):
                                return normalize_text(value)

                except Exception:
                    continue

        return None

    def _preprocess(self, image: Any) -> Any:
        """Improve screenshot OCR quality."""

        if ImageOps is None:
            return image

        try:
            image = image.convert("RGB")

            # Convert to grayscale.
            image = ImageOps.grayscale(image)

            # Increase contrast.
            image = ImageOps.autocontrast(image)

            # Slight sharpening.
            if ImageFilter is not None:
                image = image.filter(ImageFilter.SHARPEN)

            return image

        except Exception:
            return image

    def _fallback_tesseract(self, image_path: Path) -> str:
        """OCR fallback using Tesseract."""

        if pytesseract is None:
            logger.warning(
                "pytesseract is not installed. " "OCR will be empty for %s",
                image_path.name,
            )
            return ""

        if Image is None:
            return ""

        try:
            with Image.open(image_path) as image:
                image = self._preprocess(image)

                text = pytesseract.image_to_string(
                    image,
                    config="--psm 6",
                )

                return normalize_text(text)

        except Exception as exc:
            logger.warning(
                "OCR failed for %s: %s",
                image_path.name,
                exc,
            )

            return ""

    def extract(self, image_path: Path) -> str:
        """Extract searchable text from an image."""

        external_text = self._try_external(image_path)

        if external_text:
            return external_text

        return self._fallback_tesseract(image_path)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


class ScreenshotClassifier:
    """
    Lightweight classification for the hackathon.

    The challenge requires screenshots to be classified. We use a transparent
    keyword-based classifier so the result is deterministic and easy to demo.

    It can later be replaced by the existing AI classifier without changing
    the indexer's external behavior.
    """

    KEYWORDS: Dict[str, Tuple[str, ...]] = {
        "recipe": (
            "recipe",
            "ingredients",
            "ingredient",
            "calories",
            "cooking",
            "cook",
            "bake",
            "baking",
            "masala",
            "chicken",
            "biryani",
            "curry",
            "food",
            "sauce",
        ),
        "receipt": (
            "receipt",
            "subtotal",
            "total",
            "tax",
            "invoice",
            "bill",
            "amount",
            "payment",
            "cash",
            "gst",
        ),
        "shopping": (
            "buy",
            "cart",
            "wishlist",
            "price",
            "₹",
            "$",
            "add to cart",
            "checkout",
            "shop",
            "amazon",
            "flipkart",
            "product",
        ),
        "address": (
            "address",
            "street",
            "road",
            "avenue",
            "pin code",
            "pincode",
            "zip code",
            "location",
            "latitude",
            "longitude",
        ),
        "coding": (
            "python",
            "javascript",
            "typescript",
            "java",
            "error",
            "exception",
            "traceback",
            "stack trace",
            "terminal",
            "console",
            "code",
            "github",
            "fastapi",
            "react",
            "sql",
            "debug",
        ),
        "education": (
            "college",
            "university",
            "student",
            "timetable",
            "schedule",
            "exam",
            "semester",
            "subject",
            "assignment",
            "marks",
            "course",
            "class",
        ),
        "finance": (
            "bank",
            "account",
            "transaction",
            "balance",
            "debit",
            "credit",
            "upi",
            "salary",
            "investment",
            "statement",
        ),
        "travel": (
            "flight",
            "boarding",
            "hotel",
            "booking",
            "train",
            "ticket",
            "airport",
            "trip",
            "travel",
            "reservation",
        ),
        "social": (
            "instagram",
            "facebook",
            "whatsapp",
            "telegram",
            "twitter",
            "linkedin",
            "comment",
            "message",
            "followers",
        ),
        "entertainment": (
            "movie",
            "film",
            "song",
            "music",
            "netflix",
            "youtube",
            "series",
            "episode",
            "game",
        ),
        "document": (
            "certificate",
            "document",
            "application",
            "form",
            "identity",
            "aadhaar",
            "passport",
            "license",
            "signature",
            "official",
        ),
    }

    def classify(self, text: str) -> str:
        """Return the strongest matching category."""

        if not text:
            return "other"

        normalized = text.lower()

        scores: Dict[str, int] = {}

        for category, keywords in self.KEYWORDS.items():
            score = 0

            for keyword in keywords:
                if keyword.lower() in normalized:
                    score += 1

            if score:
                scores[category] = score

        if not scores:
            return "other"

        return max(
            scores,
            key=scores.get,
        )

    def extract_entities(self, text: str) -> List[str]:
        """Extract useful high-level entities."""

        if not text:
            return []

        normalized = text.lower()

        candidates = [
            "Python",
            "FastAPI",
            "JavaScript",
            "TypeScript",
            "React",
            "GitHub",
            "Aadhaar",
            "UPI",
            "WhatsApp",
            "Instagram",
            "YouTube",
            "Amazon",
            "Flipkart",
            "Google",
            "Microsoft",
            "ChatGPT",
            "Chicken",
            "Biryani",
            "College",
            "University",
            "Timetable",
            "Receipt",
            "Invoice",
            "Recipe",
        ]

        found = []

        for entity in candidates:
            if entity.lower() in normalized:
                found.append(entity)

        return found[:20]

    def summarize(
        self,
        text: str,
        category: str,
        filename: str,
    ) -> str:
        """
        Create a compact human-readable summary.

        We intentionally keep this deterministic for the MVP.
        """

        if text:
            first_line = ""

            for line in text.splitlines():
                line = line.strip()

                if len(line) >= 5:
                    first_line = line
                    break

            if first_line:
                if len(first_line) > 140:
                    first_line = first_line[:137] + "..."

                return f"{category.title()} screenshot: {first_line}"

        return f"{category.title()} screenshot: {filename}"


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


class EmbeddingEngine:
    """Semantic embedding engine."""

    def __init__(self) -> None:
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers is not installed.\n"
                "Install it with:\n"
                "pip install sentence-transformers"
            )

        logger.info(
            "Loading embedding model: %s",
            DEFAULT_EMBEDDING_MODEL,
        )

        self.model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)

        self.dimension = self.model.get_sentence_embedding_dimension()

        logger.info(
            "Embedding dimension: %s",
            self.dimension,
        )

    def encode(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """Create normalized embeddings."""

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype=np.float32,
            )

        vectors = self.model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        return np.asarray(
            vectors,
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Persistent index
# ---------------------------------------------------------------------------


class PersistentIndex:
    """FAISS + JSON metadata persistence."""

    def __init__(self, dimension: int) -> None:
        if faiss is None:
            raise RuntimeError(
                "faiss is not installed.\n" "Install it with:\n" "pip install faiss-cpu"
            )

        self.dimension = dimension

        INDEX_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.metadata: List[Dict[str, Any]] = []

        self.index = self._load_index()

        self._load_metadata()

    def _load_index(self):
        """Load an existing FAISS index or create a new one."""

        if INDEX_FILE.exists():
            try:
                index = faiss.read_index(str(INDEX_FILE))

                if index.d != self.dimension:
                    logger.warning(
                        "Existing FAISS dimension %s differs from "
                        "current dimension %s. Rebuilding.",
                        index.d,
                        self.dimension,
                    )

                else:
                    logger.info(
                        "Loaded existing FAISS index: %s vectors",
                        index.ntotal,
                    )

                    return index

            except Exception as exc:
                logger.warning(
                    "Could not load existing FAISS index: %s",
                    exc,
                )

        # Inner product on normalized vectors == cosine similarity.
        return faiss.IndexFlatIP(self.dimension)

    def _load_metadata(self) -> None:
        """Load JSON metadata."""

        if not METADATA_FILE.exists():
            return

        try:
            with METADATA_FILE.open(
                "r",
                encoding="utf-8",
            ) as file:
                data = json.load(file)

            if isinstance(data, list):
                self.metadata = data

        except Exception as exc:
            logger.warning(
                "Could not load metadata: %s",
                exc,
            )

    def known_hashes(self) -> set[str]:
        """Return content hashes already indexed."""

        return {
            item.get("content_hash")
            for item in self.metadata
            if item.get("content_hash")
        }

    def add(
        self,
        embeddings: np.ndarray,
        memories: List[ScreenshotMemory],
    ) -> None:
        """Add vectors and matching metadata."""

        if len(memories) == 0:
            return

        if len(embeddings) != len(memories):
            raise ValueError("Embedding count does not match memory count.")

        start_index = self.index.ntotal

        self.index.add(
            np.asarray(
                embeddings,
                dtype=np.float32,
            )
        )

        for offset, memory in enumerate(memories):
            memory.embedding_index = start_index + offset

            self.metadata.append(asdict(memory))

    def save(self) -> None:
        """Persist FAISS and metadata."""

        faiss.write_index(
            self.index,
            str(INDEX_FILE),
        )

        with METADATA_FILE.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                self.metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )

        state = {
            "total_vectors": int(self.index.ntotal),
            "total_memories": len(self.metadata),
            "dimension": self.dimension,
            "updated_at": datetime.now().isoformat(),
            "index_file": str(INDEX_FILE),
            "metadata_file": str(METADATA_FILE),
        }

        with STATE_FILE.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                state,
                file,
                indent=2,
            )

    @property
    def size(self) -> int:
        return int(self.index.ntotal)


# ---------------------------------------------------------------------------
# Main indexer
# ---------------------------------------------------------------------------


class ScreenshotIndexer:
    """
    Main MemoryOS screenshot indexing engine.
    """

    def __init__(
        self,
        screenshot_dir: Optional[Path] = None,
    ) -> None:

        ensure_directories()

        self.screenshot_dir = Path(screenshot_dir) if screenshot_dir else SCREENSHOT_DIR

        self.ocr = OCRProcessor()

        self.classifier = ScreenshotClassifier()

        self.embedder = EmbeddingEngine()

        self.index = PersistentIndex(self.embedder.dimension)

    # ------------------------------------------------------------------
    # Image validation
    # ------------------------------------------------------------------

    def validate_image(
        self,
        image_path: Path,
    ) -> Tuple[int, int]:
        """Validate image and return width/height."""

        if Image is None:
            raise RuntimeError(
                "Pillow is not installed.\n" "Install it with:\n" "pip install pillow"
            )

        with Image.open(image_path) as image:
            image.verify()

        with Image.open(image_path) as image:
            width, height = image.size

        if width <= 0 or height <= 0:
            raise ValueError("Invalid image dimensions.")

        return width, height

    # ------------------------------------------------------------------
    # Search document
    # ------------------------------------------------------------------

    def build_search_document(
        self,
        filename: str,
        category: str,
        summary: str,
        entities: List[str],
        ocr_text: str,
    ) -> str:
        """
        Combine all useful information into one semantic document.

        This is what the embedding model actually sees.
        """

        entity_text = ", ".join(entities)

        return "\n".join(
            [
                f"Filename: {filename}",
                f"Category: {category}",
                f"Summary: {summary}",
                f"Entities: {entity_text}",
                f"Visible text: {ocr_text}",
            ]
        )

    # ------------------------------------------------------------------
    # Process single screenshot
    # ------------------------------------------------------------------

    def process_image(
        self,
        image_path: Path,
    ) -> Tuple[ScreenshotMemory, str]:
        """
        Process one screenshot.

        Returns:
            ScreenshotMemory
            Search document
        """

        width, height = self.validate_image(image_path)

        content_hash = file_hash(image_path)

        memory_id = make_memory_id(content_hash)

        ocr_text = self.ocr.extract(image_path)

        ocr_text = normalize_text(ocr_text)

        category = self.classifier.classify(ocr_text)

        entities = self.classifier.extract_entities(ocr_text)

        summary = self.classifier.summarize(
            text=ocr_text,
            category=category,
            filename=image_path.name,
        )

        search_document = self.build_search_document(
            filename=image_path.name,
            category=category,
            summary=summary,
            entities=entities,
            ocr_text=ocr_text,
        )

        indexed_at = datetime.now().isoformat()

        created_at = datetime.fromtimestamp(image_path.stat().st_mtime).isoformat()

        memory = ScreenshotMemory(
            id=memory_id,
            filename=image_path.name,
            path=str(image_path.relative_to(PROJECT_ROOT)),
            category=category,
            ocr_text=ocr_text,
            summary=summary,
            entities=entities,
            file_size=image_path.stat().st_size,
            width=width,
            height=height,
            created_at=created_at,
            indexed_at=indexed_at,
            content_hash=content_hash,
            embedding_index=-1,
        )

        return memory, search_document

    # ------------------------------------------------------------------
    # Index everything
    # ------------------------------------------------------------------

    def index_all(
        self,
        force: bool = False,
    ) -> Dict[str, Any]:
        """
        Scan and index all screenshots.

        Args:
            force:
                Re-index everything.

        Returns:
            Indexing statistics.
        """

        started_at = time.time()

        images = discover_images(self.screenshot_dir)

        logger.info("=" * 72)
        logger.info("MEMORYOS SCREENSHOT INDEXER")
        logger.info("=" * 72)
        logger.info(
            "Screenshot directory: %s",
            self.screenshot_dir,
        )
        logger.info(
            "Images discovered: %s",
            len(images),
        )
        logger.info(
            "Already indexed: %s",
            self.index.size,
        )

        known_hashes = set() if force else self.index.known_hashes()

        candidates = []

        skipped = 0

        for image_path in images:
            try:
                image_hash = file_hash(image_path)

                if not force and image_hash in known_hashes:
                    skipped += 1
                    continue

                candidates.append(
                    (
                        image_path,
                        image_hash,
                    )
                )

            except Exception as exc:
                logger.warning(
                    "Could not hash %s: %s",
                    image_path.name,
                    exc,
                )

        logger.info(
            "New screenshots to process: %s",
            len(candidates),
        )

        if not candidates:
            logger.info("Nothing new to index.")

            return {
                "status": "complete",
                "discovered": len(images),
                "processed": 0,
                "skipped": skipped,
                "failed": 0,
                "total_indexed": self.index.size,
                "duration_seconds": round(
                    time.time() - started_at,
                    2,
                ),
            }

        processed = 0
        failed = 0

        batch_memories: List[ScreenshotMemory] = []
        batch_documents: List[str] = []

        for position, (
            image_path,
            image_hash,
        ) in enumerate(
            candidates,
            start=1,
        ):

            logger.info(
                "[%s/%s] Processing %s",
                position,
                len(candidates),
                image_path.name,
            )

            try:
                memory, search_document = self.process_image(image_path)

                batch_memories.append(memory)

                batch_documents.append(search_document)

                processed += 1

                logger.info(
                    "    category=%s | OCR=%s chars | entities=%s",
                    memory.category,
                    len(memory.ocr_text),
                    len(memory.entities),
                )

            except Exception as exc:
                failed += 1

                logger.exception(
                    "Failed to process %s: %s",
                    image_path,
                    exc,
                )

            # Flush embedding batch.
            if len(batch_memories) >= BATCH_SIZE or position == len(candidates):

                if batch_memories:

                    logger.info(
                        "Creating embeddings for %s screenshots...",
                        len(batch_memories),
                    )

                    try:
                        embeddings = self.embedder.encode(batch_documents)

                        self.index.add(
                            embeddings,
                            batch_memories,
                        )

                        self.index.save()

                        logger.info(
                            "Index saved. Total vectors: %s",
                            self.index.size,
                        )

                    except Exception as exc:
                        failed += len(batch_memories)

                        processed -= len(batch_memories)

                        logger.exception(
                            "Embedding/index batch failed: %s",
                            exc,
                        )

                    finally:
                        batch_memories = []
                        batch_documents = []

        duration = time.time() - started_at

        result = {
            "status": "complete",
            "discovered": len(images),
            "candidates": len(candidates),
            "processed": processed,
            "skipped": skipped,
            "failed": failed,
            "total_indexed": self.index.size,
            "duration_seconds": round(
                duration,
                2,
            ),
            "index_file": str(INDEX_FILE),
            "metadata_file": str(METADATA_FILE),
        }

        logger.info("=" * 72)
        logger.info("INDEXING COMPLETE")
        logger.info("=" * 72)
        logger.info(
            "Discovered : %s",
            result["discovered"],
        )
        logger.info(
            "Processed  : %s",
            result["processed"],
        )
        logger.info(
            "Skipped    : %s",
            result["skipped"],
        )
        logger.info(
            "Failed     : %s",
            result["failed"],
        )
        logger.info(
            "Total      : %s",
            result["total_indexed"],
        )
        logger.info(
            "Duration   : %ss",
            result["duration_seconds"],
        )

        return result


# ---------------------------------------------------------------------------
# Search helper
# ---------------------------------------------------------------------------


class ScreenshotSearch:
    """
    Lightweight semantic search over the generated FAISS index.

    This is used by the API layer later.
    """

    def __init__(self) -> None:

        if faiss is None:
            raise RuntimeError("faiss is not installed.")

        if not INDEX_FILE.exists():
            raise FileNotFoundError(
                "Screenshot index does not exist yet.\n"
                "Run:\n"
                "python scripts/index_screenshots.py"
            )

        self.embedder = EmbeddingEngine()

        self.index = faiss.read_index(str(INDEX_FILE))

        if METADATA_FILE.exists():
            with METADATA_FILE.open(
                "r",
                encoding="utf-8",
            ) as file:
                self.metadata = json.load(file)
        else:
            self.metadata = []

    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """Semantic search."""

        query = normalize_text(query)

        if not query:
            return []

        if self.index.ntotal == 0:
            return []

        query_vector = self.embedder.encode([query])

        limit = min(
            max(top_k, 1),
            self.index.ntotal,
        )

        scores, indices = self.index.search(
            query_vector,
            limit,
        )

        results = []

        for score, index_position in zip(
            scores[0],
            indices[0],
        ):

            if index_position < 0:
                continue

            if index_position >= len(self.metadata):
                continue

            memory = dict(self.metadata[index_position])

            memory["score"] = round(
                float(score),
                4,
            )

            memory["why_matched"] = self._explain_match(
                query,
                memory,
                float(score),
            )

            results.append(memory)

        return results

    def _explain_match(
        self,
        query: str,
        memory: Dict[str, Any],
        score: float,
    ) -> List[str]:
        """
        Generate a simple explanation for the demo.
        """

        reasons = []

        query_words = set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                query.lower(),
            )
        )

        searchable_text = " ".join(
            [
                str(
                    memory.get(
                        "ocr_text",
                        "",
                    )
                ),
                str(
                    memory.get(
                        "summary",
                        "",
                    )
                ),
                str(
                    memory.get(
                        "category",
                        "",
                    )
                ),
                " ".join(
                    memory.get(
                        "entities",
                        [],
                    )
                ),
            ]
        ).lower()

        matched_words = [
            word for word in query_words if len(word) > 2 and word in searchable_text
        ]

        if matched_words:
            reasons.append(
                "Query terms found in screenshot content: "
                + ", ".join(matched_words[:5])
            )

        category = memory.get("category")

        if category:
            reasons.append(f"Classified as {category}")

        if score >= 0.75:
            reasons.append("High semantic similarity")
        elif score >= 0.55:
            reasons.append("Good semantic similarity")
        else:
            reasons.append("Semantic similarity match")

        return reasons


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def index_screenshots(
    screenshot_dir: Optional[str] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """
    Public helper for scripts/API usage.
    """

    directory = Path(screenshot_dir) if screenshot_dir else SCREENSHOT_DIR

    indexer = ScreenshotIndexer(screenshot_dir=directory)

    return indexer.index_all(force=force)


def semantic_search(
    query: str,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """
    Public semantic search helper.
    """

    search_engine = ScreenshotSearch()

    return search_engine.search(
        query=query,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Command-line support
# ---------------------------------------------------------------------------


def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(
        description=("MemoryOS automatic screenshot indexer")
    )

    parser.add_argument(
        "--folder",
        type=str,
        default=str(SCREENSHOT_DIR),
        help=("Folder containing screenshots."),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help=("Re-index screenshots even if " "they were already indexed."),
    )

    args = parser.parse_args()

    result = index_screenshots(
        screenshot_dir=args.folder,
        force=args.force,
    )

    print(
        json.dumps(
            result,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
