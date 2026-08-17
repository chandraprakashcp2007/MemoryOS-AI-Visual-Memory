"""
MemoryOS - Screenshot Processor
================================

Connects the screenshot scanner to the semantic-memory pipeline.

Pipeline:

    Screenshot
        ↓
    OCR
        ↓
    Clean text
        ↓
    Classification
        ↓
    Entities
        ↓
    Summary
        ↓
    Semantic document
        ↓
    Embedding
        ↓
    FAISS / Vector Store

Hackathon MVP:
- simple
- fault tolerant
- incremental
- reusable
- no upload dependency
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.services.screenshot_scanner import (
    ScreenshotFile,
    ScreenshotScanner,
)

logger = logging.getLogger("memoryos.screenshot_processor")

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SCREENSHOT_FOLDER = PROJECT_ROOT / "data" / "screenshots"

PROCESSING_DIR = PROJECT_ROOT / "data" / "memory_index"

MEMORY_FILE = PROCESSING_DIR / "memories.json"


# ============================================================================
# TEXT UTILITIES
# ============================================================================


def clean_text(text: str) -> str:
    """
    Clean OCR text while preserving useful information.
    """

    if not text:
        return ""

    text = text.replace(
        "\x00",
        " ",
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    text = text.strip()

    return text[:12000]


def extract_keywords(
    text: str,
    limit: int = 30,
) -> List[str]:
    """
    Extract simple searchable keywords.

    This is intentionally lightweight.
    Semantic search will handle meaning later.
    """

    if not text:
        return []

    words = re.findall(
        r"[A-Za-z0-9₹$@#._-]+",
        text.lower(),
    )

    stop_words = {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "you",
        "your",
        "are",
        "was",
        "have",
        "has",
        "but",
        "not",
        "www",
        "http",
        "https",
    }

    result = []

    seen = set()

    for word in words:

        word = word.strip("._-")

        if len(word) < 3:
            continue

        if word in stop_words:
            continue

        if word in seen:
            continue

        seen.add(word)

        result.append(word)

        if len(result) >= limit:
            break

    return result


# ============================================================================
# PROCESSOR
# ============================================================================


class ScreenshotProcessor:
    """
    Main automatic screenshot-to-memory processor.

    The processor is deliberately designed to work even when some of the
    older MemoryOS services have different interfaces.

    It first attempts to reuse existing services and falls back to the
    components created for the new screenshot pipeline.
    """

    def __init__(
        self,
        screenshot_folder: Optional[str | Path] = None,
    ) -> None:

        PROCESSING_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.screenshot_folder = Path(screenshot_folder or DEFAULT_SCREENSHOT_FOLDER)

        self.scanner = ScreenshotScanner(self.screenshot_folder)

        self.memories = self._load_memories()

        self.ocr_service = self._load_service(
            [
                (
                    "backend.services.ocr_service",
                    "OCRService",
                ),
                (
                    "backend.services.ocr_service",
                    "ocr_service",
                ),
                (
                    "backend.app.services.ocr.ocr_service",
                    "OCRService",
                ),
            ]
        )

        self.classifier = self._load_service(
            [
                (
                    "backend.services.classification_service",
                    "ClassificationService",
                ),
                (
                    "backend.services.classification_service",
                    "classifier",
                ),
                (
                    "backend.services.classification_service",
                    "classify",
                ),
            ]
        )

        self.entity_service = self._load_service(
            [
                (
                    "backend.services.entity_service",
                    "EntityService",
                ),
                (
                    "backend.services.entity_service",
                    "entity_service",
                ),
            ]
        )

        self.embedding_service = self._load_service(
            [
                (
                    "backend.services.embedding_service",
                    "EmbeddingService",
                ),
                (
                    "backend.services.embedding_service",
                    "embedding_service",
                ),
            ]
        )

        self.vector_store = self._load_service(
            [
                (
                    "backend.services.vector_store",
                    "VectorStore",
                ),
                (
                    "backend.services.vector_store",
                    "vector_store",
                ),
            ]
        )

        logger.info("ScreenshotProcessor initialized.")

    # =========================================================================
    # SERVICE LOADING
    # =========================================================================

    def _load_service(
        self,
        candidates: List[tuple[str, str]],
    ) -> Any:
        """
        Safely discover an existing MemoryOS service.
        """

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

                if isinstance(
                    attribute,
                    type,
                ):

                    try:
                        return attribute()

                    except Exception:
                        continue

                return attribute

            except Exception:
                continue

        return None

    # =========================================================================
    # MEMORY STORAGE
    # =========================================================================

    def _load_memories(self) -> Dict[str, Dict]:
        """
        Load processed screenshot metadata.
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

            logger.warning(
                "Could not load memory file: %s",
                exc,
            )

        return {}

    def _save_memories(self) -> None:
        """
        Persist screenshot metadata.
        """

        temporary_file = MEMORY_FILE.with_suffix(".tmp")

        with temporary_file.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                self.memories,
                file,
                indent=2,
                ensure_ascii=False,
            )

        temporary_file.replace(MEMORY_FILE)

    # =========================================================================
    # OCR
    # =========================================================================

    def extract_text(
        self,
        image_path: Path,
    ) -> str:
        """
        Extract OCR text.

        First tries the existing OCR service.
        Falls back to the OCR implementation from the indexer.
        """

        if self.ocr_service is not None:

            methods = [
                "extract_text",
                "extract",
                "process",
                "ocr",
                "recognize",
                "run",
            ]

            for method_name in methods:

                method = getattr(
                    self.ocr_service,
                    method_name,
                    None,
                )

                if not callable(method):
                    continue

                for argument in (
                    str(image_path),
                    image_path,
                ):

                    try:

                        result = method(argument)

                        text = self._extract_text_from_result(result)

                        if text:
                            return clean_text(text)

                    except Exception:
                        continue

        # Fallback to our indexer's OCR processor.
        try:

            from backend.services.screenshot_indexer import (
                OCRProcessor,
            )

            processor = OCRProcessor()

            return clean_text(processor.extract(image_path))

        except Exception as exc:

            logger.warning(
                "OCR fallback failed for %s: %s",
                image_path.name,
                exc,
            )

            return ""

    @staticmethod
    def _extract_text_from_result(
        result: Any,
    ) -> str:
        """
        Normalize different OCR service response formats.
        """

        if isinstance(
            result,
            str,
        ):
            return result

        if isinstance(
            result,
            dict,
        ):

            for key in (
                "text",
                "ocr_text",
                "content",
                "raw_text",
            ):

                value = result.get(key)

                if isinstance(
                    value,
                    str,
                ):
                    return value

        return ""

    # =========================================================================
    # CLASSIFICATION
    # =========================================================================

    def classify(
        self,
        text: str,
        filename: str,
    ) -> str:
        """
        Classify screenshot.

        Existing classifier first.
        Lightweight fallback second.
        """

        if self.classifier is not None:

            methods = [
                "classify",
                "predict",
                "analyze",
                "run",
            ]

            for method_name in methods:

                method = getattr(
                    self.classifier,
                    method_name,
                    None,
                )

                if not callable(method):
                    continue

                attempts = [
                    lambda: method(text),
                    lambda: method(text=text),
                    lambda: method(
                        filename=filename,
                        text=text,
                    ),
                ]

                for attempt in attempts:

                    try:

                        result = attempt()

                        category = self._extract_category(result)

                        if category:
                            return category

                    except Exception:
                        continue

        # Fallback classifier.
        from backend.services.screenshot_indexer import (
            ScreenshotClassifier,
        )

        classifier = ScreenshotClassifier()

        return classifier.classify(text)

    @staticmethod
    def _extract_category(
        result: Any,
    ) -> str:
        """
        Normalize classifier output.
        """

        if isinstance(
            result,
            str,
        ):
            return result

        if isinstance(
            result,
            dict,
        ):

            for key in (
                "category",
                "class",
                "label",
                "type",
            ):

                value = result.get(key)

                if value:
                    return str(value)

        return ""

    # =========================================================================
    # ENTITIES
    # =========================================================================

    def extract_entities(
        self,
        text: str,
    ) -> List[str]:
        """
        Extract entities using the existing service when possible.
        """

        if self.entity_service is not None:

            methods = [
                "extract",
                "extract_entities",
                "process",
                "run",
            ]

            for method_name in methods:

                method = getattr(
                    self.entity_service,
                    method_name,
                    None,
                )

                if not callable(method):
                    continue

                for argument in (
                    text,
                    {"text": text},
                ):

                    try:

                        result = method(argument)

                        entities = self._normalize_entities(result)

                        if entities:
                            return entities

                    except Exception:
                        continue

        # Fallback.
        from backend.services.screenshot_indexer import (
            ScreenshotClassifier,
        )

        classifier = ScreenshotClassifier()

        return classifier.extract_entities(text)

    @staticmethod
    def _normalize_entities(
        result: Any,
    ) -> List[str]:
        """
        Normalize entity service output.
        """

        if isinstance(
            result,
            list,
        ):

            return [str(item) for item in result[:20]]

        if isinstance(
            result,
            dict,
        ):

            for key in (
                "entities",
                "items",
                "results",
            ):

                value = result.get(key)

                if isinstance(
                    value,
                    list,
                ):

                    return [str(item) for item in value[:20]]

        return []

    # =========================================================================
    # SUMMARY
    # =========================================================================

    def create_summary(
        self,
        text: str,
        category: str,
        filename: str,
    ) -> str:
        """
        Generate a compact summary.

        The existing AI parser is intentionally not mandatory for the MVP.
        """

        # Try existing AI parser.
        try:

            from backend.services.ai_parser import (
                AIParser,
            )

            parser = AIParser()

            for method_name in (
                "summarize",
                "parse",
                "analyze",
            ):

                method = getattr(
                    parser,
                    method_name,
                    None,
                )

                if not callable(method):
                    continue

                for argument in (
                    text,
                    {
                        "text": text,
                        "category": category,
                    },
                ):

                    try:

                        result = method(argument)

                        summary = self._extract_summary(result)

                        if summary:
                            return summary

                    except Exception:
                        continue

        except Exception:
            pass

        # Reliable fallback.
        if text:

            lines = [line.strip() for line in text.splitlines() if line.strip()]

            if lines:

                first = lines[0]

                if len(first) > 150:
                    first = first[:147] + "..."

                return f"{category.title()} screenshot: " f"{first}"

        return f"{category.title()} screenshot: " f"{filename}"

    @staticmethod
    def _extract_summary(
        result: Any,
    ) -> str:

        if isinstance(
            result,
            str,
        ):
            return result.strip()

        if isinstance(
            result,
            dict,
        ):

            for key in (
                "summary",
                "description",
                "text",
            ):

                value = result.get(key)

                if value:
                    return str(value).strip()

        return ""

    # =========================================================================
    # SEARCH DOCUMENT
    # =========================================================================

    def build_search_document(
        self,
        memory: Dict[str, Any],
    ) -> str:
        """
        Build the text representation used for semantic search.

        This is one of the most important pieces of MemoryOS.

        We don't embed only OCR.

        We embed:

            OCR
            +
            category
            +
            summary
            +
            entities
            +
            keywords
        """

        entities = memory.get(
            "entities",
            [],
        )

        keywords = memory.get(
            "keywords",
            [],
        )

        return "\n".join(
            [
                f"Category: {memory.get('category', '')}",
                f"Summary: {memory.get('summary', '')}",
                (
                    "Entities: "
                    + ", ".join(
                        map(
                            str,
                            entities,
                        )
                    )
                ),
                (
                    "Keywords: "
                    + ", ".join(
                        map(
                            str,
                            keywords,
                        )
                    )
                ),
                (
                    "Visible screenshot text: "
                    + memory.get(
                        "ocr_text",
                        "",
                    )
                ),
            ]
        )

    # =========================================================================
    # EMBEDDING
    # =========================================================================

    def create_embedding(
        self,
        document: str,
    ) -> Any:
        """
        Create semantic embedding.

        Existing embedding service first.
        """

        if self.embedding_service is not None:

            methods = [
                "embed",
                "encode",
                "create_embedding",
                "get_embedding",
            ]

            for method_name in methods:

                method = getattr(
                    self.embedding_service,
                    method_name,
                    None,
                )

                if not callable(method):
                    continue

                try:

                    result = method(document)

                    if result is not None:
                        return result

                except Exception:
                    continue

        # Fallback to the indexer's embedding engine.
        from backend.services.screenshot_indexer import (
            EmbeddingEngine,
        )

        engine = EmbeddingEngine()

        return engine.encode([document])[0]

    # =========================================================================
    # VECTOR STORE
    # =========================================================================

    def store_embedding(
        self,
        embedding: Any,
        memory: Dict[str, Any],
    ) -> bool:
        """
        Store embedding in the existing vector store when possible.

        Returns True if successful.
        """

        if self.vector_store is None:
            return False

        methods = [
            "add",
            "insert",
            "upsert",
            "store",
            "add_vector",
        ]

        for method_name in methods:

            method = getattr(
                self.vector_store,
                method_name,
                None,
            )

            if not callable(method):
                continue

            attempts = [
                lambda: method(
                    embedding,
                    memory,
                ),
                lambda: method(
                    vector=embedding,
                    metadata=memory,
                ),
                lambda: method(
                    embedding=embedding,
                    metadata=memory,
                ),
            ]

            for attempt in attempts:

                try:

                    attempt()

                    return True

                except Exception:
                    continue

        return False

    # =========================================================================
    # PROCESS ONE SCREENSHOT
    # =========================================================================

    def process(
        self,
        screenshot: ScreenshotFile,
    ) -> Dict[str, Any]:
        """
        Process one screenshot from start to finish.
        """

        started = datetime.now()

        absolute_path = PROJECT_ROOT / screenshot.path

        logger.info(
            "Processing: %s",
            screenshot.filename,
        )

        # -------------------------------------------------------------
        # OCR
        # -------------------------------------------------------------

        ocr_text = self.extract_text(absolute_path)

        # -------------------------------------------------------------
        # Classification
        # -------------------------------------------------------------

        category = self.classify(
            text=ocr_text,
            filename=screenshot.filename,
        )

        # -------------------------------------------------------------
        # Entities
        # -------------------------------------------------------------

        entities = self.extract_entities(ocr_text)

        # -------------------------------------------------------------
        # Keywords
        # -------------------------------------------------------------

        keywords = extract_keywords(ocr_text)

        # -------------------------------------------------------------
        # Summary
        # -------------------------------------------------------------

        summary = self.create_summary(
            text=ocr_text,
            category=category,
            filename=screenshot.filename,
        )

        # -------------------------------------------------------------
        # Memory object
        # -------------------------------------------------------------

        memory = {
            **asdict(screenshot),
            "category": category,
            "ocr_text": ocr_text,
            "entities": entities,
            "keywords": keywords,
            "summary": summary,
            "indexed_at": datetime.now().isoformat(),
            "processing_time_ms": int(
                (datetime.now() - started).total_seconds() * 1000
            ),
            "status": "processed",
        }

        # -------------------------------------------------------------
        # Semantic document
        # -------------------------------------------------------------

        search_document = self.build_search_document(memory)

        memory["search_document"] = search_document

        # -------------------------------------------------------------
        # Embedding
        # -------------------------------------------------------------

        try:

            embedding = self.create_embedding(search_document)

            memory["embedding_created"] = True

            # ---------------------------------------------------------
            # Vector store
            # ---------------------------------------------------------

            stored = self.store_embedding(
                embedding,
                memory,
            )

            memory["vector_stored"] = stored

        except Exception as exc:

            logger.warning(
                "Embedding failed for %s: %s",
                screenshot.filename,
                exc,
            )

            memory["embedding_created"] = False

            memory["vector_stored"] = False

        # -------------------------------------------------------------
        # Save metadata
        # -------------------------------------------------------------

        self.memories[screenshot.id] = memory

        self._save_memories()

        logger.info(
            "Processed %s | category=%s | OCR=%s chars",
            screenshot.filename,
            category,
            len(ocr_text),
        )

        return memory

    # =========================================================================
    # PROCESS ALL NEW SCREENSHOTS
    # =========================================================================

    def process_new(
        self,
    ) -> Dict[str, Any]:
        """
        Scan the folder and process new screenshots.
        """

        started = datetime.now()

        logger.info("=" * 72)

        logger.info("MEMORYOS AUTOMATIC SCREENSHOT PROCESSING")

        logger.info("=" * 72)

        screenshots = self.scanner.scan_new()

        logger.info(
            "New screenshots found: %s",
            len(screenshots),
        )

        processed = 0
        failed = 0

        results = []

        for index, screenshot in enumerate(
            screenshots,
            start=1,
        ):

            logger.info(
                "[%s/%s] %s",
                index,
                len(screenshots),
                screenshot.filename,
            )

            try:

                memory = self.process(screenshot)

                results.append(memory)

                processed += 1

            except Exception as exc:

                failed += 1

                logger.exception(
                    "Failed processing %s: %s",
                    screenshot.filename,
                    exc,
                )

        # Important:
        # Save processed hashes back into scanner state.
        self.scanner._save_state()

        duration = (datetime.now() - started).total_seconds()

        result = {
            "status": "complete",
            "discovered": len(screenshots),
            "processed": processed,
            "failed": failed,
            "total_memories": len(self.memories),
            "duration_seconds": round(
                duration,
                2,
            ),
            "updated_at": datetime.now().isoformat(),
        }

        logger.info("=" * 72)

        logger.info("PROCESSING COMPLETE")

        logger.info(
            "New screenshots : %s",
            result["discovered"],
        )

        logger.info(
            "Processed        : %s",
            result["processed"],
        )

        logger.info(
            "Failed           : %s",
            result["failed"],
        )

        logger.info(
            "Total memories   : %s",
            result["total_memories"],
        )

        logger.info(
            "Duration         : %ss",
            result["duration_seconds"],
        )

        logger.info("=" * 72)

        return result


# ============================================================================
# PUBLIC FUNCTION
# ============================================================================


def process_screenshots(
    folder: Optional[str | Path] = None,
) -> Dict[str, Any]:
    """
    Convenience function for scripts/API.
    """

    processor = ScreenshotProcessor(screenshot_folder=folder)

    return processor.process_new()


# ============================================================================
# COMMAND LINE
# ============================================================================


def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(
        description=("MemoryOS automatic screenshot processor")
    )

    parser.add_argument(
        "--folder",
        type=str,
        default=str(DEFAULT_SCREENSHOT_FOLDER),
        help=("Folder containing screenshots."),
    )

    args = parser.parse_args()

    result = process_screenshots(folder=args.folder)

    print()
    print("=" * 64)
    print("MEMORYOS SCREENSHOT PROCESSOR")
    print("=" * 64)
    print(
        json.dumps(
            result,
            indent=2,
        )
    )
    print("=" * 64)


if __name__ == "__main__":
    main()
