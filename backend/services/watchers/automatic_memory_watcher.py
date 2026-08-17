"""
MemoryOS - Automatic Memory Watcher
===================================

Watches the screenshot directory continuously.

When a new screenshot appears:

    screenshot
        ↓
    scanner
        ↓
    OCR
        ↓
    classification
        ↓
    entities
        ↓
    summary
        ↓
    embedding
        ↓
    vector store
        ↓
    searchable memory

This is the bridge between:
    "I have 4,000 screenshots"
and
    "MemoryOS automatically remembers them."
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

from backend.services.screenshot_processor import (
    ScreenshotProcessor,
)

logger = logging.getLogger("memoryos.automatic_watcher")


class AutomaticMemoryWatcher:
    """
    Lightweight polling watcher.

    No heavy infrastructure required for the hackathon.

    It checks the screenshot directory periodically and processes
    only screenshots that have not already been indexed.
    """

    SUPPORTED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".gif",
        ".tiff",
        ".tif",
    }

    def __init__(
        self,
        screenshot_folder: str | Path,
        interval: int = 5,
    ) -> None:

        self.screenshot_folder = Path(screenshot_folder).resolve()

        self.interval = max(
            1,
            interval,
        )

        self.processor = ScreenshotProcessor(screenshot_folder=(self.screenshot_folder))

        self.running = False

        self._thread: Optional[threading.Thread] = None

        self._stop_event = threading.Event()

    # ========================================================================
    # DIRECTORY
    # ========================================================================

    def ensure_directory(self) -> None:

        self.screenshot_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ========================================================================
    # COUNT
    # ========================================================================

    def count_screenshots(self) -> int:

        self.ensure_directory()

        count = 0

        for path in self.screenshot_folder.iterdir():

            if not path.is_file():
                continue

            if path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                count += 1

        return count

    # ========================================================================
    # ONE SCAN
    # ========================================================================

    def scan_once(self) -> dict:
        """
        Perform one automatic indexing pass.
        """

        logger.info("Checking screenshot folder...")

        try:

            result = self.processor.process_new()

            if result.get(
                "discovered",
                0,
            ):

                logger.info(
                    "MemoryOS indexed %s new screenshots.",
                    result["discovered"],
                )

            return result

        except Exception as exc:

            logger.exception(
                "Automatic indexing failed: %s",
                exc,
            )

            return {
                "status": "error",
                "error": str(exc),
                "discovered": 0,
                "processed": 0,
                "failed": 1,
            }

    # ========================================================================
    # BACKGROUND LOOP
    # ========================================================================

    def _watch_loop(self) -> None:

        logger.info("MemoryOS automatic watcher started.")

        logger.info(
            "Watching: %s",
            self.screenshot_folder,
        )

        while not self._stop_event.is_set():

            try:

                self.scan_once()

            except Exception as exc:

                logger.exception(
                    "Watcher cycle failed: %s",
                    exc,
                )

            self._stop_event.wait(self.interval)

        logger.info("MemoryOS automatic watcher stopped.")

    # ========================================================================
    # START
    # ========================================================================

    def start(
        self,
        run_initial_scan: bool = True,
    ) -> None:

        if self.running:
            logger.warning("Watcher is already running.")
            return

        self.ensure_directory()

        if run_initial_scan:

            logger.info("Running initial screenshot scan...")

            self.scan_once()

        self.running = True

        self._stop_event.clear()

        self._thread = threading.Thread(
            target=self._watch_loop,
            name="MemoryOSWatcher",
            daemon=True,
        )

        self._thread.start()

        logger.info("✓ Automatic memory watcher is running.")

    # ========================================================================
    # STOP
    # ========================================================================

    def stop(self) -> None:

        if not self.running:
            return

        logger.info("Stopping MemoryOS watcher...")

        self.running = False

        self._stop_event.set()

        if self._thread and self._thread.is_alive():

            self._thread.join(timeout=3)

        self._thread = None

        logger.info("✓ Watcher stopped.")

    # ========================================================================
    # STATUS
    # ========================================================================

    def status(self) -> dict:

        return {
            "running": self.running,
            "folder": str(self.screenshot_folder),
            "interval_seconds": (self.interval),
            "screenshots_on_disk": (self.count_screenshots()),
            "indexed_memories": (len(self.processor.memories)),
        }


# ============================================================================
# SIMPLE FACTORY
# ============================================================================


def create_memory_watcher(
    screenshot_folder: str | Path,
    interval: int = 5,
) -> AutomaticMemoryWatcher:

    return AutomaticMemoryWatcher(
        screenshot_folder=(screenshot_folder),
        interval=interval,
    )
