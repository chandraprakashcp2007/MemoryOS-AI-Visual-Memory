"""
MemoryOS - Automatic Screenshot Indexer
=======================================

Hackathon entry point.

Scans:
    data/screenshots/

Then automatically:

    discover
        ↓
    validate
        ↓
    deduplicate
        ↓
    OCR
        ↓
    classify
        ↓
    entities
        ↓
    summary
        ↓
    embedding
        ↓
    vector store

Run:

    python scripts/index_screenshots.py

Or:

    python scripts/index_screenshots.py --folder data/screenshots
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# ============================================================================
# PROJECT ROOT
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


# ============================================================================
# MEMORYOS IMPORT
# ============================================================================

from backend.services.screenshot_processor import (
    ScreenshotProcessor,
)

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s | " "%(levelname)s | " "%(message)s"),
)

logger = logging.getLogger("memoryos.indexer")


# ============================================================================
# DEFAULT FOLDER
# ============================================================================

DEFAULT_FOLDER = PROJECT_ROOT / "data" / "screenshots"


# ============================================================================
# DISPLAY
# ============================================================================


def print_banner() -> None:

    print()
    print("=" * 72)
    print("                    MEMORYOS")
    print("          VISUAL MEMORY INDEX ENGINE")
    print("=" * 72)
    print()
    print("Automatically turning screenshots into searchable memories.")
    print()


def print_stage(
    number: int,
    title: str,
) -> None:

    print()
    print(f"[{number}] {title}")
    print("-" * 60)


def progress_bar(
    current: int,
    total: int,
    width: int = 35,
) -> None:

    if total <= 0:
        return

    ratio = current / total

    filled = int(width * ratio)

    bar = "█" * filled + "░" * (width - filled)

    percent = int(ratio * 100)

    print(
        f"\r{bar} {percent:3d}% " f"({current}/{total})",
        end="",
        flush=True,
    )


# ============================================================================
# MAIN
# ============================================================================


def main() -> int:

    parser = argparse.ArgumentParser(
        description=("MemoryOS automatic screenshot indexing engine.")
    )

    parser.add_argument(
        "--folder",
        type=str,
        default=str(DEFAULT_FOLDER),
        help=("Folder containing screenshots."),
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=("Reprocess all screenshots instead " "of only new screenshots."),
    )

    args = parser.parse_args()

    screenshot_folder = Path(args.folder).resolve()

    print_banner()

    # ------------------------------------------------------------------
    # CHECK FOLDER
    # ------------------------------------------------------------------

    print_stage(
        1,
        "SCREENSHOT SOURCE",
    )

    print(f"Folder: {screenshot_folder}")

    screenshot_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # INITIALIZE PROCESSOR
    # ------------------------------------------------------------------

    print_stage(
        2,
        "INITIALIZING MEMORY ENGINE",
    )

    try:

        processor = ScreenshotProcessor(screenshot_folder=screenshot_folder)

    except Exception as exc:

        print()
        print("❌ Failed to initialize MemoryOS.")

        print(f"Error: {exc}")

        return 1

    print("✓ Scanner ready")

    print("✓ OCR pipeline ready")

    print("✓ AI pipeline ready")

    print("✓ Embedding pipeline ready")

    print("✓ Vector store ready")

    # ------------------------------------------------------------------
    # DISCOVER
    # ------------------------------------------------------------------

    print_stage(
        3,
        "DISCOVERING SCREENSHOTS",
    )

    start = time.perf_counter()

    try:

        if args.all:

            screenshots = processor.scanner.scan(include_duplicates=False)

        else:

            screenshots = processor.scanner.scan_new()

    except Exception as exc:

        print()
        print("❌ Screenshot discovery failed.")

        print(f"Error: {exc}")

        return 1

    discovery_time = time.perf_counter() - start

    print(f"✓ Found {len(screenshots):,} " f"screenshots")

    print(f"✓ Discovery completed in " f"{discovery_time:.2f}s")

    if not screenshots:

        print()
        print("🎉 No new screenshots need processing.")

        print()
        print("MemoryOS is already up to date.")

        return 0

    # ------------------------------------------------------------------
    # PROCESS
    # ------------------------------------------------------------------

    print_stage(
        4,
        "BUILDING VISUAL MEMORIES",
    )

    print(f"Processing {len(screenshots):,} screenshots...")

    print()

    processed = 0
    failed = 0

    processing_start = time.perf_counter()

    for index, screenshot in enumerate(
        screenshots,
        start=1,
    ):

        try:

            memory = processor.process(screenshot)

            processed += 1

            # Compact terminal feedback.
            category = memory.get(
                "category",
                "unclassified",
            )

            ocr_length = len(
                memory.get(
                    "ocr_text",
                    "",
                )
            )

            print(
                f"✓ {screenshot.filename[:45]:45} "
                f"| {category[:18]:18} "
                f"| OCR {ocr_length:5} chars"
            )

        except Exception as exc:

            failed += 1

            logger.error(
                "Failed: %s | %s",
                screenshot.filename,
                exc,
            )

            print(f"✗ {screenshot.filename[:45]:45} " f"| FAILED")

        progress_bar(
            index,
            len(screenshots),
        )

    print()

    processing_time = time.perf_counter() - processing_start

    # ------------------------------------------------------------------
    # SAVE STATE
    # ------------------------------------------------------------------

    print_stage(
        5,
        "FINALIZING INDEX",
    )

    try:

        processor.scanner._save_state()

        processor._save_memories()

        print("✓ Memory metadata saved")

        print("✓ Screenshot state saved")

    except Exception as exc:

        print(f"⚠ State save warning: {exc}")

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------

    total_time = time.perf_counter() - start

    print()
    print("=" * 72)
    print("                     COMPLETE")
    print("=" * 72)

    print()

    print(f"  Screenshots found : {len(screenshots):,}")

    print(f"  Successfully built: {processed:,}")

    print(f"  Failed            : {failed:,}")

    print(f"  Total memories    : " f"{len(processor.memories):,}")

    print(f"  Processing time   : " f"{processing_time:.2f}s")

    print(f"  Total time        : " f"{total_time:.2f}s")

    print()

    if failed == 0:

        print("🚀 ALL SCREENSHOTS ARE NOW MEMORYOS MEMORIES.")

        print()

        print("You can now search by meaning instead of filename.")

    else:

        print("⚠ Processing completed with some failures.")

    print()
    print("=" * 72)
    print()

    return 0 if failed == 0 else 1


# ============================================================================
# ENTRY POINT
# ============================================================================


if __name__ == "__main__":
    raise SystemExit(main())
