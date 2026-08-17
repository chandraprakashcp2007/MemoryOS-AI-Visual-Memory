"""
MemoryOS - Premium Screenshot Scanner
=====================================

Production-ready screenshot discovery engine for the MemoryOS MVP.

Responsibilities
----------------
1. Recursively discover supported images
2. Validate image integrity
3. Calculate SHA-256 hashes
4. Detect true duplicate images
5. Track previously indexed files
6. Support external screenshot folders
7. Store stable absolute paths
8. Support full rebuilds
9. Support incremental scanning
10. Persist scanner state safely

Hackathon principle
-------------------
The scanner must distinguish between:

    DUPLICATE IMAGE
        Same SHA-256 content

and

    PREVIOUSLY INDEXED IMAGE
        Same image already processed before

Those are NOT the same thing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from PIL import Image
except ImportError:
    Image = None


# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_SCREENSHOT_FOLDER = PROJECT_ROOT / "data" / "screenshots"

INDEX_FOLDER = PROJECT_ROOT / "data" / "faiss"

STATE_FILE = INDEX_FOLDER / "scanner_state.json"


# ============================================================================
# SUPPORTED FILES
# ============================================================================

SUPPORTED_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger("memoryos.screenshot_scanner")


# ============================================================================
# DATA MODEL
# ============================================================================


@dataclass
class ScreenshotFile:
    """
    Represents one discovered screenshot.
    """

    id: str

    filename: str

    path: str

    extension: str

    size_bytes: int

    width: int

    height: int

    sha256: str

    modified_at: str

    discovered_at: str

    duplicate: bool = False


# ============================================================================
# SCANNER
# ============================================================================


class ScreenshotScanner:
    """
    Premium MemoryOS screenshot discovery engine.

    Important:

    `duplicate`
        Means the image content is identical to another image.

    `previously_indexed`
        Means MemoryOS has already seen the image before.

    We intentionally keep these concepts separate.
    """

    def __init__(
        self,
        screenshot_folder: Optional[str | Path] = None,
    ) -> None:

        self.screenshot_folder = (
            Path(screenshot_folder or DEFAULT_SCREENSHOT_FOLDER).expanduser().resolve()
        )

        # We intentionally allow folders outside PROJECT_ROOT.
        self.screenshot_folder.mkdir(
            parents=True,
            exist_ok=True,
        )

        INDEX_FOLDER.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.state = self._load_state()

    # ========================================================================
    # STATE
    # ========================================================================

    def _empty_state(self) -> Dict:
        return {
            "version": 2,
            "files": {},
            "hashes": {},
            "updated_at": None,
        }

    def _load_state(self) -> Dict:

        if not STATE_FILE.exists():
            return self._empty_state()

        try:

            with STATE_FILE.open(
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            if not isinstance(data, dict):
                raise ValueError("Scanner state must be a JSON object.")

            data.setdefault("version", 2)
            data.setdefault("files", {})
            data.setdefault("hashes", {})
            data.setdefault("updated_at", None)

            return data

        except Exception as exc:

            logger.warning(
                "Could not load scanner state: %s",
                exc,
            )

            return self._empty_state()

    def _save_state(self) -> None:

        self.state["version"] = 2

        self.state["updated_at"] = datetime.now().isoformat()

        INDEX_FOLDER.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Atomic write.
        #
        # This prevents a partially-written JSON state file if the process
        # gets interrupted.
        fd, temp_name = tempfile.mkstemp(
            prefix="scanner_state_",
            suffix=".tmp",
            dir=str(INDEX_FOLDER),
        )

        try:

            with os.fdopen(
                fd,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    self.state,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

                file.flush()

                os.fsync(file.fileno())

            Path(temp_name).replace(STATE_FILE)

        finally:

            temp_path = Path(temp_name)

            if temp_path.exists():

                try:
                    temp_path.unlink()
                except OSError:
                    pass

    # ========================================================================
    # RESET
    # ========================================================================

    def reset_state(self) -> None:
        """
        Completely reset scanner state.

        This does NOT delete screenshots.

        It only forgets which screenshots have previously been discovered.
        """

        self.state = self._empty_state()

        if STATE_FILE.exists():

            try:
                STATE_FILE.unlink()
            except OSError as exc:

                logger.warning(
                    "Could not remove scanner state: %s",
                    exc,
                )

    # ========================================================================
    # FILE DISCOVERY
    # ========================================================================

    def discover_files(self) -> List[Path]:
        """
        Recursively discover supported image files.
        """

        files: List[Path] = []

        if not self.screenshot_folder.exists():

            logger.warning(
                "Screenshot folder does not exist: %s",
                self.screenshot_folder,
            )

            return files

        for path in self.screenshot_folder.rglob("*"):

            try:

                if not path.is_file():
                    continue

                if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue

                files.append(path.resolve())

            except OSError as exc:

                logger.warning(
                    "Could not inspect filesystem entry %s: %s",
                    path,
                    exc,
                )

        # Stable ordering.
        files.sort(
            key=lambda item: str(item.relative_to(self.screenshot_folder)).lower()
        )

        return files

    # ========================================================================
    # HASHING
    # ========================================================================

    @staticmethod
    def calculate_hash(
        path: Path,
        chunk_size: int = 1024 * 1024,
    ) -> str:

        sha256 = hashlib.sha256()

        with path.open("rb") as file:

            while True:

                chunk = file.read(chunk_size)

                if not chunk:
                    break

                sha256.update(chunk)

        return sha256.hexdigest()

    # ========================================================================
    # IMAGE VALIDATION
    # ========================================================================

    @staticmethod
    def validate_image(
        path: Path,
    ) -> Tuple[int, int]:

        if Image is None:

            raise RuntimeError(
                "Pillow is required for screenshot scanning.\n"
                "Install it with:\n"
                "pip install pillow"
            )

        try:

            # Integrity check.
            with Image.open(path) as image:

                image.verify()

            # Re-open after verify().
            with Image.open(path) as image:

                width, height = image.size

            if width <= 0 or height <= 0:

                raise ValueError("Image has invalid dimensions.")

            return width, height

        except Exception as exc:

            raise ValueError(f"Invalid image: {exc}") from exc

    # ========================================================================
    # ID
    # ========================================================================

    @staticmethod
    def create_id(
        sha256: str,
    ) -> str:

        return "mem_" + sha256[:16]

    # ========================================================================
    # PATH REPRESENTATION
    # ========================================================================

    @staticmethod
    def serialize_path(
        path: Path,
    ) -> str:
        """
        Store an absolute normalized path.

        This is critical because the screenshot folder can live outside
        PROJECT_ROOT.

        Example:

            C:/Users/GODWIN/Pictures/Screenshots/Screenshot (10).png

        instead of incorrectly assuming:

            PROJECT_ROOT/Screenshot (10).png
        """

        return str(path.resolve())

    # ========================================================================
    # BUILD RECORD
    # ========================================================================

    def build_record(
        self,
        path: Path,
        sha256: str,
        width: int,
        height: int,
        duplicate: bool,
    ) -> ScreenshotFile:

        stat = path.stat()

        return ScreenshotFile(
            id=self.create_id(sha256),
            filename=path.name,
            path=self.serialize_path(path),
            extension=path.suffix.lower(),
            size_bytes=stat.st_size,
            width=width,
            height=height,
            sha256=sha256,
            modified_at=datetime.fromtimestamp(stat.st_mtime).isoformat(),
            discovered_at=datetime.now().isoformat(),
            duplicate=duplicate,
        )

    # ========================================================================
    # SINGLE FILE
    # ========================================================================

    def inspect(
        self,
        path: Path,
        known_scan_hashes: Optional[Set[str]] = None,
    ) -> Optional[ScreenshotFile]:

        try:

            width, height = self.validate_image(path)

            sha256 = self.calculate_hash(path)

            duplicate = False

            if known_scan_hashes is not None:

                duplicate = sha256 in known_scan_hashes

            return self.build_record(
                path=path,
                sha256=sha256,
                width=width,
                height=height,
                duplicate=duplicate,
            )

        except Exception as exc:

            logger.warning(
                "Skipping invalid screenshot %s: %s",
                path.name,
                exc,
            )

            return None

    # ========================================================================
    # FULL SCAN
    # ========================================================================

    def scan(
        self,
        include_duplicates: bool = False,
        save_state: bool = True,
        force_rebuild: bool = False,
    ) -> List[ScreenshotFile]:
        """
        Scan the entire screenshot folder.

        Parameters
        ----------
        include_duplicates:
            Include true duplicate images.

        save_state:
            Persist scanner state.

        force_rebuild:
            Ignore previous indexing state.

        IMPORTANT
        ---------
        Previous indexing is NOT considered duplication.

        Only identical SHA-256 content is duplication.
        """

        logger.info(
            "Scanning screenshot folder: %s",
            self.screenshot_folder,
        )

        files = self.discover_files()

        logger.info(
            "Found %s image files.",
            len(files),
        )

        results: List[ScreenshotFile] = []

        # Only hashes from THIS scan are used to detect true duplicates.
        seen_hashes: Set[str] = set()

        duplicates = 0

        invalid = 0

        previously_indexed = 0

        for position, path in enumerate(
            files,
            start=1,
        ):

            logger.info(
                "[%s/%s] Inspecting %s",
                position,
                len(files),
                path.name,
            )

            record = self.inspect(
                path=path,
                known_scan_hashes=seen_hashes,
            )

            if record is None:

                invalid += 1

                continue

            # --------------------------------------------------------------
            # TRUE DUPLICATE
            # --------------------------------------------------------------

            if record.sha256 in seen_hashes:

                record.duplicate = True

                duplicates += 1

            else:

                seen_hashes.add(record.sha256)

            # --------------------------------------------------------------
            # PREVIOUSLY INDEXED
            # --------------------------------------------------------------

            if not force_rebuild:

                if record.sha256 in self.state["hashes"]:

                    previously_indexed += 1

            # --------------------------------------------------------------
            # SAVE CURRENT FILE STATE
            # --------------------------------------------------------------

            self.state["files"][record.path] = asdict(record)

            # Do not let a previous run make this image a duplicate.
            self.state["hashes"][record.sha256] = record.id

            # --------------------------------------------------------------
            # TRUE DUPLICATE FILTER
            # --------------------------------------------------------------

            if record.duplicate and not include_duplicates:

                continue

            results.append(record)

        if save_state:

            self._save_state()

        logger.info("Scan complete.")

        logger.info(
            "Total files: %s",
            len(files),
        )

        logger.info(
            "Unique screenshots: %s",
            len(results),
        )

        logger.info(
            "Duplicates: %s",
            duplicates,
        )

        logger.info(
            "Previously indexed: %s",
            previously_indexed,
        )

        logger.info(
            "Invalid files: %s",
            invalid,
        )

        return results

    # ========================================================================
    # NEW SCREENSHOTS ONLY
    # ========================================================================

    def scan_new(
        self,
    ) -> List[ScreenshotFile]:
        """
        Return screenshots that have not previously been indexed.
        """

        files = self.discover_files()

        new_records: List[ScreenshotFile] = []

        known_hashes = set(self.state.get("hashes", {}).keys())

        current_hashes: Set[str] = set()

        for path in files:

            try:

                sha256 = self.calculate_hash(path)

                # Already processed previously.
                if sha256 in known_hashes:

                    continue

                # Duplicate inside this scan.
                if sha256 in current_hashes:

                    continue

                width, height = self.validate_image(path)

                record = self.build_record(
                    path=path,
                    sha256=sha256,
                    width=width,
                    height=height,
                    duplicate=False,
                )

                current_hashes.add(sha256)

                new_records.append(record)

            except Exception as exc:

                logger.warning(
                    "Could not inspect %s: %s",
                    path.name,
                    exc,
                )

        return new_records

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def statistics(self) -> Dict:

        files = self.state.get(
            "files",
            {},
        )

        hashes = self.state.get(
            "hashes",
            {},
        )

        categories: Dict[str, int] = {}

        for record in files.values():

            category = record.get(
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
            "screenshot_folder": str(self.screenshot_folder),
            "files_discovered": len(files),
            "unique_images": len(hashes),
            "categories": categories,
            "updated_at": self.state.get("updated_at"),
        }


# ============================================================================
# PUBLIC HELPERS
# ============================================================================


def scan_screenshots(
    folder: Optional[str | Path] = None,
) -> List[Dict]:

    scanner = ScreenshotScanner(screenshot_folder=folder)

    records = scanner.scan()

    return [asdict(record) for record in records]


def find_new_screenshots(
    folder: Optional[str | Path] = None,
) -> List[Dict]:

    scanner = ScreenshotScanner(screenshot_folder=folder)

    records = scanner.scan_new()

    return [asdict(record) for record in records]


# ============================================================================
# COMMAND LINE
# ============================================================================


def main() -> None:

    import argparse

    parser = argparse.ArgumentParser(
        description=("MemoryOS Premium Screenshot Scanner")
    )

    parser.add_argument(
        "--folder",
        type=str,
        default=str(DEFAULT_SCREENSHOT_FOLDER),
        help="Folder containing screenshots.",
    )

    parser.add_argument(
        "--include-duplicates",
        action="store_true",
        help="Include true duplicate images.",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset scanner state before scanning.",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Force a full scan/rebuild. " "Previously indexed screenshots are allowed."
        ),
    )

    args = parser.parse_args()

    scanner = ScreenshotScanner(screenshot_folder=args.folder)

    if args.reset:

        print("Resetting scanner state...")

        scanner.reset_state()

    records = scanner.scan(
        include_duplicates=args.include_duplicates,
        force_rebuild=args.all,
    )

    print()
    print("=" * 70)
    print("MEMORYOS SCREENSHOT SCANNER")
    print("=" * 70)

    print(f"Folder       : {scanner.screenshot_folder}")

    print(f"Screenshots  : {len(records)}")

    print(f"State file   : {STATE_FILE}")

    print("=" * 70)

    if records:

        print()
        print("First screenshots:")

        for record in records[:10]:

            print(f"  ✓ {record.filename}" f"  [{record.width}x{record.height}]")

    print()


if __name__ == "__main__":

    main()
