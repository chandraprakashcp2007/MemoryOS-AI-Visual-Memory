"""Resumable OCR + visual-search backfill for existing MemoryOS memories.

Run from the project root:
    .venv\\Scripts\\python scripts/backfill_visual_index.py

Use --force only when you intentionally want to reanalyse completed records.
The script never moves/replaces images and preserves every existing memory ID.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
MEMORY_FILE = PROJECT_ROOT / "data" / "memory_index" / "memories.json"
STATE_FILE = PROJECT_ROOT / "data" / "backfill" / "visual_index_state.json"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def backfill(*, force: bool = False, limit: int | None = None) -> dict[str, int]:
    from backend.api.upload import persist_search_memory, resolve_memory_image
    from backend.services.processing_service import process_image

    memories = _read_json(MEMORY_FILE)
    state = _read_json(STATE_FILE)
    totals = {"processed": 0, "skipped": 0, "missing": 0, "degraded": 0, "failed": 0}

    for memory_id, old_memory in memories.items():
        if limit is not None and totals["processed"] + totals["degraded"] + totals["failed"] >= limit:
            break
        if not isinstance(old_memory, dict):
            totals["failed"] += 1
            continue

        prior = state.get(memory_id, {})
        if not force and prior.get("status") == "completed":
            totals["skipped"] += 1
            continue

        source = resolve_memory_image(memory_id)
        if source is None or not source.is_file():
            state[memory_id] = {"status": "missing", "updated_at": datetime.now(timezone.utc).isoformat()}
            totals["missing"] += 1
            _write_json(STATE_FILE, state)
            continue

        metadata = {
            "memory_id": memory_id,
            "filename": str(old_memory.get("filename") or source.name),
            "original_path": str(source.resolve()),
            "image_hash": str(old_memory.get("sha256") or old_memory.get("image_hash") or ""),
            "source": "backfill",
        }
        try:
            result = process_image(
                source,
                source_name=metadata["filename"],
                metadata=metadata,
                index=True,
                allow_partial_success=True,
            ).to_dict()
            persist_search_memory(memory_id=memory_id, metadata=metadata, processing=result)

            vision = result.get("vision") or {}
            visual_ok = isinstance(vision, dict) and vision.get("success") is True
            status = "completed" if result.get("success") and visual_ok else "degraded"
            state[memory_id] = {
                "status": status,
                "source_sha256": result.get("sha256"),
                "vision_error": vision.get("error") if isinstance(vision, dict) else None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            totals["processed" if status == "completed" else "degraded"] += 1
        except Exception as exc:
            state[memory_id] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            totals["failed"] += 1
        finally:
            _write_json(STATE_FILE, state)

    return totals


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill OCR and visual semantic indexing for existing memories.")
    parser.add_argument("--force", action="store_true", help="Reanalyse records already marked completed.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum records to process in this run.")
    args = parser.parse_args()
    print(json.dumps(backfill(force=args.force, limit=args.limit), indent=2))
