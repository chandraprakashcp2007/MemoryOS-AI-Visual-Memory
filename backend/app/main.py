"""
MemoryOS - FastAPI Application
==============================

Main backend entry point.

MemoryOS automatically turns a screenshot collection into
searchable visual memories.

Startup flow:

    FastAPI starts
        ↓
    Screenshot watcher starts
        ↓
    data/screenshots/ is scanned
        ↓
    New screenshots are processed
        ↓
    OCR → AI → entities → summary → embedding
        ↓
    Vector store
        ↓
    /search

Run from project root:

    python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"

SCREENSHOT_FOLDER = DATA_DIR / "screenshots"

MEMORY_INDEX_FOLDER = DATA_DIR / "memory_index"

LOG_FOLDER = DATA_DIR / "logs"


# ============================================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================================

for directory in (
    DATA_DIR,
    SCREENSHOT_FOLDER,
    MEMORY_INDEX_FOLDER,
    LOG_FOLDER,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s | " "%(levelname)s | " "%(name)s | " "%(message)s"),
)

logger = logging.getLogger("memoryos")


# ============================================================================
# AUTOMATIC SCREENSHOT WATCHER
# ============================================================================

try:

    from backend.services.watchers.automatic_memory_watcher import (
        AutomaticMemoryWatcher,
    )

    memory_watcher = AutomaticMemoryWatcher(
        screenshot_folder=SCREENSHOT_FOLDER,
        interval=5,
    )

    WATCHER_AVAILABLE = True

except Exception as exc:

    memory_watcher = None

    WATCHER_AVAILABLE = False

    logger.warning(
        "Automatic screenshot watcher unavailable: %s",
        exc,
    )


# ============================================================================
# APPLICATION LIFESPAN
# ============================================================================


@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    MemoryOS application lifecycle.

    Startup:
        - prepare directories
        - start automatic screenshot watcher
        - process existing screenshots

    Shutdown:
        - stop watcher cleanly
    """

    logger.info("=" * 72)

    logger.info("MEMORYOS STARTING")

    logger.info("=" * 72)

    logger.info(
        "Project root: %s",
        PROJECT_ROOT,
    )

    logger.info(
        "Screenshot folder: %s",
        SCREENSHOT_FOLDER,
    )

    # ------------------------------------------------------------------------
    # START AUTOMATIC MEMORY ENGINE
    # ------------------------------------------------------------------------

    if WATCHER_AVAILABLE and memory_watcher is not None:

        try:

            memory_watcher.start(run_initial_scan=True)

            logger.info("✓ Automatic screenshot indexing ENABLED")

        except Exception as exc:

            logger.exception(
                "Could not start screenshot watcher: %s",
                exc,
            )

    else:

        logger.warning("⚠ Automatic screenshot watcher is disabled.")

    logger.info("MemoryOS backend is ready.")

    logger.info("=" * 72)

    try:

        yield

    finally:

        # --------------------------------------------------------------------
        # SHUTDOWN
        # --------------------------------------------------------------------

        logger.info("=" * 72)

        logger.info("MEMORYOS SHUTTING DOWN")

        logger.info("=" * 72)

        if WATCHER_AVAILABLE and memory_watcher is not None:

            try:

                memory_watcher.stop()

                logger.info("✓ Screenshot watcher stopped.")

            except Exception as exc:

                logger.warning(
                    "Watcher shutdown warning: %s",
                    exc,
                )

        logger.info("MemoryOS shutdown complete.")


# ============================================================================
# FASTAPI APPLICATION
# ============================================================================


app = FastAPI(
    title="MemoryOS",
    description=(
        "AI-powered visual memory and semantic screenshot " "retrieval engine."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================================
# CORS
# ============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",
    ],
    allow_credentials=True,
    allow_methods=[
        "*",
    ],
    allow_headers=[
        "*",
    ],
)


# ============================================================================
# ROUTERS
# ============================================================================

# Import routers safely so one optional component does not prevent
# the entire backend from starting.

try:

    from backend.api.health import (
        router as health_router,
    )

    app.include_router(health_router)

    logger.info("✓ Health router registered")

except Exception as exc:

    logger.warning(
        "Health router unavailable: %s",
        exc,
    )


try:

    from backend.api.stats import (
        router as stats_router,
    )

    app.include_router(stats_router)

    logger.info("✓ Statistics router registered")

except Exception as exc:

    logger.warning(
        "Statistics router unavailable: %s",
        exc,
    )


try:

    from backend.api.search import (
        router as search_router,
    )

    app.include_router(search_router)

    logger.info("✓ Search router registered")

except Exception as exc:

    logger.warning(
        "Search router unavailable: %s",
        exc,
    )


try:

    from backend.api.memories import (
        router as memories_router,
    )

    app.include_router(memories_router)

    logger.info("✓ Memories router registered")

except Exception as exc:

    logger.warning(
        "Memories router unavailable: %s",
        exc,
    )


try:

    from backend.api.upload import (
        router as upload_router,
    )

    app.include_router(upload_router)

    logger.info("✓ Upload router registered")

except Exception as exc:

    logger.warning(
        "Upload router unavailable: %s",
        exc,
    )


try:

    from backend.api.ocr import (
        router as ocr_router,
    )

    app.include_router(ocr_router)

    logger.info("✓ OCR router registered")

except Exception as exc:

    logger.warning(
        "OCR router unavailable: %s",
        exc,
    )


# ============================================================================
# ROOT ENDPOINT
# ============================================================================


@app.get(
    "/",
    tags=["System"],
)
async def root():
    """
    MemoryOS API information.
    """

    searchable_memories = 0

    if memory_watcher is not None and hasattr(
        memory_watcher,
        "processor",
    ):

        try:

            searchable_memories = len(memory_watcher.processor.memories)

        except Exception:
            searchable_memories = 0

    return {
        "name": "MemoryOS",
        "version": "1.0.0",
        "status": "running",
        "description": (
            "AI-powered visual memory " "and semantic screenshot retrieval."
        ),
        "problem": ("Find anything inside thousands " "of screenshots by meaning."),
        "automatic_indexing": (WATCHER_AVAILABLE),
        "screenshot_folder": str(SCREENSHOT_FOLDER),
        "searchable_memories": (searchable_memories),
        "docs": "/docs",
        "search": "/search",
    }


# ============================================================================
# HEALTH ENDPOINT
# ============================================================================


@app.get(
    "/health",
    tags=["System"],
)
async def health():
    """
    Simple backend health check.
    """

    watcher_running = False

    indexed_memories = 0

    if memory_watcher is not None:

        try:

            status = memory_watcher.status()

            watcher_running = bool(
                status.get(
                    "running",
                    False,
                )
            )

            indexed_memories = int(
                status.get(
                    "indexed_memories",
                    0,
                )
            )

        except Exception:
            pass

    return {
        "status": "healthy",
        "service": "MemoryOS",
        "automatic_indexing": (WATCHER_AVAILABLE),
        "watcher_running": (watcher_running),
        "indexed_memories": (indexed_memories),
        "screenshot_directory": (str(SCREENSHOT_FOLDER)),
    }


# ============================================================================
# MEMORY ENGINE STATUS
# ============================================================================


@app.get(
    "/memory-engine",
    tags=["System"],
)
async def memory_engine_status():
    """
    Show the status of the automatic visual-memory engine.
    """

    if memory_watcher is None or not WATCHER_AVAILABLE:

        return {
            "status": "unavailable",
            "automatic_indexing": False,
        }

    try:

        status = memory_watcher.status()

        return {
            "status": ("running" if status.get("running") else "stopped"),
            "automatic_indexing": True,
            **status,
        }

    except Exception as exc:

        return {
            "status": "error",
            "automatic_indexing": True,
            "error": str(exc),
        }


# ============================================================================
# MANUAL SCAN
# ============================================================================


@app.post(
    "/memory-engine/scan",
    tags=["System"],
)
async def manual_scan():
    """
    Manually trigger one screenshot indexing pass.

    Useful for:
        - testing
        - hackathon demo
        - recovering after adding many screenshots
    """

    if memory_watcher is None or not WATCHER_AVAILABLE:

        return {
            "status": "unavailable",
            "message": ("Automatic memory watcher " "is not available."),
        }

    try:

        result = memory_watcher.scan_once()

        return {
            "status": "complete",
            **result,
        }

    except Exception as exc:

        logger.exception("Manual memory scan failed.")

        return {
            "status": "error",
            "error": str(exc),
        }


# ============================================================================
# APPLICATION INFO
# ============================================================================


@app.get(
    "/info",
    tags=["System"],
)
async def application_info():
    """
    Detailed MemoryOS configuration.
    """

    watcher_status = {}

    if memory_watcher is not None:

        try:

            watcher_status = memory_watcher.status()

        except Exception:
            watcher_status = {}

    return {
        "application": {
            "name": "MemoryOS",
            "version": "1.0.0",
        },
        "pipeline": [
            "Screenshot discovery",
            "Image validation",
            "OCR",
            "Text cleaning",
            "AI understanding",
            "Classification",
            "Entity extraction",
            "Summary",
            "Embedding",
            "Vector indexing",
            "Semantic search",
            "Explainable retrieval",
        ],
        "storage": {
            "screenshots": str(SCREENSHOT_FOLDER),
            "memory_index": str(MEMORY_INDEX_FOLDER),
        },
        "watcher": watcher_status,
        "endpoints": {
            "search": "/search",
            "health": "/health",
            "memory_engine": ("/memory-engine"),
            "manual_scan": ("/memory-engine/scan"),
            "docs": "/docs",
        },
    }


# ============================================================================
# LOCAL DEVELOPMENT
# ============================================================================


if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
