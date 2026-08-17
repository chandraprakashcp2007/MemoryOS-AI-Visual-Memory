"""
MemoryOS - Health API
=====================

Production-oriented health and readiness endpoints.

Endpoints
---------
GET /health
    Lightweight liveness check.

GET /health/ready
    Dependency readiness check.

The health layer intentionally avoids performing expensive AI operations.
It should be safe to call frequently from:
- browser clients
- Docker/container orchestration
- monitoring systems
- local development
- hackathon demonstrations
"""

from __future__ import annotations

import importlib.util
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


# ============================================================================
# CONSTANTS
# ============================================================================

HEALTHY = "healthy"
DEGRADED = "degraded"
UNHEALTHY = "unhealthy"

READY = "ready"
NOT_READY = "not_ready"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATA_DIR = PROJECT_ROOT / "data"

DEFAULT_FAISS_DIR = DEFAULT_DATA_DIR / "faiss"

DEFAULT_DATABASE_DIR = DEFAULT_DATA_DIR / "database"


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class DependencyHealth(BaseModel):
    """
    Health state of one MemoryOS dependency.
    """

    model_config = ConfigDict(
        extra="allow",
    )

    status: str

    available: bool

    message: str

    latency_ms: float | None = None


class HealthResponse(BaseModel):
    """
    Liveness response.
    """

    status: str

    service: str

    version: str

    timestamp: str

    uptime_seconds: float

    checks: dict[str, DependencyHealth] = Field(default_factory=dict)


class ReadinessResponse(BaseModel):
    """
    Readiness response.
    """

    status: str

    service: str

    ready: bool

    timestamp: str

    checks: dict[str, DependencyHealth] = Field(default_factory=dict)

    failed_checks: list[str] = Field(default_factory=list)

    warnings: list[str] = Field(default_factory=list)


# ============================================================================
# PROCESS STATE
# ============================================================================

_PROCESS_STARTED_AT = time.monotonic()


# ============================================================================
# GENERAL HELPERS
# ============================================================================


def _timestamp() -> str:
    """
    Return UTC ISO-8601 timestamp.
    """
    return datetime.now(timezone.utc).isoformat()


def _version() -> str:
    """
    Read application version from environment.

    Configuration will eventually expose this centrally.
    """
    return os.getenv(
        "MEMORYOS_VERSION",
        "1.0.0",
    )


def _dependency(
    *,
    status: str,
    available: bool,
    message: str,
    latency_ms: float | None = None,
) -> DependencyHealth:
    """
    Create a normalized dependency health object.
    """
    return DependencyHealth(
        status=status,
        available=available,
        message=message,
        latency_ms=(round(latency_ms, 2) if latency_ms is not None else None),
    )


# ============================================================================
# PYTHON RUNTIME CHECK
# ============================================================================


def _check_python() -> DependencyHealth:
    """
    Check that the Python runtime is usable.
    """
    started = time.perf_counter()

    try:
        import sys

        version = sys.version_info

        version_text = f"{version.major}." f"{version.minor}." f"{version.micro}"

        elapsed_ms = (time.perf_counter() - started) * 1000

        return _dependency(
            status=HEALTHY,
            available=True,
            message=(f"Python {version_text} is running."),
            latency_ms=elapsed_ms,
        )

    except Exception as exc:
        return _dependency(
            status=UNHEALTHY,
            available=False,
            message=(f"Python runtime check failed: {exc}"),
        )


# ============================================================================
# FILESYSTEM CHECK
# ============================================================================


def _check_filesystem() -> DependencyHealth:
    """
    Check that the application can access its project directory.
    """
    started = time.perf_counter()

    try:
        if not PROJECT_ROOT.exists():
            return _dependency(
                status=UNHEALTHY,
                available=False,
                message=("Project root does not exist."),
            )

        if not PROJECT_ROOT.is_dir():
            return _dependency(
                status=UNHEALTHY,
                available=False,
                message=("Project root is not a directory."),
            )

        elapsed_ms = (time.perf_counter() - started) * 1000

        return _dependency(
            status=HEALTHY,
            available=True,
            message=("Project filesystem is accessible."),
            latency_ms=elapsed_ms,
        )

    except Exception as exc:
        return _dependency(
            status=UNHEALTHY,
            available=False,
            message=(f"Filesystem check failed: {exc}"),
        )


# ============================================================================
# DATA DIRECTORY CHECK
# ============================================================================


def _check_data_directories() -> DependencyHealth:
    """
    Check whether the primary data directories are accessible.

    We do not require them to already exist because the application may
    create them during startup.
    """
    started = time.perf_counter()

    directories = (
        DEFAULT_DATA_DIR,
        DEFAULT_FAISS_DIR,
        DEFAULT_DATABASE_DIR,
    )

    missing = [str(path) for path in directories if not path.exists()]

    elapsed_ms = (time.perf_counter() - started) * 1000

    if missing:
        return _dependency(
            status=DEGRADED,
            available=True,
            message=(
                "Data directories are not fully initialized: " + ", ".join(missing)
            ),
            latency_ms=elapsed_ms,
        )

    return _dependency(
        status=HEALTHY,
        available=True,
        message=("MemoryOS data directories are accessible."),
        latency_ms=elapsed_ms,
    )


# ============================================================================
# SQLITE CHECK
# ============================================================================


def _check_sqlite() -> DependencyHealth:
    """
    Check that Python SQLite support is available.

    This does not execute application queries because the SQLAlchemy
    database layer is responsible for that.
    """
    started = time.perf_counter()

    try:
        import sqlite3

        connection = sqlite3.connect(":memory:")

        try:
            cursor = connection.cursor()

            cursor.execute("SELECT 1")

            result = cursor.fetchone()

        finally:
            connection.close()

        elapsed_ms = (time.perf_counter() - started) * 1000

        if result != (1,):
            return _dependency(
                status=UNHEALTHY,
                available=False,
                message=("SQLite returned an unexpected result."),
                latency_ms=elapsed_ms,
            )

        return _dependency(
            status=HEALTHY,
            available=True,
            message=(f"SQLite {sqlite3.sqlite_version} is available."),
            latency_ms=elapsed_ms,
        )

    except Exception as exc:
        return _dependency(
            status=UNHEALTHY,
            available=False,
            message=(f"SQLite check failed: {exc}"),
        )


# ============================================================================
# FAISS PACKAGE CHECK
# ============================================================================


def _check_faiss() -> DependencyHealth:
    """
    Check whether FAISS can be imported.

    This deliberately does not load the persisted index.
    """
    started = time.perf_counter()

    try:
        if importlib.util.find_spec("faiss") is None:
            return _dependency(
                status=UNHEALTHY,
                available=False,
                message="FAISS package is not installed.",
            )

        import faiss

        dimension = 384

        index = faiss.IndexFlatIP(dimension)

        if index.d != dimension:
            return _dependency(
                status=UNHEALTHY,
                available=False,
                message=("FAISS initialized with an unexpected dimension."),
            )

        elapsed_ms = (time.perf_counter() - started) * 1000

        return _dependency(
            status=HEALTHY,
            available=True,
            message=("FAISS CPU is available for 384-dimensional vectors."),
            latency_ms=elapsed_ms,
        )

    except Exception as exc:
        logger.warning(
            "FAISS health check failed.",
            exc_info=True,
        )

        return _dependency(
            status=UNHEALTHY,
            available=False,
            message=(f"FAISS check failed: {exc}"),
        )


# ============================================================================
# EMBEDDING PACKAGE CHECK
# ============================================================================


def _check_embedding_package() -> DependencyHealth:
    """
    Check whether sentence-transformers is installed.

    The actual model is NOT loaded here.

    Loading all-MiniLM-L6-v2 on every health request would be expensive
    and would make the health endpoint unsuitable for monitoring.
    """
    started = time.perf_counter()

    try:
        if importlib.util.find_spec("sentence_transformers") is None:
            return _dependency(
                status=UNHEALTHY,
                available=False,
                message=("sentence-transformers is not installed."),
            )

        elapsed_ms = (time.perf_counter() - started) * 1000

        return _dependency(
            status=HEALTHY,
            available=True,
            message=("sentence-transformers package is available."),
            latency_ms=elapsed_ms,
        )

    except Exception as exc:
        return _dependency(
            status=UNHEALTHY,
            available=False,
            message=(f"Embedding package check failed: {exc}"),
        )


# ============================================================================
# OCR PACKAGE CHECK
# ============================================================================


def _check_ocr() -> DependencyHealth:
    """
    Check pytesseract availability.

    Missing Tesseract is treated as DEGRADED rather than completely
    unhealthy because Gemini Vision can still provide image analysis.
    """
    started = time.perf_counter()

    try:
        if importlib.util.find_spec("pytesseract") is None:
            return _dependency(
                status=DEGRADED,
                available=False,
                message=(
                    "pytesseract is unavailable. " "OCR functionality is degraded."
                ),
            )

        import pytesseract

        try:
            version = pytesseract.get_tesseract_version()

            message = "Tesseract OCR is available: " f"{version}"

            state = HEALTHY
            available = True

        except Exception:
            message = (
                "pytesseract is installed, but the "
                "Tesseract executable is unavailable."
            )

            state = DEGRADED
            available = False

        elapsed_ms = (time.perf_counter() - started) * 1000

        return _dependency(
            status=state,
            available=available,
            message=message,
            latency_ms=elapsed_ms,
        )

    except Exception as exc:
        return _dependency(
            status=DEGRADED,
            available=False,
            message=(f"OCR check failed: {exc}"),
        )


# ============================================================================
# GEMINI CONFIGURATION CHECK
# ============================================================================


def _check_gemini() -> DependencyHealth:
    """
    Check Gemini configuration without making an external API call.

    API availability should be checked by the Gemini service itself
    when an actual AI request is performed.
    """
    started = time.perf_counter()

    api_key = os.getenv("GEMINI_API_KEY")

    elapsed_ms = (time.perf_counter() - started) * 1000

    if not api_key:
        return _dependency(
            status=DEGRADED,
            available=False,
            message=(
                "GEMINI_API_KEY is not configured. "
                "Vision analysis will use fallback behavior."
            ),
            latency_ms=elapsed_ms,
        )

    if len(api_key.strip()) < 10:
        return _dependency(
            status=DEGRADED,
            available=False,
            message=("GEMINI_API_KEY appears invalid."),
            latency_ms=elapsed_ms,
        )

    return _dependency(
        status=HEALTHY,
        available=True,
        message=("Gemini API credentials are configured."),
        latency_ms=elapsed_ms,
    )


# ============================================================================
# DATABASE FILE CHECK
# ============================================================================


def _check_database_path() -> DependencyHealth:
    """
    Check the expected SQLite database path.

    The database is allowed to not exist yet during first startup.
    """
    started = time.perf_counter()

    database_path = DEFAULT_DATABASE_DIR / "memoryos.db"

    elapsed_ms = (time.perf_counter() - started) * 1000

    if not DEFAULT_DATABASE_DIR.exists():
        return _dependency(
            status=DEGRADED,
            available=True,
            message=("Database directory has not been initialized yet."),
            latency_ms=elapsed_ms,
        )

    if database_path.exists():
        if not database_path.is_file():
            return _dependency(
                status=UNHEALTHY,
                available=False,
                message=("SQLite database path is not a file."),
                latency_ms=elapsed_ms,
            )

        return _dependency(
            status=HEALTHY,
            available=True,
            message=("SQLite database file is available."),
            latency_ms=elapsed_ms,
        )

    return _dependency(
        status=DEGRADED,
        available=True,
        message=("SQLite database has not been created yet."),
        latency_ms=elapsed_ms,
    )


# ============================================================================
# CHECK COLLECTION
# ============================================================================


def _collect_checks() -> dict[str, DependencyHealth]:
    """
    Collect all lightweight dependency checks.

    Individual failures are isolated so one broken optional dependency
    does not prevent the health endpoint from reporting the others.
    """
    checks: dict[str, DependencyHealth] = {}

    check_functions = {
        "python": _check_python,
        "filesystem": _check_filesystem,
        "data_directories": _check_data_directories,
        "sqlite": _check_sqlite,
        "database_path": _check_database_path,
        "faiss": _check_faiss,
        "embedding": _check_embedding_package,
        "ocr": _check_ocr,
        "gemini": _check_gemini,
    }

    for name, check_function in check_functions.items():
        try:
            checks[name] = check_function()

        except Exception as exc:
            logger.exception(
                "Unexpected health-check failure: %s",
                name,
            )

            checks[name] = _dependency(
                status=UNHEALTHY,
                available=False,
                message=(f"Health check failed unexpectedly: {exc}"),
            )

    return checks


# ============================================================================
# READINESS LOGIC
# ============================================================================


def _evaluate_readiness(
    checks: dict[str, DependencyHealth],
) -> tuple[
    bool,
    list[str],
    list[str],
]:
    """
    Determine whether MemoryOS is ready to serve requests.

    Required:
        Python
        filesystem
        SQLite support
        FAISS
        embedding package

    Optional/degraded:
        OCR
        Gemini
        first-run data/database directories
    """
    required_checks = {
        "python",
        "filesystem",
        "sqlite",
        "faiss",
        "embedding",
    }

    failed: list[str] = []
    warnings: list[str] = []

    for name in required_checks:
        check = checks.get(name)

        if check is None:
            failed.append(name)
            continue

        if not check.available:
            failed.append(name)

    optional_checks = {
        "data_directories",
        "database_path",
        "ocr",
        "gemini",
    }

    for name in optional_checks:
        check = checks.get(name)

        if check is None:
            continue

        if check.status == DEGRADED:
            warnings.append(f"{name}: {check.message}")

    return (
        len(failed) == 0,
        failed,
        warnings,
    )


# ============================================================================
# LIVENESS ENDPOINT
# ============================================================================


@router.get(
    "",
    response_model=HealthResponse,
    summary="Check MemoryOS liveness",
    description=("Lightweight health check confirming that the API process is alive."),
)
async def health_check() -> HealthResponse:
    """
    Lightweight liveness endpoint.

    This endpoint should remain fast and should not require every
    external service to be operational.
    """
    uptime_seconds = time.monotonic() - _PROCESS_STARTED_AT

    return HealthResponse(
        status=HEALTHY,
        service="MemoryOS",
        version=_version(),
        timestamp=_timestamp(),
        uptime_seconds=round(
            uptime_seconds,
            3,
        ),
        checks={},
    )


# ============================================================================
# READINESS ENDPOINT
# ============================================================================


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Check MemoryOS readiness",
    description=("Check whether MemoryOS dependencies are ready to serve requests."),
)
async def readiness_check() -> ReadinessResponse:
    """
    Detailed readiness endpoint.

    Unlike /health, this checks the local runtime dependencies.
    """
    checks = _collect_checks()

    ready, failed_checks, warnings = _evaluate_readiness(checks)

    if ready:
        if warnings:
            overall_status = DEGRADED
        else:
            overall_status = HEALTHY
    else:
        overall_status = UNHEALTHY

    return ReadinessResponse(
        status=overall_status,
        service="MemoryOS",
        ready=ready,
        timestamp=_timestamp(),
        checks=checks,
        failed_checks=failed_checks,
        warnings=warnings,
    )


# ============================================================================
# PUBLIC EXPORTS
# ============================================================================

__all__ = [
    "router",
    "DependencyHealth",
    "HealthResponse",
    "ReadinessResponse",
]
