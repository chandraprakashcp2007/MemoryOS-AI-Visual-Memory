"""
MemoryOS — FastAPI Application
Premium MVP Entry Point
"""

from __future__ import annotations

import logging
import os
import sys
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# ============================================================================
# PATHS
# ============================================================================

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
THUMBNAIL_DIR = DATA_DIR / "thumbnails"
FAISS_DIR = DATA_DIR / "faiss"
DATABASE_DIR = DATA_DIR / "database"
LOG_DIR = DATA_DIR / "logs"

# ============================================================================
# APP CONFIG
# ============================================================================

APP_NAME = "MemoryOS"
APP_VERSION = os.getenv("MEMORYOS_VERSION", "1.0.0")

ENVIRONMENT = (
    os.getenv(
        "MEMORYOS_ENV",
        "development",
    )
    .strip()
    .lower()
)

DEBUG = os.getenv(
    "MEMORYOS_DEBUG",
    "false",
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

APP_DESCRIPTION = """
MemoryOS is an AI-powered visual memory and semantic retrieval system.
"""

LOG_LEVEL = os.getenv(
    "MEMORYOS_LOG_LEVEL",
    "INFO",
).upper()

# ============================================================================
# LOGGING
# ============================================================================


def configure_logging() -> None:
    numeric_level = getattr(
        logging,
        LOG_LEVEL,
        logging.INFO,
    )

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root_logger = logging.getLogger()

    if not root_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    root_logger.setLevel(numeric_level)


configure_logging()

logger = logging.getLogger("memoryos")

# ============================================================================
# DIRECTORIES
# ============================================================================


def ensure_directories() -> None:
    directories = (
        DATA_DIR,
        UPLOAD_DIR,
        THUMBNAIL_DIR,
        FAISS_DIR,
        DATABASE_DIR,
        LOG_DIR,
    )

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def validate_production_configuration() -> None:
    """Reject the old laptop-only architecture before it can serve traffic."""
    if ENVIRONMENT != "production":
        return

    database_url = os.getenv("DATABASE_URL", "").strip()
    storage_backend = os.getenv("MEMORYOS_STORAGE_BACKEND", "").strip().lower()
    cors_origins = _parse_cors_origins()
    missing: list[str] = []

    if not database_url or database_url.startswith("sqlite"):
        missing.append("DATABASE_URL must be a managed PostgreSQL URL (not SQLite)")
    if storage_backend != "supabase":
        missing.append("MEMORYOS_STORAGE_BACKEND must be 'supabase' for this deployment")
    if storage_backend == "supabase":
        if not os.getenv("SUPABASE_URL", "").strip():
            missing.append("SUPABASE_URL")
        if not os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip():
            missing.append("SUPABASE_SERVICE_ROLE_KEY")
    if not os.getenv("MEMORYOS_CLOUD_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
        missing.append("MEMORYOS_CLOUD_MODE=true")
    if not os.getenv("MEMORYOS_AUTH_SECRET", "").strip():
        missing.append("MEMORYOS_AUTH_SECRET")
    if os.getenv("MEMORYOS_VECTOR_BACKEND", "pgvector").strip().lower() != "pgvector":
        missing.append("MEMORYOS_VECTOR_BACKEND=pgvector")
    if not os.getenv("MEMORYOS_CORS_ORIGINS", "").strip():
        missing.append("MEMORYOS_CORS_ORIGINS must name the deployed frontend origin")
    elif "*" in cors_origins:
        missing.append("MEMORYOS_CORS_ORIGINS must not contain '*'")

    if missing:
        raise RuntimeError(
            "Unsafe production configuration: " + "; ".join(missing) + ". "
            "MemoryOS refuses to start with device-local persistence in production."
        )


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================


def initialize_database() -> None:
    """
    Initialize SQLite tables.

    This imports models before create_all() so SQLAlchemy knows
    every MemoryOS table.
    """

    try:
        from backend.database import Base, engine

        # Import models so all ORM tables are registered.
        from backend import models  # noqa: F401
        if os.getenv("MEMORYOS_CLOUD_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
            from backend import cloud_models  # noqa: F401

        Base.metadata.create_all(bind=engine)

        logger.info("SQLite database initialized successfully.")

    except Exception:
        logger.exception("Database initialization failed.")
        raise


# ============================================================================
# CORS
# ============================================================================


def _parse_cors_origins() -> list[str]:
    raw = os.getenv(
        "MEMORYOS_CORS_ORIGINS",
        "",
    ).strip()

    if not raw:
        return [
            "http://localhost:3000",
            "http://localhost:5173",
            "http://localhost:5500",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:5500",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
        ]

    origins = [origin.strip().rstrip("/") for origin in raw.split(",") if origin.strip()]
    if ENVIRONMENT == "production" and "*" in origins:
        raise RuntimeError("MEMORYOS_CORS_ORIGINS must not contain '*' in production.")
    return origins


CORS_ORIGINS = _parse_cors_origins()

# ============================================================================
# REQUEST TIMING
# ============================================================================


class RequestTimingMiddleware(BaseHTTPMiddleware):

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):
        started = time.perf_counter()

        try:
            response = await call_next(request)

        except Exception:
            elapsed_ms = (time.perf_counter() - started) * 1000

            logger.exception(
                "%s %s -> ERROR | %.2f ms",
                request.method,
                request.url.path,
                elapsed_ms,
            )

            raise

        elapsed_ms = (time.perf_counter() - started) * 1000

        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"

        logger.info(
            "%s %s -> %s | %.2f ms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )

        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add browser-safe defaults without imposing a CSP on the Vite UI."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Small in-process guard for costly public endpoints.

    It is intentionally disabled by default in local development. Production
    deployments enable it with MEMORYOS_RATE_LIMIT_ENABLED=true or replace it
    with an upstream gateway limiter when running multiple API workers.
    """

    _windows: dict[tuple[str, str], deque[float]] = defaultdict(deque)
    _limited_prefixes = ("/upload", "/search")

    async def dispatch(self, request: Request, call_next):
        enabled = str(os.getenv("MEMORYOS_RATE_LIMIT_ENABLED", str(ENVIRONMENT == "production"))).strip().lower() in {"1", "true", "yes", "on"}
        if not enabled or not request.url.path.startswith(self._limited_prefixes):
            return await call_next(request)

        limit = int(os.getenv("MEMORYOS_RATE_LIMIT_PER_MINUTE", "60"))
        if limit < 1:
            return await call_next(request)
        client = request.client.host if request.client else "unknown"
        key = (client, request.url.path.split("/", 2)[1])
        now = time.monotonic()
        window = self._windows[key]
        while window and now - window[0] >= 60:
            window.popleft()
        if len(window) >= limit:
            return JSONResponse(status_code=429, content={"success": False, "error": "rate_limited", "message": "Too many requests. Please try again shortly."}, headers={"Retry-After": "60"})
        window.append(now)
        return await call_next(request)


# ============================================================================
# EXCEPTION HANDLER
# ============================================================================


async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:

    logger.exception(
        "Unhandled application exception: %s %s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "internal_server_error",
            "message": "MemoryOS encountered an unexpected internal error.",
        },
    )


# ============================================================================
# LIFESPAN
# ============================================================================


@asynccontextmanager
async def lifespan(
    application: FastAPI,
) -> AsyncIterator[None]:

    started = time.perf_counter()

    logger.info("============================================================")
    logger.info(
        "Starting %s v%s",
        APP_NAME,
        APP_VERSION,
    )
    logger.info(
        "Environment: %s",
        ENVIRONMENT,
    )
    logger.info("============================================================")

    try:

        validate_production_configuration()

        # ------------------------------------------------------------
        # DIRECTORIES
        # ------------------------------------------------------------

        cloud_mode = os.getenv("MEMORYOS_CLOUD_MODE", "").strip().lower() in {"1", "true", "yes", "on"}
        if not cloud_mode:
            ensure_directories()
            logger.info("Local runtime directories ready.")
        else:
            logger.info("Cloud mode: durable storage is external; no memory directories created.")

        # ------------------------------------------------------------
        # DATABASE
        # ------------------------------------------------------------

        initialize_database()

        # ------------------------------------------------------------
        # STARTUP COMPLETE
        # ------------------------------------------------------------

        elapsed_ms = (time.perf_counter() - started) * 1000

        logger.info(
            "MemoryOS startup completed in %.2f ms",
            elapsed_ms,
        )

        yield

    except Exception:
        logger.exception("MemoryOS startup failed.")
        raise

    finally:

        logger.info(
            "Shutting down %s...",
            APP_NAME,
        )

        logger.info(
            "%s shutdown completed.",
            APP_NAME,
        )


# ============================================================================
# ROUTER REGISTRATION
# ============================================================================


def _register_routers(
    application: FastAPI,
) -> None:

    # Cloud mode deliberately mounts a replacement API before any legacy
    # filesystem route.  The old routes remain available for local migration
    # only and cannot become a production source of truth.
    if os.getenv("MEMORYOS_CLOUD_MODE", "").strip().lower() in {"1", "true", "yes", "on"}:
        from backend.api.cloud import router as cloud_router
        application.include_router(cloud_router)
        logger.info("Registered router: Cloud persistence")
        return

    # ------------------------------------------------------------------------
    # HEALTH
    # ------------------------------------------------------------------------

    try:
        from backend.api.health import router as health_router

        application.include_router(health_router)

        logger.info("Registered router: Health")

    except Exception:
        logger.exception("Failed to register Health router")

    # ------------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------------

    try:
        from backend.api.stats import router as stats_router

        application.include_router(stats_router)

        logger.info("Registered router: Statistics")

    except Exception:
        logger.exception("Failed to register Statistics router")

    # ------------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------------

    try:
        from backend.api.search import router as search_router

        application.include_router(search_router)

        logger.info("Registered router: Search")

    except Exception:
        logger.exception("Failed to register Search router")

    # ------------------------------------------------------------------------
    # MEMORIES
    # ------------------------------------------------------------------------

    try:
        from backend.api.memories import router as memories_router

        application.include_router(memories_router)

        logger.info("Registered router: Memories")

    except Exception:
        logger.exception("Failed to register Memories router")

    # ------------------------------------------------------------------------
    # UPLOAD
    # ------------------------------------------------------------------------

    try:
        from backend.api.upload import router as upload_router

        application.include_router(upload_router)

        logger.info("Registered router: Upload")

    except Exception:
        logger.exception("Failed to register Upload router")

    # ------------------------------------------------------------------------
    # FAST GALLERY
    # ------------------------------------------------------------------------

    try:
        from backend.api.gallery import router as gallery_router

        application.include_router(gallery_router)

        logger.info("Registered router: Gallery")

    except Exception:
        logger.exception("Failed to register Gallery router")

    # ------------------------------------------------------------------------
    # MEMORYOS AI ASSISTANT
    # ------------------------------------------------------------------------

    try:
        from backend.api.assistant import router as assistant_router

        application.include_router(assistant_router)

        logger.info("Registered router: Assistant")

    except Exception:
        logger.exception("Failed to register Assistant router")

    # ------------------------------------------------------------------------
    # SMART COLLECTIONS
    # ------------------------------------------------------------------------

    try:
        from backend.api.collections import router as collections_router

        application.include_router(collections_router)

        logger.info("Registered router: Collections")

    except Exception:
        logger.exception("Failed to register Collections router")

    # ------------------------------------------------------------------------
    # OCR
    # ------------------------------------------------------------------------

    try:
        from backend.api.ocr import router as ocr_router

        application.include_router(ocr_router)

        logger.info("Registered router: OCR")

    except Exception:
        logger.exception("Failed to register OCR router")


# ============================================================================
# APPLICATION FACTORY
# ============================================================================


def create_app() -> FastAPI:

    application = FastAPI(
        title=APP_NAME,
        description=APP_DESCRIPTION,
        version=APP_VERSION,
        debug=DEBUG,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # ------------------------------------------------------------------------
    # MIDDLEWARE
    # ------------------------------------------------------------------------

    application.add_middleware(RequestTimingMiddleware)
    application.add_middleware(SecurityHeadersMiddleware)
    application.add_middleware(RateLimitMiddleware)

    application.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Process-Time-Ms"],
    )

    # ------------------------------------------------------------------------
    # EXCEPTIONS
    # ------------------------------------------------------------------------

    application.add_exception_handler(
        Exception,
        unhandled_exception_handler,
    )

    # ------------------------------------------------------------------------
    # ROUTERS
    # ------------------------------------------------------------------------

    _register_routers(application)

    # ------------------------------------------------------------------------
    # ROOT
    # ------------------------------------------------------------------------

    @application.get(
        "/",
        tags=["System"],
        summary="MemoryOS API information",
    )
    async def root() -> dict[str, object]:

        return {
            "success": True,
            "service": APP_NAME,
            "version": APP_VERSION,
            "environment": ENVIRONMENT,
            "message": "MemoryOS API is running.",
            "docs": "/docs",
            "redoc": "/redoc",
            "openapi": "/openapi.json",
            "health": "/health",
            "readiness": "/health/ready",
        }

    # ------------------------------------------------------------------------
    # ROUTE DEBUG ENDPOINT
    # ------------------------------------------------------------------------

    @application.get(
        "/routes",
        tags=["System"],
        summary="List registered API routes",
    )
    async def routes():

        routes = []

        for route in application.routes:

            methods = getattr(
                route,
                "methods",
                None,
            )

            if methods:

                routes.append(
                    {
                        "path": route.path,
                        "methods": sorted(methods),
                    }
                )

        return {
            "success": True,
            "count": len(routes),
            "routes": routes,
        }

    return application


# ============================================================================
# APPLICATION INSTANCE
# ============================================================================

app = create_app()


# ============================================================================
# DEVELOPMENT ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host=os.getenv(
            "MEMORYOS_HOST",
            "127.0.0.1",
        ),
        port=int(
            os.getenv(
                "MEMORYOS_PORT",
                "8000",
            )
        ),
        reload=(ENVIRONMENT == "development"),
        log_level=LOG_LEVEL.lower(),
    )


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    "app",
    "create_app",
]
