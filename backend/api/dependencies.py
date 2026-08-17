"""
MemoryOS - API Dependencies
===========================

Centralized FastAPI dependency providers.

This module is responsible for exposing shared application components
to API route handlers.

Design goals
------------
- Avoid creating heavyweight AI/vector objects per request.
- Keep dependency construction centralized.
- Make testing easy through dependency overrides.
- Keep API routers thin.
- Avoid global business logic inside route modules.
- Fail gracefully when optional services are unavailable.

Dependency graph
----------------

FastAPI
   │
   └── API dependencies
          │
          ├── EmbeddingService
          │
          ├── VectorStore
          │
          └── SearchService
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING

from fastapi import HTTPException, status

from backend.services.embedding_service import EmbeddingService
from backend.services.search_service import SearchService
from backend.services.vector_store import VectorStore

if TYPE_CHECKING:
    from typing import Any


# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# ERRORS
# ============================================================================


class DependencyInitializationError(RuntimeError):
    """Raised when an application dependency cannot be initialized."""


# ============================================================================
# APPLICATION CONTAINER
# ============================================================================


class ApplicationContainer:
    """
    Lazily initialized application dependency container.

    Heavyweight objects such as sentence-transformers models and FAISS
    indexes should not be recreated for every HTTP request.

    The container creates them once and reuses them for the application
    lifetime.
    """

    def __init__(
        self,
        *,
        faiss_index_path: str | Path | None = None,
        faiss_mapping_path: str | Path | None = None,
    ) -> None:
        self._faiss_index_path = (
            Path(faiss_index_path)
            if faiss_index_path is not None
            else Path("data/faiss/memory.index")
        )

        self._faiss_mapping_path = (
            Path(faiss_mapping_path)
            if faiss_mapping_path is not None
            else Path("data/faiss/memory_ids.json")
        )

        self._embedding_service: EmbeddingService | None = None
        self._vector_store: VectorStore | None = None
        self._search_service: SearchService | None = None

        self._lock = Lock()

    # =========================================================================
    # EMBEDDING SERVICE
    # =========================================================================

    def get_embedding_service(self) -> EmbeddingService:
        """
        Return the shared EmbeddingService.

        EmbeddingService internally handles model loading/caching, so the API
        layer should reuse a single service instance.
        """
        if self._embedding_service is not None:
            return self._embedding_service

        with self._lock:
            if self._embedding_service is None:
                logger.info("Initializing MemoryOS EmbeddingService.")

                try:
                    self._embedding_service = EmbeddingService()

                except Exception as exc:
                    logger.exception("Failed to initialize EmbeddingService.")

                    raise DependencyInitializationError(
                        "Embedding service could not be initialized."
                    ) from exc

        return self._embedding_service

    # =========================================================================
    # VECTOR STORE
    # =========================================================================

    def get_vector_store(self) -> VectorStore:
        """
        Return the shared persistent FAISS VectorStore.
        """
        if self._vector_store is not None:
            return self._vector_store

        with self._lock:
            if self._vector_store is None:
                logger.info("Initializing MemoryOS VectorStore.")

                try:
                    embedding_service = self.get_embedding_service()

                    # Try to obtain the embedding dimension from the
                    # existing EmbeddingService rather than duplicating
                    # model-specific dimensions in the API layer.
                    dimension = self._resolve_embedding_dimension(embedding_service)

                    self._vector_store = VectorStore(
                        dimension=dimension,
                        index_path=self._faiss_index_path,
                        mapping_path=self._faiss_mapping_path,
                    )

                except Exception as exc:
                    logger.exception("Failed to initialize VectorStore.")

                    raise DependencyInitializationError(
                        "Vector store could not be initialized."
                    ) from exc

        return self._vector_store

    # =========================================================================
    # SEARCH SERVICE
    # =========================================================================

    def get_search_service(self) -> SearchService:
        """
        Return the shared SearchService.

        SearchService orchestrates:
            query
              ↓
            embedding
              ↓
            FAISS
              ↓
            keyword retrieval
              ↓
            candidate merge
              ↓
            reranking
        """
        if self._search_service is not None:
            return self._search_service

        with self._lock:
            if self._search_service is None:
                logger.info("Initializing MemoryOS SearchService.")

                try:
                    embedding_service = self.get_embedding_service()

                    vector_store = self.get_vector_store()

                    self._search_service = SearchService(
                        embedding_service=embedding_service,
                        vector_store=vector_store,
                    )

                except Exception as exc:
                    logger.exception("Failed to initialize SearchService.")

                    raise DependencyInitializationError(
                        "Search service could not be initialized."
                    ) from exc

        return self._search_service

    # =========================================================================
    # RESET
    # =========================================================================

    def reset(self) -> None:
        """
        Reset cached dependencies.

        Primarily useful for:
        - automated tests
        - development reloads
        - controlled application reinitialization
        """
        with self._lock:
            self._search_service = None
            self._vector_store = None
            self._embedding_service = None

        logger.info("MemoryOS application dependency container reset.")

    # =========================================================================
    # HELPERS
    # =========================================================================

    @staticmethod
    def _resolve_embedding_dimension(
        embedding_service: EmbeddingService,
    ) -> int:
        """
        Resolve the configured embedding dimension without hard-coding
        model-specific dimensions inside this module.

        Supports common EmbeddingService metadata patterns while keeping
        this dependency layer defensive.
        """

        # Preferred public-style attributes.
        candidates = (
            getattr(
                embedding_service,
                "dimension",
                None,
            ),
            getattr(
                embedding_service,
                "embedding_dimension",
                None,
            ),
        )

        for value in candidates:
            if isinstance(value, int) and value > 0:
                return value

        # Some implementations expose model metadata.
        model_metadata = getattr(
            embedding_service,
            "model_metadata",
            None,
        )

        if callable(model_metadata):
            try:
                metadata = model_metadata()
            except Exception:
                metadata = None

            if isinstance(metadata, dict):
                dimension = metadata.get("dimension")

                if isinstance(dimension, int) and dimension > 0:
                    return dimension

        # Current MemoryOS model:
        #
        # all-MiniLM-L6-v2
        #
        # Expected dimension:
        # 384
        #
        # This fallback exists only for compatibility with the service
        # implementation if it does not expose dimension metadata.
        logger.warning(
            "EmbeddingService did not expose embedding dimension; "
            "using MemoryOS default dimension 384."
        )

        return 384


# ============================================================================
# GLOBAL APPLICATION CONTAINER
# ============================================================================

_application_container = ApplicationContainer()


# ============================================================================
# FASTAPI DEPENDENCIES
# ============================================================================


def get_application_container() -> ApplicationContainer:
    """
    FastAPI dependency for the shared application container.
    """
    return _application_container


def get_embedding_service() -> EmbeddingService:
    """
    FastAPI dependency providing the shared EmbeddingService.
    """
    try:
        return _application_container.get_embedding_service()

    except DependencyInitializationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "embedding_service_unavailable",
                "message": str(exc),
            },
        ) from exc


def get_vector_store() -> VectorStore:
    """
    FastAPI dependency providing the shared VectorStore.
    """
    try:
        return _application_container.get_vector_store()

    except DependencyInitializationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "vector_store_unavailable",
                "message": str(exc),
            },
        ) from exc


def get_search_service() -> SearchService:
    """
    FastAPI dependency providing the shared SearchService.
    """
    try:
        return _application_container.get_search_service()

    except DependencyInitializationError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "search_service_unavailable",
                "message": str(exc),
            },
        ) from exc


# ============================================================================
# TESTING / APPLICATION LIFECYCLE HELPERS
# ============================================================================


def reset_application_dependencies() -> None:
    """
    Reset all cached dependencies.

    This function is intentionally explicit so tests can isolate
    application state between test cases.
    """
    _application_container.reset()


@lru_cache(maxsize=1)
def get_runtime_info() -> dict[str, Any]:
    """
    Return static runtime information.

    Cached because this information does not change during normal
    application execution.

    Kept intentionally lightweight so health/status endpoints can use it
    without initializing heavyweight AI models.
    """
    return {
        "service": "MemoryOS",
        "environment": "local",
        "embedding_model": "all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "vector_backend": "FAISS",
        "vector_metric": "inner_product",
    }


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    "ApplicationContainer",
    "DependencyInitializationError",
    "get_application_container",
    "get_embedding_service",
    "get_vector_store",
    "get_search_service",
    "get_runtime_info",
    "reset_application_dependencies",
]
