"""
MemoryOS embedding compatibility service.

The previous version of this file was accidentally replaced by a copy of
ProcessingService.

The canonical text embedding implementation now lives in:

    backend.services.semantic_embedding_service

This module exists because several existing MemoryOS services import:

    backend.services.embedding_service.EmbeddingService
"""

from __future__ import annotations

from backend.services.semantic_embedding_service import (
    EmbeddingService,
    get_embedding_service,
)


# Shared lazy service used by older MemoryOS modules.
embedding_service = get_embedding_service()


__all__ = [
    "EmbeddingService",
    "get_embedding_service",
    "embedding_service",
]
