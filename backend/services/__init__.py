"""
MemoryOS Backend Package
========================

Core backend package for the MemoryOS application.

MemoryOS transforms visual memories such as screenshots and images
into structured, searchable AI memories.

Backend architecture:

    API
      ↓
    Services
      ↓
    AI / Processing
      ↓
    Database + Vector Index


Package responsibilities
------------------------

The ``backend`` package contains:

- Application configuration
- Database models and persistence
- API routes
- AI services
- OCR processing
- Image processing
- Embedding generation
- FAISS vector search
- Hybrid retrieval
- Reranking
- Memory management

This module intentionally contains no application startup logic.

FastAPI application startup is handled by:

    backend.main


Keeping package initialization lightweight prevents:

- circular imports
- unnecessary database initialization
- expensive ML model loading
- side effects during testing
- slow CLI imports
"""

from __future__ import annotations

# ============================================================================
# PACKAGE METADATA
# ============================================================================

__title__ = "MemoryOS Backend"
__description__ = "AI-powered visual memory and semantic search backend."
__version__ = "1.0.0"
__author__ = "MemoryOS Team"
__license__ = "MIT"


# ============================================================================
# PUBLIC PACKAGE API
# ============================================================================

__all__ = [
    "__title__",
    "__description__",
    "__version__",
    "__author__",
    "__license__",
]
