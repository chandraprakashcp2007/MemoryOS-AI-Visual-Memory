"""
MemoryOS - SQLAlchemy Model Registry
====================================

Central import registry for all MemoryOS SQLAlchemy models.

Responsibilities
----------------
- Import every database model in one place.
- Ensure SQLAlchemy registers every model with Base.metadata.
- Provide a stable public model API.
- Prevent scattered model imports throughout the application.
- Support database initialization, migrations, relationships, and services.

Architecture
------------

    backend/database.py
           │
           │ Base
           ▼
    backend/models/
           │
           ├── memory.py
           ├── image.py
           ├── entity.py
           ├── processing_run.py
           └── search_history.py
           │
           ▼
      Base.metadata
           │
           ▼
      SQLAlchemy
           │
           ▼
         SQLite

Important
---------
This module intentionally contains NO database engine creation,
NO sessions, and NO application startup logic.

Its primary job is model registration.

Before calling:

    Base.metadata.create_all(bind=engine)

the application should import:

    from backend import models

This guarantees that all model classes have been registered with
SQLAlchemy's metadata registry.
"""

from __future__ import annotations

# ============================================================================
# MODEL IMPORTS
# ============================================================================
#
# IMPORTANT:
#
# Keep these imports here even if another part of the application does not
# directly reference every model.
#
# Importing the model classes causes SQLAlchemy to register their tables with:
#
#     Base.metadata
#
# Without these imports, create_all() may silently fail to create tables for
# models that have never otherwise been imported.
# ============================================================================

from backend.models.entity import Entity
from backend.models.image import ImageAsset
from backend.models.memory import Memory
from backend.models.processing_run import ProcessingRun
from backend.models.search_history import SearchHistory

# ============================================================================
# PUBLIC MODEL REGISTRY
# ============================================================================
#
# This tuple provides one canonical collection of every MemoryOS model.
#
# It is useful for:
#
# - testing
# - database inspection
# - migration tooling
# - administrative tooling
# - debugging
# - future schema management
#
# Example:
#
#     from backend.models import ALL_MODELS
#
#     for model in ALL_MODELS:
#         print(model.__tablename__)
#
# ============================================================================

ALL_MODELS = (
    Memory,
    ImageAsset,
    Entity,
    ProcessingRun,
    SearchHistory,
)


# ============================================================================
# MODEL NAME REGISTRY
# ============================================================================
#
# A stable name → model mapping can be useful for internal tooling.
#
# Example:
#
#     model = MODEL_REGISTRY["memory"]
#
# ============================================================================

MODEL_REGISTRY = {
    "memory": Memory,
    "image": ImageAsset,
    "entity": Entity,
    "processing_run": ProcessingRun,
    "search_history": SearchHistory,
}


# ============================================================================
# TABLE NAME REGISTRY
# ============================================================================
#
# This gives application code a lightweight way to inspect the expected
# database tables without importing individual model modules.
# ============================================================================

TABLE_NAMES = tuple(model.__tablename__ for model in ALL_MODELS)


# ============================================================================
# MODEL REGISTRY VALIDATION
# ============================================================================
#
# Fail early if:
#
# - a model accidentally has no table name
# - duplicate table names are introduced
#
# This is deliberately lightweight and does not connect to the database.
# ============================================================================


def _validate_model_registry() -> None:
    """
    Validate the centralized MemoryOS model registry.

    This runs at module import time and catches common model registration
    mistakes early during development and application startup.
    """

    if not ALL_MODELS:
        raise RuntimeError("MemoryOS model registry is empty.")

    if len(TABLE_NAMES) != len(set(TABLE_NAMES)):
        raise RuntimeError(
            "Duplicate SQLAlchemy table names detected in MemoryOS models."
        )

    for model in ALL_MODELS:
        table_name = getattr(
            model,
            "__tablename__",
            None,
        )

        if not table_name:
            raise RuntimeError(
                f"SQLAlchemy model {model.__name__!r} " "does not define __tablename__."
            )


_validate_model_registry()


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Models
    "Memory",
    "ImageAsset",
    "Entity",
    "ProcessingRun",
    "SearchHistory",
    # Registries
    "ALL_MODELS",
    "MODEL_REGISTRY",
    "TABLE_NAMES",
]
