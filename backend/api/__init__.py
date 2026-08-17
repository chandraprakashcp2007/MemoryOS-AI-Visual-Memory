"""
MemoryOS API Package
====================

REST API layer for the MemoryOS application.

The API package contains FastAPI routers responsible for exposing
MemoryOS functionality to the frontend and external clients.

Architecture
------------

Frontend
    ↓
FastAPI
    ↓
API Routers
    ↓
Application Services
    ↓
Database / AI / Vector Store

Router modules should remain thin.

Business logic belongs inside the service layer rather than inside
individual API route handlers.
"""

from __future__ import annotations

__all__: list[str] = []
