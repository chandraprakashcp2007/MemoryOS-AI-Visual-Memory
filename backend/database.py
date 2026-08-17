"""
MemoryOS - Database Foundation

Centralized SQLAlchemy database configuration.

Responsibilities:
    - Create the database engine
    - Configure SQLite safely
    - Create the session factory
    - Provide request-scoped database sessions
    - Provide transaction helpers
    - Initialize database tables
    - Support testing
    - Provide database health checks

Architecture:

    FastAPI
       │
       ▼
    database.py
       │
       ├── Engine
       │
       ├── Session Factory
       │
       └── Database Session
              │
              ▼
           Models
              │
              ▼
           SQLite

The rest of MemoryOS should NOT create SQLAlchemy engines or sessions
directly. Everything should use this module.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import PROJECT_ROOT, settings

# ============================================================================
# DATABASE CONSTANTS
# ============================================================================

DEFAULT_SQLITE_TIMEOUT = 30

SQLITE_JOURNAL_MODE = "WAL"

SQLITE_SYNCHRONOUS_MODE = "NORMAL"


# ============================================================================
# BASE MODEL
# ============================================================================


class Base(DeclarativeBase):
    """
    Base class for every SQLAlchemy model in MemoryOS.

    Future models will inherit from this class:

        class Memory(Base):
            ...

    This gives SQLAlchemy a single metadata registry that can be used
    to create and inspect all MemoryOS tables.
    """

    pass


# ============================================================================
# DATABASE URL
# ============================================================================


def get_database_url() -> str:
    """
    Return the configured database URL.

    For SQLite, we normalize relative paths against the MemoryOS project root.
    This prevents the database location from unexpectedly changing depending
    on which directory was used to launch Python.
    """

    database_url = settings.database_url.strip()

    if not database_url:
        raise ValueError("DATABASE_URL cannot be empty.")

    if not database_url.startswith("sqlite"):
        return database_url

    # Standard in-memory SQLite database.
    if database_url in {
        "sqlite://",
        "sqlite:///:memory:",
    }:
        return database_url

    prefix = "sqlite:///"

    if not database_url.startswith(prefix):
        return database_url

    database_path_string = database_url[len(prefix) :]

    database_path = Path(database_path_string)

    if not database_path.is_absolute():
        database_path = PROJECT_ROOT / database_path

    database_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    return "sqlite:///" + database_path.as_posix()


# ============================================================================
# ENGINE OPTIONS
# ============================================================================


def get_engine_kwargs() -> dict[str, Any]:
    """
    Build SQLAlchemy engine options based on the active database.

    SQLite gets special configuration because SQLite has different connection
    and concurrency behavior compared with PostgreSQL/MySQL.
    """

    database_url = get_database_url()

    if database_url.startswith("sqlite"):
        return {
            "echo": settings.database_echo,
            "connect_args": {
                "check_same_thread": False,
                "timeout": DEFAULT_SQLITE_TIMEOUT,
            },
        }

    return {
        "echo": settings.database_echo,
        "pool_pre_ping": True,
    }


# ============================================================================
# ENGINE
# ============================================================================


engine: Engine = create_engine(
    get_database_url(),
    **get_engine_kwargs(),
)


# ============================================================================
# SQLITE PRAGMAS
# ============================================================================


@event.listens_for(
    engine,
    "connect",
)
def configure_sqlite_connection(
    dbapi_connection: Any,
    connection_record: Any,
) -> None:
    """
    Configure SQLite connections.

    WAL:
        Allows readers and writers to work more efficiently.

    NORMAL synchronous mode:
        Good balance between durability and performance for this MVP.

    Foreign keys:
        Ensures relational integrity.
    """

    if not engine.url.drivername.startswith("sqlite"):
        return

    cursor = dbapi_connection.cursor()

    try:
        cursor.execute("PRAGMA foreign_keys=ON")

        cursor.execute(f"PRAGMA journal_mode={SQLITE_JOURNAL_MODE}")

        cursor.execute(f"PRAGMA synchronous={SQLITE_SYNCHRONOUS_MODE}")

    finally:
        cursor.close()


# ============================================================================
# SESSION FACTORY
# ============================================================================


SessionLocal = sessionmaker(
    bind=engine,
    class_=Session,
    autoflush=False,
    autocommit=False,
    expire_on_commit=False,
)


# ============================================================================
# DATABASE SESSION
# ============================================================================


def get_db() -> Generator[
    Session,
    None,
    None,
]:
    """
    FastAPI-compatible database dependency.

    Usage:

        @router.get("/memories")
        def get_memories(
            db: Session = Depends(get_db),
        ):
            ...

    The session is always closed after the request.
    """

    db = SessionLocal()

    try:
        yield db

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


# ============================================================================
# CONTEXT-MANAGED SESSION
# ============================================================================


@contextmanager
def database_session() -> Generator[
    Session,
    None,
    None,
]:
    """
    Context-managed database session.

    Useful outside FastAPI routes.

    Example:

        with database_session() as db:
            db.add(memory)
            db.commit()

    Transactions are rolled back automatically when an exception occurs.
    """

    db = SessionLocal()

    try:
        yield db
        db.commit()

    except Exception:
        db.rollback()
        raise

    finally:
        db.close()


# ============================================================================
# TRANSACTION HELPER
# ============================================================================


@contextmanager
def transaction(
    db: Session,
) -> Generator[
    Session,
    None,
    None,
]:
    """
    Execute multiple database operations as one transaction.

    Example:

        with transaction(db):
            db.add(...)
            db.add(...)
            db.add(...)

    If anything fails, the entire transaction is rolled back.
    """

    try:
        yield db
        db.commit()

    except Exception:
        db.rollback()
        raise


# ============================================================================
# TABLE INITIALIZATION
# ============================================================================


def init_db() -> None:
    """
    Create all registered database tables.

    Models must be imported before calling this function so SQLAlchemy knows
    about every model.

    Example:

        from backend import models

        init_db()

    Calling create_all repeatedly is safe because SQLAlchemy only creates
    tables that do not already exist.
    """

    settings.ensure_directories()

    Base.metadata.create_all(
        bind=engine,
    )


# ============================================================================
# DATABASE RESET
# ============================================================================


def drop_all_tables() -> None:
    """
    Drop all registered tables.

    WARNING:
        This destroys database data.

    Intended only for:
        - automated tests
        - development resets
        - controlled local debugging
    """

    if settings.app_environment == "production":
        raise RuntimeError("Dropping database tables is disabled in production.")

    Base.metadata.drop_all(
        bind=engine,
    )


def reset_database() -> None:
    """
    Drop and recreate all registered tables.

    WARNING:
        This destroys existing database data.

    Intended for development/testing only.
    """

    drop_all_tables()
    init_db()


# ============================================================================
# DATABASE HEALTH
# ============================================================================


def check_database_connection() -> bool:
    """
    Check whether the database can execute a simple query.

    Returns:
        True  -> database is reachable
        False -> database is unavailable
    """

    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))

        return True

    except SQLAlchemyError:
        return False


# ============================================================================
# DATABASE HEALTH DETAILS
# ============================================================================


def get_database_health() -> dict[str, Any]:
    """
    Return detailed database health information.

    This will later be useful for:

        GET /api/health

    and the premium dashboard.
    """

    try:
        with engine.connect() as connection:
            result = connection.execute(text("SELECT 1"))

            result.scalar_one()

        return {
            "status": "healthy",
            "database": engine.url.database,
            "driver": engine.url.drivername,
            "connected": True,
        }

    except SQLAlchemyError as exc:
        return {
            "status": "unhealthy",
            "database": engine.url.database,
            "driver": engine.url.drivername,
            "connected": False,
            "error": str(exc),
        }


# ============================================================================
# TABLE INSPECTION
# ============================================================================


def get_table_names() -> list[str]:
    """
    Return all currently available database table names.
    """

    inspector = inspect(engine)

    return inspector.get_table_names()


def table_exists(
    table_name: str,
) -> bool:
    """
    Check whether a particular table exists.
    """

    if not table_name:
        return False

    return table_name in get_table_names()


# ============================================================================
# SESSION HEALTH
# ============================================================================


def ping_session(
    db: Session,
) -> bool:
    """
    Verify that an existing SQLAlchemy session is usable.
    """

    try:
        db.execute(text("SELECT 1"))

        return True

    except SQLAlchemyError:
        return False


# ============================================================================
# DATABASE SHUTDOWN
# ============================================================================


def dispose_engine() -> None:
    """
    Dispose all active database connections.

    Useful during:
        - application shutdown
        - tests
        - development reloads
    """

    engine.dispose()


# ============================================================================
# DATABASE INFORMATION
# ============================================================================


def get_database_info() -> dict[str, Any]:
    """
    Return non-sensitive database configuration information.

    No passwords or API keys are returned.
    """

    return {
        "url": str(engine.url.render_as_string(hide_password=True)),
        "driver": engine.url.drivername,
        "database": engine.url.database,
        "dialect": engine.dialect.name,
        "tables": get_table_names(),
    }


# ============================================================================
# SQLITE-SPECIFIC INFORMATION
# ============================================================================


def get_sqlite_pragmas() -> dict[str, Any]:
    """
    Return important SQLite runtime settings.

    If MemoryOS is later migrated to PostgreSQL, this function can remain
    unused without affecting the rest of the architecture.
    """

    if not engine.url.drivername.startswith("sqlite"):
        return {}

    pragmas = (
        "journal_mode",
        "synchronous",
        "foreign_keys",
    )

    result: dict[str, Any] = {}

    try:
        with engine.connect() as connection:
            for pragma in pragmas:
                value = connection.execute(text(f"PRAGMA {pragma}")).scalar()

                result[pragma] = value

    except SQLAlchemyError:
        return {}

    return result


# ============================================================================
# PUBLIC API
# ============================================================================


__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_database_url",
    "get_engine_kwargs",
    "get_db",
    "database_session",
    "transaction",
    "init_db",
    "drop_all_tables",
    "reset_database",
    "check_database_connection",
    "get_database_health",
    "get_table_names",
    "table_exists",
    "ping_session",
    "dispose_engine",
    "get_database_info",
    "get_sqlite_pragmas",
]
