"""Idempotent cloud schema migration, including pgvector's durable index."""
from __future__ import annotations
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from backend import cloud_models  # registers SQLAlchemy tables
from backend.config import settings
from backend.database import Base, engine

Base.metadata.create_all(bind=engine)
if settings.vector_backend == "pgvector":
    if engine.dialect.name != "postgresql":
        raise RuntimeError("pgvector production migration requires PostgreSQL.")
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.execute(text(f"""CREATE TABLE IF NOT EXISTS memory_embeddings (
            memory_id VARCHAR(64) PRIMARY KEY REFERENCES cloud_memories(memory_id) ON DELETE CASCADE,
            user_id VARCHAR(36) NOT NULL REFERENCES cloud_users(id) ON DELETE CASCADE,
            embedding vector({settings.vector_dimension}) NOT NULL,
            embedding_model VARCHAR(255) NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"""))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_memory_embeddings_user ON memory_embeddings(user_id)"))
    # HNSW needs a sufficiently recent pgvector build.  The durable vector
    # table remains valid when a managed provider has not exposed it yet.
    try:
        with engine.begin() as connection:
            connection.execute(text("CREATE INDEX IF NOT EXISTS ix_memory_embeddings_cosine ON memory_embeddings USING hnsw (embedding vector_cosine_ops)"))
    except SQLAlchemyError as exc:
        print(f"HNSW index was not created by this pgvector installation: {exc.__class__.__name__}")
print("Cloud schema migration complete.")
