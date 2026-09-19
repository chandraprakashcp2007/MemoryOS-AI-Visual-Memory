"""
MemoryOS semantic text embedding service.

Uses the pretrained sentence-transformers/all-MiniLM-L6-v2 model.

No model training is required.

The model is downloaded automatically on first use and then cached locally.
"""

from __future__ import annotations

import os
from threading import RLock
from typing import Iterable, Sequence

import numpy as np


DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingService:
    """
    Shared MemoryOS text embedding service.

    Provides compatibility methods used by both the legacy and current
    MemoryOS pipeline.
    """

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = (
            model_name
            or os.getenv("MEMORYOS_TEXT_EMBEDDING_MODEL")
            or DEFAULT_MODEL
        )

        # all-MiniLM-L6-v2 produces 384-dimensional embeddings.
        self.embedding_dimension = 384
        self.dimension = 384

        self._model = None
        self._lock = RLock()

    def _get_model(self):
        """
        Lazily load the pretrained model.

        Default behaviour allows the official model library to download the
        model the first time it is required.

        Set MEMORYOS_EMBEDDING_LOCAL_ONLY=1 after the model is cached when
        completely offline operation is desired.
        """

        with self._lock:
            if self._model is None:

                from sentence_transformers import SentenceTransformer

                local_only = (
                    os.getenv(
                        "MEMORYOS_EMBEDDING_LOCAL_ONLY",
                        "0",
                    ).strip().lower()
                    in {"1", "true", "yes"}
                )

                self._model = SentenceTransformer(
                    self.model_name,
                    local_files_only=local_only,
                )

                try:
                    detected_dimension = (
                        self._model.get_sentence_embedding_dimension()
                    )

                    if detected_dimension:
                        self.embedding_dimension = int(detected_dimension)
                        self.dimension = int(detected_dimension)

                except Exception:
                    pass

            return self._model

    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed one text string as a normalized float32 vector.
        """

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Embedding text must be a non-empty string."
            )

        model = self._get_model()

        vector = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        vector = np.asarray(
            vector,
            dtype=np.float32,
        ).reshape(-1)

        if vector.shape != (self.embedding_dimension,):
            raise ValueError(
                "Unexpected embedding dimension: "
                f"{vector.shape}; expected "
                f"({self.embedding_dimension},)"
            )

        return vector

    # ------------------------------------------------------------
    # Compatibility aliases
    # ------------------------------------------------------------

    def embed(self, text: str) -> np.ndarray:
        return self.embed_text(text)

    def create_embedding(self, text: str) -> np.ndarray:
        return self.embed_text(text)

    def get_embedding(self, text: str) -> np.ndarray:
        return self.embed_text(text)

    def encode(self, texts):
        """
        Compatible with both:

            encode("hello")
            encode(["hello", "world"])
        """

        if isinstance(texts, str):
            return self.embed_text(texts)

        values = list(texts)

        if not values:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=np.float32,
            )

        model = self._get_model()

        vectors = model.encode(
            values,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(
            vectors,
            dtype=np.float32,
        )

    def warmup(self) -> int:
        """
        Force model download/loading and return vector dimension.
        """

        self._get_model()
        return self.embedding_dimension


_service: EmbeddingService | None = None
_service_lock = RLock()


def get_embedding_service() -> EmbeddingService:
    global _service

    with _service_lock:

        if _service is None:
            _service = EmbeddingService()

        return _service
