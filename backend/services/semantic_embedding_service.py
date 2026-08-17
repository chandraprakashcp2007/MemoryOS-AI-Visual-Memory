"""Local text embeddings shared by upload indexing and semantic search."""

from __future__ import annotations

from threading import RLock

import numpy as np


class EmbeddingService:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self.embedding_dimension = 384
        self._model = None
        self._lock = RLock()

    def _get_model(self):
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                # Search must stay responsive when the optional model is not
                # cached or the machine is offline. Deployment installs the
                # model explicitly; runtime search never blocks on downloads.
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            return self._model

    def embed_text(self, text: str) -> np.ndarray:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding text must be a non-empty string.")
        vector = self._get_model().encode(text, convert_to_numpy=True, normalize_embeddings=True)
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape != (self.embedding_dimension,):
            raise ValueError(f"Expected {self.embedding_dimension} dimensions, got {vector.shape}.")
        return vector


_service: EmbeddingService | None = None
_lock = RLock()


def get_embedding_service() -> EmbeddingService:
    global _service
    with _lock:
        if _service is None:
            _service = EmbeddingService()
        return _service
