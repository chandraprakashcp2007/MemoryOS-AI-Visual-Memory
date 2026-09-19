"""
MemoryOS persistent visual FAISS index.

This index stores 512-dimensional CLIP vectors.

IMPORTANT:
Do not mix these vectors with the 384-dimensional MiniLM text index.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from threading import RLock
from typing import Iterable

import faiss
import numpy as np


class VisualVectorStore:

    def __init__(
        self,
        *,
        dimension: int = 512,
        index_path: str | Path = "data/faiss/visual.index",
        mapping_path: str | Path = "data/faiss/visual_memory_ids.json",
    ) -> None:

        self.dimension = int(dimension)

        self.index_path = Path(index_path)
        self.mapping_path = Path(mapping_path)

        self.index_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.mapping_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._lock = RLock()

        self.index = faiss.IndexFlatIP(
            self.dimension
        )

        self.memory_ids: list[str] = []

        self._load()

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    def _load(self) -> None:

        with self._lock:

            if self.index_path.exists():

                try:

                    loaded = faiss.read_index(
                        str(self.index_path)
                    )

                    if loaded.d != self.dimension:
                        raise ValueError(
                            "Visual FAISS dimension mismatch: "
                            f"{loaded.d} != {self.dimension}"
                        )

                    self.index = loaded

                except Exception:

                    # Preserve recoverability.
                    # Rebuild can occur from persisted memories.
                    self.index = faiss.IndexFlatIP(
                        self.dimension
                    )

            if self.mapping_path.exists():

                try:

                    data = json.loads(
                        self.mapping_path.read_text(
                            encoding="utf-8"
                        )
                    )

                    if isinstance(data, list):
                        self.memory_ids = [
                            str(item)
                            for item in data
                        ]

                except Exception:

                    self.memory_ids = []

            # If mapping/index disagree, do not silently
            # continue with unsafe ID alignment.
            if self.index.ntotal != len(
                self.memory_ids
            ):

                self.index = faiss.IndexFlatIP(
                    self.dimension
                )

                self.memory_ids = []

    # --------------------------------------------------------
    # VECTOR VALIDATION
    # --------------------------------------------------------

    def _vector(
        self,
        vector,
    ) -> np.ndarray:

        array = np.asarray(
            vector,
            dtype=np.float32,
        ).reshape(-1)

        if array.shape != (
            self.dimension,
        ):

            raise ValueError(
                "Expected visual vector dimension "
                f"{self.dimension}, got {array.shape}"
            )

        norm = float(
            np.linalg.norm(array)
        )

        if norm <= 1e-12:
            raise ValueError(
                "Cannot index zero visual vector."
            )

        return (
            array / norm
        ).astype(np.float32)

    # --------------------------------------------------------
    # ADD / UPDATE
    # --------------------------------------------------------

    def add(
        self,
        memory_id: str,
        vector,
    ) -> None:

        memory_id = str(memory_id).strip()

        if not memory_id:
            raise ValueError(
                "memory_id cannot be empty."
            )

        vector = self._vector(vector)

        with self._lock:

            if memory_id in self.memory_ids:

                self._replace_existing(
                    memory_id,
                    vector,
                )

            else:

                self.index.add(
                    vector.reshape(1, -1)
                )

                self.memory_ids.append(
                    memory_id
                )

            self.save()

    def _replace_existing(
        self,
        memory_id: str,
        vector: np.ndarray,
    ) -> None:

        position = self.memory_ids.index(
            memory_id
        )

        vectors = []

        for i in range(
            self.index.ntotal
        ):

            vectors.append(
                self.index.reconstruct(i)
            )

        vectors[position] = vector

        rebuilt = faiss.IndexFlatIP(
            self.dimension
        )

        if vectors:

            matrix = np.asarray(
                vectors,
                dtype=np.float32,
            )

            rebuilt.add(matrix)

        self.index = rebuilt

    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    def search(
        self,
        query_vector,
        *,
        top_k: int = 20,
    ) -> list[dict]:

        if top_k <= 0:
            return []

        query = self._vector(
            query_vector
        ).reshape(1, -1)

        with self._lock:

            if self.index.ntotal == 0:
                return []

            k = min(
                int(top_k),
                self.index.ntotal,
            )

            scores, indices = self.index.search(
                query,
                k,
            )

            results = []

            for score, position in zip(
                scores[0],
                indices[0],
            ):

                if position < 0:
                    continue

                if position >= len(
                    self.memory_ids
                ):
                    continue

                results.append(
                    {
                        "memory_id": self.memory_ids[position],
                        "score": float(score),
                    }
                )

            return results

    # --------------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------------

    def save(self) -> None:

        with self._lock:

            # Write FAISS to a temporary file first.

            temp_index = self.index_path.with_suffix(
                self.index_path.suffix + ".tmp"
            )

            faiss.write_index(
                self.index,
                str(temp_index),
            )

            os.replace(
                temp_index,
                self.index_path,
            )

            temp_mapping = self.mapping_path.with_suffix(
                self.mapping_path.suffix + ".tmp"
            )

            temp_mapping.write_text(
                json.dumps(
                    self.memory_ids,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            os.replace(
                temp_mapping,
                self.mapping_path,
            )

    # --------------------------------------------------------
    # INFORMATION
    # --------------------------------------------------------

    @property
    def count(self) -> int:

        return int(
            self.index.ntotal
        )

    def clear(self) -> None:

        with self._lock:

            self.index = faiss.IndexFlatIP(
                self.dimension
            )

            self.memory_ids = []

            self.save()


_store: VisualVectorStore | None = None
_store_lock = RLock()


def get_visual_vector_store() -> VisualVectorStore:

    global _store

    with _store_lock:

        if _store is None:
            _store = VisualVectorStore()

        return _store


__all__ = [
    "VisualVectorStore",
    "get_visual_vector_store",
]
