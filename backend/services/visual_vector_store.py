"""
MemoryOS persistent CLIP FAISS index.

512-dimensional visual vectors remain completely separate from
MiniLM's 384-dimensional semantic vectors.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import RLock

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

        self.dimension = int(
            dimension
        )

        self.index_path = Path(
            index_path
        )

        self.mapping_path = Path(
            mapping_path
        )

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

        self.memory_ids = []

        self._load()

    def _load(self):

        with self._lock:

            loaded = None
            ids = []

            if self.index_path.exists():

                try:

                    loaded = (
                        faiss.read_index(
                            str(
                                self.index_path
                            )
                        )
                    )

                    if (
                        loaded.d
                        != self.dimension
                    ):

                        loaded = None

                except Exception:
                    loaded = None

            if self.mapping_path.exists():

                try:

                    data = json.loads(
                        self.mapping_path
                        .read_text(
                            encoding="utf-8"
                        )
                    )

                    if isinstance(
                        data,
                        list,
                    ):

                        ids = [
                            str(item)
                            for item in data
                        ]

                except Exception:
                    ids = []

            if (
                loaded is not None
                and loaded.ntotal
                == len(ids)
            ):

                self.index = loaded
                self.memory_ids = ids

    def _vector(
        self,
        vector,
    ):

        array = np.asarray(
            vector,
            dtype=np.float32,
        ).reshape(-1)

        if array.shape != (
            self.dimension,
        ):

            raise ValueError(
                "Visual embedding "
                f"dimension mismatch: {array.shape}"
            )

        norm = float(
            np.linalg.norm(
                array
            )
        )

        if norm <= 1e-12:

            raise ValueError(
                "Zero vector cannot be indexed."
            )

        return (
            array / norm
        ).astype(
            np.float32,
            copy=False,
        )

    def contains(
        self,
        memory_id: str,
    ) -> bool:

        with self._lock:

            return str(
                memory_id
            ) in self.memory_ids

    def add(
        self,
        memory_id: str,
        vector,
    ):

        self.add_many(
            [
                (
                    memory_id,
                    vector,
                )
            ]
        )

    def add_many(
        self,
        items,
        *,
        persist=True,
    ):
        incoming = {}
        order = []

        for memory_id, vector in items:

            memory_id = str(memory_id).strip()

            if not memory_id:
                continue

            normalized = self._vector(vector)

            if memory_id not in incoming:
                order.append(memory_id)

            incoming[memory_id] = normalized

        prepared = [
            (memory_id, incoming[memory_id])
            for memory_id in order
        ]

        if not prepared:
            return 0

        with self._lock:

            positions = {
                memory_id: index
                for index, memory_id
                in enumerate(self.memory_ids)
            }

            has_updates = any(
                memory_id in positions
                for memory_id, _ in prepared
            )

            # FAST PATH:
            # normal gallery ingestion is almost always new IDs.
            # Append the whole matrix in a single FAISS call.
            if not has_updates:

                matrix = np.asarray(
                    [
                        vector
                        for _, vector
                        in prepared
                    ],
                    dtype=np.float32,
                )

                self.index.add(matrix)

                self.memory_ids.extend(
                    memory_id
                    for memory_id, _
                    in prepared
                )

            # UPDATE PATH:
            # only rebuild if an existing memory vector is replaced.
            else:

                vectors = [
                    self.index.reconstruct(index)
                    for index
                    in range(self.index.ntotal)
                ]

                ids = list(self.memory_ids)

                for memory_id, vector in prepared:

                    position = positions.get(memory_id)

                    if position is None:

                        positions[memory_id] = len(ids)

                        ids.append(memory_id)
                        vectors.append(vector)

                    else:

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
                self.memory_ids = ids

            if persist:
                self.save()

        return len(prepared)

    def search(
        self,
        query_vector,
        *,
        top_k=20,
    ):

        if top_k <= 0:
            return []

        query = self._vector(
            query_vector
        ).reshape(
            1,
            -1,
        )

        with self._lock:

            if (
                self.index.ntotal
                == 0
            ):
                return []

            k = min(
                int(top_k),
                self.index.ntotal,
            )

            scores, indices = (
                self.index.search(
                    query,
                    k,
                )
            )

            results = []

            for (
                rank,
                pair,
            ) in enumerate(
                zip(
                    scores[0],
                    indices[0],
                ),
                start=1,
            ):

                score, index = pair

                if (
                    index < 0
                    or index >=
                    len(
                        self.memory_ids
                    )
                ):
                    continue

                results.append(
                    {
                        "memory_id":
                            self.memory_ids[
                                index
                            ],
                        "score":
                            float(score),
                        "rank":
                            rank,
                    }
                )

            return results

    def save(self):

        temporary_index = (
            self.index_path
            .with_suffix(
                self.index_path.suffix
                + ".tmp"
            )
        )

        faiss.write_index(
            self.index,
            str(
                temporary_index
            ),
        )

        os.replace(
            temporary_index,
            self.index_path,
        )

        temporary_mapping = (
            self.mapping_path
            .with_suffix(
                self.mapping_path.suffix
                + ".tmp"
            )
        )

        temporary_mapping.write_text(
            json.dumps(
                self.memory_ids,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        os.replace(
            temporary_mapping,
            self.mapping_path,
        )

    @property
    def count(self):

        return int(
            self.index.ntotal
        )


_store = None
_store_lock = RLock()


def get_visual_vector_store():

    global _store

    with _store_lock:

        if _store is None:

            _store = (
                VisualVectorStore()
            )

        return _store


__all__ = [
    "VisualVectorStore",
    "get_visual_vector_store",
]
