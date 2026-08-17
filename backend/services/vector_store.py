"""
MemoryOS - Persistent FAISS Vector Store
========================================

Production-grade local vector storage for MemoryOS.

Responsibilities
----------------
- Maintain a persistent FAISS index.
- Store normalized float32 embedding vectors.
- Map FAISS integer IDs to MemoryOS memory IDs.
- Support single and batch insertion.
- Support semantic similarity search.
- Support deletion and index rebuilding.
- Detect corrupted or inconsistent persisted state.
- Persist atomically where possible.
- Provide thread-safe operations.
- Remain independent from SQLite and higher-level search orchestration.

FAISS strategy
--------------
MemoryOS uses:

    faiss.IndexFlatIP

Because EmbeddingService normalizes vectors before FAISS storage:

    cosine_similarity(a, b) == inner_product(a, b)

for normalized vectors.

Expected embedding dimension:

    384

Expected dtype:

    numpy.float32
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Iterable, Mapping, Sequence

import faiss
import numpy as np

# ============================================================================
# LOGGING
# ============================================================================

logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_DIMENSION = 384
DEFAULT_INDEX_FILENAME = "memory.index"
DEFAULT_MAPPING_FILENAME = "memory_ids.json"

MAPPING_SCHEMA_VERSION = 1

SUPPORTED_INDEX_TYPE = "IndexIDMap2(IndexFlatIP)"

MIN_SEARCH_K = 1


# ============================================================================
# EXCEPTIONS
# ============================================================================


class VectorStoreError(RuntimeError):
    """Base exception for all MemoryOS vector-store errors."""


class VectorStoreValidationError(VectorStoreError):
    """Raised when vector-store input is invalid."""


class VectorStorePersistenceError(VectorStoreError):
    """Raised when index/mapping persistence fails."""


class VectorStoreCorruptionError(VectorStoreError):
    """Raised when persisted FAISS state is inconsistent or corrupted."""


class DuplicateMemoryError(VectorStoreError):
    """Raised when a memory ID already exists in the vector store."""


class MemoryNotFoundError(VectorStoreError):
    """Raised when a requested memory ID does not exist."""


class VectorStoreClosedError(VectorStoreError):
    """Raised when an operation is attempted after close()."""


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass(frozen=True, slots=True)
class VectorSearchResult:
    """
    One semantic search result.

    Attributes
    ----------
    memory_id:
        Application-level MemoryOS identifier.

    score:
        FAISS inner-product score. For normalized vectors this corresponds
        to cosine similarity.

    faiss_id:
        Internal persistent FAISS identifier.
    """

    memory_id: str
    score: float
    faiss_id: int


@dataclass(frozen=True, slots=True)
class VectorStoreStats:
    """
    Snapshot of vector-store state.
    """

    dimension: int
    vector_count: int
    mapped_memory_count: int
    index_type: str
    index_path: str
    mapping_path: str
    is_trained: bool
    next_faiss_id: int


# ============================================================================
# VECTOR STORE
# ============================================================================


class VectorStore:
    """
    Persistent FAISS vector store for MemoryOS.

    Parameters
    ----------
    index_path:
        Location of the persistent FAISS index.

    mapping_path:
        Location of the JSON mapping between FAISS IDs and memory IDs.

    dimension:
        Embedding dimension. Defaults to 384 for all-MiniLM-L6-v2.

    auto_load:
        If True, load an existing index/mapping pair automatically.

    strict_persistence:
        If True, persistence failures raise immediately.

    Notes
    -----
    SQLite remains the authoritative metadata store.

    This class only owns:
        embedding vector
        ↕
        FAISS ID
        ↕
        memory_id
    """

    def __init__(
        self,
        index_path: str | Path = Path("data") / "faiss" / DEFAULT_INDEX_FILENAME,
        mapping_path: str | Path = Path("data") / "faiss" / DEFAULT_MAPPING_FILENAME,
        *,
        dimension: int = DEFAULT_DIMENSION,
        auto_load: bool = True,
        strict_persistence: bool = True,
    ) -> None:
        self._lock = RLock()
        self._closed = False

        self.dimension = self._validate_dimension(dimension)
        self.strict_persistence = bool(strict_persistence)

        self.index_path = Path(index_path).expanduser().resolve()
        self.mapping_path = Path(mapping_path).expanduser().resolve()

        if self.index_path == self.mapping_path:
            raise VectorStoreValidationError(
                "index_path and mapping_path must be different files."
            )

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.mapping_path.parent.mkdir(parents=True, exist_ok=True)

        self._index: faiss.IndexIDMap2 | None = None

        # Persistent application-level mappings.
        #
        # FAISS ID -> memory ID
        self._faiss_to_memory: dict[int, str] = {}

        # memory ID -> FAISS ID
        self._memory_to_faiss: dict[str, int] = {}

        # Never reuse an ID automatically.
        #
        # This makes persisted mappings easier to reason about and avoids
        # accidentally attaching a new memory to an old FAISS identifier.
        self._next_faiss_id = 1

        if auto_load:
            self.load()

    # ------------------------------------------------------------------------
    # LIFECYCLE
    # ------------------------------------------------------------------------

    def close(self) -> None:
        """
        Persist the current state and release the active index.

        Safe to call multiple times.
        """
        with self._lock:
            if self._closed:
                return

            try:
                self.persist()
            finally:
                self._closed = True
                self._index = None

                logger.info("VectorStore closed.")

    def __enter__(self) -> "VectorStore":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    # ------------------------------------------------------------------------
    # INDEX CREATION
    # ------------------------------------------------------------------------

    def _create_empty_index(self) -> faiss.IndexIDMap2:
        """
        Create a fresh FAISS IndexIDMap2 wrapping IndexFlatIP.
        """
        base_index = faiss.IndexFlatIP(self.dimension)
        index = faiss.IndexIDMap2(base_index)

        if index.d != self.dimension:
            raise VectorStoreCorruptionError(
                "FAISS created an index with an unexpected dimension: "
                f"{index.d}; expected {self.dimension}."
            )

        return index

    @property
    def index(self) -> faiss.IndexIDMap2:
        """
        Access the active FAISS index.

        The property exists mainly for diagnostics and controlled
        integration with future services.
        """
        with self._lock:
            self._ensure_open()

            if self._index is None:
                self._index = self._create_empty_index()

            return self._index

    # ------------------------------------------------------------------------
    # LOADING
    # ------------------------------------------------------------------------

    def load(self) -> None:
        """
        Load an existing FAISS index and its persistent ID mapping.

        Cases
        -----
        Neither file exists:
            Start with an empty index.

        Both files exist:
            Load and validate them.

        Only one exists:
            Treat as corrupted/incomplete state.
        """
        with self._lock:
            self._ensure_open()

            index_exists = self.index_path.exists()
            mapping_exists = self.mapping_path.exists()

            if not index_exists and not mapping_exists:
                self._reset_memory_state()

                logger.info("No existing FAISS state found. Starting with empty index.")
                return

            if index_exists != mapping_exists:
                raise VectorStoreCorruptionError(
                    "FAISS persistence is incomplete: "
                    f"index_exists={index_exists}, "
                    f"mapping_exists={mapping_exists}. "
                    "Both files must exist together."
                )

            try:
                loaded_index = faiss.read_index(str(self.index_path))

                if loaded_index.d != self.dimension:
                    raise VectorStoreCorruptionError(
                        "FAISS dimension mismatch. "
                        f"Persisted={loaded_index.d}, expected={self.dimension}."
                    )

                if not isinstance(loaded_index, faiss.IndexIDMap2):
                    raise VectorStoreCorruptionError(
                        "Unexpected FAISS index type. "
                        f"Expected IndexIDMap2, got {type(loaded_index).__name__}."
                    )

                mapping = self._read_mapping()

                self._validate_loaded_state(loaded_index, mapping)

                self._index = loaded_index

                self._faiss_to_memory = {
                    int(key): str(value)
                    for key, value in mapping["faiss_to_memory"].items()
                }

                self._memory_to_faiss = {
                    str(key): int(value)
                    for key, value in mapping["memory_to_faiss"].items()
                }

                persisted_next_id = int(mapping["next_faiss_id"])

                max_existing_id = max(
                    self._faiss_to_memory.keys(),
                    default=0,
                )

                self._next_faiss_id = max(
                    persisted_next_id,
                    max_existing_id + 1,
                    1,
                )

                logger.info(
                    "Loaded FAISS vector store: vectors=%d dimension=%d",
                    self._index.ntotal,
                    self.dimension,
                )

            except VectorStoreError:
                raise
            except Exception as exc:
                logger.exception("Failed to load FAISS vector store.")
                raise VectorStorePersistenceError(
                    f"Unable to load vector store from '{self.index_path}'."
                ) from exc

    # ------------------------------------------------------------------------
    # PERSISTENCE
    # ------------------------------------------------------------------------

    def persist(self) -> None:
        """
        Persist FAISS index and mapping.

        Persistence is performed using temporary files followed by
        os.replace(), which provides atomic replacement semantics on
        supported local filesystems.

        The two files are persisted independently. Therefore load()
        verifies that both files exist and contain mutually consistent data.
        """
        with self._lock:
            self._ensure_open()

            if self._index is None:
                self._index = self._create_empty_index()

            self._validate_runtime_state()

            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.mapping_path.parent.mkdir(parents=True, exist_ok=True)

            index_tmp: Path | None = None
            mapping_tmp: Path | None = None

            try:
                index_tmp = self._create_temp_path(self.index_path)
                mapping_tmp = self._create_temp_path(self.mapping_path)

                faiss.write_index(self._index, str(index_tmp))

                mapping_payload = self._build_mapping_payload()

                self._write_json_atomic_temp(
                    mapping_tmp,
                    mapping_payload,
                )

                # Replace only after successful writes.
                os.replace(index_tmp, self.index_path)
                index_tmp = None

                os.replace(mapping_tmp, self.mapping_path)
                mapping_tmp = None

                logger.info(
                    "Persisted FAISS vector store: vectors=%d",
                    self._index.ntotal,
                )

            except Exception as exc:
                logger.exception("Failed to persist FAISS vector store.")

                if self.strict_persistence:
                    raise VectorStorePersistenceError(
                        "Failed to persist FAISS vector store."
                    ) from exc

            finally:
                self._safe_unlink(index_tmp)
                self._safe_unlink(mapping_tmp)

    def _create_temp_path(self, target: Path) -> Path:
        """
        Create a unique temporary file path in the target directory.

        The file is created first so the path is guaranteed to be unique.
        """
        fd, raw_path = tempfile.mkstemp(
            prefix=f".{target.stem}.",
            suffix=f"{target.suffix}.tmp",
            dir=str(target.parent),
        )

        os.close(fd)

        return Path(raw_path)

    @staticmethod
    def _write_json_atomic_temp(
        path: Path,
        payload: Mapping[str, object],
    ) -> None:
        """
        Write JSON to an already-created temporary file.
        """
        with path.open("w", encoding="utf-8") as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            file.flush()
            os.fsync(file.fileno())

    @staticmethod
    def _safe_unlink(path: Path | None) -> None:
        if path is None:
            return

        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "Could not remove temporary file: %s",
                path,
                exc_info=True,
            )

    # ------------------------------------------------------------------------
    # MAPPING
    # ------------------------------------------------------------------------

    def _build_mapping_payload(self) -> dict[str, object]:
        """
        Build the persistent mapping document.
        """
        return {
            "schema_version": MAPPING_SCHEMA_VERSION,
            "dimension": self.dimension,
            "index_type": SUPPORTED_INDEX_TYPE,
            "faiss_to_memory": {
                str(faiss_id): memory_id
                for faiss_id, memory_id in self._faiss_to_memory.items()
            },
            "memory_to_faiss": {
                memory_id: faiss_id
                for memory_id, faiss_id in self._memory_to_faiss.items()
            },
            "next_faiss_id": self._next_faiss_id,
        }

    def _read_mapping(self) -> dict[str, object]:
        """
        Read and minimally validate the mapping JSON.
        """
        try:
            with self.mapping_path.open("r", encoding="utf-8") as file:
                payload = json.load(file)
        except Exception as exc:
            raise VectorStoreCorruptionError(
                f"Unable to read FAISS mapping file: {self.mapping_path}"
            ) from exc

        if not isinstance(payload, dict):
            raise VectorStoreCorruptionError(
                "FAISS mapping root must be a JSON object."
            )

        required_keys = {
            "schema_version",
            "dimension",
            "index_type",
            "faiss_to_memory",
            "memory_to_faiss",
            "next_faiss_id",
        }

        missing = required_keys.difference(payload)

        if missing:
            raise VectorStoreCorruptionError(
                "FAISS mapping is missing required fields: "
                + ", ".join(sorted(missing))
            )

        if int(payload["schema_version"]) != MAPPING_SCHEMA_VERSION:
            raise VectorStoreCorruptionError(
                "Unsupported FAISS mapping schema version: "
                f"{payload['schema_version']}"
            )

        if int(payload["dimension"]) != self.dimension:
            raise VectorStoreCorruptionError(
                "Mapping dimension mismatch. "
                f"Persisted={payload['dimension']}, "
                f"expected={self.dimension}."
            )

        if payload["index_type"] != SUPPORTED_INDEX_TYPE:
            raise VectorStoreCorruptionError(
                "Unexpected persisted index type: " f"{payload['index_type']}"
            )

        if not isinstance(payload["faiss_to_memory"], dict):
            raise VectorStoreCorruptionError("'faiss_to_memory' must be an object.")

        if not isinstance(payload["memory_to_faiss"], dict):
            raise VectorStoreCorruptionError("'memory_to_faiss' must be an object.")

        return payload

    # ------------------------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------------------------

    @staticmethod
    def _validate_dimension(dimension: int) -> int:
        if isinstance(dimension, bool) or not isinstance(dimension, int):
            raise VectorStoreValidationError("Vector dimension must be an integer.")

        if dimension <= 0:
            raise VectorStoreValidationError(
                "Vector dimension must be greater than zero."
            )

        return dimension

    @staticmethod
    def _validate_memory_id(memory_id: str) -> str:
        if not isinstance(memory_id, str):
            raise VectorStoreValidationError("memory_id must be a string.")

        normalized = memory_id.strip()

        if not normalized:
            raise VectorStoreValidationError("memory_id cannot be empty.")

        if len(normalized) > 512:
            raise VectorStoreValidationError(
                "memory_id is too long. Maximum length is 512 characters."
            )

        return normalized

    def _prepare_vectors(
        self,
        vectors: np.ndarray | Sequence[Sequence[float]],
        *,
        expected_count: int | None = None,
    ) -> np.ndarray:
        """
        Validate and normalize vector-array representation.

        This method intentionally does NOT normalize values mathematically.

        EmbeddingService owns embedding normalization.

        VectorStore only ensures:
            shape
            dtype
            contiguity
            finite values
            expected dimension
        """
        try:
            array = np.asarray(vectors)
        except Exception as exc:
            raise VectorStoreValidationError(
                "Unable to convert vectors to a NumPy array."
            ) from exc

        if array.ndim == 1:
            array = array.reshape(1, -1)

        if array.ndim != 2:
            raise VectorStoreValidationError(
                "Vectors must be a 2D array with shape "
                f"(count, {self.dimension}). Received shape={array.shape}."
            )

        if array.shape[1] != self.dimension:
            raise VectorStoreValidationError(
                "Vector dimension mismatch. "
                f"Received={array.shape[1]}, expected={self.dimension}."
            )

        if expected_count is not None and array.shape[0] != expected_count:
            raise VectorStoreValidationError(
                "Vector count does not match memory ID count. "
                f"Vectors={array.shape[0]}, IDs={expected_count}."
            )

        if array.shape[0] == 0:
            raise VectorStoreValidationError("At least one vector is required.")

        try:
            array = np.asarray(array, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise VectorStoreValidationError(
                "Vectors must contain numeric values."
            ) from exc

        if not np.isfinite(array).all():
            raise VectorStoreValidationError("Vectors contain NaN or infinite values.")

        array = np.ascontiguousarray(array, dtype=np.float32)

        return array

    def _validate_loaded_state(
        self,
        index: faiss.IndexIDMap2,
        mapping: Mapping[str, object],
    ) -> None:
        faiss_to_memory_raw = mapping["faiss_to_memory"]
        memory_to_faiss_raw = mapping["memory_to_faiss"]

        if not isinstance(faiss_to_memory_raw, dict):
            raise VectorStoreCorruptionError("Invalid FAISS-to-memory mapping.")

        if not isinstance(memory_to_faiss_raw, dict):
            raise VectorStoreCorruptionError("Invalid memory-to-FAISS mapping.")

        if index.ntotal != len(faiss_to_memory_raw):
            raise VectorStoreCorruptionError(
                "FAISS vector count does not match mapping count. "
                f"FAISS={index.ntotal}, mapping={len(faiss_to_memory_raw)}."
            )

        if len(faiss_to_memory_raw) != len(memory_to_faiss_raw):
            raise VectorStoreCorruptionError(
                "FAISS mapping directions have different sizes."
            )

        reverse_mapping: dict[str, int] = {}

        for raw_faiss_id, memory_id in faiss_to_memory_raw.items():
            try:
                faiss_id = int(raw_faiss_id)
            except (TypeError, ValueError) as exc:
                raise VectorStoreCorruptionError(
                    f"Invalid FAISS ID in mapping: {raw_faiss_id!r}"
                ) from exc

            if not isinstance(memory_id, str) or not memory_id.strip():
                raise VectorStoreCorruptionError(
                    f"Invalid memory ID for FAISS ID {faiss_id}."
                )

            reverse_mapping[memory_id] = faiss_id

        for memory_id, raw_faiss_id in memory_to_faiss_raw.items():
            try:
                faiss_id = int(raw_faiss_id)
            except (TypeError, ValueError) as exc:
                raise VectorStoreCorruptionError(
                    f"Invalid reverse FAISS ID for memory {memory_id!r}."
                ) from exc

            if reverse_mapping.get(memory_id) != faiss_id:
                raise VectorStoreCorruptionError(
                    "Bidirectional FAISS mapping is inconsistent for "
                    f"memory_id={memory_id!r}."
                )

    def _validate_runtime_state(self) -> None:
        if self._index is None:
            raise VectorStoreCorruptionError(
                "Vector store index has not been initialized."
            )

        if self._index.d != self.dimension:
            raise VectorStoreCorruptionError("Runtime FAISS dimension mismatch.")

        if self._index.ntotal != len(self._faiss_to_memory):
            raise VectorStoreCorruptionError(
                "Runtime FAISS count does not match mapping count."
            )

        if len(self._faiss_to_memory) != len(self._memory_to_faiss):
            raise VectorStoreCorruptionError(
                "Runtime mapping directions have different sizes."
            )

    # ------------------------------------------------------------------------
    # INSERTION
    # ------------------------------------------------------------------------

    def add(
        self,
        memory_id: str,
        vector: np.ndarray | Sequence[float],
        *,
        persist: bool = True,
    ) -> int:
        """
        Add one memory vector.

        Returns
        -------
        int
            Persistent FAISS ID assigned to the memory.
        """
        with self._lock:
            self._ensure_open()

            memory_id = self._validate_memory_id(memory_id)

            if memory_id in self._memory_to_faiss:
                raise DuplicateMemoryError(f"Memory ID already exists: {memory_id}")

            prepared = self._prepare_vectors(vector, expected_count=1)

            faiss_id = self._allocate_faiss_id()

            try:
                self.index.add_with_ids(
                    prepared,
                    np.asarray([faiss_id], dtype=np.int64),
                )

                self._faiss_to_memory[faiss_id] = memory_id
                self._memory_to_faiss[memory_id] = faiss_id

                if persist:
                    self.persist()

                logger.debug(
                    "Added memory vector: memory_id=%s faiss_id=%d",
                    memory_id,
                    faiss_id,
                )

                return faiss_id

            except Exception as exc:
                # Roll back in-memory state if insertion failed.
                self._faiss_to_memory.pop(faiss_id, None)
                self._memory_to_faiss.pop(memory_id, None)

                raise VectorStoreError(
                    f"Failed to add vector for memory '{memory_id}'."
                ) from exc

    def add_batch(
        self,
        memory_ids: Sequence[str],
        vectors: np.ndarray | Sequence[Sequence[float]],
        *,
        persist: bool = True,
        reject_duplicates: bool = True,
    ) -> list[int]:
        """
        Add multiple memory vectors atomically at the logical mapping level.

        If validation fails before FAISS insertion, nothing is inserted.

        If FAISS insertion itself fails, the operation is rejected and the
        in-memory mapping remains untouched.
        """
        with self._lock:
            self._ensure_open()

            normalized_ids = [
                self._validate_memory_id(memory_id) for memory_id in memory_ids
            ]

            if not normalized_ids:
                raise VectorStoreValidationError("memory_ids cannot be empty.")

            if len(set(normalized_ids)) != len(normalized_ids):
                raise VectorStoreValidationError("memory_ids contains duplicates.")

            prepared = self._prepare_vectors(
                vectors,
                expected_count=len(normalized_ids),
            )

            existing = [
                memory_id
                for memory_id in normalized_ids
                if memory_id in self._memory_to_faiss
            ]

            if existing and reject_duplicates:
                raise DuplicateMemoryError(
                    "One or more memory IDs already exist: " + ", ".join(existing[:10])
                )

            if existing:
                # This branch is intentionally explicit. In production use,
                # silently skipping duplicate vectors makes batch indexing
                # difficult to reason about.
                raise DuplicateMemoryError(
                    "Duplicate memory IDs are not silently skipped."
                )

            faiss_ids = self._allocate_faiss_ids(len(normalized_ids))

            try:
                self.index.add_with_ids(
                    prepared,
                    np.asarray(faiss_ids, dtype=np.int64),
                )

                for memory_id, faiss_id in zip(
                    normalized_ids,
                    faiss_ids,
                    strict=True,
                ):
                    self._faiss_to_memory[faiss_id] = memory_id
                    self._memory_to_faiss[memory_id] = faiss_id

                if persist:
                    self.persist()

                logger.info(
                    "Added batch of %d memory vectors.",
                    len(normalized_ids),
                )

                return faiss_ids

            except Exception as exc:
                # The FAISS operation is expected to fail before mapping
                # mutation. We additionally rebuild the mapping if necessary.
                for faiss_id in faiss_ids:
                    self._faiss_to_memory.pop(faiss_id, None)

                for memory_id in normalized_ids:
                    self._memory_to_faiss.pop(memory_id, None)

                raise VectorStoreError(
                    "Failed to add batch of memory vectors."
                ) from exc

    # ------------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------------

    def search(
        self,
        query_vector: np.ndarray | Sequence[float],
        *,
        top_k: int = 10,
        min_score: float | None = None,
    ) -> list[VectorSearchResult]:
        """
        Search the FAISS index using an embedding vector.

        Parameters
        ----------
        query_vector:
            One query embedding.

        top_k:
            Maximum number of results.

        min_score:
            Optional lower bound for similarity score.

        Returns
        -------
        list[VectorSearchResult]
            Results sorted by descending similarity.
        """
        with self._lock:
            self._ensure_open()

            if isinstance(top_k, bool) or not isinstance(top_k, int):
                raise VectorStoreValidationError("top_k must be an integer.")

            if top_k < MIN_SEARCH_K:
                raise VectorStoreValidationError(f"top_k must be >= {MIN_SEARCH_K}.")

            if min_score is not None:
                try:
                    min_score = float(min_score)
                except (TypeError, ValueError) as exc:
                    raise VectorStoreValidationError(
                        "min_score must be numeric."
                    ) from exc

                if not np.isfinite(min_score):
                    raise VectorStoreValidationError("min_score must be finite.")

            if self._index is None or self._index.ntotal == 0:
                return []

            query = self._prepare_vectors(
                query_vector,
                expected_count=1,
            )

            search_k = min(top_k, self._index.ntotal)

            scores, faiss_ids = self._index.search(
                query,
                search_k,
            )

            results: list[VectorSearchResult] = []

            for score, raw_faiss_id in zip(
                scores[0],
                faiss_ids[0],
                strict=True,
            ):
                faiss_id = int(raw_faiss_id)

                # FAISS can return -1 when fewer results exist.
                if faiss_id < 0:
                    continue

                memory_id = self._faiss_to_memory.get(faiss_id)

                if memory_id is None:
                    raise VectorStoreCorruptionError(
                        "FAISS returned an ID that does not exist in the "
                        f"persistent mapping: {faiss_id}"
                    )

                similarity = float(score)

                if min_score is not None and similarity < min_score:
                    continue

                results.append(
                    VectorSearchResult(
                        memory_id=memory_id,
                        score=similarity,
                        faiss_id=faiss_id,
                    )
                )

            return results

    # Alias useful for semantic-search-oriented callers.
    search_vectors = search

    # ------------------------------------------------------------------------
    # LOOKUPS
    # ------------------------------------------------------------------------

    def contains(self, memory_id: str) -> bool:
        """
        Check whether a memory has an indexed vector.
        """
        with self._lock:
            self._ensure_open()

            normalized = self._validate_memory_id(memory_id)

            return normalized in self._memory_to_faiss

    def get_faiss_id(self, memory_id: str) -> int | None:
        """
        Return the FAISS ID for a memory, or None if not indexed.
        """
        with self._lock:
            self._ensure_open()

            normalized = self._validate_memory_id(memory_id)

            return self._memory_to_faiss.get(normalized)

    def get_memory_id(self, faiss_id: int) -> str | None:
        """
        Return the memory ID associated with a FAISS ID.
        """
        with self._lock:
            self._ensure_open()

            try:
                normalized = int(faiss_id)
            except (TypeError, ValueError) as exc:
                raise VectorStoreValidationError(
                    "faiss_id must be an integer."
                ) from exc

            return self._faiss_to_memory.get(normalized)

    def memory_ids(self) -> tuple[str, ...]:
        """
        Return all indexed memory IDs.

        Ordering is deterministic by FAISS ID.
        """
        with self._lock:
            self._ensure_open()

            return tuple(
                memory_id for _, memory_id in sorted(self._faiss_to_memory.items())
            )

    # ------------------------------------------------------------------------
    # DELETE
    # ------------------------------------------------------------------------

    def remove(
        self,
        memory_id: str,
        *,
        persist: bool = True,
        missing_ok: bool = False,
    ) -> bool:
        """
        Remove one memory vector.

        Returns
        -------
        bool
            True if a vector was removed.
        """
        with self._lock:
            self._ensure_open()

            normalized = self._validate_memory_id(memory_id)

            faiss_id = self._memory_to_faiss.get(normalized)

            if faiss_id is None:
                if missing_ok:
                    return False

                raise MemoryNotFoundError(f"Memory ID is not indexed: {normalized}")

            try:
                removed = self.index.remove_ids(np.asarray([faiss_id], dtype=np.int64))

                if int(removed) != 1:
                    raise VectorStoreCorruptionError(
                        "FAISS reported an unexpected number of removed "
                        f"vectors: {removed}"
                    )

                del self._memory_to_faiss[normalized]
                del self._faiss_to_memory[faiss_id]

                if persist:
                    self.persist()

                logger.info(
                    "Removed memory vector: memory_id=%s faiss_id=%d",
                    normalized,
                    faiss_id,
                )

                return True

            except VectorStoreError:
                raise
            except Exception as exc:
                raise VectorStoreError(
                    f"Failed to remove memory '{normalized}'."
                ) from exc

    # Alias for callers using delete terminology.
    delete = remove

    def remove_batch(
        self,
        memory_ids: Sequence[str],
        *,
        persist: bool = True,
        missing_ok: bool = False,
    ) -> int:
        """
        Remove multiple vectors.
        """
        with self._lock:
            self._ensure_open()

            normalized_ids = [
                self._validate_memory_id(memory_id) for memory_id in memory_ids
            ]

            if not normalized_ids:
                return 0

            faiss_ids: list[int] = []

            for memory_id in normalized_ids:
                faiss_id = self._memory_to_faiss.get(memory_id)

                if faiss_id is None:
                    if missing_ok:
                        continue

                    raise MemoryNotFoundError(f"Memory ID is not indexed: {memory_id}")

                faiss_ids.append(faiss_id)

            if not faiss_ids:
                return 0

            try:
                removed = int(
                    self.index.remove_ids(np.asarray(faiss_ids, dtype=np.int64))
                )

                if removed != len(faiss_ids):
                    raise VectorStoreCorruptionError(
                        "FAISS removed an unexpected number of vectors. "
                        f"Expected={len(faiss_ids)}, removed={removed}."
                    )

                for memory_id in normalized_ids:
                    faiss_id = self._memory_to_faiss.pop(
                        memory_id,
                        None,
                    )

                    if faiss_id is not None:
                        self._faiss_to_memory.pop(faiss_id, None)

                if persist:
                    self.persist()

                return removed

            except VectorStoreError:
                raise
            except Exception as exc:
                raise VectorStoreError(
                    "Failed to remove batch of memory vectors."
                ) from exc

    # ------------------------------------------------------------------------
    # REBUILD
    # ------------------------------------------------------------------------

    def rebuild(
        self,
        memory_vectors: Mapping[
            str,
            np.ndarray | Sequence[float],
        ],
        *,
        persist: bool = True,
    ) -> None:
        """
        Completely rebuild the vector index from authoritative memory data.

        This is intentionally independent of SQLite.

        Future MemoryService/database integration can retrieve all valid
        embeddings from SQLite and call this method when repairing/rebuilding
        the FAISS layer.
        """
        with self._lock:
            self._ensure_open()

            if not isinstance(memory_vectors, Mapping):
                raise VectorStoreValidationError(
                    "memory_vectors must be a mapping of memory_id -> vector."
                )

            if not memory_vectors:
                self._reset_memory_state()

                if persist:
                    self.persist()

                logger.info("Rebuilt FAISS index as empty.")
                return

            normalized_ids: list[str] = []
            vector_list: list[np.ndarray] = []

            for memory_id, vector in memory_vectors.items():
                normalized_id = self._validate_memory_id(memory_id)

                normalized_ids.append(normalized_id)

                prepared = self._prepare_vectors(
                    vector,
                    expected_count=1,
                )

                vector_list.append(prepared[0])

            if len(set(normalized_ids)) != len(normalized_ids):
                raise VectorStoreValidationError(
                    "Duplicate memory IDs detected during rebuild."
                )

            matrix = np.ascontiguousarray(
                np.vstack(vector_list),
                dtype=np.float32,
            )

            new_index = self._create_empty_index()

            new_faiss_to_memory: dict[int, str] = {}
            new_memory_to_faiss: dict[str, int] = {}

            faiss_ids = list(
                range(
                    1,
                    len(normalized_ids) + 1,
                )
            )

            new_index.add_with_ids(
                matrix,
                np.asarray(faiss_ids, dtype=np.int64),
            )

            for memory_id, faiss_id in zip(
                normalized_ids,
                faiss_ids,
                strict=True,
            ):
                new_faiss_to_memory[faiss_id] = memory_id
                new_memory_to_faiss[memory_id] = faiss_id

            # Replace only after successful construction.
            self._index = new_index
            self._faiss_to_memory = new_faiss_to_memory
            self._memory_to_faiss = new_memory_to_faiss
            self._next_faiss_id = len(faiss_ids) + 1

            if persist:
                self.persist()

            logger.info(
                "Rebuilt FAISS index: vectors=%d",
                len(normalized_ids),
            )

    # ------------------------------------------------------------------------
    # RESET
    # ------------------------------------------------------------------------

    def reset(self, *, persist: bool = True) -> None:
        """
        Completely reset the vector store.
        """
        with self._lock:
            self._ensure_open()

            self._reset_memory_state()

            if persist:
                self.persist()

            logger.info("Vector store reset.")

    def _reset_memory_state(self) -> None:
        self._index = self._create_empty_index()
        self._faiss_to_memory.clear()
        self._memory_to_faiss.clear()
        self._next_faiss_id = 1

    # ------------------------------------------------------------------------
    # ID ALLOCATION
    # ------------------------------------------------------------------------

    def _allocate_faiss_id(self, count: int = 1) -> int:
        """
        Allocate one persistent FAISS ID.
        """
        return self._allocate_faiss_ids(count)[0]

    def _allocate_faiss_ids(self, count: int) -> list[int]:
        """
        Allocate a consecutive range of persistent FAISS IDs.
        """
        if count <= 0:
            raise VectorStoreValidationError("ID allocation count must be positive.")

        start = self._next_faiss_id
        end = start + count

        self._next_faiss_id = end

        return list(range(start, end))

    # ------------------------------------------------------------------------
    # STATISTICS
    # ------------------------------------------------------------------------

    def stats(self) -> VectorStoreStats:
        """
        Return a safe snapshot of vector-store statistics.
        """
        with self._lock:
            self._ensure_open()

            index = self._index or self._create_empty_index()

            return VectorStoreStats(
                dimension=self.dimension,
                vector_count=int(index.ntotal),
                mapped_memory_count=len(self._memory_to_faiss),
                index_type=SUPPORTED_INDEX_TYPE,
                index_path=str(self.index_path),
                mapping_path=str(self.mapping_path),
                is_trained=bool(index.is_trained),
                next_faiss_id=self._next_faiss_id,
            )

    @property
    def count(self) -> int:
        """
        Number of indexed vectors.
        """
        with self._lock:
            self._ensure_open()

            return int(self._index.ntotal) if self._index else 0

    @property
    def is_empty(self) -> bool:
        """
        Whether the vector store contains no vectors.
        """
        return self.count == 0

    # ------------------------------------------------------------------------
    # HEALTH / VALIDATION
    # ------------------------------------------------------------------------

    def validate(self) -> None:
        """
        Perform a consistency check.

        Raises
        ------
        VectorStoreCorruptionError
            If the runtime state is inconsistent.
        """
        with self._lock:
            self._ensure_open()

            self._validate_runtime_state()

            if self._index is None:
                return

            if self._index.ntotal == 0:
                if self._faiss_to_memory or self._memory_to_faiss:
                    raise VectorStoreCorruptionError(
                        "Empty FAISS index has non-empty mappings."
                    )

                return

            if self._index.ntotal != len(self._memory_to_faiss):
                raise VectorStoreCorruptionError(
                    "FAISS and memory mappings are inconsistent."
                )

            for faiss_id, memory_id in self._faiss_to_memory.items():
                if self._memory_to_faiss.get(memory_id) != faiss_id:
                    raise VectorStoreCorruptionError(
                        "Bidirectional mapping inconsistency detected."
                    )

    # ------------------------------------------------------------------------
    # EXPORT / SNAPSHOT
    # ------------------------------------------------------------------------

    def export_mapping(self) -> dict[str, int]:
        """
        Return a copy of memory_id -> FAISS ID mapping.

        Useful for diagnostics and future database integration.
        """
        with self._lock:
            self._ensure_open()

            return dict(self._memory_to_faiss)

    # ------------------------------------------------------------------------
    # INTERNAL SAFETY
    # ------------------------------------------------------------------------

    def _ensure_open(self) -> None:
        if self._closed:
            raise VectorStoreClosedError("VectorStore has been closed.")


# ============================================================================
# DEFAULT FACTORY
# ============================================================================


def create_vector_store(
    *,
    index_path: str | Path = Path("data") / "faiss" / DEFAULT_INDEX_FILENAME,
    mapping_path: str | Path = Path("data") / "faiss" / DEFAULT_MAPPING_FILENAME,
    dimension: int = DEFAULT_DIMENSION,
) -> VectorStore:
    """
    Convenience factory for MemoryOS application startup.

    Keeping construction here makes it easy for FastAPI dependency injection
    to instantiate the vector store later without spreading filesystem logic
    throughout the application.
    """
    return VectorStore(
        index_path=index_path,
        mapping_path=mapping_path,
        dimension=dimension,
        auto_load=True,
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================


__all__ = [
    "DEFAULT_DIMENSION",
    "DEFAULT_INDEX_FILENAME",
    "DEFAULT_MAPPING_FILENAME",
    "DuplicateMemoryError",
    "MemoryNotFoundError",
    "VectorSearchResult",
    "VectorStore",
    "VectorStoreClosedError",
    "VectorStoreCorruptionError",
    "VectorStoreError",
    "VectorStorePersistenceError",
    "VectorStoreStats",
    "VectorStoreValidationError",
    "create_vector_store",
]
