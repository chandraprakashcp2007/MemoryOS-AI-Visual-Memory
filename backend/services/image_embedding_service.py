"""
MemoryOS fast local CLIP service.

Features:
- lazy loading
- automatic CUDA / CPU selection
- batch image embeddings
- batch text embeddings
- normalized float32 vectors
- Transformers v4/v5 compatibility
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from threading import RLock
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


DEFAULT_MODEL = "openai/clip-vit-base-patch32"
DEFAULT_BATCH_SIZE = 32


class ImageEmbeddingService:

    def __init__(
        self,
        model_name: str | None = None,
    ) -> None:

        self.model_name = (
            model_name
            or os.getenv("MEMORYOS_VISUAL_EMBEDDING_MODEL")
            or DEFAULT_MODEL
        )

        self.embedding_dimension = 512
        self.dimension = 512

        self._model = None
        self._processor = None
        self._torch = None
        self._device = "cpu"

        self._load_lock = RLock()
        self._inference_lock = RLock()

    @property
    def device(self) -> str:
        self._ensure_loaded()
        return self._device

    def _ensure_loaded(self) -> None:

        with self._load_lock:

            if self._model is not None:
                return

            import torch

            from transformers import (
                CLIPModel,
                CLIPProcessor,
            )

            local_only = (
                os.getenv(
                    "MEMORYOS_VISUAL_LOCAL_ONLY",
                    "1",
                ).strip().lower()
                in {"1", "true", "yes", "on"}
            )

            requested = (
                os.getenv(
                    "MEMORYOS_VISUAL_DEVICE",
                    "auto",
                ).strip().lower()
            )

            if requested == "auto":

                self._device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "cpu"
                )

            elif requested.startswith("cuda"):

                self._device = (
                    requested
                    if torch.cuda.is_available()
                    else "cpu"
                )

            else:

                self._device = requested or "cpu"

            self._processor = CLIPProcessor.from_pretrained(
                self.model_name,
                local_files_only=local_only,
            )

            self._model = CLIPModel.from_pretrained(
                self.model_name,
                local_files_only=local_only,
            )

            self._model.eval()

            self._model.to(
                self._device
            )

            self._torch = torch

            try:

                self.embedding_dimension = int(
                    self._model.config.projection_dim
                )

                self.dimension = (
                    self.embedding_dimension
                )

            except Exception:
                pass

    @staticmethod
    def _unwrap(output):

        if hasattr(
            output,
            "pooler_output",
        ):
            return output.pooler_output

        return output

    @staticmethod
    def _normalize(tensor):

        return (
            tensor /
            tensor.norm(
                dim=-1,
                keepdim=True,
            ).clamp(min=1e-12)
        )

    @staticmethod
    def _load_image(
        source: Any,
    ) -> Image.Image:

        if isinstance(
            source,
            Image.Image,
        ):

            return source.convert(
                "RGB"
            )

        if isinstance(
            source,
            (str, Path),
        ):

            source = Path(source)

            if not source.exists():

                raise FileNotFoundError(
                    source
                )

            with Image.open(
                source
            ) as image:

                return image.convert(
                    "RGB"
                )

        if isinstance(
            source,
            bytes,
        ):

            with Image.open(
                io.BytesIO(source)
            ) as image:

                return image.convert(
                    "RGB"
                )

        raise TypeError(
            f"Unsupported image source: {type(source)!r}"
        )

    def _resolve_batch_size(
        self,
        value: int | None,
    ) -> int:

        if value is None:

            try:
                value = int(
                    os.getenv(
                        "MEMORYOS_VISUAL_BATCH_SIZE",
                        str(DEFAULT_BATCH_SIZE),
                    )
                )
            except ValueError:
                value = DEFAULT_BATCH_SIZE

        return max(
            1,
            min(
                int(value),
                128,
            ),
        )

    def embed_images(
        self,
        sources: Sequence[Any] | Iterable[Any],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:

        self._ensure_loaded()

        source_list = list(
            sources
        )

        if not source_list:

            return np.empty(
                (
                    0,
                    self.embedding_dimension,
                ),
                dtype=np.float32,
            )

        batch_size = (
            self._resolve_batch_size(
                batch_size
            )
        )

        output_vectors = []

        with self._inference_lock:

            for start in range(
                0,
                len(source_list),
                batch_size,
            ):

                batch_sources = (
                    source_list[
                        start:
                        start + batch_size
                    ]
                )

                images = [
                    self._load_image(source)
                    for source
                    in batch_sources
                ]

                inputs = self._processor(
                    images=images,
                    return_tensors="pt",
                )

                pixels = (
                    inputs[
                        "pixel_values"
                    ].to(
                        self._device
                    )
                )

                with self._torch.inference_mode():

                    raw = (
                        self._model
                        .get_image_features(
                            pixel_values=pixels
                        )
                    )

                    vectors = (
                        self._normalize(
                            self._unwrap(
                                raw
                            )
                        )
                    )

                output_vectors.append(
                    vectors
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

        matrix = np.concatenate(
            output_vectors,
            axis=0,
        )

        expected = (
            len(source_list),
            self.embedding_dimension,
        )

        if matrix.shape != expected:

            raise ValueError(
                f"Unexpected CLIP matrix: "
                f"{matrix.shape}, expected {expected}"
            )

        return matrix

    def embed_image(
        self,
        source: Any,
    ) -> np.ndarray:

        return self.embed_images(
            [source],
            batch_size=1,
        )[0]

    def embed_texts(
        self,
        texts: Sequence[str] | Iterable[str],
        *,
        batch_size: int | None = None,
    ) -> np.ndarray:

        self._ensure_loaded()

        texts = [
            str(value).strip()
            for value in texts
        ]

        if not texts:

            return np.empty(
                (
                    0,
                    self.embedding_dimension,
                ),
                dtype=np.float32,
            )

        if any(
            not text
            for text in texts
        ):

            raise ValueError(
                "CLIP query cannot be empty."
            )

        batch_size = (
            self._resolve_batch_size(
                batch_size
            )
        )

        outputs = []

        with self._inference_lock:

            for start in range(
                0,
                len(texts),
                batch_size,
            ):

                batch = (
                    texts[
                        start:
                        start + batch_size
                    ]
                )

                inputs = self._processor(
                    text=batch,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )

                kwargs = {
                    "input_ids":
                        inputs[
                            "input_ids"
                        ].to(
                            self._device
                        )
                }

                if (
                    "attention_mask"
                    in inputs
                ):

                    kwargs[
                        "attention_mask"
                    ] = (
                        inputs[
                            "attention_mask"
                        ].to(
                            self._device
                        )
                    )

                with self._torch.inference_mode():

                    raw = (
                        self._model
                        .get_text_features(
                            **kwargs
                        )
                    )

                    vectors = (
                        self._normalize(
                            self._unwrap(
                                raw
                            )
                        )
                    )

                outputs.append(
                    vectors
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(
                        np.float32,
                        copy=False,
                    )
                )

        return np.concatenate(
            outputs,
            axis=0,
        )

    def embed_text(
        self,
        text: str,
    ) -> np.ndarray:

        return self.embed_texts(
            [text],
            batch_size=1,
        )[0]

    def embed_query(
        self,
        text: str,
    ) -> np.ndarray:

        return self.embed_text(
            text
        )

    def warmup(self) -> int:

        self._ensure_loaded()

        return (
            self.embedding_dimension
        )


_service = None
_service_lock = RLock()


def get_image_embedding_service():

    global _service

    with _service_lock:

        if _service is None:

            _service = (
                ImageEmbeddingService()
            )

        return _service


__all__ = [
    "ImageEmbeddingService",
    "get_image_embedding_service",
]
