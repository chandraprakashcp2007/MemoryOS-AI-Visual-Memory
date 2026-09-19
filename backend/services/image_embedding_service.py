"""
MemoryOS local multimodal visual embedding service.

Uses pretrained OpenAI CLIP ViT-B/32.

Purpose
-------
Image -> 512-dimensional visual embedding
Text  -> 512-dimensional visual-semantic embedding

These vectors live in a SEPARATE index from MiniLM's 384-dimensional
semantic text vectors.

No training is required.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from threading import RLock
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_MODEL = "openai/clip-vit-base-patch32"


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

        self._lock = RLock()

    # --------------------------------------------------------
    # MODEL LOADING
    # --------------------------------------------------------

    def _ensure_loaded(self):

        with self._lock:

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
                in {"1", "true", "yes"}
            )

            self._processor = CLIPProcessor.from_pretrained(
                self.model_name,
                local_files_only=local_only,
            )

            self._model = CLIPModel.from_pretrained(
                self.model_name,
                local_files_only=local_only,
            )

            self._model.eval()

            self._torch = torch

            try:
                self.embedding_dimension = int(
                    self._model.config.projection_dim
                )
                self.dimension = self.embedding_dimension

            except Exception:
                pass

    # --------------------------------------------------------
    # IMAGE NORMALIZATION
    # --------------------------------------------------------

    @staticmethod
    def _load_image(source: Any) -> Image.Image:

        if isinstance(source, Image.Image):
            return source.convert("RGB")

        if isinstance(source, (str, Path)):

            path = Path(source)

            if not path.exists():
                raise FileNotFoundError(
                    f"Image not found: {path}"
                )

            with Image.open(path) as image:
                return image.convert("RGB")

        if isinstance(source, bytes):

            with Image.open(io.BytesIO(source)) as image:
                return image.convert("RGB")

        if hasattr(source, "read"):

            data = source.read()

            if hasattr(source, "seek"):
                try:
                    source.seek(0)
                except Exception:
                    pass

            with Image.open(io.BytesIO(data)) as image:
                return image.convert("RGB")

        raise TypeError(
            f"Unsupported image source: {type(source)!r}"
        )

    # --------------------------------------------------------
    # OUTPUT COMPATIBILITY
    # --------------------------------------------------------

    @staticmethod
    def _unwrap_features(output):

        # Transformers v5:
        # BaseModelOutputWithPooling.pooler_output

        if hasattr(output, "pooler_output"):
            return output.pooler_output

        # Earlier transformers versions:
        # direct tensor

        return output

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    @staticmethod
    def _normalize(vector):

        norm = vector.norm(
            dim=-1,
            keepdim=True,
        )

        norm = norm.clamp(
            min=1e-12
        )

        return vector / norm

    # --------------------------------------------------------
    # IMAGE EMBEDDING
    # --------------------------------------------------------

    def embed_image(
        self,
        source: Any,
    ) -> np.ndarray:

        self._ensure_loaded()

        image = self._load_image(source)

        inputs = self._processor(
            images=image,
            return_tensors="pt",
        )

        with self._torch.inference_mode():

            output = self._model.get_image_features(
                pixel_values=inputs["pixel_values"],
            )

            vector = self._unwrap_features(output)

            vector = self._normalize(vector)

        array = (
            vector[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if array.shape != (
            self.embedding_dimension,
        ):
            raise ValueError(
                "Unexpected visual embedding shape: "
                f"{array.shape}"
            )

        return array

    # --------------------------------------------------------
    # TEXT QUERY EMBEDDING
    # --------------------------------------------------------

    def embed_text(
        self,
        text: str,
    ) -> np.ndarray:

        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "Visual search text cannot be empty."
            )

        self._ensure_loaded()

        inputs = self._processor(
            text=[text],
            return_tensors="pt",
            padding=True,
        )

        with self._torch.inference_mode():

            output = self._model.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get(
                    "attention_mask"
                ),
            )

            vector = self._unwrap_features(output)

            vector = self._normalize(vector)

        array = (
            vector[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        if array.shape != (
            self.embedding_dimension,
        ):
            raise ValueError(
                "Unexpected visual query embedding shape: "
                f"{array.shape}"
            )

        return array

    # Compatibility alias
    def embed_query(
        self,
        text: str,
    ) -> np.ndarray:

        return self.embed_text(text)

    def warmup(self) -> int:

        self._ensure_loaded()

        return self.embedding_dimension


_service: ImageEmbeddingService | None = None
_service_lock = RLock()


def get_image_embedding_service() -> ImageEmbeddingService:

    global _service

    with _service_lock:

        if _service is None:
            _service = ImageEmbeddingService()

        return _service


__all__ = [
    "ImageEmbeddingService",
    "get_image_embedding_service",
]
