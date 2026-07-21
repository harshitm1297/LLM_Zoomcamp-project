from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from typing import Protocol

import numpy as np

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    @property
    def dimensions(self) -> int: ...

    def documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimensions = int(self._model.get_sentence_embedding_dimension())

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                list(texts), normalize_embeddings=True, show_progress_bar=False
            ),
            dtype=np.float32,
        )

    def query(self, text: str) -> np.ndarray:
        encoded = self._model.encode(
            [f"{QUERY_PREFIX}{text.strip()}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(encoded[0], dtype=np.float32)


class HashingEmbedder:
    """Small deterministic fallback for smoke tests, not a replacement for semantic BGE search."""

    model_name = "cultural-mood-tracker-hashing-v1"

    def __init__(self, dimensions: int = 512) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimensions, dtype=np.float32)
        words = re.findall(r"[a-z0-9]+", text.casefold())
        features = [*words, *(f"{a}_{b}" for a, b in zip(words, words[1:], strict=False))]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            vector[value % self._dimensions] += 1.0 if value & 1 else -1.0
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def documents(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._encode(text) for text in texts])

    def query(self, text: str) -> np.ndarray:
        return self._encode(text)


def make_embedder(backend: str, model_name: str) -> Embedder:
    if backend == "hashing":
        return HashingEmbedder()
    if backend == "sentence-transformers":
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unsupported embedding backend: {backend}")
