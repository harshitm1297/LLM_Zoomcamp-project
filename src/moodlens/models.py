from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MediaKind = Literal["movie", "tv"]


@dataclass(frozen=True)
class Document:
    document_id: str
    title: str
    media_kind: MediaKind
    text: str
    source: str
    source_url: str = ""
    year: int | None = None
    genres: tuple[str, ...] = ()
    document_type: str = "overview"

    def metadata(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "title": self.title,
            "media_kind": self.media_kind,
            "source": self.source,
            "source_url": self.source_url,
            "year": self.year,
            "genres": list(self.genres),
            "document_type": self.document_type,
        }


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    position: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Answer:
    text: str
    hits: tuple[SearchHit, ...]
    model: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

