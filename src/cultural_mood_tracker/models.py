from __future__ import annotations

import json
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

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Document:
        genres = record.get("genres") or []
        if isinstance(genres, str):
            try:
                genres = json.loads(genres)
            except json.JSONDecodeError:
                genres = [genres]
        media_kind = str(record["media_kind"])
        if media_kind not in {"movie", "tv"}:
            raise ValueError(f"Unsupported media kind: {media_kind}")
        year = record.get("year")
        return cls(
            document_id=str(record["document_id"]),
            title=str(record["title"]),
            media_kind=media_kind,
            text=str(record["text"]),
            source=str(record["source"]),
            source_url=str(record.get("source_url") or ""),
            year=int(year) if year is not None else None,
            genres=tuple(str(value) for value in genres),
            document_type=str(record.get("document_type") or "overview"),
        )


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
