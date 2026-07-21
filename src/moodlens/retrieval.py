from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .database import Database
from .embeddings import Embedder
from .models import Chunk, SearchHit

SPACE = re.compile(r"\s+")
TOKEN = re.compile(r"[a-z0-9]+")
STRATEGIES = ("bm25", "vector", "hybrid", "vector_reranked")

QUERY_EXPANSIONS: dict[str, tuple[str, ...]] = {
    "adaptation": ("accessibility", "disability", "hearing"),
    "celebrity": ("fame", "public", "attention", "performers"),
    "climate": ("environmental", "weather", "justice"),
    "environmental": ("climate", "weather", "justice", "inequality"),
    "grief": ("loss", "mourning", "absence"),
    "soundtrack": ("music", "composer", "score"),
    "translator": ("language", "communication", "voice"),
}


def normalize_query(query: str) -> str:
    normalized = SPACE.sub(" ", query).strip()
    if not normalized:
        raise ValueError("question must not be empty")
    return normalized


def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.casefold())


def rewrite_query(query: str) -> str:
    normalized = normalize_query(query)
    terms = set(tokenize(normalized))
    additions = [value for term in sorted(terms) for value in QUERY_EXPANSIONS.get(term, ())]
    return f"{normalized} {' '.join(additions)}".strip()


@dataclass
class BM25Index:
    chunks: list[Chunk]
    k1: float = 1.5
    b: float = 0.75

    def __post_init__(self) -> None:
        self.documents = [tokenize(self._searchable_text(chunk)) for chunk in self.chunks]
        self.frequencies = [Counter(document) for document in self.documents]
        self.lengths = [len(document) for document in self.documents]
        self.average_length = sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        document_frequency: Counter[str] = Counter()
        for document in self.documents:
            document_frequency.update(set(document))
        count = len(self.documents)
        self.idf = {
            term: math.log(1 + (count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    @staticmethod
    def _searchable_text(chunk: Chunk) -> str:
        metadata = chunk.metadata
        genres = metadata.get("genres") or []
        genre_text = (
            " ".join(str(value) for value in genres)
            if isinstance(genres, list)
            else str(genres)
        )
        return f"{metadata.get('title', '')} {genre_text} {chunk.text}"

    def search(self, query: str, *, top_k: int) -> list[SearchHit]:
        terms = tokenize(query)
        scored: list[tuple[float, int]] = []
        for index, frequencies in enumerate(self.frequencies):
            score = 0.0
            for term in terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                denominator = frequency + self.k1 * (
                    1 - self.b
                    + self.b * self.lengths[index] / max(self.average_length, 1.0)
                )
                score += self.idf.get(term, 0.0) * frequency * (self.k1 + 1) / denominator
            if score > 0:
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], self.chunks[item[1]].chunk_id))
        maximum = scored[0][0] if scored else 1.0
        return [
            SearchHit(chunk=self.chunks[index], score=score / maximum)
            for score, index in scored[:top_k]
        ]


def reciprocal_rank_fusion(
    rankings: list[list[SearchHit]], *, top_k: int, rank_constant: int = 60
) -> list[SearchHit]:
    scores: Counter[str] = Counter()
    hits: dict[str, SearchHit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            scores[hit.chunk.chunk_id] += 1.0 / (rank_constant + rank)
            hits.setdefault(hit.chunk.chunk_id, hit)
    ordered = sorted(scores, key=lambda identifier: (-scores[identifier], identifier))[:top_k]
    maximum = max((scores[identifier] for identifier in ordered), default=1.0)
    return [
        SearchHit(hits[identifier].chunk, scores[identifier] / maximum)
        for identifier in ordered
    ]


class Retriever:
    def __init__(
        self,
        database: Database,
        embedder: Embedder,
        *,
        strategy: str = "hybrid",
        enable_query_rewriting: bool = True,
    ) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {', '.join(STRATEGIES)}")
        self.database = database
        self.embedder = embedder
        self.strategy = strategy
        self.enable_query_rewriting = enable_query_rewriting
        self._bm25: BM25Index | None = None

    def _lexical(self) -> BM25Index:
        if self._bm25 is None:
            chunks, _ = self.database.all_chunks_with_embeddings()
            self._bm25 = BM25Index(chunks)
        return self._bm25

    def _vector(self, query: str, *, top_k: int) -> list[SearchHit]:
        return self.database.search(self.embedder.query(query), top_k=top_k)

    @staticmethod
    def _rerank(query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        query_terms = set(tokenize(query))
        rescored: list[SearchHit] = []
        for hit in hits:
            title_terms = set(tokenize(str(hit.chunk.metadata.get("title") or "")))
            text_terms = set(tokenize(hit.chunk.text))
            title_overlap = len(query_terms & title_terms) / max(len(query_terms), 1)
            text_overlap = len(query_terms & text_terms) / max(len(query_terms), 1)
            vector_score = max(0.0, min(1.0, hit.score))
            score = 0.65 * vector_score + 0.25 * text_overlap + 0.10 * title_overlap
            rescored.append(SearchHit(hit.chunk, score))
        return sorted(rescored, key=lambda hit: (-hit.score, hit.chunk.chunk_id))[:top_k]

    def search(
        self, query: str, *, top_k: int = 5, strategy: str | None = None
    ) -> list[SearchHit]:
        normalized = normalize_query(query)
        effective = rewrite_query(normalized) if self.enable_query_rewriting else normalized
        selected = strategy or self.strategy
        if selected not in STRATEGIES:
            raise ValueError(f"Unknown retrieval strategy: {selected}")
        if selected == "bm25":
            return self._lexical().search(effective, top_k=top_k)
        if selected == "vector":
            return self._vector(effective, top_k=top_k)
        candidate_k = max(top_k, 20)
        if selected == "vector_reranked":
            candidates = self._vector(effective, top_k=candidate_k)
            return self._rerank(effective, candidates, top_k=top_k)
        vector = self._vector(effective, top_k=candidate_k)
        lexical = self._lexical().search(effective, top_k=candidate_k)
        return reciprocal_rank_fusion([vector, lexical], top_k=top_k)
