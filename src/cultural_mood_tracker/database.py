from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .models import Chunk, SearchHit

SCHEMA = """
CREATE TABLE IF NOT EXISTS corpus_runs (
    run_id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    document_count INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding_dimensions INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES corpus_runs(run_id) ON DELETE CASCADE,
    document_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    text TEXT NOT NULL,
    metadata_json TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_run_id ON chunks(run_id);

CREATE TABLE IF NOT EXISTS interactions (
    interaction_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    model TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    retrieved_ids_json TEXT NOT NULL,
    mean_similarity REAL,
    error TEXT
);
CREATE TABLE IF NOT EXISTS feedback (
    interaction_id TEXT PRIMARY KEY REFERENCES interactions(interaction_id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    score INTEGER NOT NULL CHECK (score IN (-1, 1)),
    comment TEXT
);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def replace_corpus(
        self,
        *,
        run_id: str,
        source: str,
        created_at: str,
        document_count: int,
        chunks: Sequence[Chunk],
        embeddings: np.ndarray,
        embedding_model: str,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must contain the same number of rows")
        if embeddings.ndim != 2 or not len(chunks):
            raise ValueError("embeddings must be a non-empty two-dimensional matrix")
        with self.connect() as connection:
            connection.execute("DELETE FROM chunks")
            connection.execute("DELETE FROM corpus_runs")
            connection.execute(
                """
                INSERT INTO corpus_runs (
                    run_id, source, created_at, document_count, chunk_count,
                    embedding_model, embedding_dimensions
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    source,
                    created_at,
                    document_count,
                    len(chunks),
                    embedding_model,
                    int(embeddings.shape[1]),
                ),
            )
            connection.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, run_id, document_id, position, text, metadata_json, embedding
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        run_id,
                        chunk.document_id,
                        chunk.position,
                        chunk.text,
                        json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                        np.asarray(vector, dtype=np.float32).tobytes(),
                    )
                    for chunk, vector in zip(chunks, embeddings, strict=True)
                ],
            )

    def corpus_status(self) -> dict[str, object] | None:
        with self.connect() as connection:
            row = connection.execute(
                "SELECT * FROM corpus_runs ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        return dict(row) if row else None

    def all_chunks_with_embeddings(self) -> tuple[list[Chunk], np.ndarray]:
        status = self.corpus_status()
        if not status:
            raise RuntimeError("No corpus is indexed. Run the ingestion command first.")
        dimensions = int(status["embedding_dimensions"])
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT chunk_id, document_id, position, text, metadata_json, embedding FROM chunks"
            ).fetchall()
        chunks = [
            Chunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                position=int(row["position"]),
                text=row["text"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]
        vectors = np.vstack(
            [np.frombuffer(row["embedding"], dtype=np.float32, count=dimensions) for row in rows]
        )
        return chunks, vectors

    def search(self, query_vector: np.ndarray, *, top_k: int) -> list[SearchHit]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        chunks, vectors = self.all_chunks_with_embeddings()
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if vectors.shape[1] != query.shape[0]:
            raise ValueError("query vector dimensions do not match the indexed corpus")
        scores = vectors @ query
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return [
            SearchHit(chunk=chunks[int(index)], score=float(scores[index]))
            for index in indices
        ]
