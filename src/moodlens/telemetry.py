from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any

from .database import Database
from .models import Answer


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


class Telemetry:
    def __init__(self, database: Database) -> None:
        self.database = database

    def record(
        self,
        *,
        session_id: str,
        question: str,
        answer: Answer | None,
        error: str | None = None,
    ) -> str:
        interaction_id = str(uuid.uuid4())
        hits = list(answer.hits) if answer else []
        mean_similarity = sum(hit.score for hit in hits) / len(hits) if hits else None
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO interactions (
                    interaction_id, session_id, created_at, question, answer, model,
                    latency_ms, prompt_tokens, completion_tokens, retrieved_ids_json,
                    mean_similarity, error
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    interaction_id,
                    session_id,
                    _now(),
                    question,
                    answer.text if answer else "",
                    answer.model if answer else "unknown",
                    answer.latency_ms if answer else 0.0,
                    answer.prompt_tokens if answer else None,
                    answer.completion_tokens if answer else None,
                    json.dumps([hit.chunk.chunk_id for hit in hits]),
                    mean_similarity,
                    error,
                ),
            )
        return interaction_id

    def feedback(self, interaction_id: str, score: int, comment: str | None = None) -> None:
        if score not in {-1, 1}:
            raise ValueError("feedback score must be -1 or 1")
        with self.database.connect() as connection:
            connection.execute(
                """
                INSERT INTO feedback (interaction_id, created_at, score, comment)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(interaction_id) DO UPDATE SET
                    created_at=excluded.created_at,
                    score=excluded.score,
                    comment=excluded.comment
                """,
                (interaction_id, _now(), score, comment),
            )

    def rows(self, *, limit: int = 500) -> list[dict[str, Any]]:
        with self.database.connect() as connection:
            rows = connection.execute(
                """
                SELECT i.*, f.score AS feedback_score
                FROM interactions i
                LEFT JOIN feedback f USING (interaction_id)
                ORDER BY i.created_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def summary(self) -> dict[str, Any]:
        with self.database.connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS requests,
                       SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors,
                       AVG(latency_ms) AS average_latency_ms,
                       AVG(mean_similarity) AS average_similarity,
                       SUM(COALESCE(prompt_tokens, 0) + COALESCE(completion_tokens, 0)) AS tokens
                FROM interactions
                """
            ).fetchone()
            feedback = connection.execute(
                """
                SELECT COUNT(*) AS feedback_count,
                       AVG(CASE WHEN score = 1 THEN 1.0 ELSE 0.0 END) AS positive_rate
                FROM feedback
                """
            ).fetchone()
        return {**dict(row), **dict(feedback)}

