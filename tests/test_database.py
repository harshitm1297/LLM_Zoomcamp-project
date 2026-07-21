import tempfile
import unittest
from pathlib import Path

import numpy as np

from cultural_mood_tracker.database import Database
from cultural_mood_tracker.models import Chunk


def _chunk(identifier: str, title: str) -> Chunk:
    return Chunk(identifier, identifier.split("::")[0], title, 0, {"title": title})


class DatabaseTests(unittest.TestCase):
    def test_vector_search_and_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Database(Path(directory) / "test.sqlite3")
            chunks = [_chunk("a::c000", "Alpha"), _chunk("b::c000", "Beta")]
            database.replace_corpus(
                run_id="run-1",
                source="test",
                created_at="2026-01-01T00:00:00Z",
                document_count=2,
                chunks=chunks,
                embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
                embedding_model="fake",
            )
            hits = database.search(np.asarray([0.9, 0.1], dtype=np.float32), top_k=1)
            self.assertEqual(hits[0].chunk.chunk_id, "a::c000")

            database.replace_corpus(
                run_id="run-2",
                source="test",
                created_at="2026-01-02T00:00:00Z",
                document_count=1,
                chunks=[_chunk("c::c000", "Gamma")],
                embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
                embedding_model="fake",
            )
            self.assertEqual(database.corpus_status()["chunk_count"], 1)
