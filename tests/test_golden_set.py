import json
import unittest
from pathlib import Path

from cultural_mood_tracker.chunking import chunk_documents
from cultural_mood_tracker.sample_data import demo_documents


class GoldenSetTests(unittest.TestCase):
    def test_retrieval_golden_ids_exist_in_demo_corpus(self) -> None:
        root = Path(__file__).resolve().parents[1]
        chunks = chunk_documents(demo_documents(), words=180, overlap=30)
        available = {chunk.chunk_id for chunk in chunks}
        with (root / "data/evaluation/retrieval_golden.jsonl").open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        expected = {chunk_id for row in rows for chunk_id in row["relevant_chunk_ids"]}
        self.assertLessEqual(expected, available)
