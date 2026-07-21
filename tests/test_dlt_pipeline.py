from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from moodlens.config import settings
from moodlens.dlt_pipeline import (
    inspect_local_pipeline,
    load_demo_source,
    load_document_resource,
)
from moodlens.models import Document
from moodlens.sample_data import demo_source_path


class DltPipelineTests(unittest.TestCase):
    def config(self, root: Path):
        return replace(
            settings(),
            database_path=root / "search.sqlite3",
            dlt_database_path=root / "pipeline.duckdb",
            artifacts_dir=root / "artifacts",
            embedding_backend="hashing",
        )

    def test_demo_filesystem_load_is_incremental_and_inspectable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            first = load_demo_source(demo_source_path(), config=config, run_id="first")
            second = load_demo_source(demo_source_path(), config=config, run_id="second")
            inspection = inspect_local_pipeline(config, "demo")

            self.assertEqual(len(first.documents), 8)
            self.assertEqual(len(first.metadata["load_ids"]), 1)
            self.assertEqual(second.metadata["load_ids"], [])
            self.assertEqual(inspection["tables"], ["documents"])
            self.assertEqual(inspection["document_rows"], 8)

    def test_keyed_resource_merges_updates_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = self.config(Path(directory))
            original = Document("test:1", "Title", "movie", "First", "test")
            historical = Document("test:2", "Old", "movie", "Historical", "test")
            updated = Document("test:1", "Title", "movie", "Updated", "test")

            load_document_resource(
                [original, historical], source="test", config=config, run_id="one"
            )
            result = load_document_resource(
                [updated], source="test", config=config, run_id="two"
            )
            inspection = inspect_local_pipeline(config, "test")

            self.assertEqual(len(result.documents), 1)
            self.assertEqual(result.documents[0].text, "Updated")
            self.assertEqual(inspection["document_rows"], 2)


if __name__ == "__main__":
    unittest.main()
