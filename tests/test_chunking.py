import unittest

from moodlens.chunking import chunk_document
from moodlens.models import Document


class ChunkingTests(unittest.TestCase):
    def test_chunking_preserves_overlap_and_metadata(self) -> None:
        document = Document(
            document_id="test:one",
            title="Test",
            media_kind="movie",
            text="one two three four five six seven",
            source="test",
        )
        chunks = chunk_document(document, words=4, overlap=2)
        self.assertEqual(
            [chunk.text for chunk in chunks],
            ["one two three four", "three four five six", "five six seven"],
        )
        self.assertEqual(chunks[0].chunk_id, "test:one::c000")
        self.assertEqual(chunks[0].metadata["title"], "Test")

    def test_invalid_chunk_settings_are_rejected(self) -> None:
        document = Document("id", "Title", "movie", "some text", "test")
        with self.assertRaisesRegex(ValueError, "overlap"):
            chunk_document(document, words=4, overlap=4)
