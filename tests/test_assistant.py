import unittest

from cultural_mood_tracker.assistant import CulturalMoodTrackerAssistant
from cultural_mood_tracker.generation import GenerationResult
from cultural_mood_tracker.models import Chunk, SearchHit


class FakeRetriever:
    def search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        return [
            SearchHit(
                Chunk("doc::c000", "doc", "grounded evidence", 0, {"title": "Demo"}),
                0.9,
            )
        ]


class FakeGenerator:
    model = "fake-model"

    def generate(self, question: str, hits: list[SearchHit]) -> GenerationResult:
        assert question == "What is the theme?"
        assert hits[0].chunk.text == "grounded evidence"
        return GenerationResult("The theme is grounded [1].", 10, 5)


class AssistantTests(unittest.TestCase):
    def test_assistant_uses_retrieval_and_generation(self) -> None:
        result = CulturalMoodTrackerAssistant(FakeRetriever(), FakeGenerator()).answer(
            "What is the theme?"
        )
        self.assertTrue(result.text.endswith("[1]."))
        self.assertEqual(result.model, "fake-model")
        self.assertEqual(result.prompt_tokens, 10)
        self.assertEqual(result.hits[0].chunk.chunk_id, "doc::c000")
