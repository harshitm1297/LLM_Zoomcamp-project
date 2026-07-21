import unittest

from cultural_mood_tracker.generation import SYSTEM_PROMPT, build_user_prompt
from cultural_mood_tracker.models import Chunk, SearchHit


class PromptTests(unittest.TestCase):
    def test_prompt_contains_numbered_evidence_and_refusal_rule(self) -> None:
        hit = SearchHit(
            Chunk("one", "doc", "Evidence text", 0, {"title": "A Title", "media_kind": "tv"}),
            0.8,
        )
        prompt = build_user_prompt("What happened?", [hit])
        self.assertIn("[1] A Title", prompt)
        self.assertIn("Evidence text", prompt)
        self.assertIn("only the supplied evidence", SYSTEM_PROMPT)
        self.assertIn("don't have enough indexed evidence", SYSTEM_PROMPT)
