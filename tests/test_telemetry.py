import tempfile
import unittest
from pathlib import Path

from cultural_mood_tracker.database import Database
from cultural_mood_tracker.models import Answer, Chunk, SearchHit
from cultural_mood_tracker.telemetry import Telemetry


class TelemetryTests(unittest.TestCase):
    def test_interaction_and_feedback_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            telemetry = Telemetry(Database(Path(directory) / "telemetry.sqlite3"))
            answer = Answer(
                text="answer",
                hits=(SearchHit(Chunk("c1", "d1", "text", 0), 0.75),),
                model="test",
                latency_ms=12,
                prompt_tokens=20,
                completion_tokens=5,
            )
            identifier = telemetry.record(session_id="session", question="question", answer=answer)
            telemetry.feedback(identifier, 1)
            summary = telemetry.summary()
            self.assertEqual(summary["requests"], 1)
            self.assertEqual(summary["tokens"], 25)
            self.assertEqual(summary["positive_rate"], 1.0)
