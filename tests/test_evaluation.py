import unittest

from moodlens.evaluation import (
    RetrievalCase,
    evaluate_retrieval,
    evaluate_retrieval_strategies,
    reciprocal_rank,
)
from moodlens.models import Chunk, SearchHit


class FakeRetriever:
    strategy = "vector"

    def search(
        self, query: str, *, top_k: int = 5, strategy: str | None = None
    ) -> list[SearchHit]:
        return [
            SearchHit(Chunk("wrong", "wrong", "x", 0), 0.9),
            SearchHit(Chunk("right", "right", "y", 0), 0.8),
        ]


class EvaluationTests(unittest.TestCase):
    def test_reciprocal_rank_and_report(self) -> None:
        self.assertEqual(reciprocal_rank(["wrong", "right"], frozenset({"right"})), 0.5)
        report = evaluate_retrieval(
            FakeRetriever(),
            [RetrievalCase("one", "question", frozenset({"right"}))],
            top_k=2,
        )
        self.assertEqual(report["metrics"]["hit_rate"], 1.0)
        self.assertEqual(report["metrics"]["mrr"], 0.5)
        self.assertEqual(report["metrics"]["precision_at_k"], 0.5)

    def test_multiple_strategies_select_by_mrr(self) -> None:
        report = evaluate_retrieval_strategies(
            FakeRetriever(),
            [RetrievalCase("one", "question", frozenset({"right"}))],
            strategies=("bm25", "vector"),
            top_k=2,
        )
        self.assertEqual(report["best_strategy"], "bm25")
        self.assertEqual(set(report["strategies"]), {"bm25", "vector"})
