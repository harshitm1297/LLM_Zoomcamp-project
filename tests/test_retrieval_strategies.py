import unittest

from moodlens.models import Chunk, SearchHit
from moodlens.retrieval import BM25Index, reciprocal_rank_fusion, rewrite_query


class RetrievalStrategyTests(unittest.TestCase):
    def test_bm25_ranks_matching_title_and_text(self) -> None:
        chunks = [
            Chunk("weather", "weather", "residents face endless storms", 0, {"title": "Weather"}),
            Chunk("music", "music", "an orchestra performs", 0, {"title": "Orchestra"}),
        ]
        hits = BM25Index(chunks).search("endless storms", top_k=2)
        self.assertEqual(hits[0].chunk.chunk_id, "weather")

    def test_rank_fusion_rewards_cross_list_evidence(self) -> None:
        a = SearchHit(Chunk("a", "a", "a", 0), 0.9)
        b = SearchHit(Chunk("b", "b", "b", 0), 0.8)
        fused = reciprocal_rank_fusion([[a, b], [b, a]], top_k=2)
        self.assertEqual({hit.chunk.chunk_id for hit in fused}, {"a", "b"})
        self.assertEqual(fused[0].score, fused[1].score)

    def test_query_rewriting_expands_domain_terms(self) -> None:
        rewritten = rewrite_query("A story about environmental problems")
        self.assertIn("weather", rewritten)
        self.assertIn("justice", rewritten)

