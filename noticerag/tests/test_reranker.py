import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from retrieval.reranker.config import BATCH_SIZE, DEVICE, RERANK_MODEL, TOP_K_RERANK
from retrieval.reranker.evaluator import RerankerEvaluator
from retrieval.reranker.reranker import DocumentReranker
from retrieval.reranker.scorer import CrossEncoderScorer


class TestCrossEncoderReranker(unittest.TestCase):
    def setUp(self):
        self.mock_model = MagicMock()
        self.mock_model.predict.return_value = [0.95, -1.2, 0.45]

    def test_model_loading_and_config(self):
        with patch("retrieval.reranker.scorer.CrossEncoder") as mock_ce:
            mock_ce.return_value = self.mock_model
            scorer = CrossEncoderScorer(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")
            self.assertEqual(scorer.model_name, "cross-encoder/ms-marco-MiniLM-L-6-v2")
            self.assertEqual(scorer.device, "cpu")
            mock_ce.assert_called_once_with("cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu")

    def test_batch_scoring(self):
        with patch("retrieval.reranker.scorer.CrossEncoder") as mock_ce:
            mock_ce.return_value = self.mock_model
            scorer = CrossEncoderScorer(device="cpu", batch_size=2)
            pairs = [
                ("holiday notice", "Institute closed for holiday"),
                ("hostel fee", "Mess fees notification"),
                ("placement interview", "Amazon campus recruitment drive"),
            ]
            scores, latency_ms, chunks_per_sec = scorer.score_pairs(pairs, batch_size=2)
            self.assertEqual(len(scores), 3)
            self.assertEqual(scores[0], 0.95)
            self.assertGreaterEqual(latency_ms, 0.0)
            self.assertGreater(chunks_per_sec, 0.0)
            self.mock_model.predict.assert_called_once_with(pairs, batch_size=2, show_progress_bar=False)

    def test_sorting_correctness(self):
        with patch("retrieval.reranker.scorer.CrossEncoder") as mock_ce:
            mock_ce.return_value = self.mock_model
            scorer = CrossEncoderScorer(device="cpu")
            reranker = DocumentReranker(scorer=scorer, top_k=3)

            retrieved = [
                {"chunk_id": "c1", "chunk_text": "text 1", "score": 0.02},
                {"chunk_id": "c2", "chunk_text": "text 2", "score": 0.03},
                {"chunk_id": "c3", "chunk_text": "text 3", "score": 0.01},
            ]
            res = reranker.rerank("query", retrieved, top_k=3)
            results = res["results"]
            self.assertEqual(len(results), 3)
            self.assertEqual(results[0]["chunk_id"], "c1")
            self.assertEqual(results[0]["rerank_score"], 0.95)
            self.assertEqual(results[1]["chunk_id"], "c3")
            self.assertEqual(results[1]["rerank_score"], 0.45)
            self.assertEqual(results[2]["chunk_id"], "c2")
            self.assertEqual(results[2]["rerank_score"], -1.2)

    def test_top_k_selection(self):
        with patch("retrieval.reranker.scorer.CrossEncoder") as mock_ce:
            custom_mock = MagicMock()
            custom_mock.predict.return_value = [0.1, 0.9, 0.5, 0.8, 0.2, 0.7]
            mock_ce.return_value = custom_mock

            scorer = CrossEncoderScorer(device="cpu")
            reranker = DocumentReranker(scorer=scorer, top_k=3)

            retrieved = [
                {"chunk_id": f"c{i}", "chunk_text": f"text {i}", "score": 0.01 * i}
                for i in range(1, 7)
            ]
            res = reranker.rerank("query", retrieved, top_k=3)
            self.assertEqual(len(res["results"]), 3)
            self.assertEqual(res["results"][0]["chunk_id"], "c2")
            self.assertEqual(res["results"][1]["chunk_id"], "c4")
            self.assertEqual(res["results"][2]["chunk_id"], "c6")

    def test_failure_and_empty_handling(self):
        with patch("retrieval.reranker.scorer.CrossEncoder") as mock_ce:
            mock_ce.return_value = self.mock_model
            scorer = CrossEncoderScorer(device="cpu")
            reranker = DocumentReranker(scorer=scorer, top_k=5)

            res_empty = reranker.rerank("query", [])
            self.assertEqual(res_empty["chunks_scored"], 0)
            self.assertEqual(res_empty["results"], [])

            mock_fail = MagicMock()
            mock_fail.predict.side_effect = RuntimeError("GPU out of memory")
            scorer_fail = CrossEncoderScorer(device="cpu")
            scorer_fail.model = mock_fail
            reranker_fail = DocumentReranker(scorer=scorer_fail, top_k=2)

            retrieved = [
                {"chunk_id": "c1", "chunk_text": "t1", "score": 0.5},
                {"chunk_id": "c2", "chunk_text": "t2", "score": 0.4},
            ]
            res_fail = reranker_fail.rerank("query", retrieved)
            self.assertEqual(len(res_fail["results"]), 2)
            self.assertEqual(res_fail["results"][0]["chunk_id"], "c1")

    def test_evaluation_metric_computations(self):
        hits = [
            {"document_id": "doc_other_1", "chunk_id": "doc_other_1_c001"},
            {"document_id": "doc_target", "chunk_id": "doc_target_c001"},
            {"document_id": "doc_other_2", "chunk_id": "doc_other_2_c001"},
        ]

        mrr = RerankerEvaluator.compute_mrr(hits, "doc_target")
        self.assertAlmostEqual(mrr, 0.5)

        ndcg5 = RerankerEvaluator.compute_ndcg_at_k(hits, "doc_target", k=5)
        self.assertGreater(ndcg5, 0.0)

        p5 = RerankerEvaluator.compute_precision_at_k(hits, "doc_target", k=5)
        self.assertAlmostEqual(p5, 0.2)

        hit_rate = RerankerEvaluator.compute_hit_rate(hits, "doc_target", k=5)
        self.assertEqual(hit_rate, 1.0)

        hit_rate_k1 = RerankerEvaluator.compute_hit_rate(hits, "doc_target", k=1)
        self.assertEqual(hit_rate_k1, 0.0)


if __name__ == "__main__":
    unittest.main()
