import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from retrieval.hybrid.hybrid_search import HybridSearcher
from retrieval.reranker.config import (
    DEVICE,
    RERANK_MODEL,
    RERANKER_METRICS_FILE,
    RETRIEVAL_TEST_SET_FILE,
    TOP_K_RERANK,
    TOP_K_RETRIEVAL,
)
from retrieval.reranker.reranker import DocumentReranker

logger = logging.getLogger("noticerag.retrieval.reranker.evaluator")


class RerankerEvaluator:
    def __init__(
        self,
        hybrid_searcher: Optional[HybridSearcher] = None,
        reranker: Optional[DocumentReranker] = None,
        test_set_path: Path = RETRIEVAL_TEST_SET_FILE,
        metrics_output_path: Path = RERANKER_METRICS_FILE,
        top_k_retrieval: int = TOP_K_RETRIEVAL,
        top_k_rerank: int = TOP_K_RERANK,
    ):
        self.searcher = hybrid_searcher or HybridSearcher()
        self.reranker = reranker or DocumentReranker(top_k=top_k_rerank)
        self.test_set_path = Path(test_set_path)
        self.metrics_output_path = Path(metrics_output_path)
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank

    def load_test_set(self) -> List[Dict[str, Any]]:
        if not self.test_set_path.exists():
            raise FileNotFoundError(f"Evaluation test set not found at {self.test_set_path}")

        with open(self.test_set_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _is_match(hit: Dict[str, Any], expected_doc_id: str) -> bool:
        hit_doc = hit.get("document_id", "")
        hit_chunk = hit.get("chunk_id", "")
        return hit_doc == expected_doc_id or expected_doc_id in hit_chunk

    @classmethod
    def compute_mrr(cls, hits: List[Dict[str, Any]], expected_doc_id: str) -> float:
        for rank, hit in enumerate(hits, 1):
            if cls._is_match(hit, expected_doc_id):
                return 1.0 / rank
        return 0.0

    @classmethod
    def compute_ndcg_at_k(cls, hits: List[Dict[str, Any]], expected_doc_id: str, k: int) -> float:
        dcg = 0.0
        idcg = 1.0

        for rank, hit in enumerate(hits[:k], 1):
            if cls._is_match(hit, expected_doc_id):
                dcg = 1.0 / math.log2(rank + 1)
                break

        return dcg / idcg

    @classmethod
    def compute_precision_at_k(cls, hits: List[Dict[str, Any]], expected_doc_id: str, k: int) -> float:
        relevant_count = 0
        for hit in hits[:k]:
            if cls._is_match(hit, expected_doc_id):
                relevant_count += 1
        return relevant_count / float(k) if k > 0 else 0.0

    @classmethod
    def compute_hit_rate(cls, hits: List[Dict[str, Any]], expected_doc_id: str, k: int) -> float:
        for hit in hits[:k]:
            if cls._is_match(hit, expected_doc_id):
                return 1.0
        return 0.0

    def evaluate(self, test_set: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        items = test_set or self.load_test_set()
        logger.info("Evaluating reranker on %d benchmark queries...", len(items))

        hybrid_mrr_list = []
        rerank_mrr_list = []
        hybrid_ndcg5_list = []
        rerank_ndcg5_list = []
        hybrid_ndcg10_list = []
        rerank_ndcg10_list = []
        hybrid_p5_list = []
        rerank_p5_list = []
        hybrid_hit_list = []
        rerank_hit_list = []
        rerank_latencies = []
        chunks_scored_list = []

        query_evaluations = []

        for item in items:
            q = item["query"]
            expected = item["expected_document"]

            retrieval_res = self.searcher.search_hybrid(
                query=q,
                top_k=self.top_k_retrieval,
                retrieval_limit=self.top_k_retrieval,
            )
            raw_hits = retrieval_res["results"]

            rerank_res = self.reranker.rerank(
                query=q,
                retrieved_chunks=raw_hits,
                top_k=self.top_k_rerank,
            )
            rerank_hits = rerank_res["results"]

            h_mrr = self.compute_mrr(raw_hits, expected)
            r_mrr = self.compute_mrr(rerank_hits, expected)

            h_ndcg5 = self.compute_ndcg_at_k(raw_hits, expected, k=5)
            r_ndcg5 = self.compute_ndcg_at_k(rerank_hits, expected, k=5)

            h_ndcg10 = self.compute_ndcg_at_k(raw_hits, expected, k=10)
            r_ndcg10 = self.compute_ndcg_at_k(rerank_hits, expected, k=10)

            h_p5 = self.compute_precision_at_k(raw_hits, expected, k=5)
            r_p5 = self.compute_precision_at_k(rerank_hits, expected, k=5)

            h_hit = self.compute_hit_rate(raw_hits, expected, k=self.top_k_rerank)
            r_hit = self.compute_hit_rate(rerank_hits, expected, k=self.top_k_rerank)

            hybrid_mrr_list.append(h_mrr)
            rerank_mrr_list.append(r_mrr)
            hybrid_ndcg5_list.append(h_ndcg5)
            rerank_ndcg5_list.append(r_ndcg5)
            hybrid_ndcg10_list.append(h_ndcg10)
            rerank_ndcg10_list.append(r_ndcg10)
            hybrid_p5_list.append(h_p5)
            rerank_p5_list.append(r_p5)
            hybrid_hit_list.append(h_hit)
            rerank_hit_list.append(r_hit)
            rerank_latencies.append(rerank_res["latency_ms"])
            chunks_scored_list.append(rerank_res["chunks_scored"])

            top_rerank_sample = rerank_hits[0] if rerank_hits else {}
            query_evaluations.append(
                {
                    "query": q,
                    "expected_document": expected,
                    "expected_title": item.get("expected_title", ""),
                    "hybrid_mrr": round(h_mrr, 4),
                    "reranked_mrr": round(r_mrr, 4),
                    "hybrid_ndcg@5": round(h_ndcg5, 4),
                    "reranked_ndcg@5": round(r_ndcg5, 4),
                    "hybrid_precision@5": round(h_p5, 4),
                    "reranked_precision@5": round(r_p5, 4),
                    "hybrid_hit_rate": round(h_hit, 4),
                    "reranked_hit_rate": round(r_hit, 4),
                    "rerank_latency_ms": rerank_res["latency_ms"],
                    "top_reranked_title": top_rerank_sample.get("title", ""),
                    "top_reranked_doc_id": top_rerank_sample.get("document_id", ""),
                    "top_reranked_score": round(top_rerank_sample.get("rerank_score", 0.0), 4),
                }
            )

        n = len(items) if items else 1
        summary = {
            "model_name": RERANK_MODEL,
            "device": DEVICE,
            "queries_count": len(items),
            "top_k_retrieval": self.top_k_retrieval,
            "top_k_rerank": self.top_k_rerank,
            "hybrid_mrr": round(sum(hybrid_mrr_list) / n, 4),
            "reranked_mrr": round(sum(rerank_mrr_list) / n, 4),
            "hybrid_ndcg@5": round(sum(hybrid_ndcg5_list) / n, 4),
            "reranked_ndcg@5": round(sum(rerank_ndcg5_list) / n, 4),
            "hybrid_ndcg@10": round(sum(hybrid_ndcg10_list) / n, 4),
            "reranked_ndcg@10": round(sum(rerank_ndcg10_list) / n, 4),
            "hybrid_precision@5": round(sum(hybrid_p5_list) / n, 4),
            "reranked_precision@5": round(sum(rerank_p5_list) / n, 4),
            "hybrid_hit_rate": round(sum(hybrid_hit_list) / n, 4),
            "reranked_hit_rate": round(sum(rerank_hit_list) / n, 4),
            "average_rerank_latency_ms": round(sum(rerank_latencies) / n, 2),
            "total_chunks_scored": sum(chunks_scored_list),
            "average_chunks_per_second": round(
                (sum(chunks_scored_list) / (sum(rerank_latencies) / 1000.0))
                if sum(rerank_latencies) > 0
                else 0.0,
                2,
            ),
        }

        report_payload = {
            "summary_metrics": summary,
            "query_evaluations": query_evaluations,
        }

        self.metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_output_path, "w", encoding="utf-8") as out:
            json.dump(report_payload, out, indent=2)

        logger.info("Saved reranker evaluation report to %s", self.metrics_output_path)
        return report_payload
