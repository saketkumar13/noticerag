"""
Retrieval Evaluation Framework.
Generates ground-truth evaluation datasets from notice metadata,
computes Recall@5, Recall@10, and Mean Reciprocal Rank (MRR) for Dense, BM25, and Hybrid,
measures latency metrics, and outputs data/reports/retrieval_metrics.json.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from noticerag.config import EVALUATION_DIR, METADATA_DIR, REPORTS_DIR
from retrieval.hybrid.hybrid_search import HybridSearcher

logger = logging.getLogger("noticerag.retrieval.hybrid.evaluator")


class RetrievalEvaluator:
    """
    Evaluates Dense, BM25, and Hybrid retrieval across Recall@K, MRR, and search latencies.
    """

    DEFAULT_TEST_QUERIES = [
        ("holiday", "Holiday Notice", ["holiday", "janmashtami", "closed"]),
        ("janmashtami", "Holiday Notice", ["janmashtami"]),
        ("hostel", "Hostel Notice", ["hostel", "warden", "room"]),
        ("placement", "Placement", ["placement", "recruiter", "internship", "t&p"]),
        ("scholarship", "Scholarship", ["scholarship", "fellowship", "stipend"]),
        ("recruitment", "Recruitment", ["recruitment", "jrf", "advertisement", "post"]),
        ("tender", "Tender", ["tender", "quotation", "price bid", "financial bid"]),
        ("academic calendar", "Academic Notice", ["academic", "calendar", "examination", "admission"]),
    ]

    def __init__(
        self,
        searcher: Optional[HybridSearcher] = None,
        metadata_dir: Path = METADATA_DIR,
        eval_dir: Path = EVALUATION_DIR,
        reports_dir: Path = REPORTS_DIR,
    ):
        self.searcher = searcher or HybridSearcher()
        self.metadata_dir = Path(metadata_dir)
        self.eval_dir = Path(eval_dir)
        self.reports_dir = Path(reports_dir)
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_evaluation_test_set(self) -> List[Dict[str, Any]]:
        """
        Synthesizes ground truth evaluation dataset from metadata catalog.
        Finds the most relevant ground-truth document for each canonical validation query.
        """
        # Load all metadata documents
        metadata_records = []
        for mpath in sorted(self.metadata_dir.glob("*.json")):
            try:
                with open(mpath, "r", encoding="utf-8") as mf:
                    data = json.load(mf)
                    if data.get("document_id"):
                        metadata_records.append(data)
            except Exception:
                pass

        test_set = []

        for query, target_type, keywords in self.DEFAULT_TEST_QUERIES:
            matched_doc_id = None
            matched_title = ""

            # 1. First search for type match + keyword in title or doc_id
            for rec in metadata_records:
                doc_title = rec.get("title", "").lower()
                doc_id = rec.get("document_id", "").lower()
                doc_type = rec.get("document_type", "")

                if doc_type == target_type and any(kw in doc_title or kw in doc_id for kw in keywords):
                    matched_doc_id = rec["document_id"]
                    matched_title = rec.get("title", "")
                    break

            # 2. Fallback: match by keywords in title anywhere
            if not matched_doc_id:
                for rec in metadata_records:
                    doc_title = rec.get("title", "").lower()
                    doc_id = rec.get("document_id", "").lower()
                    if any(kw in doc_title or kw in doc_id for kw in keywords):
                        matched_doc_id = rec["document_id"]
                        matched_title = rec.get("title", "")
                        break

            # 3. Last fallback: match by target_type
            if not matched_doc_id:
                for rec in metadata_records:
                    if rec.get("document_type") == target_type:
                        matched_doc_id = rec["document_id"]
                        matched_title = rec.get("title", "")
                        break

            if matched_doc_id:
                test_set.append(
                    {
                        "query": query,
                        "expected_document": matched_doc_id,
                        "expected_title": matched_title,
                        "target_type": target_type,
                    }
                )

        # Save data/evaluation/retrieval_test_set.json
        test_set_file = self.eval_dir / "retrieval_test_set.json"
        with open(test_set_file, "w", encoding="utf-8") as f:
            json.dump(test_set, f, indent=2)
        logger.info("Saved evaluation test set with %d queries to %s", len(test_set), test_set_file)

        return test_set

    @staticmethod
    def _evaluate_ranking(hits: List[Dict[str, Any]], expected_doc_id: str) -> Dict[str, float]:
        """
        Calculates Recall@5, Recall@10, and Reciprocal Rank (RR) for a single query.
        """
        # Document matches if hit's document_id equals expected_doc_id
        # or if expected_doc_id is a substring of hit's chunk_id
        ranks = []
        for idx, hit in enumerate(hits, 1):
            hit_doc = hit.get("document_id", "")
            hit_chunk = hit.get("chunk_id", "")
            if hit_doc == expected_doc_id or expected_doc_id in hit_chunk:
                ranks.append(idx)
                break

        if ranks:
            first_rank = ranks[0]
            r5 = 1.0 if first_rank <= 5 else 0.0
            r10 = 1.0 if first_rank <= 10 else 0.0
            rr = 1.0 / first_rank
        else:
            r5 = 0.0
            r10 = 0.0
            rr = 0.0

        return {"recall@5": r5, "recall@10": r10, "mrr": rr}

    def evaluate_all(self, test_set: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Runs benchmark queries across Dense, BM25, and Hybrid.
        Computes comparative Recall@5, Recall@10, MRR, and search latencies.
        """
        if test_set is None:
            test_set = self.generate_evaluation_test_set()

        logger.info("Evaluating retrieval performance across %d queries...", len(test_set))

        results_by_mode = {
            "dense": {"recall@5": [], "recall@10": [], "mrr": [], "latencies": []},
            "bm25": {"recall@5": [], "recall@10": [], "mrr": [], "latencies": []},
            "hybrid": {"recall@5": [], "recall@10": [], "mrr": [], "latencies": []},
        }

        query_reports = []

        for item in test_set:
            q = item["query"]
            expected = item["expected_document"]

            # 1. Dense Retrieval
            dense_hits, dense_lat = self.searcher.search_dense(q, top_k=20)
            d_eval = self._evaluate_ranking(dense_hits, expected)
            results_by_mode["dense"]["recall@5"].append(d_eval["recall@5"])
            results_by_mode["dense"]["recall@10"].append(d_eval["recall@10"])
            results_by_mode["dense"]["mrr"].append(d_eval["mrr"])
            results_by_mode["dense"]["latencies"].append(dense_lat)

            # 2. BM25 Retrieval
            bm25_hits, bm25_lat = self.searcher.search_bm25(q, top_k=20)
            b_eval = self._evaluate_ranking(bm25_hits, expected)
            results_by_mode["bm25"]["recall@5"].append(b_eval["recall@5"])
            results_by_mode["bm25"]["recall@10"].append(b_eval["recall@10"])
            results_by_mode["bm25"]["mrr"].append(b_eval["mrr"])
            results_by_mode["bm25"]["latencies"].append(bm25_lat)

            # 3. Hybrid Retrieval
            hybrid_res = self.searcher.search_hybrid(q, top_k=10, retrieval_limit=20)
            h_eval = self._evaluate_ranking(hybrid_res["results"], expected)
            results_by_mode["hybrid"]["recall@5"].append(h_eval["recall@5"])
            results_by_mode["hybrid"]["recall@10"].append(h_eval["recall@10"])
            results_by_mode["hybrid"]["mrr"].append(h_eval["mrr"])
            results_by_mode["hybrid"]["latencies"].append(hybrid_res["latencies_ms"]["hybrid_total"])

            top_sample = hybrid_res["results"][0] if hybrid_res["results"] else {}
            query_reports.append(
                {
                    "query": q,
                    "expected_document": expected,
                    "expected_title": item["expected_title"],
                    "dense_recall@5": d_eval["recall@5"],
                    "bm25_recall@5": b_eval["recall@5"],
                    "hybrid_recall@5": h_eval["recall@5"],
                    "hybrid_mrr": round(h_eval["mrr"], 3),
                    "top_result_title": top_sample.get("title", ""),
                    "top_result_type": top_sample.get("document_type", ""),
                    "top_result_score": top_sample.get("score", 0.0),
                }
            )

        # Calculate averages
        summary_metrics = {}
        for mode in ["dense", "bm25", "hybrid"]:
            n = len(test_set)
            r5_avg = sum(results_by_mode[mode]["recall@5"]) / n if n else 0.0
            r10_avg = sum(results_by_mode[mode]["recall@10"]) / n if n else 0.0
            mrr_avg = sum(results_by_mode[mode]["mrr"]) / n if n else 0.0
            lat_avg = sum(results_by_mode[mode]["latencies"]) / n if n else 0.0

            summary_metrics[mode] = {
                "recall@5": round(r5_avg, 3),
                "recall@10": round(r10_avg, 3),
                "mrr": round(mrr_avg, 3),
                "avg_latency_ms": round(lat_avg, 2),
            }

        # Load Qdrant and BM25 reports for overall performance metrics
        qdrant_rep_path = self.reports_dir / "qdrant_report.json"
        bm25_rep_path = self.reports_dir / "bm25_stats.json"

        qdrant_data = json.load(open(qdrant_rep_path, "r", encoding="utf-8")) if qdrant_rep_path.exists() else {}
        bm25_data = json.load(open(bm25_rep_path, "r", encoding="utf-8")) if bm25_rep_path.exists() else {}

        report_data = {
            "summary_metrics": summary_metrics,
            "performance_latencies": {
                "embedding_time_seconds": qdrant_data.get("embedding_time_seconds", 0.0),
                "indexing_time_seconds": qdrant_data.get("indexing_time_seconds", 0.0),
                "bm25_index_build_seconds": bm25_data.get("index_build_time_seconds", 0.0),
                "average_dense_latency_ms": summary_metrics["dense"]["avg_latency_ms"],
                "average_bm25_latency_ms": summary_metrics["bm25"]["avg_latency_ms"],
                "average_hybrid_latency_ms": summary_metrics["hybrid"]["avg_latency_ms"],
            },
            "queries_evaluated_count": len(test_set),
            "query_evaluations": query_reports,
        }

        # Save data/reports/retrieval_metrics.json
        metrics_file = self.reports_dir / "retrieval_metrics.json"
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        logger.info("Saved retrieval metrics report to %s", metrics_file)

        return report_data
