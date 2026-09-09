"""
Reciprocal Rank Fusion (RRF) Module.
Merges ranked lists from multiple search modalities (Dense + BM25) into a unified ranking.
Formula: score += weight / (k + rank)
"""

from typing import Any, Dict, List, Optional

from noticerag.config import RRF_K


class ReciprocalRankFusion:
    """
    Combines dense and sparse search rankings using Reciprocal Rank Fusion.
    """

    @classmethod
    def fuse_rankings(
        cls,
        dense_results: List[Dict[str, Any]],
        bm25_results: List[Dict[str, Any]],
        top_k: int = 10,
        k: int = RRF_K,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ) -> List[Dict[str, Any]]:
        """
        Merges dense and BM25 result lists via RRF.
        score = dense_weight / (k + rank_dense) + bm25_weight / (k + rank_bm25)
        """
        combined_scores: Dict[str, float] = {}
        chunk_data: Dict[str, Dict[str, Any]] = {}
        sources: Dict[str, List[str]] = {}

        # 1. Process Dense Rankings
        for rank, hit in enumerate(dense_results, 1):
            cid = hit["chunk_id"]
            rrf_score = dense_weight / (k + rank)
            combined_scores[cid] = combined_scores.get(cid, 0.0) + rrf_score
            sources.setdefault(cid, []).append("dense")
            if cid not in chunk_data:
                chunk_data[cid] = dict(hit)

        # 2. Process BM25 Rankings
        for rank, hit in enumerate(bm25_results, 1):
            cid = hit["chunk_id"]
            rrf_score = bm25_weight / (k + rank)
            combined_scores[cid] = combined_scores.get(cid, 0.0) + rrf_score
            sources.setdefault(cid, []).append("bm25")
            if cid not in chunk_data:
                chunk_data[cid] = dict(hit)

        # 3. Sort by combined RRF score descending
        sorted_chunks = sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)

        final_results = []
        for rank, (cid, score) in enumerate(sorted_chunks[:top_k], 1):
            item = dict(chunk_data[cid])
            item["score"] = round(score, 6)
            item["rank"] = rank

            chunk_srcs = sources.get(cid, [])
            if len(chunk_srcs) > 1:
                item["retrieval_source"] = "hybrid"
            elif "dense" in chunk_srcs:
                item["retrieval_source"] = "dense"
            else:
                item["retrieval_source"] = "bm25"

            final_results.append(item)

        return final_results
