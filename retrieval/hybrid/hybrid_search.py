"""
Hybrid Search Coordinator Module.
Queries dense embeddings from Qdrant and sparse keywords from BM25,
merges and re-ranks via Reciprocal Rank Fusion (RRF), and profiles execution latency.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from retrieval.bm25.bm25_index import BM25Index
from retrieval.embeddings.embedder import DenseEmbedder
from retrieval.embeddings.qdrant_manager import QdrantManager
from retrieval.hybrid.fusion import ReciprocalRankFusion

logger = logging.getLogger("noticerag.retrieval.hybrid.search")


class HybridSearcher:
    """
    Executes hybrid search combining Qdrant dense vector similarity
    and BM25 keyword matching with Reciprocal Rank Fusion.
    """

    def __init__(
        self,
        qdrant_manager: Optional[QdrantManager] = None,
        embedder: Optional[DenseEmbedder] = None,
        bm25_index: Optional[BM25Index] = None,
    ):
        self.qdrant = qdrant_manager or QdrantManager()
        self.embedder = embedder or DenseEmbedder()
        self.bm25 = bm25_index or BM25Index()

    def search_dense(self, query: str, top_k: int = 20) -> Tuple[List[Dict[str, Any]], float]:
        """Runs vector search via Qdrant."""
        t0 = time.time()
        vec = self.embedder.embed_query(query)
        hits = self.qdrant.search_dense(vec, top_k=top_k)
        latency = (time.time() - t0) * 1000.0  # ms
        return hits, latency

    def search_bm25(self, query: str, top_k: int = 20) -> Tuple[List[Dict[str, Any]], float]:
        """Runs sparse search via BM25."""
        t0 = time.time()
        hits = self.bm25.search(query, top_k=top_k)
        latency = (time.time() - t0) * 1000.0  # ms
        return hits, latency

    def search_hybrid(
        self,
        query: str,
        top_k: int = 10,
        retrieval_limit: int = 20,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
        k: int = 60,
    ) -> Dict[str, Any]:
        """
        Executes hybrid search:
        1. Dense Top-20
        2. BM25 Top-20
        3. Reciprocal Rank Fusion
        Returns fused results and latency metrics.
        """
        t_start = time.time()

        dense_hits, dense_ms = self.search_dense(query, top_k=retrieval_limit)
        bm25_hits, bm25_ms = self.search_bm25(query, top_k=retrieval_limit)

        fused = ReciprocalRankFusion.fuse_rankings(
            dense_results=dense_hits,
            bm25_results=bm25_hits,
            top_k=top_k,
            k=k,
            dense_weight=dense_weight,
            bm25_weight=bm25_weight,
        )

        total_ms = (time.time() - t_start) * 1000.0

        return {
            "query": query,
            "results": fused,
            "dense_count": len(dense_hits),
            "bm25_count": len(bm25_hits),
            "total_returned": len(fused),
            "latencies_ms": {
                "dense": round(dense_ms, 2),
                "bm25": round(bm25_ms, 2),
                "hybrid_total": round(total_ms, 2),
            },
        }
