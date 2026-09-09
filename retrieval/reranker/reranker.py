import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from retrieval.reranker.config import (
    BATCH_SIZE,
    RERANKER_LOG_FILE,
    TOP_K_RERANK,
)
from retrieval.reranker.scorer import CrossEncoderScorer

logger = logging.getLogger("noticerag.retrieval.reranker")


class DocumentReranker:
    def __init__(
        self,
        scorer: Optional[CrossEncoderScorer] = None,
        top_k: int = TOP_K_RERANK,
        batch_size: int = BATCH_SIZE,
    ):
        self.scorer = scorer or CrossEncoderScorer(batch_size=batch_size)
        self.top_k = top_k
        self.batch_size = batch_size
        self._ensure_logger()

    def _ensure_logger(self) -> None:
        if not logger.handlers:
            formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
            )
            fh = logging.FileHandler(str(RERANKER_LOG_FILE), encoding="utf-8")
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            logger.setLevel(logging.INFO)

    def rerank(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        k = top_k if top_k is not None else self.top_k

        if not retrieved_chunks:
            self._log_event(
                query=query,
                retrieved_count=0,
                reranked_results=[],
                latency_ms=0.0,
                chunks_per_sec=0.0,
                error=None,
            )
            return {
                "query": query,
                "chunks_scored": 0,
                "latency_ms": 0.0,
                "chunks_per_sec": 0.0,
                "results": [],
            }

        pairs = []
        for chunk in retrieved_chunks:
            text = chunk.get("chunk_text") or chunk.get("text") or ""
            pairs.append((query, text))

        try:
            scores, latency_ms, chunks_per_sec = self.scorer.score_pairs(
                pairs, batch_size=self.batch_size
            )
        except Exception as exc:
            logger.error("Reranking failed for query '%s': %s", query, exc)
            self._log_event(
                query=query,
                retrieved_count=len(retrieved_chunks),
                reranked_results=[],
                latency_ms=0.0,
                chunks_per_sec=0.0,
                error=str(exc),
            )
            return {
                "query": query,
                "chunks_scored": len(retrieved_chunks),
                "latency_ms": 0.0,
                "chunks_per_sec": 0.0,
                "results": retrieved_chunks[:k],
            }

        scored_chunks = []
        for chunk, score in zip(retrieved_chunks, scores):
            item = dict(chunk)
            item["retrieval_score"] = float(
                chunk.get("score") or chunk.get("retrieval_score") or 0.0
            )
            item["rerank_score"] = float(score)
            scored_chunks.append(item)

        scored_chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
        final_top = scored_chunks[:k]

        self._log_event(
            query=query,
            retrieved_count=len(retrieved_chunks),
            reranked_results=final_top,
            latency_ms=latency_ms,
            chunks_per_sec=chunks_per_sec,
            error=None,
        )

        return {
            "query": query,
            "chunks_scored": len(retrieved_chunks),
            "latency_ms": round(latency_ms, 2),
            "chunks_per_sec": round(chunks_per_sec, 2),
            "results": final_top,
        }

    def _log_event(
        self,
        query: str,
        retrieved_count: int,
        reranked_results: List[Dict[str, Any]],
        latency_ms: float,
        chunks_per_sec: float,
        error: Optional[str],
    ) -> None:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "retrieved_chunks_count": retrieved_count,
            "latency_ms": round(latency_ms, 2),
            "chunks_per_sec": round(chunks_per_sec, 2),
            "error": error,
            "top_results": [
                {
                    "chunk_id": r.get("chunk_id", ""),
                    "document_id": r.get("document_id", ""),
                    "title": r.get("title", ""),
                    "retrieval_score": round(r.get("retrieval_score", 0.0), 5),
                    "rerank_score": round(r.get("rerank_score", 0.0), 5),
                }
                for r in reranked_results
            ],
        }
        logger.info(json.dumps(log_entry))
