import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, Optional

from generation.answer_generator import generate_answer
from generation.config import GENERATION_LOG_FILE, TOP_K
from generation.prompt_builder import build_prompt
from retrieval.hybrid.hybrid_search import HybridSearcher

logger = logging.getLogger("noticerag.generation.rag_pipeline")


def _log_generation_event(
    query: str,
    retrieved_chunk_count: int,
    generation_time_ms: float,
    status: str,
    error: Optional[str] = None,
) -> None:
    event = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "retrieved_chunk_count": retrieved_chunk_count,
        "generation_time_ms": round(generation_time_ms, 2),
        "status": status,
        "error": error,
    }
    try:
        with open(GENERATION_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as exc:
        logger.error("Failed to write generation log: %s", exc)


class RAGPipeline:
    def __init__(
        self,
        searcher: Optional[HybridSearcher] = None,
        top_k: int = TOP_K,
    ):
        self.searcher = searcher or HybridSearcher()
        self.top_k = top_k

    def answer_question(self, query: str) -> Dict[str, str]:
        t0 = time.time()
        error_msg = None

        try:
            search_res = self.searcher.search_hybrid(query=query, top_k=self.top_k)
            retrieved_chunks = search_res.get("results", [])
        except Exception as exc:
            error_msg = f"Retrieval failed: {exc}"
            logger.error(error_msg)
            retrieved_chunks = []

        if not retrieved_chunks:
            prompt = build_prompt(query, [])
        else:
            prompt = build_prompt(query, retrieved_chunks)

        try:
            answer = generate_answer(prompt)
            status = "success" if not answer.startswith("Error") else "error"
        except Exception as exc:
            status = "error"
            error_msg = str(exc)
            answer = f"Error during generation: {exc}"

        elapsed_ms = (time.time() - t0) * 1000.0
        _log_generation_event(
            query=query,
            retrieved_chunk_count=len(retrieved_chunks),
            generation_time_ms=elapsed_ms,
            status=status,
            error=error_msg,
        )

        return {
            "question": query,
            "answer": answer,
        }


_default_pipeline: Optional[RAGPipeline] = None


def answer_question(query: str) -> Dict[str, str]:
    global _default_pipeline
    if _default_pipeline is None:
        _default_pipeline = RAGPipeline()
    return _default_pipeline.answer_question(query)
