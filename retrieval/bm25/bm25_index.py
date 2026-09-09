"""
BM25 Sparse Index Module.
Builds, persists, and queries a BM25Okapi index over document chunks.
"""

import json
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

from noticerag.config import BM25_DIR, CHUNKS_PARQUET, REPORTS_DIR
from retrieval.bm25.tokenizer import BM25Tokenizer

logger = logging.getLogger("noticerag.retrieval.bm25.index")


class BM25Index:
    """
    Persistent BM25 sparse keyword index over notice chunks.
    """

    def __init__(self, index_dir: Path = BM25_DIR):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_dir / "bm25_model.pkl"
        self.stats_file = self.index_dir / "bm25_stats.json"

        self.bm25: Optional[BM25Okapi] = None
        self.chunks_lookup: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []

    def build_index(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tokenizes chunks, builds BM25Okapi index, and persists to disk.
        """
        start_time = time.time()
        logger.info("Building BM25 index over %d chunks...", len(chunks))

        self.chunks_lookup = chunks
        self.tokenized_corpus = []

        total_tokens = 0
        for c in chunks:
            # Combine title and chunk text for stronger keyword signals
            combined_text = f"{c.get('title', '')} {c.get('chunk_text', '')}"
            tokens = BM25Tokenizer.tokenize(combined_text)
            self.tokenized_corpus.append(tokens)
            total_tokens += len(tokens)

        # Build BM25Okapi model
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        elapsed = time.time() - start_time
        logger.info("BM25 index built in %.2f seconds (%d total tokens).", elapsed, total_tokens)

        # Persist index
        self.save()

        stats = {
            "document_count": len(chunks),
            "total_tokens": total_tokens,
            "average_tokens_per_chunk": round(total_tokens / len(chunks), 2) if chunks else 0.0,
            "index_build_time_seconds": round(elapsed, 2),
            "index_file": str(self.index_file),
        }

        with open(self.stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        return stats

    def save(self) -> None:
        """Persists the BM25 index and chunk lookup table."""
        payload = {
            "bm25": self.bm25,
            "chunks_lookup": self.chunks_lookup,
            "tokenized_corpus": self.tokenized_corpus,
        }
        with open(self.index_file, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("BM25 index saved to %s", self.index_file)

    def load(self) -> bool:
        """Loads the persisted BM25 index if available."""
        if not self.index_file.exists():
            return False

        try:
            with open(self.index_file, "rb") as f:
                data = pickle.load(f)
            self.bm25 = data["bm25"]
            self.chunks_lookup = data["chunks_lookup"]
            self.tokenized_corpus = data["tokenized_corpus"]
            logger.info("Loaded BM25 index with %d chunks.", len(self.chunks_lookup))
            return True
        except Exception as exc:
            logger.error("Could not load BM25 index from %s: %s", self.index_file, exc)
            return False

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Scores query against corpus using BM25 and returns top_k ranked chunks.
        """
        if self.bm25 is None:
            if not self.load():
                raise RuntimeError("BM25 index is not initialized or loaded.")

        tokens = BM25Tokenizer.tokenize(query)
        if not tokens:
            return []

        doc_scores = self.bm25.get_scores(tokens)

        # Fallback epsilon for matching terms when N is very small (where Robertson IDF is 0)
        for i, score in enumerate(doc_scores):
            if score <= 0.0:
                doc_toks = set(self.tokenized_corpus[i])
                matches = sum(1 for t in tokens if t in doc_toks)
                if matches > 0:
                    doc_scores[i] = 0.001 * matches

        # Get top_k indices sorted descending
        top_indices = np.argsort(doc_scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            score = float(doc_scores[idx])
            if score <= 0.0:
                continue

            chunk = self.chunks_lookup[idx]
            results.append(
                {
                    "chunk_id": chunk.get("chunk_id", ""),
                    "document_id": chunk.get("document_id", ""),
                    "score": round(score, 4),
                    "rank": rank,
                    "retrieval_source": "bm25",
                    "title": chunk.get("title", ""),
                    "date": chunk.get("date", ""),
                    "document_type": chunk.get("document_type", ""),
                    "department": chunk.get("department", ""),
                    "chunk_text": chunk.get("chunk_text", ""),
                }
            )

        return results
