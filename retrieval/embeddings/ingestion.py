"""
Dense Vector Ingestion Pipeline.
Loads document chunks, computes BAAI/bge-small-en-v1.5 embeddings in batches,
persists vectors to Qdrant, and generates data/reports/qdrant_report.json.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from noticerag.config import (
    CHUNKS_PARQUET,
    EMBEDDING_DIM,
    QDRANT_COLLECTION,
    QDRANT_DIR,
    REPORTS_DIR,
)
from retrieval.embeddings.embedder import DenseEmbedder
from retrieval.embeddings.qdrant_manager import QdrantManager

logger = logging.getLogger("noticerag.retrieval.embeddings.ingestion")


class VectorIngestionPipeline:
    """
    Orchestrates dense vector generation and Qdrant ingestion.
    """

    def __init__(
        self,
        chunks_file: Path = CHUNKS_PARQUET,
        qdrant_dir: Path = QDRANT_DIR,
        reports_dir: Path = REPORTS_DIR,
        embedder: Optional[DenseEmbedder] = None,
        qdrant_manager: Optional[QdrantManager] = None,
    ):
        self.chunks_file = Path(chunks_file)
        self.qdrant_dir = Path(qdrant_dir)
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder or DenseEmbedder()
        self.qdrant = qdrant_manager or QdrantManager(qdrant_dir=self.qdrant_dir)

    def run_ingestion(self, chunks: Optional[List[Dict[str, Any]]] = None, recreate: bool = True) -> Dict[str, Any]:
        """
        Runs complete embedding generation and Qdrant upsertion.
        """
        start_time = time.time()
        logger.info("=== Starting Dense Vector Ingestion ===")

        if chunks is None:
            if not self.chunks_file.exists():
                raise FileNotFoundError(f"Chunks file not found: {self.chunks_file}")
            df = pd.read_parquet(self.chunks_file)
            chunks = df.to_dict(orient="records")

        total_chunks = len(chunks)
        logger.info("Loaded %d chunks for embedding generation.", total_chunks)

        # 1. Reset/Ensure Qdrant collection
        if recreate:
            self.qdrant.ensure_collection(recreate=True)

        # 2. Extract texts to embed
        # BGE models benefit from notice title prefix for contextual clarity
        texts_to_embed = [
            f"Title: {c.get('title', '')}\nText: {c.get('chunk_text', '')}"
            for c in chunks
        ]

        # 3. Generate embeddings
        t_embed_start = time.time()
        vectors = self.embedder.embed_texts(texts_to_embed, batch_size=64)
        embedding_time = time.time() - t_embed_start
        logger.info("Computed %d embeddings in %.2f seconds.", len(vectors), embedding_time)

        # 4. Upsert to Qdrant
        t_upsert_start = time.time()
        upserted_count = self.qdrant.upsert_chunks(chunks, vectors, batch_size=100)
        indexing_time = time.time() - t_upsert_start
        logger.info("Upserted %d vectors to Qdrant in %.2f seconds.", upserted_count, indexing_time)

        # 5. Validate Qdrant collection
        coll_info = self.qdrant.get_collection_info()
        is_valid = (
            coll_info["points_count"] == total_chunks
            and coll_info["vector_size"] == EMBEDDING_DIM
        )

        total_elapsed = time.time() - start_time

        report_data = {
            "collection_name": QDRANT_COLLECTION,
            "status": coll_info["status"],
            "total_chunks_provided": total_chunks,
            "vectors_stored": coll_info["points_count"],
            "embedding_dimension": coll_info["vector_size"],
            "embedding_time_seconds": round(embedding_time, 2),
            "indexing_time_seconds": round(indexing_time, 2),
            "total_time_seconds": round(total_elapsed, 2),
            "is_valid": is_valid,
        }

        # Save data/reports/qdrant_report.json
        qdrant_report_file = self.reports_dir / "qdrant_report.json"
        with open(qdrant_report_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        logger.info("Saved Qdrant report to %s", qdrant_report_file)

        return report_data
