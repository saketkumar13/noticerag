"""
Qdrant Vector Database Manager.
Manages persistent collection initialization, chunk upserting, and cosine similarity search.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from noticerag.config import EMBEDDING_DIM, QDRANT_COLLECTION, QDRANT_DIR

logger = logging.getLogger("noticerag.retrieval.embeddings.qdrant")


class QdrantManager:
    """
    Manages local persistent Qdrant vector database.
    """

    def __init__(
        self,
        qdrant_dir: Path = QDRANT_DIR,
        collection_name: str = QDRANT_COLLECTION,
        dim: int = EMBEDDING_DIM,
    ):
        self.qdrant_dir = Path(qdrant_dir)
        self.qdrant_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.dim = dim
        self.client = QdrantClient(path=str(self.qdrant_dir))
        self.ensure_collection()

    def ensure_collection(self, recreate: bool = False) -> None:
        """
        Creates or validates the collection with cosine distance and 384 dimensions.
        """
        collections = [c.name for c in self.client.get_collections().collections]

        if recreate and self.collection_name in collections:
            logger.info("Recreating collection %s...", self.collection_name)
            self.client.delete_collection(self.collection_name)
            collections.remove(self.collection_name)

        if self.collection_name not in collections:
            logger.info("Creating Qdrant collection %s (dim=%d, Cosine)...", self.collection_name, self.dim)
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(
                    size=self.dim,
                    distance=qmodels.Distance.COSINE,
                ),
            )
        else:
            logger.info("Qdrant collection %s already exists.", self.collection_name)

    def upsert_chunks(
        self,
        chunks: List[Dict[str, Any]],
        vectors: List[List[float]],
        batch_size: int = 100,
    ) -> int:
        """
        Upserts chunk payloads and vectors in batches into Qdrant.
        """
        if len(chunks) != len(vectors):
            raise ValueError(f"Mismatched chunks ({len(chunks)}) and vectors ({len(vectors)})")

        total_upserted = 0
        total_points = len(chunks)

        for i in range(0, total_points, batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_vectors = vectors[i : i + batch_size]

            points = []
            for idx, (chunk, vec) in enumerate(zip(batch_chunks, batch_vectors)):
                point_id = i + idx + 1  # Integer or UUID point ID
                payload = {
                    "chunk_id": chunk.get("chunk_id", ""),
                    "document_id": chunk.get("document_id", ""),
                    "title": chunk.get("title", ""),
                    "date": chunk.get("date", ""),
                    "document_type": chunk.get("document_type", ""),
                    "department": chunk.get("department", ""),
                    "issuer": chunk.get("issuer", ""),
                    "chunk_text": chunk.get("chunk_text", ""),
                    "chunk_index": chunk.get("chunk_index", 0),
                }

                points.append(
                    qmodels.PointStruct(
                        id=point_id,
                        vector=vec,
                        payload=payload,
                    )
                )

            self.client.upsert(collection_name=self.collection_name, points=points)
            total_upserted += len(points)
            logger.debug("Upserted batch %d/%d points into Qdrant", total_upserted, total_points)

        logger.info("Total %d vectors successfully upserted into Qdrant collection %s.", total_upserted, self.collection_name)
        return total_upserted

    def search_dense(
        self,
        query_vector: List[float],
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Performs vector similarity search.
        Returns list of hits with scores and chunk payloads.
        """
        # Supports both client.query_points and client.search
        try:
            hits = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                with_payload=True,
            ).points
        except AttributeError:
            hits = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
            )

        results = []
        for rank, hit in enumerate(hits, 1):
            p = hit.payload or {}
            results.append(
                {
                    "chunk_id": p.get("chunk_id", ""),
                    "document_id": p.get("document_id", ""),
                    "score": float(hit.score),
                    "rank": rank,
                    "retrieval_source": "dense",
                    "title": p.get("title", ""),
                    "date": p.get("date", ""),
                    "document_type": p.get("document_type", ""),
                    "department": p.get("department", ""),
                    "chunk_text": p.get("chunk_text", ""),
                }
            )

        return results

    def get_collection_info(self) -> Dict[str, Any]:
        """
        Returns stats about the collection (points count, vector size, status).
        """
        info = self.client.get_collection(self.collection_name)
        return {
            "collection_name": self.collection_name,
            "status": str(info.status),
            "points_count": info.points_count,
            "indexed_vectors_count": getattr(info, "indexed_vectors_count", info.points_count),
            "vector_size": self.dim,
        }
