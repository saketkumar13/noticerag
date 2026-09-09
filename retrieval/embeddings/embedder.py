"""
Dense Embedding Module using BAAI/bge-small-en-v1.5.
Provides batch embedding generation, device selection, and retry resilience.
"""

import logging
import time
from typing import List, Optional

import numpy as np
from fastembed import TextEmbedding

from noticerag.config import EMBEDDING_DIM, EMBEDDING_MODEL_NAME

logger = logging.getLogger("noticerag.retrieval.embeddings.embedder")


class DenseEmbedder:
    """
    Generates normalized dense embeddings using BAAI/bge-small-en-v1.5.
    """

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL_NAME,
        batch_size: int = 64,
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.dim = EMBEDDING_DIM
        self._model: Optional[TextEmbedding] = None
        self._init_model()

    def _init_model(self) -> None:
        """Initializes the fastembed TextEmbedding model."""
        try:
            logger.info("Initializing embedding model %s...", self.model_name)
            self._model = TextEmbedding(model_name=self.model_name)
            logger.info("Embedding model %s initialized successfully.", self.model_name)
        except Exception as exc:
            logger.error("Failed to initialize embedding model: %s", exc)
            raise

    def embed_texts(self, texts: List[str], batch_size: Optional[int] = None) -> List[List[float]]:
        """
        Embeds a list of texts into 384-dimensional dense vectors with retries.
        """
        if not texts:
            return []

        bs = batch_size or self.batch_size

        for attempt in range(1, self.max_retries + 1):
            try:
                # FastEmbed returns an iterator of numpy ndarrays
                embeddings_iter = self._model.embed(texts, batch_size=bs)
                vectors = [v.tolist() if isinstance(v, np.ndarray) else list(v) for v in embeddings_iter]
                return vectors
            except Exception as exc:
                logger.warning("Embedding attempt %d/%d failed: %s", attempt, self.max_retries, exc)
                if attempt < self.max_retries:
                    time.sleep(1.0 * attempt)
                else:
                    logger.error("All embedding retry attempts exhausted.")
                    raise

        return []

    def embed_query(self, query: str) -> List[float]:
        """
        Embeds a single query string.
        """
        results = self.embed_texts([query])
        return results[0] if results else [0.0] * self.dim
