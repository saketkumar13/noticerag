import logging
import time
from typing import List, Optional, Tuple
import numpy as np
from sentence_transformers import CrossEncoder

from retrieval.reranker.config import BATCH_SIZE, DEVICE, RERANK_MODEL

logger = logging.getLogger("noticerag.retrieval.reranker.scorer")


class CrossEncoderScorer:
    def __init__(
        self,
        model_name: str = RERANK_MODEL,
        device: Optional[str] = None,
        batch_size: int = BATCH_SIZE,
    ):
        self.model_name = model_name
        self.device = device or DEVICE
        self.batch_size = batch_size
        self.model: Optional[CrossEncoder] = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            logger.info("Loading CrossEncoder model %s on device %s", self.model_name, self.device)
            self.model = CrossEncoder(self.model_name, device=self.device)
            logger.info("CrossEncoder model %s successfully loaded", self.model_name)
        except Exception as exc:
            logger.error("Failed to load CrossEncoder model %s: %s", self.model_name, exc)
            raise

    def score_pairs(
        self,
        pairs: List[Tuple[str, str]],
        batch_size: Optional[int] = None,
    ) -> Tuple[List[float], float, float]:
        if not pairs:
            return [], 0.0, 0.0

        bs = batch_size or self.batch_size
        t0 = time.perf_counter()

        try:
            raw_scores = self.model.predict(
                pairs,
                batch_size=bs,
                show_progress_bar=False,
            )

            if isinstance(raw_scores, np.ndarray):
                scores = raw_scores.tolist()
            elif isinstance(raw_scores, (list, tuple)):
                scores = list(raw_scores)
            else:
                scores = [float(raw_scores)]

            scores = [float(s) for s in scores]

        except Exception as exc:
            logger.error("Batch prediction failed: %s", exc)
            scores = [0.0] * len(pairs)

        elapsed = time.perf_counter() - t0
        latency_ms = elapsed * 1000.0
        effective_elapsed = max(elapsed, 1e-6)
        chunks_per_sec = len(pairs) / effective_elapsed

        return scores, latency_ms, chunks_per_sec
