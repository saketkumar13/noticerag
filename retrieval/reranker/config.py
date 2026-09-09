import os
from pathlib import Path
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
LOG_DIR = DATA_DIR / "logs"
REPORTS_DIR = DATA_DIR / "reports"
EVALUATION_DIR = DATA_DIR / "evaluation"

for directory in (LOG_DIR, REPORTS_DIR, EVALUATION_DIR):
    directory.mkdir(parents=True, exist_ok=True)

RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
SUPPORTED_MODELS = [
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "BAAI/bge-reranker-base",
]

TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "25"))
TOP_K_RERANK = int(os.getenv("TOP_K_RERANK", "5"))
BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "32"))

if torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

DEVICE = os.getenv("RERANK_DEVICE", DEVICE)

RERANKER_LOG_FILE = LOG_DIR / "reranker.log"
RERANKER_METRICS_FILE = REPORTS_DIR / "reranker_metrics.json"
RETRIEVAL_TEST_SET_FILE = EVALUATION_DIR / "retrieval_test_set.json"
