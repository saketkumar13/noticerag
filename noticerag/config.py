"""
Configuration module for the NITA NoticeRAG ingestion pipeline.
Centralizes paths, URLs, limits, headers, timeouts, and logging settings.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
METADATA_DIR = DATA_DIR / "metadata"
CHECKPOINT_DIR = DATA_DIR / "checkpoints"
LOG_DIR = DATA_DIR / "logs"

EXTRACTED_TEXT_DIR = DATA_DIR / "extracted_text"
REPORTS_DIR = DATA_DIR / "reports"
PARQUET_FILE = METADATA_DIR / "master_metadata.parquet"

# Retrieval directories
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_PARQUET = CHUNKS_DIR / "chunks.parquet"
CHUNKS_JSONL = CHUNKS_DIR / "chunks.jsonl"
QDRANT_DIR = DATA_DIR / "qdrant_db"
BM25_DIR = DATA_DIR / "indexes" / "bm25"
EVALUATION_DIR = DATA_DIR / "evaluation"

# Retrieval Settings
QDRANT_COLLECTION = "nita_documents"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
RRF_K = 60

# Ensure runtime directories exist
for directory in (
    DATA_DIR, PDF_DIR, METADATA_DIR, CHECKPOINT_DIR, LOG_DIR,
    EXTRACTED_TEXT_DIR, REPORTS_DIR, CHUNKS_DIR, QDRANT_DIR,
    BM25_DIR, EVALUATION_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)

# Database
DB_PATH = DATA_DIR / "noticerag.db"

# Target URLs
NITA_BASE_URL = "https://www.nita.ac.in"
NOTICE_BOARD_URL = "https://www.nita.ac.in/UserPanel/ViewAllNewsAndEvents.aspx?nModuleID=gi"
AZURE_BLOB_BASE = "https://nitaappstorage.blob.core.windows.net/nitagartalawebdocuments/"

# Crawler Settings
INITIAL_DOWNLOAD_LIMIT = 100
REQUEST_TIMEOUT_SECONDS = 30
DOWNLOAD_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
RETRY_BACKOFF_FACTOR = 1.5
RATE_LIMIT_DELAY_SECONDS = 0.25  # polite crawl delay
MAX_DOMAIN_PAGES_TO_DISCOVER = 15  # additional domain pages to check for PDFs

# HTTP Headers
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Logging
LOG_FILE = LOG_DIR / "pipeline.log"
LOG_FORMAT = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
LOG_LEVEL = "INFO"

# OCR Configuration
OCR_MIN_CHARS_PER_PAGE = 60
OCR_DPI = 200
TESSERACT_CMD = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
OCR_PRIMARY_ENGINE = "RapidOCR"  # Options: "RapidOCR", "Tesseract"
OCR_TIMEOUT_PER_PAGE = 30

