"""
OCR Pipeline Orchestrator for Phase 2.
Coordinates reading downloaded PDFs from SQLite, hybrid text extraction and OCR,
persisting OCR metadata to SQLite, saving structured RAG-ready JSON outputs,
and maintaining incremental checkpoints.
"""

import json
import logging
import logging.handlers
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure package root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
for p in (str(BASE_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from noticerag.config import (
    DB_PATH,
    EXTRACTED_TEXT_DIR,
    LOG_FILE,
    LOG_FORMAT,
    LOG_LEVEL,
    PDF_DIR,
)
from noticerag.crawler.database import DatabaseManager
from noticerag.ocr.pdf_processor import PDFProcessor
from noticerag.ocr.validator import OCRValidator

logger = logging.getLogger("noticerag.ocr.pipeline")


def get_utc_now_iso() -> str:
    """Returns current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def setup_ocr_logging() -> None:
    """Configures structured file and console logging for OCR pipeline."""
    root_logger = logging.getLogger("noticerag")
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console = logging.StreamHandler()
        console.setFormatter(logging.Formatter(LOG_FORMAT))
        root_logger.addHandler(console)

    if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
        try:
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            fhandler = logging.handlers.RotatingFileHandler(
                str(LOG_FILE), maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
            )
            fhandler.setFormatter(logging.Formatter(LOG_FORMAT))
            root_logger.addHandler(fhandler)
        except Exception as exc:
            print(f"Warning: could not initialize file logger: {exc}")


class OCRPipeline:
    """
    Production-grade OCR pipeline processing downloaded PDFs into clean, structured JSON.
    """

    def __init__(
        self,
        db_path: Path = DB_PATH,
        extracted_dir: Path = EXTRACTED_TEXT_DIR,
        pdf_dir: Path = PDF_DIR,
    ):
        setup_ocr_logging()
        self.db = DatabaseManager(db_path)
        self.extracted_dir = Path(extracted_dir)
        self.pdf_dir = Path(pdf_dir)
        self.extracted_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_processor = PDFProcessor()
        self.validator = OCRValidator(self.db, self.extracted_dir, self.pdf_dir)
        self._init_ocr_schema()

    def _init_ocr_schema(self) -> None:
        """Initializes SQLite schema for OCR metadata tracking."""
        with self.db._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ocr_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    pdf_id INTEGER UNIQUE,
                    document_id TEXT UNIQUE NOT NULL,
                    title TEXT,
                    pdf_url TEXT,
                    file_path TEXT,
                    sha256 TEXT,
                    pages INTEGER NOT NULL DEFAULT 0,
                    text_length INTEGER NOT NULL DEFAULT 0,
                    word_count INTEGER NOT NULL DEFAULT 0,
                    ocr_used BOOLEAN NOT NULL DEFAULT 0,
                    ocr_engine TEXT,
                    status TEXT NOT NULL DEFAULT 'pending',
                    error_message TEXT,
                    processed_at TEXT NOT NULL,
                    FOREIGN KEY (pdf_id) REFERENCES PDFs(id)
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ocr_pdf_id ON ocr_metadata(pdf_id);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ocr_status ON ocr_metadata(status);")
            conn.commit()

    def get_already_processed_ids(self) -> set:
        """Returns set of pdf_ids that have already been successfully processed."""
        with self.db._get_connection() as conn:
            rows = conn.execute(
                "SELECT pdf_id, document_id FROM ocr_metadata WHERE status = 'success'"
            ).fetchall()
            return {r["pdf_id"] for r in rows}

    def process_document(self, pdf_row: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a single PDF row from SQLite PDFs table.
        Extracts text, runs OCR if needed, updates SQLite ocr_metadata,
        and saves structured JSON.
        """
        pdf_id = pdf_row["id"]
        title = pdf_row.get("title", "")
        pdf_url = pdf_row.get("pdf_url", "")
        file_path_str = pdf_row.get("file_path", "")
        sha256 = pdf_row.get("file_hash", "")

        file_path = Path(file_path_str) if file_path_str else (self.pdf_dir / f"doc_{pdf_id}.pdf")
        document_id = file_path.stem

        logger.info("[PDF started] ID=%s: %s (%s)", pdf_id, title[:50], file_path.name)
        start_time = time.time()

        result = self.pdf_processor.process_pdf(file_path)

        if not result["success"]:
            logger.error("[Error] Processing failed for ID=%s: %s", pdf_id, result.get("error"))
            self._record_db_result(
                pdf_id=pdf_id,
                document_id=document_id,
                title=title,
                pdf_url=pdf_url,
                file_path=str(file_path),
                sha256=sha256,
                pages=result.get("total_pages", 0),
                text_length=0,
                word_count=0,
                ocr_used=False,
                ocr_engine="None",
                status="failed",
                error_message=result.get("error"),
                processed_at=get_utc_now_iso(),
            )
            return {"success": False, "document_id": document_id, "error": result.get("error")}

        if result["ocr_used"]:
            logger.info("[OCR triggered & completed] ID=%s used %s", pdf_id, result["ocr_engine"])

        processed_at = get_utc_now_iso()

        # Build output structure matching requirements
        output_data = {
            "document_id": document_id,
            "title": title,
            "text": result["text"],
            "metadata": {
                "pdf_id": pdf_id,
                "title": title,
                "pdf_url": pdf_url,
                "file_path": str(file_path.resolve()),
                "sha256": sha256,
                "pages": result["total_pages"],
                "text_length": result["text_length"],
                "word_count": result["word_count"],
                "ocr_used": result["ocr_used"],
                "ocr_engine": result["ocr_engine"],
                "pages_digital": result["pages_digital_count"],
                "pages_ocr": result["pages_ocr_count"],
                "processed_at": processed_at,
            },
        }

        # Save to data/extracted_text/<document_id>.json
        json_file = self.extracted_dir / f"{document_id}.json"
        with open(json_file, "w", encoding="utf-8") as jf:
            json.dump(output_data, jf, indent=2, ensure_ascii=False)

        # Update SQLite
        self._record_db_result(
            pdf_id=pdf_id,
            document_id=document_id,
            title=title,
            pdf_url=pdf_url,
            file_path=str(file_path),
            sha256=sha256,
            pages=result["total_pages"],
            text_length=result["text_length"],
            word_count=result["word_count"],
            ocr_used=result["ocr_used"],
            ocr_engine=result["ocr_engine"],
            status="success",
            error_message=None,
            processed_at=processed_at,
        )

        elapsed = time.time() - start_time
        logger.info(
            "[PDF completed] ID=%s in %.2fs: %d chars, %d words (OCR: %s)",
            pdf_id,
            elapsed,
            result["text_length"],
            result["word_count"],
            result["ocr_used"],
        )
        return {"success": True, "document_id": document_id, "text_length": result["text_length"]}

    def _record_db_result(
        self,
        pdf_id: int,
        document_id: str,
        title: str,
        pdf_url: str,
        file_path: str,
        sha256: str,
        pages: int,
        text_length: int,
        word_count: int,
        ocr_used: bool,
        ocr_engine: str,
        status: str,
        error_message: Optional[str],
        processed_at: str,
    ) -> None:
        """Inserts or updates SQLite ocr_metadata record."""
        with self.db._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO ocr_metadata (
                    pdf_id, document_id, title, pdf_url, file_path, sha256,
                    pages, text_length, word_count, ocr_used, ocr_engine,
                    status, error_message, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pdf_id) DO UPDATE SET
                    document_id = excluded.document_id,
                    title = excluded.title,
                    pdf_url = excluded.pdf_url,
                    file_path = excluded.file_path,
                    sha256 = excluded.sha256,
                    pages = excluded.pages,
                    text_length = excluded.text_length,
                    word_count = excluded.word_count,
                    ocr_used = excluded.ocr_used,
                    ocr_engine = excluded.ocr_engine,
                    status = excluded.status,
                    error_message = excluded.error_message,
                    processed_at = excluded.processed_at
                """,
                (
                    pdf_id,
                    document_id,
                    title,
                    pdf_url,
                    file_path,
                    sha256,
                    pages,
                    text_length,
                    word_count,
                    1 if ocr_used else 0,
                    ocr_engine,
                    status,
                    error_message,
                    processed_at,
                ),
            )
            conn.commit()

    def run(self, limit: Optional[int] = None, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Executes the OCR pipeline over downloaded PDFs.
        Supports incremental processing by skipping already processed files.
        """
        logger.info("=== OCR Pipeline Started ===")
        start_time = time.time()

        # Retrieve downloaded PDFs from database
        all_downloaded = self.db.get_downloaded_pdfs()
        logger.info("Found %d downloaded PDFs in database.", len(all_downloaded))

        already_processed = set() if force_reprocess else self.get_already_processed_ids()
        pending = [p for p in all_downloaded if p["id"] not in already_processed]

        if limit:
            pending = pending[:limit]

        logger.info("Scheduling %d PDFs for text extraction/OCR (already processed: %d).", len(pending), len(already_processed))

        processed_count = 0
        failed_count = 0

        for idx, pdf_row in enumerate(pending, 1):
            logger.info("--- [%d/%d] Processing PDF ID %s ---", idx, len(pending), pdf_row["id"])
            try:
                res = self.process_document(pdf_row)
                if res["success"]:
                    processed_count += 1
                else:
                    failed_count += 1
            except Exception as exc:
                logger.error("Unhandled error processing PDF %s: %s", pdf_row["id"], exc)
                failed_count += 1

        elapsed = time.time() - start_time
        logger.info("=== OCR Pipeline Completed in %.2f seconds (Processed: %d, Failed: %d) ===", elapsed, processed_count, failed_count)

        report = self.validator.generate_report()
        return {
            "elapsed_seconds": round(elapsed, 2),
            "processed_count": processed_count,
            "failed_count": failed_count,
            "report": report,
        }


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pipeline = OCRPipeline()
    res = pipeline.run()
    try:
        print("\n" + res["report"])
    except UnicodeEncodeError:
        safe_report = res["report"].encode("ascii", errors="replace").decode("ascii")
        print("\n" + safe_report)
