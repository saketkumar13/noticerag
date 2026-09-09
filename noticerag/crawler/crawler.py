"""
Main Crawler Orchestrator for NITA NoticeRAG.
Coordinates PDF discovery, SQLite storage, latest 100 downloading, incremental updates,
checkpoints, and validation reporting.
"""

import logging
import logging.handlers
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from noticerag.config import (
    DB_PATH,
    INITIAL_DOWNLOAD_LIMIT,
    LOG_FILE,
    LOG_FORMAT,
    LOG_LEVEL,
    METADATA_DIR,
    PDF_DIR,
    RATE_LIMIT_DELAY_SECONDS,
)
from noticerag.crawler.checkpoint import CheckpointManager
from noticerag.crawler.database import DatabaseManager
from noticerag.crawler.downloader import PDFDownloader
from noticerag.crawler.pdf_discovery import PDFDiscovery
from noticerag.crawler.validators import PipelineValidator

logger = logging.getLogger("noticerag.crawler")


def setup_pipeline_logging() -> None:
    """Configures both file and console logging."""
    root_logger = logging.getLogger("noticerag")
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

    # Avoid duplicate handlers
    if root_logger.handlers:
        return

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    root_logger.addHandler(console_handler)

    # File Handler
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            str(LOG_FILE), maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
        root_logger.addHandler(file_handler)
    except Exception as exc:
        print(f"Warning: could not configure file logging: {exc}")


class NoticeRAGPipeline:
    """
    Pipeline orchestrating discovery, persistence, selective download, and checkpoints.
    """

    def __init__(
        self,
        db_path: Path = DB_PATH,
        download_limit: int = INITIAL_DOWNLOAD_LIMIT,
    ):
        setup_pipeline_logging()
        self.db = DatabaseManager(db_path)
        self.discovery = PDFDiscovery()
        self.downloader = PDFDownloader(self.db)
        self.checkpoint_mgr = CheckpointManager(self.db)
        self.validator = PipelineValidator(self.db)
        self.download_limit = download_limit

    def run(self, incremental: bool = False, crawl_domain: bool = True) -> Dict[str, Any]:
        """
        Executes pipeline run.
        If incremental=False and first run: downloads latest N (default 100) PDFs.
        If incremental=True: crawls, detects only newly published PDFs, inserts them,
        and downloads only the new PDFs.
        """
        start_time = time.time()
        logger.info("=== crawl started (mode: %s, limit: %s) ===", "incremental" if incremental else "initial/resumable", self.download_limit)

        # 1. Check existing state
        stats_before = self.db.get_stats()
        latest_checkpoint = self.checkpoint_mgr.load_latest_checkpoint()
        is_first_run = stats_before["total_pdfs"] == 0 and latest_checkpoint is None

        # 2. Discover PDFs from NITA
        logger.info("Starting Scrapling PDF discovery...")
        discovered_items = self.discovery.discover_all(crawl_domain=crawl_domain)
        disc_val = self.validator.validate_discovery(discovered_items)
        logger.info(
            "Discovery complete: %d total candidates, %d unique, %d duplicate, %d invalid.",
            disc_val["total_discovered"],
            disc_val["unique_urls"],
            disc_val["duplicate_count"],
            disc_val["invalid_urls_count"],
        )

        # 3. Insert discovered items into SQLite with deduplication
        inserted_count, skipped_count = self.db.insert_many_pdfs(discovered_items)
        logger.info("Database ingestion: %d newly inserted, %d already present.", inserted_count, skipped_count)
        logger.info("=== new PDFs found: %d ===", inserted_count)

        # 4. Determine download candidates
        to_download: List[Dict[str, Any]] = []

        if is_first_run and not incremental:
            # Initial Run: Sort already done by discovery (newest date first / notice index)
            # Pick the top N (default 100)
            candidates = discovered_items[: self.download_limit]
            logger.info("Initial run: selected top %d newest PDFs for download.", len(candidates))
            for item in candidates:
                rec = self.db.get_pdf_by_url(item["pdf_url"])
                if rec and rec["status"] != "downloaded":
                    to_download.append(rec)
        elif incremental:
            # Incremental run: only download newly inserted items
            logger.info("Incremental run: downloading only newly detected PDFs...")
            if inserted_count > 0:
                # Discovered items that were newly inserted
                new_urls = {
                    item["pdf_url"]
                    for item in discovered_items
                    if self.db.get_pdf_by_url(item["pdf_url"])
                    and self.db.get_pdf_by_url(item["pdf_url"])["status"] == "discovered"
                }
                for url in new_urls:
                    rec = self.db.get_pdf_by_url(url)
                    if rec and rec["status"] != "downloaded":
                        to_download.append(rec)
            logger.info("Found %d newly published PDFs to download.", len(to_download))
        else:
            # Resumable run: download pending from previous run up to download_limit
            logger.info("Resumable run: fetching pending download queue from database...")
            to_download = self.db.get_pending_downloads(limit=self.download_limit)

        logger.info("Total PDFs scheduled for download in this execution: %d", len(to_download))

        # 5. Execute downloads with progress and rate limiting
        succeeded_downloads = 0
        failed_downloads = 0
        latest_notice_date = discovered_items[0].get("notice_date") if discovered_items else None

        for idx, rec in enumerate(to_download, 1):
            logger.info("[%d/%d] Processing download: %s", idx, len(to_download), rec["title"][:50])
            result = self.downloader.download_pdf(rec)
            if result:
                succeeded_downloads += 1
            else:
                failed_downloads += 1

            # Politeness delay
            if idx < len(to_download):
                time.sleep(RATE_LIMIT_DELAY_SECONDS)

            # Periodic checkpoint every 25 downloads
            if idx % 25 == 0:
                self.checkpoint_mgr.save_checkpoint(
                    total_discovered=self.db.get_stats()["total_pdfs"],
                    downloaded_count=self.db.get_stats()["downloaded"],
                    last_notice_date=latest_notice_date,
                    status="in_progress",
                )
                logger.info("=== checkpoint saved (interim progress %d/%d) ===", idx, len(to_download))

        # 6. Save final checkpoint
        final_stats = self.db.get_stats()
        self.checkpoint_mgr.save_checkpoint(
            total_discovered=final_stats["total_pdfs"],
            downloaded_count=final_stats["downloaded"],
            last_notice_date=latest_notice_date,
            status="completed",
        )
        logger.info("=== checkpoint saved ===")
        logger.info("=== downloads succeeded: %d ===", succeeded_downloads)
        logger.info("=== downloads failed: %d ===", failed_downloads)

        elapsed = time.time() - start_time
        logger.info("=== crawl finished in %.2f seconds ===", elapsed)

        # 7. Run validation and print report
        download_val = self.validator.validate_downloads()
        report = self.validator.generate_report(
            discovery_validation=disc_val,
            download_validation=download_val,
        )

        return {
            "elapsed_seconds": round(elapsed, 2),
            "discovered_count": len(discovered_items),
            "new_inserted_count": inserted_count,
            "scheduled_download_count": len(to_download),
            "succeeded_downloads": succeeded_downloads,
            "failed_downloads": failed_downloads,
            "stats": final_stats,
            "validation_report": report,
        }
