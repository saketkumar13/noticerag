"""
CLI Entry Point for NITA NoticeRAG - Phase 1 Ingestion Pipeline.
"""

import argparse
import sys
from pathlib import Path

# Ensure package root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
for p in (str(BASE_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from noticerag.config import DB_PATH, INITIAL_DOWNLOAD_LIMIT
from noticerag.crawler.checkpoint import CheckpointManager
from noticerag.crawler.crawler import NoticeRAGPipeline
from noticerag.crawler.database import DatabaseManager
from noticerag.crawler.validators import PipelineValidator


def display_status(db_path: Path) -> None:
    """Displays current SQLite and Checkpoint statistics."""
    db = DatabaseManager(db_path)
    stats = db.get_stats()
    ck_mgr = CheckpointManager(db)
    latest_ck = ck_mgr.load_latest_checkpoint()

    print("\n============================================================")
    print("             NITA NOTICERAG - CURRENT STATUS                ")
    print("============================================================")
    print(f"Database Location:    {db_path}")
    print(f"Total PDFs in DB:     {stats['total_pdfs']}")
    print(f"Successfully Downloaded: {stats['downloaded']}")
    print(f"Discovered Only:      {stats['discovered']}")
    print(f"Failed Downloads:     {stats['failed']}")
    print(f"Checkpoints Count:    {stats['checkpoints_count']}")
    print("------------------------------------------------------------")
    if latest_ck:
        print(f"Latest Checkpoint:    {latest_ck.get('last_run')}")
        print(f"Last Notice Date:     {latest_ck.get('last_notice_date')}")
        print(f"Checkpoint Status:    {latest_ck.get('status')}")
    else:
        print("Latest Checkpoint:    None (pipeline has not run yet)")
    print("============================================================\n")


def run_verification(db_path: Path) -> None:
    """Runs data integrity checks and prints formatted report."""
    db = DatabaseManager(db_path)
    validator = PipelineValidator(db)
    download_val = validator.validate_downloads()
    report = validator.generate_report(download_validation=download_val)
    print("\n" + report + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="NITA NoticeRAG - Phase 1 PDF Discovery and Ingestion Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Execute the pipeline (discovers notices, persists metadata, downloads PDFs)",
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="Run in incremental mode: detect newly published PDFs only and skip existing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=INITIAL_DOWNLOAD_LIMIT,
        help=f"Number of PDFs to download on initial/resumable run (default: {INITIAL_DOWNLOAD_LIMIT})",
    )
    parser.add_argument(
        "--skip-domain",
        action="store_true",
        help="Skip crawling additional domain pages (crawl notice board only)",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Run validation checks on existing downloads, SQLite DB, and OCR metadata",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Display summary statistics from database and latest checkpoint",
    )

    args = parser.parse_args()

    # Default action if no flag specified is status or help
    if not any([args.run, args.verify, args.status]):
        parser.print_help()
        sys.exit(0)

    if args.status:
        display_status(DB_PATH)

    if args.verify:
        run_verification(DB_PATH)

    if args.run:
        pipeline = NoticeRAGPipeline(
            db_path=DB_PATH,
            download_limit=args.limit,
        )
        results = pipeline.run(
            incremental=args.incremental,
            crawl_domain=not args.skip_domain,
        )
        print("\n" + results["validation_report"] + "\n")


if __name__ == "__main__":
    main()
