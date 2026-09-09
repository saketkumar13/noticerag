"""
Unit and Integration Tests for NITA NoticeRAG Pipeline.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from noticerag.crawler.checkpoint import CheckpointManager
from noticerag.crawler.database import DatabaseManager
from noticerag.crawler.downloader import PDFDownloader, compute_sha256, sanitize_filename
from noticerag.crawler.pdf_discovery import (
    PDFDiscovery,
    extract_date_from_filename_or_text,
    normalize_url,
    parse_date_string,
)
from noticerag.crawler.validators import PipelineValidator


class TestDatabase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_rag.db"
        self.db = DatabaseManager(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tables_created(self):
        with self.db._get_connection() as conn:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            self.assertIn("PDFs", tables)
            self.assertIn("Crawl Checkpoints", tables)

    def test_insert_and_deduplication(self):
        inserted = self.db.insert_pdf(
            title="Notice 1",
            pdf_url="https://www.nita.ac.in/notice1.pdf",
            source_page="https://www.nita.ac.in/notices",
        )
        self.assertTrue(inserted)

        # Duplicate insert should return False
        dup_inserted = self.db.insert_pdf(
            title="Notice 1 Duplicate",
            pdf_url="https://www.nita.ac.in/notice1.pdf",
            source_page="https://www.nita.ac.in/notices",
        )
        self.assertFalse(dup_inserted)

        # Bulk insert test
        records = [
            {"title": "Notice 1", "pdf_url": "https://www.nita.ac.in/notice1.pdf", "source_page": "p"},
            {"title": "Notice 2", "pdf_url": "https://www.nita.ac.in/notice2.pdf", "source_page": "p"},
            {"title": "Notice 3", "pdf_url": "https://www.nita.ac.in/notice3.pdf", "source_page": "p"},
        ]
        ins, skip = self.db.insert_many_pdfs(records)
        self.assertEqual(ins, 2)
        self.assertEqual(skip, 1)

        stats = self.db.get_stats()
        self.assertEqual(stats["total_pdfs"], 3)
        self.assertEqual(stats["discovered"], 3)

    def test_update_downloaded_status(self):
        self.db.insert_pdf(
            title="Test Notice",
            pdf_url="https://www.nita.ac.in/doc.pdf",
            source_page="https://www.nita.ac.in",
        )
        updated = self.db.update_pdf_downloaded(
            pdf_url="https://www.nita.ac.in/doc.pdf",
            file_path="/data/pdfs/doc.pdf",
            file_hash="dummyhash123",
        )
        self.assertTrue(updated)

        rec = self.db.get_pdf_by_url("https://www.nita.ac.in/doc.pdf")
        self.assertIsNotNone(rec)
        self.assertEqual(rec["status"], "downloaded")
        self.assertEqual(rec["file_hash"], "dummyhash123")


class TestCheckpoints(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_rag.db"
        self.ck_dir = Path(self.temp_dir) / "checkpoints"
        self.db = DatabaseManager(self.db_path)
        self.ck_mgr = CheckpointManager(self.db, checkpoint_dir=self.ck_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_checkpoint(self):
        # Initial state: no checkpoint
        initial = self.ck_mgr.load_latest_checkpoint()
        self.assertIsNone(initial)

        # Save checkpoint
        saved = self.ck_mgr.save_checkpoint(
            total_discovered=1050,
            downloaded_count=100,
            last_notice_date="2026-09-03",
            status="completed",
        )
        self.assertEqual(saved["total_pdfs"], 1050)
        self.assertEqual(saved["downloaded_count"], 100)

        # Reload
        loaded = self.ck_mgr.load_latest_checkpoint()
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["total_pdfs"], 1050)
        self.assertEqual(loaded["downloaded_count"], 100)
        self.assertEqual(loaded["last_notice_date"], "2026-09-03")


class TestDownloaderAndOCRMetadata(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_rag.db"
        self.pdf_dir = Path(self.temp_dir) / "pdfs"
        self.meta_dir = Path(self.temp_dir) / "metadata"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        self.db = DatabaseManager(self.db_path)
        self.downloader = PDFDownloader(
            db=self.db,
            pdf_dir=self.pdf_dir,
            metadata_dir=self.meta_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_filename_sanitizer(self):
        self.assertEqual(sanitize_filename("Valid_Name.pdf"), "Valid_Name.pdf")
        self.assertEqual(sanitize_filename('Bad/Name:With*Chars?.pdf'), "Bad_Name_With_Chars_.pdf")

    def test_mock_pdf_download_and_ocr_metadata(self):
        dummy_pdf_bytes = b"%PDF-1.4\n%test content line\n%%EOF"
        pdf_url = "https://www.nita.ac.in/dummy_notice.pdf"

        self.db.insert_pdf(
            title="Sample Notice Title",
            pdf_url=pdf_url,
            source_page="https://www.nita.ac.in/UserPanel/ViewAllNewsAndEvents.aspx",
        )

        # Mock requests download stream
        with patch.object(self.downloader, "_download_stream", return_value=(True, dummy_pdf_bytes, pdf_url)):
            rec = self.db.get_pdf_by_url(pdf_url)
            ocr_meta = self.downloader.download_pdf(rec)

            self.assertIsNotNone(ocr_meta)
            self.assertEqual(ocr_meta["title"], "Sample Notice Title")
            self.assertEqual(ocr_meta["pdf_url"], pdf_url)
            self.assertTrue(Path(ocr_meta["file_path"]).exists())
            self.assertTrue(ocr_meta["sha256"])

            # Verify saved OCR JSON
            json_files = list(self.meta_dir.glob("*.json"))
            self.assertEqual(len(json_files), 1)

            with open(json_files[0], "r", encoding="utf-8") as f:
                saved_json = json.load(f)
                self.assertEqual(saved_json["title"], "Sample Notice Title")
                self.assertEqual(saved_json["pdf_url"], pdf_url)
                self.assertIn("sha256", saved_json)
                self.assertIn("downloaded_at", saved_json)


class TestValidators(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_rag.db"
        self.pdf_dir = Path(self.temp_dir) / "pdfs"
        self.meta_dir = Path(self.temp_dir) / "metadata"
        self.pdf_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        self.db = DatabaseManager(self.db_path)
        self.validator = PipelineValidator(
            db=self.db,
            pdf_dir=self.pdf_dir,
            metadata_dir=self.meta_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_discovery_validation(self):
        items = [
            {"pdf_url": "https://www.nita.ac.in/a.pdf"},
            {"pdf_url": "https://www.nita.ac.in/b.pdf"},
            {"pdf_url": "https://www.nita.ac.in/a.pdf"},  # duplicate
            {"pdf_url": "javascript:void(0)"},          # invalid
        ]
        res = self.validator.validate_discovery(items)
        self.assertEqual(res["total_discovered"], 4)
        self.assertEqual(res["unique_urls"], 3)
        self.assertEqual(res["duplicate_count"], 1)
        self.assertEqual(res["invalid_urls_count"], 1)
        self.assertFalse(res["is_valid"])


class TestDateParsingAndNormalization(unittest.TestCase):
    def test_date_parser(self):
        dt = parse_date_string("03/09/2026")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.day, 3)
        self.assertEqual(dt.month, 9)
        self.assertEqual(dt.year, 2026)

        dt_hyphen = parse_date_string("25-08-2026")
        self.assertIsNotNone(dt_hyphen)
        self.assertEqual(dt_hyphen.day, 25)

    def test_extract_date_regex(self):
        dt = extract_date_from_filename_or_text("Notice_03-09-2026_Holiday-Janmashtami_admin.pdf")
        self.assertIsNotNone(dt)
        self.assertEqual(dt.day, 3)
        self.assertEqual(dt.month, 9)
        self.assertEqual(dt.year, 2026)

    def test_normalize_url(self):
        base = "https://www.nita.ac.in/UserPanel/ViewAllNewsAndEvents.aspx?nModuleID=gi"
        rel = "../Notice_03-09-2026.pdf"
        norm = normalize_url(base, rel)
        self.assertEqual(norm, "https://www.nita.ac.in/Notice_03-09-2026.pdf")


class TestLiveNITA(unittest.TestCase):
    """Verifies live Scrapling reachability to NITA notice board."""

    def test_scrapling_notice_discovery(self):
        discovery = PDFDiscovery()
        notices = discovery.discover_from_notice_board()
        self.assertGreater(len(notices), 50, "Expected at least 50 notices from NITA")
        first = notices[0]
        self.assertTrue(first["pdf_url"].startswith("http"))
        self.assertTrue(".pdf" in first["pdf_url"].lower())


if __name__ == "__main__":
    unittest.main()
