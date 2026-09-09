"""
Tests for NoticeRAG Validators.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from noticerag.crawler.database import DatabaseManager
from noticerag.crawler.downloader import compute_sha256
from noticerag.crawler.validators import PipelineValidator


class TestPipelineValidatorDetails(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "val_test.db"
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

    def test_download_validation_clean(self):
        # Create a genuine dummy PDF on disk
        pdf_file = self.pdf_dir / "valid_notice.pdf"
        dummy_content = b"%PDF-1.4\n1 0 obj<<>>endobj\ntrailer<<>>%%EOF"
        pdf_file.write_bytes(dummy_content)
        file_hash = compute_sha256(pdf_file)

        # Create matching OCR metadata JSON
        meta_file = self.meta_dir / "valid_notice.json"
        meta_content = {
            "title": "Valid Notice",
            "pdf_url": "https://www.nita.ac.in/valid_notice.pdf",
            "source_page": "https://www.nita.ac.in/notices",
            "file_path": str(pdf_file.resolve()),
            "sha256": file_hash,
            "downloaded_at": "2026-09-06T18:00:00Z",
        }
        meta_file.write_text(json.dumps(meta_content), encoding="utf-8")

        # Insert into DB
        self.db.insert_pdf(
            title="Valid Notice",
            pdf_url="https://www.nita.ac.in/valid_notice.pdf",
            source_page="https://www.nita.ac.in/notices",
        )
        self.db.update_pdf_downloaded(
            pdf_url="https://www.nita.ac.in/valid_notice.pdf",
            file_path=str(pdf_file.resolve()),
            file_hash=file_hash,
            downloaded_at="2026-09-06T18:00:00Z",
        )

        res = self.validator.validate_downloads()
        self.assertTrue(res["is_valid"])
        self.assertEqual(res["status_downloaded"], 1)
        self.assertEqual(res["missing_files_count"], 0)
        self.assertEqual(res["zero_byte_files_count"], 0)
        self.assertEqual(res["hash_mismatches_count"], 0)
        self.assertEqual(res["missing_ocr_metadata_count"], 0)
        self.assertEqual(res["download_success_rate_pct"], 100.0)

        report = self.validator.generate_report(download_validation=res)
        self.assertIn("NITA NOTICERAG VALIDATION REPORT", report)
        self.assertIn("Storage Valid:        PASSED", report)

    def test_download_validation_corrupt_and_missing(self):
        # 1. Missing file in DB
        self.db.insert_pdf(
            title="Missing Notice",
            pdf_url="https://www.nita.ac.in/missing.pdf",
            source_page="https://www.nita.ac.in/notices",
        )
        self.db.update_pdf_downloaded(
            pdf_url="https://www.nita.ac.in/missing.pdf",
            file_path=str(self.pdf_dir / "non_existent.pdf"),
            file_hash="fakehash",
        )

        # 2. Zero-byte file
        zero_file = self.pdf_dir / "zero.pdf"
        zero_file.write_bytes(b"")
        self.db.insert_pdf(
            title="Zero Notice",
            pdf_url="https://www.nita.ac.in/zero.pdf",
            source_page="https://www.nita.ac.in/notices",
        )
        self.db.update_pdf_downloaded(
            pdf_url="https://www.nita.ac.in/zero.pdf",
            file_path=str(zero_file.resolve()),
            file_hash="fakehash2",
        )

        res = self.validator.validate_downloads()
        self.assertFalse(res["is_valid"])
        self.assertEqual(res["missing_files_count"], 1)
        self.assertEqual(res["zero_byte_files_count"], 1)


if __name__ == "__main__":
    unittest.main()
