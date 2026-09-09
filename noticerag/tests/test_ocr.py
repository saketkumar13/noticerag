"""
Unit and Integration Tests for Phase 2 OCR Pipeline.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from noticerag.crawler.database import DatabaseManager
from noticerag.ocr.cleaner import TextCleaner
from noticerag.ocr.ocr_engine import OCREngine
from noticerag.ocr.pdf_processor import PDFProcessor
from noticerag.ocr.pipeline import OCRPipeline
from noticerag.ocr.text_extractor import PDFTextExtractor
from noticerag.ocr.validator import OCRValidator


class TestTextCleaner(unittest.TestCase):
    def test_unicode_normalization(self):
        text = "Hello\u00a0World\u200b with \x07control chars"
        cleaned = TextCleaner.normalize_unicode(text)
        self.assertNotIn("\u00a0", cleaned)
        self.assertNotIn("\u200b", cleaned)
        self.assertNotIn("\x07", cleaned)
        self.assertIn("Hello World", cleaned)

    def test_repair_hyphenation(self):
        text = "This is an adminis-\ntration notice with B.Sc.-B.Ed."
        cleaned = TextCleaner.repair_hyphenation(text)
        self.assertIn("administration", cleaned)
        self.assertIn("B.Sc.-B.Ed.", cleaned)

    def test_clean_ocr_artifacts(self):
        text = "Valid header\n~\n.\nValid body text\n====================\nValid footer"
        cleaned = TextCleaner.clean_ocr_artifacts(text)
        lines = [l.strip() for l in cleaned.splitlines()]
        self.assertNotIn("~", lines)
        self.assertNotIn(".", lines)
        self.assertIn("Valid header", lines)
        self.assertIn("Valid body text", lines)

    def test_full_clean(self):
        raw = "  Notice Regarding \u00a0 Holiday   \n\n\n\nDate: 04-\n09-2026   "
        res = TextCleaner.clean(raw)
        self.assertEqual(res, "Notice Regarding Holiday\nDate: 04-\n09-2026")


class TestTextExtractor(unittest.TestCase):
    def test_is_text_sufficient(self):
        # Sparse text
        self.assertFalse(PDFTextExtractor.is_text_sufficient("Too short", min_chars=60))
        # Garbage text
        self.assertFalse(PDFTextExtractor.is_text_sufficient("$$%^&*(#@!~`}{][|\\:;?/><," * 5, min_chars=60))
        # Valid text
        valid_sample = "National Institute of Technology Agartala Notice regarding academic calendar and holiday schedule for year 2026."
        self.assertTrue(PDFTextExtractor.is_text_sufficient(valid_sample, min_chars=60))


class TestOCREngine(unittest.TestCase):
    def test_ocr_synthetic_image(self):
        # Create synthetic image with readable text
        img = Image.new("RGB", (400, 100), color=(255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((20, 35), "NATIONAL INSTITUTE", fill=(0, 0, 0))

        engine = OCREngine(primary_engine="RapidOCR")
        text, engine_name, conf = engine.ocr_image(img)
        self.assertIn(engine_name, ["RapidOCR", "Tesseract"])
        self.assertTrue(len(text) > 0)
        self.assertIn("NATIONAL", text.upper())


class TestOCRPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_ocr.db"
        self.ext_dir = Path(self.temp_dir) / "extracted"
        self.pdf_dir = Path(self.temp_dir) / "pdfs"
        self.ext_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dir.mkdir(parents=True, exist_ok=True)

        self.db = DatabaseManager(self.db_path)
        self.pipeline = OCRPipeline(
            db_path=self.db_path,
            extracted_dir=self.ext_dir,
            pdf_dir=self.pdf_dir,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_ocr_metadata_schema_created(self):
        with self.db._get_connection() as conn:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]
            self.assertIn("ocr_metadata", tables)

    def test_process_non_existent_pdf(self):
        self.db.insert_pdf(
            title="Non-existent PDF",
            pdf_url="https://example.com/none.pdf",
            source_page="https://example.com",
        )
        pdf_row = self.db.get_pdf_by_url("https://example.com/none.pdf")
        pdf_row["file_path"] = str(self.pdf_dir / "non_existent.pdf")
        res = self.pipeline.process_document(pdf_row)
        self.assertFalse(res["success"])

        with self.db._get_connection() as conn:
            row = conn.execute("SELECT * FROM ocr_metadata WHERE pdf_id = ?", (pdf_row["id"],)).fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row["status"], "failed")

    def test_validator_statistics_empty(self):
        validator = OCRValidator(self.db, self.ext_dir, self.pdf_dir)
        stats = validator.get_pipeline_statistics()
        self.assertEqual(stats["processed_pdfs"], 0)
        self.assertEqual(stats["failed_pdfs"], 0)


if __name__ == "__main__":
    unittest.main()
