"""
Unit and Integration Tests for Verification and Metadata Extraction Packages.
"""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from metadata.classifiers import DocumentClassifier
from metadata.extractor import MetadataExtractor
from metadata.metadata_pipeline import MetadataPipeline
from metadata.regex_extractors import RegexExtractors
from metadata.validators import MetadataValidator
from verification.quality_metrics import QualityMetrics
from verification.reports import ReportGenerator
from verification.sampling import QualitySampler
from verification.validators import DatasetValidator


class TestQualityMetrics(unittest.TestCase):
    def test_clean_text_health_score(self):
        clean_text = (
            "National Institute of Technology Agartala Notification.\n"
            "This is to inform all faculty members and students that the institute "
            "will observe the holiday on account of Janmashtami on 4th September 2026. "
            "All classes and offices will remain closed. Essential services like water, "
            "electricity and security will function as usual.\n"
            "Issued by the Registrar."
        )
        res = QualityMetrics.compute_health_score(clean_text)
        self.assertGreaterEqual(res["quality_score"], 75)
        self.assertIn(res["tier"], ["High Quality", "Medium Quality"])
        self.assertFalse(res["is_suspicious"])

    def test_garbage_text_detection(self):
        garbage_text = "%%%%% IIIIII OOOOOO 111111 |||||| ^^^^^^ ~~~~~~"
        res = QualityMetrics.compute_health_score(garbage_text)
        self.assertTrue(res["is_suspicious"])
        self.assertLess(res["quality_score"], 50)
        self.assertEqual(res["tier"], "Low Quality")

    def test_empty_text(self):
        res = QualityMetrics.compute_health_score("")
        self.assertEqual(res["quality_score"], 0)
        self.assertEqual(res["tier"], "Failed")


class TestRegexExtractors(unittest.TestCase):
    def test_date_extraction_formats(self):
        # Format 1: Dated: 03/09/2026
        d1, conf1 = RegexExtractors.extract_date("Dated: 03/09/2026\nSubject: Holiday")
        self.assertEqual(d1, "2026-09-03")
        self.assertGreater(conf1, 0.9)

        # Format 2: Date: 28.07.2026
        d2, conf2 = RegexExtractors.extract_date("Date: 28.07.2026\nNotification")
        self.assertEqual(d2, "2026-07-28")

        # Format 3: 15th August 2026
        d3, conf3 = RegexExtractors.extract_date("Celebration of Independence Day on 15th August 2026")
        self.assertEqual(d3, "2026-08-15")

        # Format 4: Filename fallback
        d4, conf4 = RegexExtractors.extract_date("Some notice text without date", fallback_filename="Notice_01-09-2026_Admin.pdf")
        self.assertEqual(d4, "2026-09-01")

    def test_memo_reference(self):
        text = "National Institute of Technology Agartala\nNo.F.NITA.3(6-Gen)/2011/4082-84\nDate: 03/09/2026"
        ref = RegexExtractors.extract_memo_reference(text)
        self.assertIsNotNone(ref)
        self.assertIn("NITA.3(6-Gen)/2011/4082-84", ref)


class TestClassifiers(unittest.TestCase):
    def test_holiday_classification(self):
        res = DocumentClassifier.classify("Declaration of Holiday on 4th September on account of Janmashtami")
        self.assertEqual(res["document_type"], "Holiday Notice")
        self.assertGreaterEqual(res["confidence"], 0.8)

    def test_tender_classification(self):
        res = DocumentClassifier.classify("Notice inviting Price Bid and Spot Quotation for Air Conditioners")
        self.assertEqual(res["document_type"], "Tender")
        self.assertGreater(res["confidence"], 0.8)

    def test_hostel_classification(self):
        res = DocumentClassifier.classify("Notification for Hostel Admission and Chief Warden Committee for First Year")
        self.assertEqual(res["document_type"], "Hostel Notice")

    def test_academic_classification(self):
        res = DocumentClassifier.classify("Notification for Provisional Admission in Ph.D and M.Tech Programme 2026")
        self.assertEqual(res["document_type"], "Academic Notice")


class TestExtractor(unittest.TestCase):
    def test_title_extraction(self):
        text = (
            "National Institute of Technology Agartala\n"
            "Office of the Registrar\n"
            "Date: 03/09/2026\n"
            "NOTICE\n"
            "Sub: Declaration of Holiday on account of Janmashtami.\n"
            "In pursuance of the holiday list..."
        )
        title, conf = MetadataExtractor.extract_title(text)
        self.assertIn("Holiday on account of Janmashtami", title)
        self.assertGreaterEqual(conf, 0.9)

    def test_department_and_issuer_extraction(self):
        text = (
            "National Institute of Technology Agartala\n"
            "Office of the Dean Academic\n"
            "nitadeanacademic@gmail.com\n"
            "Notice regarding branch change.\n\n"
            "Dean (Academic)"
        )
        dept, d_conf = MetadataExtractor.extract_department(text)
        self.assertEqual(dept, "Dean (Academic)")
        self.assertGreaterEqual(d_conf, 0.9)

        issuer, i_conf = MetadataExtractor.extract_issuer(text, department=dept)
        self.assertEqual(issuer, "Dean (Academic)")


class TestMetadataPipeline(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.ext_dir = Path(self.temp_dir) / "extracted"
        self.meta_dir = Path(self.temp_dir) / "metadata"
        self.parquet_path = self.meta_dir / "test_master.parquet"
        self.ext_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

        # Create dummy extracted document
        doc_data = {
            "document_id": "test_doc_01",
            "title": "Notice_03-09-2026_Holiday-Janmashtami_admin",
            "text": (
                "National Institute of Technology Agartala\n"
                "No.F.NITA.3(6-Gen)/2011/4082-84\n"
                "Date: 03/09/2026\n"
                "NOTICE\n"
                "Sub: Declaration of Holiday on 4th September 2026 on account of Janmashtami.\n"
                "The institute will remain closed.\n"
                "Registrar"
            ),
            "metadata": {
                "pdf_id": 1,
                "pdf_url": "https://example.com/holiday.pdf",
                "pages": 1,
            },
        }
        with open(self.ext_dir / "test_doc_01.json", "w", encoding="utf-8") as f:
            json.dump(doc_data, f)

        self.pipeline = MetadataPipeline(
            extracted_dirs=[self.ext_dir],
            metadata_dir=self.meta_dir,
            parquet_file=self.parquet_path,
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_pipeline_run_and_parquet(self):
        res = self.pipeline.run()
        self.assertEqual(res["total_processed"], 1)

        # Check JSON generated
        meta_json = self.meta_dir / "test_doc_01.json"
        self.assertTrue(meta_json.exists())
        with open(meta_json, "r", encoding="utf-8") as jf:
            m = json.load(jf)
            self.assertEqual(m["document_id"], "test_doc_01")
            self.assertEqual(m["date"], "2026-09-03")
            self.assertEqual(m["document_type"], "Holiday Notice")
            self.assertEqual(m["issuer"], "Registrar")

        # Check Parquet generated and readable by pandas
        self.assertTrue(self.parquet_path.exists())
        df = pd.read_parquet(self.parquet_path)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["document_id"], "test_doc_01")
        self.assertEqual(df.iloc[0]["document_type"], "Holiday Notice")


if __name__ == "__main__":
    unittest.main()
