"""
Validation and Quality Evaluation Module for Phase 2 OCR Pipeline.
Computes comprehensive processing metrics, validates text outputs,
and generates structured quality reports with random document sampling.
"""

import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from noticerag.config import EXTRACTED_TEXT_DIR, PDF_DIR
from noticerag.crawler.database import DatabaseManager

logger = logging.getLogger("noticerag.ocr.validator")


class OCRValidator:
    """
    Validates OCR pipeline execution, calculates metrics, and samples output quality.
    """

    def __init__(
        self,
        db: DatabaseManager,
        extracted_dir: Path = EXTRACTED_TEXT_DIR,
        pdf_dir: Path = PDF_DIR,
    ):
        self.db = db
        self.extracted_dir = Path(extracted_dir)
        self.pdf_dir = Path(pdf_dir)

    def get_pipeline_statistics(self) -> Dict[str, Any]:
        """
        Calculates complete statistics across database and extracted text files.
        """
        # Read from SQLite ocr_metadata table
        with self.db._get_connection() as conn:
            # Check if ocr_metadata table exists
            table_check = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='ocr_metadata'"
            ).fetchone()

            if not table_check:
                return {
                    "total_pdfs": 0,
                    "processed_pdfs": 0,
                    "ocr_pdfs": 0,
                    "digital_pdfs": 0,
                    "failed_pdfs": 0,
                    "ocr_usage_pct": 0.0,
                    "average_text_length": 0.0,
                    "average_word_count": 0.0,
                }

            rows = conn.execute("SELECT * FROM ocr_metadata").fetchall()
            total_pdfs = conn.execute("SELECT COUNT(*) FROM PDFs WHERE status = 'downloaded'").fetchone()[0]

        records = [dict(r) for r in rows]
        processed_count = len(records)
        ocr_count = sum(1 for r in records if r.get("ocr_used") == 1 or r.get("ocr_used") is True)
        digital_count = sum(1 for r in records if (r.get("ocr_used") == 0 or r.get("ocr_used") is False) and r.get("status") == "success")
        failed_count = sum(1 for r in records if r.get("status") == "failed")

        text_lengths = [r.get("text_length", 0) for r in records if r.get("status") == "success"]
        avg_text_length = (sum(text_lengths) / len(text_lengths)) if text_lengths else 0.0

        ocr_pct = (ocr_count / processed_count * 100.0) if processed_count > 0 else 0.0

        return {
            "total_pdfs": total_pdfs,
            "processed_pdfs": processed_count,
            "ocr_pdfs": ocr_count,
            "digital_pdfs": digital_count,
            "failed_pdfs": failed_count,
            "ocr_usage_pct": round(ocr_pct, 2),
            "average_text_length": round(avg_text_length, 2),
        }

    def sample_quality(self, sample_size: int = 10) -> List[Dict[str, Any]]:
        """
        Randomly samples processed JSON documents to evaluate extraction quality.
        Returns for each sampled document:
          - document_id
          - title
          - first 500 characters
          - ocr_used
          - text_length
          - word_count
        """
        json_files = list(self.extracted_dir.glob("*.json"))
        if not json_files:
            return []

        sampled_files = random.sample(json_files, min(sample_size, len(json_files)))
        samples: List[Dict[str, Any]] = []

        for jf in sampled_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)

                text = data.get("text", "")
                metadata = data.get("metadata", {})
                snippet = text[:500] if len(text) > 500 else text

                samples.append(
                    {
                        "document_id": data.get("document_id", jf.stem),
                        "title": data.get("title", ""),
                        "ocr_used": metadata.get("ocr_used", False),
                        "ocr_engine": metadata.get("ocr_engine", "None"),
                        "text_length": len(text),
                        "word_count": len(text.split()),
                        "first_500_chars": snippet,
                    }
                )
            except Exception as exc:
                logger.error("Error reading %s for quality sample: %s", jf.name, exc)

        return samples

    def generate_report(self) -> str:
        """
        Generates formatted terminal report containing pipeline statistics
        and 10 quality verification samples.
        """
        stats = self.get_pipeline_statistics()
        samples = self.sample_quality(10)

        lines = [
            "============================================================",
            "        NITA CAMPUS INTELLIGENCE - OCR PIPELINE REPORT      ",
            "============================================================",
            f"Total Downloaded PDFs:  {stats['total_pdfs']}",
            f"Processed PDFs:         {stats['processed_pdfs']}",
            f"OCR Scanned PDFs:       {stats['ocr_pdfs']}",
            f"Digital PDFs:           {stats['digital_pdfs']}",
            f"Failed PDFs:            {stats['failed_pdfs']}",
            f"OCR Usage Percentage:   {stats['ocr_usage_pct']}%",
            f"Average Text Length:    {stats['average_text_length']} characters",
            "============================================================",
            "",
            "--- QUALITY EVALUATION SAMPLES (10 Processed PDFs) ---",
        ]

        for idx, sample in enumerate(samples, 1):
            method = f"OCR ({sample['ocr_engine']})" if sample["ocr_used"] else "Digital Direct"
            preview = sample["first_500_chars"].replace("\n", " ")
            if len(preview) > 200:
                preview = preview[:200] + "..."

            lines.extend(
                [
                    f"\n[{idx}] Title: {sample['title'][:70]}",
                    f"    Doc ID:      {sample['document_id']}",
                    f"    Method:      {method}",
                    f"    Text Length: {sample['text_length']} chars | Words: {sample['word_count']}",
                    f"    Preview:     \"{preview}\"",
                ]
            )

        lines.append("\n============================================================\n")
        return "\n".join(lines)
