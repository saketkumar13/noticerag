"""
Verification Runner Module for OCR Quality Audit.
Coordinates document audit, health score computation, sampling, and report generation.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from noticerag.config import DATA_DIR, EXTRACTED_TEXT_DIR, REPORTS_DIR
from verification.quality_metrics import QualityMetrics
from verification.reports import ReportGenerator
from verification.sampling import QualitySampler
from verification.validators import DatasetValidator

logger = logging.getLogger("noticerag.verification")


class OCRAuditor:
    """
    Orchestrates OCR quality auditing across extracted text JSON documents.
    """

    def __init__(
        self,
        extracted_dirs: Optional[List[Path]] = None,
        reports_dir: Path = REPORTS_DIR,
    ):
        self.extracted_dirs = extracted_dirs or [
            EXTRACTED_TEXT_DIR,
            BASE_DIR / "data" / "extracted_text",
            BASE_DIR / "noticerag" / "data" / "extracted_text",
        ]
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run_audit(self, sample_size: int = 10) -> Dict[str, Any]:
        """
        Runs complete OCR quality audit on all discovered documents.
        """
        files = DatasetValidator.find_extracted_files(self.extracted_dirs)
        logger.info("Found %d extracted documents to audit.", len(files))

        audit_results: List[Dict[str, Any]] = []

        for fpath in files:
            is_valid, data, err = DatasetValidator.validate_file(fpath)
            if not is_valid:
                doc_id = data.get("document_id", fpath.stem) if data else fpath.stem
                title = data.get("title", "") if data else ""
                audit_results.append(
                    {
                        "document_id": doc_id,
                        "title": title,
                        "file_path": str(fpath),
                        "quality_score": 0,
                        "tier": "Failed",
                        "is_suspicious": True,
                        "suspicious_patterns": [err or "Validation failure"],
                        "stats": {"char_count": 0, "word_count": 0, "line_count": 0, "paragraph_count": 0},
                        "metrics": {},
                        "text": "",
                    }
                )
                continue

            doc_id = data.get("document_id", fpath.stem)
            title = data.get("title", "")
            text = data.get("text", "")

            score_data = QualityMetrics.compute_health_score(text)
            audit_results.append(
                {
                    "document_id": doc_id,
                    "title": title,
                    "file_path": str(fpath),
                    "quality_score": score_data["quality_score"],
                    "tier": score_data["tier"],
                    "is_suspicious": score_data["is_suspicious"],
                    "suspicious_patterns": score_data["suspicious_patterns"],
                    "stats": score_data["stats"],
                    "metrics": score_data["metrics"],
                    "text": text,
                }
            )

        # Generate summary and samples
        summary = ReportGenerator.generate_summary(audit_results)
        samples = QualitySampler.sample_documents(audit_results, sample_size=sample_size)

        # Save reports
        json_report_path = self.reports_dir / "ocr_report.json"
        html_report_path = self.reports_dir / "ocr_report.html"

        ReportGenerator.save_json_report(summary, samples, audit_results, json_report_path)
        ReportGenerator.save_html_report(summary, samples, html_report_path)

        logger.info("Saved JSON report to %s", json_report_path)
        logger.info("Saved HTML report to %s", html_report_path)

        return {
            "summary": summary,
            "samples": samples,
            "audit_results": audit_results,
            "json_report_path": json_report_path,
            "html_report_path": html_report_path,
        }


def main():
    auditor = OCRAuditor()
    results = auditor.run_audit()
    summary = results["summary"]

    print("\n============================================================")
    print("           OCR QUALITY AUDIT - SUMMARY REPORT               ")
    print("============================================================")
    print(f"Total Documents:         {summary['total_documents']}")
    print(f"Average Quality Score:   {summary['average_quality_score']} / 100")
    print(f"Average Word Count:      {summary['average_word_count']}")
    print(f"Average Character Count: {summary['average_character_count']}")
    print(f"High Quality (>=80):     {summary['high_quality_documents']}")
    print(f"Medium Quality (50-79):  {summary['medium_quality_documents']}")
    print(f"Low Quality (<50):       {summary['low_quality_documents']}")
    print(f"Failed Documents:        {summary['failed_documents']}")
    print(f"Suspicious Documents:    {summary['suspicious_documents_count']}")
    print(f"JSON Report Saved:       {results['json_report_path']}")
    print(f"HTML Report Saved:       {results['html_report_path']}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
