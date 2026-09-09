"""
Metadata Pipeline Orchestrator.
Processes all extracted OCR documents, extracts canonical metadata,
persists individual JSON files in data/metadata/, generates master_metadata.parquet,
and computes extraction coverage statistics.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from noticerag.config import EXTRACTED_TEXT_DIR, METADATA_DIR, PARQUET_FILE
from metadata.classifiers import DocumentClassifier
from metadata.extractor import MetadataExtractor
from metadata.regex_extractors import RegexExtractors
from metadata.validators import MetadataValidator
from verification.quality_metrics import QualityMetrics
from verification.validators import DatasetValidator

logger = logging.getLogger("noticerag.metadata.pipeline")


class MetadataPipeline:
    """
    Coordinates extraction of metadata from OCR texts, individual JSON storage,
    and export of master_metadata.parquet.
    """

    def __init__(
        self,
        extracted_dirs: Optional[List[Path]] = None,
        metadata_dir: Path = METADATA_DIR,
        parquet_file: Path = PARQUET_FILE,
    ):
        self.extracted_dirs = extracted_dirs or [
            EXTRACTED_TEXT_DIR,
            BASE_DIR / "data" / "extracted_text",
            BASE_DIR / "noticerag" / "data" / "extracted_text",
        ]
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.parquet_file = Path(parquet_file)

    def process_document(self, doc_data: Dict[str, Any], file_path: Path) -> Dict[str, Any]:
        """
        Extracts all canonical metadata fields from a single document.
        """
        document_id = doc_data.get("document_id", file_path.stem)
        raw_title = doc_data.get("title", "")
        text = doc_data.get("text", "")
        existing_meta = doc_data.get("metadata", {})
        source_url = existing_meta.get("pdf_url", "")

        # 1. OCR Quality Score
        health = QualityMetrics.compute_health_score(text)
        quality_score = health["quality_score"]

        # 2. Title Extraction
        title, title_conf = MetadataExtractor.extract_title(text, fallback_title=raw_title)

        # 3. Date Extraction
        date_val, date_conf = RegexExtractors.extract_date(text, fallback_filename=document_id)
        if not date_val:
            date_val = "Unknown"

        # 4. Document Type Classification
        class_res = DocumentClassifier.classify(text, title=title, filename=document_id)
        doc_type = class_res["document_type"]
        type_conf = class_res["confidence"]

        # 5. Department Extraction
        dept, dept_conf = MetadataExtractor.extract_department(text, title=title)

        # 6. Issuer Extraction
        issuer, issuer_conf = MetadataExtractor.extract_issuer(text, department=dept)

        # 7. Memo / Reference number
        memo_ref = RegexExtractors.extract_memo_reference(text)

        metadata_record = {
            "document_id": document_id,
            "title": title,
            "date": date_val,
            "document_type": doc_type,
            "source_url": source_url,
            "issuer": issuer,
            "department": dept,
            "ocr_quality_score": quality_score,
            "memo_reference": memo_ref or "",
            "confidence_scores": {
                "title": title_conf,
                "date": date_conf,
                "document_type": type_conf,
                "department": dept_conf,
                "issuer": issuer_conf,
            },
            "stats": {
                "pages": existing_meta.get("pages", 1),
                "word_count": health["stats"]["word_count"],
                "char_count": health["stats"]["char_count"],
            },
        }

        # Save individual JSON: data/metadata/<document_id>.json
        json_target = self.metadata_dir / f"{document_id}.json"
        with open(json_target, "w", encoding="utf-8") as jf:
            json.dump(metadata_record, jf, indent=2, ensure_ascii=False)

        return metadata_record

    def run(self) -> Dict[str, Any]:
        """
        Executes metadata extraction across all discovered documents.
        Exports master_metadata.parquet and generates coverage metrics.
        """
        files = DatasetValidator.find_extracted_files(self.extracted_dirs)
        logger.info("Found %d extracted files for metadata extraction.", len(files))

        all_records = []
        for fpath in files:
            is_valid, data, err = DatasetValidator.validate_file(fpath)
            if is_valid and data:
                rec = self.process_document(data, fpath)
                all_records.append(rec)

        # Compute coverage metrics
        coverage = MetadataValidator.calculate_coverage(all_records)

        # Export master_metadata.parquet
        parquet_rows = []
        for r in all_records:
            parquet_rows.append(
                {
                    "document_id": r["document_id"],
                    "title": r["title"],
                    "date": r["date"],
                    "document_type": r["document_type"],
                    "source_url": r["source_url"],
                    "issuer": r["issuer"],
                    "department": r["department"],
                    "ocr_quality_score": r["ocr_quality_score"],
                    "memo_reference": r["memo_reference"],
                    "pages": r["stats"]["pages"],
                    "word_count": r["stats"]["word_count"],
                    "char_count": r["stats"]["char_count"],
                    "title_confidence": r["confidence_scores"]["title"],
                    "date_confidence": r["confidence_scores"]["date"],
                    "type_confidence": r["confidence_scores"]["document_type"],
                    "dept_confidence": r["confidence_scores"]["department"],
                    "issuer_confidence": r["confidence_scores"]["issuer"],
                }
            )

        df = pd.DataFrame(parquet_rows)
        df.to_parquet(self.parquet_file, index=False, engine="pyarrow")
        logger.info("Saved master metadata parquet to %s (%d rows)", self.parquet_file, len(df))

        return {
            "total_processed": len(all_records),
            "coverage": coverage,
            "parquet_file": str(self.parquet_file),
            "records": all_records,
        }


def main():
    pipeline = MetadataPipeline()
    results = pipeline.run()
    cov = results["coverage"]

    print("\n============================================================")
    print("         METADATA EXTRACTION - COVERAGE REPORT              ")
    print("============================================================")
    print(f"Total Documents Processed: {cov['total_documents']}")
    print(f"Title Coverage:            {cov['title_coverage_pct']}%")
    print(f"Date Coverage:             {cov['date_coverage_pct']}%")
    print(f"Document Type Coverage:    {cov['type_coverage_pct']}%")
    print(f"Department Coverage:       {cov['department_coverage_pct']}%")
    print(f"Issuer Coverage:           {cov['issuer_coverage_pct']}%")
    print(f"Parquet Dataset:           {results['parquet_file']}")
    print("============================================================\n")


if __name__ == "__main__":
    main()
