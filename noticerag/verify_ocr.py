"""
Master Verification CLI Script: OCR Quality Audit & Metadata Extraction.
"""

import sys
from pathlib import Path

# Ensure utf-8 stdout encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
for p in (str(BASE_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from metadata.metadata_pipeline import MetadataPipeline
from verification.verify_ocr import OCRAuditor


def main():
    auditor = OCRAuditor()
    audit_res = auditor.run_audit(sample_size=10)
    summary = audit_res["summary"]

    meta_pipe = MetadataPipeline()
    meta_res = meta_pipe.run()
    coverage = meta_res["coverage"]
    records = meta_res["records"]

    print("\n============================================================")
    print("      NITA CAMPUS INTELLIGENCE - OCR & METADATA AUDIT       ")
    print("============================================================")
    print(f"Total Documents:              {summary['total_documents']}")
    print(f"Average OCR Quality Score:    {summary['average_quality_score']} / 100")
    print(f"Failed OCR Documents:         {summary['failed_documents']}")
    print(f"High Quality (>=80):          {summary['high_quality_documents']}")
    print(f"Medium Quality (50-79):       {summary['medium_quality_documents']}")
    print(f"Low Quality (<50):            {summary['low_quality_documents']}")
    print("------------------------------------------------------------")
    print(f"Title Extraction Coverage:    {coverage['title_coverage_pct']}%")
    print(f"Date Extraction Coverage:     {coverage['date_coverage_pct']}%")
    print(f"Document Type Coverage:       {coverage['type_coverage_pct']}%")
    print(f"Department Coverage:          {coverage['department_coverage_pct']}%")
    print(f"Issuer Coverage:              {coverage['issuer_coverage_pct']}%")
    print("------------------------------------------------------------")
    print(f"Reports Generated:            {audit_res['json_report_path']}")
    print(f"                              {audit_res['html_report_path']}")
    print(f"Master Parquet Dataset:       {meta_res['parquet_file']}")
    print("============================================================\n")

    print("--- SAMPLE DOCUMENTS (Manual Inspection) ---\n")
    sample_docs = records[:5]
    for idx, doc in enumerate(sample_docs, 1):
        print(f"[{idx}] Document ID: {doc['document_id']}")
        print(f"    Title:         {doc['title'][:75]}")
        print(f"    Date:          {doc['date']}")
        print(f"    Type:          {doc['document_type']} (conf: {doc['confidence_scores']['document_type']})")
        print(f"    Department:    {doc['department']} (conf: {doc['confidence_scores']['department']})")
        print(f"    Issuer:        {doc['issuer']} (conf: {doc['confidence_scores']['issuer']})")
        print(f"    Quality Score: {doc['ocr_quality_score']} / 100")
        if doc.get("memo_reference"):
            print(f"    Reference:     {doc['memo_reference']}")
        print()

    print("============================================================\n")


if __name__ == "__main__":
    main()
