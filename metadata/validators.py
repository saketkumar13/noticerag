"""
Metadata Validation and Coverage Metrics Module.
Calculates extraction coverage across canonical metadata fields:
Title, Date, Document Type, Department, and Issuer.
"""

from typing import Any, Dict, List


class MetadataValidator:
    """
    Validates metadata completeness and calculates coverage statistics.
    """

    @classmethod
    def calculate_coverage(cls, metadata_records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates extraction coverage percentages across the corpus.
        """
        total = len(metadata_records)
        if total == 0:
            return {
                "total_documents": 0,
                "title_coverage_pct": 0.0,
                "date_coverage_pct": 0.0,
                "type_coverage_pct": 0.0,
                "department_coverage_pct": 0.0,
                "issuer_coverage_pct": 0.0,
                "counts": {
                    "title": 0,
                    "date": 0,
                    "document_type": 0,
                    "department": 0,
                    "issuer": 0,
                },
            }

        title_count = sum(1 for r in metadata_records if r.get("title") and r.get("title") != "University Notice")
        # Accept valid YYYY-MM-DD format
        date_count = sum(1 for r in metadata_records if r.get("date") and r.get("date") != "Unknown" and len(r.get("date")) == 10)
        type_count = sum(1 for r in metadata_records if r.get("document_type") and r.get("document_type") != "Unknown")
        dept_count = sum(1 for r in metadata_records if r.get("department") and r.get("department") != "General Administration")
        issuer_count = sum(1 for r in metadata_records if r.get("issuer") and r.get("issuer") != "Competent Authority")

        return {
            "total_documents": total,
            "title_coverage_pct": round(title_count / total * 100.0, 1),
            "date_coverage_pct": round(date_count / total * 100.0, 1),
            "type_coverage_pct": round(type_count / total * 100.0, 1),
            "department_coverage_pct": round(dept_count / total * 100.0, 1),
            "issuer_coverage_pct": round(issuer_count / total * 100.0, 1),
            "counts": {
                "title": title_count,
                "date": date_count,
                "document_type": type_count,
                "department": dept_count,
                "issuer": issuer_count,
            },
        }
