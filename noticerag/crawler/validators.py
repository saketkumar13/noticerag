"""
Validation Module for NITA NoticeRAG.
Provides discovery and download validation, data integrity checks, and report generation.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

from noticerag.config import METADATA_DIR, PDF_DIR
from noticerag.crawler.database import DatabaseManager

logger = logging.getLogger("noticerag.validators")


def is_valid_url(url: str) -> bool:
    """Checks if a URL has valid scheme and network location."""
    if not url:
        return False
    parsed = urlparse(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def compute_file_sha256(path: Path) -> str:
    """Computes SHA256 of file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha.update(chunk)
    return sha.hexdigest()


class PipelineValidator:
    """
    Validates discovery output, downloaded files, database integrity, and OCR readiness.
    """

    def __init__(
        self,
        db: DatabaseManager,
        pdf_dir: Path = PDF_DIR,
        metadata_dir: Path = METADATA_DIR,
    ):
        self.db = db
        self.pdf_dir = Path(pdf_dir)
        self.metadata_dir = Path(metadata_dir)

    def validate_discovery(
        self, discovered_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validates discovery records:
        - total PDFs discovered
        - duplicate count
        - invalid URLs
        """
        total_discovered = len(discovered_records)
        seen_urls: Set[str] = set()
        duplicate_count = 0
        invalid_urls: List[str] = []

        for item in discovered_records:
            url = item.get("pdf_url", "")
            if not is_valid_url(url):
                invalid_urls.append(url)
            if url in seen_urls:
                duplicate_count += 1
            else:
                seen_urls.add(url)

        return {
            "total_discovered": total_discovered,
            "unique_urls": len(seen_urls),
            "duplicate_count": duplicate_count,
            "invalid_urls_count": len(invalid_urls),
            "invalid_urls": invalid_urls[:10],  # preview up to 10
            "is_valid": len(invalid_urls) == 0,
        }

    def validate_downloads(self) -> Dict[str, Any]:
        """
        Validates downloaded files on disk and in database:
        - download success rate
        - missing files
        - zero-byte files
        - hash discrepancies
        - OCR metadata readiness
        """
        db_records = self.db.get_all_pdfs()
        total_in_db = len(db_records)

        downloaded_records = [r for r in db_records if r["status"] == "downloaded"]
        failed_records = [r for r in db_records if r["status"] == "failed"]
        discovered_only = [r for r in db_records if r["status"] == "discovered"]

        missing_files: List[str] = []
        zero_byte_files: List[str] = []
        hash_mismatches: List[str] = []
        invalid_pdf_headers: List[str] = []
        missing_ocr_metadata: List[str] = []

        ocr_required_keys = {"title", "pdf_url", "source_page", "file_path", "sha256", "downloaded_at"}

        for rec in downloaded_records:
            file_path_str = rec.get("file_path")
            if not file_path_str:
                missing_files.append(rec["pdf_url"])
                continue

            file_path = Path(file_path_str)
            if not file_path.exists():
                missing_files.append(str(file_path))
                continue

            file_size = file_path.stat().st_size
            if file_size == 0:
                zero_byte_files.append(str(file_path))
                continue

            # Verify magic bytes
            try:
                with open(file_path, "rb") as f:
                    header = f.read(5)
                    if not header.startswith(b"%PDF-"):
                        invalid_pdf_headers.append(str(file_path))
            except Exception:
                invalid_pdf_headers.append(str(file_path))

            # Verify SHA256 checksum
            stored_hash = rec.get("file_hash")
            if stored_hash:
                actual_hash = compute_file_sha256(file_path)
                if actual_hash != stored_hash:
                    hash_mismatches.append(str(file_path))

            # Verify OCR metadata file
            meta_path = self.metadata_dir / f"{file_path.stem}.json"
            if not meta_path.exists():
                missing_ocr_metadata.append(str(meta_path))
            else:
                try:
                    with open(meta_path, "r", encoding="utf-8") as mf:
                        m_data = json.load(mf)
                        if not ocr_required_keys.issubset(m_data.keys()):
                            missing_ocr_metadata.append(str(meta_path))
                except Exception:
                    missing_ocr_metadata.append(str(meta_path))

        total_downloaded = len(downloaded_records)
        total_attempted = total_downloaded + len(failed_records)
        success_rate = (total_downloaded / total_attempted * 100.0) if total_attempted > 0 else 0.0

        all_valid = (
            len(missing_files) == 0
            and len(zero_byte_files) == 0
            and len(hash_mismatches) == 0
            and len(invalid_pdf_headers) == 0
            and len(missing_ocr_metadata) == 0
        )

        return {
            "total_records_in_db": total_in_db,
            "status_downloaded": total_downloaded,
            "status_failed": len(failed_records),
            "status_discovered_only": len(discovered_only),
            "download_success_rate_pct": round(success_rate, 2),
            "missing_files_count": len(missing_files),
            "missing_files": missing_files[:10],
            "zero_byte_files_count": len(zero_byte_files),
            "zero_byte_files": zero_byte_files[:10],
            "hash_mismatches_count": len(hash_mismatches),
            "invalid_pdf_headers_count": len(invalid_pdf_headers),
            "missing_ocr_metadata_count": len(missing_ocr_metadata),
            "is_valid": all_valid,
        }

    def generate_report(
        self,
        discovery_validation: Optional[Dict[str, Any]] = None,
        download_validation: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Formats validation results into a clean, human-readable report.
        """
        if download_validation is None:
            download_validation = self.validate_downloads()

        report_lines = [
            "============================================================",
            "           NITA NOTICERAG VALIDATION REPORT                 ",
            "============================================================",
        ]

        if discovery_validation:
            report_lines.extend(
                [
                    "--- Discovery Validation ---",
                    f"Total Discovered:     {discovery_validation.get('total_discovered', 0)}",
                    f"Unique URLs:          {discovery_validation.get('unique_urls', 0)}",
                    f"Duplicate Count:      {discovery_validation.get('duplicate_count', 0)}",
                    f"Invalid URLs Count:   {discovery_validation.get('invalid_urls_count', 0)}",
                    f"Discovery Valid:      {'PASSED' if discovery_validation.get('is_valid') else 'FAILED'}",
                    "",
                ]
            )

        report_lines.extend(
            [
                "--- Download & Storage Validation ---",
                f"Total Records in DB:  {download_validation.get('total_records_in_db', 0)}",
                f"Status Downloaded:    {download_validation.get('status_downloaded', 0)}",
                f"Status Failed:        {download_validation.get('status_failed', 0)}",
                f"Status Discovered:    {download_validation.get('status_discovered_only', 0)}",
                f"Success Rate:         {download_validation.get('download_success_rate_pct', 0.0)}%",
                f"Missing Files:        {download_validation.get('missing_files_count', 0)}",
                f"Zero-Byte Files:      {download_validation.get('zero_byte_files_count', 0)}",
                f"Invalid PDF Headers:  {download_validation.get('invalid_pdf_headers_count', 0)}",
                f"Hash Mismatches:      {download_validation.get('hash_mismatches_count', 0)}",
                f"Missing OCR Metadata: {download_validation.get('missing_ocr_metadata_count', 0)}",
                f"Storage Valid:        {'PASSED' if download_validation.get('is_valid') else 'FAILED'}",
                "============================================================",
            ]
        )

        return "\n".join(report_lines)
