"""
PDF Downloader Module for NITA NoticeRAG.
Handles streaming downloads, ASP.NET/Azure Blob unwrapping, safe filename generation,
SHA256 checksum calculation, verification, and OCR metadata JSON generation.
"""

import hashlib
import json
import logging
import re
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import requests
import urllib3
from scrapling.parser import Selector

from noticerag.config import (
    AZURE_BLOB_BASE,
    DEFAULT_HEADERS,
    DOWNLOAD_TIMEOUT_SECONDS,
    MAX_RETRIES,
    METADATA_DIR,
    PDF_DIR,
    RETRY_BACKOFF_FACTOR,
)
from noticerag.crawler.database import DatabaseManager

# Suppress insecure request warnings if server has TLS intermediate chain issues
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger("noticerag.downloader")


def get_utc_now_iso() -> str:
    """Returns current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def sanitize_filename(name: str, max_length: int = 150) -> str:
    """
    Sanitizes string into a valid and safe filename for Windows and POSIX.
    """
    # Replace illegal filesystem chars <>:"/\|?* with underscore
    clean = re.sub(r'[\\/*?:"<>|]', "_", name)
    # Remove control characters and strip spaces/periods
    clean = re.sub(r"[\x00-\x1f\x7f]", "", clean).strip(". ")
    # Collapse multiple underscores
    clean = re.sub(r"_+", "_", clean)
    if not clean:
        clean = "document"

    if len(clean) > max_length:
        clean = clean[:max_length].rstrip("_")

    return clean


def compute_sha256(file_path: Path) -> str:
    """Calculates SHA256 checksum of a file in 64KB blocks."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


class PDFDownloader:
    """
    Resilient PDF Downloader with ASP.NET unwrapping and OCR metadata preparation.
    """

    def __init__(
        self,
        db: DatabaseManager,
        pdf_dir: Path = PDF_DIR,
        metadata_dir: Path = METADATA_DIR,
        timeout: int = DOWNLOAD_TIMEOUT_SECONDS,
    ):
        self.db = db
        self.pdf_dir = Path(pdf_dir)
        self.metadata_dir = Path(metadata_dir)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(DEFAULT_HEADERS)

    def _generate_target_path(self, title: str, pdf_url: str) -> Path:
        """
        Creates a distinct, collision-free file path for saving the PDF.
        """
        parsed_url = urllib.parse.urlparse(pdf_url)
        url_filename = Path(parsed_url.path).name

        # Create base stem from url filename or title
        base_name = url_filename if url_filename.lower().endswith(".pdf") else f"{title}.pdf"
        stem = Path(base_name).stem
        safe_stem = sanitize_filename(stem)

        # Append short URL hash to guarantee uniqueness and avoid collisions
        url_hash = hashlib.md5(pdf_url.encode("utf-8")).hexdigest()[:8]
        file_name = f"{safe_stem}_{url_hash}.pdf"
        return self.pdf_dir / file_name

    def _extract_blob_url_from_html(self, html_text: str, original_url: str) -> Optional[str]:
        """
        NITA ASP.NET notice endpoints often return an HTML preview page embedding
        the genuine PDF via an <embed id="fileViewer" src="..."> or <iframe>.
        This parses the embed src or falls back to direct Azure Blob storage.
        """
        try:
            selector = Selector(html_text)
            # Check embed or iframe
            embed_elements = selector.css("embed#fileViewer, iframe#fileViewer, embed, iframe, object")
            for elem in embed_elements:
                src = elem.attrib.get("src") or elem.attrib.get("data")
                if src and ".pdf" in src.lower():
                    logger.debug("Found embedded PDF URL: %s", src)
                    return src
        except Exception as exc:
            logger.debug("Error parsing HTML wrapper: %s", exc)

        # Fallback: extract the PDF filename and check Azure Blob Storage directly
        parsed = urllib.parse.urlparse(original_url)
        pdf_name = Path(parsed.path).name
        if pdf_name and pdf_name.lower().endswith(".pdf"):
            direct_blob = urllib.parse.urljoin(AZURE_BLOB_BASE, pdf_name)
            logger.debug("Constructed fallback Azure blob URL: %s", direct_blob)
            return direct_blob

        return None

    def _download_stream(self, url: str) -> Tuple[bool, bytes, str]:
        """
        Downloads URL content, unwrapping ASP.NET HTML viewer if necessary.
        Returns (is_pdf, content_bytes, effective_url).
        """
        response = self.session.get(url, timeout=self.timeout, verify=False, stream=True)
        response.raise_for_status()

        # Read initial chunk to inspect magic bytes
        initial_chunk = response.raw.read(1024)
        if initial_chunk.startswith(b"%PDF-"):
            # Direct binary PDF! Read remainder
            remaining = response.raw.read()
            return True, initial_chunk + remaining, url

        # Content is likely HTML wrapper or preview
        rest_body = response.raw.read()
        full_body = initial_chunk + rest_body

        try:
            html_text = full_body.decode("utf-8", errors="replace")
        except Exception:
            html_text = ""

        # Attempt to unwrap embed/iframe/blob
        real_pdf_url = self._extract_blob_url_from_html(html_text, url)
        if real_pdf_url and real_pdf_url != url:
            logger.info("Unwrapping ASP.NET viewer to real blob: %s", real_pdf_url)
            blob_resp = self.session.get(real_pdf_url, timeout=self.timeout, verify=False)
            blob_resp.raise_for_status()
            if blob_resp.content.startswith(b"%PDF-"):
                return True, blob_resp.content, real_pdf_url

        # As last fallback, test if full body contains PDF stream anywhere
        if b"%PDF-" in full_body:
            pdf_start = full_body.find(b"%PDF-")
            return True, full_body[pdf_start:], url

        return False, full_body, url

    def download_pdf(self, pdf_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Downloads a single PDF record with retry logic, SHA256 validation,
        and OCR metadata generation.
        """
        pdf_url = pdf_record["pdf_url"]
        title = pdf_record.get("title", "")
        source_page = pdf_record.get("source_page", "")

        target_file = self._generate_target_path(title, pdf_url)

        # Check if already downloaded on disk and DB
        if target_file.exists() and target_file.stat().st_size > 0:
            if pdf_record.get("status") == "downloaded" and pdf_record.get("file_hash"):
                logger.info("PDF already downloaded and verified: %s", target_file.name)
                return self._create_ocr_metadata(
                    title=title,
                    pdf_url=pdf_url,
                    source_page=source_page,
                    file_path=str(target_file.resolve()),
                    sha256=pdf_record["file_hash"],
                    downloaded_at=pdf_record.get("downloaded_at") or get_utc_now_iso(),
                )

        logger.info("Downloading PDF: %s -> %s", pdf_url, target_file.name)

        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                is_pdf, content, effective_url = self._download_stream(pdf_url)
                if not is_pdf:
                    raise ValueError(f"Fetched content is not a valid PDF for {pdf_url}")

                # Write to disk safely
                temp_file = target_file.with_suffix(".tmp")
                with open(temp_file, "wb") as f:
                    f.write(content)

                # Validate downloaded file
                if not temp_file.exists() or temp_file.stat().st_size == 0:
                    temp_file.unlink(missing_ok=True)
                    raise ValueError("Downloaded file is empty")

                # Rename temp file to target
                temp_file.replace(target_file)

                # Compute checksum
                file_hash = compute_sha256(target_file)
                downloaded_at = get_utc_now_iso()

                # Verify file size > 0 and hash generated
                if not file_hash or target_file.stat().st_size == 0:
                    raise ValueError("Verification failed: empty hash or zero byte file")

                # Update database
                self.db.update_pdf_downloaded(
                    pdf_url=pdf_url,
                    file_path=str(target_file.resolve()),
                    file_hash=file_hash,
                    downloaded_at=downloaded_at,
                )

                # Generate and store OCR preparation metadata
                ocr_metadata = self._create_ocr_metadata(
                    title=title,
                    pdf_url=pdf_url,
                    source_page=source_page,
                    file_path=str(target_file.resolve()),
                    sha256=file_hash,
                    downloaded_at=downloaded_at,
                )

                logger.info(
                    "Successfully downloaded [%s] (%d bytes, sha256=%s...)",
                    target_file.name,
                    target_file.stat().st_size,
                    file_hash[:8],
                )
                return ocr_metadata

            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Download attempt %d/%d failed for %s: %s",
                    attempt,
                    MAX_RETRIES,
                    pdf_url,
                    exc,
                )
                if attempt < MAX_RETRIES:
                    time.sleep(RETRY_BACKOFF_FACTOR * attempt)

        # All retries exhausted
        logger.error("All download attempts failed for %s: %s", pdf_url, last_error)
        self.db.update_pdf_failed(pdf_url)
        return None

    def _create_ocr_metadata(
        self,
        title: str,
        pdf_url: str,
        source_page: str,
        file_path: str,
        sha256: str,
        downloaded_at: str,
    ) -> Dict[str, Any]:
        """
        Creates and stores the required OCR preparation metadata JSON file.
        Matches exact schema:
        {
          "title": "",
          "pdf_url": "",
          "source_page": "",
          "file_path": "",
          "sha256": "",
          "downloaded_at": ""
        }
        """
        metadata = {
            "title": title,
            "pdf_url": pdf_url,
            "source_page": source_page,
            "file_path": file_path,
            "sha256": sha256,
            "downloaded_at": downloaded_at,
        }

        # Save individual JSON metadata file matching the PDF stem
        pdf_stem = Path(file_path).stem
        json_path = self.metadata_dir / f"{pdf_stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return metadata
