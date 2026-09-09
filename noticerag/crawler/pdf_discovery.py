"""
PDF Discovery Module using Scrapling.
Extracts PDF links, titles, source pages, and notice dates from NITA notice board
and reachable domain pages.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

from scrapling.fetchers import Fetcher
from scrapling.parser import Selector

from noticerag.config import (
    DEFAULT_HEADERS,
    MAX_DOMAIN_PAGES_TO_DISCOVER,
    NITA_BASE_URL,
    NOTICE_BOARD_URL,
    REQUEST_TIMEOUT_SECONDS,
)

logger = logging.getLogger("noticerag.pdf_discovery")


def normalize_url(base_url: str, link: str) -> Optional[str]:
    """
    Normalizes a link relative to base_url.
    Returns None if the link is invalid, javascript, or external/malformed.
    """
    if not link:
        return None

    link = link.strip()
    if link.startswith(("javascript:", "mailto:", "tel:", "#")):
        return None

    # Resolve relative URL
    absolute_url = urljoin(base_url, link)
    parsed = urlparse(absolute_url)

    # Basic validity check
    if not (parsed.scheme in ("http", "https") and parsed.netloc):
        return None

    # Remove URL fragment
    normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
    if parsed.query:
        normalized += f"?{parsed.query}"

    return normalized


def parse_date_string(date_str: str) -> Optional[datetime]:
    """
    Tries to parse date string in common Indian / academic formats:
    - DD/MM/YYYY
    - DD-MM-YYYY
    - YYYY-MM-DD
    """
    if not date_str:
        return None

    cleaned = date_str.strip()
    for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y", "%d-%m-%y"):
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue

    return None


def extract_date_from_filename_or_text(text: str) -> Optional[datetime]:
    """
    Attempts to extract DD-MM-YYYY or DD_MM_YYYY pattern from filename or title.
    """
    if not text:
        return None

    match = re.search(r"(\d{2})[-_](\d{2})[-_](\d{4})", text)
    if match:
        day, month, year = match.groups()
        try:
            return datetime(int(year), int(month), int(day))
        except ValueError:
            pass

    return None


class PDFDiscovery:
    """
    Discovers PDFs from NITA notice board and reachable domain pages using Scrapling.
    """

    def __init__(
        self,
        notice_board_url: str = NOTICE_BOARD_URL,
        base_url: str = NITA_BASE_URL,
        timeout: int = REQUEST_TIMEOUT_SECONDS,
    ):
        self.notice_board_url = notice_board_url
        self.base_url = base_url
        self.timeout = timeout

    def fetch_page_selector(self, url: str) -> Optional[Selector]:
        """
        Fetches a web page using Scrapling Fetcher and returns a Selector.
        """
        try:
            logger.info("Fetching %s with Scrapling Fetcher...", url)
            response = Fetcher.get(
                url,
                headers=DEFAULT_HEADERS,
                timeout=self.timeout,
            )
            if response.status == 200:
                return response
            else:
                logger.warning("Failed to fetch %s, status code: %s", url, response.status)
                return None
        except Exception as exc:
            logger.error("Scrapling Fetcher error on %s: %s", url, exc)
            return None

    def discover_from_notice_board(self) -> List[Dict[str, Any]]:
        """
        Parses the primary ASP.NET notice board table.
        Columns:
          [0] Sr. No.
          [1] Description (HTML with <a> tag to PDF)
          [2] Title
          [3] Uploaded Date (DD/MM/YYYY)
        """
        selector = self.fetch_page_selector(self.notice_board_url)
        if not selector:
            logger.error("Could not load notice board at %s", self.notice_board_url)
            return []

        rows = selector.css("table tr")
        logger.info("Found %d table rows on notice board.", len(rows))

        discovered: List[Dict[str, Any]] = []
        seen_urls: Set[str] = set()

        notice_index = 0
        for row in rows:
            tds = row.css("td")
            if len(tds) < 3:
                # Header row or irrelevant tr
                continue

            # Look for <a> tags linking to pdf in column 1 or anywhere in the row
            anchors = row.css("a")
            pdf_href = None
            link_text = ""

            for a in anchors:
                href = a.attrib.get("href", "")
                if ".pdf" in href.lower():
                    pdf_href = href
                    link_text = a.text.strip()
                    break

            if not pdf_href:
                continue

            # Normalize PDF URL
            full_pdf_url = normalize_url(self.notice_board_url, pdf_href)
            if not full_pdf_url or full_pdf_url in seen_urls:
                continue

            # Extract Title
            # Column 2 usually contains title span, column 1 has description
            title_text = ""
            if len(tds) >= 3:
                title_text = tds[2].text.strip()
            if not title_text and len(tds) >= 2:
                title_text = tds[1].text.strip()
            if not title_text:
                title_text = link_text or full_pdf_url.split("/")[-1].replace(".pdf", "")

            # Extract Date
            uploaded_date_str = ""
            if len(tds) >= 4:
                uploaded_date_str = tds[3].text.strip()

            parsed_date = parse_date_string(uploaded_date_str)
            if not parsed_date:
                # Fallback to filename/title regex date
                parsed_date = extract_date_from_filename_or_text(full_pdf_url)
            if not parsed_date:
                parsed_date = extract_date_from_filename_or_text(title_text)

            notice_index += 1
            seen_urls.add(full_pdf_url)

            discovered.append(
                {
                    "title": title_text,
                    "pdf_url": full_pdf_url,
                    "source_page": self.notice_board_url,
                    "notice_date": parsed_date.isoformat() if parsed_date else None,
                    "notice_date_raw": uploaded_date_str,
                    "notice_index": notice_index,
                }
            )

        logger.info("Successfully discovered %d unique PDFs from notice board.", len(discovered))
        return discovered

    def discover_from_domain_pages(
        self,
        max_pages: int = MAX_DOMAIN_PAGES_TO_DISCOVER,
        existing_urls: Optional[Set[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Discovers additional PDFs from reachable pages within nita.ac.in.
        """
        if existing_urls is None:
            existing_urls = set()

        seen_urls = set(existing_urls)
        discovered: List[Dict[str, Any]] = []

        # Fetch homepage first to find navigable section links
        home_sel = self.fetch_page_selector(self.base_url)
        if not home_sel:
            return discovered

        candidate_pages: List[str] = []
        base_netloc = urlparse(self.base_url).netloc.lower()

        for a in home_sel.css("a"):
            href = a.attrib.get("href", "")
            norm_link = normalize_url(self.base_url, href)
            if not norm_link:
                continue

            parsed = urlparse(norm_link)
            if base_netloc in parsed.netloc.lower():
                if norm_link.lower().endswith(".pdf"):
                    if norm_link not in seen_urls:
                        seen_urls.add(norm_link)
                        discovered.append(
                            {
                                "title": a.text.strip() or norm_link.split("/")[-1],
                                "pdf_url": norm_link,
                                "source_page": self.base_url,
                                "notice_date": None,
                                "notice_date_raw": "",
                                "notice_index": 999999,
                            }
                        )
                elif norm_link not in candidate_pages and norm_link != self.notice_board_url:
                    candidate_pages.append(norm_link)

        # Inspect subset of candidate domain pages
        pages_checked = 0
        for page_url in candidate_pages:
            if pages_checked >= max_pages:
                break

            pages_checked += 1
            sel = self.fetch_page_selector(page_url)
            if not sel:
                continue

            for a in sel.css("a"):
                href = a.attrib.get("href", "")
                if ".pdf" in href.lower():
                    pdf_url = normalize_url(page_url, href)
                    if pdf_url and pdf_url not in seen_urls:
                        seen_urls.add(pdf_url)
                        title = a.text.strip() or pdf_url.split("/")[-1]
                        parsed_date = extract_date_from_filename_or_text(pdf_url)
                        discovered.append(
                            {
                                "title": title,
                                "pdf_url": pdf_url,
                                "source_page": page_url,
                                "notice_date": parsed_date.isoformat() if parsed_date else None,
                                "notice_date_raw": "",
                                "notice_index": 999999 + pages_checked,
                            }
                        )

        logger.info("Discovered %d additional PDFs from %d domain pages.", len(discovered), pages_checked)
        return discovered

    def discover_all(self, crawl_domain: bool = True) -> List[Dict[str, Any]]:
        """
        Discovers all PDFs starting from the notice board, then domain pages.
        Returns deduplicated list sorted by newest date first (descending),
        falling back to notice board row index.
        """
        notice_pdfs = self.discover_from_notice_board()
        seen_urls = {p["pdf_url"] for p in notice_pdfs}

        domain_pdfs: List[Dict[str, Any]] = []
        if crawl_domain:
            domain_pdfs = self.discover_from_domain_pages(existing_urls=seen_urls)

        all_pdfs = notice_pdfs + domain_pdfs

        # Sort key:
        # 1. Has date: newest date first (e.g. 2026-09-03 > 2026-08-20)
        # 2. No date: notice_index ascending (Row 1 before Row 2)
        def sort_key(item: Dict[str, Any]) -> Tuple[int, str, int]:
            date_str = item.get("notice_date")
            index = item.get("notice_index", 999999)
            if date_str:
                # 0 comes before 1, so dated items come first, sorted inversely by date
                return (0, date_str, -index)
            else:
                return (1, "", -index)

        # For descending date sort:
        all_pdfs.sort(
            key=lambda x: (
                x.get("notice_date") or "1970-01-01",
                -x.get("notice_index", 999999),
            ),
            reverse=True,
        )

        logger.info("Total sorted PDF candidates discovered: %d", len(all_pdfs))
        return all_pdfs
