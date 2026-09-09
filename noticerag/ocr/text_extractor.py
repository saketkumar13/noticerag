"""
Text Extractor Module for Digital PDFs.
Extracts native embedded text from PDF pages and assesses whether
the page has sufficient digital content or requires OCR fallback.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pypdf

from noticerag.config import OCR_MIN_CHARS_PER_PAGE

logger = logging.getLogger("noticerag.ocr.text_extractor")


class PDFTextExtractor:
    """
    Extracts native digital text from PDF files using pypdf.
    """

    def __init__(self, min_chars_per_page: int = OCR_MIN_CHARS_PER_PAGE):
        self.min_chars_per_page = min_chars_per_page

    @staticmethod
    def is_text_sufficient(text: str, min_chars: int = OCR_MIN_CHARS_PER_PAGE) -> bool:
        """
        Determines whether extracted text is sufficient to treat the page as digital.
        Checks:
          1. Character count >= min_chars
          2. Printable alphanumeric character density >= 50%
          3. Contains at least a few recognizable words
        """
        if not text:
            return False

        stripped = text.strip()
        if len(stripped) < min_chars:
            return False

        # Check alphanumeric ratio to weed out binary font encoding garbage
        alnum_chars = sum(1 for c in stripped if c.isalnum() or c.isspace())
        ratio = alnum_chars / len(stripped) if stripped else 0
        if ratio < 0.6:
            return False

        words = stripped.split()
        if len(words) < 5:
            return False

        return True

    def extract_digital_pages(self, pdf_path: Path) -> Tuple[bool, List[Dict[str, Any]], int]:
        """
        Extracts digital text page-by-page.
        Returns:
            (is_all_digital, pages_data, total_pages)
            where pages_data is a list of dicts:
            {
                "page_number": int,
                "text": str,
                "char_count": int,
                "word_count": int,
                "is_digital": bool
            }
        """
        pdf_path = Path(pdf_path)
        pages_data: List[Dict[str, Any]] = []

        try:
            reader = pypdf.PdfReader(str(pdf_path))
            total_pages = len(reader.pages)

            all_digital = True
            for idx, page in enumerate(reader.pages, 1):
                try:
                    raw_text = page.extract_text() or ""
                except Exception as exc:
                    logger.warning("Error extracting text from page %d of %s: %s", idx, pdf_path.name, exc)
                    raw_text = ""

                clean_text = raw_text.strip()
                char_count = len(clean_text)
                words = clean_text.split()
                word_count = len(words)
                is_digital = self.is_text_sufficient(clean_text, self.min_chars_per_page)

                if not is_digital:
                    all_digital = False

                pages_data.append(
                    {
                        "page_number": idx,
                        "text": clean_text,
                        "char_count": char_count,
                        "word_count": word_count,
                        "is_digital": is_digital,
                    }
                )

            return all_digital, pages_data, total_pages

        except Exception as exc:
            logger.error("Failed to open PDF %s: %s", pdf_path.name, exc)
            return False, [], 0
