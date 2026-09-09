"""
PDF Processor Module.
Coordinates page-by-page hybrid extraction: uses digital text when sufficient,
and automatically rasterizes and triggers OCR for scanned or mixed pages.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pypdfium2

from noticerag.config import OCR_DPI, OCR_MIN_CHARS_PER_PAGE
from noticerag.ocr.cleaner import TextCleaner
from noticerag.ocr.ocr_engine import OCREngine
from noticerag.ocr.text_extractor import PDFTextExtractor

logger = logging.getLogger("noticerag.ocr.pdf_processor")


class PDFProcessor:
    """
    Hybrid PDF processor capable of handling digital PDFs, scanned notices,
    and mixed documents.
    """

    def __init__(
        self,
        ocr_engine: Optional[OCREngine] = None,
        min_chars_per_page: int = OCR_MIN_CHARS_PER_PAGE,
        dpi: int = OCR_DPI,
    ):
        self.ocr_engine = ocr_engine or OCREngine()
        self.text_extractor = PDFTextExtractor(min_chars_per_page=min_chars_per_page)
        self.dpi = dpi

    def _render_page_image(self, pdf_path: Path, page_index: int):
        """
        Renders a specific page of a PDF as a PIL Image at the configured DPI.
        """
        try:
            pdf = pypdfium2.PdfDocument(str(pdf_path))
            page = pdf[page_index]
            # scale: 72 DPI is 1.0, so 200 DPI scale is 200 / 72 ≈ 2.777
            scale = self.dpi / 72.0
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            pdf.close()
            return pil_image
        except Exception as exc:
            logger.error("Failed to render page %d of %s with pypdfium2: %s", page_index + 1, pdf_path.name, exc)
            return None

    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """
        Processes a single PDF:
        - Attempts direct text extraction per page.
        - Falls back to OCR for scanned or sparse pages.
        - Cleans text and aggregates document statistics.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            return {
                "success": False,
                "error": f"File does not exist: {pdf_path}",
                "text": "",
                "total_pages": 0,
                "text_length": 0,
                "word_count": 0,
                "ocr_used": False,
                "ocr_engine": "None",
                "pages_digital_count": 0,
                "pages_ocr_count": 0,
                "page_details": [],
            }

        if pdf_path.stat().st_size == 0:
            return {
                "success": False,
                "error": f"File is empty (0 bytes): {pdf_path}",
                "text": "",
                "total_pages": 0,
                "text_length": 0,
                "word_count": 0,
                "ocr_used": False,
                "ocr_engine": "None",
                "pages_digital_count": 0,
                "pages_ocr_count": 0,
                "page_details": [],
            }

        try:
            all_digital, digital_pages, total_pages = self.text_extractor.extract_digital_pages(pdf_path)
        except Exception as exc:
            logger.error("Could not read digital pages for %s: %s", pdf_path.name, exc)
            digital_pages = []
            total_pages = 0

        if total_pages == 0:
            # Try to get total pages using pypdfium2
            try:
                pdf_doc = pypdfium2.PdfDocument(str(pdf_path))
                total_pages = len(pdf_doc)
                pdf_doc.close()
                digital_pages = [
                    {"page_number": i + 1, "text": "", "char_count": 0, "word_count": 0, "is_digital": False}
                    for i in range(total_pages)
                ]
            except Exception as exc:
                return {
                    "success": False,
                    "error": f"Corrupted or password-protected PDF: {exc}",
                    "text": "",
                    "total_pages": 0,
                    "text_length": 0,
                    "word_count": 0,
                    "ocr_used": False,
                    "ocr_engine": "None",
                    "pages_digital_count": 0,
                    "pages_ocr_count": 0,
                    "page_details": [],
                }

        page_texts: List[str] = []
        page_details: List[Dict[str, Any]] = []
        ocr_used = False
        engines_used = set()
        pages_digital = 0
        pages_ocr = 0

        for page_idx in range(total_pages):
            page_num = page_idx + 1
            d_info = digital_pages[page_idx] if page_idx < len(digital_pages) else None
            is_digital = d_info["is_digital"] if d_info else False

            if is_digital and d_info:
                # Digital text sufficient!
                raw_page_text = d_info["text"]
                cleaned_page_text = TextCleaner.clean(raw_page_text)
                page_texts.append(cleaned_page_text)
                pages_digital += 1
                page_details.append(
                    {
                        "page_number": page_num,
                        "method": "digital",
                        "chars": len(cleaned_page_text),
                        "words": len(cleaned_page_text.split()),
                    }
                )
            else:
                # Scanned or sparse page: Run OCR
                logger.info("Page %d of %s requires OCR. Rasterizing at %d DPI...", page_num, pdf_path.name, self.dpi)
                page_img = self._render_page_image(pdf_path, page_idx)
                if page_img is not None:
                    ocr_text, engine_name, conf = self.ocr_engine.ocr_image(page_img)
                    cleaned_ocr_text = TextCleaner.clean(ocr_text)
                    page_texts.append(cleaned_ocr_text)
                    ocr_used = True
                    pages_ocr += 1
                    engines_used.add(engine_name)
                    page_details.append(
                        {
                            "page_number": page_num,
                            "method": f"ocr_{engine_name.lower()}",
                            "confidence": round(conf, 2),
                            "chars": len(cleaned_ocr_text),
                            "words": len(cleaned_ocr_text.split()),
                        }
                    )
                else:
                    # Fallback to whatever digital text existed
                    fallback_text = TextCleaner.clean(d_info["text"]) if d_info else ""
                    page_texts.append(fallback_text)
                    page_details.append(
                        {
                            "page_number": page_num,
                            "method": "render_failed_fallback",
                            "chars": len(fallback_text),
                            "words": len(fallback_text.split()),
                        }
                    )

        full_document_text = "\n\n".join(t for t in page_texts if t.strip())
        cleaned_final_text = TextCleaner.clean(full_document_text)

        engine_summary = "+".join(sorted(engines_used)) if engines_used else "None"

        return {
            "success": True,
            "error": None,
            "text": cleaned_final_text,
            "total_pages": total_pages,
            "text_length": len(cleaned_final_text),
            "word_count": len(cleaned_final_text.split()),
            "ocr_used": ocr_used,
            "ocr_engine": engine_summary,
            "pages_digital_count": pages_digital,
            "pages_ocr_count": pages_ocr,
            "page_details": page_details,
        }
