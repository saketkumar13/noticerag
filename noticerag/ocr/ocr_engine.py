"""
OCR Engine Module.
Provides RapidOCR (PaddleOCR-compatible deep learning ONNX models) as primary engine
with Tesseract 5.5.3 as fallback for resilient text extraction on scanned notices,
stamps, signatures, and degraded photocopies.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

try:
    from rapidocr_onnxruntime import RapidOCR
    RAPID_OCR_AVAILABLE = True
except ImportError:
    RAPID_OCR_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from noticerag.config import TESSERACT_CMD

logger = logging.getLogger("noticerag.ocr.engine")

# Configure tesseract executable if present
if TESSERACT_AVAILABLE and os.path.exists(TESSERACT_CMD):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


class OCREngine:
    """
    Robust OCR Engine supporting RapidOCR (PaddleOCR ONNX) and Tesseract.
    """

    def __init__(self, primary_engine: str = "RapidOCR"):
        self.primary_engine = primary_engine
        self._rapid_ocr: Optional[Any] = None

        if primary_engine == "RapidOCR" and RAPID_OCR_AVAILABLE:
            try:
                # Initialize RapidOCR with angle classification enabled
                self._rapid_ocr = RapidOCR()
                logger.info("RapidOCR (PaddleOCR ONNX) initialized successfully.")
            except Exception as exc:
                logger.warning("Failed to initialize RapidOCR: %s. Falling back to Tesseract.", exc)
                self.primary_engine = "Tesseract"
        elif primary_engine == "RapidOCR" and not RAPID_OCR_AVAILABLE:
            logger.warning("rapidocr_onnxruntime is not installed. Falling back to Tesseract.")
            self.primary_engine = "Tesseract"

    @staticmethod
    def preprocess_image(image: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """
        Preprocesses image for OCR: converts to grayscale and normalizes contrast.
        """
        if isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"))
        else:
            img_np = image.copy()

        # If image has 3 channels, convert to BGR for OpenCV
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            # Check if likely RGB
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img_np

        return img_bgr

    def ocr_image(self, image: Union[Image.Image, np.ndarray]) -> Tuple[str, str, float]:
        """
        Performs OCR on an image.
        Returns:
            (extracted_text, engine_used, average_confidence)
        """
        img_preprocessed = self.preprocess_image(image)

        # 1. Attempt RapidOCR if primary
        if self.primary_engine == "RapidOCR" and self._rapid_ocr is not None:
            try:
                result, elapse = self._rapid_ocr(img_preprocessed)
                if result:
                    lines = []
                    scores = []
                    for box, text, score in result:
                        text_str = str(text).strip()
                        if text_str:
                            lines.append(text_str)
                            try:
                                scores.append(float(score))
                            except (ValueError, TypeError):
                                pass

                    avg_conf = sum(scores) / len(scores) if scores else 1.0
                    full_text = "\n".join(lines)
                    logger.debug("RapidOCR extracted %d lines (conf: %.2f)", len(lines), avg_conf)
                    return full_text, "RapidOCR", avg_conf
                else:
                    logger.debug("RapidOCR returned empty result. Trying fallback.")
            except Exception as exc:
                logger.warning("RapidOCR execution failed: %s. Trying Tesseract fallback.", exc)

        # 2. Tesseract Fallback
        if TESSERACT_AVAILABLE and os.path.exists(TESSERACT_CMD):
            try:
                if isinstance(image, np.ndarray):
                    pil_img = Image.fromarray(cv2.cvtColor(img_preprocessed, cv2.COLOR_BGR2RGB))
                else:
                    pil_img = image

                # Run tesseract with psm 1 (automatic page segmentation with OSD)
                text = pytesseract.image_to_string(pil_img, config="--psm 1 --oem 3")
                logger.debug("Tesseract extracted %d characters", len(text))
                return text.strip(), "Tesseract", 0.8
            except Exception as exc:
                logger.error("Tesseract execution failed: %s", exc)

        return "", "None", 0.0
