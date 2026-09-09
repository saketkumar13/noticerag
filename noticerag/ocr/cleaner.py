"""
Text Cleaning and Normalization Module.
Prepares extracted and OCR'd notice text for downstream RAG chunking and GraphRAG.
"""

import re
import unicodedata
from typing import Optional


class TextCleaner:
    """
    Cleans, normalizes, and repairs text extracted from administrative notices and scans.
    """

    @staticmethod
    def normalize_unicode(text: str) -> str:
        """Normalizes Unicode characters (NFKC) and strips non-printable control chars."""
        if not text:
            return ""
        # Normalize NFKC (converts ligatures like ﬁ -> fi, ½ -> 1/2)
        norm = unicodedata.normalize("NFKC", text)
        # Replace non-breaking spaces and zero-width spaces
        norm = norm.replace("\u00a0", " ").replace("\u200b", "").replace("\ufeff", "")
        # Remove unprintable control characters except standard whitespace \n, \r, \t
        return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]", "", norm)

    @staticmethod
    def repair_hyphenation(text: str) -> str:
        """
        Repairs line-split words like 'adminis-\\ntration' -> 'administration'.
        Preserves genuine compound words like 'Non-CCMT' or 'B.Sc.-B.Ed.'.
        """
        if not text:
            return ""
        # Matches lowercase letter + hyphen + optional spaces + newline + optional spaces + lowercase letter
        return re.sub(r"([a-zA-Z])-\s*\n\s*([a-zA-Z])", r"\1\2", text)

    @staticmethod
    def clean_ocr_artifacts(text: str) -> str:
        """
        Removes isolated scanner noise (specks, lone punctuation marks on a line).
        """
        if not text:
            return ""

        lines = text.split("\n")
        cleaned_lines = []

        for line in lines:
            stripped = line.strip()
            # If line is only 1-2 punctuation specks (like `.` or `~` or `_`), skip
            if len(stripped) <= 2 and all(not ch.isalnum() for ch in stripped):
                continue
            # Remove repeated dashes / underscores representing dotted scan borders if excessive
            if len(stripped) > 20 and len(set(stripped)) == 1 and stripped[0] in "-_=.~*":
                continue
            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    @staticmethod
    def collapse_whitespace(text: str) -> str:
        """
        Collapses excessive horizontal spaces and empty lines.
        Limits consecutive blank lines to at most two.
        """
        if not text:
            return ""

        # Collapse horizontal tabs and spaces (except newlines)
        text = re.sub(r"[ \t]+", " ", text)
        # Strip trailing whitespace on each line
        lines = [line.strip() for line in text.splitlines()]
        # Collapse multiple empty lines
        joined = "\n".join(lines)
        return re.sub(r"\n{3,}", "\n\n", joined).strip()

    @classmethod
    def clean(cls, text: Optional[str]) -> str:
        """Runs the complete text cleaning pipeline."""
        if not text:
            return ""

        t = cls.normalize_unicode(text)
        t = cls.repair_hyphenation(t)
        t = cls.clean_ocr_artifacts(t)
        t = cls.collapse_whitespace(t)
        return t
