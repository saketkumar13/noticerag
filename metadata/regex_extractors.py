"""
Regex and Rule-based Extractors for Dates, Reference Numbers, and Structured Identifiers.
Normalizes dates to canonical ISO 8601 (YYYY-MM-DD).
"""

import re
from datetime import datetime
from typing import List, Optional, Tuple

MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


class RegexExtractors:
    """
    Regex utilities for extracting publication dates and official dispatch/memo numbers.
    """

    # Memo / File Reference patterns: e.g. F.NITA.3(6-Gen)/2011/..., No.NITA.5/(143-Acad)/...
    MEMO_PATTERNS = [
        re.compile(r"(?:No\.|File No\.|F\.No\.|F\.)\s*([A-Za-z0-9\.\(\)\/\-_]+(?:\/[A-Za-z0-9\.\(\)\-_]+){2,})", re.I),
        re.compile(r"(NITA[\w\.\(\)\/\-_]{8,})", re.I),
    ]

    # Explicit Date prefix patterns (highest priority)
    DATED_PREFIX_PATTERNS = [
        re.compile(r"(?:Dated?|Date)\s*[:\-\.]?\s*([0-3]?\d[\/\-\.][0-1]?\d[\/\-\.][12]\d{3})", re.I),
        re.compile(r"(?:Dated?|Date)\s*[:\-\.]?\s*([0-3]?\d\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[\s,]+[12]\d{3})", re.I),
    ]

    # General Date patterns
    DATE_NUMERIC_PATTERN = re.compile(r"\b([0-3]?\d)[\/\-\.]([0-1]?\d)[\/\-\.]([12]\d{3})\b")
    DATE_TEXTUAL_PATTERN = re.compile(
        r"\b([0-3]?\d)(?:st|nd|rd|th)?\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,\s]+([12]\d{3})\b",
        re.I,
    )

    @classmethod
    def _parse_and_validate(cls, day: int, month: int, year: int) -> Optional[str]:
        """
        Validates day, month, year ranges and returns normalized YYYY-MM-DD.
        Accommodates occasional DD/MM vs MM/DD swaps if day > 12.
        """
        # Academic notice years are typically between 2000 and 2035
        if not (2000 <= year <= 2035):
            return None

        # Try DD/MM/YYYY first (standard in Indian university notices)
        try:
            dt = datetime(year, month, day)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            # If day and month might be inverted (e.g. month was > 12)
            try:
                dt = datetime(year, day, month)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                return None

    @classmethod
    def extract_date(cls, text: str, fallback_filename: str = "") -> Tuple[Optional[str], float]:
        """
        Extracts primary notice publication date and normalizes to YYYY-MM-DD.
        Returns (normalized_date_str, confidence).
        """
        if not text:
            text = ""

        # First 1500 characters usually contain the notice header and publication date
        header_text = text[:1500]

        # 1. Look for explicit "Date:" or "Dated:" prefixes
        for pattern in cls.DATED_PREFIX_PATTERNS:
            match = pattern.search(header_text)
            if match:
                raw_val = match.group(1).strip()
                parsed = cls.normalize_raw_date(raw_val)
                if parsed:
                    return parsed, 0.98

        # 2. Textual month pattern in header: e.g. "4th September 2026" or "15 August 2026"
        tmatch = cls.DATE_TEXTUAL_PATTERN.search(header_text)
        if tmatch:
            day_str, month_str, year_str = tmatch.groups()
            month_num = MONTH_MAP.get(month_str.lower()[:3])
            if month_num:
                parsed = cls._parse_and_validate(int(day_str), month_num, int(year_str))
                if parsed:
                    return parsed, 0.95

        # 3. Numeric pattern in header (DD/MM/YYYY or DD-MM-YYYY)
        for dmatch in cls.DATE_NUMERIC_PATTERN.finditer(header_text):
            day_str, month_str, year_str = dmatch.groups()
            parsed = cls._parse_and_validate(int(day_str), int(month_str), int(year_str))
            if parsed:
                return parsed, 0.90

        # 4. Fallback: Search in entire document
        tmatch_full = cls.DATE_TEXTUAL_PATTERN.search(text)
        if tmatch_full:
            day_str, month_str, year_str = tmatch_full.groups()
            month_num = MONTH_MAP.get(month_str.lower()[:3])
            if month_num:
                parsed = cls._parse_and_validate(int(day_str), month_num, int(year_str))
                if parsed:
                    return parsed, 0.85

        # 5. Fallback: Regex extraction from document filename/ID
        # e.g. Notice_03-09-2026_Holiday-Janmashtami_admin...
        if fallback_filename:
            fn_match = re.search(r"(\d{2})[-_](\d{2})[-_](\d{4})", fallback_filename)
            if fn_match:
                d, m, y = fn_match.groups()
                parsed = cls._parse_and_validate(int(d), int(m), int(y))
                if parsed:
                    return parsed, 0.80

        return None, 0.0

    @classmethod
    def normalize_raw_date(cls, raw: str) -> Optional[str]:
        """
        Normalizes varying date strings (e.g. 03/09/2026, 3-9-2026, 4th Sept 2026) to YYYY-MM-DD.
        """
        raw = raw.strip()
        # Clean OCR artifacts where 'O' was recognized instead of '0'
        raw = re.sub(r"\bO([0-9])", r"0\1", raw)

        # Try standard formats
        for fmt in ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%Y-%m-%d", "%d %B %Y", "%d %b %Y"):
            try:
                dt = datetime.strptime(raw, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue

        # Try regex parsing
        num_match = cls.DATE_NUMERIC_PATTERN.search(raw)
        if num_match:
            d, m, y = num_match.groups()
            return cls._parse_and_validate(int(d), int(m), int(y))

        txt_match = cls.DATE_TEXTUAL_PATTERN.search(raw)
        if txt_match:
            d, m, y = txt_match.groups()
            month_num = MONTH_MAP.get(m.lower()[:3])
            if month_num:
                return cls._parse_and_validate(int(d), month_num, int(y))

        return None

    @classmethod
    def extract_memo_reference(cls, text: str) -> Optional[str]:
        """
        Extracts official NITA notice memo / dispatch reference code if present.
        """
        if not text:
            return None

        header = text[:1500]
        for pattern in cls.MEMO_PATTERNS:
            match = pattern.search(header)
            if match:
                val = match.group(1).strip(" .:,")
                if len(val) >= 8:
                    return val
        return None
