"""
Metadata Extraction Module.
Extracts Title, Department, and Issuer from document text and structural headers.
"""

import re
from typing import Any, Dict, List, Optional, Tuple


class MetadataExtractor:
    """
    Extracts semantic entities: Title, Department, and Issuer with confidence scores.
    """

    # Department patterns and canonical names
    DEPARTMENT_MAPPINGS = [
        (re.compile(r"hostel\s+(?:affairs|administration|committee)|chief\s+warden", re.I), "Hostel Administration", 0.95),
        (re.compile(r"dean\s*(?:of\s+)?academic|office\s+of\s+the\s+dean\s+academic|nitadeanacademic", re.I), "Dean (Academic)", 0.95),
        (re.compile(r"dean\s*(?:of\s+)?students['’]?\s*welfare|office\s+of\s+the\s+dean\s*sw|deansw", re.I), "Dean (Students' Welfare)", 0.95),
        (re.compile(r"training\s*&\s*placement|t&p\s+cell|placement\s+cell", re.I), "Training & Placement Cell", 0.95),
        (re.compile(r"registrar\s+office|office\s+of\s+the\s+registrar|f\.nita\.3\(6-gen\)", re.I), "Registrar Office", 0.92),
        (re.compile(r"estate\s*(?:civil|electrical|section)|nita\/estate", re.I), "Estate Section", 0.92),
        (re.compile(r"sports\s+section|annual\s*sports|gymkhana", re.I), "Sports & Gymkhana Section", 0.92),
        (re.compile(r"computer\s+science|cse\s+department", re.I), "Department of Computer Science & Engineering", 0.90),
        (re.compile(r"mechanical\s+engineering", re.I), "Department of Mechanical Engineering", 0.90),
        (re.compile(r"civil\s+engineering", re.I), "Department of Civil Engineering", 0.90),
        (re.compile(r"electrical\s+engineering", re.I), "Department of Electrical Engineering", 0.90),
        (re.compile(r"electronics\s*&\s*communication|ece\b", re.I), "Department of Electronics & Communication Engineering", 0.90),
        (re.compile(r"bio[\-\s]?engineering|icmr", re.I), "Department of Bio-Engineering", 0.90),
        (re.compile(r"management,\s*humanities|mhss|mba\b", re.I), "Department of Management Studies", 0.88),
    ]

    # Issuer designations and canonical names
    ISSUER_MAPPINGS = [
        (re.compile(r"(?:chief\s+warden|warden)", re.I), "Chief Warden", 0.94),
        (re.compile(r"dean\s*(?:\(acad(?:emic)?\)|\s+academic)", re.I), "Dean (Academic)", 0.95),
        (re.compile(r"dean\s*(?:\(sw\)|\s+students['’]?\s*welfare)", re.I), "Dean (Students' Welfare)", 0.95),
        (re.compile(r"(?:associate\s+dean\s*\(acad\)|\bassoc\.\s*dean)", re.I), "Associate Dean (Academic)", 0.92),
        (re.compile(r"\bregistrar\b", re.I), "Registrar", 0.92),
        (re.compile(r"\bdirector\b", re.I), "Director", 0.90),
        (re.compile(r"training\s*&\s*placement\s+officer|tpo\b", re.I), "Training & Placement Officer", 0.92),
        (re.compile(r"faculty\s+coordinator|coordinator", re.I), "Faculty Coordinator", 0.85),
        (re.compile(r"head\s+of\s+department|\bhod\b", re.I), "Head of Department", 0.85),
        (re.compile(r"assistant\s+registrar|\basst\.\s*registrar", re.I), "Assistant Registrar", 0.88),
    ]

    # Subject line extraction pattern
    SUB_PATTERNS = [
        re.compile(r"(?:Sub|Subject)\s*[:\-\.]\s*([^\n\r]+(?:\n[^\n\r]+)?)", re.I),
        re.compile(r"\bNOTIFICATION\s*[:\-\.]?\s*([^\n\r]+(?:\n[^\n\r]+)?)", re.I),
        re.compile(r"\bNOTICE\s*[:\-\.]?\s*([^\n\r]+(?:\n[^\n\r]+)?)", re.I),
        re.compile(r"\bCIRCULAR\s*[:\-\.]?\s*([^\n\r]+(?:\n[^\n\r]+)?)", re.I),
        re.compile(r"\bMEMO\s*[:\-\.]?\s*([^\n\r]+(?:\n[^\n\r]+)?)", re.I),
    ]

    @classmethod
    def extract_title(cls, text: str, fallback_title: str = "") -> Tuple[str, float]:
        """
        Extracts document title using multiple hierarchical strategies.
        Returns (title, confidence).
        """
        if not text:
            clean_fb = cls._clean_fallback_title(fallback_title)
            return clean_fb, 0.60

        header_text = text[:1800]

        # Strategy 1: Explicit Sub / Subject line
        for pattern in cls.SUB_PATTERNS:
            match = pattern.search(header_text)
            if match:
                raw_sub = match.group(1).strip()
                # Clean up trailing punctuation or next section tokens
                clean_sub = re.split(r"\b(?:In pursuance|Based on|This is to|Applications are|The |1\.)\b", raw_sub)[0]
                clean_sub = clean_sub.strip(" :-\n\t.,")
                if len(clean_sub) >= 15:
                    return clean_sub, 0.95

        # Strategy 2: Pre-heading block (e.g. lines right after NITA header)
        lines = [line.strip() for line in header_text.splitlines() if line.strip()]
        for idx, line in enumerate(lines[:12]):
            if any(term in line.lower() for term in ["admission notification", "quotation notice", "tender notice"]):
                return line, 0.90
            if line.upper() in ["NOTIFICATION", "NOTICE", "CIRCULAR", "MEMORANDUM", "ADVERTISEMENT"]:
                # The next non-empty line is often the actual title
                if idx + 1 < len(lines):
                    next_line = lines[idx + 1]
                    if len(next_line) >= 15 and not next_line.lower().startswith("date"):
                        return next_line, 0.88

        # Strategy 3: Clean fallback from filename or DB title
        if fallback_title:
            clean_fb = cls._clean_fallback_title(fallback_title)
            if len(clean_fb) >= 10:
                return clean_fb, 0.75

        # Strategy 4: Use first substantial sentence
        for line in lines[:8]:
            if len(line) > 25 and not line.lower().startswith(("national institute", "agartala", "barjala", "f.nita", "dated")):
                return line[:150], 0.65

        return "University Notice", 0.40

    @classmethod
    def _clean_fallback_title(cls, title: str) -> str:
        """Cleans filename stems into human-readable titles."""
        if not title:
            return ""
        # Remove prefixes like Notice_03-09-2026_ or suffixes like _admin
        cleaned = re.sub(r"^Notice_\d{2}[-_]\d{2}[-_]\d{4}_?", "", title, flags=re.I)
        cleaned = re.sub(r"_admin.*$", "", cleaned, flags=re.I)
        # Replace hyphens and underscores with spaces
        cleaned = re.sub(r"[-_]+", " ", cleaned).strip()
        return cleaned

    @classmethod
    def extract_department(cls, text: str, title: str = "") -> Tuple[str, float]:
        """
        Extracts issuing department or administrative unit.
        Returns (department, confidence).
        """
        combined = f"{title}\n{text[:2500]}".lower()

        for pattern, dept_name, conf in cls.DEPARTMENT_MAPPINGS:
            if pattern.search(combined):
                return dept_name, conf

        return "General Administration", 0.50

    @classmethod
    def extract_issuer(cls, text: str, department: str = "") -> Tuple[str, float]:
        """
        Extracts the authority / signatory of the document.
        Returns (issuer, confidence).
        """
        # Look both in footer (where signatures appear) and in header
        lines = text.splitlines()
        footer_text = "\n".join(lines[-15:]) if len(lines) >= 15 else text
        header_text = text[:1500]

        # Check footer first (direct signatory)
        for pattern, issuer_name, conf in cls.ISSUER_MAPPINGS:
            if pattern.search(footer_text):
                return issuer_name, conf

        # Check header
        for pattern, issuer_name, conf in cls.ISSUER_MAPPINGS:
            if pattern.search(header_text):
                return issuer_name, max(0.50, conf - 0.10)

        # Fallback based on department
        if "Academic" in department:
            return "Dean (Academic)", 0.60
        if "Students' Welfare" in department or "Hostel" in department:
            return "Dean (Students' Welfare)", 0.60
        if "Registrar" in department:
            return "Registrar", 0.60

        return "Competent Authority", 0.45
