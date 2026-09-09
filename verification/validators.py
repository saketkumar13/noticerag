"""
Dataset and File Validation Module.
Verifies file existence, readability, non-empty text, and schema compliance
across the extracted text corpus.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("noticerag.verification.validators")


class DatasetValidator:
    """
    Validates physical existence, readability, and text integrity of OCR files.
    """

    @classmethod
    def find_extracted_files(cls, base_dirs: List[Path]) -> List[Path]:
        """
        Discovers extracted JSON files across potential directories.
        """
        discovered = []
        seen = set()

        for bdir in base_dirs:
            p = Path(bdir)
            if p.exists():
                for jf in p.glob("*.json"):
                    if jf.name not in seen:
                        seen.add(jf.name)
                        discovered.append(jf)

        return sorted(discovered)

    @classmethod
    def validate_file(cls, file_path: Path) -> Tuple[bool, Optional[Dict[str, Any]], Optional[str]]:
        """
        Validates individual file:
        - exists
        - readable
        - valid JSON
        - text field exists and is non-empty
        Returns (is_valid, parsed_data, error_message).
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return False, None, f"File does not exist: {file_path}"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            return False, None, f"Could not parse JSON: {exc}"

        text = data.get("text", "")
        if not isinstance(text, str) or not text.strip():
            return False, data, "Extracted text is empty or missing"

        return True, data, None
