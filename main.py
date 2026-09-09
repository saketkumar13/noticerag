"""
Root bridge entry point for NoticeRAG.
Allows running `python main.py` directly from the workspace root.
"""

import sys
from pathlib import Path

# Ensure noticerag package directory is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from noticerag.main import main

if __name__ == "__main__":
    main()
