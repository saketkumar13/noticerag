import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
for p in (str(BASE_DIR), str(BASE_DIR / "noticerag"), str(BASE_DIR / "retrieval")):
    if p not in sys.path:
        sys.path.insert(0, p)

from retrieval.verify_reranker import main

if __name__ == "__main__":
    main()
