import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "noticerag"), str(BASE_DIR / "retrieval")):
    if p not in sys.path:
        sys.path.insert(0, p)

from generation.config import GENERATION_LOG_FILE, MODEL_NAME, TOP_K
from generation.rag_pipeline import RAGPipeline

TEST_QUERIES = [
    "holiday notice",
    "hostel notice",
    "placement notice",
    "scholarship",
    "recruitment",
    "tender",
    "academic calendar",
]


def main():
    print("\n============================================================")
    print("      NITA CAMPUS INTELLIGENCE - GENERATION VERIFICATION    ")
    print("============================================================\n")

    print(f"LLM Model:         {MODEL_NAME}")
    print(f"Retrieval Top-K:   {TOP_K}")
    print(f"Log File:          {GENERATION_LOG_FILE}\n")

    pipeline = RAGPipeline(top_k=TOP_K)

    for q in TEST_QUERIES:
        t0 = time.time()
        res = pipeline.answer_question(q)
        latency_ms = (time.time() - t0) * 1000.0
        ans = res["answer"]
        success = not ans.startswith("Error")

        print("============================================================")
        print(f"QUESTION:\n{q}\n")
        print(f"GENERATION TIME:   {latency_ms:.1f} ms")
        print(f"STATUS:            {'SUCCESS' if success else 'FAILURE'}\n")
        print(f"ANSWER:\n{ans}")
        print("============================================================\n")


if __name__ == "__main__":
    main()
