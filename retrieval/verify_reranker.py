import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "noticerag"), str(BASE_DIR / "retrieval")):
    if p not in sys.path:
        sys.path.insert(0, p)

from retrieval.hybrid.hybrid_search import HybridSearcher
from retrieval.reranker.config import (
    BATCH_SIZE,
    DEVICE,
    RERANK_MODEL,
    RERANKER_METRICS_FILE,
    TOP_K_RERANK,
    TOP_K_RETRIEVAL,
)
from retrieval.reranker.evaluator import RerankerEvaluator
from retrieval.reranker.reranker import DocumentReranker
from retrieval.reranker.scorer import CrossEncoderScorer


def main():
    print("\n============================================================")
    print("       NITA CAMPUS INTELLIGENCE - RERANKER AUDIT            ")
    print("============================================================\n")

    print(f"Model Loaded:      {RERANK_MODEL}")
    print(f"Device Used:       {DEVICE}")
    print(f"Top-K Retrieval:   {TOP_K_RETRIEVAL}")
    print(f"Top-K Rerank:      {TOP_K_RERANK}")
    print(f"Batch Size:        {BATCH_SIZE}\n")

    scorer = CrossEncoderScorer(model_name=RERANK_MODEL, device=DEVICE, batch_size=BATCH_SIZE)
    reranker = DocumentReranker(scorer=scorer, top_k=TOP_K_RERANK, batch_size=BATCH_SIZE)
    searcher = HybridSearcher()

    evaluator = RerankerEvaluator(
        hybrid_searcher=searcher,
        reranker=reranker,
        top_k_retrieval=TOP_K_RETRIEVAL,
        top_k_rerank=TOP_K_RERANK,
    )

    print("Running evaluation on retrieval benchmark query set...")
    report = evaluator.evaluate()
    summary = report["summary_metrics"]

    print(f"Chunks Scored:     {summary['total_chunks_scored']}")
    print(f"Average Latency:   {summary['average_rerank_latency_ms']} ms")
    print(f"Throughput:        {summary['average_chunks_per_second']} chunks/sec\n")

    print("------------------------------------------------------------")
    print("METRICS COMPARISON: HYBRID RETRIEVAL vs. HYBRID + RERANKER")
    print("------------------------------------------------------------")
    print(f"MRR Before:        {summary['hybrid_mrr']:.4f}")
    print(f"MRR After:         {summary['reranked_mrr']:.4f}")
    print(f"nDCG@5 Before:     {summary['hybrid_ndcg@5']:.4f}")
    print(f"nDCG@5 After:      {summary['reranked_ndcg@5']:.4f}")
    print(f"nDCG@10 Before:    {summary['hybrid_ndcg@10']:.4f}")
    print(f"nDCG@10 After:     {summary['reranked_ndcg@10']:.4f}")
    print(f"Precision@5 Before:{summary['hybrid_precision@5']:.4f}")
    print(f"Precision@5 After: {summary['reranked_precision@5']:.4f}")
    print(f"Hit Rate Before:   {summary['hybrid_hit_rate']:.4f}")
    print(f"Hit Rate After:    {summary['reranked_hit_rate']:.4f}")
    print("------------------------------------------------------------")
    print(f"Report Generated:  {RERANKER_METRICS_FILE}\n")

    print("============================================================")
    print("           SAMPLE RERANKED QUERY OUTPUTS                    ")
    print("============================================================\n")

    show_queries = ["holiday", "hostel", "placement"]

    for q in show_queries:
        retrieval_res = searcher.search_hybrid(q, top_k=TOP_K_RETRIEVAL, retrieval_limit=TOP_K_RETRIEVAL)
        raw_hits = retrieval_res["results"]
        rerank_res = reranker.rerank(q, raw_hits, top_k=TOP_K_RERANK)

        print(f"Query: \"{q}\"  (Scored {rerank_res['chunks_scored']} chunks in {rerank_res['latency_ms']} ms)")
        for rank, hit in enumerate(rerank_res["results"], 1):
            title = (hit.get("title") or "")[:60]
            r_score = hit.get("retrieval_score", 0.0)
            rr_score = hit.get("rerank_score", 0.0)
            dtype = hit.get("document_type", "Notice")
            snippet = (hit.get("chunk_text") or hit.get("text") or "").replace("\n", " ")[:120]
            print(f"  [{rank}] Rerank Score: {rr_score:+.4f} | Retrieval Score: {r_score:.4f} | Type: {dtype}")
            print(f"      Title:   {title}")
            print(f"      Snippet: {snippet}...")
        print()

    print("============================================================\n")


if __name__ == "__main__":
    main()
