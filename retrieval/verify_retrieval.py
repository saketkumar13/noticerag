"""
Master Retrieval Verification CLI Script.
Executes chunking, Qdrant BGE-small dense ingestion, BM25 indexing,
hybrid search evaluation (Recall@5, Recall@10, MRR, latencies),
and displays sample retrieval outputs for required benchmark queries.
"""

import sys
import time
from pathlib import Path

# Ensure utf-8 stdout encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Ensure package roots are in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
for p in (str(BASE_DIR), str(BASE_DIR / "noticerag"), str(BASE_DIR / "retrieval")):
    if p not in sys.path:
        sys.path.insert(0, p)

from noticerag.config import (
    BM25_DIR,
    CHUNKS_JSONL,
    CHUNKS_PARQUET,
    EXTRACTED_TEXT_DIR,
    METADATA_DIR,
    QDRANT_DIR,
    REPORTS_DIR,
)
from retrieval.bm25.bm25_index import BM25Index
from retrieval.chunking.chunker import DocumentChunker
from retrieval.chunking.reports import ChunkReporter
from retrieval.chunking.validators import ChunkValidator
from retrieval.embeddings.embedder import DenseEmbedder
from retrieval.embeddings.ingestion import VectorIngestionPipeline
from retrieval.embeddings.qdrant_manager import QdrantManager
from retrieval.hybrid.evaluator import RetrievalEvaluator
from retrieval.hybrid.hybrid_search import HybridSearcher


def main():
    print("\n============================================================")
    print("      NITA CAMPUS INTELLIGENCE - RETRIEVAL LAYER AUDIT      ")
    print("============================================================\n")

    # 1. Phase 1: Chunking
    print("[1/4] Running Document Chunking (RecursiveCharacterSplitter 800/150)...")
    chunker = DocumentChunker()
    chunks = chunker.chunk_all_documents()
    chunk_stats = ChunkValidator.calculate_statistics(chunks)
    chunk_rep = ChunkReporter.generate_report(chunks)
    print(f"      -> Total Documents: {chunk_stats['total_documents']}")
    print(f"      -> Total Chunks:    {chunk_stats['total_chunks']}")
    print(f"      -> Avg Chunks/Doc:  {chunk_stats['average_chunks_per_document']}")
    print(f"      -> Avg Chunk Len:   {chunk_stats['average_chunk_length']} chars\n")

    # 2. Phase 2: Dense Embeddings (Qdrant)
    print("[2/4] Generating Dense Embeddings (BAAI/bge-small-en-v1.5) & Upserting to Qdrant...")
    embedder = DenseEmbedder()
    qdrant_mgr = QdrantManager(qdrant_dir=QDRANT_DIR)
    ingestion = VectorIngestionPipeline(embedder=embedder, qdrant_manager=qdrant_mgr)
    qdrant_report = ingestion.run_ingestion(chunks=chunks, recreate=True)
    print(f"      -> Vectors Stored:  {qdrant_report['vectors_stored']}")
    print(f"      -> Embedding Time:  {qdrant_report['embedding_time_seconds']}s")
    print(f"      -> Indexing Time:   {qdrant_report['indexing_time_seconds']}s\n")

    # 3. Phase 3: BM25 Sparse Index
    print("[3/4] Building Persistent BM25 Index (BM25Okapi)...")
    bm25_idx = BM25Index(index_dir=BM25_DIR)
    bm25_stats = bm25_idx.build_index(chunks)
    print(f"      -> BM25 Documents:  {bm25_stats['document_count']}")
    print(f"      -> Total Tokens:    {bm25_stats['total_tokens']}\n")

    # 4. Phase 4 & 5: Hybrid Search & Evaluation
    print("[4/4] Evaluating Retrieval Performance across Dense, BM25, and Hybrid (RRF)...")
    searcher = HybridSearcher(qdrant_manager=qdrant_mgr, embedder=embedder, bm25_index=bm25_idx)
    evaluator = RetrievalEvaluator(searcher=searcher)
    eval_metrics = evaluator.evaluate_all()
    summary = eval_metrics["summary_metrics"]

    # Summary Report Table
    print("\n============================================================")
    print("                 RETRIEVAL SUMMARY METRICS                  ")
    print("============================================================")
    print(f"Documents Indexed:        {chunk_stats['total_documents']}")
    print(f"Chunks Indexed:           {chunk_stats['total_chunks']}")
    print(f"Vectors Stored:           {qdrant_report['vectors_stored']}")
    print(f"BM25 Documents:           {bm25_stats['document_count']}")
    print(f"Hybrid Retrieval Working: YES (Qdrant Dense + BM25 with RRF)")
    print("------------------------------------------------------------")
    print("Modalities Evaluation:")
    print(f"  * Dense:   Recall@5: {summary['dense']['recall@5']:.3f} | Recall@10: {summary['dense']['recall@10']:.3f} | MRR: {summary['dense']['mrr']:.3f} | Latency: {summary['dense']['avg_latency_ms']:.1f}ms")
    print(f"  * BM25:    Recall@5: {summary['bm25']['recall@5']:.3f} | Recall@10: {summary['bm25']['recall@10']:.3f} | MRR: {summary['bm25']['mrr']:.3f} | Latency: {summary['bm25']['avg_latency_ms']:.1f}ms")
    print(f"  * Hybrid:  Recall@5: {summary['hybrid']['recall@5']:.3f} | Recall@10: {summary['hybrid']['recall@10']:.3f} | MRR: {summary['hybrid']['mrr']:.3f} | Latency: {summary['hybrid']['avg_latency_ms']:.1f}ms")
    print("------------------------------------------------------------")
    print(f"Reports Generated:        {REPORTS_DIR / 'chunk_report.json'}")
    print(f"                          {REPORTS_DIR / 'qdrant_report.json'}")
    print(f"                          {REPORTS_DIR / 'retrieval_metrics.json'}")
    print("============================================================\n")

    # Sample Retrieval Queries Display
    print("============================================================")
    print("            SAMPLE HYBRID RETRIEVAL RESULTS                 ")
    print("============================================================\n")

    validation_queries = [
        "holiday",
        "janmashtami",
        "hostel",
        "placement",
        "scholarship",
        "recruitment",
        "tender",
        "academic calendar",
    ]

    for q in validation_queries:
        res = searcher.search_hybrid(q, top_k=3)
        hits = res["results"]
        print(f"Query: \"{q}\"  (Dense: {res['latencies_ms']['dense']}ms | BM25: {res['latencies_ms']['bm25']}ms | Hybrid Total: {res['latencies_ms']['hybrid_total']}ms)")
        if not hits:
            print("  [No matching chunks found]")
        for rank, hit in enumerate(hits, 1):
            src = hit.get("retrieval_source", "hybrid").upper()
            title = hit.get("title", "")[:65]
            dtype = hit.get("document_type", "Notice")
            score = hit.get("score", 0.0)
            preview = hit.get("chunk_text", "").replace("\n", " ")[:130]
            print(f"  [{rank}] [{src}] Score: {score:.4f} | Type: {dtype}")
            print(f"      Title:   {title}")
            print(f"      Snippet: {preview}...")
        print()

    print("============================================================\n")


if __name__ == "__main__":
    main()
