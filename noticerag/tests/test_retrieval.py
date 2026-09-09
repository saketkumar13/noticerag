"""
Unit and Integration Tests for Retrieval Layer.
Verifies chunking, BGE-small embeddings, Qdrant manager, BM25 indexing,
Reciprocal Rank Fusion, and retrieval evaluation metrics.
"""

import shutil
import tempfile
import unittest
from pathlib import Path

from retrieval.bm25.bm25_index import BM25Index
from retrieval.bm25.tokenizer import BM25Tokenizer
from retrieval.chunking.chunker import DocumentChunker
from retrieval.chunking.validators import ChunkValidator
from retrieval.embeddings.embedder import DenseEmbedder
from retrieval.embeddings.qdrant_manager import QdrantManager
from retrieval.hybrid.evaluator import RetrievalEvaluator
from retrieval.hybrid.fusion import ReciprocalRankFusion


class TestChunking(unittest.TestCase):
    def setUp(self):
        self.chunker = DocumentChunker(chunk_size=300, chunk_overlap=50)

    def test_chunk_splitting_and_lineage(self):
        doc = {
            "document_id": "test_doc_01",
            "title": "Janmashtami Notice",
            "text": (
                "National Institute of Technology Agartala.\n"
                "Declaration of holiday on account of Janmashtami on 4th September 2026. "
                "All offices and academic departments will remain closed for the entire day. "
                "Essential campus facilities will continue operating under duty staff. "
            ) * 5,
        }
        chunks = self.chunker.split_document(doc)
        self.assertGreater(len(chunks), 1)

        first_chunk = chunks[0]
        self.assertEqual(first_chunk["document_id"], "test_doc_01")
        self.assertEqual(first_chunk["chunk_index"], 0)
        self.assertEqual(first_chunk["chunk_id"], "test_doc_01_c000")
        self.assertIn("Janmashtami Notice", first_chunk["title"])

        # Second chunk
        second_chunk = chunks[1]
        self.assertEqual(second_chunk["chunk_id"], "test_doc_01_c001")
        self.assertEqual(second_chunk["chunk_index"], 1)

    def test_chunk_validator(self):
        chunks = [
            {"document_id": "d1", "char_count": 500, "chunk_text": "text1"},
            {"document_id": "d1", "char_count": 600, "chunk_text": "text2"},
            {"document_id": "d2", "char_count": 700, "chunk_text": "text3"},
        ]
        stats = ChunkValidator.calculate_statistics(chunks)
        self.assertEqual(stats["total_documents"], 2)
        self.assertEqual(stats["total_chunks"], 3)
        self.assertEqual(stats["min_chunk_length"], 500)
        self.assertEqual(stats["max_chunk_length"], 700)
        self.assertEqual(stats["average_chunk_length"], 600.0)


class TestBM25(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.bm25_dir = Path(self.temp_dir) / "bm25"
        self.bm25_index = BM25Index(index_dir=self.bm25_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_tokenizer(self):
        text = "Notice: Holiday on 4th September 2026 for B.Sc.-B.Ed."
        tokens = BM25Tokenizer.tokenize(text)
        self.assertIn("notice", tokens)
        self.assertIn("holiday", tokens)
        self.assertIn("2026", tokens)
        self.assertNotIn("on", tokens)  # stopword
        self.assertNotIn("for", tokens) # stopword

    def test_bm25_build_and_search(self):
        chunks = [
            {
                "chunk_id": "c1",
                "document_id": "d1",
                "title": "Holiday Notice",
                "chunk_text": "Declaration of holiday on account of Janmashtami festival.",
            },
            {
                "chunk_id": "c2",
                "document_id": "d2",
                "title": "Hostel Admission",
                "chunk_text": "Chief warden hostel allotment for first year engineering students.",
            },
        ]
        stats = self.bm25_index.build_index(chunks)
        self.assertEqual(stats["document_count"], 2)

        # Search for holiday
        hits = self.bm25_index.search("janmashtami", top_k=5)
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0]["chunk_id"], "c1")

        # Persistence test: reload from disk
        new_index = BM25Index(index_dir=self.bm25_dir)
        loaded = new_index.load()
        self.assertTrue(loaded)
        reloaded_hits = new_index.search("hostel", top_k=5)
        self.assertGreaterEqual(len(reloaded_hits), 1)
        self.assertEqual(reloaded_hits[0]["chunk_id"], "c2")


class TestFusion(unittest.TestCase):
    def test_rrf_scoring(self):
        dense = [
            {"chunk_id": "c1", "score": 0.95, "title": "Doc 1"},
            {"chunk_id": "c2", "score": 0.85, "title": "Doc 2"},
        ]
        bm25 = [
            {"chunk_id": "c2", "score": 12.5, "title": "Doc 2"},
            {"chunk_id": "c3", "score": 8.0, "title": "Doc 3"},
        ]
        fused = ReciprocalRankFusion.fuse_rankings(dense, bm25, top_k=5, k=60)
        self.assertGreaterEqual(len(fused), 3)

        # c2 appeared in both (rank 2 in dense, rank 1 in bm25) -> should have highest RRF score
        # c2 score: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0163 = 0.0325
        # c1 score: 1/(60+1) = 0.0163
        self.assertEqual(fused[0]["chunk_id"], "c2")
        self.assertEqual(fused[0]["retrieval_source"], "hybrid")


class TestEmbeddingsAndQdrant(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.qdrant_dir = Path(self.temp_dir) / "qdrant"
        self.qdrant = QdrantManager(qdrant_dir=self.qdrant_dir, collection_name="test_coll")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_embedder_dimension(self):
        embedder = DenseEmbedder()
        vec = embedder.embed_query("Holiday in Agartala")
        self.assertEqual(len(vec), 384)

    def test_qdrant_upsert_and_search(self):
        chunks = [
            {"chunk_id": "c1", "document_id": "d1", "title": "Holiday", "chunk_text": "Holiday notice"},
            {"chunk_id": "c2", "document_id": "d2", "title": "Hostel", "chunk_text": "Hostel notice"},
        ]
        embedder = DenseEmbedder()
        vectors = embedder.embed_texts([c["chunk_text"] for c in chunks])
        upserted = self.qdrant.upsert_chunks(chunks, vectors)
        self.assertEqual(upserted, 2)

        info = self.qdrant.get_collection_info()
        self.assertEqual(info["points_count"], 2)

        q_vec = embedder.embed_query("Holiday notice")
        hits = self.qdrant.search_dense(q_vec, top_k=2)
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0]["chunk_id"], "c1")


class TestEvaluationMetrics(unittest.TestCase):
    def test_evaluate_ranking(self):
        hits = [
            {"document_id": "docA", "chunk_id": "docA_c000"},
            {"document_id": "docB", "chunk_id": "docB_c000"},
            {"document_id": "docC", "chunk_id": "docC_c000"},
        ]
        # Match at rank 1
        res1 = RetrievalEvaluator._evaluate_ranking(hits, "docA")
        self.assertEqual(res1["recall@5"], 1.0)
        self.assertEqual(res1["recall@10"], 1.0)
        self.assertEqual(res1["mrr"], 1.0)

        # Match at rank 3
        res3 = RetrievalEvaluator._evaluate_ranking(hits, "docC")
        self.assertEqual(res3["recall@5"], 1.0)
        self.assertAlmostEqual(res3["mrr"], 1.0 / 3.0)

        # No match
        res_none = RetrievalEvaluator._evaluate_ranking(hits, "docZ")
        self.assertEqual(res_none["recall@5"], 0.0)
        self.assertEqual(res_none["mrr"], 0.0)


if __name__ == "__main__":
    unittest.main()
