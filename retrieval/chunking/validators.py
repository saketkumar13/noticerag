"""
Chunk Validation Module.
Computes chunk distributions, document chunk frequencies, and statistical bounds.
"""

from typing import Any, Dict, List


class ChunkValidator:
    """
    Validates chunk counts, character lengths, and source document coverage.
    """

    @classmethod
    def calculate_statistics(cls, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculates required chunking statistics:
        - Total Documents
        - Total Chunks
        - Average Chunks Per Document
        - Average Chunk Length
        - Min Chunk Length
        - Max Chunk Length
        """
        if not chunks:
            return {
                "total_documents": 0,
                "total_chunks": 0,
                "average_chunks_per_document": 0.0,
                "average_chunk_length": 0.0,
                "min_chunk_length": 0,
                "max_chunk_length": 0,
            }

        total_chunks = len(chunks)
        doc_ids = {c["document_id"] for c in chunks}
        total_docs = len(doc_ids)

        lengths = [c.get("char_count", len(c.get("chunk_text", ""))) for c in chunks]
        min_len = min(lengths) if lengths else 0
        max_len = max(lengths) if lengths else 0
        avg_len = (sum(lengths) / total_chunks) if total_chunks else 0.0
        avg_chunks_per_doc = (total_chunks / total_docs) if total_docs else 0.0

        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "average_chunks_per_document": round(avg_chunks_per_doc, 2),
            "average_chunk_length": round(avg_len, 2),
            "min_chunk_length": min_len,
            "max_chunk_length": max_len,
        }
