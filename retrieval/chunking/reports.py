"""
Chunk Reporting Module.
Samples representative chunks and generates data/reports/chunk_report.json.
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from noticerag.config import REPORTS_DIR
from retrieval.chunking.validators import ChunkValidator


class ChunkReporter:
    """
    Samples chunks and compiles the chunk validation report.
    """

    @classmethod
    def generate_report(
        cls,
        chunks: List[Dict[str, Any]],
        sample_size: int = 20,
        output_file: Optional[Path] = None,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Calculates statistics, samples 20 chunks, and writes data/reports/chunk_report.json.
        """
        output_path = output_file or (REPORTS_DIR / "chunk_report.json")
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        stats = ChunkValidator.calculate_statistics(chunks)

        # Sample chunks
        rng = random.Random(seed)
        shuffled = list(chunks)
        rng.shuffle(shuffled)
        sample_chunks = shuffled[: min(sample_size, len(shuffled))]

        formatted_samples = []
        for c in sample_chunks:
            formatted_samples.append(
                {
                    "chunk_id": c["chunk_id"],
                    "document_id": c["document_id"],
                    "chunk_index": c["chunk_index"],
                    "title": c.get("title", ""),
                    "date": c.get("date", ""),
                    "document_type": c.get("document_type", ""),
                    "department": c.get("department", ""),
                    "char_count": c.get("char_count", len(c.get("chunk_text", ""))),
                    "word_count": c.get("word_count", len(c.get("chunk_text", "").split())),
                    "chunk_preview": c.get("chunk_text", "")[:250],
                }
            )

        report_data = {
            "statistics": stats,
            "sample_count": len(formatted_samples),
            "samples": formatted_samples,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return report_data
