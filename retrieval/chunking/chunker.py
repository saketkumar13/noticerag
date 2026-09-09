"""
Document Chunker Module.
Partitions extracted notices using RecursiveCharacterTextSplitter,
preserves document lineage and metadata, generates deterministic chunk IDs,
and saves chunks as Parquet and JSONL.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

from noticerag.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_DIR,
    CHUNKS_JSONL,
    CHUNKS_PARQUET,
    EXTRACTED_TEXT_DIR,
    METADATA_DIR,
)

logger = logging.getLogger("noticerag.retrieval.chunking")


class DocumentChunker:
    """
    Chunks documents while maintaining complete metadata lineage.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        chunks_dir: Path = CHUNKS_DIR,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks_dir = Path(chunks_dir)
        self.chunks_dir.mkdir(parents=True, exist_ok=True)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True,
        )

    def split_document(self, doc_data: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Splits a single document into metadata-enriched chunks.
        """
        document_id = doc_data.get("document_id", "")
        title = doc_data.get("title", "")
        text = doc_data.get("text", "")

        meta = metadata or {}
        date_str = meta.get("date") or doc_data.get("metadata", {}).get("date", "")
        doc_type = meta.get("document_type") or doc_data.get("metadata", {}).get("document_type", "General Notice")
        department = meta.get("department") or doc_data.get("metadata", {}).get("department", "General Administration")
        issuer = meta.get("issuer") or doc_data.get("metadata", {}).get("issuer", "Competent Authority")

        if not text or not text.strip():
            return []

        raw_splits = self.splitter.split_text(text)
        chunks = []

        for idx, chunk_str in enumerate(raw_splits):
            chunk_text = chunk_str.strip()
            if not chunk_text:
                continue

            chunk_id = f"{document_id}_c{idx:03d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "chunk_index": idx,
                    "chunk_text": chunk_text,
                    "title": title,
                    "date": date_str,
                    "document_type": doc_type,
                    "department": department,
                    "issuer": issuer,
                    "char_count": len(chunk_text),
                    "word_count": len(chunk_text.split()),
                }
            )

        return chunks

    def chunk_all_documents(
        self,
        extracted_dir: Path = EXTRACTED_TEXT_DIR,
        metadata_dir: Path = METADATA_DIR,
    ) -> List[Dict[str, Any]]:
        """
        Loads all extracted documents, joins metadata, generates chunks,
        and saves outputs to data/chunks/chunks.parquet and data/chunks/chunks.jsonl.
        """
        extracted_dir = Path(extracted_dir)
        metadata_dir = Path(metadata_dir)

        json_files = sorted(list(extracted_dir.glob("*.json")))
        logger.info("Found %d extracted documents for chunking.", len(json_files))

        # Build metadata lookup
        metadata_lookup = {}
        for mpath in metadata_dir.glob("*.json"):
            try:
                with open(mpath, "r", encoding="utf-8") as mf:
                    mdata = json.load(mf)
                    doc_id = mdata.get("document_id")
                    if doc_id:
                        metadata_lookup[doc_id] = mdata
            except Exception:
                pass

        all_chunks: List[Dict[str, Any]] = []

        for jf in json_files:
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    doc = json.load(f)

                doc_id = doc.get("document_id", jf.stem)
                meta = metadata_lookup.get(doc_id, {})
                doc_chunks = self.split_document(doc, metadata=meta)
                all_chunks.extend(doc_chunks)
            except Exception as exc:
                logger.error("Failed to chunk %s: %s", jf.name, exc)

        logger.info("Generated %d total chunks across %d documents.", len(all_chunks), len(json_files))

        # 1. Save chunks.jsonl
        with open(CHUNKS_JSONL, "w", encoding="utf-8") as jf:
            for chunk in all_chunks:
                jf.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        logger.info("Saved chunks to %s", CHUNKS_JSONL)

        # 2. Save chunks.parquet
        df = pd.DataFrame(all_chunks)
        df.to_parquet(CHUNKS_PARQUET, index=False, engine="pyarrow")
        logger.info("Saved chunks to %s (%d rows)", CHUNKS_PARQUET, len(df))

        return all_chunks
