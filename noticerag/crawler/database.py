"""
Database management module for NITA NoticeRAG.
Handles SQLite connection, schema initialization, and transactional CRUD operations.
"""

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def get_utc_now_iso() -> str:
    """Returns current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class DatabaseManager:
    """
    Thread-safe SQLite manager for PDF metadata and crawl checkpoints.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_db(self) -> None:
        """Initializes tables matching the Phase 1 schema requirements."""
        with self._lock, self._get_connection() as conn:
            # Table: PDFs
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS PDFs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT,
                    pdf_url TEXT UNIQUE NOT NULL,
                    source_page TEXT,
                    discovered_at TEXT NOT NULL,
                    downloaded_at TEXT,
                    file_path TEXT,
                    file_hash TEXT,
                    status TEXT NOT NULL DEFAULT 'discovered'
                );
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pdfs_url ON PDFs(pdf_url);")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_pdfs_status ON PDFs(status);")

            # Table: Crawl Checkpoints
            # Supports exact table name Crawl_Checkpoints and alias
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS [Crawl Checkpoints] (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    last_run TEXT NOT NULL,
                    last_notice_date TEXT,
                    total_pdfs INTEGER NOT NULL DEFAULT 0,
                    downloaded_count INTEGER NOT NULL DEFAULT 0
                );
                """
            )
            conn.commit()

    def insert_pdf(
        self,
        title: str,
        pdf_url: str,
        source_page: str,
        discovered_at: Optional[str] = None,
        status: str = "discovered",
    ) -> bool:
        """
        Inserts a discovered PDF if not already present.
        Returns True if inserted, False if already existed.
        """
        if not discovered_at:
            discovered_at = get_utc_now_iso()

        with self._lock, self._get_connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO PDFs (title, pdf_url, source_page, discovered_at, status)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (title.strip() if title else "", pdf_url.strip(), source_page.strip(), discovered_at, status),
                )
                conn.commit()
                return True
            except sqlite3.IntegrityError:
                # Duplicate pdf_url
                return False

    def insert_many_pdfs(self, pdf_records: List[Dict[str, Any]]) -> Tuple[int, int]:
        """
        Bulk insert PDFs with deduplication.
        Returns (inserted_count, skipped_count).
        """
        inserted = 0
        skipped = 0
        now_ts = get_utc_now_iso()

        with self._lock, self._get_connection() as conn:
            for rec in pdf_records:
                title = rec.get("title", "").strip()
                pdf_url = rec.get("pdf_url", "").strip()
                source_page = rec.get("source_page", "").strip()
                discovered_at = rec.get("discovered_at") or now_ts
                status = rec.get("status", "discovered")

                try:
                    conn.execute(
                        """
                        INSERT INTO PDFs (title, pdf_url, source_page, discovered_at, status)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (title, pdf_url, source_page, discovered_at, status),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    skipped += 1
            conn.commit()

        return inserted, skipped

    def update_pdf_downloaded(
        self,
        pdf_url: str,
        file_path: str,
        file_hash: str,
        downloaded_at: Optional[str] = None,
    ) -> bool:
        """
        Updates a PDF status to downloaded with its file path and SHA256 hash.
        """
        if not downloaded_at:
            downloaded_at = get_utc_now_iso()

        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE PDFs
                SET downloaded_at = ?,
                    file_path = ?,
                    file_hash = ?,
                    status = 'downloaded'
                WHERE pdf_url = ?
                """,
                (downloaded_at, file_path, file_hash, pdf_url),
            )
            conn.commit()
            return cursor.rowcount > 0

    def update_pdf_failed(self, pdf_url: str) -> bool:
        """Updates a PDF status to 'failed'."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                UPDATE PDFs
                SET status = 'failed'
                WHERE pdf_url = ?
                """,
                (pdf_url,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_pdf_by_url(self, pdf_url: str) -> Optional[Dict[str, Any]]:
        """Retrieves a single PDF record by URL."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM PDFs WHERE pdf_url = ?",
                (pdf_url,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_all_pdfs(self) -> List[Dict[str, Any]]:
        """Retrieves all PDF records."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM PDFs ORDER BY id ASC")
            return [dict(row) for row in cursor.fetchall()]

    def get_pending_downloads(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Retrieves PDFs marked as 'discovered' or 'failed' needing download.
        """
        query = "SELECT * FROM PDFs WHERE status != 'downloaded' ORDER BY id ASC"
        if limit:
            query += f" LIMIT {int(limit)}"

        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def get_downloaded_pdfs(self) -> List[Dict[str, Any]]:
        """Retrieves all successfully downloaded PDFs."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute("SELECT * FROM PDFs WHERE status = 'downloaded' ORDER BY id ASC")
            return [dict(row) for row in cursor.fetchall()]

    def record_checkpoint(
        self,
        last_run: str,
        last_notice_date: Optional[str],
        total_pdfs: int,
        downloaded_count: int,
    ) -> int:
        """Records a new checkpoint snapshot."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO [Crawl Checkpoints] (last_run, last_notice_date, total_pdfs, downloaded_count)
                VALUES (?, ?, ?, ?)
                """,
                (last_run, last_notice_date or "", total_pdfs, downloaded_count),
            )
            conn.commit()
            return cursor.lastrowid

    def get_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """Retrieves the most recent checkpoint."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM [Crawl Checkpoints] ORDER BY id DESC LIMIT 1"
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_stats(self) -> Dict[str, int]:
        """Returns row counts by status and checkpoint total."""
        with self._lock, self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status = 'downloaded' THEN 1 ELSE 0 END) as downloaded,
                    SUM(CASE WHEN status = 'discovered' THEN 1 ELSE 0 END) as discovered,
                    SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) as failed
                FROM PDFs
                """
            )
            row = cursor.fetchone()
            total_ck = conn.execute("SELECT COUNT(*) FROM [Crawl Checkpoints]").fetchone()[0]

            return {
                "total_pdfs": row["total"] or 0,
                "downloaded": row["downloaded"] or 0,
                "discovered": row["discovered"] or 0,
                "failed": row["failed"] or 0,
                "checkpoints_count": total_ck,
            }
