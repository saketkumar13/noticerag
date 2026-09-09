"""
Checkpoint Management Module for NITA NoticeRAG.
Provides state persistence across crashes and interruptions, tracking crawl timestamps,
discovered totals, downloaded totals, and pending queues in both SQLite and JSON files.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from noticerag.config import CHECKPOINT_DIR
from noticerag.crawler.database import DatabaseManager

logger = logging.getLogger("noticerag.checkpoint")


def get_utc_now_iso() -> str:
    """Returns current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class CheckpointManager:
    """
    Manages persistent checkpoints in both SQLite and disk JSON snapshots.
    """

    def __init__(
        self,
        db: DatabaseManager,
        checkpoint_dir: Path = CHECKPOINT_DIR,
    ):
        self.db = db
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.latest_json_path = self.checkpoint_dir / "latest_checkpoint.json"

    def load_latest_checkpoint(self) -> Optional[Dict[str, Any]]:
        """
        Loads the latest checkpoint from disk or database.
        """
        # First check JSON on disk
        if self.latest_json_path.exists():
            try:
                with open(self.latest_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    logger.info("Loaded latest checkpoint from JSON: %s", data.get("last_run"))
                    return data
            except Exception as exc:
                logger.warning("Could not read %s: %s", self.latest_json_path, exc)

        # Fallback to SQLite
        db_ck = self.db.get_latest_checkpoint()
        if db_ck:
            logger.info("Loaded latest checkpoint from SQLite: %s", db_ck.get("last_run"))
            return {
                "last_run": db_ck["last_run"],
                "last_notice_date": db_ck["last_notice_date"],
                "total_pdfs": db_ck["total_pdfs"],
                "downloaded_count": db_ck.get("downloaded_count", 0),
                "status": "completed",
            }

        return None

    def save_checkpoint(
        self,
        total_discovered: int,
        downloaded_count: int,
        last_notice_date: Optional[str] = None,
        status: str = "completed",
        pending_urls: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Persists checkpoint to both SQLite table and disk JSON files.
        """
        now_ts = get_utc_now_iso()
        ck_data = {
            "last_run": now_ts,
            "last_notice_date": last_notice_date or "",
            "total_pdfs": total_discovered,
            "downloaded_count": downloaded_count,
            "status": status,
            "pending_urls": pending_urls or [],
        }

        # 1. Save to SQLite
        try:
            self.db.record_checkpoint(
                last_run=now_ts,
                last_notice_date=last_notice_date,
                total_pdfs=total_discovered,
                downloaded_count=downloaded_count,
            )
        except Exception as exc:
            logger.error("Failed to record checkpoint in SQLite: %s", exc)

        # 2. Save latest to JSON
        try:
            with open(self.latest_json_path, "w", encoding="utf-8") as f:
                json.dump(ck_data, f, indent=2)

            # Also create timestamped snapshot for audit history
            safe_ts = now_ts.replace(":", "-").replace(".", "_")
            snap_file = self.checkpoint_dir / f"checkpoint_{safe_ts}.json"
            with open(snap_file, "w", encoding="utf-8") as f:
                json.dump(ck_data, f, indent=2)

            logger.info("Checkpoint saved: %s (Total: %d, Downloaded: %d)", now_ts, total_discovered, downloaded_count)
        except Exception as exc:
            logger.error("Failed to save JSON checkpoint: %s", exc)

        return ck_data

    def is_resumable(self) -> bool:
        """
        Returns True if there are pending downloads from an interrupted state.
        """
        pending = self.db.get_pending_downloads(limit=1)
        return len(pending) > 0
