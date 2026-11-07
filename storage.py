"""SQLite persistence for analysis history. All queries are parameterised."""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from utils import utc_now_iso

log = logging.getLogger("ventureiq.storage")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL CHECK (kind IN ('predict','explore')),
    created_at TEXT NOT NULL,
    title TEXT NOT NULL,
    mode TEXT NOT NULL,
    model_version TEXT,
    parent_id INTEGER,
    summary TEXT,
    result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
"""


class StorageError(Exception):
    """Persistence failed; the caller must not report a save."""


class HistoryStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)

    def _connect(self) -> sqlite3.Connection:
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.db_path), timeout=5)
        except (sqlite3.Error, OSError) as exc:
            raise StorageError(f"Could not open the history database ({type(exc).__name__}).") from exc
        conn.row_factory = sqlite3.Row
        return conn

    def init(self) -> None:
        try:
            with closing(self._connect()) as conn:
                conn.executescript(_SCHEMA)
                conn.commit()
        except sqlite3.Error as exc:
            raise StorageError(f"Could not initialise the history database ({type(exc).__name__}).") from exc

    def is_available(self) -> tuple[bool, str]:
        try:
            self.init()
            return True, "SQLite history is writable."
        except StorageError as exc:
            return False, str(exc)

    def save(self, result: dict[str, Any]) -> int:
        """Persist a result and return its id. Raises StorageError on any failure."""
        kind = result.get("kind")
        if kind not in ("predict", "explore"):
            raise StorageError("Unknown analysis type; nothing was saved.")
        try:
            payload = json.dumps(result, default=str)
            with closing(self._connect()) as conn:
                conn.executescript(_SCHEMA)
                cur = conn.execute(
                    "INSERT INTO analyses (kind, created_at, title, mode, model_version, parent_id, summary, result_json)"
                    " VALUES (?,?,?,?,?,?,?,?)",
                    (kind, result.get("created_at") or utc_now_iso(), str(result.get("title", "Untitled"))[:200],
                     str(result.get("mode", "offline")), (result.get("model_info") or {}).get("model_version"),
                     result.get("parent_id"), str(result.get("summary_line", ""))[:300], payload))
                conn.commit()
                new_id = int(cur.lastrowid)
                row = conn.execute("SELECT id FROM analyses WHERE id = ?", (new_id,)).fetchone()
                if row is None:
                    raise StorageError("The record could not be read back after saving.")
                return new_id
        except (sqlite3.Error, TypeError, ValueError) as exc:
            log.error("History save failed: %s", type(exc).__name__)
            raise StorageError(f"Saving failed ({type(exc).__name__}). The report was not added to history.") from exc

    def get(self, analysis_id: int) -> dict[str, Any] | None:
        try:
            with closing(self._connect()) as conn:
                row = conn.execute("SELECT id, result_json FROM analyses WHERE id = ?", (int(analysis_id),)).fetchone()
        except sqlite3.Error as exc:
            raise StorageError(f"Could not read history ({type(exc).__name__}).") from exc
        if row is None:
            return None
        try:
            data = json.loads(row["result_json"])
        except ValueError as exc:
            raise StorageError("The stored report is corrupted and cannot be opened.") from exc
        data["history_id"] = row["id"]
        return data

    def list(self, kind: str | None = None, search: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        sql = "SELECT id, kind, created_at, title, mode, model_version, parent_id, summary FROM analyses"
        where, params = [], []
        if kind:
            where.append("kind = ?"); params.append(kind)
        if search:
            esc = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            where.append("(title LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\')")
            params += [f"%{esc}%", f"%{esc}%"]
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(int(limit))
        try:
            with closing(self._connect()) as conn:
                conn.executescript(_SCHEMA)
                return [dict(r) for r in conn.execute(sql, params).fetchall()]
        except sqlite3.Error as exc:
            raise StorageError(f"Could not read history ({type(exc).__name__}).") from exc

    def count(self) -> int:
        try:
            with closing(self._connect()) as conn:
                conn.executescript(_SCHEMA)
                return int(conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0])
        except sqlite3.Error as exc:
            raise StorageError(f"Could not read history ({type(exc).__name__}).") from exc

    def delete(self, analysis_id: int) -> bool:
        try:
            with closing(self._connect()) as conn:
                cur = conn.execute("DELETE FROM analyses WHERE id = ?", (int(analysis_id),))
                conn.commit()
                return cur.rowcount > 0
        except sqlite3.Error as exc:
            raise StorageError(f"Could not delete ({type(exc).__name__}).") from exc
