from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4


_CREATE_BASELINES = """
CREATE TABLE IF NOT EXISTS baselines (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    prompt_text TEXT NOT NULL,
    model_name  TEXT,
    outputs     TEXT NOT NULL,
    n_outputs   INTEGER NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_baselines_prompt_hash ON baselines(prompt_hash);
CREATE INDEX IF NOT EXISTS idx_baselines_name ON baselines(name);
"""


@dataclass
class Baseline:
    id: str
    name: str
    prompt_hash: str
    prompt_text: str
    model_name: str | None
    outputs: str  # raw JSON string — caller uses json.loads()
    n_outputs: int
    version: int
    created_at: str
    updated_at: str


@dataclass
class BaselineSummary:
    id: str
    name: str
    prompt_hash: str
    n_outputs: int
    version: int
    model_name: str | None
    created_at: str


def _prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class BaselineStore:
    """SQLite-backed storage for baseline output collections."""

    def __init__(self, db_path: str = "spaniq.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_CREATE_BASELINES)

    def create(
        self,
        name: str,
        prompt: str,
        outputs: list[str],
        model_name: str | None = None,
    ) -> str:
        baseline_id = str(uuid4())
        now = _now()
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO baselines
                  (id, name, prompt_hash, prompt_text, model_name,
                   outputs, n_outputs, version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    baseline_id,
                    name,
                    _prompt_hash(prompt),
                    prompt,
                    model_name,
                    json.dumps(outputs),
                    len(outputs),
                    now,
                    now,
                ),
            )
        return baseline_id

    def get(self, baseline_id: str) -> Baseline:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM baselines WHERE id = ?", (baseline_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"Baseline not found: {baseline_id}")
        return Baseline(**dict(row))

    def get_by_name(self, name: str) -> Baseline:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM baselines WHERE name = ? ORDER BY version DESC LIMIT 1",
                (name,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Baseline not found: {name}")
        return Baseline(**dict(row))

    def get_by_prompt(self, prompt: str) -> Baseline | None:
        ph = _prompt_hash(prompt)
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM baselines WHERE prompt_hash = ? ORDER BY version DESC LIMIT 1",
                (ph,),
            ).fetchone()
        if row is None:
            return None
        return Baseline(**dict(row))

    def update(self, baseline_id: str, new_outputs: list[str]) -> str:
        existing = self.get(baseline_id)
        merged = json.loads(existing.outputs) + new_outputs
        now = _now()
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE baselines
                SET outputs = ?, n_outputs = ?, version = version + 1, updated_at = ?
                WHERE id = ?
                """,
                (json.dumps(merged), len(merged), now, baseline_id),
            )
        return baseline_id

    def list_all(self) -> list[BaselineSummary]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT id, name, prompt_hash, n_outputs, version, model_name, created_at "
                "FROM baselines ORDER BY created_at DESC"
            ).fetchall()
        return [BaselineSummary(**dict(r)) for r in rows]

    def delete(self, baseline_id: str) -> None:
        with self._conn() as conn:
            conn.execute("DELETE FROM baselines WHERE id = ?", (baseline_id,))
            has_timeline = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='timeline'"
            ).fetchone()
            if has_timeline:
                conn.execute(
                    "DELETE FROM timeline WHERE baseline_id = ?", (baseline_id,)
                )
