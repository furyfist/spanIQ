from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

import numpy as np

_CREATE_TIMELINE = """
CREATE TABLE IF NOT EXISTS timeline (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    trace_id    TEXT NOT NULL,
    baseline_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    score       REAL NOT NULL,
    threshold   REAL NOT NULL,
    passed      INTEGER NOT NULL,
    timestamp   TEXT NOT NULL,
    metadata    TEXT
);
CREATE INDEX IF NOT EXISTS idx_timeline_ts ON timeline(timestamp);
CREATE INDEX IF NOT EXISTS idx_timeline_metric ON timeline(metric_name, timestamp);
CREATE INDEX IF NOT EXISTS idx_timeline_baseline ON timeline(baseline_id, timestamp);
"""


@dataclass
class TimelineRow:
    id: int
    trace_id: str
    baseline_id: str
    metric_name: str
    score: float
    threshold: float
    passed: bool
    timestamp: str
    metadata: dict | None


@dataclass
class TimelineSummary:
    metric_name: str
    n: int
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    pass_rate: float
    trend: float  # linear regression slope — positive = worsening for drift metrics


class TimelineStore:
    """SQLite-backed time series of metric scores."""

    def __init__(self, db_path: str = "spaniq.db") -> None:
        self.db_path = db_path
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(_CREATE_TIMELINE)

    def record(
        self,
        trace_id: str,
        baseline_id: str,
        metric_name: str,
        score: float,
        threshold: float,
        passed: bool,
        timestamp: str,
        metadata: dict | None = None,
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO timeline
                  (trace_id, baseline_id, metric_name, score, threshold, passed, timestamp, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trace_id,
                    baseline_id,
                    metric_name,
                    score,
                    threshold,
                    int(passed),
                    timestamp,
                    json.dumps(metadata) if metadata else None,
                ),
            )

    def query(
        self,
        metric_name: str,
        last_n: int = 200,
        since: str | None = None,
    ) -> list[TimelineRow]:
        sql = "SELECT * FROM timeline WHERE metric_name = ?"
        params: list = [metric_name]
        if since:
            sql += " AND timestamp >= ?"
            params.append(since)
        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(last_n)

        with self._conn() as conn:
            rows = conn.execute(sql, params).fetchall()

        return list(reversed([
            TimelineRow(
                id=r["id"],
                trace_id=r["trace_id"],
                baseline_id=r["baseline_id"],
                metric_name=r["metric_name"],
                score=r["score"],
                threshold=r["threshold"],
                passed=bool(r["passed"]),
                timestamp=r["timestamp"],
                metadata=json.loads(r["metadata"]) if r["metadata"] else None,
            )
            for r in rows
        ]))

    def summary(self, metric_name: str, last_n: int = 200) -> TimelineSummary:
        rows = self.query(metric_name, last_n=last_n)
        if not rows:
            return TimelineSummary(
                metric_name=metric_name,
                n=0,
                mean_score=0.0,
                std_score=0.0,
                min_score=0.0,
                max_score=0.0,
                pass_rate=0.0,
                trend=0.0,
            )
        scores = np.array([r.score for r in rows], dtype=float)
        passed = np.array([int(r.passed) for r in rows], dtype=float)

        trend = 0.0
        if len(scores) >= 2:
            x = np.arange(len(scores), dtype=float)
            slope = float(np.polyfit(x, scores, 1)[0])
            trend = slope

        return TimelineSummary(
            metric_name=metric_name,
            n=len(rows),
            mean_score=float(np.mean(scores)),
            std_score=float(np.std(scores)),
            min_score=float(np.min(scores)),
            max_score=float(np.max(scores)),
            pass_rate=float(np.mean(passed)),
            trend=trend,
        )

    def count(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM timeline").fetchone()[0]
