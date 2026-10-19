from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone

from spaniq.monitor._sqlite import connect

_CREATE = """
CREATE TABLE IF NOT EXISTS judge_budget (
    day  TEXT PRIMARY KEY,
    used INTEGER NOT NULL DEFAULT 0
);
"""


def _utc_day() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class JudgeBudget:
    """A daily cap on LLM judge calls, shared by every process using the same database.

    try_spend() reserves calls atomically, so two monitors writing to one
    spaniq.db cannot both spend the last call. The day rolls over at UTC midnight.
    """

    def __init__(
        self,
        db_path: str,
        daily_limit: int,
        clock: Callable[[], str] = _utc_day,
    ) -> None:
        if daily_limit < 0:
            raise ValueError("daily_limit must be >= 0")
        self.db_path = db_path
        self.daily_limit = daily_limit
        self._clock = clock
        with connect(db_path) as conn:
            conn.executescript(_CREATE)

    def used(self) -> int:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT used FROM judge_budget WHERE day = ?", (self._clock(),)
            ).fetchone()
        return int(row["used"]) if row else 0

    def remaining(self) -> int:
        return max(0, self.daily_limit - self.used())

    def try_spend(self, n: int = 1) -> int:
        """Reserve up to n calls for today; returns how many were granted (possibly 0)."""
        if n <= 0:
            return 0
        day = self._clock()
        with connect(self.db_path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT used FROM judge_budget WHERE day = ?", (day,)).fetchone()
            used = int(row["used"]) if row else 0
            granted = max(0, min(n, self.daily_limit - used))
            if granted:
                conn.execute(
                    "INSERT INTO judge_budget (day, used) VALUES (?, ?) "
                    "ON CONFLICT(day) DO UPDATE SET used = used + excluded.used",
                    (day, granted),
                )
        return granted
