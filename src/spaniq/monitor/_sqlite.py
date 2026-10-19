"""One place that opens SQLite connections for the stores.

The OTel receiver threads write while the Streamlit dashboard reads the same
file, so connections use WAL (readers never block the writer) and a busy
timeout instead of failing at once with "database is locked". Connections are
always closed: `with sqlite3.connect(...)` only commits, and leaked handles keep
the file locked on Windows.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

BUSY_TIMEOUT_SEC = 30.0


@contextmanager
def connect(db_path: str) -> Iterator[sqlite3.Connection]:
    """Open, yield, commit on success (rollback on error), and always close."""
    conn = sqlite3.connect(db_path, timeout=BUSY_TIMEOUT_SEC)
    conn.row_factory = sqlite3.Row
    try:
        if db_path != ":memory:":
            conn.execute("PRAGMA journal_mode=WAL")
        with conn:
            yield conn
    finally:
        conn.close()
