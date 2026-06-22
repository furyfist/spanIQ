from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DashboardConfig:
    db_path: str = "spaniq.db"
    refresh_interval: int = 10
    page_title: str = "spanIQ Dashboard"
    max_rows: int = 500
