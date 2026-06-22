"""Tests for dashboard alert log data logic (Step 13)."""
from __future__ import annotations

import io
import pytest

from spaniq.monitor.timeline_store import TimelineStore


@pytest.fixture()
def alerts_db(tmp_path):
    db = str(tmp_path / "alerts.db")
    store = TimelineStore(db)
    for i in range(5):
        store.record_alert(
            timestamp=f"2024-01-0{i+1}T12:00:00",
            metric_name="ResponseDriftMetric",
            score=0.3 + i * 0.05,
            threshold=0.2,
            message=f"Alert {i}",
            component="retriever" if i < 3 else "generator",
            severity="warning",
        )
    return db, store


def test_alert_table_renders_with_data(alerts_db):
    _, store = alerts_db
    alerts = store.query_alerts(last_n=10)
    assert len(alerts) == 5


def test_component_filter_works(alerts_db):
    _, store = alerts_db
    retriever_alerts = store.query_alerts(component="retriever")
    assert len(retriever_alerts) == 3


def test_empty_alert_log_returns_zero(tmp_path):
    store = TimelineStore(str(tmp_path / "empty.db"))
    assert store.alert_count() == 0
    alerts = store.query_alerts()
    assert alerts == []


def test_csv_export_valid(alerts_db):
    import pandas as pd
    _, store = alerts_db
    alerts = store.query_alerts()
    df = pd.DataFrame(alerts)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    df2 = pd.read_csv(io.BytesIO(csv_bytes))
    assert len(df2) == 5
    assert "metric_name" in df2.columns
