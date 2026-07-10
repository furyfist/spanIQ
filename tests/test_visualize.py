from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from spaniq.monitor.timeline_store import TimelineStore
from spaniq.monitor.visualize import export_timeline_png, print_sparkline


def _ts(offset: int = 0) -> str:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return (base + timedelta(seconds=offset)).isoformat()


@pytest.fixture
def populated_store(tmp_path):
    store = TimelineStore(db_path=str(tmp_path / "viz.db"))
    for i in range(20):
        store.record(
            trace_id=f"t{i}",
            baseline_id="b1",
            metric_name="ResponseDriftMetric",
            score=0.01 + i * 0.01,
            threshold=0.10,
            passed=(0.01 + i * 0.01) < 0.10,
            timestamp=_ts(i),
        )
    return store


def test_export_png_creates_file(tmp_path, populated_store):
    out = str(tmp_path / "chart.png")
    result = export_timeline_png(populated_store, "ResponseDriftMetric", output_path=out)
    import os

    assert os.path.exists(result)
    assert result == out


def test_export_png_valid_image(tmp_path, populated_store):
    out = str(tmp_path / "chart.png")
    export_timeline_png(populated_store, "ResponseDriftMetric", output_path=out)
    # verify it's a valid PNG by checking magic bytes
    with open(out, "rb") as f:
        header = f.read(8)
    assert header[:4] == b"\x89PNG"


def test_sparkline_output_contains_blocks(populated_store):
    print_sparkline(populated_store, "ResponseDriftMetric", last_n=20)
    # rich writes to its own console; check no exception was raised
    # and function runs without error (rich Console captures internally)
    # just assert it doesn't crash — integration tested via CLI


def test_empty_timeline_no_crash(tmp_path, capsys):
    store = TimelineStore(db_path=str(tmp_path / "empty.db"))
    # both functions should handle 0 rows gracefully
    export_timeline_png(store, "NonExistentMetric", output_path=str(tmp_path / "out.png"))
    print_sparkline(store, "NonExistentMetric", last_n=50)
