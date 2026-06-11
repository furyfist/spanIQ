"""E2E attribution test using committed cascade fixtures."""
from __future__ import annotations

import tempfile
import os
from pathlib import Path

import pytest

FIXTURE_PATH = Path(__file__).parent.parent / "src/spaniq/demos/fixtures/cascade/traces.jsonl"


@pytest.mark.skipif(not FIXTURE_PATH.exists(), reason="cascade fixtures not generated")
def test_e2e_cascade_attribution():
    from spaniq.attribution.attributor import attribute
    from spaniq.attribution.pipeline_monitor import PipelineMonitor
    from spaniq.monitor.collectors.file import FileCollector
    from spaniq.monitor.timeline_store import TimelineStore

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        collector = FileCollector(str(FIXTURE_PATH))
        pm = PipelineMonitor(
            pipeline_name="e2e-test",
            collector=collector,
            db_path=db_path,
            window_size=20,
            warmup=20,
        )
        report = pm.run()

        assert report.total_traces == 200
        assert "retrieval" in report.components_seen
        assert "generation" in report.components_seen
        assert "search_tool" in report.components_seen

        store = TimelineStore(db_path)
        components = store.components()

        metrics = ["ResponseDriftMetric", "SemanticSimilarityMetric", "OutputStabilityMetric"]
        result = attribute(
            timeline=store,
            components=components,
            metrics=metrics,
            last_n=200,
            cusum_alarms=report.online_alarms,
        )

        if result.root_cause is not None:
            all_breaks = [result.root_cause] + result.cascade
            break_components = [b.component for b in all_breaks]
            assert "retrieval" in break_components or "generation" in break_components

        assert "search_tool" in result.healthy or result.root_cause is None or \
            result.root_cause.component != "search_tool"

        if result.root_cause and result.cascade:
            assert result.root_cause.break_trace_index <= result.cascade[0].break_trace_index

    finally:
        try:
            os.unlink(db_path)
        except OSError:
            pass
