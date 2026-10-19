import numpy as np
import pytest

from spaniq.attribution.drift_attribution import (
    DriftAttribution,
    attribute_drift,
    estimate_change_start,
)
from spaniq.drift import compare


def _rows(rng, n, coverage=0.9, chars=100.0):
    return [
        {
            "question_coverage": float(rng.normal(coverage, 0.05)),
            "chars": float(rng.normal(chars, 5)),
        }
        for _ in range(n)
    ]


def test_change_start_found():
    rng = np.random.default_rng(0)
    series = list(rng.normal(0, 1, 70)) + list(rng.normal(4, 1, 30))
    assert abs(estimate_change_start(series) - 70) <= 3


def test_change_start_flat_or_short():
    assert estimate_change_start([1.0] * 50) == 0
    assert estimate_change_start([1.0, 2.0]) == 0


def test_upstream_component_is_blamed_even_when_downstream_trips_first():
    rng = np.random.default_rng(1)
    baselines = {"retrieval": _rows(rng, 100), "generation": _rows(rng, 100)}
    # 70 normal traces then 30 where retrieval coverage fell and generation got much shorter
    windows = {
        "retrieval": _rows(rng, 70) + _rows(rng, 30, coverage=0.84),
        "generation": _rows(rng, 70) + _rows(rng, 30, chars=40.0),
    }
    trigger_report = compare(baselines["generation"], windows["generation"])
    assert trigger_report.drifted
    # on the whole window the retriever's shift is diluted below significance
    assert not compare(baselines["retrieval"], windows["retrieval"]).drifted
    result = attribute_drift(
        "generation", trigger_report, windows, baselines, ["retrieval", "generation"]
    )
    assert result.root == "retrieval"
    assert result.implicated == ["retrieval", "generation"]
    assert 60 <= result.change_start <= 75
    assert "retrieval is the most upstream" in result.verdict


def test_trigger_alone_when_upstream_is_stable():
    rng = np.random.default_rng(2)
    baselines = {"retrieval": _rows(rng, 100), "generation": _rows(rng, 100)}
    windows = {
        "retrieval": _rows(rng, 100),
        "generation": _rows(rng, 60) + _rows(rng, 40, chars=40.0),
    }
    report = compare(baselines["generation"], windows["generation"])
    result = attribute_drift("generation", report, windows, baselines, ["retrieval", "generation"])
    assert result.root == "generation"
    assert result.verdict == "generation shifted; no other component did"


def test_empty_attribution_verdict():
    assert DriftAttribution(root=None, implicated=[], change_start=0).verdict == (
        "no component shifted"
    )


def test_pipeline_monitor_reports_attribution(tmp_path):
    from spaniq.attribution.component import ComponentKind, ComponentSpan
    from spaniq.attribution.pipeline_monitor import PipelineMonitor
    from spaniq.metrics.output_stability import OutputStabilityMetric
    from spaniq.monitor.trace import Trace
    from tests.test_monitor_drift import GOOD, ListCollector

    traces = []
    for i in range(240):
        broken = i >= 140
        context = (
            "Unrelated passage about volcanoes."
            if broken
            else ("Rayleigh scattering of sunlight by air makes the sky blue.")
        )
        answer = "I'm sorry, I can't help with that." if broken else GOOD[i % 4]
        traces.append(
            Trace(
                input="why is the sky blue?",
                output=answer,
                trace_id=f"p{i}",
                timestamp=f"2026-01-01T{i // 3600:02d}:{i // 60 % 60:02d}:{i % 60:02d}+00:00",
                components=[
                    ComponentSpan("retrieval", ComponentKind.RETRIEVAL, context),
                    ComponentSpan("generation", ComponentKind.GENERATION, answer),
                ],
            )
        )
    pm = PipelineMonitor(
        "rag",
        ListCollector(traces),
        metrics=[OutputStabilityMetric()],
        db_path=str(tmp_path / "p.db"),
        warmup=60,
        drift_window=40,
        drift_stride=10,
    )
    report = pm.run()
    assert len(report.drift_attributions) == 1
    assert report.drift_attributions[0].root == "retrieval"


def test_signal_names_and_attribute_cli(tmp_path, capsys):
    from spaniq.cli.runner import main
    from spaniq.monitor.timeline_store import TimelineStore

    db = str(tmp_path / "s.db")
    store = TimelineStore(db)
    for i in range(80):
        for comp, value in (("retrieval", 1.0 if i < 50 else 0.1), ("generation", 1.0)):
            store.record_signals(
                f"t{i}",
                f"rag/{comp}",
                {"question_coverage": value + 0.01 * (i % 3)},
                f"2026-01-01T00:{i // 60:02d}:{i % 60:02d}+00:00",
                component=comp,
            )
    assert store.signal_names(pipeline="rag") == ["signal.question_coverage"]
    assert store.signal_names(pipeline="other") == []
    main(
        [
            "attribute",
            "--db",
            db,
            "--pipeline",
            "rag",
            "--metrics",
            "OutputStability",
            "--signals",
            "--warmup",
            "0",
            "--json",
        ]
    )
    out = capsys.readouterr().out
    assert '"component": "retrieval"' in out


def test_attribute_signals_flag_without_series(tmp_path, capsys):
    from spaniq.cli.runner import main
    from spaniq.monitor.timeline_store import TimelineStore

    db = str(tmp_path / "e.db")
    TimelineStore(db).record(
        "t", "rag/gen", "OutputStabilityMetric", 0.1, 0.15, True, "2026", component="gen"
    )
    with pytest.raises(SystemExit):
        main(["attribute", "--db", db, "--pipeline", "rag", "--signals"])
    assert "no signal series" in capsys.readouterr().err
