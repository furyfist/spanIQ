"""CLI behaviour: flag placement, clear errors instead of tracebacks, flags that used to be ignored."""

from __future__ import annotations

import json

import pytest

from spaniq.cli.runner import _build_parser, _trace_to_json, main
from spaniq.monitor.baseline_store import BaselineStore
from spaniq.monitor.timeline_store import TimelineStore


def _error(capsys, argv) -> str:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 1
    return capsys.readouterr().err


def test_db_flag_accepted_after_subcommand(tmp_path, capsys):
    db = str(tmp_path / "x.db")
    BaselineStore(db).create(name="b", prompt="p", outputs=["a"])
    main(["baseline", "list", "--db", db])
    assert "b" in capsys.readouterr().out
    assert _build_parser().parse_args(["baseline", "--db", "y.db", "list"]).db == "y.db"


def test_missing_baseline_is_a_clean_error(tmp_path, capsys):
    traces = tmp_path / "t.jsonl"
    traces.write_text(json.dumps({"input": "q", "output": "a"}) + "\n", encoding="utf-8")
    err = _error(
        capsys,
        [
            "monitor",
            "run",
            "--db",
            str(tmp_path / "e.db"),
            "--baseline",
            "nope",
            "--path",
            str(traces),
        ],
    )
    assert "Baseline not found: nope" in err


def test_unknown_metric_is_rejected(tmp_path, capsys):
    db = str(tmp_path / "m.db")
    BaselineStore(db).create(name="b", prompt="p", outputs=["a"])
    traces = tmp_path / "t.jsonl"
    traces.write_text(json.dumps({"input": "q", "output": "a"}) + "\n", encoding="utf-8")
    err = _error(
        capsys,
        [
            "monitor",
            "run",
            "--db",
            db,
            "--baseline",
            "b",
            "--path",
            str(traces),
            "--metrics",
            "Drfit",
        ],
    )
    assert "unknown metric 'Drfit'" in err
    assert "ResponseDriftMetric" in err


def test_missing_trace_file_is_a_clean_error(tmp_path, capsys):
    err = _error(
        capsys,
        [
            "monitor",
            "run",
            "--db",
            str(tmp_path / "f.db"),
            "--baseline",
            "b",
            "--path",
            "missing.jsonl",
        ],
    )
    assert "missing.jsonl" in err


def test_baseline_file_skips_lines_without_output(tmp_path, capsys):
    path = tmp_path / "base.jsonl"
    path.write_text(
        "\n".join(
            [json.dumps({"output": "one"}), json.dumps({"text": "no output key"}), "plain line"]
        ),
        encoding="utf-8",
    )
    db = str(tmp_path / "b.db")
    main(
        [
            "baseline",
            "collect",
            "--db",
            db,
            "--name",
            "n",
            "--prompt",
            "p",
            "--source",
            "file",
            "--path",
            str(path),
        ]
    )
    assert json.loads(BaselineStore(db).get_by_name("n").outputs) == ["one", "plain line"]
    assert "no 'output' field" in capsys.readouterr().err


def test_collect_otel_store_only_does_not_need_baseline():
    args = _build_parser().parse_args(["collect-otel", "--store-only"])
    assert args.baseline is None
    assert args.host == "127.0.0.1"


def test_collect_otel_without_baseline_or_store_only_errors(capsys):
    err = _error(capsys, ["collect-otel"])
    assert "--baseline is required" in err


def test_trace_to_json_round_trips_through_file_collector(tmp_path):
    from spaniq.attribution.component import ComponentKind, ComponentSpan
    from spaniq.monitor.collectors.file import FileCollector
    from spaniq.monitor.trace import Trace

    trace = Trace(
        input="q",
        output="a",
        trace_id="abc",
        components=[ComponentSpan("retriever", ComponentKind.RETRIEVAL, "docs", latency_ms=3.0)],
    )
    path = tmp_path / "otel.jsonl"
    path.write_text(_trace_to_json(trace) + "\n", encoding="utf-8")
    (back,) = list(FileCollector(str(path)).collect())
    assert back.trace_id == "abc"
    assert back.components[0].name == "retriever"
    assert back.components[0].kind is ComponentKind.RETRIEVAL


def test_attribute_uses_pipeline_flag(tmp_path, capsys):
    db = str(tmp_path / "p.db")
    store = TimelineStore(db)
    store.record(
        "t1",
        "other/gen",
        "ResponseDriftMetric",
        0.1,
        0.1,
        True,
        "2024-01-01T00:00:00",
        component="gen",
    )
    err = _error(capsys, ["attribute", "--db", db, "--pipeline", "mine"])
    assert "no data for pipeline 'mine'" in err
    assert "other" in err


def test_pipeline_status_uses_name(tmp_path, capsys):
    db = str(tmp_path / "s.db")
    store = TimelineStore(db)
    for pipe, score in (("a", 0.1), ("b", 0.9)):
        store.record(
            f"{pipe}-1",
            f"{pipe}/gen",
            "ResponseDriftMetric",
            score,
            0.1,
            True,
            "2024-01-01T00:00:00",
            component="gen",
        )
    main(["pipeline", "status", "--db", db, "--name", "b"])
    out = capsys.readouterr().out
    assert "0.9000" in out
    assert "0.1000" not in out


def test_benchmark_outside_checkout_explains_how_to_run(monkeypatch, capsys):
    import builtins

    real_import = builtins.__import__

    def no_benchmarks(name, *args, **kwargs):
        if name.startswith("benchmarks"):
            raise ImportError("No module named 'benchmarks'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_benchmarks)
    err = _error(capsys, ["benchmark", "--tool", "spaniq"])
    assert "source checkout" in err
