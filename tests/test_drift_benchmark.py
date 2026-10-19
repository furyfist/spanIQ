import json

import pytest

from benchmarks.drift.detectors import LegacyStreakDetector
from benchmarks.drift.harness import (
    judge_sampling_delay,
    replay,
    run_scenario,
    summarize,
)
from benchmarks.drift.run import main, render_summary
from benchmarks.drift.scenarios import SCENARIOS, build_stream, load_items


@pytest.fixture(scope="module")
def items():
    return load_items()


def test_items_have_context_and_answers(items):
    assert len(items) >= 8
    assert all(i.context and i.answers for i in items)


def test_streams_are_deterministic(items):
    a = build_stream(SCENARIOS["model_swap"], items, seed=3, n_baseline=10, n_pre=10, n_post=10)
    b = build_stream(SCENARIOS["model_swap"], items, seed=3, n_baseline=10, n_pre=10, n_post=10)
    assert a == b


def test_control_has_no_change_point(items):
    _, stream, change_at = build_stream(SCENARIOS["control"], items, 0, 5, 5, 5)
    assert change_at is None
    assert len(stream) == 10


def test_perturbation_only_after_change(items):
    control = build_stream(SCENARIOS["control"], items, 1, 5, 20, 20)[1]
    _, stream, change_at = build_stream(SCENARIOS["prompt_format_change"], items, 1, 5, 20, 20)
    assert stream[:change_at] == control[:change_at]
    assert all(r["output"].lstrip().startswith(("-", "##")) for r in stream[change_at:])


def test_records_are_file_collector_shaped(items):
    _, stream, _ = build_stream(SCENARIOS["retriever_degraded"], items, 0, 2, 2, 2)
    for r in stream:
        assert {"trace_id", "timestamp", "input", "output", "components"} <= set(r)
        assert [c["name"] for c in r["components"]] == ["retrieval", "generation"]
        assert r["components"][1]["output"] == r["output"]


class _AlarmAt:
    name = "fake"

    def __init__(self, at, root="generation"):
        self.at, self.root, self.i = set(at), root, -1

    def fit(self, baseline):
        pass

    def update(self, record):
        self.i += 1
        return self.i in self.at

    def root_cause(self):
        return self.root


def test_replay_counts_false_alarms_and_delay():
    stream = [{}] * 50
    r = replay(_AlarmAt({5, 30}), SCENARIOS["model_swap"], 0, [], stream, change_at=20)
    assert r.false_alarms == 1
    assert r.first_alarm == 30
    assert r.delay == 10
    assert r.found_root == "generation"


def test_summarize_rates():
    stream = [{}] * 40
    results = [
        replay(_AlarmAt({25}), SCENARIOS["model_swap"], 0, [], stream, 20),
        replay(_AlarmAt(set()), SCENARIOS["model_swap"], 1, [], stream, 20),
        replay(_AlarmAt({3}), SCENARIOS["control"], 0, [], stream, None),
    ]
    rows = {r["scenario"]: r for r in summarize(results)}
    assert rows["model_swap"]["detection_rate"] == 0.5
    assert rows["model_swap"]["median_delay"] == 5
    assert rows["model_swap"]["root_accuracy"] == 1.0
    assert rows["control"]["detection_rate"] is None
    assert rows["control"]["false_alarms_per_1k"] == pytest.approx(25.0)


def test_legacy_detector_catches_a_format_change(items):
    r = run_scenario(LegacyStreakDetector, SCENARIOS["verbose_drift"], items, 0, 50, 60, 60)
    assert r.first_alarm is not None
    assert r.found_root == "generation"


def test_judge_sampling_model():
    assert judge_sampling_delay(0.05, 0.6) == pytest.approx(100.0)
    assert judge_sampling_delay(0.0, 0.6) == float("inf")


def test_main_writes_results(tmp_path):
    rows = main(
        detectors=["legacy"], scenarios=["control", "verbose_drift"], runs=1, output_dir=tmp_path
    )
    assert {r["scenario"] for r in rows} == {"control", "verbose_drift"}
    assert (tmp_path / "results.csv").exists()
    summary = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| legacy | verbose_drift |" in summary
    assert "Sampled LLM judge, modeled" in summary


def test_saved_traces_replay(tmp_path):
    main(
        detectors=["legacy"],
        scenarios=["control"],
        runs=1,
        output_dir=tmp_path / "a",
        save_traces=str(tmp_path / "traces"),
    )
    saved = json.loads((tmp_path / "traces" / "control_seed0.json").read_text(encoding="utf-8"))
    assert saved["change_at"] is None
    rows = main(
        detectors=["legacy"],
        scenarios=["control"],
        runs=1,
        output_dir=tmp_path / "b",
        traces_dir=str(tmp_path / "traces"),
    )
    assert rows[0]["runs"] == 1


def test_unknown_names_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown detector"):
        main(detectors=["nope"], output_dir=tmp_path)
    with pytest.raises(ValueError, match="unknown scenario"):
        main(scenarios=["nope"], output_dir=tmp_path)


def test_render_summary_handles_missing_values():
    text = render_summary(
        [
            {
                "detector": "d",
                "scenario": "control",
                "runs": 1,
                "detection_rate": None,
                "median_delay": None,
                "false_alarms_per_1k": 0.0,
                "root_accuracy": None,
            }
        ],
        "synthetic streams",
    )
    assert "| d | control | 1 | n/a | n/a | 0.0 | n/a |" in text


def test_cli_drift_suite(tmp_path):
    from spaniq.cli.runner import main as cli_main

    cli_main(
        [
            "benchmark",
            "--suite",
            "drift",
            "--detectors",
            "legacy",
            "--scenarios",
            "control",
            "--runs",
            "1",
            "--output",
            str(tmp_path),
        ]
    )
    assert (tmp_path / "summary.md").exists()


def test_cli_drift_suite_bad_detector(tmp_path, capsys):
    from spaniq.cli.runner import main as cli_main

    with pytest.raises(SystemExit) as exc:
        cli_main(
            ["benchmark", "--suite", "drift", "--detectors", "nope", "--output", str(tmp_path)]
        )
    assert exc.value.code == 1
    assert "unknown detector" in capsys.readouterr().err


class _FakeGroq:
    def __init__(self):
        self.calls = []
        self.chat = self
        self.completions = self

    def create(self, **kwargs):
        from types import SimpleNamespace

        self.calls.append(kwargs)
        msg = SimpleNamespace(content=f"answer from {kwargs['model']}")
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_live_stream_switches_config_at_change(items):
    from benchmarks.drift.live import LIVE_CONFIGS, build_live_stream

    client = _FakeGroq()
    baseline, stream, change_at = build_live_stream(
        "model_swap", items, 0, n_baseline=2, n_pre=3, n_post=4, client=client
    )
    assert change_at == 3
    assert len(baseline) == 2 and len(stream) == 7
    swapped = LIVE_CONFIGS["model_swap"].model
    assert all(swapped not in r["output"] for r in stream[:3])
    assert all(swapped in r["output"] for r in stream[3:])
    assert len({r["trace_id"] for r in baseline + stream}) == 9


def test_fact_swap_keeps_shape_but_changes_facts(items):
    import random

    from benchmarks.drift.scenarios import _fact_swap, make_record

    rng = random.Random(0)
    record = next(
        r
        for r in (make_record(items, i, rng) for i in range(50))
        if any(c.isdigit() for c in r["output"])
    )
    swapped = _fact_swap(record, items, rng)
    assert swapped["output"] != record["output"]
    assert abs(len(swapped["output"].split()) - len(record["output"].split())) <= 1


def test_fraction_limits_how_many_traces_change(items):
    _, stream, change_at = build_stream(SCENARIOS["partial_retriever"], items, 0, 5, 10, 400)
    control = build_stream(SCENARIOS["control"], items, 0, 5, 10, 400)[1]
    changed = sum(
        a["metadata"]["context"] != b["metadata"]["context"]
        for a, b in zip(stream[change_at:], control[change_at:], strict=True)
    )
    assert 0.1 < changed / 400 < 0.6
