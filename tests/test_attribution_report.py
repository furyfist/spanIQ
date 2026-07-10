"""Tests for AttributionReport (terminal, JSON, PNG)."""

from __future__ import annotations

import json

from spaniq.attribution.attributor import AttributionResult, ComponentBreak
from spaniq.attribution.report import attribution_to_dict, print_attribution, save_attribution_json


def _make_result(root_comp="retrieval", cascade_comp="generation"):
    root = ComponentBreak(
        component=root_comp,
        break_trace_index=80,
        cusum_alarm_index=85,
        broken_metrics=["ResponseDriftMetric", "SemanticSimilarityMetric"],
        confidence=0.75,
    )
    cascade = ComponentBreak(
        component=cascade_comp,
        break_trace_index=90,
        cusum_alarm_index=95,
        broken_metrics=["ResponseDriftMetric"],
        confidence=0.45,
    )
    return AttributionResult(
        event_window=(80, 200),
        root_cause=root,
        cascade=[cascade],
        healthy=["search_tool"],
        verdict=f"{root_comp} broke first by 10 trace(s); {cascade_comp} drift is cascade",
    )


def test_attribution_to_dict_structure():
    result = _make_result()
    d = attribution_to_dict(result)
    assert d["root_cause"]["component"] == "retrieval"
    assert d["cascade"][0]["component"] == "generation"
    assert d["healthy"] == ["search_tool"]
    assert "broke first" in d["verdict"]
    assert d["event_window"] == [80, 200]


def test_attribution_to_dict_none_root():
    result = AttributionResult(
        event_window=(0, 200),
        root_cause=None,
        cascade=[],
        healthy=["a", "b"],
        verdict="no degradation detected",
    )
    d = attribution_to_dict(result)
    assert d["root_cause"] is None
    assert d["healthy"] == ["a", "b"]


def test_save_attribution_json(tmp_path):
    result = _make_result()
    path = str(tmp_path / "out.json")
    save_attribution_json(result, path)
    with open(path) as f:
        loaded = json.load(f)
    assert loaded["root_cause"]["component"] == "retrieval"
    assert loaded["cascade"][0]["break_trace_index"] == 90


def test_print_attribution_no_crash(capsys):
    result = _make_result()
    print_attribution(result)
