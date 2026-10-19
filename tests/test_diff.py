import json

import pytest

from spaniq.drift.diff import diff_traces, load_traces, upstream_root
from spaniq.drift.report import render_markdown
from spaniq.monitor.trace import Trace

CONTEXT = "The Greenfield Accord was signed by 195 nations in 2021 and pledges 45% cuts by 2035."
REFUSAL = "I'm sorry, I can't help with that."


def _trace(i, answer, context=CONTEXT):
    return Trace.from_dict(
        {
            "trace_id": f"t{i}",
            "input": "Who signed the Greenfield Accord?",
            "output": answer,
            "components": [
                {"name": "retrieval", "kind": "retrieval", "output": context},
                {"name": "generation", "kind": "chat", "output": answer},
            ],
        }
    )


GOOD = [
    "195 nations signed the Greenfield Accord in 2021.",
    "The Greenfield Accord was signed by 195 nations.",
    "In 2021, 195 nations signed it, pledging 45% cuts by 2035.",
    "It was signed by 195 nations and pledges 45% cuts.",
]


def _run(n, start=0, answer=None, context=CONTEXT):
    return [_trace(start + i, answer or GOOD[i % len(GOOD)], context) for i in range(n)]


def test_unchanged_runs_do_not_drift():
    result = diff_traces(_run(60), _run(60, start=100))
    assert not result.drifted
    assert result.root_cause is None
    assert result.review == []


def test_generation_change_is_blamed_on_generation():
    bad = "Dr. Elena Marsh says 3,400 nations signed it. " * 3
    result = diff_traces(_run(60), _run(60, 100, answer=bad))
    assert result.drifted
    assert result.drifted_units == ["generation"]
    assert result.root_cause == "generation"
    assert result.review[0].trace_id.startswith("t1")
    assert any("unsupported" in r for r in result.review[0].reasons)


def test_retrieval_change_is_blamed_upstream():
    wrong = "Aurora borealis is caused by charged solar particles hitting the atmosphere."
    result = diff_traces(_run(60), _run(60, 100, context=wrong))
    assert "retrieval" in result.drifted_units
    assert result.root_cause == "retrieval"


def test_output_unit_dropped_when_it_repeats_a_span():
    result = diff_traces(_run(20), _run(20, 100))
    assert [u.unit for u in result.units] == ["retrieval", "generation"]


def test_plain_traces_use_the_output_unit():
    base = [Trace(input="q", output=GOOD[i % 4]) for i in range(30)]
    cand = [Trace(input="q", output=REFUSAL) for _ in range(30)]
    result = diff_traces(base, cand)
    assert result.drifted_units == ["output"]
    assert result.root_cause == "output"


def test_upstream_root():
    assert upstream_root([], {}) is None
    positions = {"retrieval": 0, "generation": 1, "output": float("inf")}
    assert upstream_root(["output", "generation"], positions) == "generation"


def test_to_dict_and_markdown():
    result = diff_traces(_run(40), _run(40, 100, answer=REFUSAL))
    d = json.loads(json.dumps(result.to_dict()))
    assert d["drifted"] is True
    assert d["root_cause"] == "generation"
    md = render_markdown(result)
    assert "## spanIQ diff" in md
    assert "### generation: drifted" in md
    assert "### Review first" in md


def test_load_traces_skips_bad_lines(tmp_path):
    path = tmp_path / "run.jsonl"
    path.write_text('{"input": "q", "output": "a"}\nnot json\n{"input": "q"}\n\n', encoding="utf-8")
    assert len(load_traces(str(path))) == 1


def _write(path, traces):
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "trace_id": t.trace_id,
                    "input": t.input,
                    "output": t.output,
                    "components": [
                        {"name": c.name, "kind": c.kind.value, "output": c.output}
                        for c in t.components
                    ],
                }
            )
            for t in traces
        ),
        encoding="utf-8",
    )


def test_cli_exit_codes(tmp_path, capsys):
    from spaniq.cli.runner import main

    base, same, bad = tmp_path / "b.jsonl", tmp_path / "s.jsonl", tmp_path / "x.jsonl"
    _write(base, _run(40))
    _write(same, _run(40, 100))
    _write(bad, _run(40, 100, answer=REFUSAL))

    main(["diff", str(base), str(same)])
    assert "no drift detected" in capsys.readouterr().out

    with pytest.raises(SystemExit) as exc:
        main(["diff", str(base), str(bad), "--markdown", str(tmp_path / "r.md")])
    assert exc.value.code == 1
    assert (tmp_path / "r.md").read_text(encoding="utf-8").startswith("## spanIQ diff")
    capsys.readouterr()

    main(["diff", str(base), str(bad), "--no-fail", "--json"])
    assert json.loads(capsys.readouterr().out)["drifted"] is True


def test_cli_missing_file(tmp_path, capsys):
    from spaniq.cli.runner import main

    with pytest.raises(SystemExit) as exc:
        main(["diff", str(tmp_path / "nope.jsonl"), str(tmp_path / "nope2.jsonl")])
    assert exc.value.code == 1
    assert "trace file not found" in capsys.readouterr().err


def test_embeddings_flag_adds_mmd(monkeypatch):
    import numpy as np

    import spaniq.statistical.embeddings as emb

    def fake_embed(texts):
        return np.array([[float("sorry" in t), len(t) / 100, 1.0] for t in texts])

    monkeypatch.setattr(emb, "embed", fake_embed)
    result = diff_traces(_run(30), _run(30, 100, answer=REFUSAL), embeddings=True)
    generation = next(u for u in result.units if u.unit == "generation")
    assert generation.report.get("embedding") is not None
