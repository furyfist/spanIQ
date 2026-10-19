import json

import pytest

from benchmarks.grounding.loaders import (
    GroundingExample,
    load,
    load_aggrefact,
    load_halueval,
    load_local,
    load_ragtruth,
    sample,
)
from benchmarks.grounding.run import evaluate, main, parse_datasets


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_local_examples_are_balanced():
    examples = load_local()
    assert len(examples) == 32
    assert sum(e.hallucinated for e in examples) == 16
    assert all(e.context for e in examples)


def test_ragtruth_loader(tmp_path):
    _write_jsonl(
        tmp_path / "source_info.jsonl",
        [
            {
                "source_id": "s1",
                "task_type": "QA",
                "source_info": {"question": "q", "passages": "P1"},
            },
            {"source_id": "s2", "task_type": "Summary", "source_info": "Doc text"},
        ],
    )
    _write_jsonl(
        tmp_path / "response.jsonl",
        [
            {"id": "1", "source_id": "s1", "response": "a", "labels": [], "split": "test"},
            {
                "id": "2",
                "source_id": "s2",
                "response": "b",
                "labels": [{"text": "x"}],
                "split": "test",
            },
            {"id": "3", "source_id": "s2", "response": "c", "labels": [], "split": "train"},
        ],
    )
    examples = load_ragtruth(tmp_path)
    assert [(e.context, e.output, e.hallucinated) for e in examples] == [
        ("P1", "a", False),
        ("Doc text", "b", True),
    ]
    assert examples[0].source == "ragtruth/QA"


def test_halueval_loader(tmp_path):
    path = tmp_path / "qa.json"
    _write_jsonl(
        path, [{"knowledge": "k", "question": "q", "right_answer": "r", "hallucinated_answer": "h"}]
    )
    examples = load_halueval(path, "qa")
    assert [(e.output, e.hallucinated) for e in examples] == [("r", False), ("h", True)]
    with pytest.raises(ValueError, match="unknown HaluEval task"):
        load_halueval(path, "nope")


def test_aggrefact_local(tmp_path):
    path = tmp_path / "agg.jsonl"
    _write_jsonl(
        path,
        [
            {"dataset": "X", "doc": "d", "claim": "c", "label": 1},
            {"dataset": "X", "doc": "d", "claim": "c2", "label": 0},
        ],
    )
    assert [e.hallucinated for e in load_aggrefact(path)] == [False, True]


def test_load_requires_path():
    with pytest.raises(ValueError, match="needs a local path"):
        load("ragtruth")
    with pytest.raises(ValueError, match="unknown dataset"):
        load("nope", "x")


def test_sample_balances():
    examples = [GroundingExample("s", "c", str(i), i % 4 == 0) for i in range(100)]
    picked = sample(examples, 20)
    assert len(picked) == 20
    assert sum(e.hallucinated for e in picked) == 10
    assert sample(examples, None) == examples


def test_evaluate_perfect_and_inverted():
    examples = [GroundingExample("s", "c", "o", h) for h in (True, False, True, False)]
    assert evaluate(examples, [0.9, 0.1, 0.8, 0.2])["auc"] == 1.0
    assert evaluate(examples, [0.1, 0.9, 0.2, 0.8])["auc"] == 0.0


def test_parse_datasets():
    assert parse_datasets("local, ragtruth=/x/y ,") == [("local", None), ("ragtruth", "/x/y")]


def test_main_on_local(tmp_path):
    rows = main(scorers=["unsupported_claims", "novel_words"], output_dir=tmp_path)
    assert {r["scorer"] for r in rows} == {"unsupported_claims", "novel_words"}
    assert all(0.0 <= r["auc"] <= 1.0 for r in rows)
    assert "| local | unsupported_claims |" in (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert (tmp_path / "predictions.csv").exists()


def test_unavailable_scorer_is_skipped(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    rows = main(scorers=["judge", "novel_words"], output_dir=tmp_path)
    assert {r["scorer"] for r in rows} == {"novel_words"}


def test_cli_grounding_suite(tmp_path):
    from spaniq.cli.runner import main as cli_main

    cli_main(
        ["benchmark", "--suite", "grounding", "--scorers", "novel_words", "--output", str(tmp_path)]
    )
    assert (tmp_path / "summary.md").exists()
