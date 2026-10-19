from types import SimpleNamespace

import pytest

from spaniq.escalation import (
    Candidate,
    GroqJudge,
    JudgeBudget,
    JudgeError,
    escalate,
    parse_reply,
    select,
)
from spaniq.escalation.judge import make_judge, with_retry
from spaniq.monitor.timeline_store import TimelineStore


class FakeJudge:
    name = "fake"

    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def judge(self, input, output, context):  # noqa: A002
        self.calls.append(output)
        if output in self.fail_on:
            raise JudgeError("rate limited")
        return (0.1, "refuses to answer") if "sorry" in output else (0.9, "supported")


def test_budget_caps_calls_per_day(tmp_path):
    day = ["2026-10-20"]
    budget = JudgeBudget(str(tmp_path / "b.db"), daily_limit=5, clock=lambda: day[0])
    assert budget.try_spend(3) == 3
    assert budget.try_spend(3) == 2
    assert budget.try_spend(1) == 0
    assert budget.remaining() == 0
    day[0] = "2026-10-21"
    assert budget.remaining() == 5
    assert budget.try_spend(0) == 0


def test_budget_is_shared_through_the_database(tmp_path):
    db = str(tmp_path / "b.db")
    a = JudgeBudget(db, 4, clock=lambda: "d")
    b = JudgeBudget(db, 4, clock=lambda: "d")
    assert a.try_spend(3) == 3
    assert b.try_spend(3) == 1


def test_budget_rejects_negative_limit(tmp_path):
    with pytest.raises(ValueError):
        JudgeBudget(str(tmp_path / "b.db"), -1)


def test_parse_reply():
    assert parse_reply("10\nAll claims supported.") == (1.0, "All claims supported.")
    assert parse_reply("Score: 1") == (0.0, "")
    with pytest.raises(JudgeError):
        parse_reply("no idea")


def test_with_retry_backs_off_then_raises():
    sleeps = []

    def flaky():
        raise RuntimeError("503")

    with pytest.raises(JudgeError, match="503"):
        with_retry(flaky, attempts=3, base_delay=1.0, sleep=sleeps.append)
    assert sleeps == [1.0, 2.0]


def test_with_retry_stops_on_permanent_error():
    class BadRequest(Exception):
        status_code = 400

    calls = []

    def call():
        calls.append(1)
        raise BadRequest("model not found")

    with pytest.raises(JudgeError):
        with_retry(call, attempts=5, sleep=lambda s: None)
    assert len(calls) == 1


def test_groq_judge_with_fake_client():
    class Client:
        def __init__(self):
            self.chat = self
            self.completions = self
            self.prompts = []

        def create(self, model, messages, max_tokens):
            self.prompts.append(messages[0]["content"])
            msg = SimpleNamespace(content="3\nInvents a date.")
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    client = Client()
    judge = GroqJudge(model="m", client=client, max_context_chars=10)
    score, reason = judge.judge("q", "a", "0123456789ABCDEF")
    assert score == pytest.approx(2 / 9)
    assert reason == "Invents a date."
    assert "0123456789" in client.prompts[0] and "ABCDEF" not in client.prompts[0]


def test_groq_judge_without_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises((OSError, ImportError)):
        GroqJudge().judge("q", "a", None)


def test_make_judge():
    assert make_judge("groq").name == "groq"
    with pytest.raises(ValueError, match="unknown judge"):
        make_judge("nope")


def _cands(n):
    return [Candidate(f"t{i}", "q", f"out {i}", priority=float(i)) for i in range(n)]


def test_select_top_priority_and_exploration():
    cands = _cands(10)
    top = select(cands, 3)
    assert [c.trace_id for c in top] == ["t9", "t8", "t7"]
    mixed = select(cands, 4, explore=0.5, seed=1)
    assert [c.trace_id for c in mixed[:2]] == ["t9", "t8"]
    assert len(mixed) == 4 and len({c.trace_id for c in mixed}) == 4
    assert select(cands, 0) == []
    assert select([], 3) == []


def test_select_fills_with_random_when_nothing_moved():
    flat = [Candidate(f"t{i}", "q", "o") for i in range(5)]
    assert len(select(flat, 3)) == 3


def test_escalate_respects_budget_and_records(tmp_path):
    db = str(tmp_path / "e.db")
    budget = JudgeBudget(db, 2, clock=lambda: "d")
    judge = FakeJudge()
    cands = [
        Candidate("a", "q", "I'm sorry", priority=3),
        Candidate("b", "q", "fine", priority=2),
        Candidate("c", "q", "fine", priority=1),
    ]
    verdicts = escalate(cands, judge, budget, timeline=TimelineStore(db), timestamp="2026")
    assert [v.trace_id for v in verdicts] == ["a", "b"]
    assert not verdicts[0].passed and verdicts[1].passed
    rows = TimelineStore(db).query("judge", last_n=10, ascending=True)
    assert [(r.trace_id, r.passed) for r in rows] == [("a", False), ("b", True)]
    assert rows[0].metadata["reason"] == "refuses to answer"


def test_escalate_skips_failed_calls_without_scoring():
    judge = FakeJudge(fail_on={"bad"})
    cands = [Candidate("x", "q", "bad"), Candidate("y", "q", "fine")]
    verdicts = escalate(cands, judge)
    assert [v.trace_id for v in verdicts] == ["y"]


def test_monitor_escalates_on_drift(tmp_path):
    from tests.test_monitor_drift import GOOD, REFUSAL, _monitor

    outputs = [GOOD[i % 4] for i in range(60)] + [REFUSAL] * 80
    budget = JudgeBudget(str(tmp_path / "budget.db"), 3, clock=lambda: "d")
    judge = FakeJudge()
    monitor, db = _monitor(
        tmp_path,
        outputs,
        drift_window=40,
        drift_stride=10,
        judge=judge,
        judge_budget=budget,
        judge_per_incident=5,
    )
    report = monitor.run()
    assert len(report.drift_incidents) == 1
    assert len(report.verdicts) == 3
    assert all(not v.passed for v in report.verdicts)
    assert len(TimelineStore(db).query("judge", last_n=10)) == 3


def test_diff_cli_with_judge(tmp_path, monkeypatch, capsys):
    import json

    import spaniq.escalation as esc
    from spaniq.cli.runner import main
    from tests.test_diff import REFUSAL, _run, _write

    monkeypatch.setattr(esc, "make_judge", lambda name: FakeJudge())
    base, bad = tmp_path / "b.jsonl", tmp_path / "x.jsonl"
    _write(base, _run(40))
    _write(bad, _run(40, 100, answer=REFUSAL))
    main(
        [
            "diff",
            str(base),
            str(bad),
            "--no-fail",
            "--json",
            "--judge",
            "groq",
            "--judge-budget",
            "4",
            "--top",
            "6",
            "--db",
            str(tmp_path / "j.db"),
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert len(out["verdicts"]) == 4
    assert out["verdicts"][0]["judge"] == "fake"
