"""The judge retry policy retries transient failures and never fabricates a score."""

from __future__ import annotations

import asyncio

import pytest

from benchmarks.runners._judge import JudgeError, awith_retry, score_from_ten, with_retry


def test_with_retry_recovers_from_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("429 rate limited")
        return "ok"

    assert with_retry(flaky, base_delay=0.0) == "ok"
    assert calls["n"] == 3


def test_with_retry_raises_judge_error_when_exhausted():
    def always_fails():
        raise ConnectionError("429 rate limited")

    with pytest.raises(JudgeError, match="429"):
        with_retry(always_fails, attempts=3, base_delay=0.0)


def test_awith_retry_raises_judge_error_when_exhausted():
    async def always_fails():
        raise ConnectionError("down")

    with pytest.raises(JudgeError):
        asyncio.run(awith_retry(always_fails, attempts=2, base_delay=0.0))


def test_score_from_ten_parses_and_rejects():
    assert score_from_ten("8") == pytest.approx(0.8)
    assert score_from_ten(" 10 \n") == pytest.approx(1.0)
    with pytest.raises(JudgeError):
        score_from_ten("")
    with pytest.raises(JudgeError):
        score_from_ten("great answer")


def test_permanent_client_errors_are_not_retried():
    class NotFound(Exception):
        status_code = 404

    calls = {"n": 0}

    def missing_model():
        calls["n"] += 1
        raise NotFound("model_not_found")

    with pytest.raises(JudgeError, match="1 attempt"):
        with_retry(missing_model, base_delay=0.0)
    assert calls["n"] == 1


def test_rate_limits_are_retried():
    class RateLimited(Exception):
        status_code = 429

    calls = {"n": 0}

    def limited():
        calls["n"] += 1
        raise RateLimited("slow down")

    with pytest.raises(JudgeError):
        with_retry(limited, attempts=3, base_delay=0.0)
    assert calls["n"] == 3


def test_score_from_ten_finds_score_in_prose():
    assert score_from_ten("Score: 7") == pytest.approx(0.7)
