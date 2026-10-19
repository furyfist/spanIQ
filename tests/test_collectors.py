from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from spaniq.monitor.collectors.file import FileCollector
from spaniq.monitor.collectors.sdk import SDKCollector

# ── FileCollector ──────────────────────────────────────────────────────────────


def _write_jsonl(path, records: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")


def test_file_collector_batch(tmp_path):
    p = tmp_path / "traces.jsonl"
    records = [{"input": f"q{i}", "output": f"a{i}"} for i in range(5)]
    _write_jsonl(p, records)
    collector = FileCollector(str(p))
    traces = list(collector.collect())
    assert len(traces) == 5


def test_file_collector_parses_fields(tmp_path):
    p = tmp_path / "t.jsonl"
    _write_jsonl(
        p,
        [
            {"input": "hi", "output": "hello", "trace_id": "abc123", "metadata": {"k": "v"}},
        ],
    )
    collector = FileCollector(str(p))
    traces = list(collector.collect())
    t = traces[0]
    assert t.input == "hi"
    assert t.output == "hello"
    assert t.trace_id == "abc123"
    assert t.metadata == {"k": "v"}


def test_file_collector_handles_malformed(tmp_path, caplog):
    p = tmp_path / "bad.jsonl"
    with open(p, "w") as fh:
        fh.write("not json\n")
        fh.write(json.dumps({"input": "ok", "output": "fine"}) + "\n")
    import logging

    with caplog.at_level(logging.WARNING):
        traces = list(FileCollector(str(p)).collect())
    assert len(traces) == 1
    assert traces[0].input == "ok"


def test_file_collector_skips_missing_fields(tmp_path):
    p = tmp_path / "partial.jsonl"
    _write_jsonl(
        p,
        [
            {"input": "only input"},  # missing output → skip
            {"output": "only output"},  # missing input → skip
            {"input": "both", "output": "ok"},
        ],
    )
    traces = list(FileCollector(str(p)).collect())
    assert len(traces) == 1
    assert traces[0].input == "both"


def test_file_collector_generates_trace_id_when_absent(tmp_path):
    p = tmp_path / "noid.jsonl"
    _write_jsonl(p, [{"input": "q", "output": "a"}])
    traces = list(FileCollector(str(p)).collect())
    assert traces[0].trace_id  # non-empty, auto-generated


# ── SDKCollector ──────────────────────────────────────────────────────────────


def test_sdk_collector_ingest_and_collect():
    collector = SDKCollector()
    collector.ingest(input="hello", output="world")
    collector.ingest(input="foo", output="bar")
    collector.close()
    traces = list(collector.collect())
    assert len(traces) == 2
    assert traces[0].input == "hello"
    assert traces[1].input == "foo"


def test_sdk_collector_close_stops_iteration():
    collector = SDKCollector()
    collector.close()
    traces = list(collector.collect())
    assert traces == []


def test_sdk_collector_ingest_sets_fields():
    collector = SDKCollector()
    collector.ingest(input="q", output="a", trace_id="t1", metadata={"m": 1})
    collector.close()
    traces = list(collector.collect())
    assert traces[0].trace_id == "t1"
    assert traces[0].metadata == {"m": 1}


# ── LangfuseCollector ─────────────────────────────────────────────────────────


def test_langfuse_collector_raises_without_dep():
    with patch.dict("sys.modules", {"langfuse": None}):
        from spaniq.monitor.collectors.langfuse import LangfuseCollector

        collector = LangfuseCollector(poll_interval=0)
        with pytest.raises(ImportError, match="pip install spaniq\\[langfuse\\]"):
            next(collector.collect())


def test_langfuse_collector_yields_traces_from_mock():
    obs1 = MagicMock()
    obs1.input = "what is 2+2?"
    obs1.output = "4"
    obs1.trace_id = "trace-abc"
    obs1.start_time = None
    obs1.model = "gpt-4"
    obs1.id = "obs-1"

    mock_resp = MagicMock()
    mock_resp.data = [obs1]

    mock_client = MagicMock()
    mock_client.api.observations.get_many.return_value = mock_resp

    collected: list = []
    call_count = 0

    def _fake_sleep(n):
        nonlocal call_count
        call_count += 1
        if call_count >= 1:
            raise RuntimeError("stop polling")

    # get_client is imported inside collect(), so patch via langfuse module
    mock_langfuse_module = MagicMock()
    mock_langfuse_module.get_client.return_value = mock_client

    from spaniq.monitor.collectors.langfuse import LangfuseCollector

    collector = LangfuseCollector(poll_interval=0)

    with (
        patch.dict("sys.modules", {"langfuse": mock_langfuse_module}),
        patch("spaniq.monitor.collectors.langfuse.time.sleep", side_effect=_fake_sleep),
    ):
        try:
            for t in collector.collect():
                collected.append(t)
        except RuntimeError:
            pass

    assert len(collected) == 1
    assert collected[0].input == "what is 2+2?"
    assert collected[0].output == "4"
    assert collected[0].trace_id == "trace-abc"
    assert collected[0].metadata["model"] == "gpt-4"


def _obs(i, start):
    from datetime import datetime, timezone

    o = MagicMock()
    o.id = f"obs-{i}"
    o.trace_id = f"trace-{i}"
    o.input = [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": f"question {i}"},
    ]
    o.output = {"role": "assistant", "content": f"answer {i}"}
    o.model = "m"
    o.start_time = datetime(2026, 1, 1, 0, 0, start, tzinfo=timezone.utc)
    return o


def _resp(data):
    r = MagicMock()
    r.data = data
    r.meta.total_pages = None
    return r


def test_langfuse_reads_every_page_and_extracts_message_text():
    from spaniq.monitor.collectors.langfuse import LangfuseCollector

    collector = LangfuseCollector(poll_interval=0, limit=2)
    client = MagicMock()
    pages = {1: [_obs(1, 1), _obs(2, 2)], 2: [_obs(3, 3), _obs(4, 4)], 3: [_obs(5, 5)]}
    client.api.observations.get_many.side_effect = lambda **kw: _resp(pages[kw["page"]])

    traces = list(collector._new_traces(collector._fetch_all(client)))
    assert [t.trace_id for t in traces] == [f"trace-{i}" for i in range(1, 6)]
    assert traces[0].input == "question 1"
    assert traces[0].output == "answer 1"


def test_langfuse_cursor_does_not_repeat_boundary_observation():
    from spaniq.monitor.collectors.langfuse import LangfuseCollector

    collector = LangfuseCollector(poll_interval=0)
    first = list(collector._new_traces([_obs(1, 1), _obs(2, 2)]))
    # the API's from_start_time is inclusive, so obs-2 comes back with a new one
    second = list(collector._new_traces([_obs(2, 2), _obs(3, 2), _obs(4, 3)]))
    assert [t.trace_id for t in first] == ["trace-1", "trace-2"]
    assert [t.trace_id for t in second] == ["trace-3", "trace-4"]


def test_file_collector_tail_waits_for_partial_line(tmp_path):
    import json
    import threading

    path = tmp_path / "live.jsonl"
    record = json.dumps({"input": "q", "output": "a"})
    path.write_text(record[:10], encoding="utf-8")

    def finish_line():
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(record[10:] + "\n")

    collector = FileCollector(str(path), tail=True, interval=0.05)
    timer = threading.Timer(0.2, finish_line)
    timer.start()
    trace = next(collector.collect())
    timer.join()
    assert trace.input == "q"
