import re

from spaniq.monitor.trace import Trace


def test_defaults_populated():
    t = Trace(input="hello", output="world")
    assert t.input == "hello"
    assert t.output == "world"
    assert t.trace_id  # non-empty
    assert t.timestamp  # non-empty
    assert t.metadata is None


def test_trace_id_is_uuid():
    t = Trace(input="a", output="b")
    uuid_re = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
    )
    assert uuid_re.match(t.trace_id)


def test_timestamp_is_utc_iso():
    t = Trace(input="a", output="b")
    from datetime import datetime
    dt = datetime.fromisoformat(t.timestamp)
    assert dt.tzinfo is not None


def test_unique_ids():
    t1 = Trace(input="x", output="y")
    t2 = Trace(input="x", output="y")
    assert t1.trace_id != t2.trace_id


def test_custom_fields():
    t = Trace(
        input="q",
        output="a",
        trace_id="custom-id",
        timestamp="2024-01-01T00:00:00+00:00",
        metadata={"model": "gpt-4"},
    )
    assert t.trace_id == "custom-id"
    assert t.metadata == {"model": "gpt-4"}
