from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from spaniq.attribution.component import ComponentKind, ComponentSpan


@dataclass
class Trace:
    """A single LLM interaction captured from production."""

    input: str
    output: str
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict | None = None
    components: list[ComponentSpan] | None = None

    @classmethod
    def from_dict(cls, obj: dict) -> Trace:
        """A Trace from one JSONL record; raises KeyError without 'input' and 'output'."""
        spans = []
        for c in obj.get("components") or []:
            if not isinstance(c, dict) or "name" not in c or "output" not in c:
                continue
            try:
                kind = ComponentKind(c.get("kind", "default"))
            except ValueError:
                kind = ComponentKind.DEFAULT
            spans.append(
                ComponentSpan(
                    name=c["name"],
                    kind=kind,
                    output=c["output"],
                    latency_ms=c.get("latency_ms"),
                    error=bool(c.get("error", False)),
                    metadata=c.get("metadata"),
                )
            )
        return cls(
            input=obj["input"],
            output=obj["output"],
            trace_id=obj.get("trace_id") or str(uuid4()),
            timestamp=obj.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            metadata=obj.get("metadata"),
            components=spans or None,
        )
