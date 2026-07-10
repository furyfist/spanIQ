from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from uuid import uuid4

from spaniq.attribution.component import ComponentSpan


@dataclass
class Trace:
    """A single LLM interaction captured from production."""

    input: str
    output: str
    trace_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict | None = None
    components: list[ComponentSpan] | None = None
