from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ComponentKind(str, Enum):
    RETRIEVAL = "retrieval"
    TOOL = "execute_tool"
    GENERATION = "chat"
    AGENT = "invoke_agent"
    DEFAULT = "default"


@dataclass
class ComponentSpan:
    name: str
    kind: ComponentKind
    output: str
    latency_ms: float | None = None
    error: bool = False
    metadata: dict | None = field(default=None)
