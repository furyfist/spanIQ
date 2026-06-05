from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Iterator
from uuid import uuid4

from spaniq.monitor.collectors.base import BaseCollector
from spaniq.monitor.trace import Trace

logger = logging.getLogger(__name__)


class FileCollector(BaseCollector):
    """Reads traces from a JSONL file.

    Each line must be a JSON object with at least 'input' and 'output' keys.
    Optional keys: 'trace_id', 'timestamp', 'metadata'.

    In tail mode the collector continues polling for new lines after the file
    is exhausted, behaving like ``tail -f``.
    """

    def __init__(self, path: str, tail: bool = False, interval: float = 1.0) -> None:
        self.path = path
        self.tail = tail
        self.interval = interval

    def name(self) -> str:
        return f"file:{self.path}"

    def collect(self) -> Iterator[Trace]:
        with open(self.path, encoding="utf-8") as fh:
            for line in fh:
                trace = self._parse_line(line)
                if trace is not None:
                    yield trace
            if self.tail:
                while True:
                    line = fh.readline()
                    if line:
                        trace = self._parse_line(line)
                        if trace is not None:
                            yield trace
                    else:
                        time.sleep(self.interval)

    def _parse_line(self, line: str) -> Trace | None:
        line = line.strip()
        if not line:
            return None
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("FileCollector: skipping malformed line: %r", line[:80])
            return None
        if not isinstance(obj, dict):
            logger.warning("FileCollector: skipping non-object line: %r", line[:80])
            return None
        if "input" not in obj or "output" not in obj:
            logger.warning("FileCollector: skipping line missing input/output: %r", line[:80])
            return None
        return Trace(
            input=obj["input"],
            output=obj["output"],
            trace_id=obj.get("trace_id") or str(uuid4()),
            timestamp=obj.get("timestamp") or datetime.now(timezone.utc).isoformat(),
            metadata=obj.get("metadata"),
        )
