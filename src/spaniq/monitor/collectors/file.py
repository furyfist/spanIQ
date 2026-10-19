from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator

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
            # In tail mode a writer may be mid-line; hold the fragment until its
            # newline arrives instead of parsing it as a malformed line.
            pending = ""
            while True:
                line = fh.readline()
                if line:
                    pending += line
                    if self.tail and not pending.endswith("\n"):
                        continue
                    trace = self._parse_line(pending)
                    pending = ""
                    if trace is not None:
                        yield trace
                elif self.tail:
                    time.sleep(self.interval)
                else:
                    return

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
        return Trace.from_dict(obj)
