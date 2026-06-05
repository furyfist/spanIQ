from spaniq.monitor.baseline_store import BaselineStore
from spaniq.monitor.trace import Trace

__all__ = [
    "BaselineStore",
    "Trace",
]


def __getattr__(name: str):
    if name == "Monitor":
        from spaniq.monitor.monitor import Monitor
        return Monitor
    if name == "TimelineStore":
        from spaniq.monitor.timeline_store import TimelineStore
        return TimelineStore
    if name == "SDKCollector":
        from spaniq.monitor.collectors.sdk import SDKCollector
        return SDKCollector
    raise AttributeError(f"module 'spaniq.monitor' has no attribute {name!r}")
