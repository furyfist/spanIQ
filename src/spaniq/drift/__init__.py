"""Distribution-level change detection over per-output signals."""

from spaniq.drift.detector import (
    EMBEDDING,
    DriftReport,
    SignalDrift,
    WindowDriftDetector,
    compare,
)
from spaniq.drift.diff import DiffResult, ReviewItem, UnitDiff, diff_traces, load_traces
from spaniq.drift.stats import benjamini_hochberg, effect_size, mmd_rbf, mmd_test, psi

__all__ = [
    "EMBEDDING",
    "DiffResult",
    "DriftReport",
    "ReviewItem",
    "SignalDrift",
    "UnitDiff",
    "WindowDriftDetector",
    "benjamini_hochberg",
    "compare",
    "diff_traces",
    "effect_size",
    "load_traces",
    "mmd_rbf",
    "mmd_test",
    "psi",
]
