"""Distribution-level change detection over per-output signals."""

from spaniq.drift.detector import (
    EMBEDDING,
    DriftReport,
    SignalDrift,
    WindowDriftDetector,
    compare,
)
from spaniq.drift.stats import benjamini_hochberg, effect_size, mmd_rbf, mmd_test, psi

__all__ = [
    "EMBEDDING",
    "DriftReport",
    "SignalDrift",
    "WindowDriftDetector",
    "benjamini_hochberg",
    "compare",
    "effect_size",
    "mmd_rbf",
    "mmd_test",
    "psi",
]
