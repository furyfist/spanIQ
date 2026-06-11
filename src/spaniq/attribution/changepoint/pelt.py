from __future__ import annotations

import numpy as np
import ruptures as rpt


def detect_changepoints(
    series: np.ndarray,
    penalty: float = 3.0,
    min_size: int = 10,
) -> list[int]:
    """Exact offline segmentation via PELT with RBF cost (Killick et al. 2012).
    Returns changepoint indices excluding the trailing n. min_size prevents
    spurious breaks on tiny segments. penalty=3.0 is a sensible starting point;
    BIC-style pen=3*log(n) is a data-driven fallback."""
    n = len(series)
    if n < min_size * 2:
        return []
    algo = rpt.Pelt(model="rbf", min_size=min_size).fit(series.reshape(-1, 1))
    breakpoints = algo.predict(pen=penalty)
    return [bp for bp in breakpoints if bp < n]


def bic_penalty(n: int) -> float:
    """BIC-style penalty: 3 * log(n), a data-driven fallback per ruptures docs."""
    return 3.0 * float(np.log(n))
