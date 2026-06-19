from __future__ import annotations

import numpy as np
import ruptures as rpt


def bic_penalty(n: int) -> float:
    """BIC-style penalty: 3 * log(n) per ruptures docs."""
    return 3.0 * float(np.log(n))


def detect_changepoints(
    series: np.ndarray,
    penalty: float | None = None,
    min_size: int = 10,
) -> list[int]:
    """Exact offline segmentation via PELT with RBF cost (Killick et al. 2012).
    Returns changepoint indices excluding the trailing n. Defaults to BIC penalty
    (3*log(n)) which scales with series length and prevents spurious early breaks."""
    n = len(series)
    if n < min_size * 2:
        return []
    pen = penalty if penalty is not None else bic_penalty(n)
    algo = rpt.Pelt(model="rbf", min_size=min_size).fit(series.reshape(-1, 1))
    breakpoints = algo.predict(pen=pen)
    return [bp for bp in breakpoints if bp < n]
