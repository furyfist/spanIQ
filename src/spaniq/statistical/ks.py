from typing import NamedTuple

import numpy as np
from scipy import stats


class KSResult(NamedTuple):
    statistic: float
    p_value: float
    significant: bool


def compute_ks(baseline: np.ndarray, current: np.ndarray, alpha: float = 0.05) -> KSResult:
    """Two-sample KS test between two distributions.

    significant=True means the distributions are statistically different.
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    if len(baseline) == 0 or len(current) == 0:
        raise ValueError("baseline and current must be non-empty")

    result = stats.ks_2samp(baseline, current)
    return KSResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        significant=bool(result.pvalue < alpha),
    )
