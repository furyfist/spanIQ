import numpy as np
from scipy.spatial.distance import jensenshannon


def compute_js(baseline: np.ndarray, current: np.ndarray, n_bins: int = 50) -> float:
    """JS distance between two distributions. Bounded [0, 1].

    Uses shared bin edges across both arrays so the histograms are comparable.
    Returns JS distance (square root of divergence), not divergence itself.
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    if len(baseline) == 0 or len(current) == 0:
        raise ValueError("baseline and current must be non-empty")

    combined_min = min(baseline.min(), current.min())
    combined_max = max(baseline.max(), current.max())

    if combined_min == combined_max:
        return 0.0

    bin_edges = np.linspace(combined_min, combined_max, n_bins + 1)

    eps = 1e-6
    baseline_hist, _ = np.histogram(baseline, bins=bin_edges)
    current_hist, _ = np.histogram(current, bins=bin_edges)

    baseline_dist = (baseline_hist + eps) / (baseline_hist.sum() + eps * len(baseline_hist))
    current_dist = (current_hist + eps) / (current_hist.sum() + eps * len(current_hist))

    return float(jensenshannon(baseline_dist, current_dist))
