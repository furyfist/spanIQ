import numpy as np


def compute_psi(baseline: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Population Stability Index between two distributions.

    PSI < 0.1: no significant drift
    PSI 0.1-0.25: moderate drift
    PSI > 0.25: significant drift
    """
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)

    if len(baseline) == 0 or len(current) == 0:
        raise ValueError("baseline and current must be non-empty")

    bin_edges = np.percentile(baseline, np.linspace(0, 100, n_bins + 1))
    bin_edges = np.unique(bin_edges)

    baseline_counts, _ = np.histogram(baseline, bins=bin_edges)
    current_counts, _ = np.histogram(current, bins=bin_edges)

    eps = 1e-6
    baseline_pct = (baseline_counts + eps) / (baseline_counts.sum() + eps * len(baseline_counts))
    current_pct = (current_counts + eps) / (current_counts.sum() + eps * len(current_counts))

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)
