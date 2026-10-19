"""Two-sample statistics for comparing a baseline window of signals with a candidate window."""

from __future__ import annotations

import numpy as np

from spaniq.statistical.psi import compute_psi

_MAX_EFFECT = 99.0
_EPS = 1e-6


def effect_size(baseline: np.ndarray, candidate: np.ndarray) -> float:
    """Standardised mean difference (candidate minus baseline) over the pooled std.

    Capped at +/-99 so a signal that was constant in the baseline and moved in
    the candidate reads as a very large shift instead of infinity.
    """
    b, c = np.asarray(baseline, float), np.asarray(candidate, float)
    diff = float(c.mean() - b.mean())
    pooled = float(np.sqrt((b.var() + c.var()) / 2))
    if pooled < 1e-12:
        return 0.0 if abs(diff) < 1e-12 else float(np.sign(diff)) * _MAX_EFFECT
    return float(np.clip(diff / pooled, -_MAX_EFFECT, _MAX_EFFECT))


def psi(baseline: np.ndarray, candidate: np.ndarray, max_categories: int = 10) -> float:
    """PSI that also works for flags and counts: few distinct values are compared as categories."""
    b, c = np.asarray(baseline, float), np.asarray(candidate, float)
    values = np.unique(np.concatenate([b, c]))
    if len(values) <= max_categories:
        b_pct = np.array([(b == v).mean() for v in values]) + _EPS
        c_pct = np.array([(c == v).mean() for v in values]) + _EPS
        return float(np.sum((c_pct - b_pct) * np.log(c_pct / b_pct)))
    return compute_psi(b, c)


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Benjamini-Hochberg adjusted p-values, in the input order."""
    p = np.asarray(p_values, float)
    n = len(p)
    if n == 0:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(adjusted, 1.0)
    return out.tolist()


def _rbf_kernel(x: np.ndarray, y: np.ndarray, gamma: float) -> np.ndarray:
    sq = (x**2).sum(1)[:, None] + (y**2).sum(1)[None, :] - 2 * x @ y.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


def mmd_rbf(x: np.ndarray, y: np.ndarray, gamma: float | None = None) -> float:
    """Squared maximum mean discrepancy with an RBF kernel (biased estimator).

    gamma defaults to the median heuristic on the pooled sample.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if gamma is None:
        z = np.vstack([x, y])
        sq = (z**2).sum(1)[:, None] + (z**2).sum(1)[None, :] - 2 * z @ z.T
        med = float(np.median(sq[np.triu_indices(len(z), 1)])) if len(z) > 1 else 1.0
        gamma = 1.0 / max(med, 1e-12)
    kxx = _rbf_kernel(x, x, gamma).mean()
    kyy = _rbf_kernel(y, y, gamma).mean()
    kxy = _rbf_kernel(x, y, gamma).mean()
    return float(kxx + kyy - 2 * kxy)


def mmd_test(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 200,
    seed: int = 0,
    max_points: int = 300,
) -> tuple[float, float]:
    """(MMD^2, permutation p-value) for whether x and y come from the same distribution.

    Used on output embeddings, where a change in meaning shows up even when
    length and format stay the same. Inputs are subsampled to max_points each.
    """
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) > max_points:
        x = x[rng.choice(len(x), max_points, replace=False)]
    if len(y) > max_points:
        y = y[rng.choice(len(y), max_points, replace=False)]
    z = np.vstack([x, y])
    sq = (z**2).sum(1)[:, None] + (z**2).sum(1)[None, :] - 2 * z @ z.T
    med = float(np.median(sq[np.triu_indices(len(z), 1)])) if len(z) > 1 else 1.0
    gamma = 1.0 / max(med, 1e-12)
    observed = mmd_rbf(x, y, gamma)
    n = len(x)
    exceed = 0
    for _ in range(n_permutations):
        idx = rng.permutation(len(z))
        if mmd_rbf(z[idx[:n]], z[idx[n:]], gamma) >= observed:
            exceed += 1
    return observed, (exceed + 1) / (n_permutations + 1)
