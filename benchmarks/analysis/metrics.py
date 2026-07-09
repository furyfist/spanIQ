"""Accuracy metrics for the benchmark — precision, recall, F1, AUC, AP.

Pure functions over (true_label, score) pairs. The positive class is "bad" (the
failure we want to catch), per docs/plans/benchmark_v2_accuracy.md. A tool's
`score` is "how good the output looks" in [0, 1], so a LOW score means "likely
bad": the decision is `predicted = bad if score < threshold else good`.

No third-party dependency — the AUC and average-precision implementations are
small and self-contained so the benchmark stays reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass

POSITIVE = "bad"
NEGATIVE = "good"


@dataclass(frozen=True)
class Confusion:
    tp: int  # labeled bad, predicted bad
    fp: int  # labeled good, predicted bad
    fn: int  # labeled bad, predicted good
    tn: int  # labeled good, predicted good

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


def predict(score: float, threshold: float) -> str:
    """A low score (output looks bad) is flagged as the positive class 'bad'."""
    return NEGATIVE if score >= threshold else POSITIVE


def confusion(true_labels: list[str], scores: list[float], threshold: float) -> Confusion:
    tp = fp = fn = tn = 0
    for true, score in zip(true_labels, scores):
        pred = predict(score, threshold)
        if true == POSITIVE and pred == POSITIVE:
            tp += 1
        elif true == NEGATIVE and pred == POSITIVE:
            fp += 1
        elif true == POSITIVE and pred == NEGATIVE:
            fn += 1
        else:
            tn += 1
    return Confusion(tp=tp, fp=fp, fn=fn, tn=tn)


def precision(c: Confusion) -> float:
    return c.tp / (c.tp + c.fp) if (c.tp + c.fp) else 0.0


def recall(c: Confusion) -> float:
    return c.tp / (c.tp + c.fn) if (c.tp + c.fn) else 0.0


def f1(c: Confusion) -> float:
    p, r = precision(c), recall(c)
    return 2 * p * r / (p + r) if (p + r) else 0.0


def accuracy(c: Confusion) -> float:
    return (c.tp + c.tn) / c.total if c.total else 0.0


def _pos_scores(true_labels: list[str], scores: list[float]) -> tuple[list[float], list[float]]:
    """Split judge scores by true label. Because low score => bad (positive), the
    'positive score' used for ranking is (1 - score): high means more bad-like."""
    pos = [1.0 - s for t, s in zip(true_labels, scores) if t == POSITIVE]
    neg = [1.0 - s for t, s in zip(true_labels, scores) if t == NEGATIVE]
    return pos, neg


def roc_auc(true_labels: list[str], scores: list[float]) -> float:
    """Area under the ROC curve via the rank / Mann-Whitney-U identity.

    AUC = P(a random positive ranks above a random negative). Ties count 0.5.
    Threshold-free: measures whether the tool's score separates bad from good.
    """
    pos, neg = _pos_scores(true_labels, scores)
    if not pos or not neg:
        return float("nan")
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def average_precision(true_labels: list[str], scores: list[float]) -> float:
    """Area under the precision-recall curve (threshold-free), positive = bad.

    Ranks items by bad-likeness (1 - score) descending and averages precision at
    each true-positive hit.
    """
    ranked = sorted(
        ((1.0 - s, t) for t, s in zip(true_labels, scores)),
        key=lambda x: x[0],
        reverse=True,
    )
    n_pos = sum(1 for t in true_labels if t == POSITIVE)
    if n_pos == 0:
        return float("nan")
    tp = 0
    seen = 0
    ap = 0.0
    for _, true in ranked:
        seen += 1
        if true == POSITIVE:
            tp += 1
            ap += tp / seen  # precision at this recall step
    return ap / n_pos


def best_f1_threshold(true_labels: list[str], scores: list[float]) -> float:
    """Threshold on `score` that maximizes F1 for the positive ('bad') class.

    Candidate thresholds are the midpoints between sorted unique scores plus the
    extremes, so every distinct decision boundary is considered.
    """
    uniq = sorted(set(scores))
    if not uniq:
        return 0.5
    candidates = [uniq[0] - 1e-6, uniq[-1] + 1e-6]
    for a, b in zip(uniq, uniq[1:]):
        candidates.append((a + b) / 2)
    best_t, best = uniq[0], -1.0
    for t in candidates:
        score = f1(confusion(true_labels, scores, t))
        if score > best:
            best, best_t = score, t
    return best_t
