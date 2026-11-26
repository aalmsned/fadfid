"""
Evaluation metrics: precision/recall, rates with bootstrap CIs, McNemar's test.
"""

import math

import numpy as np


def bootstrapCi(values, nBoot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean and 95% bootstrap confidence interval."""
    arr = np.asarray([v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if len(arr) == 0:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(nBoot, len(arr)), replace=True).mean(axis=1)
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def precisionRecall(gold: list[int], pred: list[int], threshold: int = 2) -> dict:
    pairs = [(g, p) for g, p in zip(gold, pred) if g is not None and p is not None]
    if not pairs:
        return {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "n": 0}
    tp = sum(g >= threshold and p >= threshold for g, p in pairs)
    fp = sum(g < threshold and p >= threshold for g, p in pairs)
    fn = sum(g >= threshold and p < threshold for g, p in pairs)
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if precision and recall and not math.isnan(precision + recall) else float("nan")
    return {"precision": precision, "recall": recall, "f1": f1, "n": len(pairs)}


def mcnemar(a: list[bool], b: list[bool]) -> dict:
    """Exact McNemar test on paired yes/no outcomes (e.g. unsafe in condition A vs B)."""
    onlyA = sum(x and not y for x, y in zip(a, b))
    onlyB = sum(y and not x for x, y in zip(a, b))
    n = onlyA + onlyB
    if n == 0:
        return {"onlyA": 0, "onlyB": 0, "p": 1.0}
    k = min(onlyA, onlyB)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"onlyA": onlyA, "onlyB": onlyB, "p": p}


def cohenKappa(a: list, b: list) -> float:
    """Agreement between two labellers (or a human and the LLM judge)."""
    labels = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return float("nan")
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def weightedKappa(a: list[int], b: list[int], minScore: int, maxScore: int) -> float:
    """Quadratic-weighted kappa for ordered scores (crisis tiers 0-4, empathy 0-6)."""
    pairs = [(int(x), int(y)) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    k = maxScore - minScore + 1
    observed = np.zeros((k, k))
    for x, y in pairs:
        observed[x - minScore, y - minScore] += 1
    observed /= len(pairs)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    idx = np.arange(k)
    weights = (idx[:, None] - idx[None, :]) ** 2 / (k - 1) ** 2
    denom = (weights * expected).sum()
    return float(1 - (weights * observed).sum() / denom) if denom > 0 else 1.0
