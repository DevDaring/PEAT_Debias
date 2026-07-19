"""
PEAT — Instance-level significance and effect-size tests (WP-B).

Reviewers #2/#3/#4 asked for formal significance testing and effect sizes rather
than three-seed Student-t intervals. The statistical power for a stereotype-score
comparison lives in the 1,508 *paired* per-instance outcomes, not in the seed
count, so these tests operate on the per-pair `prefers_stereo` vectors written to
results/raw/.../ss_<model>.csv. All functions are pure (numpy only; no GPU) and
are wired into aggregation.py to emit table_significance.csv / table_effect_sizes.csv.

# References:
#   Dror et al., "The Hitchhiker's Guide to Testing Statistical Significance in
#     NLP", ACL 2018 — non-parametric paired tests are the documented best
#     practice for paired binary NLP outcomes.
#   McNemar, "Note on the sampling error of the difference between correlated
#     proportions or percentages", Psychometrika 1947.
#   Cohen, "Statistical Power Analysis for the Behavioral Sciences", 1988 (h).
#   Holm, "A simple sequentially rejective multiple test procedure", 1979.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# McNemar's exact test on discordant pairs
# ---------------------------------------------------------------------------
def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact (binomial) McNemar test for two paired binary vectors.

    a, b: per-pair `prefers_stereo` (0/1) for methods A and B on the SAME pairs.
    Tests whether the two methods disagree symmetrically on discordant pairs.
    Returns discordant counts, the exact two-sided p-value, and the odds ratio.
    """
    a = np.asarray(a).astype(int); b = np.asarray(b).astype(int)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    b01 = int(np.sum((a == 0) & (b == 1)))   # A neutral, B stereo
    b10 = int(np.sum((a == 1) & (b == 0)))   # A stereo, B neutral
    n_disc = b01 + b10
    if n_disc == 0:
        return {"b01": 0, "b10": 0, "n_discordant": 0, "p_value": 1.0, "odds_ratio": float("nan")}
    k = min(b01, b10)
    # two-sided exact binomial p at prob 0.5
    p = 2.0 * sum(math.comb(n_disc, i) for i in range(0, k + 1)) / (2 ** n_disc)
    p = min(1.0, p)
    odds = (b10 / b01) if b01 > 0 else float("inf")
    return {"b01": b01, "b10": b10, "n_discordant": n_disc, "p_value": p, "odds_ratio": odds}


# ---------------------------------------------------------------------------
# Paired permutation test on the SS difference
# ---------------------------------------------------------------------------
def permutation_test_ss(a: np.ndarray, b: np.ndarray, n_resamples: int = 10000,
                        seed: int = 42) -> dict:
    """Two-sided paired permutation test for SS(A) - SS(B).

    Under H0 the per-pair labels of A and B are exchangeable; each resample flips
    a random subset of pairs. SS = 100 * mean(prefers_stereo). Returns the
    observed difference and the permutation p-value.
    """
    rng = np.random.default_rng(seed)
    a = np.asarray(a).astype(float); b = np.asarray(b).astype(float)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    obs = 100.0 * (a.mean() - b.mean())
    d = a - b
    count = 0
    for _ in range(n_resamples):
        signs = rng.choice([1.0, -1.0], size=n)
        stat = 100.0 * (signs * d).mean()
        if abs(stat) >= abs(obs) - 1e-12:
            count += 1
    return {"observed_diff": obs, "p_value": (count + 1) / (n_resamples + 1), "n_pairs": n}


# ---------------------------------------------------------------------------
# Paired bootstrap CI on the SS difference
# ---------------------------------------------------------------------------
def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, n_resamples: int = 1000,
                        seed: int = 42, alpha: float = 0.05) -> dict:
    """Percentile bootstrap CI for SS(A) - SS(B), resampling pairs with replacement."""
    rng = np.random.default_rng(seed)
    a = np.asarray(a).astype(float); b = np.asarray(b).astype(float)
    n = min(len(a), len(b)); a, b = a[:n], b[:n]
    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        diffs[i] = 100.0 * (a[idx].mean() - b[idx].mean())
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff_mean": float(100.0 * (a.mean() - b.mean())),
            "ci_lo": float(lo), "ci_hi": float(hi),
            "excludes_zero": bool(lo > 0 or hi < 0)}


# ---------------------------------------------------------------------------
# Hierarchical bootstrap over (seeds, pairs)
# ---------------------------------------------------------------------------
def hierarchical_bootstrap_ss(per_seed_vectors: list[np.ndarray], n_resamples: int = 1000,
                              seed: int = 42, alpha: float = 0.05) -> dict:
    """Two-level bootstrap for a single method's SS: resample seeds, then pairs.

    per_seed_vectors: list of per-pair prefers_stereo arrays, one per seed.
    """
    rng = np.random.default_rng(seed)
    vecs = [np.asarray(v).astype(float) for v in per_seed_vectors if len(v) > 0]
    if not vecs:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan")}
    S = len(vecs)
    means = np.empty(n_resamples)
    for i in range(n_resamples):
        s = vecs[rng.integers(0, S)]
        idx = rng.integers(0, len(s), size=len(s))
        means[i] = 100.0 * s[idx].mean()
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    grand = 100.0 * np.mean([v.mean() for v in vecs])
    return {"mean": float(grand), "ci_lo": float(lo), "ci_hi": float(hi)}


# ---------------------------------------------------------------------------
# Effect sizes and flip counts
# ---------------------------------------------------------------------------
def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h between two proportions (e.g. stereotype-preference rates)."""
    phi = lambda p: 2.0 * math.asin(math.sqrt(min(max(p, 0.0), 1.0)))
    return phi(p1) - phi(p2)


def flip_counts(base: np.ndarray, method: np.ndarray) -> dict:
    """Count per-pair preference changes from Base to the method.

    'de-biased' = a pair that preferred the stereotype under Base but no longer
    does under the method (the practical-significance quantity Reviewer #3 asked
    for: how many of the 1,508 decisions change).
    """
    base = np.asarray(base).astype(int); method = np.asarray(method).astype(int)
    n = min(len(base), len(method)); base, method = base[:n], method[:n]
    debiased = int(np.sum((base == 1) & (method == 0)))
    worsened = int(np.sum((base == 0) & (method == 1)))
    return {"n_pairs": n, "debiased_flips": debiased, "worsened_flips": worsened,
            "net_debiased": debiased - worsened,
            "cohens_h": cohens_h(method.mean(), base.mean())}


# ---------------------------------------------------------------------------
# Holm–Bonferroni multiple-comparison correction
# ---------------------------------------------------------------------------
def holm_bonferroni(pvalues: dict, alpha: float = 0.05) -> dict:
    """Holm step-down correction. Input {label: p}; returns {label: (p_adj, reject)}."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out = {}
    running_max = 0.0
    for rank, (label, p) in enumerate(items):
        p_adj = min(1.0, (m - rank) * p)
        running_max = max(running_max, p_adj)   # enforce monotonicity
        out[label] = {"p_raw": p, "p_adj": running_max, "reject": running_max < alpha}
    return out


# ---------------------------------------------------------------------------
# CSV convenience loader
# ---------------------------------------------------------------------------
def load_prefers_stereo(csv_path) -> np.ndarray:
    """Read the per-pair `prefers_stereo` column from an ss_<model>.csv file."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    if "prefers_stereo" not in df.columns:
        return np.array([])
    return df["prefers_stereo"].to_numpy().astype(int)
