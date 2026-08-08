"""Grounded extensions (v1.1): distribution-shift metrics.

Population Stability Index [11], KL divergence [12], Wasserstein distance [13],
energy distance [14], and RBF-kernel Maximum Mean Discrepancy [15]. These methods
are standard in the literature but were **not** run in the source thesis; each is
implemented here from its published definition, cited to origin, and validated on
data with a known planted answer (see ``tests/test_shift.py``).

Each ships with an **operating envelope**: PSI is unstable when quantile bins are
sparse; KL is asymmetric and bin-count sensitive; MMD is subsampled for
tractability. The validation criterion is *ranking*: the metrics must order the
planted shift ``internal < temporal < external`` -- they are not claimed to be
unbiased point estimates of any particular divergence in finite samples.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wasserstein_distance

__all__ = [
    "population_stability_index",
    "kl_divergence",
    "energy_distance",
    "mmd_rbf",
    "shift_report",
]


def population_stability_index(ref, cur, n_bins: int = 10, eps: float = 1e-6) -> float:
    """Population Stability Index between reference and current samples [11].

    Bin edges are quantiles of ``ref`` (outer edges extended to +/-inf so all of
    ``cur`` is captured). **Operating envelope:** unstable when a bin's reference
    count is small, since PSI is a sum of ``(c - r) * log(c / r)`` terms that blow
    up as ``r -> 0``; ``eps``-clipping bounds this but does not eliminate it, so PSI
    on small or heavy-tailed samples should be read as directional, not precise.
    """
    edges = np.quantile(ref, np.linspace(0, 1, n_bins + 1))
    edges[0] = -np.inf
    edges[-1] = np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r = np.clip(r, eps, None)
    c = np.clip(c, eps, None)
    return float(np.sum((c - r) * np.log(c / r)))


def kl_divergence(ref, cur, n_bins: int = 20, eps: float = 1e-6) -> float:
    """KL(ref || cur) via shared fixed-width histogram binning [12].

    **Operating envelope:** asymmetric (``KL(ref, cur) != KL(cur, ref)``) and
    sensitive to ``n_bins`` -- report the bin count alongside the value.
    """
    lo = min(ref.min(), cur.min())
    hi = max(ref.max(), cur.max())
    edges = np.linspace(lo, hi, n_bins + 1)
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r = np.clip(r, eps, None)
    c = np.clip(c, eps, None)
    return float(np.sum(r * np.log(r / c)))


def energy_distance(x, y) -> float:
    """Energy distance between two 1-D samples [14]: ``2E|X-Y| - E|X-X'| - E|Y-Y'|``.

    Falls back to a chunked mean for samples large enough that the full pairwise
    matrix (``len(x) * len(y)``) would exceed ~4e6 entries.
    """
    x = np.sort(x)
    y = np.sort(y)

    def _m(a, b):
        if len(a) * len(b) < 4_000_000:
            return np.mean(np.abs(a[:, None] - b[None, :]))
        return np.mean([np.mean(np.abs(ai - b)) for ai in a])

    return float(2 * _m(x, y) - _m(x, x) - _m(y, y))


def mmd_rbf(x, y, gamma: float | None = None, max_n: int = 1500, seed: int = 0) -> float:
    """RBF-kernel Maximum Mean Discrepancy (biased estimator) [15].

    **Operating envelope:** subsampled to at most ``max_n`` points per sample for
    tractability (an ``O(n^2)`` kernel matrix otherwise); ``gamma`` defaults to the
    median-heuristic bandwidth. Subsampling makes this a stochastic estimate --
    fix ``seed`` for reproducibility, not exactness.
    """
    rng = np.random.default_rng(seed)
    x = x[rng.choice(len(x), min(len(x), max_n), replace=False)]
    y = y[rng.choice(len(y), min(len(y), max_n), replace=False)]
    if gamma is None:
        med = np.median(np.abs(np.subtract.outer(np.r_[x, y], np.r_[x, y])))
        gamma = 1.0 / (2 * med**2 + 1e-12)

    def k(a, b):
        return np.exp(-gamma * np.subtract.outer(a, b) ** 2)

    return float(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())


def shift_report(ref_df: pd.DataFrame, cur_df: pd.DataFrame, features) -> pd.DataFrame:
    """Per-feature shift report: PSI, KL, Wasserstein, energy distance, and KS p-value."""
    return pd.DataFrame([
        dict(
            feature=f,
            PSI=round(population_stability_index(ref_df[f].values, cur_df[f].values), 4),
            KL=round(kl_divergence(ref_df[f].values, cur_df[f].values), 4),
            Wasserstein=round(wasserstein_distance(ref_df[f].values, cur_df[f].values), 4),
            energy=round(energy_distance(ref_df[f].values, cur_df[f].values), 4),
            KS_p=round(stats.ks_2samp(ref_df[f].values, cur_df[f].values).pvalue, 4),
        )
        for f in features
    ])
