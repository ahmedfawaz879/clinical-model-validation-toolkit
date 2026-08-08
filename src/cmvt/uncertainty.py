"""Tier-1 uncertainty quantification: the patient-level (cluster) bootstrap.

Confidence intervals use a nonparametric bootstrap [10] that resamples **patients**,
not rows: in ICU data many rows can share a patient, so row-level resampling
understates uncertainty. Single-statistic and paired-difference variants are
provided; the paired bootstrap gives a distribution-free companion to the paired
DeLong test in :mod:`cmvt.metrics`. Validation anchor (see
``tests/test_uncertainty.py``): the bootstrap standard error of the AUROC should
reproduce the analytic DeLong standard error.

Multi-seed variance reporting
------------------------------
A single training run conflates true performance with optimisation noise (random
initialisation, data order, stochastic solvers). Repeating the fit across seeds and
reporting mean +/- SD separates the two. This estimates *optimisation* variance
only -- **not** sampling/data uncertainty, for which the bootstrap above is the
right tool. The distinction should be stated wherever multi-seed numbers appear.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cmvt.metrics import brier_score, expected_calibration_error

__all__ = ["bootstrap_ci", "bootstrap_paired_diff", "multiseed_run"]


def bootstrap_ci(y_true, y_prob, stat_fn, *, group_id=None, n_boot: int = 2000,
                  alpha: float = 0.05, seed: int = 42) -> dict:
    """Cluster (patient-level) bootstrap CI for an arbitrary ``stat_fn(y_true, y_prob)``.

    Parameters
    ----------
    stat_fn : callable
        ``stat_fn(y_true, y_prob) -> float``.
    group_id : array-like, optional
        Cluster/patient identifiers. Rows sharing a ``group_id`` are always resampled
        together. Defaults to one group per row (no clustering).
    n_boot : int
        Number of bootstrap resamples.
    alpha : float
        Two-sided significance level; the interval reported is
        ``[alpha/2, 1 - alpha/2]`` percentile.

    Returns
    -------
    dict with keys ``point``, ``lo``, ``hi``, ``se``, ``n_boot``.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, float)
    group_id = np.arange(len(y_true)) if group_id is None else np.asarray(group_id)
    groups = np.unique(group_id)
    gmap = {g: np.where(group_id == g)[0] for g in groups}
    point = stat_fn(y_true, y_prob)
    sb = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([gmap[g] for g in rng.choice(groups, len(groups), replace=True)])
        sb[b] = stat_fn(y_true[idx], y_prob[idx])
    lo, hi = np.percentile(sb, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return dict(point=float(point), lo=float(lo), hi=float(hi), se=float(np.std(sb, ddof=1)), n_boot=n_boot)


def bootstrap_paired_diff(y_true, score_a, score_b, stat_fn, *, group_id=None,
                           n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> dict:
    """Cluster bootstrap for the *paired* difference ``stat_fn(y,a) - stat_fn(y,b)``.

    Distribution-free companion to :func:`cmvt.metrics.delong_paired_test`: both
    score vectors are resampled together (same bootstrap patients each iteration),
    preserving their correlation.

    Returns
    -------
    dict with keys ``point``, ``lo``, ``hi``, ``p_two_sided`` (the bootstrap p-value:
    twice the smaller tail proportion crossing zero, clipped to 1.0).
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    a = np.asarray(score_a, float)
    b = np.asarray(score_b, float)
    group_id = np.arange(len(y_true)) if group_id is None else np.asarray(group_id)
    groups = np.unique(group_id)
    gmap = {g: np.where(group_id == g)[0] for g in groups}
    point = stat_fn(y_true, a) - stat_fn(y_true, b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([gmap[g] for g in rng.choice(groups, len(groups), replace=True)])
        diffs[i] = stat_fn(y_true[idx], a[idx]) - stat_fn(y_true[idx], b[idx])
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_two_sided = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return dict(point=float(point), lo=float(lo), hi=float(hi), p_two_sided=float(p_two_sided))


def multiseed_run(fit_predict_fn, cohort_df, feature_cols, *, seeds=(0, 1, 2, 3, 4), metrics=None):
    """Refit across ``seeds`` and report mean +/- SD per metric -- optimisation variance only.

    Parameters
    ----------
    fit_predict_fn : callable
        ``fit_predict_fn(cohort_df, feature_cols, seed) -> (y_test, p_test)``.
    metrics : dict[str, callable], optional
        Maps metric name to ``fn(y_true, y_prob) -> float``. Defaults to AUROC,
        Brier, and ECE.

    Returns
    -------
    df : pandas.DataFrame
        One row per seed.
    agg : pandas.DataFrame
        ``mean``, ``sd``, ``cv_pct`` (coefficient of variation, %) per metric.
    """
    if metrics is None:
        metrics = {
            "auroc": lambda yt, yp: roc_auc_score(yt, yp),
            "brier": lambda yt, yp: brier_score(yt, yp),
            "ece": lambda yt, yp: expected_calibration_error(yt, yp),
        }
    rows = []
    for s in seeds:
        yte_s, p_s = fit_predict_fn(cohort_df, feature_cols, s)
        rows.append({m: fn(yte_s, p_s) for m, fn in metrics.items()} | {"seed": s})
    df = pd.DataFrame(rows)
    agg = df[list(metrics)].agg(["mean", "std"]).T
    agg.columns = ["mean", "sd"]
    agg["cv_pct"] = 100 * agg["sd"] / agg["mean"].abs()
    return df, agg
