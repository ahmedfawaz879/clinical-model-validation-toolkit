"""Tier-1 core statistics: needs only ``(y_true, y_prob)`` per cohort.

Everything in this module needs only ``(y_true, y_prob)`` -- no model, no private
data. This is the "Tier-1" contract that makes the toolkit runnable by anyone
holding only a table of predictions. Each statistic is validated (see
``tests/test_metrics.py``) against an independent reference: ``sklearn`` for the
AUROC point estimate, the Hanley-McNeil closed form for its standard error, and a
closed-form anchor (``ECE(constant c) = |c - mean(y)|``) for calibration error.

Discrimination
--------------
AUROC is estimated by the midrank algorithm of Sun & Xu [4], which also yields the
DeLong [5] variance/covariance of one or more correlated AUROCs in ``O(n log n)``.
For a single AUROC the estimator's structural components are

.. math::

    V_{10,i} = \\frac{1}{n}\\sum_j \\mathbf{1}[X_i > Y_j], \\qquad
    V_{01,j} = \\frac{1}{m}\\sum_i \\mathbf{1}[X_i > Y_j],

with ``m`` positives ``X`` and ``n`` negatives ``Y``; the variance combines the
sample covariances of these components.

Comparing two AUROCs -- paired vs unpaired, and the self-comparison guard
--------------------------------------------------------------------------
Two models on the **same** patients give *correlated* AUROCs; the correct test is
the paired DeLong test, whose variance subtracts twice the covariance. Two models on
**different** cohorts are *independent*; their variances add (unpaired test). Using
the paired test across cohorts is a real statistical error, so :func:`compare_auc`
auto-selects based on whether a second ``y_true`` is supplied.

A **self-comparison guard** refuses to test a score vector against itself. This
encodes a specific error caught during the source thesis: an AUROC "comparison" of a
model with itself yields a zero difference and a degenerate variance, which must
raise rather than silently return a meaningless *p*-value. Encoding a caught failure
as a permanent guard is a design principle of this toolkit -- where a method breaks
is documented in the code that prevents it.

Calibration
-----------
Discrimination says nothing about whether a predicted 0.2 means a 20% event rate.
Calibration is assessed with a 10-bin reliability decomposition and the **Expected
Calibration Error** (weighted mean ``|confidence - accuracy|`` across bins) [7].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import brier_score_loss
from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

__all__ = [
    "delong_auc_var",
    "compare_auc",
    "delong_paired_test",
    "delong_unpaired_test",
    "mcnemar_test",
    "tost_auc_equivalence",
    "reliability_table",
    "expected_calibration_error",
    "brier_score",
]


def _compute_midrank(x: np.ndarray) -> np.ndarray:
    """Midranks of ``x``, tie-averaged (Sun & Xu [4], Algorithm step 1)."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Fast DeLong algorithm (Sun & Xu [4]) for one or more correlated AUROCs.

    Parameters
    ----------
    preds_sorted : ndarray of shape (k, n_total)
        ``k`` score vectors, each with the ``m`` positive-class scores first.
    m : int
        Number of positives.

    Returns
    -------
    aucs : ndarray of shape (k,)
    cov : ndarray of shape (k, k)
        DeLong covariance matrix of the ``k`` AUROC estimates.
    """
    k, n_tot = preds_sorted.shape
    n = n_tot - m
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, n_tot))
    for r in range(k):
        tx[r] = _compute_midrank(preds_sorted[r, :m])
        ty[r] = _compute_midrank(preds_sorted[r, m:])
        tz[r] = _compute_midrank(preds_sorted[r])
    aucs = (tz[:, :m].sum(1) / m - (m + 1) / 2.0) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01) / m + np.cov(v10) / n)
    return aucs, cov


def delong_auc_var(y_true, y_score) -> tuple[float, float]:
    """AUROC and its DeLong variance, validated against ``sklearn.roc_auc_score``.

    Parameters
    ----------
    y_true : array-like of {0,1}
    y_score : array-like of float
        Predicted probabilities or scores.

    Returns
    -------
    auc : float
    var : float
        DeLong variance of the AUROC estimate (``sqrt(var)`` is the standard error,
        which should agree with the Hanley-McNeil [6] closed form).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, float)
    order = np.argsort(-y_true)
    m = int(y_true.sum())
    aucs, cov = _fast_delong(y_score[order][np.newaxis, :], m)
    return float(aucs[0]), float(cov[0, 0])


def delong_paired_test(y_true, score_a, score_b) -> dict:
    """Paired DeLong test for two correlated AUROCs on the same patients.

    Returns a dict with ``auc_a``, ``auc_b``, ``delta``, ``z``, ``p`` (and
    ``kind="paired"``), or ``z=p=nan`` with a ``"degenerate variance"`` note when the
    paired variance is non-positive (e.g. near-identical scores).
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true)
    m = int(y_true.sum())
    preds = np.vstack([np.asarray(score_a, float)[order], np.asarray(score_b, float)[order]])
    aucs, cov = _fast_delong(preds, m)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return dict(
            auc_a=float(aucs[0]), auc_b=float(aucs[1]), delta=float(aucs[0] - aucs[1]),
            z=np.nan, p=np.nan, note="degenerate variance",
        )
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return dict(
        auc_a=float(aucs[0]), auc_b=float(aucs[1]), delta=float(aucs[0] - aucs[1]),
        z=float(z), p=float(2 * stats.norm.sf(abs(z))), kind="paired",
    )


def delong_unpaired_test(yt_a, sc_a, yt_b, sc_b) -> dict:
    """Unpaired DeLong test for two independent AUROCs (different cohorts/patients)."""
    a, av = delong_auc_var(yt_a, sc_a)
    b, bv = delong_auc_var(yt_b, sc_b)
    se = np.sqrt(av + bv)
    return dict(
        auc_a=a, auc_b=b, delta=a - b, se=float(se), z=float((a - b) / se),
        p=float(2 * stats.norm.sf(abs((a - b) / se))), kind="unpaired",
    )


def compare_auc(y_true_a, score_a, score_b=None, *, y_true_b=None, paired=None) -> dict:
    """Dispatch to the paired or unpaired DeLong test.

    ``paired`` defaults to ``True`` when only a single ``y_true`` is supplied (same
    patients, two score vectors) and ``False`` when ``y_true_b`` is supplied (two
    independent cohorts). Using the paired test across cohorts, or the unpaired test
    within one, is a real statistical error -- the auto-selection exists to prevent it.

    Raises
    ------
    ValueError
        If a paired comparison is requested with identical score vectors (the
        self-comparison guard), or if ``score_b`` is missing for a paired comparison.
    """
    if paired is None:
        paired = y_true_b is None
    if paired:
        if score_b is None:
            raise ValueError("paired comparison needs two score vectors")
        if np.array_equal(np.asarray(score_a), np.asarray(score_b)):
            raise ValueError("SELF-COMPARISON GUARD: identical score vectors cannot be compared.")
        return delong_paired_test(y_true_a, score_a, score_b)
    return delong_unpaired_test(y_true_a, score_a, y_true_b, score_b)


def mcnemar_test(y_true, pred_a, pred_b) -> dict:
    """McNemar's test on the discordant-prediction counts between two classifiers."""
    ca = np.asarray(pred_a) == np.asarray(y_true)
    cb = np.asarray(pred_b) == np.asarray(y_true)
    n01 = int(np.sum(ca & ~cb))
    n10 = int(np.sum(~ca & cb))
    res = sm_mcnemar([[0, n01], [n10, 0]], exact=(n01 + n10 < 25))
    return dict(n_a_only_correct=n01, n_b_only_correct=n10, statistic=float(res.statistic), p=float(res.pvalue))


def tost_auc_equivalence(y_true, score_a, score_b, margin: float = 0.02) -> dict:
    """Two one-sided tests (TOST) for AUROC equivalence within ``+/- margin``.

    A significant DeLong difference is not necessarily a *meaningful* one: two
    models can differ significantly (small p-value) yet be statistically
    equivalent within a pre-specified clinically negligible margin.
    """
    r = delong_paired_test(y_true, score_a, score_b)
    se = abs(r["delta"]) / abs(r["z"]) if (r["z"] and np.isfinite(r["z"])) else np.nan
    if not np.isfinite(se) or se == 0:
        return dict(delta=r["delta"], margin=margin, equivalent=None)
    p = max(stats.norm.sf((r["delta"] + margin) / se), stats.norm.cdf((r["delta"] - margin) / se))
    return dict(delta=r["delta"], se=float(se), margin=margin, p_tost=float(p), equivalent=bool(p < 0.05))


def reliability_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Bin predictions into ``n_bins`` equal-width bins and report mean confidence vs observed rate."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        rows.append(dict(
            bin=b, lo=edges[b], hi=edges[b + 1], n=int(m.sum()),
            conf=float(y_prob[m].mean()) if m.sum() else np.nan,
            obs=float(y_true[m].mean()) if m.sum() else np.nan,
        ))
    return pd.DataFrame(rows)


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Weighted mean |confidence - observed| across bins.

    Closed-form anchor used for validation: for a constant predictor
    ``y_prob == c``, ``ECE == |c - mean(y_true)|`` exactly (all mass falls in one
    bin, whose confidence is ``c`` and observed rate is the base rate).
    """
    t = reliability_table(y_true, y_prob, n_bins).dropna()
    w = t["n"] / t["n"].sum()
    return float((w * (t["conf"] - t["obs"]).abs()).sum())


def brier_score(y_true, y_prob) -> float:
    """Brier score [7]: mean squared error between predicted probability and outcome."""
    return float(brier_score_loss(y_true, y_prob))
