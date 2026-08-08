"""Calibration summary and recalibration maps.

Builds on the reliability decomposition in :mod:`cmvt.metrics` (reliability table,
Expected Calibration Error) to provide (i) a one-line calibration summary combining
the Brier score [7] with ECE and the base-rate/mean-prediction gap, and (ii) two
recalibration maps: isotonic regression [8] and Platt scaling [9].

Validation anchors (see ``tests/test_calibration.py``): a constant predictor ``c``
has ``ECE == |c - mean(y)|`` exactly, and recalibrating a monotonically-miscalibrated
score with isotonic regression must not *worsen* ECE.
"""

from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from cmvt.metrics import brier_score, expected_calibration_error

__all__ = ["calibration_summary", "recalibrate"]


def calibration_summary(y_true, y_prob, n_bins: int = 10) -> dict:
    """One-line calibration summary: Brier score, ECE, base rate, and mean prediction.

    ``base_rate`` and ``mean_pred`` are reported alongside ``ece`` so a reader can
    see at a glance whether miscalibration is a systematic over/under-prediction
    (``mean_pred`` far from ``base_rate``) or a shape mismatch that only ECE's
    per-bin decomposition reveals.
    """
    return dict(
        brier=brier_score(y_true, y_prob),
        ece=expected_calibration_error(y_true, y_prob, n_bins),
        base_rate=float(np.mean(y_true)),
        mean_pred=float(np.mean(y_prob)),
    )


def recalibrate(y_true, y_prob, method: str = "isotonic"):
    """Recalibrate predicted probabilities against observed outcomes.

    Parameters
    ----------
    method : {"isotonic", "platt"}
        ``"isotonic"`` fits a monotone step function (Zadrozny & Elkan [8]);
        ``"platt"`` fits a 1-D logistic regression on the raw score (Platt [9]).
        Both are fit and applied in-sample here; out-of-sample use requires the
        caller to fit on a held-out calibration split and apply to test scores.

    Returns
    -------
    p_recal : ndarray
        Recalibrated probabilities.
    fitted
        The fitted ``IsotonicRegression`` or ``LogisticRegression`` object, so the
        caller can apply it to new data via ``.predict`` / ``.predict_proba``.
    """
    if method == "isotonic":
        ir = IsotonicRegression(out_of_bounds="clip").fit(y_prob, y_true)
        return ir.predict(y_prob), ir
    if method == "platt":
        lr = LogisticRegression().fit(np.asarray(y_prob).reshape(-1, 1), y_true)
        return lr.predict_proba(np.asarray(y_prob).reshape(-1, 1))[:, 1], lr
    raise ValueError(f"unknown recalibration method: {method!r} (expected 'isotonic' or 'platt')")
