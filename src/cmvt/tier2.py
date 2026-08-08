"""Tier-2 model-in modules (v2): ablation, clinical utility, fairness.

Everything in :mod:`cmvt.metrics`, :mod:`cmvt.uncertainty`, :mod:`cmvt.engine`,
:mod:`cmvt.shift`, and :mod:`cmvt.integrity` needs only predictions. This module
needs the **model** -- a framework-neutral ``fit``/``predict`` callable (here,
anything exposing ``.fit(X, y)`` and ``.predict_proba(X)``, e.g. an
``sklearn`` estimator or pipeline). That is the Tier-1/Tier-2 boundary:
predictions-in statistics run for anyone with a CSV of outputs; model-in modules
require the user to wire in their estimator, but unlock ablation, clinical
utility, and fairness analyses.

Component-removal ablation runner
------------------------------------
A factorial leave-one-component-out ablation [22] quantifies how much each block of
features (or architectural component) contributes. It ships with the **verified-
zeroing guard** -- a direct lesson from the source thesis, where an ablation that
failed to actually disable a component would silently report a null effect. Every
configuration asserts that the removed columns are genuinely absent from the model
input before its result is trusted.

Clinical utility -- decision-curve analysis
-----------------------------------------------
Discrimination and calibration do not tell a clinician whether *using* the model
produces net benefit at their decision threshold. Decision-curve analysis [18]
computes net benefit

.. math::

    \\mathrm{NB}(p_t) = \\frac{TP}{N} - \\frac{FP}{N} \\cdot \\frac{p_t}{1-p_t}

across threshold probabilities and compares against treat-all and treat-none.
Validation anchor: a perfect model has net benefit exactly equal to prevalence at
every threshold (see ``tests/test_tier2.py``).

Fairness subgroup panel
--------------------------
:func:`fairness_panel` reports per-subgroup discrimination, calibration, and
operating-point sensitivity/specificity, with the max-min disparity across
subgroups [19]. The panel **reports** disparities; it does **not** diagnose their
cause or prescribe mitigation. Whether a disparity reflects labelling, sampling,
true physiology, or model bias is a causal question a performance table cannot
answer -- asserting a cause would be the over-reach this toolkit refuses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cmvt.metrics import expected_calibration_error

__all__ = ["ablation_runner", "decision_curve", "fairness_panel"]


def ablation_runner(cohort_df: pd.DataFrame, base_features, components: dict, *,
                     seed: int = 42, test_size: float = 0.3, label: str = "y") -> pd.DataFrame:
    """Leave-one-component-out ablation with a verified-zeroing guard.

    Parameters
    ----------
    base_features : list[str]
        Feature columns always included (never ablated).
    components : dict[str, list[str]]
        Maps a component name to the feature columns that make it up. Each
        component is removed one at a time from the full feature set.

    Returns
    -------
    pandas.DataFrame
        One row per configuration (``"full"`` plus one per removed component) with
        ``auroc``, ``delta`` (vs full), and ``verified`` (whether the removed
        columns are actually absent from the design matrix used to fit/evaluate).

    Raises
    ------
    AssertionError
        If any configuration's ``verified`` flag is ``False`` -- i.e. the removed
        component's columns were not actually excluded from the model input, which
        would otherwise silently report a null (and wrong) ablation effect.
    """
    tr, te = train_test_split(cohort_df, test_size=test_size, random_state=seed, stratify=cohort_df[label])

    def _fit_eval(cols):
        m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)).fit(
            tr[cols].values, tr[label].values)
        return roc_auc_score(te[label].values, m.predict_proba(te[cols].values)[:, 1])

    full = base_features + [c for comp in components.values() for c in comp]
    fa = _fit_eval(full)
    rows = [dict(config="full", removed="—", n_features=len(full), auroc=fa, delta=0.0, verified=True)]
    for name, cols in components.items():
        kept = [c for c in full if c not in cols]
        verified = all(c not in kept for c in cols) and len(kept) < len(full)  # <-- verified-zeroing guard
        rows.append(dict(config=f"−{name}", removed=name, n_features=len(kept),
                          auroc=_fit_eval(kept), delta=_fit_eval(kept) - fa, verified=verified))
    df = pd.DataFrame(rows)
    assert df["verified"].all(), "verified-zeroing failed"
    return df


def decision_curve(y_true, y_prob, thresholds=None) -> pd.DataFrame:
    """Decision-curve analysis [18]: net benefit of the model vs treat-all/treat-none.

    Parameters
    ----------
    thresholds : array-like, optional
        Threshold probabilities to evaluate. Defaults to ``linspace(0.01, 0.60, 60)``.

    Returns
    -------
    pandas.DataFrame
        Columns ``threshold``, ``nb_model``, ``nb_treat_all``, ``nb_treat_none``
        (always 0). At every threshold, a perfect model's ``nb_model`` equals the
        outcome prevalence exactly -- the validation anchor used in
        ``tests/test_tier2.py``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, float)
    N = len(y_true)
    prev = y_true.mean()
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.60, 60)
    rows = []
    for pt in thresholds:
        pred = y_prob >= pt
        tp = np.sum(pred & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        rows.append(dict(threshold=pt, nb_model=tp / N - fp / N * (pt / (1 - pt)),
                          nb_treat_all=prev - (1 - prev) * (pt / (1 - pt)), nb_treat_none=0.0))
    return pd.DataFrame(rows)


def fairness_panel(y_true, y_prob, group, threshold: float = 0.5) -> tuple[pd.DataFrame, dict]:
    """Per-subgroup discrimination/calibration/operating-point panel and max-min disparity.

    Returns
    -------
    df : pandas.DataFrame
        One row per subgroup: ``n``, ``auroc``, ``ece``, ``sens``, ``spec``,
        ``prevalence``. Subgroups with only one outcome class get ``NaN`` metrics
        (AUROC/sens/spec undefined without both classes present).
    disparity : dict
        Max-min gap across subgroups for ``auroc``, ``ece``, ``sens``, ``spec``.
        Reported only -- no cause is inferred or claimed.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, float)
    group = np.asarray(group)
    rows = []
    for g in pd.unique(group):
        mm = group == g
        yt = y_true[mm]
        yp = y_prob[mm]
        if len(np.unique(yt)) < 2:
            rows.append(dict(subgroup=g, n=int(mm.sum()), auroc=np.nan, ece=np.nan,
                              sens=np.nan, spec=np.nan, prevalence=float(yt.mean())))
            continue
        pred = yp >= threshold
        tp = np.sum(pred & (yt == 1))
        fn = np.sum(~pred & (yt == 1))
        tn = np.sum(~pred & (yt == 0))
        fp = np.sum(pred & (yt == 0))
        rows.append(dict(subgroup=g, n=int(mm.sum()), auroc=roc_auc_score(yt, yp),
                          ece=expected_calibration_error(yt, yp),
                          sens=tp / (tp + fn) if tp + fn else np.nan,
                          spec=tn / (tn + fp) if tn + fp else np.nan,
                          prevalence=float(yt.mean())))
    df = pd.DataFrame(rows)
    disparity = {c: float(df[c].max() - df[c].min()) for c in ["auroc", "ece", "sens", "spec"]}
    return df, disparity
