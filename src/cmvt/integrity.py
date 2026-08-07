"""Grounded extensions (v1.1): decidable data-integrity checks.

Decidable leakage checks only
------------------------------
Automatic detection of *semantic* leakage (a feature computed using the outcome) is
**undecidable from values alone** -- it requires domain knowledge of how each column
was derived (see the negative control in ``tests/test_negative_controls.py``,
Section 6.3 of the source notebook). This module therefore ships only checks that
are decidable facts: patient overlap across splits, duplicate IDs, rows dated after
a temporal cutoff, and a feature identical to the label. A scanner that emitted a
single "leakage score" would be over-claiming; :func:`leakage_scan` reports
verifiable findings and refuses to guess the rest.

Cohort-transparency exclusion tracker
---------------------------------------
Following the reproducibility/equity reporting literature [16],
:class:`ExclusionTracker` records cohort *composition* (size, prevalence, subgroup
fractions) at every exclusion step. It is purely descriptive: it reports what each
filter did to the population, and issues **no** bias verdict -- deciding whether an
exclusion is unfair is a clinical judgement, not a value computation.

Missingness description and Little's MCAR test
-------------------------------------------------
:func:`missingness_report` reports per-feature missing fractions and
:func:`littles_mcar_test` implements Little's MCAR test [17]. Rejecting MCAR tells
you the data are **not** missing completely at random -- but it does **not**
identify whether they are MAR or MNAR, because the information needed to
distinguish those is, by definition, unobserved (demonstrated directly in
``tests/test_negative_controls.py``, Section 6.1). No "MNAR likelihood" is produced
by this module; claiming one would be a statistical impossibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = ["leakage_scan", "ExclusionTracker", "missingness_report", "littles_mcar_test"]


def leakage_scan(train_df: pd.DataFrame, test_df: pd.DataFrame, id_col: str = "patient_id",
                  time_col: str | None = None, label_col: str = "y", split_time=None) -> pd.DataFrame:
    """Run decidable leakage checks and return one row per check.

    Checks: patient overlap between ``train_df``/``test_df`` (critical), duplicate
    IDs within ``train_df`` (warning), rows in ``train_df`` after ``split_time`` when
    ``time_col``/``split_time`` are given (critical), and any numeric binary column
    identical to ``label_col`` (critical). Each row reports ``passed`` (bool) and a
    human-readable ``detail`` string.
    """
    f = []
    ov = set(train_df[id_col]) & set(test_df[id_col])
    f.append(dict(check="patient_overlap_train_test", severity="critical", n=len(ov), passed=len(ov) == 0,
                  detail=f"{len(ov)} patients in BOTH splits"))
    dup = int(train_df[id_col].duplicated().sum())
    f.append(dict(check="duplicate_ids_in_train", severity="warning", n=dup, passed=dup == 0,
                  detail=f"{dup} duplicate ids in train"))
    if time_col and split_time is not None:
        lk = int((train_df[time_col] > split_time).sum())
        f.append(dict(check="train_rows_after_split_time", severity="critical", n=lk, passed=lk == 0,
                      detail=f"{lk} train rows after cutoff"))
    leaky = [
        c for c in train_df.select_dtypes("number").columns
        if c != label_col and set(np.unique(train_df[c].dropna())) <= {0, 1}
        and train_df[c].nunique() > 1
        and np.array_equal(train_df[c].values.astype(int), train_df[label_col].values)
    ]
    f.append(dict(check="feature_identical_to_label", severity="critical", n=len(leaky), passed=len(leaky) == 0,
                  detail=f"label-identical columns: {leaky}"))
    return pd.DataFrame(f)


class ExclusionTracker:
    """Track cohort composition through a sequence of exclusion filters.

    Purely descriptive: records size, outcome prevalence, and subgroup fractions
    (for each column in ``strata`` that is present) at every step. Issues no
    judgement about whether an exclusion is appropriate.
    """

    def __init__(self, df: pd.DataFrame, name: str = "cohort", strata=("sex", "race")):
        self.steps: list[dict] = []
        self.strata = [s for s in strata if s in df.columns]
        self.df = df
        self._record("initial", df)

    def _record(self, label: str, df: pd.DataFrame) -> None:
        row = {"step": label, "n": len(df), "prevalence": round(df["y"].mean(), 4) if "y" in df else np.nan}
        for s in self.strata:
            for k, v in df[s].value_counts(normalize=True).items():
                row[f"{s}={k}"] = round(v, 3)
        self.steps.append(row)

    def apply(self, mask, label: str) -> ExclusionTracker:
        """Apply a boolean ``mask`` to the current cohort, recording a new step. Returns ``self``."""
        self.df = self.df[mask].copy()
        self._record(label, self.df)
        return self

    def report(self) -> pd.DataFrame:
        """Return the step-by-step composition table, indexed by step, with ``n_excluded`` per step."""
        rep = pd.DataFrame(self.steps).set_index("step")
        rep["n_excluded"] = (-rep["n"].diff()).fillna(0).astype(int)
        return rep


def missingness_report(df: pd.DataFrame, feature_cols) -> pd.DataFrame:
    """Per-feature missing count and percentage, sorted descending by ``pct_missing``."""
    return pd.DataFrame([
        dict(feature=f, n_missing=int(df[f].isna().sum()), pct_missing=round(100 * df[f].isna().mean(), 2))
        for f in feature_cols
    ]).sort_values("pct_missing", ascending=False)


def littles_mcar_test(df: pd.DataFrame, feature_cols) -> dict:
    """Little's [17] chi-square test of Missing Completely At Random.

    Groups rows by missingness pattern and compares each pattern's observed-column
    means to the grand mean via a Mahalanobis-type statistic. Rejecting the null
    (small ``p``) means the data are *not* MCAR -- it does **not** tell you whether
    the mechanism is MAR or MNAR (see the module docstring and
    ``tests/test_negative_controls.py``).

    Returns
    -------
    dict with keys ``chi2``, ``df``, ``p``.
    """
    X = df[feature_cols].copy()
    R = X.isna()
    patterns = R.apply(lambda r: tuple(r), axis=1)
    grand = X.mean()
    S = X.cov()
    d2 = 0.0
    dof = 0
    for pat, idx in patterns.groupby(patterns).groups.items():
        obs = [i for i, m in enumerate(pat) if not m]
        if not obs:
            continue
        sub = X.iloc[[X.index.get_loc(j) for j in idx], obs]
        nj = len(sub)
        diff = sub.mean().values - grand.values[obs]
        d2 += nj * float(diff @ np.linalg.pinv(S.values[np.ix_(obs, obs)]) @ diff)
        dof += len(obs)
    dof = max(dof - len(feature_cols), 1)
    return dict(chi2=float(d2), df=int(dof), p=float(stats.chi2.sf(d2, dof)))
