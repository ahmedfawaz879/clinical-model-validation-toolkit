"""The validation engine (flagship): separating true distribution shift from information loss.

External validation of a clinical model almost always reports a single number: the
AUROC fell from X internally to Y externally. That number conflates two very
different things:

1. **True distribution shift** -- the patients and/or the feature-outcome
   relationship genuinely differ across sites.
2. **Information loss** -- the external cohort simply does not record some features
   the model was trained on, so at deployment they are imputed (often as zero),
   discarding real predictive signal.

Attributing a naive-transfer collapse entirely to "the model doesn't generalise" is
a **confound**: some of the drop is shift, some is missing information. The engine
separates them with two analyses on the same cohorts:

- **Naive transfer** (:func:`naive_transfer`) -- train on the full internal feature
  set, deploy elsewhere by zero-imputing absent features. Deployment-realistic, but
  confounded.
- **Shared-feature validation** (:func:`shared_feature_validation`) -- intersect the
  feature schemas programmatically, retrain from scratch on the shared space, and
  evaluate every cohort on *identical* inputs. This removes the information-loss
  confound; the remaining gap is attributable to distribution/concept shift.

:func:`validation_plan` runs both and reports the difference between the two gaps
as the information-loss confound itself, quantified. This shared-feature design is
the methodological core carried over from the source thesis; the three-tier
internal -> temporal -> external hierarchy follows the external-validation
literature [1,2,3].

This is a Tier-2 module in the sense that it fits models internally, but its public
entry points (:func:`naive_transfer`, :func:`shared_feature_validation`,
:func:`validation_plan`) only require a :class:`CohortRegistry` of
``(y, features)`` tables -- no external model object is threaded through.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = ["CohortRegistry", "naive_transfer", "shared_feature_validation", "validation_plan"]


class CohortRegistry:
    """Registry of cohorts with roles and per-cohort recorded feature schemas.

    Each cohort is registered with the feature columns actually *recorded* for it,
    which may differ across cohorts (see :mod:`cmvt.synthetic`'s internal-only
    features). :meth:`shared_features` computes the intersection programmatically --
    the engine never hardcodes which features are shared.
    """

    def __init__(self):
        self.cohorts: dict[str, dict] = {}

    def register(self, role: str, df: pd.DataFrame, feature_cols) -> CohortRegistry:
        """Register a cohort's data and its recorded feature columns. Returns ``self`` for chaining."""
        self.cohorts[role] = dict(df=df, features=list(feature_cols))
        return self

    def shared_features(self) -> list[str]:
        """Sorted intersection of feature columns recorded across *all* registered cohorts."""
        return sorted(set.intersection(*[set(v["features"]) for v in self.cohorts.values()]))

    def schema_report(self) -> pd.DataFrame:
        """One row per feature (union across cohorts), with per-cohort availability and a ``shared`` flag."""
        allf = sorted(set().union(*[set(v["features"]) for v in self.cohorts.values()]))
        return pd.DataFrame([
            dict(feature=f, **{r: (f in v["features"]) for r, v in self.cohorts.items()},
                 shared=all(f in v["features"] for v in self.cohorts.values()))
            for f in allf
        ])


def _make_model(seed: int = 42):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))


def naive_transfer(reg: CohortRegistry, dev_role: str, feature_cols, seed: int = 42) -> dict[str, float]:
    """Full-feature model trained on ``dev_role``, transferred by zero-imputing absent features.

    This is the deployment-realistic but *confounded* analysis: a cohort's AUROC
    drop mixes true distribution shift with the information loss from features it
    never recorded (silently zero-imputed at prediction time).
    """
    dev = reg.cohorts[dev_role]["df"]
    model = _make_model(seed).fit(dev[feature_cols].values, dev["y"].values)
    out = {}
    for role, v in reg.cohorts.items():
        d = v["df"]
        X = np.zeros((len(d), len(feature_cols)))
        for j, f in enumerate(feature_cols):
            if f in d.columns:
                X[:, j] = d[f].values
        out[role] = roc_auc_score(d["y"].values, model.predict_proba(X)[:, 1])
    return out


def shared_feature_validation(reg: CohortRegistry, dev_role: str, seed: int = 42) -> tuple[dict[str, float], list[str]]:
    """Fair transfer: retrain on the intersection feature space; evaluate all cohorts identically.

    Because every cohort is evaluated on the exact same input schema, no cohort can
    under-perform merely because it recorded fewer features -- the remaining AUROC
    gap is attributable to distribution/concept shift alone.
    """
    shared = reg.shared_features()
    dev = reg.cohorts[dev_role]["df"]
    model = _make_model(seed).fit(dev[shared].values, dev["y"].values)
    return (
        {role: roc_auc_score(v["df"]["y"].values, model.predict_proba(v["df"][shared].values)[:, 1])
         for role, v in reg.cohorts.items()},
        shared,
    )


def validation_plan(reg: CohortRegistry, dev_role: str = "internal", full_feature_cols=None, seed: int = 42):
    """Run naive transfer and shared-feature validation, and isolate the confound between them.

    Parameters
    ----------
    reg : CohortRegistry
        Registered cohorts, at least one being ``dev_role`` (the development cohort).
    dev_role : str
        Role to train on (default ``"internal"``).
    full_feature_cols : list[str], optional
        Feature columns for the naive-transfer model. Defaults to ``dev_role``'s
        full recorded feature set.

    Returns
    -------
    plan : pandas.DataFrame
        Indexed by cohort role, with columns ``naive_auroc``, ``fair_auroc``,
        ``naive_gap`` (drop from dev under naive transfer), ``fair_gap_true_shift``
        (drop from dev under shared-feature transfer -- the *true shift* estimate),
        and ``information_loss_confound`` (``naive_gap - fair_gap_true_shift``: the
        part of the naive collapse wrongly attributable to shift when it is really
        missing information).
    shared : list[str]
        The shared-feature space used for the fair analysis.
    """
    full_feature_cols = full_feature_cols or reg.cohorts[dev_role]["features"]
    naive = naive_transfer(reg, dev_role, full_feature_cols, seed)
    fair, shared = shared_feature_validation(reg, dev_role, seed)
    roles = list(reg.cohorts)
    rows = []
    for r in roles:
        ng = naive[dev_role] - naive[r]
        fg = fair[dev_role] - fair[r]
        rows.append(dict(role=r, naive_auroc=naive[r], fair_auroc=fair[r],
                          naive_gap=ng, fair_gap_true_shift=fg, information_loss_confound=ng - fg))
    return pd.DataFrame(rows).set_index("role"), shared
