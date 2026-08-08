"""Shared fixtures: the synthetic three-cohort hierarchy and a couple of baseline models.

Built once per test session (the generator is deterministic given a seed) so tests
across modules share the same planted-shift data without recomputation.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cmvt.engine import CohortRegistry
from cmvt.synthetic import INTERNAL_ONLY, SHARED_FEATURES, make_cohorts

GLOBAL_SEED = 42


@pytest.fixture(scope="session")
def cohorts():
    return make_cohorts(seed=GLOBAL_SEED)


@pytest.fixture(scope="session")
def registry(cohorts):
    return (
        CohortRegistry()
        .register("internal", cohorts["internal"], SHARED_FEATURES + INTERNAL_ONLY)
        .register("temporal", cohorts["temporal"], SHARED_FEATURES)
        .register("external", cohorts["external"], SHARED_FEATURES)
    )


@pytest.fixture(scope="session")
def internal_split(cohorts):
    """A held-out split of the internal cohort with two fitted logistic-regression models.

    ``model_a`` uses only the shared features; ``model_b`` adds one internal-only
    feature. Mirrors the notebook's live demonstration (Section 2.6): the two
    models differ slightly but by a real, DeLong-detectable margin.
    """
    d = cohorts["internal"]
    cols_a = SHARED_FEATURES
    cols_b = SHARED_FEATURES + INTERNAL_ONLY[:1]
    idx_tr, idx_te = train_test_split(
        np.arange(len(d)), test_size=0.3, random_state=GLOBAL_SEED, stratify=d["y"].values
    )
    y_tr = d["y"].values[idx_tr]
    y_te = d["y"].values[idx_te]

    def _fit(cols):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(
            d[cols].values[idx_tr], y_tr
        )

    model_a = _fit(cols_a)
    model_b = _fit(cols_b)
    p_a = model_a.predict_proba(d[cols_a].values[idx_te])[:, 1]
    p_b = model_b.predict_proba(d[cols_b].values[idx_te])[:, 1]
    return dict(y_true=y_te, p_a=p_a, p_b=p_b, cols_a=cols_a, cols_b=cols_b, df=d, idx_te=idx_te)
