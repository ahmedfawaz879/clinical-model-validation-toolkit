"""Synthetic-cohort generator tests: planted structure is actually recovered."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from cmvt.synthetic import INTERNAL_ONLY, SHARED_FEATURES, CohortSpec, generate_cohort, make_cohorts


class TestMakeCohorts:
    def test_three_roles_present(self, cohorts):
        assert set(cohorts.keys()) == {"internal", "temporal", "external"}

    def test_prevalence_matches_target(self, cohorts):
        assert cohorts["internal"]["y"].mean() == pytest.approx(0.30, abs=0.02)
        assert cohorts["temporal"]["y"].mean() == pytest.approx(0.24, abs=0.02)
        assert cohorts["external"]["y"].mean() == pytest.approx(0.17, abs=0.02)

    def test_internal_only_features_recorded_only_internally(self, cohorts):
        for f in INTERNAL_ONLY:
            assert f in cohorts["internal"].columns
            assert f not in cohorts["temporal"].columns
            assert f not in cohorts["external"].columns

    def test_deterministic_given_seed(self):
        a = make_cohorts(seed=123)
        b = make_cohorts(seed=123)
        for role in a:
            assert a[role]["y"].equals(b[role]["y"])
            assert np.allclose(a[role]["troponin"].values, b[role]["troponin"].values)

    def test_different_seeds_differ(self):
        a = make_cohorts(seed=1)
        b = make_cohorts(seed=2)
        assert not np.allclose(a["internal"]["troponin"].values, b["internal"]["troponin"].values)

    def test_oracle_auroc_is_high_and_similar_across_cohorts(self, cohorts):
        """Because BETA is fixed and rotation is norm-preserving, the Bayes-optimal
        AUROC should be roughly similar across cohorts -- any *trained*-model gap is
        transfer failure, not differing task difficulty."""
        aucs = {r: roc_auc_score(d["y"], d["y_true_prob"]) for r, d in cohorts.items()}
        for a in aucs.values():
            assert 0.65 < a < 1.0
        assert max(aucs.values()) - min(aucs.values()) < 0.15


class TestGenerateCohort:
    def test_covariate_shift_moves_feature_means(self):
        spec_a = CohortSpec("A", "internal", 2000, 0.3, shift=0.0, seed=1)
        spec_b = CohortSpec("B", "external", 2000, 0.3, shift=0.70, seed=2)
        da = generate_cohort(spec_a)
        db = generate_cohort(spec_b)
        assert db["troponin"].mean() - da["troponin"].mean() > 0.5

    def test_has_internal_only_flag_controls_columns(self):
        spec = CohortSpec("X", "temporal", 500, 0.25, has_internal_only=False, seed=5)
        d = generate_cohort(spec)
        assert not any(f in d.columns for f in INTERNAL_ONLY)

    def test_patient_ids_are_unique(self):
        spec = CohortSpec("Y", "internal", 1000, 0.3, seed=9)
        d = generate_cohort(spec)
        assert d["patient_id"].nunique() == len(d)

    def test_all_shared_features_present(self):
        spec = CohortSpec("Z", "internal", 300, 0.3, seed=3)
        d = generate_cohort(spec)
        for f in SHARED_FEATURES:
            assert f in d.columns
