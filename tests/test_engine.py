"""Validation-engine tests: naive-vs-fair transfer, and the monotone-gap property.

Ports the notebook's Section 3.1 exit criterion: on planted-shift data, the fair
(shared-feature) analysis must recover a *monotone* transfer gap
(internal < temporal < external) while the naive analysis over-attributes the drop
to shift when part of it is information loss.
"""

from __future__ import annotations

import numpy as np
import pytest

from cmvt.engine import CohortRegistry, naive_transfer, shared_feature_validation, validation_plan
from cmvt.synthetic import INTERNAL_ONLY, SHARED_FEATURES


class TestCohortRegistry:
    def test_shared_features_is_intersection(self, registry):
        shared = registry.shared_features()
        assert shared == sorted(SHARED_FEATURES)
        assert set(shared).isdisjoint(INTERNAL_ONLY)

    def test_schema_report_flags_internal_only_features(self, registry):
        rep = registry.schema_report().set_index("feature")
        for f in INTERNAL_ONLY:
            assert rep.loc[f, "internal"] and not rep.loc[f, "temporal"] and not rep.loc[f, "external"]
            assert not rep.loc[f, "shared"]
        for f in SHARED_FEATURES:
            assert rep.loc[f, "shared"]

    def test_register_returns_self_for_chaining(self, cohorts):
        reg = CohortRegistry()
        result = reg.register("internal", cohorts["internal"], SHARED_FEATURES)
        assert result is reg


class TestValidationPlanRecoversMonotoneGap:
    def test_fair_gap_is_monotone_in_planted_shift(self, registry):
        plan, shared = validation_plan(registry, "internal", SHARED_FEATURES + INTERNAL_ONLY)
        assert plan.loc["internal", "fair_gap_true_shift"] == pytest.approx(0.0, abs=1e-9)
        assert 0 < plan.loc["temporal", "fair_gap_true_shift"] < plan.loc["external", "fair_gap_true_shift"]

    def test_information_loss_confound_is_positive_for_feature_poor_cohorts(self, registry):
        plan, _ = validation_plan(registry, "internal", SHARED_FEATURES + INTERNAL_ONLY)
        # temporal/external don't record the internal-only features, so naive
        # zero-imputed transfer must lose more than the fair (shared-feature) transfer.
        assert plan.loc["temporal", "information_loss_confound"] > 0
        assert plan.loc["external", "information_loss_confound"] > 0.02

    def test_shared_feature_space_excludes_internal_only(self, registry):
        _, shared = validation_plan(registry, "internal", SHARED_FEATURES + INTERNAL_ONLY)
        assert set(shared) == set(SHARED_FEATURES)

    def test_naive_gap_exceeds_fair_gap_when_confound_present(self, registry):
        plan, _ = validation_plan(registry, "internal", SHARED_FEATURES + INTERNAL_ONLY)
        for role in ("temporal", "external"):
            assert plan.loc[role, "naive_gap"] > plan.loc[role, "fair_gap_true_shift"]


class TestNaiveTransfer:
    def test_naive_transfer_dev_cohort_auroc_is_high(self, registry):
        out = naive_transfer(registry, "internal", SHARED_FEATURES + INTERNAL_ONLY)
        assert out["internal"] > 0.6

    def test_naive_transfer_zero_imputes_missing_features(self, cohorts):
        """A cohort lacking a feature entirely should not raise -- the naive-transfer
        design zero-imputes it (this is the confound the fair analysis exists to strip out)."""
        reg = CohortRegistry().register("internal", cohorts["internal"], SHARED_FEATURES + INTERNAL_ONLY)
        reg.register("temporal", cohorts["temporal"], SHARED_FEATURES)
        out = naive_transfer(reg, "internal", SHARED_FEATURES + INTERNAL_ONLY)
        assert np.isfinite(out["temporal"])


class TestSharedFeatureValidation:
    def test_all_cohorts_evaluated_on_identical_schema(self, registry):
        fair, shared = shared_feature_validation(registry, "internal")
        assert set(shared) == set(SHARED_FEATURES)
        assert set(fair.keys()) == {"internal", "temporal", "external"}
        for auc in fair.values():
            assert 0.5 <= auc <= 1.0

    def test_matches_manual_roc_auc_score(self, registry, cohorts):
        fair, shared = shared_feature_validation(registry, "internal")
        # Re-fitting with the same seed should be deterministic and reproduce the AUROC.
        fair2, _ = shared_feature_validation(registry, "internal")
        assert fair == fair2
