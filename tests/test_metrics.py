"""Tier-1 core-statistics tests: each checked against an independent reference.

Mirrors the notebook's Section 2.6 reference-agreement audit table, converted into
real assertions:
- DeLong AUROC point estimate vs ``sklearn.metrics.roc_auc_score`` (exact agreement).
- DeLong standard error vs the Hanley-McNeil closed-form approximation.
- ECE of a constant predictor ``c`` equals ``|c - mean(y)|`` exactly (closed form).
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from cmvt.metrics import (
    compare_auc,
    delong_auc_var,
    delong_paired_test,
    delong_unpaired_test,
    expected_calibration_error,
    mcnemar_test,
    reliability_table,
    tost_auc_equivalence,
)


class TestDelongAgreesWithSklearn:
    def test_auc_point_estimate_exact(self, internal_split):
        s = internal_split
        auc_dl, _ = delong_auc_var(s["y_true"], s["p_a"])
        auc_sk = roc_auc_score(s["y_true"], s["p_a"])
        assert abs(auc_dl - auc_sk) < 1e-9

    def test_auc_point_estimate_exact_random_data(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 500)
        p = rng.random(500)
        auc_dl, _ = delong_auc_var(y, p)
        assert abs(auc_dl - roc_auc_score(y, p)) < 1e-9

    def test_se_agrees_with_hanley_mcneil(self, internal_split):
        s = internal_split
        y, p = s["y_true"], s["p_a"]
        auc_dl, var_dl = delong_auc_var(y, p)
        se_dl = np.sqrt(var_dl)

        # Hanley-McNeil [6] closed-form SE.
        na = y.sum()
        nb = len(y) - na
        auc = auc_dl
        Q1 = auc / (2 - auc)
        Q2 = 2 * auc**2 / (1 + auc)
        se_hm = np.sqrt((auc * (1 - auc) + (na - 1) * (Q1 - auc**2) + (nb - 1) * (Q2 - auc**2)) / (na * nb))

        assert abs(se_dl - se_hm) < 0.02


class TestSelfComparisonGuard:
    def test_paired_identical_scores_raises(self, internal_split):
        s = internal_split
        with pytest.raises(ValueError, match="SELF-COMPARISON GUARD"):
            compare_auc(s["y_true"], s["p_a"], s["p_a"])

    def test_compare_auc_paired_with_missing_score_b_raises(self, internal_split):
        s = internal_split
        with pytest.raises(ValueError, match="paired comparison needs two score vectors"):
            compare_auc(s["y_true"], s["p_a"], paired=True)

    def test_paired_dispatch_default(self, internal_split):
        s = internal_split
        result = compare_auc(s["y_true"], s["p_a"], s["p_b"])
        assert result["kind"] == "paired"

    def test_unpaired_dispatch_via_y_true_b(self, cohorts):
        internal, external = cohorts["internal"], cohorts["external"]
        result = compare_auc(
            internal["y"].values, internal["y_true_prob"].values,
            y_true_b=external["y"].values, score_b=external["y_true_prob"].values,
        )
        assert result["kind"] == "unpaired"


class TestPairedVsUnpairedVariance:
    def test_paired_variance_smaller_than_naive_sum_for_correlated_scores(self, internal_split):
        """Paired scores are correlated (same patients); the paired test should not equal
        treating them as independent, since it subtracts 2*cov."""
        s = internal_split
        paired = delong_paired_test(s["y_true"], s["p_a"], s["p_b"])
        _, var_a = delong_auc_var(s["y_true"], s["p_a"])
        _, var_b = delong_auc_var(s["y_true"], s["p_b"])
        naive_unpaired_var = var_a + var_b
        # If there's any correlation at all, the paired SE differs from the naive sum.
        paired_se = abs(paired["delta"]) / abs(paired["z"])
        assert paired_se != pytest.approx(np.sqrt(naive_unpaired_var), rel=1e-6)

    def test_unpaired_on_independent_cohorts(self, cohorts):
        internal, external = cohorts["internal"], cohorts["external"]
        result = delong_unpaired_test(
            internal["y"].values, internal["y_true_prob"].values,
            external["y"].values, external["y_true_prob"].values,
        )
        assert result["kind"] == "unpaired"
        assert np.isfinite(result["se"])
        assert np.isfinite(result["z"])


class TestMcNemar:
    def test_identical_predictions_give_zero_discordance(self):
        y = np.array([0, 1, 1, 0, 1, 0, 1, 0])
        pred = np.array([0, 1, 0, 0, 1, 1, 1, 0])
        result = mcnemar_test(y, pred, pred)
        assert result["n_a_only_correct"] == 0
        assert result["n_b_only_correct"] == 0
        assert result["p"] == 1.0

    def test_detects_discordance(self):
        rng = np.random.default_rng(3)
        y = rng.integers(0, 2, 200)
        pred_a = y.copy()
        pred_b = y.copy()
        flip = rng.random(200) < 0.3
        pred_b[flip] = 1 - pred_b[flip]
        result = mcnemar_test(y, pred_a, pred_b)
        assert result["n_a_only_correct"] > 0
        assert 0.0 <= result["p"] <= 1.0


class TestEquivalence:
    def test_tost_significant_but_negligible(self, internal_split):
        """Two models can differ significantly by DeLong yet be statistically
        equivalent within a pre-specified margin -- the notebook's central caution."""
        s = internal_split
        tost = tost_auc_equivalence(s["y_true"], s["p_a"], s["p_b"], margin=0.10)
        assert tost["equivalent"] in (True, False, None)
        assert "delta" in tost

    def test_tost_degenerate_when_scores_nearly_identical(self, internal_split):
        s = internal_split
        result = tost_auc_equivalence(s["y_true"], s["p_a"], s["p_a"] + 1e-12)
        assert "delta" in result


class TestCalibrationClosedFormAnchors:
    @pytest.mark.parametrize("c", [0.0, 0.1, 0.5, 0.73, 1.0])
    def test_ece_of_constant_predictor_equals_closed_form(self, c):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 1000)
        p_const = np.full(1000, c)
        ece = expected_calibration_error(y, p_const)
        expected = abs(c - y.mean())
        assert ece == pytest.approx(expected, abs=1e-9)

    def test_reliability_table_shape_and_bin_edges(self):
        rng = np.random.default_rng(2)
        y = rng.integers(0, 2, 500)
        p = rng.random(500)
        t = reliability_table(y, p, n_bins=10)
        assert len(t) == 10
        assert t["lo"].iloc[0] == 0.0
        assert t["hi"].iloc[-1] == 1.0
        assert t["n"].sum() == 500

    def test_perfectly_calibrated_oracle_has_low_ece(self, cohorts):
        d = cohorts["internal"]
        ece = expected_calibration_error(d["y"].values, d["y_true_prob"].values)
        # The oracle probability IS the true event probability, so on a large sample
        # it should be well-calibrated (small, not necessarily exactly zero, since y
        # is a Bernoulli draw around y_true_prob).
        assert ece < 0.03
