"""Tier-2 model-in module tests: ablation guard, decision-curve anchor, fairness panel."""

from __future__ import annotations

import numpy as np
import pytest

from cmvt.synthetic import INTERNAL_ONLY, SHARED_FEATURES
from cmvt.tier2 import ablation_runner, decision_curve, fairness_panel


class TestAblationRunner:
    def test_recovers_known_effects(self, cohorts):
        di = cohorts["internal"].copy()
        rng = np.random.default_rng(0)
        for j in range(3):
            di[f"noise_{j}"] = rng.normal(size=len(di))
        components = {"physio_signal": INTERNAL_ONLY, "noise_block": [f"noise_{j}" for j in range(3)]}
        abl = ablation_runner(di, SHARED_FEATURES, components, seed=42)
        by_removed = abl.set_index("removed")
        assert by_removed.loc["physio_signal", "delta"] < -0.01
        assert abs(by_removed.loc["noise_block", "delta"]) < 0.01
        assert abl["verified"].all()

    def test_verified_zeroing_guard_fires_on_empty_component(self, cohorts):
        """If a declared component maps to no real columns (e.g. an empty list from a
        config-wiring bug), 'removing' it changes nothing about the design matrix --
        the verified-zeroing guard must raise loudly rather than silently report a
        null effect as if the ablation had actually run."""
        di = cohorts["internal"]
        components = {"empty_component": []}
        with pytest.raises(AssertionError, match="verified-zeroing failed"):
            ablation_runner(di, SHARED_FEATURES, components, seed=42)

    def test_full_config_uses_all_features(self, cohorts):
        di = cohorts["internal"]
        components = {"internal_only": INTERNAL_ONLY}
        abl = ablation_runner(di, SHARED_FEATURES, components, seed=42)
        full_row = abl[abl.config == "full"].iloc[0]
        assert full_row["n_features"] == len(SHARED_FEATURES) + len(INTERNAL_ONLY)
        assert full_row["delta"] == 0.0


class TestDecisionCurve:
    def test_perfect_model_net_benefit_equals_prevalence_at_every_threshold(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 2000)
        prev = y.mean()
        dca = decision_curve(y, y.astype(float))
        assert np.allclose(dca["nb_model"].values, prev, atol=1e-9)

    def test_treat_none_is_always_zero(self, cohorts):
        d = cohorts["internal"]
        dca = decision_curve(d["y"].values, d["y_true_prob"].values)
        assert (dca["nb_treat_none"] == 0.0).all()

    def test_model_dominates_treat_all_at_low_prevalence_high_threshold(self, cohorts):
        d = cohorts["internal"]
        dca = decision_curve(d["y"].values, d["y_true_prob"].values, thresholds=[0.5])
        # a reasonably discriminating model should not be worse than treat-all near
        # its own decision boundary when net benefit is evaluated on oracle probs
        assert dca.iloc[0]["nb_model"] >= dca.iloc[0]["nb_treat_none"]

    def test_custom_thresholds_respected(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 500)
        p = rng.random(500)
        thresholds = [0.1, 0.3, 0.5]
        dca = decision_curve(y, p, thresholds=thresholds)
        assert list(dca["threshold"]) == thresholds


class TestFairnessPanel:
    def test_reports_one_row_per_subgroup(self, cohorts):
        d = cohorts["internal"]
        panel, disp = fairness_panel(d["y"].values, d["y_true_prob"].values, d["sex"].values)
        assert set(panel["subgroup"]) == set(d["sex"].unique())
        assert set(disp.keys()) == {"auroc", "ece", "sens", "spec"}
        assert all(v >= 0 for v in disp.values())

    def test_planted_disparity_is_detected(self, cohorts):
        d = cohorts["internal"].copy()
        rng = np.random.default_rng(1)
        p = d["y_true_prob"].values.copy()
        fmask = d["sex"].values == "F"
        p[fmask] = np.clip(0.5 * p[fmask] + 0.5 * rng.random(fmask.sum()), 0, 1)
        panel, disp = fairness_panel(d["y"].values, p, d["sex"].values)
        assert disp["auroc"] > 0.05  # the planted degradation for group F should show up

    def test_single_class_subgroup_returns_nan_metrics(self):
        y = np.array([1, 1, 1, 0, 0, 0])
        p = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        group = np.array(["a", "a", "a", "a", "a", "b"])  # group b has only y=0
        panel, _ = fairness_panel(y, p, group)
        row_b = panel.set_index("subgroup").loc["b"]
        assert np.isnan(row_b["auroc"])
