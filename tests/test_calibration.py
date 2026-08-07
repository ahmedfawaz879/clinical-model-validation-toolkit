"""Calibration summary and recalibration tests.

Validates: the isotonic recalibration of a monotonically-miscalibrated score must
not worsen ECE (monotone non-worsening, checked in-sample as in the notebook's
audit table), and the summary dict's closed-form fields are internally consistent.
"""

from __future__ import annotations

import numpy as np
import pytest

from cmvt.calibration import calibration_summary, recalibrate
from cmvt.metrics import expected_calibration_error


class TestCalibrationSummary:
    def test_summary_matches_component_functions(self, internal_split):
        s = internal_split
        summ = calibration_summary(s["y_true"], s["p_a"])
        assert summ["ece"] == pytest.approx(expected_calibration_error(s["y_true"], s["p_a"]))
        assert summ["base_rate"] == pytest.approx(np.mean(s["y_true"]))
        assert summ["mean_pred"] == pytest.approx(np.mean(s["p_a"]))
        assert summ["brier"] >= 0

    def test_perfect_predictions_zero_brier(self):
        y = np.array([0, 1, 1, 0, 1])
        summ = calibration_summary(y, y.astype(float))
        assert summ["brier"] == pytest.approx(0.0, abs=1e-12)


class TestRecalibration:
    def test_isotonic_does_not_worsen_ece(self, internal_split):
        s = internal_split
        p_miscal = np.clip(s["p_a"] ** 0.5, 0, 1)  # deliberately miscalibrated (monotone transform)
        ece_before = expected_calibration_error(s["y_true"], p_miscal)
        p_recal, _ = recalibrate(s["y_true"], p_miscal, "isotonic")
        ece_after = expected_calibration_error(s["y_true"], p_recal)
        assert ece_after <= ece_before + 1e-9

    def test_isotonic_recalibrated_probabilities_are_in_unit_interval(self, internal_split):
        s = internal_split
        p_recal, ir = recalibrate(s["y_true"], s["p_a"], "isotonic")
        assert (p_recal >= 0).all() and (p_recal <= 1).all()
        # the fitted transform is reusable on new data
        p_new = ir.predict(s["p_a"][:5])
        assert len(p_new) == 5

    def test_platt_recalibration_returns_valid_probabilities(self, internal_split):
        s = internal_split
        p_recal, lr = recalibrate(s["y_true"], s["p_a"], "platt")
        assert (p_recal >= 0).all() and (p_recal <= 1).all()
        assert hasattr(lr, "predict_proba")

    def test_unknown_method_raises(self, internal_split):
        s = internal_split
        with pytest.raises(ValueError):
            recalibrate(s["y_true"], s["p_a"], "not-a-method")
