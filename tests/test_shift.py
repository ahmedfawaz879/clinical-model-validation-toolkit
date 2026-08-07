"""Distribution-shift metric tests: ranking property on planted shift.

Validation criterion (Section 4.1 of the notebook): the metrics must order the
planted covariate shift ``internal < temporal < external`` -- they are not claimed
to be unbiased point estimates in finite samples, only to rank correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from cmvt.shift import energy_distance, kl_divergence, mmd_rbf, population_stability_index, shift_report
from cmvt.synthetic import SHARED_FEATURES


class TestPSIRanksPlantedShift:
    def test_psi_zero_for_self_comparison(self, cohorts):
        x = cohorts["internal"]["troponin"].values
        assert population_stability_index(x, x) == pytest.approx(0.0, abs=1e-9)

    def test_psi_ranks_shift_magnitude(self, cohorts):
        psi_temporal = np.mean([
            population_stability_index(cohorts["internal"][f].values, cohorts["temporal"][f].values)
            for f in SHARED_FEATURES
        ])
        psi_external = np.mean([
            population_stability_index(cohorts["internal"][f].values, cohorts["external"][f].values)
            for f in SHARED_FEATURES
        ])
        assert 0 < psi_temporal < psi_external


class TestOtherShiftMetricsAgreeOnRanking:
    def test_kl_divergence_ranks_shift(self, cohorts):
        kl_t = np.mean([
            kl_divergence(cohorts["internal"][f].values, cohorts["temporal"][f].values) for f in SHARED_FEATURES
        ])
        kl_e = np.mean([
            kl_divergence(cohorts["internal"][f].values, cohorts["external"][f].values) for f in SHARED_FEATURES
        ])
        assert 0 <= kl_t < kl_e

    def test_energy_distance_ranks_shift(self, cohorts):
        e_t = np.mean([
            energy_distance(cohorts["internal"][f].values, cohorts["temporal"][f].values) for f in SHARED_FEATURES
        ])
        e_e = np.mean([
            energy_distance(cohorts["internal"][f].values, cohorts["external"][f].values) for f in SHARED_FEATURES
        ])
        assert 0 <= e_t < e_e

    def test_energy_distance_self_is_zero(self, cohorts):
        x = cohorts["internal"]["troponin"].values
        assert energy_distance(x, x) == pytest.approx(0.0, abs=1e-9)

    def test_mmd_rbf_ranks_shift(self, cohorts):
        mmd_t = mmd_rbf(cohorts["internal"]["troponin"].values, cohorts["temporal"]["troponin"].values, seed=0)
        mmd_e = mmd_rbf(cohorts["internal"]["troponin"].values, cohorts["external"]["troponin"].values, seed=0)
        assert mmd_t < mmd_e

    def test_mmd_rbf_nonnegative(self, cohorts):
        mmd = mmd_rbf(cohorts["internal"]["troponin"].values, cohorts["external"]["troponin"].values, seed=0)
        assert mmd >= -1e-9


class TestShiftReport:
    def test_returns_one_row_per_feature(self, cohorts):
        rep = shift_report(cohorts["internal"], cohorts["external"], SHARED_FEATURES)
        assert len(rep) == len(SHARED_FEATURES)
        assert set(rep.columns) == {"feature", "PSI", "KL", "Wasserstein", "energy", "KS_p"}

    def test_ks_p_small_for_shifted_features(self, cohorts):
        rep = shift_report(cohorts["internal"], cohorts["external"], SHARED_FEATURES).set_index("feature")
        # troponin was drawn with a large mean shift (0.70 SD); KS should easily detect it.
        assert rep.loc["troponin", "KS_p"] < 0.01
