"""Decidable data-integrity check tests: leakage scan, exclusion tracker, missingness."""

from __future__ import annotations

import numpy as np

from cmvt.integrity import ExclusionTracker, leakage_scan, littles_mcar_test, missingness_report
from cmvt.synthetic import SHARED_FEATURES


class TestLeakageScan:
    def test_clean_split_passes_overlap_check(self, cohorts):
        d = cohorts["internal"]
        report = leakage_scan(d.iloc[:2800], d.iloc[2800:])
        row = report.set_index("check").loc["patient_overlap_train_test"]
        assert bool(row["passed"]) is True
        assert row["n"] == 0

    def test_overlapping_split_fails_overlap_check(self, cohorts):
        d = cohorts["internal"]
        report = leakage_scan(d.iloc[:2800], d.iloc[2600:])  # deliberately overlapping
        row = report.set_index("check").loc["patient_overlap_train_test"]
        assert bool(row["passed"]) is False
        assert row["n"] > 0

    def test_label_identical_feature_is_flagged(self, cohorts):
        d = cohorts["internal"]
        tr = d.iloc[:2800].copy()
        te = d.iloc[2800:].copy()
        tr["cheat"] = tr["y"].values
        report = leakage_scan(tr, te)
        row = report.set_index("check").loc["feature_identical_to_label"]
        assert bool(row["passed"]) is False
        assert "cheat" in row["detail"]

    def test_no_false_positive_on_clean_features(self, cohorts):
        d = cohorts["internal"]
        report = leakage_scan(d.iloc[:2800], d.iloc[2800:])
        row = report.set_index("check").loc["feature_identical_to_label"]
        assert bool(row["passed"]) is True

    def test_duplicate_ids_detected(self, cohorts):
        d = cohorts["internal"]
        tr = d.iloc[:100].copy()
        tr = pd_concat_dupe(tr)
        te = d.iloc[2800:]
        report = leakage_scan(tr, te)
        row = report.set_index("check").loc["duplicate_ids_in_train"]
        assert row["n"] > 0
        assert bool(row["passed"]) is False


def pd_concat_dupe(df):
    import pandas as pd

    return pd.concat([df, df.iloc[:5]], ignore_index=True)


class TestExclusionTracker:
    def test_tracks_size_and_prevalence_through_steps(self, cohorts):
        d = cohorts["external"]
        trk = ExclusionTracker(d, "eICU_like")
        trk.apply(d["age"] >= 40, "age>=40")
        trk.apply(trk.df["troponin"] > -1.5, "troponin in range")
        report = trk.report()
        assert list(report.index) == ["initial", "age>=40", "troponin in range"]
        assert report.loc["initial", "n"] == len(d)
        assert report["n"].is_monotonic_decreasing
        assert (report["n_excluded"] >= 0).all()

    def test_records_subgroup_fractions(self, cohorts):
        trk = ExclusionTracker(cohorts["internal"], strata=("sex",))
        report = trk.report()
        assert any(c.startswith("sex=") for c in report.columns)


class TestMissingness:
    def test_missingness_report_reflects_injected_nans(self, cohorts):
        rng = np.random.default_rng(0)
        base = cohorts["internal"][SHARED_FEATURES].reset_index(drop=True).copy()
        mask = rng.random(len(base)) < 0.10
        base.loc[mask, "troponin"] = np.nan
        rep = missingness_report(base, SHARED_FEATURES).set_index("feature")
        assert rep.loc["troponin", "n_missing"] == int(mask.sum())
        assert rep["pct_missing"].max() == rep.loc["troponin", "pct_missing"]

    def test_no_missingness_reports_zero(self, cohorts):
        d = cohorts["internal"]
        rep = missingness_report(d, SHARED_FEATURES)
        assert (rep["n_missing"] == 0).all()


class TestLittlesMCAR:
    def test_planted_mcar_is_not_rejected(self, cohorts):
        rng = np.random.default_rng(0)
        base = cohorts["internal"][SHARED_FEATURES].reset_index(drop=True)
        mcar = base.copy()
        mcar.loc[rng.random(len(mcar)) < 0.10, "troponin"] = np.nan
        result = littles_mcar_test(mcar, SHARED_FEATURES)
        assert result["p"] > 0.01  # should not confidently reject MCAR when it's true

    def test_planted_mar_is_rejected(self, cohorts):
        rng = np.random.default_rng(0)
        base = cohorts["internal"][SHARED_FEATURES].reset_index(drop=True)
        mar = base.copy()
        pm = 1 / (1 + np.exp(-(mar["age"] - mar["age"].mean()) / mar["age"].std()))
        mar.loc[rng.random(len(mar)) < pm * 0.4, "troponin"] = np.nan
        result = littles_mcar_test(mar, SHARED_FEATURES)
        assert result["p"] < 0.05  # MAR-driven missingness should be detected as non-MCAR
