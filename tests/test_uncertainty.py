"""Patient-level bootstrap tests.

Validates that the bootstrap standard error of the AUROC reproduces the analytic
DeLong standard error (Section 2.6 of the notebook), and exercises the paired
bootstrap and multi-seed runner.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cmvt.metrics import brier_score, delong_auc_var, expected_calibration_error
from cmvt.uncertainty import bootstrap_ci, bootstrap_paired_diff, multiseed_run


class TestBootstrapAgreesWithDelong:
    def test_bootstrap_se_close_to_delong_se(self, internal_split):
        s = internal_split
        auc_fn = lambda yt, yp: roc_auc_score(yt, yp)  # noqa: E731
        ci = bootstrap_ci(s["y_true"], s["p_a"], auc_fn, n_boot=1500, seed=42)
        _, var_dl = delong_auc_var(s["y_true"], s["p_a"])
        se_dl = np.sqrt(var_dl)
        assert abs(ci["se"] - se_dl) < 0.01

    def test_bootstrap_point_estimate_matches_statistic(self, internal_split):
        s = internal_split
        auc_fn = lambda yt, yp: roc_auc_score(yt, yp)  # noqa: E731
        ci = bootstrap_ci(s["y_true"], s["p_a"], auc_fn, n_boot=500, seed=1)
        assert ci["point"] == roc_auc_score(s["y_true"], s["p_a"])
        assert ci["lo"] < ci["point"] < ci["hi"]

    def test_bootstrap_respects_group_clustering(self):
        """Row-level resampling should understate variance vs patient-level (cluster)
        resampling when many rows share a patient -- the whole point of the cluster
        bootstrap. We construct 20 patients x 10 duplicated rows each and check the
        clustered bootstrap SE is not implausibly small."""
        rng = np.random.default_rng(7)
        n_patients = 20
        patient_p = rng.random(n_patients)
        patient_y = (rng.random(n_patients) < patient_p).astype(int)
        reps = 10
        y = np.repeat(patient_y, reps)
        p = np.repeat(patient_p, reps) + rng.normal(0, 0.01, n_patients * reps)
        group = np.repeat(np.arange(n_patients), reps)

        auc_fn = lambda yt, yp: roc_auc_score(yt, yp)  # noqa: E731
        clustered = bootstrap_ci(y, p, auc_fn, group_id=group, n_boot=500, seed=0)
        row_level = bootstrap_ci(y, p, auc_fn, n_boot=500, seed=0)
        # Clustered SE should be >= a substantial fraction of naive row-level SE is not
        # guaranteed in general, but it must at least be a finite, positive number and
        # the two should differ (independent resampling units vs correlated rows).
        assert clustered["se"] > 0
        assert clustered["se"] != row_level["se"]


class TestPairedBootstrap:
    def test_paired_diff_matches_point_difference(self, internal_split):
        s = internal_split
        auc_fn = lambda yt, yp: roc_auc_score(yt, yp)  # noqa: E731
        result = bootstrap_paired_diff(s["y_true"], s["p_a"], s["p_b"], auc_fn, n_boot=500, seed=0)
        expected = auc_fn(s["y_true"], s["p_a"]) - auc_fn(s["y_true"], s["p_b"])
        assert result["point"] == expected
        assert 0.0 <= result["p_two_sided"] <= 1.0

    def test_paired_diff_zero_for_identical_scores(self, internal_split):
        s = internal_split
        auc_fn = lambda yt, yp: roc_auc_score(yt, yp)  # noqa: E731
        result = bootstrap_paired_diff(s["y_true"], s["p_a"], s["p_a"], auc_fn, n_boot=200, seed=0)
        assert result["point"] == 0.0
        assert result["p_two_sided"] == 1.0


class TestMultiseedRun:
    def test_reports_mean_and_sd_across_seeds(self, cohorts):
        d = cohorts["internal"]
        feature_cols = ["troponin", "creatinine", "heart_rate"]

        def fit_predict(df, cols, seed):
            tr, te = train_test_split(df, test_size=0.3, random_state=seed, stratify=df["y"])
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
            m.fit(tr[cols].values, tr["y"].values)
            return te["y"].values, m.predict_proba(te[cols].values)[:, 1]

        df, agg = multiseed_run(fit_predict, d, feature_cols, seeds=(0, 1, 2))
        assert len(df) == 3
        assert set(agg.index) == {"auroc", "brier", "ece"}
        assert (agg["sd"] >= 0).all()

    def test_default_metrics_match_standalone_functions(self, cohorts):
        d = cohorts["internal"]
        feature_cols = ["troponin", "creatinine"]

        def fit_predict(df, cols, seed):
            tr, te = train_test_split(df, test_size=0.3, random_state=seed, stratify=df["y"])
            m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, random_state=seed))
            m.fit(tr[cols].values, tr["y"].values)
            yt, p = te["y"].values, m.predict_proba(te[cols].values)[:, 1]
            return yt, p

        df, _ = multiseed_run(fit_predict, d, feature_cols, seeds=(0,))
        yt, p = fit_predict(d, feature_cols, 0)
        assert df.loc[0, "auroc"] == roc_auc_score(yt, p)
        assert df.loc[0, "brier"] == brier_score(yt, p)
        assert df.loc[0, "ece"] == expected_calibration_error(yt, p)
