"""Automated validation report tests: assembled from real computed values, no template blanks."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cmvt.engine import validation_plan
from cmvt.integrity import leakage_scan
from cmvt.report import build_validation_report
from cmvt.shift import population_stability_index
from cmvt.synthetic import INTERNAL_ONLY, SHARED_FEATURES
from cmvt.tier2 import ablation_runner, decision_curve, fairness_panel


def _build_report_inputs(cohorts, registry):
    plan, shared = validation_plan(registry, "internal", SHARED_FEATURES + INTERNAL_ONLY, seed=42)
    psi_mean = float(np.mean([
        population_stability_index(cohorts["internal"][f].values, cohorts["external"][f].values)
        for f in SHARED_FEATURES
    ]))
    internal = cohorts["internal"]
    leak_clean = leakage_scan(internal.iloc[:2800], internal.iloc[2800:])

    di = internal.copy()
    rng = np.random.default_rng(0)
    for j in range(3):
        di[f"noise_{j}"] = rng.normal(size=len(di))
    components = {"physio_signal": INTERNAL_ONLY, "noise_block": [f"noise_{j}" for j in range(3)]}
    abl = ablation_runner(di, SHARED_FEATURES, components, seed=42)

    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(
        di[SHARED_FEATURES + INTERNAL_ONLY].values, di["y"].values, di["sex"].values,
        test_size=0.3, random_state=42, stratify=di["y"].values,
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    dca = decision_curve(yte, p)
    _, disp = fairness_panel(yte, p, gte)
    return dict(plan=plan, shared=shared, leak_clean=leak_clean, abl=abl, dca=dca, disp=disp, psi_mean=psi_mean)


class TestBuildValidationReport:
    def test_report_contains_all_required_sections(self, cohorts, registry):
        inputs = _build_report_inputs(cohorts, registry)
        report = build_validation_report(
            inputs["plan"], cohorts, inputs["shared"], inputs["leak_clean"],
            inputs["abl"], inputs["dca"], inputs["disp"], inputs["psi_mean"],
        )
        for heading in [
            "# Automated Validation Report",
            "## 1. Transportability",
            "## 2. Distribution shift",
            "## 3. Data-integrity",
            "## 4. Component ablation",
            "## 5. Clinical utility",
            "## 6. Subgroup disparities",
            "## 7. Deliberately omitted",
            "## 8. Reporting-standard note",
        ]:
            assert heading in report

    def test_deliberately_omitted_section_names_all_three_negative_controls(self, cohorts, registry):
        inputs = _build_report_inputs(cohorts, registry)
        report = build_validation_report(
            inputs["plan"], cohorts, inputs["shared"], inputs["leak_clean"],
            inputs["abl"], inputs["dca"], inputs["disp"], inputs["psi_mean"],
        )
        assert "MNAR" in report
        assert "quality score" in report
        assert "leakage score" in report

    def test_report_is_nonempty_string(self, cohorts, registry):
        inputs = _build_report_inputs(cohorts, registry)
        report = build_validation_report(
            inputs["plan"], cohorts, inputs["shared"], inputs["leak_clean"],
            inputs["abl"], inputs["dca"], inputs["disp"], inputs["psi_mean"],
        )
        assert isinstance(report, str)
        assert len(report) > 500

    def test_report_reflects_actual_plan_numbers(self, cohorts, registry):
        inputs = _build_report_inputs(cohorts, registry)
        report = build_validation_report(
            inputs["plan"], cohorts, inputs["shared"], inputs["leak_clean"],
            inputs["abl"], inputs["dca"], inputs["disp"], inputs["psi_mean"],
        )
        external_fair_auroc = f"{inputs['plan'].loc['external', 'fair_auroc']:.3f}"
        assert external_fair_auroc in report
