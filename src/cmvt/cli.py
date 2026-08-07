"""End-to-end demo pipeline: bundled synthetic cohorts -> a written validation report.

This is what ``cmvt-demo`` (the console-script entry point installed by
``pip install -e .``) and ``examples/run_synthetic_demo.py`` both run. It exercises
Tier-1 and Tier-2 modules on the synthetic three-cohort hierarchy and writes
``reports/validation_report.md`` -- the same clone-and-run contract exercised by CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cmvt.engine import CohortRegistry, validation_plan
from cmvt.integrity import leakage_scan
from cmvt.report import build_validation_report
from cmvt.shift import population_stability_index, shift_report
from cmvt.synthetic import INTERNAL_ONLY, SHARED_FEATURES, make_cohorts
from cmvt.tier2 import ablation_runner, decision_curve, fairness_panel

GLOBAL_SEED = 42


def main(argv: list[str] | None = None) -> int:
    """Run the full synthetic-cohort validation demo and write ``reports/validation_report.md``."""
    np.random.seed(GLOBAL_SEED)

    cohorts = make_cohorts(seed=GLOBAL_SEED)
    print("Synthetic cohorts:")
    for role, d in cohorts.items():
        print(f"  {role:9s} n={len(d):5d}  prevalence={d['y'].mean():.3f}")

    reg = (
        CohortRegistry()
        .register("internal", cohorts["internal"], SHARED_FEATURES + INTERNAL_ONLY)
        .register("temporal", cohorts["temporal"], SHARED_FEATURES)
        .register("external", cohorts["external"], SHARED_FEATURES)
    )
    plan, shared = validation_plan(reg, "internal", SHARED_FEATURES + INTERNAL_ONLY, seed=GLOBAL_SEED)
    print("\nValidation plan (naive vs shared-feature transfer):")
    print(plan.round(4).to_string())

    psi_mean = float(np.mean([
        population_stability_index(cohorts["internal"][f].values, cohorts["external"][f].values)
        for f in SHARED_FEATURES
    ]))
    shift_tbl = shift_report(cohorts["internal"], cohorts["external"], SHARED_FEATURES)
    print(f"\nMean PSI internal->external: {psi_mean:.3f}")

    internal = cohorts["internal"]
    leak_clean = leakage_scan(internal.iloc[:2800], internal.iloc[2800:])

    di = internal.copy()
    rng = np.random.default_rng(0)
    for j in range(3):
        di[f"noise_{j}"] = rng.normal(size=len(di))

    components = {"physio_signal": INTERNAL_ONLY, "noise_block": [f"noise_{j}" for j in range(3)]}
    abl = ablation_runner(di, SHARED_FEATURES, components, seed=GLOBAL_SEED)

    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(
        di[SHARED_FEATURES + INTERNAL_ONLY].values, di["y"].values, di["sex"].values,
        test_size=0.3, random_state=GLOBAL_SEED, stratify=di["y"].values,
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    p_te = model.predict_proba(Xte)[:, 1]
    dca = decision_curve(yte, p_te)
    panel, disp = fairness_panel(yte, p_te, gte)
    print(f"\nAUROC (internal held-out): {roc_auc_score(yte, p_te):.4f}")
    print(f"Fairness max-min AUROC disparity (sex): {disp['auroc']:.4f}")

    report = build_validation_report(plan, cohorts, shared, leak_clean, abl, dca, disp, psi_mean)

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "validation_report.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"\nWrote {out_path} ({len(report)} chars)")
    print(f"Per-feature shift report ({len(shift_tbl)} features) computed but not printed in full; "
          "see shift_report() for programmatic access.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
