"""cmvt — a reproducible, synthetic-first validation toolkit for clinical prediction models.

Tier 1 (predictions-in, needs only ``(y_true, y_prob)``):
    :mod:`cmvt.metrics`, :mod:`cmvt.calibration`, :mod:`cmvt.uncertainty`,
    :mod:`cmvt.engine`, :mod:`cmvt.shift`, :mod:`cmvt.integrity`.

Tier 2 (model-in, needs a ``fit``/``predict_proba`` callable):
    :mod:`cmvt.tier2`.

See the README for the problem statement, quickstart, and the "what this toolkit
refuses to claim" section (:mod:`cmvt.integrity`, and the negative-control tests in
``tests/test_negative_controls.py``).
"""

from __future__ import annotations

from cmvt.calibration import calibration_summary, recalibrate
from cmvt.engine import CohortRegistry, naive_transfer, shared_feature_validation, validation_plan
from cmvt.integrity import ExclusionTracker, leakage_scan, littles_mcar_test, missingness_report
from cmvt.metrics import (
    brier_score,
    compare_auc,
    delong_auc_var,
    expected_calibration_error,
    mcnemar_test,
    reliability_table,
    tost_auc_equivalence,
)
from cmvt.report import build_validation_report
from cmvt.shift import (
    energy_distance,
    kl_divergence,
    mmd_rbf,
    population_stability_index,
    shift_report,
)
from cmvt.synthetic import CohortSpec, generate_cohort, make_cohorts
from cmvt.tier2 import ablation_runner, decision_curve, fairness_panel
from cmvt.uncertainty import bootstrap_ci, bootstrap_paired_diff, multiseed_run

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # synthetic
    "CohortSpec", "generate_cohort", "make_cohorts",
    # metrics (Tier 1)
    "delong_auc_var", "compare_auc", "mcnemar_test", "tost_auc_equivalence",
    "reliability_table", "expected_calibration_error", "brier_score",
    # calibration (Tier 1)
    "calibration_summary", "recalibrate",
    # uncertainty (Tier 1)
    "bootstrap_ci", "bootstrap_paired_diff", "multiseed_run",
    # engine (Tier 1 / flagship)
    "CohortRegistry", "naive_transfer", "shared_feature_validation", "validation_plan",
    # shift (Tier 1)
    "population_stability_index", "kl_divergence", "energy_distance", "mmd_rbf", "shift_report",
    # integrity (Tier 1)
    "leakage_scan", "ExclusionTracker", "missingness_report", "littles_mcar_test",
    # tier2 (model-in)
    "ablation_runner", "decision_curve", "fairness_panel",
    # report
    "build_validation_report",
]
