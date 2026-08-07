#!/usr/bin/env python
"""Regenerate all six README figures into ``docs/figures/`` at 200 dpi, deterministically.

This script is the *only* source of the figures under ``docs/figures/`` -- they are
never hand-edited. Run it with:

    python scripts/generate_figures.py

Every number and plot here is produced by the installed ``cmvt`` package on the
bundled synthetic cohorts, using the fixed global seed below, so re-running this
script on a clean checkout reproduces the exact same PNGs (bit-for-bit reproducible
data; minor anti-aliasing differences across matplotlib/font versions are possible
but the numbers are exact).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from cmvt.calibration import recalibrate
from cmvt.engine import CohortRegistry, validation_plan
from cmvt.metrics import compare_auc, expected_calibration_error, reliability_table
from cmvt.shift import population_stability_index
from cmvt.synthetic import INTERNAL_ONLY, SHARED_FEATURES, make_cohorts
from cmvt.tier2 import ablation_runner, decision_curve, fairness_panel

GLOBAL_SEED = 42
DPI = 200

COLORS = {"internal": "#2166ac", "temporal": "#f4a582", "external": "#b2182b"}
SHIFTS = {"internal": 0.0, "temporal": 0.35, "external": 0.70}


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def figure_1_synthetic_generator(cohorts: dict, out_dir: Path) -> None:
    """Figure 1: verification of the planted covariate shift, label shift, and oracle separability."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    ax = axes[0]
    for r, d in cohorts.items():
        ax.hist(d["troponin"], bins=40, density=True, alpha=0.5, color=COLORS[r],
                 label=f"{r} (shift={SHIFTS[r]})")
    ax.set_title("A. Planted covariate shift\n(shared feature: troponin)")
    ax.set_xlabel("value")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)

    ax = axes[1]
    roles = list(cohorts.keys())
    prevs = [cohorts[r]["y"].mean() for r in roles]
    ax.bar(roles, prevs, color=[COLORS[r] for r in roles])
    for i, p in enumerate(prevs):
        ax.text(i, p + 0.005, f"{p:.3f}", ha="center", fontsize=9)
    ax.set_title("B. Planted label shift\n(outcome prevalence)")
    ax.set_ylabel("prevalence")
    ax.set_ylim(0, 0.38)

    ax = axes[2]
    aucs = [roc_auc_score(cohorts[r]["y"], cohorts[r]["y_true_prob"]) for r in roles]
    ax.bar(roles, aucs, color=[COLORS[r] for r in roles])
    for i, a in enumerate(aucs):
        ax.text(i, a + 0.005, f"{a:.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, ls="--", c="grey", lw=1)
    ax.set_ylim(0.5, 1.0)
    ax.set_title("C. Oracle separability\n(Bayes-optimal AUROC per cohort)")
    ax.set_ylabel("AUROC")

    fig.suptitle("Figure 1. Synthetic multi-cohort generator — verification of planted structure", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / "figure1_synthetic_generator.png")


def figure_2_calibration(cohorts: dict, out_dir: Path) -> dict:
    """Figure 2: reliability diagram for two models, and the effect of isotonic recalibration."""
    d = cohorts["internal"]
    cols_a = SHARED_FEATURES
    cols_b = SHARED_FEATURES + INTERNAL_ONLY[:1]
    idx_tr, idx_te = train_test_split(
        np.arange(len(d)), test_size=0.3, random_state=GLOBAL_SEED, stratify=d["y"].values
    )
    y_tr, y_te = d["y"].values[idx_tr], d["y"].values[idx_te]

    def _fit(cols):
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(
            d[cols].values[idx_tr], y_tr
        )

    model_a, model_b = _fit(cols_a), _fit(cols_b)
    p_a = model_a.predict_proba(d[cols_a].values[idx_te])[:, 1]
    p_b = model_b.predict_proba(d[cols_b].values[idx_te])[:, 1]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    for name, p, c in [("model A (10 shared features)", p_a, "#2166ac"),
                        ("model B (+1 internal-only feature)", p_b, "#b2182b")]:
        t = reliability_table(y_te, p, 10).dropna()
        ax[0].plot(t["conf"], t["obs"], "o-", color=c,
                    label=f"{name} (ECE={expected_calibration_error(y_te, p):.3f})")
    ax[0].plot([0, 1], [0, 1], "--", c="grey")
    ax[0].set_xlabel("mean predicted")
    ax[0].set_ylabel("observed frequency")
    ax[0].set_title("A. Reliability diagram")
    ax[0].legend(fontsize=8)

    p_miscal = np.clip(p_a**0.5, 0, 1)
    p_recal, _ = recalibrate(y_te, p_miscal, "isotonic")
    for name, p, c in [("miscalibrated", p_miscal, "#d6604d"), ("after isotonic", p_recal, "#1a9850")]:
        t = reliability_table(y_te, p, 10).dropna()
        ax[1].plot(t["conf"], t["obs"], "o-", color=c,
                    label=f"{name} (ECE={expected_calibration_error(y_te, p):.3f})")
    ax[1].plot([0, 1], [0, 1], "--", c="grey")
    ax[1].set_xlabel("mean predicted")
    ax[1].set_ylabel("observed frequency")
    ax[1].set_title("B. Recalibration effect")
    ax[1].legend(fontsize=8)

    fig.suptitle("Figure 2. Calibration diagnostics and recalibration", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / "figure2_calibration.png")

    paired = compare_auc(y_te, p_a, p_b)
    print(f"  [fig2] paired DeLong dAUROC={paired['delta']:+.4f} z={paired['z']:.2f} p={paired['p']:.4f}")
    return dict(y_te=y_te, p_a=p_a, p_b=p_b)


def figure_3_validation_engine(cohorts: dict, out_dir: Path):
    """Figure 3: the validation-engine story -- naive vs fair transfer, and the isolated confound."""
    reg = (
        CohortRegistry()
        .register("internal", cohorts["internal"], SHARED_FEATURES + INTERNAL_ONLY)
        .register("temporal", cohorts["temporal"], SHARED_FEATURES)
        .register("external", cohorts["external"], SHARED_FEATURES)
    )
    oracle = {r: roc_auc_score(cohorts[r]["y"], cohorts[r]["y_true_prob"]) for r in cohorts}
    plan, shared = validation_plan(reg, "internal", SHARED_FEATURES + INTERNAL_ONLY, seed=GLOBAL_SEED)
    assert plan.loc["external", "fair_gap_true_shift"] > plan.loc["temporal", "fair_gap_true_shift"] > 0
    assert plan.loc["external", "information_loss_confound"] > 0.02

    roles = ["internal", "temporal", "external"]
    xs = np.arange(3)
    w = 0.35
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    ax[0].bar(xs - w / 2, [plan.loc[r, "naive_auroc"] for r in roles], w, label="naive transfer (full)", color="#d6604d")
    ax[0].bar(xs + w / 2, [plan.loc[r, "fair_auroc"] for r in roles], w, label="shared-feature (fair)", color="#4393c3")
    ax[0].plot(xs, [oracle[r] for r in roles], "k*--", ms=13, label="per-cohort oracle")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(roles)
    ax[0].set_ylim(0.5, 0.95)
    ax[0].set_ylabel("AUROC")
    ax[0].set_title("A. Naive vs fair transfer")
    ax[0].legend(fontsize=8)

    ax[1].bar(xs - w / 2, [plan.loc[r, "naive_gap"] for r in roles], w, label="naive gap", color="#d6604d")
    ax[1].bar(xs + w / 2, [plan.loc[r, "fair_gap_true_shift"] for r in roles], w, label="fair gap (true shift)", color="#4393c3")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(roles)
    ax[1].set_ylabel("AUROC drop from internal")
    ax[1].set_title("B. Naive collapse = true shift + information loss")
    ax[1].legend(fontsize=8)

    ax[2].bar(xs, [plan.loc[r, "information_loss_confound"] for r in roles], color=["#2166ac", "#f4a582", "#b2182b"])
    for i, r in enumerate(roles):
        ax[2].text(i, plan.loc[r, "information_loss_confound"] + 0.001,
                    f"{plan.loc[r, 'information_loss_confound']:+.3f}", ha="center", fontsize=9)
    ax[2].set_xticks(xs)
    ax[2].set_xticklabels(roles)
    ax[2].set_ylabel("naive gap - fair gap")
    ax[2].set_title("C. The information-loss confound\n(what naive transfer wrongly blames on shift)")

    fig.suptitle("Figure 3. Validation engine — separating true distribution shift from information loss", y=1.03)
    fig.tight_layout()
    _save(fig, out_dir / "figure3_validation_engine.png")
    return plan


def figure_4_shift_metrics(cohorts: dict, plan, out_dir: Path) -> None:
    """Figure 4: per-feature PSI ranks the planted shift, and shift magnitude tracks transfer degradation."""
    feats = SHARED_FEATURES
    psi_temporal = [population_stability_index(cohorts["internal"][f].values, cohorts["temporal"][f].values) for f in feats]
    psi_external = [population_stability_index(cohorts["internal"][f].values, cohorts["external"][f].values) for f in feats]
    assert 0 < np.mean(psi_temporal) < np.mean(psi_external), "PSI must rank the planted covariate shift"

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    xs = np.arange(len(feats))
    w = 0.38
    ax[0].bar(xs - w / 2, psi_temporal, w, label="internal→temporal", color="#f4a582")
    ax[0].bar(xs + w / 2, psi_external, w, label="internal→external", color="#b2182b")
    ax[0].axhline(0.1, ls="--", c="grey", lw=1)
    ax[0].axhline(0.25, ls="--", c="k", lw=1)
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(feats, rotation=45, ha="right", fontsize=8)
    ax[0].set_ylabel("PSI")
    ax[0].set_title("A. Per-feature PSI ranks planted shift")
    ax[0].legend(fontsize=8)

    gapd = {"temporal": plan.loc["temporal", "fair_gap_true_shift"], "external": plan.loc["external", "fair_gap_true_shift"]}
    mp = {"temporal": np.mean(psi_temporal), "external": np.mean(psi_external)}
    ax[1].scatter([mp[r] for r in ["temporal", "external"]], [gapd[r] for r in ["temporal", "external"]],
                    s=120, c=["#f4a582", "#b2182b"], zorder=3)
    for r in ["temporal", "external"]:
        ax[1].annotate(r, (mp[r], gapd[r]), fontsize=9, xytext=(5, 5), textcoords="offset points")
    ax[1].set_xlabel("mean PSI")
    ax[1].set_ylabel("fair transfer gap (AUROC)")
    ax[1].set_title("B. Shift magnitude tracks transfer degradation")

    fig.suptitle("Figure 4. Distribution-shift metrics (v1.1, validated on planted shift)", y=1.02)
    fig.tight_layout()
    _save(fig, out_dir / "figure4_shift_metrics.png")


def figure_5_tier2_modules(cohorts: dict, out_dir: Path) -> None:
    """Figure 5: component-removal ablation, decision-curve analysis, and the fairness panel."""
    di = cohorts["internal"].copy()
    rng = np.random.default_rng(0)
    for j in range(3):
        di[f"noise_{j}"] = rng.normal(size=len(di))
    components = {"physio_signal": INTERNAL_ONLY, "noise_block": [f"noise_{j}" for j in range(3)]}
    abl = ablation_runner(di, SHARED_FEATURES, components, seed=GLOBAL_SEED)
    assert abl.set_index("removed").loc["physio_signal", "delta"] < -0.01
    assert abs(abl.set_index("removed").loc["noise_block", "delta"]) < 0.01

    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(
        di[SHARED_FEATURES + INTERNAL_ONLY].values, di["y"].values, di["sex"].values,
        test_size=0.3, random_state=GLOBAL_SEED, stratify=di["y"].values,
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    dca = decision_curve(yte, model.predict_proba(Xte)[:, 1])

    pf = model.predict_proba(Xte)[:, 1].copy()
    fmask = gte == "F"
    rng2 = np.random.default_rng(1)
    pf[fmask] = np.clip(0.5 * pf[fmask] + 0.5 * rng2.random(fmask.sum()), 0, 1)  # planted disparity
    panel, disp = fairness_panel(yte, pf, gte)

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    a = abl[abl.config != "full"].sort_values("delta")
    cols = ["#b2182b" if d < 0 else "#4393c3" for d in a["delta"]]
    ax[0].barh(a["config"], a["delta"], color=cols)
    for i, (_, r) in enumerate(a.iterrows()):
        ax[0].text(r["delta"], i, f" {r['delta']:+.3f}", va="center", fontsize=9)
    ax[0].axvline(0, c="k", lw=0.8)
    ax[0].set_xlabel("deltaAUROC vs full")
    ax[0].set_title("A. Component-removal ablation\n(verified-zeroing enforced)")

    ax[1].plot(dca["threshold"], dca["nb_model"], color="#2166ac", lw=2, label="model")
    ax[1].plot(dca["threshold"], dca["nb_treat_all"], color="grey", ls="--", label="treat all")
    ax[1].axhline(0, color="k", ls=":", label="treat none")
    ax[1].set_ylim(-0.05, dca["nb_model"].max() * 1.15)
    ax[1].set_xlabel("threshold probability")
    ax[1].set_ylabel("net benefit")
    ax[1].set_title("B. Decision-curve analysis")
    ax[1].legend(fontsize=8)

    s = panel.dropna(subset=["auroc"])
    ax[2].bar(s["subgroup"], s["auroc"], color=["#4393c3", "#d6604d"])
    for i, (_, r) in enumerate(s.iterrows()):
        ax[2].text(i, r["auroc"] + 0.005, f"{r['auroc']:.3f}", ha="center", fontsize=9)
    ax[2].set_ylim(0.5, 0.95)
    ax[2].set_ylabel("subgroup AUROC")
    ax[2].set_title(f"C. Fairness panel (disparity={disp['auroc']:.3f})\nreported, not causally diagnosed")

    fig.suptitle("Figure 5. Tier-2 model-in modules — ablation, clinical utility, fairness", y=1.03)
    fig.tight_layout()
    _save(fig, out_dir / "figure5_tier2_modules.png")


def figure_6_negative_controls(out_dir: Path) -> None:
    """Figure 6: the three negative controls of Section 6 -- what the toolkit refuses to claim."""
    # NC1: MAR vs MNAR unidentifiable.
    rng = np.random.default_rng(11)
    N = 20000
    X_full = rng.integers(0, 4, N)
    obs_A = ~(rng.random(N) < np.array([0.1, 0.3, 0.5, 0.7])[X_full])
    observed_X_A = X_full[obs_A]
    target = np.bincount(observed_X_A, minlength=4) / obs_A.sum()
    X_full_B = rng.choice(4, size=N, p=[0.15, 0.2, 0.25, 0.4])
    full_B = np.bincount(X_full_B, minlength=4) / N
    keep = target / (full_B + 1e-12)
    keep /= keep.max()
    obs_B = rng.random(N) < keep[X_full_B]
    observed_X_B = X_full_B[obs_B]
    oa = np.bincount(observed_X_A, minlength=4) / obs_A.sum()
    ob = np.bincount(observed_X_B, minlength=4) / obs_B.sum()
    fa = np.bincount(X_full, minlength=4) / N
    fb = np.bincount(X_full_B, minlength=4) / N
    assert np.abs(oa - ob).max() < 0.02

    # NC2: composite quality score is weight-dependent.
    import pandas as pd

    datasets = pd.DataFrame({
        "dataset": ["Cohort_A", "Cohort_B", "Cohort_C"],
        "completeness": [0.95, 0.55, 0.65], "balance": [0.50, 0.95, 0.60],
        "size": [0.60, 0.55, 0.95], "shift_free": [0.70, 0.75, 0.72],
    }).set_index("dataset")
    dims = ["completeness", "balance", "size", "shift_free"]
    weightings = {"equal": [.25, .25, .25, .25], "clinician": [.55, .25, .10, .10],
                  "trialist": [.10, .55, .25, .10], "ml_engineer": [.10, .15, .55, .20]}
    nc2 = pd.DataFrame({
        n: pd.Series((datasets[dims].values * np.array(w)).sum(1), index=datasets.index)
            .rank(ascending=False).astype(int)
        for n, w in weightings.items()
    })
    winners = nc2.apply(lambda c: c.idxmin())
    assert winners.nunique() >= 3

    # NC3: semantic leakage undecidable from values alone.
    rng3 = np.random.default_rng(5)
    n = 8000
    y = rng3.integers(0, 2, n)
    legit = 0.9 * y + rng3.normal(0, 1, n)
    leak = 0.9 * y + rng3.normal(0, 1, n)

    def col_stats(c):
        return dict(corr_with_y=np.corrcoef(c, y)[0, 1], auroc=roc_auc_score(y, c),
                    mean_diff=c[y == 1].mean() - c[y == 0].mean())

    nc3 = pd.DataFrame({"legitimate_predictor": col_stats(legit), "outcome_derived_leak": col_stats(leak)}).T
    mx = max(abs(nc3.loc["legitimate_predictor", k] - nc3.loc["outcome_derived_leak", k]) for k in nc3.columns)
    assert mx < 0.05

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.7))
    xs = np.arange(4)
    w = 0.2
    ax[0].bar(xs - 1.5 * w, oa, w, color="#2166ac", label="A observed (MNAR)")
    ax[0].bar(xs - 0.5 * w, ob, w, color="#92c5de", label="B observed (MAR)")
    ax[0].bar(xs + 0.5 * w, fa, w, color="#b2182b", label="A full (truth)")
    ax[0].bar(xs + 1.5 * w, fb, w, color="#f4a582", label="B full (truth)")
    ax[0].set_xticks(xs)
    ax[0].set_xlabel("X value")
    ax[0].set_ylabel("probability")
    ax[0].set_title("NC1. MAR vs MNAR unidentifiable\nobserved identical, truth differs")
    ax[0].legend(fontsize=7)

    ax[1].imshow(nc2.values, cmap="RdYlGn_r", aspect="auto", vmin=1, vmax=3)
    ax[1].set_xticks(range(len(nc2.columns)))
    ax[1].set_xticklabels(nc2.columns, rotation=30, ha="right", fontsize=8)
    ax[1].set_yticks(range(len(nc2.index)))
    ax[1].set_yticklabels(nc2.index, fontsize=8)
    for i in range(nc2.shape[0]):
        for j in range(nc2.shape[1]):
            ax[1].text(j, i, int(nc2.values[i, j]), ha="center", va="center", fontsize=11, fontweight="bold")
    ax[1].set_title("NC2. 'Quality score' rank instability\neach weighting crowns a different winner")

    metrics = ["corr_with_y", "auroc", "mean_diff"]
    xm = np.arange(3)
    w2 = 0.35
    ax[2].bar(xm - w2 / 2, [nc3.loc["legitimate_predictor", m] for m in metrics], w2, color="#1a9850", label="legitimate predictor")
    ax[2].bar(xm + w2 / 2, [nc3.loc["outcome_derived_leak", m] for m in metrics], w2, color="#762a83", label="outcome-derived leak")
    ax[2].set_xticks(xm)
    ax[2].set_xticklabels(metrics, rotation=20, ha="right", fontsize=8)
    ax[2].set_title("NC3. Semantic leakage undecidable\nidentical stats, only provenance differs")
    ax[2].legend(fontsize=8)

    fig.suptitle("Figure 6. Negative controls — what the toolkit deliberately refuses to claim", y=1.03)
    fig.tight_layout()
    _save(fig, out_dir / "figure6_negative_controls.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("docs/figures"),
                         help="Output directory for the PNGs (default: docs/figures)")
    args = parser.parse_args()

    np.random.seed(GLOBAL_SEED)
    plt.rcParams["figure.dpi"] = DPI

    print(f"Generating cohorts (seed={GLOBAL_SEED})...")
    cohorts = make_cohorts(seed=GLOBAL_SEED)

    print("Figure 1/6 ...")
    figure_1_synthetic_generator(cohorts, args.out_dir)
    print("Figure 2/6 ...")
    figure_2_calibration(cohorts, args.out_dir)
    print("Figure 3/6 ...")
    plan = figure_3_validation_engine(cohorts, args.out_dir)
    print("Figure 4/6 ...")
    figure_4_shift_metrics(cohorts, plan, args.out_dir)
    print("Figure 5/6 ...")
    figure_5_tier2_modules(cohorts, args.out_dir)
    print("Figure 6/6 ...")
    figure_6_negative_controls(args.out_dir)

    print(f"\nAll 6 figures written to {args.out_dir}/ at {DPI} dpi.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
