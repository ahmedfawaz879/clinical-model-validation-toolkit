---
title: 'cmvt: A Python toolkit for reproducible external validation of clinical prediction models'
tags:
  - Python
  - clinical prediction models
  - external validation
  - calibration
  - distribution shift
  - reproducibility
authors:
  - name: Ahmed Fawaz
    affiliation: 1
affiliations:
  - name: "[Affiliation to be completed]"
    index: 1
date: 5 September 2026
bibliography: paper.bib
---

# Summary

Clinical prediction models are routinely reported with a single external-validation
headline: AUROC fell from some internal value to some lower external value. `cmvt`
(clinical-model-validation-toolkit) is a Python package that treats that number as
insufficient on its own and provides the machinery to unpack it: DeLong-based
comparison of correlated AUROCs [@delong1988], calibration diagnostics and
recalibration, patient-level cluster bootstrap confidence intervals, distribution-shift
metrics (population stability index, KL divergence, energy distance, and RBF-kernel
MMD), decidable data-integrity checks (patient overlap across splits, duplicate
identifiers, post-cutoff leakage, exact label duplication, and Little's MCAR test),
and a validation engine that decomposes an external-validation AUROC gap into a
component attributable to true distribution shift and a component attributable to
information loss from features the external site never recorded.

The package ships a synthetic three-cohort generator (internal, temporal, external)
so the full pipeline runs end-to-end with no external data dependency and no network
access: `pip install -e ".[dev]"` followed by `cmvt-demo` builds roughly 12,000
synthetic patients, runs every module, and writes a Markdown validation report in
under a minute on a laptop. All 92 tests in the test suite assert numerical agreement
against an independent reference implementation where one exists (`scipy`,
`statsmodels`, `scikit-learn`) rather than only checking that the code runs, and
continuous integration exercises the suite on Python 3.10, 3.11, and 3.12 with a
90% minimum-coverage gate.

# Statement of need

External validation is a required step before a clinical prediction model is trusted
outside the population it was developed on [@steyerberg2019; @debray2015], and
calibration failures in particular are common enough to be called predictive
analytics' "Achilles heel" [@vancalster2019]. Two distinct explanations are typically
available whenever external AUROC is lower than internal AUROC: the external
population genuinely differs from the development population (distribution shift)
[@subbaswamy2020; @finlayson2021], or the external site did not record some of the
predictors the model was trained on, so those predictors are zero-imputed or dropped
at deployment, discarding real signal. These two explanations call for different
remedies — model updating or population-specific redevelopment for genuine shift,
versus better data capture or a model redesigned around what is actually available
for information loss — but a single external AUROC number cannot distinguish them.

`cmvt.engine.validation_plan` addresses this directly: it fits the development
model on its full recorded feature set and evaluates it externally with the
external site's missing features zero-imputed (the "naive" transfer), separately
retrains on only the features every registered cohort actually shares and evaluates
that ("fair") model externally, and reports the difference between the naive and
fair AUROC gaps as an information-loss confound (\autoref{fig:engine}). Existing
general-purpose model-validation and fairness toolkits provide calibration and
subgroup-performance diagnostics but do not, to our knowledge, provide this specific
decomposition, nor construct synthetic cohorts with a known ground-truth split
between shift and missingness against which such a decomposition could be checked
for correctness during development.

A second design goal shapes the package as much as its calculations: `cmvt` is built
to show what it will not compute. Three modules ship alongside executable negative
controls (`tests/test_negative_controls.py`) demonstrating, on data constructed so the
full-data truth is known, that (1) MAR-versus-MNAR status is not identifiable from
observed data alone, so `cmvt.integrity.littles_mcar_test` tests only for MCAR and
does not extend to an MNAR-detection claim; (2) a composite "dataset quality" score
built from completeness, balance, size, and shift-freedom crowns a different dataset
as "best" depending on the (equally defensible) weighting chosen, so `cmvt` reports
sub-metrics individually and does not rank them; and (3) a values-only leakage
detector cannot distinguish a legitimate predictor from an outcome-derived leak with
identical AUROC, correlation, and mean difference against the label, so
`cmvt.integrity.leakage_scan` restricts itself to checks that are decidable from the
data alone (split overlap, duplicate IDs, post-cutoff rows, exact-label-duplicate
columns). This is intended for researchers and clinical-informatics practitioners
who need not just a working validation pipeline but one whose claims are explicitly
bounded by what is statistically identifiable.

![Output of the bundled synthetic demo (`cmvt-demo`), reproduced deterministically by `scripts/generate_figures.py`. Panel A compares naive full-feature transfer against shared-feature ("fair") transfer across three synthetic cohorts of increasing dissimilarity, against a per-cohort oracle ceiling. Panel B shows the naive AUROC drop from the internal cohort decomposing into a true-shift component (blue, the fair gap) and an information-loss component (the remainder). Panel C isolates that information-loss confound directly.\label{fig:engine}](figure3_validation_engine.png)

# Acknowledgements

We acknowledge the open-source maintainers of NumPy, pandas, SciPy, scikit-learn,
statsmodels, and Matplotlib, on which this package depends.

# References
