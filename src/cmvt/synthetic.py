"""Synthetic multi-cohort generator with planted covariate, label, and concept shift.

A validation toolkit must be *auditable*: a reader has to be able to check that each
method returns the right answer. That is only possible when the correct answer is
known. This module generates three cohorts with **planted** structure and treats the
plant as ground truth:

- ``internal``  the development cohort (no shift; role model of MIMIC-IV).
- ``temporal``  same institution, earlier era (moderate covariate + label shift;
  role model of MIMIC-III).
- ``external``  different institutions (large shift; role model of eICU-CRD).

Two structural properties of real multi-site ICU cohorts are reproduced deliberately:

1. **Feature-availability differences.** The internal cohort carries five
   *internal-only* features (e.g. arterial-waveform-derived indices) absent from the
   external cohorts. This is what motivates the shared-feature validation design in
   :mod:`cmvt.engine` -- naively transferring a full-feature model to a feature-poor
   cohort confounds distribution shift with information loss.
2. **Distribution shift.** A covariate shift of increasing magnitude
   (0.0 -> 0.35 -> 0.70 SD on the feature means) and a label shift (prevalence
   0.31 -> 0.25 -> 0.17) are planted so the validation hierarchy has a real,
   measurable gap to recover. A norm-preserving rotation of the outcome coefficients
   (a *concept* shift) is additionally applied to the temporal and external cohorts.

*Provenance:* the three-tier internal -> temporal -> external hierarchy follows the
transportability framework of Justice et al. and the validation taxonomy of
Steyerberg and of Riley, Collins et al. [1,2,3]. See the project References.

No private or credentialed data is used or required. MIMIC-IV, MIMIC-III, and
eICU-CRD motivate the design but are PhysioNet credentialed-access datasets and are
never shipped; this generator reproduces the *structural properties* those cohorts
exhibit so the methods can be demonstrated and audited by anyone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import brentq

__all__ = [
    "SHARED_FEATURES",
    "INTERNAL_ONLY",
    "BETA_BASE",
    "BETA_INTERNAL_ONLY",
    "CohortSpec",
    "generate_cohort",
    "make_cohorts",
]

#: Features recorded in every cohort (internal, temporal, external).
SHARED_FEATURES = [
    "troponin", "creatinine", "heart_rate", "sbp", "spo2",
    "resp_rate", "sodium", "hemoglobin", "wbc", "age",
]

#: Features recorded only in the internal (development) cohort.
INTERNAL_ONLY = ["art_wave_compliance", "sevr_proxy", "co_estimate", "lactate", "inr"]

#: True outcome model: fixed shared-feature coefficients, shared across cohorts
#: (before any concept-shift rotation is applied).
BETA_BASE = np.array([1.1, 0.7, 0.5, -0.6, -0.8, 0.4, 0.0, -0.3, 0.3, 0.0])

#: True outcome model: fixed internal-only-feature coefficients. These contribute to
#: the outcome in *every* cohort but are only *recorded* internally -- naive
#: zero-imputed transfer to a feature-poor cohort therefore discards real signal.
BETA_INTERNAL_ONLY = np.array([0.6, 0.5, -0.4, 0.7, -0.3])


@dataclass
class CohortSpec:
    """Specification for one synthetic cohort.

    Parameters
    ----------
    name : str
        Cohort identifier, e.g. ``"MIMIC-IV_like"``.
    role : str
        One of ``"internal"``, ``"temporal"``, ``"external"``.
    n_patients : int
        Number of rows to generate.
    prevalence : float
        Target outcome prevalence; the intercept is solved numerically to match it.
    shift : float
        Covariate-shift magnitude in standard deviations applied to feature means.
    signal_scale : float
        Multiplicative scale on the shared-feature logit contribution.
    has_internal_only : bool
        Whether the five internal-only features are recorded (only ``True`` for the
        internal/development cohort by convention).
    seed : int
        Cohort-specific random seed.
    """

    name: str
    role: str
    n_patients: int
    prevalence: float
    shift: float = 0.0
    signal_scale: float = 1.0
    has_internal_only: bool = True
    seed: int = 0


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _perturb_beta(beta: np.ndarray, concept_shift: float, rng: np.random.Generator) -> np.ndarray:
    """Concept shift: rotate coefficients (norm-preserved so per-cohort separability holds).

    The rotation preserves ``||beta||``, so the Bayes-optimal AUROC stays ~identical
    across cohorts -- the task remains equally learnable everywhere while a *trained*
    model's transfer degrades. This isolates transfer failure from task difficulty.
    """
    if concept_shift == 0:
        return beta.copy()
    noise = rng.normal(0, 1, size=beta.shape)
    noise -= noise.mean()
    b = beta + concept_shift * np.linalg.norm(beta) / np.sqrt(len(beta)) * noise
    return b * (np.linalg.norm(beta) / np.linalg.norm(b))


def generate_cohort(spec: CohortSpec, concept_shift: float = 0.0) -> pd.DataFrame:
    """Generate one cohort with a planted covariate shift, concept shift, and known outcome model.

    Internal-only features contribute to the TRUE outcome in every cohort, but are
    only RECORDED in the internal cohort -- naive transfer to feature-poor cohorts
    loses real signal (the information-loss confound that :mod:`cmvt.engine` exists
    to isolate).

    Parameters
    ----------
    spec : CohortSpec
        Cohort specification (size, prevalence, shift, seed, ...).
    concept_shift : float
        Magnitude of the norm-preserving coefficient rotation (0 = no concept shift).

    Returns
    -------
    pandas.DataFrame
        One row per patient with shared features, ``y`` (binary outcome),
        ``y_true_prob`` (the planted Bayes-optimal probability), demographics, and
        (if ``has_internal_only``) the five internal-only features.
    """
    rng = np.random.default_rng(spec.seed)
    n = spec.n_patients
    X = rng.normal(loc=spec.shift, scale=1.0, size=(n, len(SHARED_FEATURES)))
    df = pd.DataFrame(X, columns=SHARED_FEATURES)
    df["age"] = np.clip(rng.normal(65 + 5 * spec.shift, 12, n), 18, 95)
    Xv = df[SHARED_FEATURES].values
    Xz = (Xv - Xv.mean(0)) / (Xv.std(0) + 1e-9)
    beta = _perturb_beta(BETA_BASE, concept_shift, rng)
    logit = spec.signal_scale * (Xz @ beta)
    Xi = rng.normal(spec.shift, 1.0, size=(n, len(INTERNAL_ONLY)))
    Xiz = (Xi - Xi.mean(0)) / (Xi.std(0) + 1e-9)
    logit = logit + Xiz @ BETA_INTERNAL_ONLY
    b0 = brentq(lambda b: _sigmoid(logit + b).mean() - spec.prevalence, -20, 20)
    p = _sigmoid(logit + b0)
    df["y"] = rng.binomial(1, p).astype(int)
    df["y_true_prob"] = p
    df["patient_id"] = [f"{spec.name}_{i:06d}" for i in range(n)]
    df["cohort"] = spec.name
    df["role"] = spec.role
    df["sex"] = rng.choice(["F", "M"], n, p=[0.45, 0.55])
    df["race"] = rng.choice(["White", "Black", "Asian", "Other"], n, p=[0.6, 0.2, 0.12, 0.08])
    if spec.has_internal_only:
        for j, f in enumerate(INTERNAL_ONLY):
            df[f] = Xi[:, j]
    return df


def make_cohorts(seed: int = 42, cs_temporal: float = 0.5, cs_external: float = 1.3) -> dict[str, pd.DataFrame]:
    """Build the canonical 3-cohort setup used throughout the toolkit.

    Covariate shift AND concept shift grow internal -> temporal -> external, and the
    internal-only features are recorded only in the internal cohort.

    Parameters
    ----------
    seed : int
        Base seed; each cohort derives its own seed from this.
    cs_temporal, cs_external : float
        Concept-shift magnitude (coefficient-rotation strength) for the temporal and
        external cohorts respectively.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Keyed by role: ``"internal"``, ``"temporal"``, ``"external"``.
    """
    specs = [
        (CohortSpec("MIMIC-IV_like", "internal", 4000, 0.30, shift=0.00, seed=seed + 1, has_internal_only=True), 0.00),
        (CohortSpec("MIMIC-III_like", "temporal", 3000, 0.24, shift=0.35, seed=seed + 2, has_internal_only=False), cs_temporal),
        (CohortSpec("eICU_like", "external", 5000, 0.17, shift=0.70, seed=seed + 3, has_internal_only=False), cs_external),
    ]
    return {s.role: generate_cohort(s, cs) for s, cs in specs}
