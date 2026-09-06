# Contributing to cmvt

Thanks for your interest in contributing. A few things to know up front:

## Project status

`cmvt` is currently maintained by a single person (Ahmed Fawaz), in spare
time. That means review and response time on issues and PRs may be slow —
please be patient, and feel free to ping a stale PR/issue after a couple of
weeks if you haven't heard back.

## Before you propose a new metric or score

This toolkit deliberately refuses to ship certain outputs on statistical
grounds — not because nobody thought of them. Read the
["What this toolkit refuses to claim"](README.md#what-this-toolkit-refuses-to-claim)
section of the README before proposing, for example:

- an MNAR-likelihood or MNAR-detection score,
- a composite "dataset quality" score,
- a values-only leakage score (i.e. one that doesn't use provenance).

Each of these is backed by an executable proof in
`tests/test_negative_controls.py` showing the quantity is not identifiable
from the data alone. A PR adding one of these will likely be declined for the
reasons documented there, not out of neglect — so please read that section
first and, if you still think there's a gap, open an issue explaining what
the existing negative-control argument misses.

## Development setup

```bash
git clone https://github.com/ahmedfawaz879/clinical-model-validation-toolkit.git
cd clinical-model-validation-toolkit
pip install -e ".[dev]"
```

This installs `cmvt` in editable mode along with the dev toolchain: pytest,
pytest-cov, ruff, mypy, pre-commit, and the notebook-execution dependencies
(nbclient, nbconvert, ipykernel).

## Running the test suite

Before opening a PR, run:

```bash
pytest
```

CI enforces `--cov-fail-under=90` (see `.github/workflows/ci.yml`), so any new
code needs accompanying tests — a PR that drops coverage below 90% will fail
CI. `pyproject.toml` already configures `pytest` to compute coverage over
`src/cmvt` with a terminal + XML report, so plain `pytest` locally will show
you the same coverage numbers CI checks.

## Linting and type-checking

Ruff and mypy are configured in `pyproject.toml` and wired up via
`.pre-commit-config.yaml`. The easiest way to run everything CI/pre-commit
would run is:

```bash
pre-commit install   # one-time, sets up the git hook
pre-commit run --all-files
```

This runs `ruff` (with `--fix`) and `ruff-format` on the whole repo, and
`mypy` on `src/`. You can also run the checks directly:

```bash
ruff check src tests scripts examples
mypy src/cmvt
```

## Continuous integration

CI (`.github/workflows/ci.yml`) runs on every push and PR to `main`, across
Python 3.10, 3.11, and 3.12. It:

- installs the package with `pip install -e ".[dev]"`,
- runs `pytest --cov-fail-under=90`,
- runs the clone-and-run demo (`examples/run_synthetic_demo.py`) and checks
  it produces `reports/validation_report.md`,
- regenerates figures via `scripts/generate_figures.py`,
- and, in a separate job, runs `ruff check` and `mypy`.

All of this must pass on Python 3.10–3.12 before a PR can be merged.

## Filing an issue

Please use [GitHub Issues](https://github.com/ahmedfawaz879/clinical-model-validation-toolkit/issues)
on this repository. For bug reports, include a minimal reproducible example
(ideally a short script or snippet using the bundled synthetic data) —
that makes it far more likely the issue can be diagnosed and fixed quickly.

## Proposing a pull request

1. Fork the repository and create a branch off `main`.
2. Keep the PR small and scoped to a single change — separate unrelated
   fixes/features into separate PRs.
3. Make sure `pytest` and `pre-commit run --all-files` pass locally.
4. In the PR description, explain what changed and why, and link any related
   issue.

Thanks again for considering a contribution.
