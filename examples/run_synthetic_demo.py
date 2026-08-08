#!/usr/bin/env python
"""End-to-end demo on the bundled synthetic cohorts. Same pipeline as `cmvt-demo`.

Run directly with:

    python examples/run_synthetic_demo.py

Exits 0 and writes ``reports/validation_report.md`` on success -- this is the
command CI runs on a clean checkout to enforce the clone-and-run guarantee (see
README.md, "Reproducing every number").
"""

from __future__ import annotations

import sys

from cmvt.cli import main

if __name__ == "__main__":
    sys.exit(main())
