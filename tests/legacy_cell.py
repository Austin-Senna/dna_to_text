"""Run one family5 cell under the May 2026 protocol and print its record as JSON.

tests/test_acceptance.py runs this in a subprocess because the May records
were produced at the libraries' default thread count, and the old protocol's
results depend on it (measurements_2026-09.md §4).

The fit path differs from the May code in two ways that don't matter for the
cells checked here, which converge well inside max_iter: a sweep point that
does not converge is ineligible for the pick, and a refit that does not
converge raises.
"""
import json
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from linear_trainer.cell import run_cell  # noqa: E402
from linear_trainer.protocol import V2  # noqa: E402

GRID = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
LEGACY = replace(V2, name="legacy-2026-05", scale=False, dtype="float32", base_grid=GRID,
                 limits={"C": (GRID[0], GRID[-1]), "alpha": (GRID[0], GRID[-1])},
                 tol=None, max_iter=2000, threads=None)

if __name__ == "__main__":
    # A registry key, or a path to a parquet (the May features read from git).
    source = Path(sys.argv[1]) if sys.argv[1].endswith(".parquet") else sys.argv[1]
    res = run_cell(source, "family5", REPO / "data" / "splits.json", LEGACY,
                   pred_dir=Path(tempfile.mkdtemp()))
    print(json.dumps({"hp": res["hp"], "sweep": res["sweep"], "metrics": res["metrics"]}))
