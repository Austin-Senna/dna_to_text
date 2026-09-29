"""Real-data acceptance for the probe core (run with ``uv run pytest -m slow``).

LEGACY is the May 26-27 protocol that produced ``metrics_homology.json`` (no
scaler, float32, 6-point grid, lbfgs defaults; tests/legacy_cell.py). The
refactored fit path must reproduce those records exactly. They were produced
at the default thread count on the Ryzen 9 7940HX, and the old protocol is
thread-dependent: at 1 thread the 4-mer gives 0.6227, not 0.6328. So this
check runs in a subprocess at default threads, on that machine only.

V2 is the camera-ready protocol: with pooling fixed at today's picks it must
reproduce the Sept 25 preview (``measurements_2026-09.md`` §2, "new" rows) to
4 dp with identical picks.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from linear_trainer.protocol import stamp

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / "data" / "splits.json"
MAY_CPU = "AMD Ryzen 9 7940HX"
pytestmark = pytest.mark.slow


def _recorded(source: str) -> dict:
    recs = json.loads((REPO / "data" / "metrics_homology.json").read_text())
    return [r for r in recs if r.get("feature_source") == source
            and r.get("task") == "family5" and not r.get("shuffled_labels")][-1]


@pytest.mark.parametrize("source", ["kmer", "esm2_650m"])
def test_legacy_reproduces_the_may_records(source):
    if MAY_CPU not in stamp()["machine"]["cpu"]:
        pytest.skip("the May records are specific to the machine that produced them")
    env = {k: v for k, v in os.environ.items()
           if k not in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    out = subprocess.run([sys.executable, str(REPO / "tests" / "legacy_cell.py"), source],
                         env=env, capture_output=True, text=True, check=True, cwd=REPO)
    res = json.loads(out.stdout.strip().splitlines()[-1])
    old = _recorded(source)
    assert res["hp"] == old["C"]
    assert res["metrics"]["test_macro_f1"] == pytest.approx(old["test_macro_f1"], abs=1e-6)
    for got, want in zip(res["sweep"], old["C_sweep"], strict=True):
        assert got["C"] == want["C"]
        assert got["macro_f1"] == pytest.approx(want["macro_f1"], abs=1e-6)


V2_TARGETS = [
    # source, task, pick, target (4 dp), metric
    ("nt_v2_specialmean", "family5", 1.0, 0.7025, "test_macro_f1"),
    ("aa2", "family5", 0.01, 0.7591, "test_macro_f1"),
    ("kmer", "family5", 0.01, 0.6508, "test_macro_f1"),
    ("esm2_650m", "family5", 100.0, 0.9488, "test_macro_f1"),
    ("dnabert2_meanD", "genept", 1e4, 0.0752, "test_r2_macro"),
    ("aa3", "genept", 1e4, 0.0937, "test_r2_macro"),
]


@pytest.mark.parametrize("source,task,pick,target,metric", V2_TARGETS)
def test_v2_reproduces_the_sept25_preview(source, task, pick, target, metric, tmp_path):
    from linear_trainer.cell import run_cell
    from linear_trainer.protocol import V2
    res = run_cell(source, task, SPLITS, V2, pred_dir=tmp_path)
    print(f"{source} {task}: pick={res['hp']:g} edge={res['edge']} "
          f"{metric}={res['metrics'][metric]:.6f} (target {target})")
    assert res["hp"] == pytest.approx(pick)
    assert res["edge"] is False
    assert res["metrics"][metric] == pytest.approx(target, abs=5e-5)
