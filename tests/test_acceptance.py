"""Real-data acceptance for the probe core (run with ``uv run pytest -m slow``).

LEGACY is the May 26-27 protocol that produced ``metrics_homology.json`` (no
scaler, float32, 6-point grid, lbfgs defaults; tests/legacy_cell.py). The
refactored fit path must reproduce those records exactly. They were produced
at the default thread count on the Ryzen 9 7940HX, and the old protocol is
thread-dependent: at 1 thread the 4-mer gives 0.6227, not 0.6328. So this
check runs in a subprocess at default threads, on that machine only. The
Sept 30 re-extraction replaced the ESM-2 parquet, so the check reads the May
features from git at ``MAY_FEATURES_COMMIT``, verified by sha256.

V2 is the camera-ready protocol: with pooling fixed at today's picks it must
reproduce the Sept 25 preview (``measurements_2026-09.md`` §2, "new" rows) to
4 dp with identical picks, on the current features. Where the Sept 30
re-extraction moved a value, the target is re-pinned and the comment says why.
"""
from __future__ import annotations

import hashlib
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
# The last commit whose dataset parquets are the May features (before the
# Sept 30 rebuild). Only parquet-backed sources need it; kmer is computed
# from data/sequences/, which the rebuild did not touch.
MAY_FEATURES_COMMIT = "1c884ef40bc5a53eab9ddb6eb44b8762effaa646"
MAY_FEATURES_SHA256 = {
    "esm2_650m": "b978306f9a1981b6ee442ee789598d4346b6f58f260b8bf351b1b1fad7b8ca63",
}
pytestmark = pytest.mark.slow


def _recorded(source: str) -> dict:
    recs = json.loads((REPO / "data" / "metrics_homology.json").read_text())
    return [r for r in recs if r.get("feature_source") == source
            and r.get("task") == "family5" and not r.get("shuffled_labels")][-1]


def _may_features(source: str, tmp_path: Path) -> Path:
    """Write the May parquet for ``source`` from git into tmp_path and verify it."""
    out = subprocess.run(["git", "show", f"{MAY_FEATURES_COMMIT}:data/dataset_{source}.parquet"],
                         capture_output=True, cwd=REPO)
    if out.returncode != 0:
        pytest.skip(f"the May features are not in this checkout ({MAY_FEATURES_COMMIT[:7]})")
    assert hashlib.sha256(out.stdout).hexdigest() == MAY_FEATURES_SHA256[source]
    path = tmp_path / f"{source}.parquet"
    path.write_bytes(out.stdout)
    return path


@pytest.mark.parametrize("source", ["kmer", "esm2_650m"])
def test_legacy_reproduces_the_may_records(source, tmp_path):
    if MAY_CPU not in stamp()["machine"]["cpu"]:
        pytest.skip("the May records are specific to the machine that produced them")
    arg = str(_may_features(source, tmp_path)) if source in MAY_FEATURES_SHA256 else source
    env = {k: v for k, v in os.environ.items()
           if k not in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    out = subprocess.run([sys.executable, str(REPO / "tests" / "legacy_cell.py"), arg],
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
    # Sept 25 preview 0.948796 on the May features. 0.956069 on the Sept 30
    # re-extraction: the 17 full-length proteins (G5) give 0.953100, and fp32
    # instead of the May fp16 gives the rest. Same pick; 4 test genes flip.
    ("esm2_650m", "family5", 100.0, 0.9561, "test_macro_f1"),
    ("dnabert2_meanD", "genept", 1e4, 0.0752, "test_r2_macro"),
    # Sept 25 preview 0.093715 on first-stop proteins; 0.093628 once the 17
    # truncated translations run full length (G5, Sept 29). AA2 is unchanged.
    ("aa3", "genept", 1e4, 0.0936, "test_r2_macro"),
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
