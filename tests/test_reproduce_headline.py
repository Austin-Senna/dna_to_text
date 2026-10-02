"""Rule 3: the independent reimplementation stays independent, and it can fail.

``scripts/reproduce_headline.py`` was written from a spec without the pipeline's
code. If it imported a repo module, a bug there would sit on both sides of the
comparison; the scan below keeps it to the standard library, numpy, pandas and
scikit-learn. The slow test tampers with a copy of real records and checks that
the comparison catches it.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "reproduce_headline.py"
ALLOWED = {"numpy", "pandas", "sklearn"}


def foreign_imports(source: str) -> set[str]:
    """Top-level modules imported that are neither stdlib nor ALLOWED."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            found.add("." if node.level else (node.module or "").split(".")[0])
    return found - ALLOWED - set(sys.stdlib_module_names) - {"__future__"}


def test_the_reimplementation_imports_nothing_from_the_repo():
    assert foreign_imports(SCRIPT.read_text()) == set()


@pytest.mark.parametrize("line", ["from linear_trainer import fit", "import splits.leaks",
                                  "from .records import load", "    from data_loader import x"])
def test_the_import_scan_catches_a_repo_import(line):
    src = SCRIPT.read_text().replace("import numpy as np\n", f"import numpy as np\n{line.strip()}\n", 1)
    assert foreign_imports(src)


# Records to tamper with: the canonical data/v2, or another run's (e.g. a trial) via
# REPRODUCE_RECORDS. Cell 5 (the 4-mer) is the fastest headline cell.
RECORDS = Path(os.environ.get("REPRODUCE_RECORDS", REPO / "data" / "v2"))
THREADS = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def _run(records: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "--records", str(records), "--cells", "5",
                           "--n-boot", "200", "--out", str(out)],
                          env={**os.environ, **THREADS}, capture_output=True, text=True, cwd=REPO)


def _copy(tmp_path: Path) -> Path:
    dst = tmp_path / "records"
    dst.mkdir()
    for f in [*RECORDS.glob("metrics_splits.json"), *RECORDS.glob("metrics_splits_tss_disjoint.json"),
              *RECORDS.glob("null_splits*.json")]:
        shutil.copy(f, dst / f.name)
    return dst


def _kmer(recs: list[dict]) -> dict:
    return next(r for r in recs if r["feature_source"] == "kmer" and r["task"] == "family5"
                and r["arm"] == "cds")


@pytest.mark.slow
@pytest.mark.skipif(not (RECORDS / "metrics_splits.json").exists(), reason="no records to check")
@pytest.mark.parametrize("tamper", ["none", "prediction", "val_score"])
def test_a_tampered_record_fails_the_reproduction(tmp_path, tamper):
    # Sweep points reproduce bit for bit only on the BLAS kernel that fitted them
    # (measurements §8a: c6a records refitted on this laptop differ at non-picked C),
    # and there a nudged sweep point can't be told from kernel drift. Point
    # REPRODUCE_RECORDS at records fitted on this machine (e.g. the Phase 5 run).
    from linear_trainer.protocol import stamp
    fitted_on = _kmer(json.loads((RECORDS / "metrics_splits.json").read_text()))["stamp"]["machine"]["cpu"]
    if fitted_on != stamp()["machine"]["cpu"]:
        pytest.skip(f"records fitted on {fitted_on}, not this CPU; set REPRODUCE_RECORDS")
    records = _copy(tmp_path)
    path = records / "metrics_splits.json"
    recs = json.loads(path.read_text())
    rec = _kmer(recs)
    if tamper == "prediction":       # flip one stored test prediction to another class
        arrays = dict(np.load(REPO / rec["pred_file"], allow_pickle=False))
        classes = sorted(set(arrays["y_true"].tolist()))
        arrays["pred"][0] = next(c for c in classes if c != arrays["pred"][0])
        rec["pred_file"] = str(tmp_path / "tampered.npz")
        np.savez(rec["pred_file"], **arrays)
    elif tamper == "val_score":      # nudge one sweep point the pick didn't land on
        row = next(r for r in rec["C_sweep"] if r["C"] != rec["C"])
        row["macro_f1"] += 1e-6
    path.write_text(json.dumps(recs))
    res = _run(records, tmp_path / "reproduction.json")
    verdict = json.loads((tmp_path / "reproduction.json").read_text())
    assert (res.returncode == 0) is (tamper == "none"), res.stdout[-2000:]
    assert verdict["ok"] is (tamper == "none")
