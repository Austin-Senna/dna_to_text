"""G6: every probe fit goes through ``linear_trainer.fit``.

A second fit path is how protocols drift apart (max_iter 3000 vs 5000, unscaled
refits, bootstraps centred on other fits). The allowlist names the scripts
still to migrate and the phase that migrates them; it must shrink to the
out-of-scope entries by the Phase 1 exit gate.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIT_PATH = "src/linear_trainer/fit.py"
PATTERN = re.compile(r"\bLogisticRegression(CV)?\(|\bRidge(CV)?\(|\bSGD(Classifier|Regressor)\("
                     r"|\bLinearSVC\(|\bsweep_C\b|\bsweep_alpha\b|\bDEFAULT_CS\b|\bDEFAULT_ALPHAS\b"
                     r"|from sklearn\.(linear_model|svm) import|import sklearn\.(linear_model|svm)")
SCANNED = ("scripts", "src", "analysis")
ALLOWLIST = {
    "scripts/probe_enformer_homology.py": "1D: Enformer cells through run_cell",
    "scripts/build_full_table.py": "1F: off-path, delete (needs approval)",
    "scripts/compute_kappa.py": "1F: off-path, delete (needs approval)",
    "scripts/train_mlp_probe.py": "out of scope: MLP probe, not a linear probe",
    "analysis/demo/cross_modal.py": "out of scope: demo, not a paper number",
    "analysis/demo/zero_shot.py": "out of scope: demo, not a paper number",
}


def _offenders() -> set[str]:
    files = [f for d in SCANNED for f in (REPO / d).rglob("*.py")]
    rel = {f.relative_to(REPO).as_posix(): f for f in files}
    return {r for r, f in rel.items() if r != FIT_PATH and PATTERN.search(f.read_text())}


def test_no_probe_is_fitted_outside_the_single_fit_path():
    extra = _offenders() - ALLOWLIST.keys()
    assert not extra, f"fit or grid outside {FIT_PATH}: {sorted(extra)}"


def test_the_allowlist_has_no_stale_entries():
    stale = ALLOWLIST.keys() - _offenders()
    assert not stale, f"migrated, drop from ALLOWLIST: {sorted(stale)}"
