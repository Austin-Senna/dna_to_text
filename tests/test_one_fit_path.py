"""G6: every probe fit goes through ``linear_trainer.fit``.

A second fit path is how protocols drift apart (max_iter 3000 vs 5000, unscaled
refits, bootstraps centred on other fits). Since Phase 1F the allowlist holds
only out-of-scope entries (the Phase 1 exit gate), and a test keeps it so.
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
    "scripts/train_mlp_probe.py": "out of scope: MLP probe, not a linear probe",
    "analysis/demo/cross_modal.py": "out of scope: demo, not a paper number",
    "analysis/demo/zero_shot.py": "out of scope: demo, not a paper number",
    # Rule 3: a second fit path on purpose, written without this one, so a bug in
    # one can't hide in both. It produces no paper number, only reproduction.json.
    "scripts/reproduce_headline.py": "independent check: the Rule 3 reimplementation",
}
INDEPENDENT = {"scripts/reproduce_headline.py"}


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


def test_the_allowlist_holds_only_out_of_scope_entries():
    # Phase 1 exit gate: no paper-path script may be waiting to migrate.
    pending = {k for k, why in ALLOWLIST.items() if not why.startswith("out of scope:")
               and not (k in INDEPENDENT and why.startswith("independent check:"))}
    assert not pending, f"paper-path fits outside {FIT_PATH}: {sorted(pending)}"
