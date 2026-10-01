"""G15: nothing overwrites a split file to run a variant.

The May runners swapped a variant into ``data/splits.json`` and restored it in
``finally``; a crash between the two left the wrong split in place for every
later run. Runners now pass the split file's path. The SWAP entries in the
allowlist are the scripts still to retire, with the phase that retires them;
none may remain at the Phase 1 exit gate.
"""
from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCANNED = ("scripts", "src", "analysis")
# A file that names a split file (or the canonical path constant) and also
# writes anything is flagged; each flagged file is reviewed once and listed
# below with what it writes. A regex for "writes to the split" can't see
# through variables, so the scan is broad and the review is the gate.
NAMES = re.compile(r"splits\w*\.json|binary_[\w{}]*\.json|\bSPLITS(_PATH)?\b|\bSPLIT_PATH\b")
WRITE = re.compile(r"\.write_(text|bytes)\(|\bopen\([^)]*['\"][wax]b?['\"]|\.open\(\s*['\"][wax]"
                   r"|shutil\.(copy\w*|move)\(|os\.(replace|rename)\(|\.rename\("
                   r"|write_(cluster_)?splits_json\(|write_binary_subset_json\(")
ALLOWLIST = {
    # Builders: the only writers of split files.
    "scripts/make_splits.py": "builder: the primary files; refuses any other seed",
    "scripts/make_tss_disjoint_split.py": "builder: the disjoint primary; a seed needs its own --out",
    "scripts/make_seed_splits.py": "builder: splits_*seed{N}.json only; refuses seed 42",
    "scripts/make_binary_subsets.py": "builder: binary_*.json",
    # Read a split file; write something else.
    "scripts/check_split_reproduction.py": "writes a scratch split under its tempdir and the cluster TSV",
    "scripts/build_protein_pairs.py": "writes data/leaks/ only",
    "scripts/build_tss_windows.py": "writes the window manifest and its meta",
    "scripts/build_headline_ci.py": "writes table fragments",
    "scripts/build_analysis_artifacts.py": "writes analysis artifacts",
    "scripts/probe_enformer_homology.py": "writes its metrics file (1F: replaced by recompute_all)",
    "src/cluster/mmseqs_cluster.py": "writes the clustering FASTA",
    "analysis/demo/cross_modal.py": "demo outputs",
    "analysis/demo/zero_shot.py": "demo outputs",
    # Swap a variant into data/splits.json and restore it: to retire.
    "scripts/bootstrap_tss_anchored.py": "SWAP; 1D4: thin caller of linear_trainer.stats",
    "scripts/paired_tss_anchored.py": "SWAP; 1D4: thin caller of linear_trainer.stats",
    "scripts/probe_enformer_pooling.py": "SWAP; 1F: replaced by recompute_all",
    "scripts/probe_homology70.py": "SWAP; 1F: replaced by recompute_all",
    "scripts/probe_random_comparators.py": "SWAP; 1F: replaced by recompute_all",
    "scripts/probe_tss_anchored.py": "SWAP; 1F: replaced by recompute_all",
    "scripts/probe_tss_composition.py": "SWAP; 1F: replaced by recompute_all",
    "scripts/probe_tss_disjoint.py": "SWAP; 1F: replaced by recompute_all",
    "scripts/seed_sensitivity.py": "SWAP; 1F: replaced by make_seed_splits + recompute_all",
}


def _offenders() -> set[str]:
    files = [f for d in SCANNED for f in (REPO / d).rglob("*.py")]
    return {f.relative_to(REPO).as_posix() for f in files
            if NAMES.search(t := f.read_text()) and WRITE.search(t)}


def test_no_unreviewed_script_names_a_split_and_writes():
    extra = _offenders() - ALLOWLIST.keys()
    assert not extra, f"names a split file and writes; review and list it: {sorted(extra)}"


def test_the_allowlist_has_no_stale_entries():
    stale = ALLOWLIST.keys() - _offenders()
    assert not stale, f"migrated, drop from ALLOWLIST: {sorted(stale)}"


def _flags(code: str) -> bool:
    return bool(NAMES.search(code) and WRITE.search(code))


def test_the_scan_catches_each_write_idiom():
    # Tamper check: every idiom the Oct 1 review listed must be flagged.
    for code in ("SPLITS_PATH.write_bytes(DISJOINT.read_bytes())",
                 "write_cluster_splits_json(df_clustered, SPLITS, stats, seed=seed)",
                 'write_cluster_splits_json(df_p, DATA / "splits.json", stats_p, seed=args.seed)',
                 "shutil.copy(DISJOINT, SPLITS_PATH)",
                 'with open(SPLITS, "w") as f: json.dump(x, f)',
                 'SPLITS.open("w")',
                 'os.replace(tmp, DATA / "splits_tss_disjoint.json")',
                 'Path("data/splits.json").write_text(s)',
                 'out = DATA / f"binary_{task}.json"; write_binary_subset_json(task, out)'):
        assert _flags(code), code
    assert not _flags('out.write_text(json.dumps(payload))')
    assert not _flags('json.loads((DATA / "splits.json").read_text())')


def test_make_splits_refuses_a_seed_over_the_primary_files(monkeypatch):
    import sys

    import make_splits
    monkeypatch.setattr(sys, "argv", ["make_splits.py", "--seed", "7"])
    import pytest
    with pytest.raises(ValueError, match="primary"):
        make_splits.main()
