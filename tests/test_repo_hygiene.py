"""The repo is public: tracked files must not carry local absolute paths (G22).

Builders that stamp input paths into tracked outputs (split files, manifests)
write them repo-relative. KNOWN lists any file allowed to carry one; the
stale-entry check keeps it honest.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATH = re.compile(r"/(home|Users)/[A-Za-z0-9_.-]+/")
KNOWN: set[str] = set()


def _tracked_text_files() -> list[str]:
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files"], capture_output=True,
                         text=True, check=True).stdout.split("\n")
    return [f for f in out if f and not f.endswith((".parquet", ".npz", ".png", ".pdf", ".gz",
                                                    ".zip", ".npy", ".ipynb"))]


def test_no_local_paths_in_tracked_files():
    hits = set()
    for rel in _tracked_text_files():
        path = ROOT / rel
        if not path.is_file() or path.stat().st_size > 50_000_000:
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if LOCAL_PATH.search(text):
            hits.add(rel)
    assert hits - KNOWN == set(), "local absolute paths in tracked files"
    assert KNOWN - hits == set(), "stale entries in KNOWN: remove them"


# Library modules that other scripts import; everything else in scripts/ is a
# command and must do nothing on import (Sept 29: importing the unguarded
# run_tss_extract_capped.py started a GPU extraction).
LIBRARY_SCRIPTS = {"headline_cells.py"}


def test_scripts_do_nothing_on_import():
    unguarded = {p.name for p in (ROOT / "scripts").glob("*.py")
                 if '__name__ == "__main__"' not in p.read_text()}
    assert unguarded - LIBRARY_SCRIPTS == set(), "scripts without a __main__ guard"
    assert LIBRARY_SCRIPTS <= {p.name for p in (ROOT / "scripts").glob("*.py")}


def test_tests_cannot_see_a_gpu():
    """conftest hides every GPU: the local card is shared, the clean-room box has none."""
    import os

    import torch

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert not torch.cuda.is_available()
