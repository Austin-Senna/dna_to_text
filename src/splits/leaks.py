"""The evaluation purge (G2): genes masked from val and test scoring.

A val or test gene with a near-copy on the training side can score well by
remembering its partner, not by reading its own sequence. Two rules find such
pairs:

- ``protein@<id>``: Rule A, the clustering rule (E <= 1e-3, identity >= id,
  coverage >= 0.8 on both sequences), from an all-vs-all MMseqs2 search over
  the full-length proteins (``data/leaks/protein_pairs.tsv``, built by
  ``scripts/build_protein_pairs.py``). The table holds every pair at identity
  >= 0.4; stricter rules filter it.
- ``window``: TSS windows that overlap on the chromosome
  (``splits.window_leak.overlap_pairs`` on ``data/tss_windows.tsv``).

A val gene leaks when a train gene matches it (selection fits on train). A
test gene leaks when a train or val gene matches it (the refit uses
train+val). Training genes are never masked, so the fits are unchanged and
only the scored sets shrink.

Which rules apply depends on the split file and the arm (``rules_for``).
Every split the recompute uses is named there; an unknown split raises
instead of running unpurged.
"""
from __future__ import annotations

import csv
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
PAIRS = DATA / "leaks" / "protein_pairs.tsv"
WINDOWS = DATA / "tss_windows.tsv"
PAIR_MIN_ID = 0.40
SPLITS = ("train", "val", "test")

_PROTEIN = re.compile(r"protein@(0\.\d+)")


@dataclass(frozen=True)
class Purge:
    """The genes one cell masks from val selection and test scoring."""
    rules: tuple[str, ...]
    val: frozenset[str] = frozenset()
    test: frozenset[str] = frozenset()
    stamp: dict = field(default_factory=dict)

    def record(self) -> dict:
        """The record's ``purge`` field: enough to re-derive and to rescore."""
        return {"rules": list(self.rules), **self.stamp,
                "val_masked": sorted(self.val), "test_masked": sorted(self.test)}


NONE = Purge(rules=())


def rules_for(split_name: str, arm: str) -> tuple[str, ...]:
    """The purge rules for a split file and an arm ("cds" or "tss").

    - Homology splits (primary, seeds): Rule A at 40%.
    - The 70% split: Rule A at its own 70%.
    - The disjoint split and its seeds: Rule A and windows (both arms; the
      windows are disjoint by construction, so that mask should be empty).
    - TSS on the homology split is the sensitivity arm for window overlap
      itself, so only Rule A is masked there (decided Oct 1).
    - The random split is the leakage demonstration: not purged. The binary
      task subsets are not in the paper and are not purged either.
    """
    if arm not in ("cds", "tss"):
        raise ValueError(f"arm must be 'cds' or 'tss', got {arm!r}")
    if split_name == "splits_random.json":
        return ()
    if re.fullmatch(r"binary_\w+\.json", split_name):
        return ()   # random stratified subsets, not in the paper: unpurged by design
    if split_name == "splits_homology70.json":
        return ("protein@0.70",)
    if re.fullmatch(r"splits_tss_disjoint(_seed\d+)?\.json", split_name):
        return ("protein@0.40", "window")
    if re.fullmatch(r"splits(_seed\d+)?\.json", split_name):
        return ("protein@0.40",)
    raise KeyError(f"no purge rule for {split_name}: name it in leaks.rules_for")


def arm_of(source: str | Path) -> str:
    """The arm a feature source belongs to: "tss" for TSS-window and Enformer
    features, "cds" for everything else."""
    name = Path(source).stem.removeprefix("dataset_") if isinstance(source, Path) else source
    return "tss" if name.startswith(("tss_", "enformer_")) else "cds"


class UncoveredGenes(RuntimeError):
    """A split holds genes the pair table was not built over."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    return path.relative_to(REPO_ROOT).as_posix() if path.is_relative_to(REPO_ROOT) else str(path)


def read_protein_pairs(min_id: float, path: Path = PAIRS) -> list[tuple[str, str]]:
    """Unordered gene pairs passing Rule A at ``min_id`` identity."""
    if min_id < PAIR_MIN_ID:
        raise ValueError(f"{path.name} holds pairs at identity >= {PAIR_MIN_ID} only")
    with Path(path).open() as f:
        return [(r["gene_a"], r["gene_b"]) for r in csv.DictReader(f, delimiter="\t")
                if float(r["fident"]) >= min_id]


def window_pairs(path: Path = WINDOWS) -> list[tuple[str, str]]:
    from data_loader.enformer_windows import window_spans
    from splits.window_leak import overlap_pairs
    return [(a, b) for a, b, _ in overlap_pairs(window_spans(path))]


def leaky(split: dict, pairs: list[tuple[str, str]]) -> tuple[set[str], set[str]]:
    """(val genes with a train partner, test genes with a train or val partner)."""
    where = {g: s for s in SPLITS for g in split[s]}
    val, test = set(), set()
    for a, b in pairs:
        sa, sb = where.get(a), where.get(b)
        if sa is None or sb is None or sa == sb:
            continue
        for g, s, other in ((a, sa, sb), (b, sb, sa)):
            if s == "val" and other == "train":
                val.add(g)
            elif s == "test":
                test.add(g)
    return val, test


def purge_for(split_path: Path, arm: str, *, pairs_path: Path = PAIRS,
              windows_path: Path = WINDOWS) -> Purge:
    """The purge for one cell, from its split file, its arm and the tracked pair tables."""
    import json

    split_path = Path(split_path)
    rules = rules_for(split_path.name, arm)
    if not rules:
        return NONE
    split = json.loads(split_path.read_text())
    val, test, stamp = set(), set(), {"split_sha256": _sha256(split_path)}
    for rule in rules:
        m = _PROTEIN.fullmatch(rule)
        if m:
            _check_universe(split, split_path, pairs_path)
            pairs = read_protein_pairs(float(m.group(1)), pairs_path)
            stamp.update(pairs_file=_rel(pairs_path), pairs_sha256=_sha256(pairs_path))
        elif rule == "window":
            pairs = window_pairs(windows_path)
            stamp.update(windows_file=_rel(windows_path), windows_sha256=_sha256(windows_path))
        else:
            raise ValueError(f"unknown purge rule {rule!r}")
        v, t = leaky(split, pairs)
        val |= v
        test |= t
    return Purge(rules=rules, val=frozenset(val), test=frozenset(test), stamp=stamp)


def _check_universe(split: dict, split_path: Path, pairs_path: Path) -> None:
    """Every gene the split assigns must have been in the all-vs-all search."""
    import json

    meta = json.loads(Path(pairs_path).with_suffix(".json").read_text())["proteins"]["gene_universe"]
    universe_file = REPO_ROOT / meta["path"]
    if _sha256(universe_file) != meta["sha256"]:
        raise UncoveredGenes(f"{meta['path']} changed since {Path(pairs_path).name} was built")
    universe = json.loads(universe_file.read_text())
    have = {g for s in SPLITS for g in universe[s]}
    extra = {g for s in SPLITS for g in split[s]} - have
    if extra:
        raise UncoveredGenes(f"{split_path.name}: {len(extra)} genes outside the pair table's "
                             f"universe: {sorted(extra)[:5]}")
