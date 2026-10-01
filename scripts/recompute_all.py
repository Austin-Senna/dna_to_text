"""The camera-ready recompute: every probe cell, from one manifest (Phase 1D).

A cell is (split file, arm, task, source[, label seed]). The manifest is built
from the registry, so a pool, encoder or split added there is picked up here.
Each cell runs through ``linear_trainer.cell.run_cell`` in this process, with
its split file passed by path (no split file is ever swapped, G15) and its
evaluation purge from ``splits.leaks.purge_for`` (G2).

Records go to ``data/v2/metrics_<split stem>.json`` (one JSON array per split
file), predictions to ``outputs/predictions/v2/<split stem>/``. Null-band
shuffles go to ``data/v2/null_<split stem>.json``.

Guards:
  * A cell that raises, ``SystemExit`` included, fails the run (G17); at the
    end, every manifest key must have exactly one record.
  * A run resumes only into files from the same git commit and protocol; mixing
    records from two commits is refused (G7).
  * After the run, three cells are re-selected with their test labels permuted;
    their picks must not move (G1, black-box).

Run (normally via scripts/recompute_all.sh, which pins the threads):
  uv run scripts/recompute_all.py [--group main|null|all] [--only SUBSTR] [--dry-run]
"""
from __future__ import annotations

import argparse
import fcntl
import json
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data_loader.model_registry import ENCODER_SPECS, encoder_pools
from linear_trainer import sources
from linear_trainer.cell import run_cell
from linear_trainer.protocol import V2, stamp
from linear_trainer.records import COMPLETE
from splits.leaks import purge_for

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
OUT_DIR = DATA / "v2"
PRED_ROOT = REPO_ROOT / "outputs" / "predictions" / "v2"

ENCODERS = tuple(ENCODER_SPECS)
SEEDS = (1, 7, 123)
TASKS = ("family5", "genept")
COMPOSITION = ("kmer", "kmer6", "codon", "aa1", "aa2", "aa3", "gc")
ESM2 = ("esm2_150m", "esm2_650m")
NULL_SHUFFLES = 200

# Split files by role.
CDS_PRIMARY = "splits.json"
TSS_PRIMARY = "splits_tss_disjoint.json"
CDS_SPLITS = (CDS_PRIMARY, TSS_PRIMARY, "splits_homology70.json", "splits_random.json",
              *(f"splits_seed{s}.json" for s in SEEDS))
TSS_SPLITS_FULL = (TSS_PRIMARY, CDS_PRIMARY)                 # with the E5 cells
# The random split's TSS grid feeds the split comparison (random vs homology).
TSS_SPLITS_GRID = ("splits_random.json", *(f"splits_tss_disjoint_seed{s}.json" for s in SEEDS))


def cds_sources() -> list[str]:
    pooled = [f"{e}_{p}" for e in ENCODERS for p in encoder_pools(e, "CDS")]
    return pooled + list(COMPOSITION) + list(ESM2)


def tss_sources() -> list[str]:
    pooled = [f"tss_{e}_{p}" for e in ENCODERS for p in encoder_pools(e, "TSS")]
    # Enformer: whole-window everywhere; the centre readout is the E5 counterpart.
    return pooled + ["enformer_tss_4mer", "enformer_trunk_global", "enformer_trunk_center"]


def e5_sources() -> list[str]:
    anchored = [f"tss_{e}_tssanchored" for e in ENCODERS]
    comp = [f"tss_{e}_{f}" for e in ENCODERS for f in ("chunk4mergc", "chunk6mer")]
    return anchored + comp


class DirtyTree(RuntimeError):
    """The canonical outputs are written only from a committed tree (G7, G12)."""


@dataclass(frozen=True)
class Cell:
    split: str
    arm: str
    task: str
    source: str
    label_seed: int | None = None

    @property
    def key(self) -> str:
        shuf = "" if self.label_seed is None else f"/shuf{self.label_seed}"
        return f"{self.split}/{self.arm}/{self.task}/{self.source}{shuf}"

    def out(self, out_dir: Path) -> Path:
        stem = Path(self.split).stem
        return out_dir / (f"null_{stem}.json" if self.label_seed is not None else f"metrics_{stem}.json")


def main_cells() -> list[Cell]:
    cells = [Cell(s, "cds", t, src) for s in CDS_SPLITS for t in TASKS for src in cds_sources()]
    for s in TSS_SPLITS_FULL:
        cells += [Cell(s, "tss", t, src) for t in TASKS for src in tss_sources() + e5_sources()]
    for s in TSS_SPLITS_GRID:
        cells += [Cell(s, "tss", t, src) for t in TASKS for src in tss_sources()]
    return cells


def null_cells() -> list[Cell]:
    """Null bands (decided Oct 1): the 4-mer on each primary split and task, plus
    the ESM-2 650M family5 spot check on the CDS primary."""
    bands = [(CDS_PRIMARY, "cds", t, "kmer") for t in TASKS]
    bands += [(TSS_PRIMARY, "tss", t, "enformer_tss_4mer") for t in TASKS]
    bands += [(CDS_PRIMARY, "cds", "family5", "esm2_650m")]
    return [Cell(*b, label_seed=k) for b in bands for k in range(NULL_SHUFFLES)]


def manifest(group: str) -> list[Cell]:
    cells = {"main": main_cells, "null": null_cells,
             "all": lambda: main_cells() + null_cells()}[group]()
    keys = [c.key for c in cells]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate manifest keys")
    return cells


class MixedRun(RuntimeError):
    """An output file holds records from another commit or protocol (G7)."""


class IncompleteRun(RuntimeError):
    """Manifest cells without exactly one record (G17)."""


def shard(cells: list[Cell], spec: str) -> list[Cell]:
    """Cells ``i, i+n, i+2n, ...`` for ``spec = "i/n"``; the n shards partition ``cells``."""
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"--shard {spec}: need 0 <= I < N")
    return cells[i::n]


def _load(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def _append(out: Path, rec: dict) -> None:
    """Append one record atomically, under an exclusive lock, so shards that
    split one split file's cells (``--only``) can share its records file."""
    with open(out.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        recs = _load(out)
        if any(r["key"] == rec["key"] for r in recs):
            raise RuntimeError(f"{rec['key']} is already recorded in {out.name}")
        here = (rec["stamp"]["git_sha"], rec["protocol_hash"])
        if recs and (recs[0]["stamp"]["git_sha"], recs[0]["protocol_hash"]) != here:   # G7
            raise MixedRun(f"{out.name} holds records from another commit or protocol than "
                           f"{rec['key']}; shards must run from one commit")
        recs.append(rec)
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(recs, indent=2))
        tmp.replace(out)


def _check_same_run(out_dir: Path) -> None:
    """Every records file in out_dir must come from this commit, this protocol and
    the feature files on disk now, whichever group this invocation runs (G7)."""
    from linear_trainer.cell import _features_stamp

    here = stamp()["git_sha"]
    if here is None:
        raise DirtyTree("git can't read this tree; the records would carry no commit")
    current: dict[str, dict] = {}
    for path in sorted(out_dir.glob("metrics_*.json")) + sorted(out_dir.glob("null_*.json")):
        for r in _load(path):
            if r["stamp"]["git_sha"] != here or r["protocol_hash"] != V2.hash:
                raise MixedRun(f"{path.name} has records from {str(r['stamp']['git_sha'])[:7]} "
                               f"(protocol {r['protocol_hash'][:8]}); this run is {here[:7]}. "
                               "Move the old files aside to start fresh.")
            src = r["feature_source"]
            if src not in current:
                current[src] = _features_stamp(src)
            if r["features"] != current[src]:
                raise MixedRun(f"{path.name}: {r['key']} was fitted on other features than "
                               f"{src} has now; move the old files aside to start fresh")


def run_one(cell: Cell, pred_root: Path) -> dict:
    split_path = DATA / cell.split
    purge = purge_for(split_path, cell.arm)
    res = run_cell(cell.source, cell.task, split_path, V2,
                   pred_dir=pred_root / Path(cell.split).stem,
                   label_seed=cell.label_seed, purge=purge)
    return {
        "key": cell.key, "split": cell.split, "arm": cell.arm, "task": cell.task,
        "feature_source": cell.source, "shuffled_labels": cell.label_seed is not None,
        "label_seed": cell.label_seed,
        # The legacy key names, so selection.val_score and the builders read v2
        # records unchanged.
        **({"C": res["hp"], "C_sweep": res["sweep"]} if cell.task == "family5" else
           {"alpha": res["hp"], "alpha_sweep": res["sweep"], "select_by": "r2"}),
        "edge": res["edge"], "feature_dim": res["feature_dim"],
        **res["metrics"], **res["provenance"],
    }


def black_box_g1(cells: list[Cell], pred_root: Path) -> list[dict]:
    """Re-select with the test labels permuted: the pick must not move (G1)."""
    real_load = sources.load

    def permuted(source, task, split, splits_path):
        X, y, ids = real_load(source, task, split, splits_path)
        if split == "test":
            y = np.random.default_rng(0).permutation(y)
        return X, y, ids

    checked = []
    for cell in cells:
        # Separate directories: the permuted run's targets differ by design, and the
        # shared GenePT targets file refuses different content for one split.
        base = run_cell(cell.source, cell.task, DATA / cell.split, V2,
                        pred_dir=pred_root / "g1_check" / "base",
                        purge=purge_for(DATA / cell.split, cell.arm))
        sources.load = permuted
        try:
            perm = run_cell(cell.source, cell.task, DATA / cell.split, V2,
                            pred_dir=pred_root / "g1_check" / "permuted",
                            purge=purge_for(DATA / cell.split, cell.arm))
        finally:
            sources.load = real_load
        if (base["hp"], base["sweep"]) != (perm["hp"], perm["sweep"]):
            raise RuntimeError(f"G1: {cell.key} selection moved when the test labels were permuted")
        print(f"  G1 ok: {cell.key} pick {base['hp']:g} unchanged under permuted test labels", flush=True)
        checked.append({"key": cell.key, "hp": base["hp"]})
    return checked


def write_complete(out_dir: Path, cells: list[Cell], g1: list[dict]) -> None:
    """The marker builders require (G17, G1): written only by an unsharded run over
    the whole manifest, after every cell has exactly one record and G1 passed."""
    recs = [r for out in sorted({c.out(out_dir) for c in cells}) for r in _load(out)]
    stamps = {(r["stamp"]["git_sha"], r["protocol_hash"]) for r in recs}
    if len(stamps) != 1:
        raise MixedRun(f"the records span {len(stamps)} commit/protocol stamps")
    (sha, proto), = stamps
    payload = {"git_sha": sha, "protocol_hash": proto, "n_cells": len(cells), "g1": g1}
    tmp = out_dir / (COMPLETE + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(out_dir / COMPLETE)
    print(f"wrote {COMPLETE}: {len(cells)} cells, G1 on {len(g1)}")


G1_CELLS = (Cell(CDS_PRIMARY, "cds", "family5", "kmer"),
            Cell(CDS_PRIMARY, "cds", "genept", "aa3"),
            Cell(TSS_PRIMARY, "tss", "family5", "tss_nt_v2_meanD"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", choices=["main", "null", "all"], default="main")
    ap.add_argument("--only", default=None, help="run only cells whose key contains this")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="run every N-th cell from the I-th (0-based), after --only; shards "
                         "share records files safely (locked appends)")
    ap.add_argument("--dry-run", action="store_true", help="print the manifest size and exit")
    ap.add_argument("--skip-g1", action="store_true", help="skip the G1 black-box check")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR,
                    help="records directory (default data/v2; anything else is a trial run)")
    ap.add_argument("--pred-root", type=Path, default=None,
                    help="predictions directory (default outputs/predictions/v2 for the canonical "
                         "run, <out-dir>/predictions for a trial)")
    args = ap.parse_args()
    out_dir = args.out_dir
    canonical = out_dir.resolve() == OUT_DIR.resolve()
    pred_root = args.pred_root or (PRED_ROOT if canonical else out_dir / "predictions")
    if not canonical and pred_root.resolve() == PRED_ROOT.resolve():
        raise DirtyTree("a trial run must not write into the canonical predictions directory")

    cells = manifest(args.group)
    if args.only:
        cells = [c for c in cells if args.only in c.key]
    if args.shard:
        cells = shard(cells, args.shard)
    if args.dry_run:
        by = {}
        for c in cells:
            by[c.out(out_dir).name] = by.get(c.out(out_dir).name, 0) + 1
        for name, n in sorted(by.items()):
            print(f"  {name}: {n}")
        print(f"{len(cells)} cells")
        return

    if canonical and stamp()["git_dirty"]:
        raise DirtyTree("data/v2 is written only from a clean, committed tree; use --out-dir for a trial")
    out_dir.mkdir(parents=True, exist_ok=True)
    _check_same_run(out_dir)
    done = {out: {r["key"] for r in _load(out)} for out in {c.out(out_dir) for c in cells}}

    todo = [c for c in cells if c.key not in done[c.out(out_dir)]]
    print(f"=== {len(cells)} cells, {len(cells) - len(todo)} already recorded, {len(todo)} to run ===",
          flush=True)
    t_start = time.time()
    for i, cell in enumerate(todo, 1):
        t0 = time.time()
        try:
            rec = run_one(cell, pred_root)
        except BaseException:
            traceback.print_exc()
            raise RuntimeError(f"cell failed: {cell.key}") from None
        _append(cell.out(out_dir), rec)
        hp = rec.get("C", rec.get("alpha"))
        print(f"[{i}/{len(todo)}] {cell.key}: hp={hp:g} edge={rec['edge']} "
              f"({time.time() - t0:.1f}s, {(time.time() - t_start) / 3600:.2f} h)", flush=True)

    missing, extra = [], []
    for out in {c.out(out_dir) for c in cells}:
        keys = [r["key"] for r in _load(out)]
        want = {c.key for c in cells if c.out(out_dir) == out}
        missing += sorted(want - set(keys))
        extra += sorted(k for k in set(keys) if keys.count(k) > 1)
    if missing or extra:
        raise IncompleteRun(f"{len(missing)} cells without a record, {len(extra)} duplicated: "
                            f"{(missing + extra)[:5]}")
    print(f"all {len(cells)} cells recorded")

    if args.group == "all" and not (args.only or args.shard or args.skip_g1):
        write_complete(out_dir, cells, black_box_g1(list(G1_CELLS), pred_root))
    elif not args.skip_g1 and args.group == "main" and not (args.only or args.shard):
        black_box_g1(list(G1_CELLS), pred_root)
    else:
        print(f"no {COMPLETE}: only an unsharded --group all run writes it")


if __name__ == "__main__":
    main()
