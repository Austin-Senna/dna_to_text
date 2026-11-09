"""Key-aware diff of two recompute runs: Phase 4's 1-vs-6-thread determinism check
and Phase 5's clean-room comparison.

Joins the records of two runs (``metrics_*.json`` and ``null_*.json``) on cell
key and compares what the paper uses:

  * each cell's pick (C or alpha), its convergence flag, and the fields the
    statistics gate a primary-test cell on (edge, degeneracy, scored counts,
    purge, GenePT targets);
  * its test predictions: classification bit for bit (the stored arrays'
    sha256), regression within ``--tol`` (absolute, on the stored predictions);
  * its test metrics: exactly for classification, within ``--tol`` for regression
    (a metric on one side only, or NaN on one side, always differs);
  * the builder-level picks (``linear_trainer.records``): each encoder's pool, the
    best encoder, the k-mer picks, the E5 anchored-chunk composition pick. Validation scores can move enough to swap a
    pick while every cell keeps its own C, so the cell diff alone would miss it.

Provenance (commit, protocol hash, thread count, CPU) is reported, not compared.
Each differing cell carries its C or alpha, the feature dimension d and the train
size n, so the fragile reading (C >= 1e3 with d near n) can be checked.

Run: uv run scripts/diff_records.py A_DIR B_DIR [--a-root DIR] [--b-root DIR]
         [--subset] [--tol 1e-9] [--json OUT]
A relative ``pred_file`` resolves against its run's root (default: this repo).
``--subset`` lets B hold fewer cells than A (a main-group-only trial), never more.
Exit 0 when nothing differs, 1 when something does, 2 on unusable input.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from linear_trainer import records as R
from linear_trainer.selection import MissingRecord, val_score

REPO_ROOT = Path(__file__).resolve().parents[1]
# Compared exactly: build_statistics refuses a primary-test cell on these.
GATE_FIELDS = ("edge", "degenerate", "n_test_scored", "n_test_scored_by_class", "purge",
               "targets_sha256")


class BadInput(RuntimeError):
    """The two runs can't be compared: their keys differ, or a file mixes runs."""


def _load(d: Path) -> tuple[dict[str, dict], dict]:
    recs: dict[str, dict] = {}
    for path in sorted(d.glob("metrics_*.json")) + sorted(d.glob("null_*.json")):
        for r in json.loads(path.read_text()):
            if r["key"] in recs:
                raise BadInput(f"{path}: {r['key']} recorded twice")
            recs[r["key"]] = r
    if not recs:
        raise BadInput(f"no records in {d}")
    stamps = {(r["stamp"]["git_sha"], r["protocol_hash"]) for r in recs.values()}
    if len(stamps) > 1:
        raise BadInput(f"{d} mixes {len(stamps)} commit/protocol stamps")
    (sha, proto), = stamps
    threads = sorted({json.dumps(r["stamp"].get("threads")) for r in recs.values()})
    return recs, {"git_sha": sha, "protocol_hash": proto, "threads": [json.loads(t) for t in threads]}


def _hp(rec: dict) -> float:
    return rec["C"] if rec["task"] == "family5" else rec["alpha"]


def _metrics(rec: dict) -> dict[str, float]:
    """Every numeric test metric, the unpurged scores as ``unpurged.<name>``."""
    out = {k: v for k, v in rec.items() if k.startswith("test_") and isinstance(v, (int, float))}
    for k, v in (rec.get("unpurged") or {}).items():
        if isinstance(v, (int, float)):
            out[f"unpurged.{k}"] = v
    return out


def _n_train(split: str) -> int:
    return len(json.loads((REPO_ROOT / "data" / split).read_text())["train"])


def _pred(rec: dict, root: Path) -> np.ndarray:
    path = Path(rec["pred_file"])
    path = path if path.is_absolute() else root / path
    if not path.exists():
        raise BadInput(f"{rec['key']}: no prediction file {path}")
    with np.load(path, allow_pickle=False) as z:
        return z["pred"]


def _same(x: float, y: float, tol: float) -> bool:
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    return abs(y - x) <= tol


def _val(rec: dict) -> float | None:
    try:
        return val_score(rec)
    except ValueError:                     # no converged sweep point
        return None


def _cell(a: dict, b: dict, a_root: Path, b_root: Path, tol: float) -> dict | None:
    regression = a["task"] != "family5"
    what, extra = [], {}
    if _hp(a) != _hp(b):
        what.append("pick")
    if a["pred_sha256"] != b["pred_sha256"]:
        if not regression:
            what.append("predictions")
        else:
            gap = float(np.max(np.abs(_pred(a, a_root).astype(np.float64) - _pred(b, b_root))))
            if gap > tol:
                what.append("predictions")
                extra["max_abs_pred"] = gap
    ma, mb = _metrics(a), _metrics(b)
    deltas = {k: (mb[k] - ma[k] if k in ma and k in mb else None) for k in sorted(set(ma) | set(mb))
              if k not in ma or k not in mb or not _same(ma[k], mb[k], tol if regression else 0)}
    if deltas:
        what.append("metrics")
        extra["metric_deltas"] = deltas
    if a["converged"] != b["converged"]:
        what.append("converged")
    what += [f for f in GATE_FIELDS if a.get(f) != b.get(f)]
    if not what:
        return None
    return {"key": a["key"], "what": what, "hp_a": _hp(a), "hp_b": _hp(b), "d": a["feature_dim"],
            "n_train": _n_train(a["split"]), **extra}


def _picks(recs: dict[str, dict]) -> dict[str, str]:
    """Every builder-level pick the records support, by a readable name."""
    out: dict[str, str] = {}
    groups = {(r["split"], r["arm"], r["task"]) for r in recs.values() if not r["shuffled_labels"]}
    for split, arm, task in sorted(groups):
        by_source = R.cells({k: r for k, r in recs.items() if r["split"] == split}, arm, task)
        where = f"{split}/{arm}/{task}"
        named = {f"{where} {e} pool": (R.best_pool, e, arm) for e in R.ENCODERS}
        named[f"{where} best encoder"] = (R.best_encoder, arm)
        if arm == "tss":
            named |= {f"{where} {e} anchored-chunk composition":
                      (R.pick, [f"tss_{e}_chunk4mergc", f"tss_{e}_chunk6mer"]) for e in R.ENCODERS}
        if arm == "cds":
            named |= {f"{where} nt k-mer": (R.best_nt_kmer,), f"{where} aa k-mer": (R.best_aa,),
                      f"{where} nt k-mer+len": (R.best_nt_kmer_len,),
                      f"{where} aa k-mer+len": (R.best_aa_len,)}
        for name, (fn, *args) in named.items():
            try:
                out[name] = fn(by_source, *args)
            except MissingRecord:          # a split without that grid
                pass
    return out


def diff(a_dir: Path, b_dir: Path, *, a_root: Path = REPO_ROOT, b_root: Path = REPO_ROOT,
         subset: bool = False, tol: float = 1e-9) -> dict:
    if Path(a_dir).resolve() == Path(b_dir).resolve():
        raise BadInput(f"{a_dir} and {b_dir} are the same directory")
    a, a_stamp = _load(Path(a_dir))
    b, b_stamp = _load(Path(b_dir))
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))
    if only_b:
        raise BadInput(f"{len(only_b)} keys only in b, e.g. {only_b[:3]}")
    if only_a and not subset:
        raise BadInput(f"{len(only_a)} keys only in a, e.g. {only_a[:3]} (pass --subset for a partial b)")
    common = sorted(set(a) & set(b))
    cells = [c for k in common if (c := _cell(a[k], b[k], Path(a_root), Path(b_root), tol))]
    pa, pb = _picks({k: a[k] for k in common}), _picks({k: b[k] for k in common})
    picks = [{"pick": n, "a": pa[n], "b": pb.get(n)} for n in sorted(pa) if pa[n] != pb.get(n)]
    val = [abs(va - vb) for k in common if (va := _val(a[k])) is not None and (vb := _val(b[k])) is not None]
    return {"stamps": {"a": a_stamp, "b": b_stamp}, "n_common": len(common), "n_only_a": len(only_a),
            "n_iter_changed": sum(a[k]["n_iter"] != b[k]["n_iter"] for k in common),
            "max_val_delta": max(val, default=0.0), "tol": tol, "cells": cells, "picks": picks}


def differs(report: dict) -> bool:
    return bool(report["cells"] or report["picks"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--a-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--b-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    try:
        report = diff(args.a, args.b, a_root=args.a_root, b_root=args.b_root, subset=args.subset,
                      tol=args.tol)
    except BadInput as e:
        print(f"unusable input: {e}", file=sys.stderr)
        return 2
    if args.json:
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"a: {report['stamps']['a']}\nb: {report['stamps']['b']}")
    print(f"{report['n_common']} cells compared ({report['n_only_a']} only in a); "
          f"n_iter changed in {report['n_iter_changed']}; max |val delta| {report['max_val_delta']:.3g}")
    for c in report["cells"]:
        print(f"  {c['key']}: {', '.join(c['what'])}; hp {c['hp_a']:g} -> {c['hp_b']:g}, "
              f"d={c['d']}, n={c['n_train']}")
    for p in report["picks"]:
        print(f"  pick {p['pick']}: {p['a']} -> {p['b']}")
    print("no differences" if not differs(report) else
          f"{len(report['cells'])} cells and {len(report['picks'])} picks differ")
    return int(differs(report))


if __name__ == "__main__":
    sys.exit(main())
