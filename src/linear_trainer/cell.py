"""One probe cell, end to end: select on validation, refit, then score test once.

``run_cell`` is the only place test data is loaded for scoring. It loads train
and val, selects the hyperparameter, refits on train+val, and only then loads
the test split, predicts, stores the predictions and scores them. Bootstraps
and downstream analyses rescore those stored predictions instead of refitting
(ledger G8).
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import numpy as np

from linear_trainer import sources
from linear_trainer.fit import fit, score, select
from linear_trainer.protocol import REPO_ROOT, Protocol, assert_threads, stamp

FAMILY5_CLASSES = 5


def _rel(path: Path) -> str:
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _abs(path: str) -> Path:
    return REPO_ROOT / path


def arrays_sha256(arrays: dict[str, np.ndarray]) -> str:
    """Hash of named arrays (npz bytes are not deterministic, the arrays are)."""
    h = hashlib.sha256()
    for key in sorted(arrays):
        a = np.ascontiguousarray(arrays[key])
        h.update(f"{key}|{a.dtype.str}|{a.shape}|".encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _save(path: Path, arrays: dict[str, np.ndarray]) -> str:
    """Write atomically: a run killed mid-write must not leave a truncated file
    that a resumed run then trips over. The temporary name is per writer, so
    shards that store the same content-addressed file at once never share it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.partial.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return arrays_sha256(arrays)


def _save_predictions(pred_dir: Path, name: str, arrays: dict[str, np.ndarray]) -> tuple[Path, str]:
    """Store predictions under a content-addressed name, never overwriting.

    Metrics files hold several records per cell (reruns, cosine vs R^2
    selection, other split files); each keeps its own file, so an older record
    can always be rescored.
    """
    sha = arrays_sha256(arrays)
    path = Path(pred_dir) / f"{name}__{sha[:12]}.npz"
    if path.exists():
        if arrays_sha256(dict(np.load(path, allow_pickle=False))) != sha:
            raise RuntimeError(f"{path.name} exists with different content")
        return path, sha
    return path, _save(path, arrays)


def _save_targets(pred_dir: Path, ids: np.ndarray, Y: np.ndarray, splits_sha: str) -> tuple[Path, str]:
    """GenePT targets are shared by every regression cell on a split: store them once.

    A second cell on the same split must bring byte-identical targets, or the
    shared file would silently describe only the first one.
    """
    path = pred_dir / f"targets__{splits_sha[:16]}.npz"
    arrays = {"ids": ids.astype(str), "y_true": Y}
    sha = arrays_sha256(arrays)
    if path.exists():
        if arrays_sha256(dict(np.load(path, allow_pickle=False))) != sha:
            raise RuntimeError(f"targets differ from the stored {path.name} for the same split")
        return path, sha
    return path, _save(path, arrays)


def load_predictions(rec: dict) -> dict[str, np.ndarray]:
    """Stored test predictions of a record, verified against its hashes.

    Returns ``ids``, ``pred`` and ``y_true``.
    """
    if not rec.get("pred_file"):
        raise RuntimeError(f"{rec.get('run_id', '?')}: no stored predictions; rerun the cell")
    arrays = dict(np.load(_abs(rec["pred_file"]), allow_pickle=False))
    if arrays_sha256(arrays) != rec["pred_sha256"]:
        raise RuntimeError(f"{rec['pred_file']}: sha256 does not match the record")
    if "targets_file" in rec:
        targets = dict(np.load(_abs(rec["targets_file"]), allow_pickle=False))
        if arrays_sha256(targets) != rec["targets_sha256"]:
            raise RuntimeError(f"{rec['targets_file']}: sha256 does not match the record")
        pos = {g: i for i, g in enumerate(targets["ids"].tolist())}
        arrays["y_true"] = targets["y_true"][[pos[g] for g in arrays["ids"].tolist()]]
    return arrays


def scored_predictions(rec: dict) -> dict[str, np.ndarray]:
    """Stored test predictions minus the record's purged genes (G2).

    The record's metrics are computed on exactly these rows; bootstraps and
    downstream analyses rescore them, never the full stored set.
    """
    if "purge" not in rec:
        raise RuntimeError(f"{rec.get('run_id', rec.get('key', '?'))}: no purge field; a record from "
                           "before G2 cannot be rescored as if nothing were masked")
    arrays = load_predictions(rec)
    masked = set(rec["purge"]["test_masked"])
    if not masked:
        return arrays
    keep = ~np.isin(arrays["ids"], sorted(masked))
    if int((~keep).sum()) != len(masked):
        raise RuntimeError(f"{rec.get('run_id', '?')}: purged genes missing from the stored predictions")
    return {k: v[keep] for k, v in arrays.items()}


def _features_stamp(source: str | Path) -> dict:
    """What the features were read from, hashed (G7): a parquet, or a featuriser
    over the pinned CDS manifest."""
    parquet = sources._parquet_for(source)
    if parquet is not None:
        out = {"path": _rel(parquet), "sha256": sources.sha256_file(parquet)}
        if not isinstance(source, Path) and source in sources.DERIVED:
            base, copies = sources.DERIVED[source]
            out.update(derived_from=base, transform=f"hstack x{copies}")
        return out
    manifest = sources.DATA / "cds_manifest.tsv"
    # The ids, family labels and GenePT targets come from the metadata parquet.
    return {"featurizer": str(source), "cds_manifest": _rel(manifest),
            "cds_manifest_sha256": sources.sha256_file(manifest),
            "meta": _rel(sources.META_PARQUET), "meta_sha256": sources.sha256_file(sources.META_PARQUET)}


def _default_purge(split_file: Path, source: str | Path):
    """The policy purge for a cell (``splits.leaks.rules_for``). A split file
    outside data/ (a test fixture) is recorded as unpurged with the reason."""
    from splits import leaks

    if not Path(split_file).resolve().is_relative_to(sources.DATA.resolve()):
        return leaks.Purge(rules=(), stamp={"reason": "split file outside data/"})
    return leaks.purge_for(split_file, leaks.arm_of(source))


def _check_purge(purge, splits_sha: str, name: str, ids: np.ndarray) -> None:
    if not purge.rules:
        return
    if purge.stamp.get("split_sha256") != splits_sha:
        raise RuntimeError("the purge was derived from a different split file than this cell reads")
    masked = purge.val if name == "val" else purge.test
    stray = masked - set(ids.tolist())
    if stray:
        raise RuntimeError(f"{len(stray)} purged {name} genes are not in the {name} split")


def _assert_disjoint(*id_sets: np.ndarray) -> None:
    seen: set[str] = set()
    for ids in id_sets:
        s = set(ids.tolist())
        if seen & s:
            raise RuntimeError("train/val/test splits overlap")
        seen |= s


def _assert_classes(task: str, name: str, y: np.ndarray) -> None:
    if task == "family5":
        n = len(np.unique(y))
        if n != FAMILY5_CLASSES:
            raise RuntimeError(f"family5 split {name} has {n} classes, expected {FAMILY5_CLASSES}")
    elif task != "genept":
        bal = float(np.mean(y == 1))
        if not 0.45 <= bal <= 0.55:
            raise RuntimeError(f"binary split {name} class balance {bal:.3f} outside [0.45, 0.55]")


def run_cell(source: str | Path, task: str, splits_path: Path, protocol: Protocol, *,
             pred_dir: Path, label_seed: int | None = None, select_by: str = "r2",
             probe_out: Path | None = None,
             purge=None) -> dict:
    """Run one cell. Returns the pick (``hp``, ``sweep``, ``edge``), test ``metrics``
    and the ``provenance`` fields every record carries.

    ``label_seed`` permutes the train and val targets (the shuffled-label
    controls); test targets are never permuted.

    ``purge`` (a ``splits.leaks.Purge`` for this cell's split file) masks its
    val genes from selection and its test genes from scoring (G2). Training,
    the refit and the stored predictions are unchanged; the record also keeps
    the unpurged test metrics for the disclosure. Left as None, the policy
    purge for the split file and the source's arm is applied, so no cell can
    run unpurged by omission; every record carries its ``purge`` field.
    """
    assert_threads(protocol)
    kind = "ridge" if task == "genept" else "logistic"
    splits_path = Path(splits_path)
    split_file = sources.split_file(task, splits_path)
    splits_sha = sources.sha256_file(split_file)   # before the loads; re-checked after test

    X_tr, y_tr, ids_tr = sources.load(source, task, "train", splits_path)
    X_va, y_va, ids_va = sources.load(source, task, "val", splits_path)
    if kind == "logistic":
        _assert_classes(task, "train", y_tr)
        _assert_classes(task, "val", y_va)
    if label_seed is not None:
        rng = np.random.default_rng(label_seed)
        y_tr, y_va = rng.permutation(y_tr), rng.permutation(y_va)

    if purge is None:
        purge = _default_purge(split_file, source)
    _check_purge(purge, splits_sha, "val", ids_va)
    if purge.rules:
        keep_va = ~np.isin(ids_va, sorted(purge.val))
        if kind == "logistic":
            _assert_classes(task, "purged val", y_va[keep_va])
        sel = select(kind, X_tr, y_tr, X_va[keep_va], y_va[keep_va], protocol, select_by=select_by)
    else:
        sel = select(kind, X_tr, y_tr, X_va, y_va, protocol, select_by=select_by)
    stack = np.vstack if kind == "ridge" else np.concatenate
    # The pick converged on train; its train+val refit may not (shuffled labels at
    # high C). Decided Oct 1: keep the refit and record ``converged`` (False here);
    # a primary-test cell with a non-converged refit fails the statistics build,
    # and null bands count them.
    probe = fit(kind, np.vstack([X_tr, X_va]), stack([y_tr, y_va]), sel.hp, protocol, strict=False)
    if probe_out is not None and kind == "ridge":
        probe.save(probe_out)

    # Test data enters only here, after selection and refit are final.
    X_te, y_te, ids_te = sources.load(source, task, "test", splits_path)
    if sources.sha256_file(split_file) != splits_sha:
        raise RuntimeError(f"{split_file} changed while the cell was running")
    _assert_disjoint(ids_tr, ids_va, ids_te)
    _check_purge(purge, splits_sha, "test", ids_te)
    if kind == "logistic":
        _assert_classes(task, "test", y_te)
        _assert_classes(task, "purged test", y_te[~np.isin(ids_te, sorted(purge.test))])
    pred = probe.predict(X_te)

    name = sources.cell_name(source) + f"__{task}" + ("" if label_seed is None else f"__shuf{label_seed}")
    prov: dict = {}
    if kind == "logistic":
        as_str = (lambda a: a.astype(str)) if y_te.dtype.kind in "OUS" else (lambda a: a)
        arrays = {"ids": ids_te.astype(str), "pred": as_str(np.asarray(pred)), "y_true": as_str(y_te)}
    else:
        arrays = {"ids": ids_te.astype(str), "pred": pred.astype(np.float32)}
        tfile, tsha = _save_targets(Path(pred_dir), ids_te.astype(str), y_te, splits_sha)
        prov = {"targets_file": _rel(tfile), "targets_sha256": tsha}
    pred_file, pred_sha = _save_predictions(Path(pred_dir), name, arrays)
    prov = {"pred_file": _rel(pred_file), "pred_sha256": pred_sha, **prov}
    prov["purge"] = purge.record()
    full = load_predictions(prov)
    stored = scored_predictions(prov)
    metrics = score(kind, stored["y_true"], stored["pred"])
    metrics["n_test_scored"] = int(len(stored["ids"]))
    if kind == "logistic":   # G28: the purge thins families unevenly; macro-F1 weighs each equally
        classes, counts = np.unique(stored["y_true"], return_counts=True)
        metrics["n_test_scored_by_class"] = {str(c): int(n) for c, n in zip(classes, counts)}
    if purge.rules:
        metrics["unpurged"] = score(kind, full["y_true"], full["pred"])
    # G3: a probe that predicts one class (or one vector) for every test gene
    # scores at the majority level whatever its features; 1E refuses such a
    # primary-test cell.
    degenerate = (len(np.unique(stored["pred"])) == 1 if kind == "logistic"
                  else bool(np.all(np.ptp(stored["pred"], axis=0) == 0)))

    provenance = {
        "stamp": stamp(),
        "features": _features_stamp(source),
        "protocol": protocol.name,
        "protocol_hash": protocol.hash,
        "splits_file": _rel(split_file),
        "splits_sha256": splits_sha,
        "grid": sel.grid,
        "edge": sel.edge,
        "n_iter": probe.n_iter,          # the refit; each sweep row carries its own
        "converged": probe.converged,
        "degenerate": bool(degenerate),
        **prov,
    }
    return {"kind": kind, "hp": sel.hp, "sweep": sel.sweep, "edge": sel.edge,
            "feature_dim": int(X_tr.shape[1]), "metrics": metrics, "provenance": provenance}


def append_record(path: Path, entry: dict) -> None:
    """Append one record to a JSON-array metrics file."""
    path = Path(path)
    runs: list = []
    if path.exists():
        runs = json.loads(path.read_text())
        if not isinstance(runs, list):
            raise ValueError(f"{path} is not a JSON array")
    runs.append(entry)
    path.write_text(json.dumps(runs, indent=2))
