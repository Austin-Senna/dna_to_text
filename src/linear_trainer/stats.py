"""Uncertainty and tests over stored predictions (G8, G13).

Nothing here refits. Every function starts from a record's scored predictions
(``cell.scored_predictions``: the stored test predictions minus the purged
genes), checks that they reproduce the record's own test value exactly, and
resamples them.

- **Cluster bootstrap.** Test genes are not independent: genes in one 40%
  protein cluster share sequence, and so do genes whose TSS windows overlap.
  The bootstrap resamples whole groups with replacement, so the interval
  reflects the number of independent units, not the number of genes. Groups
  are 40% protein clusters for CDS cells on the homology-type splits, and
  window-and-protein groups for every TSS cell and for any cell on a disjoint
  split (``group_of``). Resampling is unstratified, so the intervals also carry
  class-mix variance the stratified split design does not have: they are
  conservative. The metric is computed exactly as the cell's (``fit.score``).
  A resample that misses a family changes what macro-F1 averages over; the
  count is reported, and more than ``MAX_SHORT_CLASS`` of resamples raises.
- **Paired bootstrap.** Two cells on the same split file (same purge) are
  resampled with the same groups each iteration. It reports the difference
  ``A - B``, its 95% interval, ``p_a_gt_b`` = the share of resamples with
  ``A - B > 0`` (a bootstrap share, not a posterior probability), and a
  one-sided p-value for ``A > B``,
  ``p = (1 + #{A - B <= 0}) / (B_iters + 1)``.
- **Holm.** Step-down adjustment over one named family of p-values. Only the
  four confirmatory tests go in it (decided Sept 29); everything else is
  reported unadjusted and labelled exploratory.
- **Null band.** The 2.5-97.5% range of a metric over label-shuffled runs of
  one cell, each with its full selection loop. It replaces the single shuffled
  run once used as "chance" (G13).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score, r2_score

from linear_trainer.cell import scored_predictions
from linear_trainer.fit import score

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
PROTEIN_CLUSTERS = DATA / "clusters" / "homology_id40.tsv"
N_ITERS = 1000
SEED = 42
# Pre-registered (Oct 1): above this share of resamples missing a family, the
# macro-F1 interval is not reported.
MAX_SHORT_CLASS = 0.01


class ShortClassResamples(RuntimeError):
    """Too many resamples drop a whole family (macro-F1 changes definition)."""

_KIND = {"family5": "logistic", "genept": "ridge"}
_METRIC = {"logistic": "test_macro_f1", "ridge": "test_r2_macro"}


class PointMismatch(RuntimeError):
    """The scored predictions do not reproduce the record's test value (G8)."""


def _kind(rec: dict) -> str:
    return _KIND[rec["task"]]


def _metric(kind: str, y: np.ndarray, pred: np.ndarray) -> float:
    if kind == "logistic":
        return float(f1_score(y, pred, average="macro"))
    return float(r2_score(y, pred, multioutput="uniform_average"))


def _scored(rec: dict) -> dict[str, np.ndarray]:
    arrays = scored_predictions(rec)
    kind = _kind(rec)
    key = _METRIC[kind]
    point = score(kind, arrays["y_true"], arrays["pred"])[key]
    if point != rec[key]:
        raise PointMismatch(f"{rec.get('key', '?')}: scored predictions give {key}={point!r}, "
                            f"record says {rec[key]!r}")
    return arrays


# --- groups --------------------------------------------------------------------

@lru_cache(maxsize=None)
def _protein_groups() -> dict[str, str]:
    from cluster.mmseqs_cluster import parse_cluster_tsv
    return parse_cluster_tsv(PROTEIN_CLUSTERS)


@lru_cache(maxsize=None)
def _disjoint_groups() -> dict[str, str]:
    from data_loader.enformer_windows import MANIFEST, window_spans
    from splits.tss_disjoint import combined_groups
    pmap = _protein_groups()
    group_of, _ = combined_groups(sorted(pmap), window_spans(MANIFEST), pmap)
    return group_of


def group_of(split_name: str, arm: str) -> dict[str, str]:
    """Gene -> resampling unit for one cell.

    Window-and-protein groups for TSS cells (window overlap is a property of the
    genes, whatever the split) and for every cell on a disjoint split (the units
    that split was built from); 40% protein clusters otherwise.
    """
    if arm not in ("cds", "tss"):
        raise ValueError(f"arm must be 'cds' or 'tss', got {arm!r}")
    if arm == "tss" or Path(split_name).name.startswith("splits_tss_disjoint"):
        return _disjoint_groups()
    return _protein_groups()


def _group_index(ids: np.ndarray, groups: dict[str, str]) -> list[np.ndarray]:
    missing = [g for g in ids.tolist() if g not in groups]
    if missing:
        raise KeyError(f"{len(missing)} test genes have no resampling group: {missing[:5]}")
    by: dict[str, list[int]] = {}
    for i, g in enumerate(ids.tolist()):
        by.setdefault(groups[g], []).append(i)
    return [np.asarray(by[k]) for k in sorted(by)]


def _resamples(units: list[np.ndarray], n_iters: int, seed: int):
    rng = np.random.default_rng(seed)
    k = len(units)
    for _ in range(n_iters):
        yield np.concatenate([units[j] for j in rng.integers(0, k, size=k)])


def _ci(values: np.ndarray) -> list[float]:
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _short_class(kind: str, y: np.ndarray, idxs: list[np.ndarray], what: str) -> int:
    if kind != "logistic":
        return 0
    n_classes = len(np.unique(y))
    short = sum(len(np.unique(y[idx])) < n_classes for idx in idxs)
    if short > MAX_SHORT_CLASS * len(idxs):
        raise ShortClassResamples(f"{what}: {short} of {len(idxs)} resamples miss a family")
    return int(short)


# --- bootstraps -----------------------------------------------------------------

def cluster_bootstrap(rec: dict, groups: dict[str, str] | None = None,
                      n_iters: int = N_ITERS, seed: int = SEED) -> dict:
    """95% cluster-bootstrap interval for one record's test metric."""
    arrays = _scored(rec)
    kind = _kind(rec)
    groups = groups if groups is not None else group_of(rec["split"], rec["arm"])
    units = _group_index(arrays["ids"], groups)
    y, pred = arrays["y_true"], arrays["pred"]
    idxs = list(_resamples(units, n_iters, seed))
    vals = np.array([_metric(kind, y[idx], pred[idx]) for idx in idxs])
    ci = _ci(vals)
    point = rec[_METRIC[kind]]
    out = {"key": rec.get("key"), "metric": _METRIC[kind], "point": point, "ci95": ci,
           "point_in_ci": bool(ci[0] <= point <= ci[1]),
           "n_test": int(len(y)), "n_groups": len(units), "n_iters": n_iters, "seed": seed,
           "n_short_class": _short_class(kind, y, idxs, str(rec.get("key")))}
    if kind == "logistic":
        kap = np.array([float(cohen_kappa_score(y[idx], pred[idx])) for idx in idxs])
        out.update(kappa_point=rec["test_kappa"], kappa_ci95=_ci(kap))
    return out


def paired_bootstrap(rec_a: dict, rec_b: dict, groups: dict[str, str] | None = None,
                     n_iters: int = N_ITERS, seed: int = SEED) -> dict:
    """Difference A - B on the same scored test genes, resampled together."""
    if rec_a["task"] != rec_b["task"]:
        raise ValueError("paired cells must share a task")
    if rec_a["splits_sha256"] != rec_b["splits_sha256"]:
        raise ValueError(f"{rec_a.get('key')} and {rec_b.get('key')} are on different split files")
    if rec_a["purge"]["test_masked"] != rec_b["purge"]["test_masked"]:
        raise ValueError(f"{rec_a.get('key')} and {rec_b.get('key')} score different test genes")
    a, b = _scored(rec_a), _scored(rec_b)
    pos = {g: i for i, g in enumerate(b["ids"].tolist())}
    if set(pos) != set(a["ids"].tolist()):
        raise ValueError("paired cells score different test genes")
    order = np.array([pos[g] for g in a["ids"].tolist()])
    y = a["y_true"]
    if _kind(rec_a) == "logistic":
        if not np.array_equal(y, b["y_true"][order]):
            raise ValueError("paired test labels differ")
    elif not np.array_equal(y, b["y_true"][order]):
        raise ValueError("paired regression targets differ")
    pa, pb = a["pred"], b["pred"][order]
    kind = _kind(rec_a)
    if groups is None:   # the coarser grouping when the two cells' arms differ
        arm = "tss" if "tss" in (rec_a["arm"], rec_b["arm"]) else "cds"
        groups = group_of(rec_a["split"], arm)
    units = _group_index(a["ids"], groups)
    idxs = list(_resamples(units, n_iters, seed))
    d = np.array([_metric(kind, y[idx], pa[idx]) - _metric(kind, y[idx], pb[idx]) for idx in idxs])
    point = rec_a[_METRIC[kind]] - rec_b[_METRIC[kind]]
    ci = _ci(d)
    return {"metric": _METRIC[kind], "a": rec_a.get("key"), "b": rec_b.get("key"),
            "delta_point": float(point), "delta_ci95": ci,
            "point_in_ci": bool(ci[0] <= point <= ci[1]),
            "p_a_gt_b": float(np.mean(d > 0)),
            "p_one_sided": float((1 + np.sum(d <= 0)) / (n_iters + 1)),
            "n_test": int(len(y)), "n_groups": len(units), "n_iters": n_iters, "seed": seed,
            "n_short_class": _short_class(kind, y, idxs, f"{rec_a.get('key')} vs {rec_b.get('key')}")}


# --- multiplicity and chance ----------------------------------------------------

def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values for one family of tests."""
    names = sorted(pvalues, key=lambda k: (pvalues[k], k))
    m = len(names)
    out, running = {}, 0.0
    for i, name in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvalues[name]))
        out[name] = running
    return out


def null_band(records: list[dict], n_expected: int) -> dict:
    """2.5-97.5% range of the test metric over label-shuffled runs of one cell."""
    if len(records) != n_expected:
        raise ValueError(f"null band has {len(records)} shuffles, expected {n_expected}")
    cells = {(r["split"], r["task"], r["feature_source"]) for r in records}
    if len(cells) != 1 or not all(r.get("shuffled_labels") for r in records):
        raise ValueError(f"a null band is one cell's shuffled runs; got {sorted(cells)}")
    seeds = [r["label_seed"] for r in records]
    if len(set(seeds)) != len(seeds):
        raise ValueError("duplicate shuffle seeds in a null band")
    key = _METRIC[_kind(records[0])]
    vals = np.array([r[key] for r in records])
    (split, task, source), = cells
    return {"split": split, "task": task, "source": source, "metric": key,
            "band95": _ci(vals), "median": float(np.median(vals)), "n": len(vals)}


def write_json(path: Path, payload: dict) -> None:
    tmp = Path(path).with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)
