"""Independent reproduction of the MINA headline probe cells.

What: recompute ten headline cells (picks, evaluation purge, sweep, refit, test score) and the
T1-T4 confirmatory statistics from the raw inputs (split files, feature parquets, CDS FASTA,
protein-pair and TSS-window tables), then compare every number with the pipeline's records.

Why: the camera-ready numbers come from one probe pipeline. This script was written from a spec
only, without reading or importing that pipeline, so a shared bug cannot hide in both. It imports
nothing from the repo: only the standard library, numpy, pandas and scikit-learn.

Usage:
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python scripts/reproduce_headline.py [--records DIR] [--data-root DIR] [--cells 5,9,10]
                                         [--n-boot 2000] [--out FILE] [--jobs N]

Prints one line per check; exits 0 only when every check passes.

Choices where the spec is silent or ambiguous (each is a decision, not taken from pipeline code):
- Row order. Fits use the split file's gene order: train for the sweep, train then val for the
  refit. lbfgs sums in row order, so a different order could move predictions at the last bit.
- TSS window interval convention. `end - start + 1` equals the 196608 bp window length in
  `tss_windows.meta.json`, so windows are closed intervals [start, end]; two windows overlap when
  they share a chromosome and `a.start <= b.end and b.start <= a.end`. The run also counts the
  pairs that only touch (`a.end == b.start`), the ones a half-open reading would drop.
- Edge rule. After the extension loop stops with `edge = false`, the interior-neighbour rule is
  applied to both hp-order neighbours of the final pick. Ties in the extension test count as
  plateau (difference in [0, 1e-4]); a strictly lower score is interior.
- Ridge sweep scores use float64 predictions; only the test R^2 uses float32-cast predictions.
- Shuffled labels: `rng.permutation(y_train)` then `rng.permutation(y_val)` on the label arrays in
  split order; the permuted labels are used in both the sweep and the refit.
- Gene metadata for k-mer/AA sources comes from `dataset_dnabert2_meanmean.parquet`; encoder
  sources use their own parquet's `family` / `y`.
- Statistics truth labels come from the parquet `family` column (the npz `y_true` is checked
  against it). Scored genes use this script's own purge masks, not the record's lists (the purge
  check compares the two separately).
- Resampling groups are connected components of "same homology_id40 cluster OR a Rule-A protein
  pair" (spec amendment, Oct 1), plus "OR overlapping TSS window" on the disjoint split and for any
  TSS cell, built over every gene in the parquet and then counted among the scored genes. A gene
  absent from the cluster table is its own cluster. The count over a graph restricted to the
  scored genes is reported as a diagnostic.
- The cluster bootstrap draws groups with `numpy.random.default_rng(20261001)`; the CI is the
  2.5/97.5 linear-interpolation percentile of the resampled deltas.
- Statistics sides T1-T4 are derived from this script's own picks. `statistics.json` is compared
  only when it exists in the records directory.
- Extra guard checks beyond the spec: npz `y_true` / Ridge targets equal the parquet truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import f1_score, r2_score
from sklearn.preprocessing import StandardScaler

THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")
SPLIT_FILES = ("splits.json", "splits_tss_disjoint.json")
ENCODERS = ("dnabert2", "nt_v2", "gena_lm", "hyena_dna")
EXCLUDED_POOL_SUFFIXES = ("meanmean3", "tssanchored", "centermean", "chunk4mergc", "chunk6mer")
META_PARQUET = "dataset_dnabert2_meanmean.parquet"
SEQ_SOURCES = ("kmer", "kmer6", "aa1", "aa2", "aa3")
MAX_ITER = 5000
BASE_EXPONENTS = range(-4, 5)
HP_LIMITS = {"C": (-6, 6), "alpha": (-6, 9)}
PLATEAU_TOL = 1e-4
BOOT_SEED = 20261001
NUCLEOTIDES = "ACGT"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
# NCBI table 1, codons enumerated with bases in TCAG order.
CODE_TABLE_1 = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TO_AA = {
    "".join(c): aa for c, aa in zip(product("TCAG", repeat=3), CODE_TABLE_1, strict=True)
}

CELLS: dict[int, tuple[str, str, str, str]] = {
    1: ("splits.json", "cds", "family5", "best encoder"),
    2: ("splits.json", "cds", "family5", "nucleotide k-mer pick"),
    3: ("splits.json", "cds", "family5", "AA k-mer pick"),
    4: ("splits.json", "cds", "family5", "esm2_650m"),
    5: ("splits.json", "cds", "family5", "kmer"),
    6: ("splits_tss_disjoint.json", "cds", "family5", "best encoder"),
    7: ("splits_tss_disjoint.json", "tss", "family5", "TSS pool pick of cell 6's encoder"),
    8: ("splits.json", "cds", "genept", "best encoder"),
    9: ("splits.json", "cds", "genept", "AA k-mer pick"),
    10: ("splits.json", "cds", "family5", "kmer null, lowest label_seed"),
}
# Cells 11+ exercise the selection rule's edge branches, which the headline cells
# may never reach: per branch, the record with the smallest feature dimension
# among those this script can featurise (``edge_cells``).
EDGE_BRANCHES = ("plateau", "limit", "nonconverged", "multi-step extension")


# ---------------------------------------------------------------- small helpers


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def hp_name(task: str) -> str:
    return "C" if task == "family5" else "alpha"


def score_name(task: str) -> str:
    return "macro_f1" if task == "family5" else "r2"


def hp_value(exponent: int) -> float:
    # Parse the decimal literal so 1e-4 is the correctly rounded double, as a config would hold it.
    return float(f"1e{exponent}")


def exponent_of(value: float) -> int:
    return round(math.log10(value))


def split_stem(split: str) -> str:
    return split.removesuffix(".json")


def resolve(root: Path, path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else root / p


def val_score(record: dict[str, Any]) -> float:
    """Best validation score among the record's converged sweep rows."""
    task = record["task"]
    rows = record[f"{hp_name(task)}_sweep"]
    scores = [r[score_name(task)] for r in rows if r["converged"]]
    return max(scores) if scores else -math.inf


class Checks:
    """Collects one pass/fail line per check and prints it as it lands."""

    def __init__(self) -> None:
        self.lines: list[dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: str) -> bool:
        self.lines.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
        return ok

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [line for line in self.lines if not line["ok"]]


# ---------------------------------------------------------------- inputs


def load_split(root: Path, split: str) -> dict[str, list[str]]:
    data = json.loads((root / "data" / split).read_text())
    return {part: list(data[part]) for part in ("train", "val", "test")}


def read_fasta(path: Path) -> str:
    lines = path.read_text().splitlines()
    if not lines or not lines[0].startswith(">"):
        raise ValueError(f"{path.name}: not a FASTA record")
    if any(line.startswith(">") for line in lines[1:]):
        raise ValueError(f"{path.name}: more than one FASTA record")
    return "".join(line.strip() for line in lines[1:]).upper()


def verify_cds(root: Path, checks: Checks) -> None:
    """Every CDS sequence must hash to its manifest sha256 (uppercase, no header or newlines)."""
    manifest = pd.read_csv(root / "data" / "cds_manifest.tsv", sep="\t", dtype=str)
    bad: list[str] = []
    for gene, digest in zip(manifest["ensembl_id"], manifest["sha256"], strict=True):
        path = root / "data" / "sequences" / f"{gene}.fa"
        if not path.exists():
            bad.append(f"{gene}: missing file")
            continue
        seq = read_fasta(path)
        if hashlib.sha256(seq.encode()).hexdigest() == digest:
            continue
        # Say what the digest does match, if anything, before failing.
        raw = path.read_text()
        body = "".join(raw.splitlines()[1:])
        alts = {
            "raw file": raw,
            "body, original case": body,
            "body with newlines": "\n".join(raw.splitlines()[1:]),
        }
        hit = [name for name, text in alts.items() if hashlib.sha256(text.encode()).hexdigest() == digest]
        bad.append(f"{gene}: no match ({'matches ' + hit[0] if hit else 'no alternative matches'})")
    checks.add(
        "cds_manifest",
        not bad,
        f"{len(manifest)} sequences, {len(bad)} mismatched" + (f"; first: {bad[:3]}" if bad else ""),
    )


def load_parquet(root: Path, source: str) -> pd.DataFrame:
    df = pd.read_parquet(root / "data" / f"dataset_{source}.parquet")
    return df.set_index("ensembl_id", drop=False)


def stack(column: pd.Series) -> np.ndarray:
    return np.stack([np.asarray(v, dtype=np.float32) for v in column.to_numpy()])


def kmer_matrix(codes: list[np.ndarray], alphabet: int, k: int) -> np.ndarray:
    """L1-normalised overlapping k-mer frequencies; -1 codes mark characters to skip."""
    out = np.zeros((len(codes), alphabet**k), dtype=np.float64)
    weights = alphabet ** np.arange(k - 1, -1, -1)  # first letter most significant
    for row, c in enumerate(codes):
        if len(c) < k:
            continue
        windows = np.lib.stride_tricks.sliding_window_view(c, k)
        windows = windows[(windows >= 0).all(axis=1)]
        counts = np.bincount(windows @ weights, minlength=alphabet**k).astype(np.float64)
        total = counts.sum()
        if total > 0:
            out[row] = counts / total
    return out.astype(np.float32)


def translate(cds: str) -> str:
    """Frame 0, table 1; non-ACGT codon -> X; internal stop -> X; final stop and remainder dropped."""
    codons = [cds[i : i + 3] for i in range(0, len(cds) - len(cds) % 3, 3)]
    protein = []
    for i, codon in enumerate(codons):
        aa = CODON_TO_AA.get(codon, "X")  # any non-ACGT base misses the table
        if aa == "*":
            if i == len(codons) - 1:
                continue
            aa = "X"
        protein.append(aa)
    return "".join(protein)


def encode(text: str, alphabet: str) -> np.ndarray:
    lookup = np.full(256, -1, dtype=np.int64)
    for i, ch in enumerate(alphabet):
        lookup[ord(ch)] = i
    return lookup[np.frombuffer(text.encode("ascii"), dtype=np.uint8)]


_CACHE: dict[str, Any] = {}


def source_frame(root: Path, source: str) -> tuple[pd.DataFrame, np.ndarray]:
    """(metadata frame indexed by ensembl_id, float32 feature matrix in that frame's row order)."""
    key = f"{root}:{source}"
    if key in _CACHE:
        return _CACHE[key]
    if source in SEQ_SOURCES:
        meta = load_parquet(root, META_PARQUET.removeprefix("dataset_").removesuffix(".parquet"))
        seqs = [read_fasta(root / "data" / "sequences" / f"{g}.fa") for g in meta["ensembl_id"]]
        if source.startswith("kmer"):
            k = 4 if source == "kmer" else 6
            x = kmer_matrix([encode(s, NUCLEOTIDES) for s in seqs], 4, k)
        else:
            k = int(source[2:])
            x = kmer_matrix([encode(translate(s), AMINO_ACIDS) for s in seqs], 20, k)
    else:
        meta = load_parquet(root, source)
        x = stack(meta["x"])
    _CACHE[key] = (meta, x)
    return meta, x


# ---------------------------------------------------------------- purge and groups


def protein_adjacency(root: Path, identity: float = 0.40) -> dict[str, set[str]]:
    pairs = pd.read_csv(root / "data" / "leaks" / "protein_pairs.tsv", sep="\t")
    keep = (
        (pairs["fident"] >= identity)
        & (pairs["evalue"] <= 1e-3)
        & (pairs["qcov"] >= 0.8)
        & (pairs["tcov"] >= 0.8)
    )
    adj: dict[str, set[str]] = {}
    for a, b in zip(pairs.loc[keep, "gene_a"], pairs.loc[keep, "gene_b"], strict=True):
        if a != b:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)
    return adj


def window_adjacency(root: Path) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Closed-interval overlaps of TSS windows on the same chromosome."""
    win = pd.read_csv(root / "data" / "tss_windows.tsv", sep="\t", dtype={"chrom": str})
    # Closed-interval length; the full (unclipped) windows should all be 196608 bp.
    lengths = (win["end"] - win["start"] + 1).to_numpy()
    adj: dict[str, set[str]] = {}
    n_pairs = n_touching = 0
    for _, grp in win.groupby("chrom"):
        grp = grp.sort_values("start")
        ids = grp["ensembl_id"].to_numpy()
        starts = grp["start"].to_numpy()
        ends = grp["end"].to_numpy()
        for i in range(len(ids)):
            j = i + 1
            # Sorted by start: once start_j > end_i, every later window starts past end_i too.
            while j < len(ids) and starts[j] <= ends[i]:
                adj.setdefault(ids[i], set()).add(ids[j])
                adj.setdefault(ids[j], set()).add(ids[i])
                n_pairs += 1
                n_touching += int(starts[j] == ends[i])
                j += 1
    info = {
        "max_window_length_closed": int(lengths.max()),
        "n_windows_shorter": int((lengths < lengths.max()).sum()),
        "overlap_pairs": n_pairs,
        "touching_only_pairs": n_touching,
    }
    return adj, info


def merge(*adjs: dict[str, set[str]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for adj in adjs:
        for gene, nbrs in adj.items():
            out.setdefault(gene, set()).update(nbrs)
    return out


def purge_masks(split: dict[str, list[str]], adj: dict[str, set[str]]) -> tuple[set[str], set[str]]:
    train = set(split["train"])
    train_val = train | set(split["val"])
    val_masked = {g for g in split["val"] if adj.get(g, set()) & train}
    test_masked = {g for g in split["test"] if adj.get(g, set()) & train_val}
    return val_masked, test_masked


def cluster_map(root: Path) -> dict[str, str]:
    tab = pd.read_csv(root / "data" / "clusters" / "homology_id40.tsv", sep="\t", header=None, dtype=str)
    return dict(zip(tab[1], tab[0], strict=True))


def components(genes: list[str], clusters: dict[str, str], links: dict[str, set[str]]) -> dict[str, int]:
    """Union-find over `genes`: same protein cluster OR a direct link (a Rule-A protein pair, and on
    window-grouped tests an overlapping TSS window)."""
    parent = {g: g for g in genes}

    def find(g: str) -> str:
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    first_of_cluster: dict[str, str] = {}
    for g in genes:
        c = clusters.get(g, g)
        if c in first_of_cluster:
            union(g, first_of_cluster[c])
        else:
            first_of_cluster[c] = g
        for h in links.get(g, ()):
            if h in parent:
                union(g, h)
    roots = sorted({find(g) for g in genes})
    index = {r: i for i, r in enumerate(roots)}
    return {g: index[find(g)] for g in genes}


# ---------------------------------------------------------------- probe protocol


def fit_logistic(c: float, x: np.ndarray, y: np.ndarray) -> tuple[LogisticRegression, bool, int]:
    model = LogisticRegression(C=c, solver="lbfgs", max_iter=MAX_ITER, tol=1e-6)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.fit(x, y)
    warned = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    n_iter = int(np.max(model.n_iter_))
    return model, (n_iter < MAX_ITER and not warned), n_iter


def fit_predict(task: str, hp: float, x_fit: np.ndarray, y_fit: np.ndarray, x_pred: np.ndarray) -> tuple[np.ndarray, bool, int | None]:
    """Standardise on the fit rows, fit the probe, predict `x_pred`."""
    scaler = StandardScaler().fit(x_fit)
    xf, xp = scaler.transform(x_fit), scaler.transform(x_pred)
    if task == "family5":
        model, converged, n_iter = fit_logistic(hp, xf, y_fit)
        return model.predict(xp), converged, n_iter
    model = Ridge(alpha=hp).fit(xf, y_fit)
    return model.predict(xp), True, None


def score(task: str, y_true: np.ndarray, pred: np.ndarray) -> float:
    if task == "family5":
        return float(f1_score(y_true, pred, average="macro"))
    return float(r2_score(y_true, pred, multioutput="uniform_average"))


def sweep_and_pick(task: str, evaluate: Any) -> tuple[int, Any, dict[int, dict[str, Any]]]:
    """Base grid, best converged point (first evaluated wins ties), then the edge extension."""
    points: dict[int, dict[str, Any]] = {}
    order: list[int] = []

    def run(e: int) -> dict[str, Any]:
        points[e] = evaluate(e)
        order.append(e)
        return points[e]

    for e in BASE_EXPONENTS:
        run(e)
    best: int | None = None
    for e in order:
        p = points[e]
        if p["converged"] and (best is None or p["score"] > points[best]["score"]):
            best = e
    if best is None:
        raise RuntimeError("no converged point on the base grid")

    lo, hi = HP_LIMITS[hp_name(task)]
    edge: Any = False
    while True:
        evaluated = sorted(points)
        if best == evaluated[-1]:
            nxt = best + 1
        elif best == evaluated[0]:
            nxt = best - 1
        else:
            break
        if not lo <= nxt <= hi:
            edge = "limit"
            break
        p = run(nxt)
        if not p["converged"]:
            edge = "nonconverged"
            break
        gain = p["score"] - points[best]["score"]
        if gain > PLATEAU_TOL:
            best = nxt
            continue
        edge = "plateau" if gain >= 0 else False
        break
    if edge is False:
        evaluated = sorted(points)
        i = evaluated.index(best)
        nbrs = [evaluated[j] for j in (i - 1, i + 1) if 0 <= j < len(evaluated)]
        if any(not points[n]["converged"] for n in nbrs):
            edge = "nonconverged"
    return best, edge, points


def refit_cell(root_s: str, record: dict[str, Any], val_masked: list[str], test_masked: list[str]) -> dict[str, Any]:
    """Re-run one cell end to end and compare with its record. Runs in a worker process."""
    t0 = time.time()
    root = Path(root_s)
    task, source = record["task"], record["feature_source"]
    split = load_split(root, record["split"])
    meta, x_all = source_frame(root, source)
    row = {g: i for i, g in enumerate(meta["ensembl_id"])}
    idx = {part: np.array([row[g] for g in split[part]]) for part in split}
    x = {part: x_all[idx[part]].astype(np.float64) for part in split}
    if task == "family5":
        labels = meta["family"].to_numpy(dtype=object).astype(str)
    else:
        labels = stack(meta["y"])
    y = {part: labels[idx[part]] for part in split}
    if record["shuffled_labels"]:
        rng = np.random.default_rng(record["label_seed"])
        y["train"] = rng.permutation(y["train"])
        y["val"] = rng.permutation(y["val"])

    val_out, test_out = set(val_masked), set(test_masked)
    val_keep = np.array([g not in val_out for g in split["val"]])
    test_keep = np.array([g not in test_out for g in split["test"]])

    def evaluate(e: int) -> dict[str, Any]:
        pred, converged, n_iter = fit_predict(task, hp_value(e), x["train"], y["train"], x["val"])
        s = score(task, y["val"][val_keep], pred[val_keep])
        return {"hp": hp_value(e), "score": s, "converged": converged, "n_iter": n_iter}

    best, edge, points = sweep_and_pick(task, evaluate)
    hp = hp_value(best)
    x_fit = np.vstack([x["train"], x["val"]])
    y_fit = np.concatenate([y["train"], y["val"]])
    pred, refit_conv, refit_iter = fit_predict(task, hp, x_fit, y_fit, x["test"])

    out: dict[str, Any] = {"key": record["key"], "hp_name": hp_name(task), "checks": []}

    def check(name: str, ok: bool, detail: str) -> None:
        out["checks"].append({"check": f"cell {record['key']} {name}", "ok": bool(ok), "detail": detail})

    rec_hp = record[hp_name(task)]
    check("pick", math.isclose(hp, rec_hp, rel_tol=1e-12), f"mine {hp:g}, record {rec_hp:g}")
    check("edge", edge == record["edge"], f"mine {edge!r}, record {record['edge']!r}")

    rec_rows = {exponent_of(r[hp_name(task)]): r for r in record[f"{hp_name(task)}_sweep"]}
    same_grid = sorted(points) == sorted(rec_rows)
    check("sweep grid", same_grid, f"mine {[hp_value(e) for e in sorted(points)]}, record {[hp_value(e) for e in sorted(rec_rows)]}")
    bad_rows = []
    max_dscore = 0.0
    for e in sorted(set(points) & set(rec_rows)):
        mine, rec = points[e], rec_rows[e]
        d = abs(mine["score"] - rec[score_name(task)])
        max_dscore = max(max_dscore, d)
        if mine["converged"] != rec["converged"] or d > 1e-9:
            bad_rows.append(
                f"{hp_value(e):g}: converged {mine['converged']}/{rec['converged']}, "
                f"score {mine['score']!r}/{rec[score_name(task)]!r}, n_iter {mine['n_iter']}/{rec.get('n_iter')}"
            )
    check("sweep points", not bad_rows, f"max |dscore| {max_dscore:.3g}" + (f"; mismatches (mine/record): {bad_rows}" if bad_rows else ""))

    test_ids = np.array(split["test"])
    pf = np.load(resolve(root, record["pred_file"]), allow_pickle=False)
    rec_pos = {g: i for i, g in enumerate(pf["ids"].tolist())}
    missing = [g for g in split["test"] if g not in rec_pos]
    if missing:
        check("pred ids", False, f"{len(missing)} test genes absent from pred_file")
        out["seconds"] = round(time.time() - t0, 1)
        return out
    order = np.array([rec_pos[g] for g in split["test"]])
    rec_pred = pf["pred"][order]

    if task == "family5":
        diff = int((rec_pred.astype(str) != pred.astype(str)).sum())
        check("test predictions", diff == 0, f"{diff} of {len(pred)} test genes differ")
        truth = labels[idx["test"]]
        rec_truth = pf["y_true"][order].astype(str)
        n_truth = int((rec_truth != truth).sum())
        check("npz y_true vs parquet family", n_truth == 0, f"{n_truth} differ")
        mine = score(task, truth[test_keep], pred[test_keep])
        rec_score = record["test_macro_f1"]
        check("test macro-F1", abs(mine - rec_score) <= 1e-12, f"mine {mine!r}, record {rec_score!r}")
    else:
        pred32 = pred.astype(np.float32)
        max_diff = float(np.max(np.abs(pred32.astype(np.float64) - rec_pred.astype(np.float64))))
        tf = np.load(resolve(root, record["targets_file"]), allow_pickle=False)
        tpos = {g: i for i, g in enumerate(tf["ids"].tolist())}
        rec_truth = tf["y_true"][np.array([tpos[g] for g in split["test"]])]
        truth = labels[idx["test"]]
        t_diff = float(np.max(np.abs(rec_truth.astype(np.float64) - truth.astype(np.float64))))
        check("targets_file vs parquet y", t_diff == 0.0, f"max |diff| {t_diff:.3g}")
        mine = score(task, truth[test_keep], pred32[test_keep])
        rec_score = record["test_r2_macro"]
        check(
            "test R2",
            abs(mine - rec_score) <= 1e-6,
            f"mine {mine!r}, record {rec_score!r}, max |pred diff| {max_diff:.3g}",
        )
        out["max_abs_pred_diff"] = max_diff

    n_scored = int(test_keep.sum())
    check("n_test_scored", n_scored == record["n_test_scored"], f"mine {n_scored}, record {record['n_test_scored']}")
    out.update(
        {
            "pick": hp,
            "edge": edge,
            "sweep": [
                {"hp": points[e]["hp"], "score": points[e]["score"], "converged": points[e]["converged"], "n_iter": points[e]["n_iter"]}
                for e in sorted(points)
            ],
            "test_score": mine,
            "record_test_score": rec_score,
            "n_test_scored": n_scored,
            "refit_converged": refit_conv,
            "refit_n_iter": refit_iter,
            "test_ids_n": len(test_ids),
            "seconds": round(time.time() - t0, 1),
        }
    )
    return out


# ---------------------------------------------------------------- picks


def argmax_records(cands: list[dict[str, Any]], label: str, checks: Checks) -> dict[str, Any]:
    """First record with the highest validation score; an exact tie is a failure."""
    best = None
    for r in cands:
        if best is None or val_score(r) > val_score(best):
            best = r
    if best is None:
        checks.add(f"pick {label}", False, "no candidates")
        raise RuntimeError(f"no candidates for {label}")
    ties = [r["feature_source"] for r in cands if r is not best and val_score(r) == val_score(best)]
    checks.add(
        f"pick {label}",
        not ties,
        f"{best['feature_source']} (val {val_score(best):.6f}) from {[r['feature_source'] for r in cands]}"
        + (f"; EXACT TIE with {ties}" if ties else ""),
    )
    return best


def compute_picks(metrics: dict[str, list[dict[str, Any]]], checks: Checks) -> tuple[dict[str, Any], dict[int, dict[str, Any]]]:
    """Return the pick report and the record behind each of the ten cells."""

    def pool(split: str, arm: str, task: str) -> list[dict[str, Any]]:
        return [r for r in metrics[split] if r["arm"] == arm and r["task"] == task and not r["shuffled_labels"]]

    def by_source(split: str, arm: str, task: str, sources: tuple[str, ...]) -> list[dict[str, Any]]:
        recs = {r["feature_source"]: r for r in pool(split, arm, task)}
        return [recs[s] for s in sources if s in recs]

    report: dict[str, Any] = {}

    def best_encoder(split: str, arm: str, task: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        per_enc: dict[str, dict[str, Any]] = {}
        tag = f"{split}/{arm}/{task}"
        for enc in ENCODERS:
            per_enc[enc] = encoder_pool_pick(split, arm, task, enc)
        best = argmax_records(list(per_enc.values()), f"{tag} best encoder", checks)
        report[f"{tag}/best_encoder"] = {"pick": best["key"], "val": val_score(best), "per_encoder": {e: r["key"] for e, r in per_enc.items()}}
        return best, per_enc

    def encoder_pool_pick(split: str, arm: str, task: str, enc: str) -> dict[str, Any]:
        prefix = ("tss_" if arm == "tss" else "") + enc + "_"
        cands = [
            r
            for r in pool(split, arm, task)
            if r["feature_source"].startswith(prefix) and not r["feature_source"].endswith(EXCLUDED_POOL_SUFFIXES)
        ]
        tag = f"{split}/{arm}/{task}/{enc} pool"
        best = argmax_records(cands, tag, checks)
        report[tag] = {"candidates": [r["feature_source"] for r in cands], "pick": best["key"], "val": val_score(best)}
        return best

    def fixed_pick(split: str, arm: str, task: str, sources: tuple[str, ...], name: str) -> dict[str, Any]:
        cands = by_source(split, arm, task, sources)
        tag = f"{split}/{arm}/{task}/{name}"
        best = argmax_records(cands, tag, checks)
        report[tag] = {"candidates": [r["feature_source"] for r in cands], "pick": best["key"], "val": val_score(best)}
        return best

    def single(split: str, arm: str, task: str, source: str) -> dict[str, Any]:
        recs = by_source(split, arm, task, (source,))
        if not recs:
            raise RuntimeError(f"no record for {split}/{arm}/{task}/{source}")
        return recs[0]

    cells: dict[int, dict[str, Any]] = {}
    cells[1], _ = best_encoder("splits.json", "cds", "family5")
    cells[2] = fixed_pick("splits.json", "cds", "family5", ("kmer", "kmer6"), "nucleotide k-mer")
    cells[3] = fixed_pick("splits.json", "cds", "family5", ("aa1", "aa2", "aa3"), "AA k-mer")
    cells[4] = single("splits.json", "cds", "family5", "esm2_650m")
    cells[5] = single("splits.json", "cds", "family5", "kmer")
    cells[6], _ = best_encoder("splits_tss_disjoint.json", "cds", "family5")
    enc6 = next(e for e in ENCODERS if cells[6]["feature_source"].startswith(e + "_"))
    cells[7] = encoder_pool_pick("splits_tss_disjoint.json", "tss", "family5", enc6)
    cells[8], _ = best_encoder("splits.json", "cds", "genept")
    fixed_pick("splits.json", "cds", "genept", ("kmer", "kmer6"), "nucleotide k-mer")
    cells[9] = fixed_pick("splits.json", "cds", "genept", ("aa1", "aa2", "aa3"), "AA k-mer")
    nulls = [
        r
        for r in metrics["null:splits.json"]
        if r["feature_source"] == "kmer" and r["task"] == "family5" and r["arm"] == "cds" and r["shuffled_labels"]
    ]
    cells[10] = min(nulls, key=lambda r: r["label_seed"])
    report["cells"] = {str(n): r["key"] for n, r in cells.items()}
    return report, cells


def edge_cells(root: Path, metrics: dict[str, list[dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    """One record per edge branch (cheapest first), from either primary split."""
    def branch(r: dict[str, Any]) -> str | None:
        if r["edge"] in ("plateau", "limit", "nonconverged"):
            return r["edge"]
        return "multi-step extension" if len(r["grid"]) >= len(BASE_EXPONENTS) + 2 else None

    def featurisable(source: str) -> bool:
        return source in SEQ_SOURCES or (root / "data" / f"dataset_{source}.parquet").exists()

    found: dict[str, dict[str, Any]] = {}
    for split in SPLIT_FILES:
        for r in metrics[split]:
            b = branch(r)
            if b is None or r["shuffled_labels"] or not featurisable(r["feature_source"]):
                continue
            if b not in found or (r["feature_dim"], r["key"]) < (found[b]["feature_dim"], found[b]["key"]):
                found[b] = r
    return found


# ---------------------------------------------------------------- statistics


def weighted_macro_f1(t: np.ndarray, p: np.ndarray, w: np.ndarray, k: int) -> float:
    """Macro-F1 over labels present in truth or prediction, with integer gene weights."""
    present = (np.bincount(t, weights=w, minlength=k) + np.bincount(p, weights=w, minlength=k)) > 0
    tp = np.bincount(t[t == p], weights=w[t == p], minlength=k)
    denom = np.bincount(t, weights=w, minlength=k) + np.bincount(p, weights=w, minlength=k)
    f1 = np.divide(2 * tp, denom, out=np.zeros(k), where=denom > 0)
    return float(f1[present].mean())


def run_statistics(
    root: Path,
    picks: dict[int, dict[str, Any]],
    masks: dict[str, tuple[set[str], set[str]]],
    clusters: dict[str, str],
    windows: dict[str, set[str]],
    protein: dict[str, set[str]],
    n_boot: int,
) -> dict[str, Any]:
    meta = load_parquet(root, META_PARQUET.removeprefix("dataset_").removesuffix(".parquet"))
    family = dict(zip(meta["ensembl_id"], meta["family"].astype(str), strict=True))
    all_genes = list(meta["ensembl_id"])
    tests = {
        "T1": (picks[1], picks[2]),
        "T2": (picks[1], picks[3]),
        "T3": (picks[4], picks[1]),
        "T4": (picks[6], picks[7]),
    }
    groups_cache: dict[str, dict[str, int]] = {}
    out: dict[str, Any] = {}
    for name, (ra, rb) in tests.items():
        split = ra["split"]
        assert rb["split"] == split
        sp = load_split(root, split)
        _, test_masked = masks[split]
        scored = [g for g in sp["test"] if g not in test_masked]
        preds = {}
        y_npz_bad = 0
        for side, rec in (("a", ra), ("b", rb)):
            pf = np.load(resolve(root, rec["pred_file"]), allow_pickle=False)
            pos = {g: i for i, g in enumerate(pf["ids"].tolist())}
            preds[side] = np.array([str(pf["pred"][pos[g]]) for g in scored])
            y_npz_bad += sum(str(pf["y_true"][pos[g]]) != family[g] for g in scored)
        truth = np.array([family[g] for g in scored])
        delta = f1_score(truth, preds["a"], average="macro") - f1_score(truth, preds["b"], average="macro")

        tss_like = split == "splits_tss_disjoint.json" or "tss" in (ra["arm"], rb["arm"])
        links = merge(protein, windows) if tss_like else protein
        kind = "windows+protein" if tss_like else "protein"
        if kind not in groups_cache:
            groups_cache[kind] = components(all_genes, clusters, links)
        gmap = groups_cache[kind]
        n_groups_restricted = len(set(components(scored, clusters, links).values()))
        glabels = [gmap[g] for g in scored]
        uniq = {gl: i for i, gl in enumerate(dict.fromkeys(glabels))}
        gidx = np.array([uniq[gl] for gl in glabels])
        n_groups = len(uniq)

        classes = sorted(set(truth) | set(preds["a"]) | set(preds["b"]))
        code = {c: i for i, c in enumerate(classes)}
        t = np.array([code[c] for c in truth])
        pa = np.array([code[c] for c in preds["a"]])
        pb = np.array([code[c] for c in preds["b"]])
        rng = np.random.default_rng(BOOT_SEED)
        deltas = np.empty(n_boot)
        for i in range(n_boot):
            draw = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)
            w = draw[gidx].astype(np.float64)
            deltas[i] = weighted_macro_f1(t, pa, w, len(classes)) - weighted_macro_f1(t, pb, w, len(classes))
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        out[name] = {
            "a": ra["key"],
            "b": rb["key"],
            "delta_point": float(delta),
            "n_test": len(scored),
            "n_groups": n_groups,
            "delta_ci95": [float(lo), float(hi)],
            "p_one_sided": float((1 + np.sum(deltas <= 0)) / (1 + n_boot)),
            "n_boot": n_boot,
            "seed": BOOT_SEED,
            "_npz_y_true_mismatches": int(y_npz_bad),
            "_n_groups_scored_only_graph": n_groups_restricted,
        }
    return out


def compare_statistics(mine: dict[str, Any], stats_path: Path, checks: Checks) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, m in mine.items():
        checks.add(f"stats {name} y_true", m["_npz_y_true_mismatches"] == 0,
                   f"{m['_npz_y_true_mismatches']} npz y_true differ from parquet family")
    if not stats_path.exists():
        print(f"SKIP statistics comparison: {stats_path.name} absent", flush=True)
        return {"skipped": "statistics.json absent"}
    conf = json.loads(stats_path.read_text())["confirmatory"]
    for name, m in mine.items():
        recs = [v for k, v in conf.items() if k.startswith(name)]
        if len(recs) != 1:
            checks.add(f"stats {name}", False, f"{len(recs)} entries start with {name}")
            continue
        r = recs[0]
        d_ci = [m["delta_ci95"][i] - r["delta_ci95"][i] for i in (0, 1)]
        d_p = m["p_one_sided"] - r["p_one_sided"]
        row = {
            "keys": (m["a"] == r["a"] and m["b"] == r["b"], f"mine ({m['a']}, {m['b']}), record ({r['a']}, {r['b']})"),
            "delta_point": (abs(m["delta_point"] - r["delta_point"]) <= 1e-12, f"mine {m['delta_point']!r}, record {r['delta_point']!r}"),
            "n_test": (m["n_test"] == r["n_test"], f"mine {m['n_test']}, record {r['n_test']}"),
            "n_groups": (
                m["n_groups"] == r["n_groups"],
                f"mine {m['n_groups']}, record {r['n_groups']}"
                + (f" (scored-only graph: {m['_n_groups_scored_only_graph']})" if m["_n_groups_scored_only_graph"] is not None else ""),
            ),
            "ci95": (
                max(abs(d) for d in d_ci) <= 0.02,
                f"mine {[round(v, 4) for v in m['delta_ci95']]}, record {[round(v, 4) for v in r['delta_ci95']]}, diff {[round(d, 4) for d in d_ci]}",
            ),
            "p_one_sided": (abs(d_p) <= 0.03, f"mine {m['p_one_sided']:.4f}, record {r['p_one_sided']:.4f}, diff {d_p:+.4f}"),
        }
        for field, (ok, detail) in row.items():
            checks.add(f"stats {name} {field}", ok, detail)
        result[name] = {field: {"ok": ok, "detail": detail} for field, (ok, detail) in row.items()}
    return result


# ---------------------------------------------------------------- main


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--records", type=Path, default=None)
    ap.add_argument("--data-root", type=Path, default=repo_root)
    ap.add_argument("--cells", default=",".join(str(c) for c in CELLS))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()

    bad_env = {v: os.environ.get(v) for v in THREAD_VARS if os.environ.get(v) != "1"}
    if bad_env:
        print(f"refusing to run: set {', '.join(THREAD_VARS)}=1 (got {bad_env})", file=sys.stderr)
        return 1

    root = args.data_root.resolve()
    records_dir = (args.records or root / "data" / "v2").resolve()
    out_path = args.out or records_dir / "reproduction.json"
    cells_wanted = [int(c) for c in args.cells.split(",") if c.strip()]
    checks = Checks()

    # Records and their digests.
    inputs = {p.name: sha256_file(p) for p in sorted(records_dir.glob("metrics_*.json")) + sorted(records_dir.glob("null_*.json"))}
    metrics: dict[str, list[dict[str, Any]]] = {}
    for split in SPLIT_FILES:
        stem = split_stem(split)
        metrics[split] = json.loads((records_dir / f"metrics_{stem}.json").read_text())
        metrics[f"null:{split}"] = json.loads((records_dir / f"null_{stem}.json").read_text())

    verify_cds(root, checks)

    # Purge.
    protein = protein_adjacency(root)
    windows, win_info = window_adjacency(root)
    print(f"INFO windows: {win_info}", flush=True)
    rules = {"splits.json": protein, "splits_tss_disjoint.json": merge(protein, windows)}
    masks: dict[str, tuple[set[str], set[str]]] = {}
    purge_report: dict[str, Any] = {"window_convention": "closed [start, end]", **win_info}
    for split in SPLIT_FILES:
        sp = load_split(root, split)
        val_m, test_m = purge_masks(sp, rules[split])
        masks[split] = (val_m, test_m)
        recs = metrics[split] + metrics[f"null:{split}"]
        bad = []
        for r in recs:
            pv, pt = set(r["purge"]["val_masked"]), set(r["purge"]["test_masked"])
            n_scored = len(sp["test"]) - len(test_m)
            if pv != val_m or pt != test_m or r["n_test_scored"] != n_scored:
                bad.append(
                    f"{r['key']}: val -{len(val_m - pv)}/+{len(pv - val_m)}, test -{len(test_m - pt)}/+{len(pt - test_m)}, "
                    f"n_test_scored {r['n_test_scored']} vs {n_scored}"
                )
        checks.add(
            f"purge {split}",
            not bad,
            f"val masked {len(val_m)}, test masked {len(test_m)}, n_test_scored {len(sp['test']) - len(test_m)}; "
            f"{len(recs)} records, {len(bad)} differ" + (f"; first: {bad[:3]}" if bad else ""),
        )
        purge_report[split] = {"val_masked": sorted(val_m), "test_masked": sorted(test_m), "records_checked": len(recs), "records_differing": len(bad)}

    # Picks.
    pick_report, cell_records = compute_picks(metrics, checks)

    # Edge-branch cells (11+): a branch with no eligible record is reported, not failed.
    edges = edge_cells(root, metrics)
    labels = {n: CELLS[n][3] for n in CELLS}
    for i, b in enumerate(EDGE_BRANCHES, start=len(CELLS) + 1):
        if b in edges:
            cell_records[i], labels[i] = edges[b], f"edge branch: {b}"
            if args.cells == ap.get_default("cells"):
                cells_wanted.append(i)
        else:
            print(f"INFO edge branch {b}: no featurisable record reaches it", flush=True)
    pick_report["edge_cells"] = {b: r["key"] for b, r in edges.items()}

    # Cells.
    cell_out: dict[str, Any] = {}
    jobs = [(n, cell_records[n]) for n in cells_wanted]
    calls = [(str(root), rec, sorted(masks[rec["split"]][0]), sorted(masks[rec["split"]][1])) for _, rec in jobs]
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(refit_cell, *zip(*calls, strict=True)))
    else:
        results = [refit_cell(*c) for c in calls]
    for (n, _), res in zip(jobs, results, strict=True):
        for line in res.pop("checks"):
            checks.add(f"[{n}] {line['check']}", line["ok"], line["detail"])
        cell_out[str(n)] = {"cell": labels[n], **res}
        print(f"INFO cell {n} took {res.get('seconds')} s", flush=True)

    # Statistics.
    clusters = cluster_map(root)
    meta_ids = set(load_parquet(root, META_PARQUET.removeprefix("dataset_").removesuffix(".parquet"))["ensembl_id"])
    checks.add("gene universe", set(clusters) == meta_ids,
               f"cluster table {len(clusters)} genes, parquet {len(meta_ids)}, differing {len(set(clusters) ^ meta_ids)}")
    stats = run_statistics(root, cell_records, masks, clusters, windows, protein, args.n_boot)
    stats_check = compare_statistics(stats, records_dir / "statistics.json", checks)
    for v in stats.values():
        v.pop("_npz_y_true_mismatches")
        v.pop("_n_groups_scored_only_graph")

    try:
        records_label = str(records_dir.relative_to(root))
    except ValueError:
        records_label = str(args.records) if args.records else str(records_dir)
    report = {
        "ok": not checks.failures,
        "records_dir": records_label,
        "inputs": inputs,
        "picks": pick_report,
        "purge": purge_report,
        "cells": cell_out,
        "statistics": stats,
        "statistics_check": stats_check,
        "cells_run": sorted(int(n) for n in cell_out),
        "cells_expected": sorted([*CELLS, *(n for n in cell_records if n > len(CELLS))]),
        "n_boot": args.n_boot,
        "checks": checks.lines,
        "failures": checks.failures,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(f"{'OK' if report['ok'] else 'FAILED'}: {len(checks.lines)} checks, {len(checks.failures)} failed; wrote {out_path.name}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
