"""The one feature-source registry: which X and which y a probe cell reads.

A source is a registry key (an encoder/pooling parquet, a comparator parquet,
or an on-the-fly compositional featuriser) or a path to a parquet. Every load
names its split file explicitly; nothing reads a default split behind the
caller's back (ledger G15).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from binary_tasks import BINARY_TASKS, load_binary_split
from composition_baseline import featurize_aa_kmer, featurize_codon, featurize_gc
from data_loader.model_registry import dataset_paths, encoder_pools
from data_loader.sequence_fetcher import fetch_cds
from kmer_baseline import featurize_kmer
from splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"
SEQUENCES_DIR = DATA / "sequences"
TASKS = ("family5", "genept") + tuple(BINARY_TASKS)
ENCODERS = ("dnabert2", "nt_v2", "gena_lm", "hyena_dna")

# One dict object, shared by every script: some register extra parquets at run
# time (E5, Enformer pooling), and all readers must see them.
DATASET_PATHS: dict[str, Path] = dataset_paths(include_variants=True)
DATASET_PATHS.update({
    "enformer_trunk_global": DATA / "dataset_enformer_trunk_global.parquet",
    "enformer_trunk_center": DATA / "dataset_enformer_trunk_center.parquet",
    "enformer_tracks_center": DATA / "dataset_enformer_tracks_center.parquet",
    "enformer_tss_4mer": DATA / "dataset_enformer_tss_4mer.parquet",
    "esm2_150m": DATA / "dataset_esm2_150m.parquet",
    "esm2_650m": DATA / "dataset_esm2_650m.parquet",
})
for _encoder in ENCODERS:
    DATASET_PATHS[f"tss_{_encoder}"] = DATA / f"dataset_tss_{_encoder}.parquet"
    # The encoder's TSS grid plus the TSS-only centermean template.
    for _variant in (*encoder_pools(_encoder, "TSS"), "centermean"):
        DATASET_PATHS[f"tss_{_encoder}_{_variant}"] = DATA / f"dataset_tss_{_encoder}_{_variant}.parquet"
    # TSS-anchored chunk pooling (E5). Not an aggregate() variant: it picks the
    # TSS-centred chunk per gene from external window info.
    DATASET_PATHS[f"tss_{_encoder}_tssanchored"] = DATA / f"dataset_tss_{_encoder}_tssanchored.parquet"
del _encoder, _variant

# On-the-fly compositional feature sources, computed from the cached CDS.
SYNTHETIC_FEATURIZERS = {
    "kmer": lambda s: featurize_kmer(s, 4),
    "kmer6": lambda s: featurize_kmer(s, 6),
    "codon": featurize_codon,
    "aa1": lambda s: featurize_aa_kmer(s, 1),
    "aa2": lambda s: featurize_aa_kmer(s, 2),
    "aa3": lambda s: featurize_aa_kmer(s, 3),
    "gc": featurize_gc,
}


def _pick_meta_parquet() -> Path:
    """Any encoder parquet supplies the shared {ensembl_id, family, y} metadata."""
    for key in ("dnabert2", "dnabert2_meanmean", "gena_lm_meanmean"):
        p = DATASET_PATHS.get(key)
        if p is not None and Path(p).exists():
            return Path(p)
    raise FileNotFoundError("no metadata parquet found for synthetic features")


META_PARQUET = _pick_meta_parquet()


def cell_name(source: str | Path) -> str:
    return Path(source).stem if isinstance(source, Path) else source


def _parquet_for(source: str | Path) -> Path | None:
    if isinstance(source, Path):
        return source
    if source in DATASET_PATHS:
        return Path(DATASET_PATHS[source])
    if source in SYNTHETIC_FEATURIZERS:
        return None
    raise ValueError(f"unknown feature source: {source!r}")


def synthetic_features(source: str, ids: list[str]) -> np.ndarray:
    fn = SYNTHETIC_FEATURIZERS[source]
    rows = []
    for eid in ids:
        seq = fetch_cds(eid, SEQUENCES_DIR)
        if not seq:
            raise RuntimeError(f"missing cached CDS for {eid}")
        rows.append(fn(seq))
    return np.stack(rows).astype(np.float32)


# A matrix whose columns barely move across genes carries no gene signal (the May
# HyenaDNA Mean-CLS: every gene got the same vector, G3). The statistic is the
# median over columns of std/|mean|, skipping all-zero columns (sparse k-mer
# features have many), so one large constant column cannot mask real signal.
# Measured Sept 29 over the 72 local parquets: HyenaDNA clsmean 3.7e-6 (CDS) and
# 7.0e-7 (TSS); the lowest real source 6.2e-3 (TSS GENA-LM clsmean); synthetic
# kmer6/aa3/gc 2.3/6.7/0.044. The threshold sits well clear of both sides.
CONSTANT_RTOL = 1e-4


class ConstantFeatures(RuntimeError):
    """A feature matrix is (numerically) the same vector for every gene."""


def check_not_constant(X: np.ndarray, name: str) -> float:
    """Raise ``ConstantFeatures`` unless the columns vary across genes (G3)."""
    X = np.asarray(X, dtype=np.float64)
    live = np.abs(X).max(axis=0) > 0
    if not live.any():
        raise ConstantFeatures(f"{name}: every feature is zero")
    sd, mu = X[:, live].std(axis=0), np.abs(X[:, live].mean(axis=0))
    spread = float(np.median(sd / np.maximum(mu, 1e-300)))
    if spread < CONSTANT_RTOL:
        raise ConstantFeatures(f"{name}: median column std/|mean| is {spread:.1e} "
                               f"(< {CONSTANT_RTOL:g}); the features carry no gene signal")
    return spread


def split_file(task: str, splits_path: Path) -> Path:
    """The split file a cell actually reads (binary tasks carry their own)."""
    if task in BINARY_TASKS:
        return DATA / f"binary_{task}.json"
    return Path(splits_path)


def load(source: str | Path, task: str, split: str, splits_path: Path
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, ensembl_ids) for one split of one cell, row-aligned.

    ``task``: ``family5`` (y = family label), ``genept`` (y = GenePT vector) or
    a binary task (y = 0/1 from its own frozen subset file).
    """
    if task not in TASKS:
        raise ValueError(f"unknown task: {task!r}")
    parquet = _parquet_for(source)
    meta_parquet = parquet if parquet is not None else META_PARQUET
    if task in BINARY_TASKS:
        X, y, meta = load_binary_split(task, split, dataset_path=meta_parquet)
    else:
        X, Y, meta = load_split(split, dataset_path=meta_parquet, splits_path=Path(splits_path))
        y = meta["family"].to_numpy() if task == "family5" else Y
    ids = meta["ensembl_id"].to_numpy()
    if parquet is None:
        X = synthetic_features(source, ids.tolist())
    check_not_constant(X, f"{cell_name(source)} ({split})")
    return X, y, ids


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
