"""Extract Enformer comparator features and matched TSS-window 4-mer features."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader.enformer_encoder import embed_all_enformer
from data_loader.enformer_windows import ENFORMER_WINDOW_LENGTH, fetch_tss_window
from kmer_baseline import featurize_sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
FEATURE_NAMES = ("trunk_global", "trunk_center", "tracks_center")


def _load_windows(meta: pd.DataFrame, cache_dir: Path, length: int) -> dict[str, str]:
    windows: dict[str, str] = {}
    for eid in meta["ensembl_id"]:
        seq = fetch_tss_window(eid, cache_dir, length=length)
        if seq:
            windows[eid] = seq
    return windows


def _write_feature_dataset(base: pd.DataFrame, features: dict[str, dict[str, np.ndarray]], name: str) -> None:
    rows = base[base["ensembl_id"].isin(features)].copy().reset_index(drop=True)
    rows["x"] = rows["ensembl_id"].map(lambda eid: features[eid][name])
    out_path = DATA / f"dataset_enformer_{name}.parquet"
    rows.to_parquet(out_path)
    print(f"  wrote {out_path.name}: {len(rows)} rows")


def _write_tss_kmer_dataset(base: pd.DataFrame, windows: dict[str, str]) -> None:
    rows = base[base["ensembl_id"].isin(windows)].copy().reset_index(drop=True)
    rows["x"] = rows["ensembl_id"].map(lambda eid: featurize_sequence(windows[eid]))
    out_path = DATA / "dataset_enformer_tss_4mer.parquet"
    rows.to_parquet(out_path)
    print(f"  wrote {out_path.name}: {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template-dataset", default=str(DATA / "dataset.parquet"))
    ap.add_argument("--window-cache", default=str(DATA / "enformer_windows"))
    ap.add_argument("--feature-cache", default=str(DATA / "enformer_features"))
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    ap.add_argument("--length", type=int, default=ENFORMER_WINDOW_LENGTH)
    ap.add_argument("--center-bins", type=int, default=16)
    ap.add_argument("--skip-model", action="store_true", help="only fetch windows and build TSS 4-mer")
    args = ap.parse_args()

    base = pd.read_parquet(args.template_dataset)
    print(f"=== Enformer comparator: {len(base)} genes from {Path(args.template_dataset).name} ===")
    windows = _load_windows(base, Path(args.window_cache), length=args.length)
    print(f"  TSS windows on disk: {len(windows)}")
    _write_tss_kmer_dataset(base, windows)

    if args.skip_model:
        return

    device = None if args.device == "auto" else args.device
    features = embed_all_enformer(
        windows,
        cache_dir=args.feature_cache,
        device=device,
        center_bins=args.center_bins,
    )
    for name in FEATURE_NAMES:
        _write_feature_dataset(base, features, name)


if __name__ == "__main__":
    main()
