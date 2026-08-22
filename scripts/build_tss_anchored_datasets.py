"""Materialize TSS-anchored pooling datasets (MLCB E5 confirmatory re-probe).

`centermean` reduces each gene to the chunk at token-count index ``n_chunks // 2``,
which the center-chunk-position diagnostic showed sits ~half a chunk *downstream*
of the TSS (and off-TSS entirely for the wide-window encoders). This rebuilds the
feature by selecting, per gene, the chunk whose content is most TSS-centered
(``argmin_k |chunk_bp_center[k] - tss_bp|``, precomputed as ``tss_chunk_idx`` in
``analysis/tss_overlap/center_chunk_offsets_<enc>.csv``), giving a correctly-labeled,
same-locus TSS pooling for a controlled family5 comparison.

No GPU / no re-extraction: the per-chunk ``mean`` vectors are already cached in
``data/tss_chunk_reductions_<enc>/*.npz``; we only pick a different row.

Run: uv run scripts/build_tss_anchored_datasets.py
Writes: data/dataset_tss_<enc>_tssanchored.parquet (one per encoder).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from data_loader.model_registry import get_encoder_spec, main_encoder_names

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
OFFSETS_DIR = REPO_ROOT / "analysis" / "tss_overlap"


# Anchor rules: which chunk index to pool per gene, from the offsets CSV.
#   nearestcenter -> tss_chunk_idx (argmin |chunk_center - TSS|); the confirmatory
#                    re-probe feature. Output stem "tssanchored".
#   containstss   -> the chunk that BRACKETS the TSS base pair (tss_contain_chunk_idx),
#                    falling back to nearestcenter where none contains it (GAP 5 anchor
#                    robustness). Output stem "tsscontain".
ANCHOR_RULES = {
    "nearestcenter": ("tss_chunk_idx", "tssanchored"),
    "containstss": ("tss_contain_chunk_idx", "tsscontain"),
}


def build_encoder(enc: str, anchor: str = "nearestcenter") -> Path:
    idx_col, out_stem = ANCHOR_RULES[anchor]
    spec = get_encoder_spec(enc)
    base_path = DATA / f"dataset_tss_{spec.dataset_stem}_centermean.parquet"
    chunk_dir = DATA / f"tss_chunk_reductions_{spec.cache_name}"
    offsets_csv = OFFSETS_DIR / f"center_chunk_offsets_{enc}.csv"
    for p in (base_path, chunk_dir, offsets_csv):
        if not p.exists():
            raise FileNotFoundError(f"{enc}: missing required input {p}")

    base = pd.read_parquet(base_path)
    anchors = pd.read_csv(offsets_csv).set_index("ensembl_id")
    if idx_col not in anchors.columns:
        raise RuntimeError(
            f"{enc}: {offsets_csv.name} lacks '{idx_col}' — regenerate with "
            "check_tss_center_chunk.py (GAP 5).")
    print(f"=== {enc}: {len(base)} genes | base={base_path.name} anchors={offsets_csv.name} "
          f"| anchor={anchor} ({idx_col}) ===")

    x_col = []
    changed = 0    # anchor chunk != n//2 center chunk
    fallback = 0   # containstss: no bracketing chunk, fell back to nearest-center
    for eid in base["ensembl_id"]:
        if eid not in anchors.index:
            raise RuntimeError(f"{enc}: no anchor row for {eid}")
        row = anchors.loc[eid]
        anchor_idx = int(row[idx_col])
        if anchor_idx < 0:  # containstss with no bracketing chunk
            anchor_idx = int(row["tss_chunk_idx"])
            fallback += 1
        n_chunks_csv = int(row["n_chunks"])
        with np.load(chunk_dir / f"{eid}.npz") as npz:
            mean = npz["mean"]
        if mean.shape[0] != n_chunks_csv:
            raise RuntimeError(
                f"{enc}/{eid}: npz n_chunks {mean.shape[0]} != diagnostic {n_chunks_csv} "
                "(tokenization drift — anchor index invalid)"
            )
        if not 0 <= anchor_idx < mean.shape[0]:
            raise RuntimeError(f"{enc}/{eid}: anchor_idx {anchor_idx} out of range {mean.shape[0]}")
        x_col.append(mean[anchor_idx].astype(np.float32))
        if anchor_idx != int(row["center_idx"]):
            changed += 1

    dim = x_col[0].shape[0]
    assert all(v.shape == (dim,) for v in x_col), f"{enc}: dim mismatch across rows"
    out = base.copy()
    out["x"] = x_col
    out_path = DATA / f"dataset_tss_{spec.dataset_stem}_{out_stem}.parquet"
    out.to_parquet(out_path)
    msg = (f"    dim={dim}  anchor!=center in {changed}/{len(base)} "
           f"({100*changed/len(base):.0f}%)")
    if anchor == "containstss":
        msg += f"  fell back to nearest-center in {fallback}/{len(base)} ({100*fallback/len(base):.0f}%)"
    print(f"{msg}  -> {out_path.name}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=list(main_encoder_names()))
    ap.add_argument("--anchor", choices=list(ANCHOR_RULES), default="nearestcenter",
                    help="chunk-selection rule (default: nearestcenter = tssanchored)")
    args = ap.parse_args()
    for enc in args.encoders:
        build_encoder(enc, anchor=args.anchor)


if __name__ == "__main__":
    main()
