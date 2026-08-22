"""Chunk-matched composition baseline for the E5 anchored result (MLCB camera-ready).

THE test that adjudicates whether gena_lm's TSS-anchored family5 signal (0.431) is real
encoder biology or trivial promoter composition. The existing `enformer_tss_4mer` baseline
is a 4-mer over the FULL 196 kb window (diluted like global pooling); this featurizes the
SAME ~2-4 kb anchored chunk the encoder pooled, per encoder, so the composition baseline is
matched to the encoder's own region.

Per gene: take the encoder's anchor chunk (tss_chunk_idx from center_chunk_offsets_<enc>.csv),
slice its DNA out of the cached TSS window, and featurize:
  chunk4mergc : 4-mer freqs (256) + GC/length (3)   -> 259-dim   (parallels enformer_tss_4mer)
  chunk6mer   : 6-mer freqs (4096)                   -> 4096-dim  (stronger composition baseline)

No GPU, no network (cached windows). Read-only except the new parquets it writes.

Run: uv run scripts/build_tss_composition_baseline.py
Writes: data/dataset_tss_<enc>_{chunk4mergc,chunk6mer}.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from data_loader.enformer_windows import ENFORMER_WINDOW_LENGTH, fetch_tss_window
from data_loader.model_registry import get_encoder_spec, main_encoder_names
from data_loader.multi_pool import _chunk_ids
from kmer_baseline import featurize_kmer
from composition_baseline import featurize_gc

# reuse the exact bp-length logic the diagnostic used (offsets for fast tokenizers,
# token-string lengths for slow ones) so chunk char-spans match the anchor index.
from check_tss_center_chunk import _bp_lengths  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
OFFSETS_DIR = REPO_ROOT / "analysis" / "tss_overlap"
WINDOW_CACHE = DATA / "enformer_windows"


def _chunk_dna(seq: str, bp_lens: np.ndarray, anchor_idx: int, max_tokens: int, stride: int) -> str:
    cum = np.concatenate([[0], np.cumsum(bp_lens)]).astype(np.int64)
    chunks = _chunk_ids(list(range(len(bp_lens))), max_tokens, stride)
    t0, t1 = chunks[anchor_idx][0], chunks[anchor_idx][-1] + 1
    return seq[int(cum[t0]):int(cum[t1])]


def build_encoder(enc: str) -> None:
    spec = get_encoder_spec(enc)
    tok = AutoTokenizer.from_pretrained(spec.model_name, trust_remote_code=True)
    base = pd.read_parquet(DATA / f"dataset_tss_{spec.dataset_stem}_centermean.parquet")
    anchors = pd.read_csv(OFFSETS_DIR / f"center_chunk_offsets_{enc}.csv").set_index("ensembl_id")
    print(f"=== {enc}: {len(base)} genes | is_fast={tok.is_fast} "
          f"max_tokens={spec.max_content_tokens} stride={spec.stride} ===", flush=True)

    x4, x6 = [], []
    skipped = 0
    for i, eid in enumerate(base["ensembl_id"]):
        seq = fetch_tss_window(eid, WINDOW_CACHE, length=ENFORMER_WINDOW_LENGTH)
        row = anchors.loc[eid]
        bp_lens = _bp_lengths(seq, tok, tok.is_fast)
        if bp_lens is None or int(row["n_chunks"]) != len(_chunk_ids(list(range(len(bp_lens))),
                                                                     spec.max_content_tokens, spec.stride)):
            skipped += 1
            # fall back to whole window so row counts stay aligned (rare); flagged below
            chunk = seq
        else:
            chunk = _chunk_dna(seq, bp_lens, int(row["tss_chunk_idx"]),
                               spec.max_content_tokens, spec.stride)
        x4.append(np.concatenate([featurize_kmer(chunk, 4), np.atleast_1d(featurize_gc(chunk))]).astype(np.float32))
        x6.append(featurize_kmer(chunk, 6).astype(np.float32))
        if (i + 1) % 800 == 0:
            print(f"    {i + 1}/{len(base)}", flush=True)
    if skipped:
        print(f"    WARNING {enc}: {skipped} genes fell back to whole window (chunk recompute mismatch)", flush=True)

    for suffix, xs in (("chunk4mergc", x4), ("chunk6mer", x6)):
        out = base.copy()
        out["x"] = xs
        p = DATA / f"dataset_tss_{spec.dataset_stem}_{suffix}.parquet"
        out.to_parquet(p)
        print(f"    dim={xs[0].shape[0]:5d} -> {p.name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=list(main_encoder_names()))
    args = ap.parse_args()
    for enc in args.encoders:
        build_encoder(enc)


if __name__ == "__main__":
    main()
