"""E5 center-chunk-position check (MLCB camera-ready diagnostic).

`centermean` pooling reduces each gene to the chunk at token-count index
``n_chunks // 2`` (``pooling_aggregator.aggregate``). For fixed-stride encoders
that chunk sits at the base-pair center of the TSS-centered window (the TSS).
gena_lm uses variable-length BPE, so ``n//2``-by-token-count can land on a
genomic region *offset* from the TSS, and on a different locus than the other
encoders' center chunks. This makes centermean a possibly-uncontrolled
comparison and is the one open confound on the gena_lm centermean outlier
(0.411 hom / 0.393 disjoint vs at-chance global pooling).

This script measures, per gene, where the ``n//2`` chunk actually lands in
base pairs relative to the TSS, using ONLY the tokenizer (no model forward
pass, no GPU, no network — the TSS windows are already cached). gena_lm is the
subject; nt_v2, hyena_dna, dnabert2 are controls.

Read-only: writes only under ``analysis/tss_overlap/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from data_loader.enformer_windows import centered_window, fetch_tss_window
from data_loader.model_registry import get_encoder_spec, main_encoder_names
from data_loader.multi_pool import _chunk_ids

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "analysis" / "tss_overlap"
WINDOW_CACHE = DATA / "enformer_windows"
LOOKUP_DIR = WINDOW_CACHE / "_lookup"

# Tokens whose surface string does not correspond to consumed base pairs.
_SPECIAL_MARKERS = ("##", "▁", "Ġ")  # WordPiece "##", SentencePiece "_", GPT2 "G-dot"


def _tss_offset_bp(gene_id: str, seq_len: int, length: int) -> int | None:
    """0-based index of the TSS within the cached forward-strand window.

    Recomputed from the cached Ensembl lookup + ``centered_window`` rather than
    assumed to be ``length // 2`` so chromosome-start-clipped genes are correct.
    """
    lookup = LOOKUP_DIR / f"{gene_id}.json"
    if not lookup.exists():
        return None
    data = json.loads(lookup.read_text())
    _, region_start, _ = centered_window(
        seq_region_name=str(data["seq_region_name"]),
        start=int(data["start"]),
        end=int(data["end"]),
        strand=int(data.get("strand", 1)),
        length=length,
    )
    tss = int(data["start"]) if int(data.get("strand", 1)) >= 0 else int(data["end"])
    offset = tss - region_start
    return int(np.clip(offset, 0, seq_len - 1))


def _bp_lengths(seq: str, tok, is_fast: bool) -> np.ndarray | None:
    """Per-token base-pair length, from exact offsets (fast) or token strings.

    Returns None if the reconstructed base-pair coverage does not match the
    sequence length (e.g. UNK/normalisation swallowed bases) so the gene can be
    excluded rather than silently mismeasured.
    """
    if is_fast:
        offs = tok(seq, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        if not offs:
            return None
        lens = np.array([b - a for a, b in offs], dtype=np.int64)
        if int(offs[-1][1]) != len(seq):
            return None
        return lens
    ids = tok(seq, add_special_tokens=False)["input_ids"]
    toks = tok.convert_ids_to_tokens(ids)
    clean = [t for t in toks]
    for m in _SPECIAL_MARKERS:
        clean = [t.replace(m, "") for t in clean]
    lens = np.array([len(t) for t in clean], dtype=np.int64)
    if int(lens.sum()) != len(seq):
        return None
    return lens


def _analyze_gene(bp_lens: np.ndarray, tss_bp: int, max_tokens: int, stride: int) -> dict:
    # cumulative base-pair position at the START of each token (len n_tokens + 1).
    cum = np.concatenate([[0], np.cumsum(bp_lens)]).astype(np.int64)
    total_bp = int(cum[-1])

    chunks = _chunk_ids(list(range(len(bp_lens))), max_tokens, stride)
    n_chunks = len(chunks)
    centers = np.empty(n_chunks, dtype=np.float64)
    starts = np.empty(n_chunks, dtype=np.int64)
    ends = np.empty(n_chunks, dtype=np.int64)
    for k, chunk in enumerate(chunks):
        t0, t1 = chunk[0], chunk[-1] + 1  # token index range [t0, t1)
        starts[k] = cum[t0]
        ends[k] = cum[t1]
        centers[k] = 0.5 * (cum[t0] + cum[t1])

    center_idx = n_chunks // 2
    tss_chunk_idx = int(np.argmin(np.abs(centers - tss_bp)))
    center_contains_tss = bool(starts[center_idx] <= tss_bp < ends[center_idx])
    # Chunks that actually bracket the TSS base pair. Chunks overlap (stride), so
    # several may contain it; pick the one whose center is nearest the TSS. -1 if
    # none contains it (the wide-window encoders: their chunk is the nearest
    # available, not a TSS-spanning one). Feeds the GAP 5 contains-TSS anchor rule.
    contains = np.flatnonzero((starts <= tss_bp) & (tss_bp < ends))
    if contains.size:
        tss_contain_chunk_idx = int(contains[np.argmin(np.abs(centers[contains] - tss_bp))])
    else:
        tss_contain_chunk_idx = -1
    anchor_contains_tss = bool(starts[tss_chunk_idx] <= tss_bp < ends[tss_chunk_idx])
    chunk_bp = ends - starts
    return {
        "n_tokens": len(bp_lens),
        "n_chunks": n_chunks,
        "total_bp": total_bp,
        "tss_bp": int(tss_bp),
        "center_idx": center_idx,
        "tss_chunk_idx": tss_chunk_idx,
        "tss_contain_chunk_idx": tss_contain_chunk_idx,
        "delta_chunks": center_idx - tss_chunk_idx,
        "center_chunk_bp_center": float(centers[center_idx]),
        "delta_bp": float(centers[center_idx] - tss_bp),
        "center_contains_tss": center_contains_tss,
        "anchor_contains_tss": anchor_contains_tss,
        "bp_per_token": total_bp / max(len(bp_lens), 1),
        "chunk_bp_min": int(chunk_bp.min()),
        "chunk_bp_max": int(chunk_bp.max()),
    }


def run_encoder(enc: str, length: int, max_genes: int | None) -> pd.DataFrame:
    spec = get_encoder_spec(enc)
    tok = AutoTokenizer.from_pretrained(spec.model_name, trust_remote_code=True)
    parquet = DATA / f"dataset_tss_{spec.cache_name}_centermean.parquet"
    gene_ids = pd.read_parquet(parquet, columns=["ensembl_id"])["ensembl_id"].tolist()
    if max_genes is not None:
        gene_ids = gene_ids[:max_genes]
    print(f"=== {enc}: {spec.model_name} | max_content_tokens={spec.max_content_tokens} "
          f"stride={spec.stride} is_fast={tok.is_fast} | {len(gene_ids)} genes ===")

    rows = []
    skipped = 0
    for i, gid in enumerate(gene_ids):
        seq = fetch_tss_window(gid, WINDOW_CACHE, length=length)
        if not seq:
            skipped += 1
            continue
        tss_bp = _tss_offset_bp(gid, len(seq), length)
        if tss_bp is None:
            skipped += 1
            continue
        bp_lens = _bp_lengths(seq, tok, tok.is_fast)
        if bp_lens is None or len(bp_lens) == 0:
            skipped += 1
            continue
        r = _analyze_gene(bp_lens, tss_bp, spec.max_content_tokens, spec.stride)
        r["ensembl_id"] = gid
        r["seq_len"] = len(seq)
        rows.append(r)
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(gene_ids)}")
    if skipped:
        print(f"    skipped {skipped} genes (missing window/lookup or bp reconstruction mismatch)")
    return pd.DataFrame(rows)


def summarize(enc: str, df: pd.DataFrame) -> dict:
    abs_bp = df["delta_bp"].abs()
    return {
        "encoder": enc,
        "n_genes": int(len(df)),
        "frac_center_contains_tss": float((df["center_contains_tss"]).mean()),
        "frac_anchor_contains_tss": float((df["anchor_contains_tss"]).mean()),
        "frac_any_containing_chunk": float((df["tss_contain_chunk_idx"] >= 0).mean()),
        "frac_center_is_tss_chunk": float((df["delta_chunks"] == 0).mean()),
        "median_abs_delta_bp": float(abs_bp.median()),
        "median_signed_delta_bp": float(df["delta_bp"].median()),
        "iqr_abs_delta_bp": [float(abs_bp.quantile(0.25)), float(abs_bp.quantile(0.75))],
        "median_delta_chunks": float(df["delta_chunks"].median()),
        "median_abs_delta_chunks": float(df["delta_chunks"].abs().median()),
        "median_n_chunks": float(df["n_chunks"].median()),
        "n_chunks_range": [int(df["n_chunks"].min()), int(df["n_chunks"].max())],
        "median_bp_per_token": float(df["bp_per_token"].median()),
        "median_chunk_bp_span": [int(df["chunk_bp_min"].median()), int(df["chunk_bp_max"].median())],
        "median_seq_len": float(df["seq_len"].median()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=list(main_encoder_names()))
    ap.add_argument("--length", type=int, default=None, help="window length; default = spec/ENFORMER")
    ap.add_argument("--max-genes", type=int, default=None, help="debug limit")
    args = ap.parse_args()

    from data_loader.enformer_windows import ENFORMER_WINDOW_LENGTH
    length = args.length or ENFORMER_WINDOW_LENGTH

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for enc in args.encoders:
        df = run_encoder(enc, length, args.max_genes)
        if df.empty:
            print(f"    {enc}: no rows, skipping outputs")
            continue
        csv = OUT_DIR / f"center_chunk_offsets_{enc}.csv"
        df.to_csv(csv, index=False)
        s = summarize(enc, df)
        summaries.append(s)
        print(f"    -> {csv.name}: contains_TSS={s['frac_center_contains_tss']:.3f} "
              f"median|Δbp|={s['median_abs_delta_bp']:.0f} "
              f"median|Δchunks|={s['median_abs_delta_chunks']:.0f} "
              f"bp/tok={s['median_bp_per_token']:.1f}")

    if summaries:
        (OUT_DIR / "center_chunk_summary.json").write_text(json.dumps(summaries, indent=2))
        print(f"\nwrote {OUT_DIR / 'center_chunk_summary.json'}")


if __name__ == "__main__":
    main()
