"""E5 center-chunk-position check (MLCB camera-ready diagnostic).

`centermean` pooling reduces each gene to the chunk at token-count index
``n_chunks // 2`` (``pooling_aggregator.aggregate``). Every window is in gene
orientation with the canonical TSS at ``TSS_INDEX``, but the ``n // 2`` chunk is
a count-based midpoint: it sits 3' of the base-pair centre (for HyenaDNA the
whole chunk lies past the TSS), so it is not the TSS chunk.
gena_lm uses variable-length BPE, so ``n//2``-by-token-count can land on a
genomic region *offset* from the TSS, and on a different locus than the other
encoders' center chunks. That would make centermean an uncontrolled comparison
across encoders, which is why the E5 features anchor on the TSS chunk instead
(``tss_chunk_idx``, read by ``build_tss_anchored_datasets.py``).

This script measures, per gene, where the ``n//2`` chunk actually lands in
base pairs relative to the TSS, using ONLY the tokenizer (no model forward
pass, no GPU, no network: the windows are read from the manifest-checked
cache). gena_lm is the subject; nt_v2, hyena_dna, dnabert2 are controls.

Read-only: writes only under ``analysis/tss_overlap/``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from data_loader.cache_meta import StaleCache
from data_loader.enformer_windows import (
    ENFORMER_WINDOW_LENGTH,
    TSS_INDEX,
    load_manifest,
    read_window,
    sha256_seq,
)
from data_loader.model_registry import get_encoder_spec, main_encoder_names
from data_loader.multi_pool import _chunk_ids

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
OUT_DIR = REPO_ROOT / "analysis" / "tss_overlap"

# Tokens whose surface string does not correspond to consumed base pairs.
_SPECIAL_MARKERS = ("##", "▁", "Ġ")  # WordPiece "##", SentencePiece "_", GPT2 "G-dot"


def verify_offsets(anchors: pd.DataFrame, spec, genes) -> None:
    """Refuse offsets computed on other windows or with another tokenizer (G19).

    ``anchors`` is an offsets CSV indexed by ``ensembl_id``. Each row carries the
    sha256 of the window it was computed on and the tokenizer revision; both
    must match the manifest and the encoder spec for every gene in ``genes``.
    """
    need = {"window_sha256", "tokenizer_revision"}
    if not need <= set(anchors.columns):
        raise StaleCache(f"{spec.name}: offsets lack {sorted(need - set(anchors.columns))}; "
                         "rerun scripts/check_tss_center_chunk.py on the current windows")
    missing = [g for g in genes if g not in anchors.index]
    if missing:
        raise StaleCache(f"{spec.name}: no offsets for {len(missing)} genes: {missing[:5]}")
    shas = load_manifest()["sha256"]
    stale = [g for g in genes if anchors.at[g, "window_sha256"] != shas[g]
             or anchors.at[g, "tokenizer_revision"] != spec.revision]
    if stale:
        raise StaleCache(f"{spec.name}: offsets for {len(stale)} genes come from other windows "
                         f"or another tokenizer revision: {stale[:5]}")


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
    tok = AutoTokenizer.from_pretrained(spec.model_name, revision=spec.revision,
                                        trust_remote_code=True)
    parquet = DATA / f"dataset_tss_{spec.cache_name}_centermean.parquet"
    gene_ids = pd.read_parquet(parquet, columns=["ensembl_id"])["ensembl_id"].tolist()
    if max_genes is not None:
        gene_ids = gene_ids[:max_genes]
    print(f"=== {enc}: {spec.model_name} | max_content_tokens={spec.max_content_tokens} "
          f"stride={spec.stride} is_fast={tok.is_fast} | {len(gene_ids)} genes ===")

    rows = []
    skipped = 0
    for i, gid in enumerate(gene_ids):
        seq = read_window(gid, length=length)
        tss_bp = TSS_INDEX  # gene orientation, N-padded at edges: the TSS never moves
        bp_lens = _bp_lengths(seq, tok, tok.is_fast)
        if bp_lens is None or len(bp_lens) == 0:
            skipped += 1
            continue
        r = _analyze_gene(bp_lens, tss_bp, spec.max_content_tokens, spec.stride)
        r["ensembl_id"] = gid
        r["seq_len"] = len(seq)
        r["window_sha256"] = sha256_seq(seq)
        r["tokenizer_revision"] = spec.revision
        rows.append(r)
        if (i + 1) % 500 == 0:
            print(f"    {i + 1}/{len(gene_ids)}")
    if skipped:
        print(f"    skipped {skipped} genes (bp reconstruction mismatch)")
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

    length = args.length or ENFORMER_WINDOW_LENGTH

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for enc in args.encoders:
        df = run_encoder(enc, length, args.max_genes)
        if df.empty:
            print(f"    {enc}: no rows, skipping outputs")
            continue
        pilot = "_pilot" if args.max_genes is not None else ""  # never overwrite the real table
        csv = OUT_DIR / f"center_chunk_offsets_{enc}{pilot}.csv"
        df.to_csv(csv, index=False)
        s = summarize(enc, df)
        summaries.append(s)
        print(f"    -> {csv.name}: contains_TSS={s['frac_center_contains_tss']:.3f} "
              f"median|Δbp|={s['median_abs_delta_bp']:.0f} "
              f"median|Δchunks|={s['median_abs_delta_chunks']:.0f} "
              f"bp/tok={s['median_bp_per_token']:.1f}")

    if summaries and args.max_genes is None:
        (OUT_DIR / "center_chunk_summary.json").write_text(json.dumps(summaries, indent=2))
        print(f"\nwrote {OUT_DIR / 'center_chunk_summary.json'}")


if __name__ == "__main__":
    main()
