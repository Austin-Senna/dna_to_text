"""Rotation-invariant Ridge->GenePT metrics for the R2-metric rebuttal (MLCB R2-W4).

Reviewer 2 noted macro-averaged per-dimension R^2 is coordinate-dependent (equal weight
to all 1,536 GenePT dims) and might miss shared structure spread across dim-combinations.
For each cell this rescores the cell's stored test predictions (the camera-ready
records in data/v2; nothing is refitted, G16) and reports three views:
  - macro_r2   : uniform_average per-dim R^2 (the reported headline)
  - pooled_r2  : variance_weighted = 1 - sum(SS_res)/sum(SS_tot)   (rotation-robust)
  - retrieval  : cosine nearest-neighbour retrieval of the true gene in GenePT space
                 (top1/top5/top10, median rank) -- coordinate-free and combination-aware.
The control row is the 4-mer GenePT null band: all 200 label-shuffled runs,
rescored the same way, each metric reported as its median (G13).
Writes data/v2/ridge_robust.json (consumed by build_paper_tables.build_ridge_robust).

Run: uv run scripts/ridge_robust_metrics.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

from linear_trainer import records as R
from linear_trainer.cell import scored_predictions

ENC_DISPLAY = {"dnabert2": "DNABERT-2", "nt_v2": "NT-v2", "gena_lm": "GENA-LM", "hyena_dna": "HyenaDNA"}
CDS = "splits.json"
TSS = "splits_tss_disjoint.json"
NULL_SHUFFLES = 200


def cells() -> tuple[list[tuple[str, dict]], list[dict], dict]:
    """(display label, record) rows: ESM-2 upper bound, then each DNA encoder at its
    validation-selected pool (the validation-selected encoder tagged), then TSS. Also the 200
    label-shuffled 4-mer GenePT runs (the control) and the stamp of every file read."""
    cds_recs, tss_recs, null_recs = R.load(CDS), R.load(TSS), R.load(CDS, null=True)
    stamp = R.stamp_of(cds_recs, tss_recs, null_recs)                 # G7
    cds = R.cells(cds_recs, "cds", "genept")
    tss = R.cells(tss_recs, "tss", "genept")
    best = {e: R.best_pool(cds, e, "cds") for e in R.ENCODERS}
    top = R.best_encoder(cds, "cds")
    rows = [("ESM-2 650M (upper bound)", cds["esm2_650m"]), ("ESM-2 150M", cds["esm2_150m"])]
    for e in R.ENCODERS:
        tag = " (best DNA)" if best[e] == top else ""
        rows.append((f"{ENC_DISPLAY[e]} {best[e].rsplit('_', 1)[1]}{tag}", cds[best[e]]))
    t = R.best_pool(tss, "dnabert2", "tss")
    rows.append((f"TSS DNABERT-2 {t.rsplit('_', 1)[1]}", tss[t]))
    null = [r for r in null_recs.values() if r["task"] == "genept" and r["feature_source"] == "kmer"]
    if len(null) != NULL_SHUFFLES:
        raise R.MissingRecord(f"{len(null)} shuffled 4-mer GenePT runs, expected {NULL_SHUFFLES}")
    return rows, null, stamp


def control(null: list[dict]) -> dict:
    """The shuffled-label control (G13): each metric's median over every shuffle,
    and the central 95% of macro R^2."""
    scored = [rescore("shuffle", r) for r in null]
    out = {"label": f"Shuffled-label control (4-mer, median of {len(scored)})", "key": "null band",
           "n_shuffles": len(scored),
           "macro_r2_band95": [float(np.percentile([r["macro_r2"] for r in scored], q)) for q in (2.5, 97.5)]}
    for k in ("macro_r2", "pooled_r2", "top1", "top5", "top10", "median_rank", "n_test",
              "chance_top1", "chance_top5"):
        out[k] = float(np.median([r[k] for r in scored]))
    return out


def pooled_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Variance-weighted (pooled) R^2 -- invariant to rotations of the target basis."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean(axis=0, keepdims=True)) ** 2)
    return 1.0 - ss_res / ss_tot


def retrieval(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Cosine nearest-neighbour retrieval of each gene's true GenePT vector."""
    a = y_pred / np.clip(np.linalg.norm(y_pred, axis=1, keepdims=True), 1e-12, None)
    b = y_true / np.clip(np.linalg.norm(y_true, axis=1, keepdims=True), 1e-12, None)
    sim = a @ b.T
    n = sim.shape[0]
    order = np.argsort(-sim, axis=1)
    ranks = np.array([np.where(order[i] == i)[0][0] + 1 for i in range(n)])
    return {
        "n_test": int(n),
        "top1": float((ranks == 1).mean()),
        "top5": float((ranks <= 5).mean()),
        "top10": float((ranks <= 10).mean()),
        "median_rank": float(np.median(ranks)),
        "chance_top1": 1.0 / n,
        "chance_top5": 5.0 / n,
    }


def rescore(label: str, rec: dict) -> dict:
    arrays = scored_predictions(rec)
    y_te, y_pred = arrays["y_true"], arrays["pred"]
    macro = float(r2_score(y_te, y_pred, multioutput="uniform_average"))
    if macro != rec["test_r2_macro"]:
        raise RuntimeError(f"{rec['key']}: rescored R^2 {macro!r} != recorded {rec['test_r2_macro']!r}")
    out = {"label": label, "key": rec["key"], "alpha": rec["alpha"],
           "macro_r2": macro, "pooled_r2": float(pooled_r2(y_te, y_pred))}
    out.update(retrieval(y_pred, y_te))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=R.V2 / "ridge_robust.json")
    args = ap.parse_args()

    cell_rows, null, stamp = cells()
    rows = [rescore(label, rec) for label, rec in cell_rows] + [control(null)]
    hdr = (f'{"cell":28s} {"macroR2":>8s} {"pooledR2":>8s} '
           f'{"top1%":>7s} {"top5%":>7s} {"top10%":>7s} {"medRank":>8s}')
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f'{r["label"][:27]:28s} {r["macro_r2"]:8.4f} {r["pooled_r2"]:8.4f} '
              f'{100*r["top1"]:7.2f} {100*r["top5"]:7.2f} {100*r["top10"]:7.2f} '
              f'{r["median_rank"]:8.0f}')
    print(f'\nn_test = {rows[0]["n_test"]}   chance top5 = {100*rows[0]["chance_top5"]:.3f}%'
          f'   chance median rank ~ {rows[0]["n_test"]//2}')
    args.out.write_text(json.dumps({"stamp": stamp, "inputs": R.input_digests(), "rows": rows}, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
