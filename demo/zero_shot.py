"""Zero-shot demo: pick a small sample of test-set genes, predict their family
with the headline NT-v2 + meanD pipeline, and report the top-5 nearest
neighbours by embedding cosine.

Mechanism:
    1. Train logistic regression on train+val of dataset_nt_v2_meanD.parquet
       (5-way family target). C=1.0 — the value Phase 4a's matrix found best
       for nt_v2 + family5; we use the same value here for the meanD variant
       to keep the demo aligned with the matrix protocol.
    2. Pick four test genes: two with the longest summaries (well-characterised,
       calibration cohort) and two with the shortest summaries (poorly
       characterised, novelty cohort).
    3. For each, print: predicted family + per-class probabilities; top-5
       nearest train+val gene neighbours by NT-v2_meanD embedding cosine.
    4. Render a markdown table to demo/output.md for embedding in the deck.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
OUT = Path(__file__).resolve().parent / "output.md"

DATASET = DATA / "dataset_nt_v2_meanD.parquet"
SPLITS = DATA / "splits.json"
PROBE_C = 1.0
TOP_K = 5


def main():
    df = pd.read_parquet(DATASET)
    splits = json.loads(SPLITS.read_text())
    train_val_ids = set(splits["train"]) | set(splits["val"])
    test_ids = set(splits["test"])

    train_val = df[df["ensembl_id"].isin(train_val_ids)].reset_index(drop=True)
    test = df[df["ensembl_id"].isin(test_ids)].reset_index(drop=True)

    X_tv = np.stack(train_val["x"].values).astype(np.float32)
    y_tv = train_val["family"].values
    X_te = np.stack(test["x"].values).astype(np.float32)

    print(f"  dataset: {DATASET.name}  X_tv={X_tv.shape}  X_te={X_te.shape}")

    print(f"\n  fitting logistic probe (C={PROBE_C}, multinomial, lbfgs)...")
    probe = LogisticRegression(C=PROBE_C, max_iter=2000, solver="lbfgs")
    probe.fit(X_tv, y_tv)
    print(f"    classes: {probe.classes_.tolist()}")

    test = test.assign(summary_len=test["summary"].fillna("").str.len())
    nonempty = test[test["summary_len"] >= 30].copy()
    well = nonempty.nlargest(2, "summary_len")
    poor = nonempty.nsmallest(2, "summary_len")
    sample = pd.concat([well.assign(cohort="well-characterised"),
                        poor.assign(cohort="poorly characterised")]).reset_index(drop=True)
    print(f"\n  picked {len(sample)} demo genes "
          f"(2 well-characterised + 2 poorly characterised, by summary length)")

    X_norm_tv = X_tv / np.clip(np.linalg.norm(X_tv, axis=1, keepdims=True), 1e-12, None)

    rows: list[dict] = []
    for _, gene in sample.iterrows():
        x = np.asarray(gene["x"], dtype=np.float32)
        x_norm = x / max(float(np.linalg.norm(x)), 1e-12)

        probs = probe.predict_proba(x[None, :])[0]
        pred_idx = int(probs.argmax())
        pred_family = probe.classes_[pred_idx]
        prob_dict = {cls: float(p) for cls, p in zip(probe.classes_, probs)}

        cos = X_norm_tv @ x_norm
        top_idx = np.argsort(-cos)[:TOP_K]
        neighbours = train_val.iloc[top_idx].reset_index(drop=True)
        neighbour_cos = cos[top_idx]

        rows.append({
            "gene": gene,
            "pred_family": pred_family,
            "true_family": gene["family"],
            "probs": prob_dict,
            "neighbours": neighbours,
            "neighbour_cos": neighbour_cos,
        })

    print(f"\n  writing demo output -> {OUT}")
    with OUT.open("w") as f:
        f.write("# Zero-shot demo — NT-v2 + meanD\n\n")
        f.write(
            "Pipeline: tokenise CDS with NT-v2 (special tokens wrapped per chunk) "
            "→ extract per-chunk mean-token vectors → reduce across chunks via "
            "`concat[first_chunk, last_chunk, mean_chunks]` (the headline `meanD` "
            "variant) → 5-way logistic probe (C=1.0) trained on train+val of the "
            "frozen 70/15/15 split.\n\n"
        )
        f.write(
            "Sample: 4 test-set genes — the two with the longest NCBI summaries "
            "(*well-characterised*, calibration) and the two with the shortest "
            "non-trivial summaries (*poorly characterised*, novelty).\n\n"
        )
        for r in rows:
            g = r["gene"]
            cohort_marker = "✅" if r["pred_family"] == r["true_family"] else "❌"
            f.write(f"## {g['symbol']}  ({g['ensembl_id']})\n\n")
            f.write(f"- **Cohort:** {g['cohort']}\n")
            f.write(f"- **True family:** `{r['true_family']}`\n")
            f.write(f"- **Predicted family:** `{r['pred_family']}` {cohort_marker}\n")
            probs_str = " · ".join(
                f"{cls}={p:.3f}" for cls, p in sorted(
                    r["probs"].items(), key=lambda kv: -kv[1]
                )
            )
            f.write(f"- **Class probabilities:** {probs_str}\n")
            f.write(f"- **Summary ({g['summary_len']} chars):** "
                    f"{(g['summary'] or '').strip()[:200]}"
                    f"{'…' if g['summary_len'] > 200 else ''}\n\n")
            f.write(f"**Top-{TOP_K} nearest neighbours in train+val "
                    f"(NT-v2 + meanD embedding cosine):**\n\n")
            f.write("| # | Symbol | Family | Cos | Summary (first 80 chars) |\n")
            f.write("|---|---|---|---:|---|\n")
            for i, (_, n) in enumerate(r["neighbours"].iterrows(), start=1):
                summary_excerpt = (n["summary"] or "").strip()[:80].replace("|", "/")
                f.write(
                    f"| {i} | `{n['symbol']}` | `{n['family']}` | "
                    f"{r['neighbour_cos'][i-1]:.4f} | "
                    f"{summary_excerpt}{'…' if len(n['summary'] or '') > 80 else ''} |\n"
                )
            n_majority = (r["neighbours"]["family"] == r["pred_family"]).sum()
            f.write(
                f"\nTop-{TOP_K} neighbour-vote agreement with predicted family: "
                f"**{n_majority}/{TOP_K}**\n\n---\n\n"
            )

    n_correct = sum(1 for r in rows if r["pred_family"] == r["true_family"])
    print(f"  predictions correct: {n_correct}/{len(rows)}")
    print(f"  done -> {OUT}")


if __name__ == "__main__":
    main()
