"""Cross-modal demo: DNA -> family + DNA -> retrieved gene summaries.

Two probes on the same DNA input:
  1. NT-v2 + meanD  -> logistic probe (C=1.0)        -> predicted family
  2. DNABERT-2 + meanG -> Ridge probe (alpha=10.0)  -> predicted GenePT vector
                                                    -> top-3 nearest train+val
                                                       gene summaries by
                                                       cosine similarity in
                                                       GenePT 1536-d text space.

Both probes are trained on the train+val split of the same frozen 70/15/15
family-stratified split (seed 42). The hyperparameters match the matrix-best
values for each cell.

Sample: 4 test genes, picked by NCBI summary length as a proxy for how
well-characterised the gene is in the literature. Summaries are NEVER a model
input -- they are only used (a) to choose which test genes to demo on, and
(b) to display retrieved neighbours alongside their actual NCBI text.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
OUT = Path(__file__).resolve().parent / "output.md"

NT_V2_DATASET = DATA / "dataset_nt_v2_meanD.parquet"
DNABERT2_DATASET = DATA / "dataset_dnabert2_meanG.parquet"
GENEPT_DATASET = DATA / "dataset.parquet"
SPLITS = DATA / "splits.json"

LOGISTIC_C = 1.0
RIDGE_ALPHA = 10.0
TOP_K = 3
SUMMARY_EXCERPT_CHARS = 220


def _stack(col):
    return np.stack(col.values).astype(np.float32)


def _normalise(M: np.ndarray) -> np.ndarray:
    norms = np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-12, None)
    return M / norms


def main():
    print("  loading datasets...")
    df_family = pd.read_parquet(NT_V2_DATASET)[["ensembl_id", "x"]].rename(columns={"x": "x_family"})
    df_text = pd.read_parquet(DNABERT2_DATASET)[["ensembl_id", "x"]].rename(columns={"x": "x_text"})
    df_meta = pd.read_parquet(GENEPT_DATASET)[
        ["ensembl_id", "symbol", "family", "summary", "y"]
    ]

    df = (
        df_meta.merge(df_family, on="ensembl_id", how="inner")
              .merge(df_text, on="ensembl_id", how="inner")
              .reset_index(drop=True)
    )
    print(f"    merged rows: {len(df)}")

    splits = json.loads(SPLITS.read_text())
    train_val_ids = set(splits["train"]) | set(splits["val"])
    test_ids = set(splits["test"])

    train_val = df[df["ensembl_id"].isin(train_val_ids)].reset_index(drop=True)
    test = df[df["ensembl_id"].isin(test_ids)].reset_index(drop=True)
    print(f"    train+val: {len(train_val)}    test: {len(test)}")

    X_family_tv = _stack(train_val["x_family"])
    y_family_tv = train_val["family"].values

    X_text_tv = _stack(train_val["x_text"])
    Y_genept_tv = _stack(train_val["y"])

    print(f"\n  fitting logistic probe (C={LOGISTIC_C})...")
    family_probe = LogisticRegression(C=LOGISTIC_C, max_iter=2000, solver="lbfgs")
    family_probe.fit(X_family_tv, y_family_tv)

    print(f"  fitting Ridge probe (alpha={RIDGE_ALPHA})...")
    text_probe = Ridge(alpha=RIDGE_ALPHA)
    text_probe.fit(X_text_tv, Y_genept_tv)

    test_with_len = test.assign(summary_len=test["summary"].fillna("").str.len())
    nonempty = test_with_len[test_with_len["summary_len"] >= 30].copy()
    well = nonempty.nlargest(2, "summary_len")
    poor = nonempty.nsmallest(2, "summary_len")
    sample = pd.concat([
        well.assign(cohort="well-characterised"),
        poor.assign(cohort="poorly characterised"),
    ]).reset_index(drop=True)
    print(f"\n  picked {len(sample)} demo genes "
          "(2 well-characterised + 2 poorly characterised, by summary length)")

    Y_genept_tv_norm = _normalise(Y_genept_tv)

    rows: list[dict] = []
    for _, gene in sample.iterrows():
        x_family = np.asarray(gene["x_family"], dtype=np.float32)[None, :]
        x_text = np.asarray(gene["x_text"], dtype=np.float32)[None, :]

        probs = family_probe.predict_proba(x_family)[0]
        pred_family = family_probe.classes_[int(probs.argmax())]
        prob_dict = {cls: float(p) for cls, p in zip(family_probe.classes_, probs)}

        y_pred = text_probe.predict(x_text)[0]
        y_pred_norm = y_pred / max(float(np.linalg.norm(y_pred)), 1e-12)

        cos = Y_genept_tv_norm @ y_pred_norm
        top_idx = np.argsort(-cos)[:TOP_K]
        retrieved = train_val.iloc[top_idx].reset_index(drop=True)
        retrieved_cos = cos[top_idx]

        rows.append({
            "gene": gene,
            "pred_family": pred_family,
            "true_family": gene["family"],
            "probs": prob_dict,
            "retrieved": retrieved,
            "retrieved_cos": retrieved_cos,
        })

    print(f"\n  writing demo output -> {OUT}")
    with OUT.open("w") as f:
        f.write("# Cross-modal zero-shot demo\n\n")
        f.write(
            "Two probes on the **same DNA input**, both trained on train+val "
            "of the frozen 70/15/15 family-stratified split (seed 42):\n\n"
            "1. **NT-v2 + meanD → logistic probe** (C=1.0) → predicted family.\n"
            "2. **DNABERT-2 + meanG → Ridge probe** (α=10.0) → predicted "
            "GenePT 1536-d vector → top-3 nearest train+val gene summaries by "
            "cosine similarity in the GenePT text-embedding space.\n\n"
            "Sample: 4 test genes — 2 with the longest NCBI summaries "
            "(well-characterised, calibration) and 2 with the shortest "
            "non-trivial summaries (poorly characterised, novelty test). "
            "Summaries are *never* a model input — only used to pick which "
            "test genes to demo and to display retrieved neighbours.\n\n"
        )

        for r in rows:
            g = r["gene"]
            mark = "✅" if r["pred_family"] == r["true_family"] else "❌"
            f.write(f"## {g['symbol']}  ({g['ensembl_id']})\n\n")
            f.write(f"- **Cohort:** {g['cohort']}\n")
            f.write(f"- **True family:** `{r['true_family']}`\n")
            f.write(f"- **Predicted family:** `{r['pred_family']}` {mark}\n")
            probs_str = " · ".join(
                f"{cls}={p:.3f}" for cls, p in
                sorted(r["probs"].items(), key=lambda kv: -kv[1])
            )
            f.write(f"- **Family probabilities:** {probs_str}\n")
            actual = (g["summary"] or "").strip()
            actual_excerpt = actual[:SUMMARY_EXCERPT_CHARS] + (
                "…" if len(actual) > SUMMARY_EXCERPT_CHARS else ""
            )
            f.write(f"- **Actual NCBI summary ({g['summary_len']} chars):** "
                    f"{actual_excerpt}\n\n")
            f.write(
                f"**Top-{TOP_K} retrieved gene summaries** "
                "(nearest train+val genes by cosine similarity in predicted "
                "GenePT 1536-d text-embedding space):\n\n"
            )
            for i, (_, n) in enumerate(r["retrieved"].iterrows(), start=1):
                summary = (n["summary"] or "").strip()
                excerpt = summary[:SUMMARY_EXCERPT_CHARS] + (
                    "…" if len(summary) > SUMMARY_EXCERPT_CHARS else ""
                )
                f.write(
                    f"{i}. `{n['symbol']}` (family `{n['family']}`, "
                    f"cos={r['retrieved_cos'][i-1]:.4f})\n"
                    f"   > {excerpt}\n\n"
                )
            n_majority = (r["retrieved"]["family"] == r["pred_family"]).sum()
            f.write(
                f"Retrieved-summary family agreement with predicted family: "
                f"**{n_majority}/{TOP_K}**\n\n---\n\n"
            )

    n_correct = sum(1 for r in rows if r["pred_family"] == r["true_family"])
    print(f"  family predictions correct: {n_correct}/{len(rows)}")
    print(f"  done -> {OUT}")


if __name__ == "__main__":
    main()
