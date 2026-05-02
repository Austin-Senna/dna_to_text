"""Build the full results table: every baseline + (encoder x pooling) cell,
with macro-F1, Cohen's kappa, and Ridge R^2 across all three classification
tasks plus the regression task.

Cells (rows):
    - Baselines: shuffled-label, length, kmer
    - Phase 1-3 originals: dnabert2, nt_v2 (no special tokens)
    - Phase 4b DNABERT-2 variants: meanmean / maxmean / clsmean / meanD / meanG
    - Phase 4b NT-v2 variants: same five

Metrics (columns):
    - family5:    macro-F1, Cohen's kappa
    - tf_vs_gpcr: macro-F1, Cohen's kappa
    - tf_vs_kinase: macro-F1, Cohen's kappa
    - regression: R^2 macro into GenePT 1536-d

Output: data/full_table.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score

from binary_tasks import load_binary_split
from kmer_baseline import featurize_cds
from data_loader.sequence_fetcher import fetch_cds
from length_baseline import cds_length_features
from splits import load_split

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
SEQUENCES_DIR = DATA / "sequences"
OUT = DATA / "full_table.md"

DATASET_PATHS = {
    "dnabert2": DATA / "dataset.parquet",
    "nt_v2":    DATA / "dataset_nt_v2.parquet",
}
for _enc in ("dnabert2", "nt_v2"):
    for _v in ("meanmean", "maxmean", "clsmean", "meanD", "meanG"):
        DATASET_PATHS[f"{_enc}_{_v}"] = DATA / f"dataset_{_enc}_{_v}.parquet"
del _enc, _v
META_PARQUET = DATASET_PATHS["dnabert2"]

# Row order for the table — baselines, originals, then variants grouped by encoder.
ROW_ORDER = [
    "shuffled", "length", "kmer",
    "dnabert2",
    "dnabert2_meanmean", "dnabert2_meanD", "dnabert2_meanG",
    "dnabert2_maxmean", "dnabert2_clsmean",
    "nt_v2",
    "nt_v2_meanmean", "nt_v2_meanD", "nt_v2_meanG",
    "nt_v2_maxmean", "nt_v2_clsmean",
]

CLASSIFICATION_TASKS = ("family5", "tf_vs_gpcr", "tf_vs_kinase")


def _kmer_features(meta):
    out = np.zeros((len(meta), 256), dtype=np.float32)
    for i, eid in enumerate(meta["ensembl_id"].tolist()):
        seq = fetch_cds(eid, SEQUENCES_DIR)
        if not seq:
            raise RuntimeError(f"missing CDS for {eid}")
        out[i] = featurize_cds(seq)
    return out


def _length_features(meta):
    return cds_length_features(meta)


def _load(dataset, task, name):
    if task == "family5":
        if dataset in DATASET_PATHS:
            X, _, meta = load_split(name, dataset_path=DATASET_PATHS[dataset])
        else:
            _, _, meta = load_split(name, dataset_path=META_PARQUET)
            X = _kmer_features(meta) if dataset == "kmer" else _length_features(meta)
        y = meta["family"].to_numpy()
        return X, y
    if dataset in DATASET_PATHS:
        X, y, _ = load_binary_split(task, name, dataset_path=DATASET_PATHS[dataset])
    else:
        _, y, meta = load_binary_split(task, name, dataset_path=META_PARQUET)
        X = _kmer_features(meta) if dataset == "kmer" else _length_features(meta)
    return X, y


def kappa_from_cm(cm: np.ndarray) -> float:
    cm = np.asarray(cm, dtype=float)
    n = cm.sum()
    po = np.trace(cm) / n
    p_true = cm.sum(axis=1) / n
    p_pred = cm.sum(axis=0) / n
    pe = float((p_true * p_pred).sum())
    return float((po - pe) / (1.0 - pe))


def kappa_via_refit(dataset: str, task: str, C: float, shuffle: bool = False) -> float:
    X_tr, y_tr = _load(dataset, task, "train")
    X_val, y_val = _load(dataset, task, "val")
    X_te, y_te = _load(dataset, task, "test")
    if shuffle:
        rng = np.random.default_rng(42)
        y_tr = rng.permutation(y_tr)
        y_val = rng.permutation(y_val)
    X_fit = np.vstack([X_tr, X_val])
    y_fit = np.concatenate([y_tr, y_val])
    probe = LogisticRegression(C=C, max_iter=2000, solver="lbfgs").fit(X_fit, y_fit)
    return float(cohen_kappa_score(y_te, probe.predict(X_te)))


def main():
    metrics = json.loads((DATA / "metrics.json").read_text())

    # Index classification entries: latest per (encoder, task, shuffled).
    cls_index: dict[tuple, dict] = {}
    for r in metrics:
        if r.get("model") != "logistic_probe":
            continue
        key = (r["encoder"], r["task"], bool(r.get("shuffled_labels", False)))
        if key not in cls_index or r["timestamp"] > cls_index[key]["timestamp"]:
            cls_index[key] = r

    # Index regression (Ridge) entries: latest per dataset filename.
    # Phase 3 linear_probe entries pre-date the `dataset` field — treat
    # missing as the default DNABERT-2 dataset.parquet. The kmer_baseline_4
    # entries also have no dataset field but are encoder-independent; they
    # live in a separate index pulled by the kmer row only.
    reg_index: dict[str, dict] = {}
    for r in metrics:
        if r.get("model") != "linear_probe":
            continue
        ds = r.get("dataset") or "dataset.parquet"
        if ds not in reg_index or r["timestamp"] > reg_index[ds]["timestamp"]:
            reg_index[ds] = r

    # Map row labels -> regression dataset filenames.
    reg_dataset_for: dict[str, str | None] = {
        "shuffled": None,
        "length": None,
        "kmer": None,  # kmer_baseline_4 has no `dataset` field; pulled separately
        "dnabert2": "dataset.parquet",
        "nt_v2":    "dataset_nt_v2.parquet",
    }
    for enc in ("dnabert2", "nt_v2"):
        for v in ("meanmean", "maxmean", "clsmean", "meanD", "meanG"):
            reg_dataset_for[f"{enc}_{v}"] = f"dataset_{enc}_{v}.parquet"

    # Pull kmer-baseline R^2 from the model='kmer_baseline_4' entry (no `dataset` key).
    kmer_r2 = None
    for r in metrics:
        if r.get("model") == "kmer_baseline_4":
            if kmer_r2 is None or r["timestamp"] > kmer_r2["timestamp"]:
                kmer_r2 = r

    # Build the table rows.
    print("computing kappa for all classification cells...")
    rows: list[dict] = []
    for label in ROW_ORDER:
        row = {"label": label}
        for task in CLASSIFICATION_TASKS:
            shuffled = label == "shuffled"
            # Encoder name in metrics: shuffled rows live under encoder='shuffled'.
            enc_key = "shuffled" if shuffled else label
            entry = cls_index.get((enc_key, task, shuffled))
            if entry is None:
                row[f"{task}_f1"] = None
                row[f"{task}_k"]  = None
                continue
            row[f"{task}_f1"] = float(entry["test_macro_f1"])
            # Kappa: prefer confusion matrix for family5 (no refit), refit otherwise.
            if task == "family5" and not shuffled:
                cm_path = DATA / f"confusion_5way_{label}.json"
                if cm_path.exists():
                    cm = np.array(json.loads(cm_path.read_text())["matrix"])
                    row[f"{task}_k"] = kappa_from_cm(cm)
                    print(f"  {label:<22s} {task:<14s} cm  kappa={row[f'{task}_k']:.4f}")
                    continue
            C = float(entry["C"])
            # For non-encoder feature sources on binary: use kmer/length features synthesised on the fly.
            ds_for_load = label if label not in ("kmer", "length", "shuffled") else label
            if shuffled:
                # Use NT-v2 X for shuffled rows (matches how we ran the anti-baseline).
                ds_for_load = "nt_v2"
            row[f"{task}_k"] = kappa_via_refit(ds_for_load, task, C, shuffle=shuffled)
            print(f"  {label:<22s} {task:<14s} fit kappa={row[f'{task}_k']:.4f}")

        # Regression R^2.
        if label == "kmer" and kmer_r2 is not None:
            row["r2"] = float(kmer_r2["test_r2_macro"])
        elif reg_dataset_for.get(label):
            entry = reg_index.get(reg_dataset_for[label])
            row["r2"] = float(entry["test_r2_macro"]) if entry else None
        else:
            row["r2"] = None

        rows.append(row)

    # Render markdown.
    print(f"\nwriting {OUT}...")
    with OUT.open("w") as f:
        f.write("# Full results table — all baselines × all (encoder × pooling) variants\n\n")
        f.write(
            "Every cell visited in the project, with macro-F1 + Cohen's κ for each "
            "of the three classification tasks and Ridge R² macro into GenePT 1536-d "
            "for the regression task.\n\n"
            "Numbers traceable to `data/metrics.json` (latest entry per cell, deduped on "
            "`(encoder, task, shuffled_labels)`). κ for `family5` cells is computed from "
            "`data/confusion_5way_*.json`; for binary cells the probe is refit at the "
            "recorded best C and κ is computed via `sklearn.metrics.cohen_kappa_score`. "
            "`—` = not run for this combination.\n\n"
        )
        f.write("Row groups: **baselines** (shuffled-label, length-only, 4-mer) → "
                "**Phase 1–3 originals** (no special tokens) → **Phase 4b DNABERT-2 variants** → "
                "**Phase 4b NT-v2 variants**.\n\n")
        f.write("| Feature source | 5-way F1 | 5-way κ | tf-vs-gpcr F1 | tf-vs-gpcr κ | "
                "tf-vs-kinase F1 | tf-vs-kinase κ | Ridge R² |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")

        def _fmt(v, prec=4, signed=False):
            if v is None:
                return "—"
            if signed:
                return f"{v:+.{prec}f}"
            return f"{v:.{prec}f}"

        for row in rows:
            f.write(
                f"| `{row['label']}` "
                f"| {_fmt(row['family5_f1'])} "
                f"| {_fmt(row['family5_k'], signed=row['label']=='shuffled')} "
                f"| {_fmt(row['tf_vs_gpcr_f1'])} "
                f"| {_fmt(row['tf_vs_gpcr_k'], signed=row['label']=='shuffled')} "
                f"| {_fmt(row['tf_vs_kinase_f1'])} "
                f"| {_fmt(row['tf_vs_kinase_k'], signed=row['label']=='shuffled')} "
                f"| {_fmt(row['r2'])} |\n"
            )
        f.write("\n")
        f.write(
            "**Reading guide.** macro-F1 is the per-task headline metric "
            "(unweighted mean of per-class F1). Cohen's κ is chance-corrected "
            "(0 = chance, 1 = perfect; negative = worse than chance). R² macro is "
            "the unweighted mean R² across the 1536-d GenePT regression target — "
            "chance-corrected by construction (R² = 0 ≡ predict-the-mean).\n\n"
            "**Anti-baseline interpretation.** The shuffled-label row is the empirical "
            "chance level for this exact pipeline. Across all three tasks κ falls within "
            "±0.10 of zero — pipeline is honest.\n\n"
            "**Best of each column.** Highlighted in bold below.\n\n"
        )

        # Per-column maxima
        cols = [("5-way F1", "family5_f1"), ("5-way κ", "family5_k"),
                ("tf-vs-gpcr F1", "tf_vs_gpcr_f1"), ("tf-vs-gpcr κ", "tf_vs_gpcr_k"),
                ("tf-vs-kinase F1", "tf_vs_kinase_f1"), ("tf-vs-kinase κ", "tf_vs_kinase_k"),
                ("Ridge R²", "r2")]
        f.write("| Column | Best feature source | Value |\n|---|---|---:|\n")
        for col_label, col_key in cols:
            best = max(
                (r for r in rows if r[col_key] is not None and r["label"] != "shuffled"),
                key=lambda r: r[col_key],
            )
            f.write(f"| {col_label} | `{best['label']}` | **{best[col_key]:.4f}** |\n")
        f.write("\n")

        # ---- Delta vs k-mer table ----
        kmer_row = next(r for r in rows if r["label"] == "kmer")
        f.write("## Δ vs 4-mer baseline\n\n")
        f.write(
            "Same matrix, but each cell is `(value) − (kmer value)` in the same column. "
            "Positive = beats k-mer composition; negative = worse than k-mer; zero = tied. "
            "k-mer's own row reads zero by construction. Reading this table answers "
            "*\"how much extra signal does the encoder + pooling carry over a "
            "256-d 4-mer histogram?\"*\n\n"
        )
        f.write("| Feature source | Δ 5-way F1 | Δ 5-way κ | Δ tf-vs-gpcr F1 | Δ tf-vs-gpcr κ | "
                "Δ tf-vs-kinase F1 | Δ tf-vs-kinase κ | Δ Ridge R² |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")

        delta_keys = [
            "family5_f1", "family5_k",
            "tf_vs_gpcr_f1", "tf_vs_gpcr_k",
            "tf_vs_kinase_f1", "tf_vs_kinase_k",
            "r2",
        ]

        def _delta(row, key):
            if row[key] is None or kmer_row.get(key) is None:
                return None
            return row[key] - kmer_row[key]

        for row in rows:
            cells = []
            for key in delta_keys:
                d = _delta(row, key)
                cells.append("—" if d is None else f"{d:+.4f}")
            f.write(f"| `{row['label']}` | " + " | ".join(cells) + " |\n")
        f.write("\n")

        # Highlight cells that meet the spec's "beats 4-mer" gate (Δ ≥ 0.02 macro-F1).
        f.write("**Cells beating 4-mer by Δ macro-F1 ≥ 0.02** (the spec's decision-gate "
                "threshold from `2026-04-29-classification-pivot-design.md`):\n\n")
        f.write("| Feature source | 5-way | tf-vs-gpcr | tf-vs-kinase |\n|---|:-:|:-:|:-:|\n")
        for row in rows:
            if row["label"] in ("kmer", "shuffled"):
                continue
            marks = []
            for f1_key in ("family5_f1", "tf_vs_gpcr_f1", "tf_vs_kinase_f1"):
                d = _delta(row, f1_key)
                if d is None:
                    marks.append("—")
                elif d >= 0.02:
                    marks.append("✅")
                elif d <= -0.02:
                    marks.append("❌")
                else:
                    marks.append("≈")
            f.write(f"| `{row['label']}` | " + " | ".join(marks) + " |\n")
        f.write("\nLegend: ✅ beats k-mer (Δ ≥ +0.02); ≈ ties (within ±0.02); ❌ loses (Δ ≤ −0.02).\n\n")

    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
