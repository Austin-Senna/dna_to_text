"""Build the v1 paper's focused family5 results table.

The older `build_full_table.py` keeps the legacy binary tasks. This script is
for the revised paper story: 5-way classification plus Ridge-to-GenePT R2.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"

MAIN_ROWS = [
    "kmer",
    "dnabert2_meanmean",
    "dnabert2_meanD",
    "dnabert2_meanG",
    "nt_v2_meanmean",
    "nt_v2_meanD",
    "nt_v2_meanG",
    "gena_lm",
    "gena_lm_meanmean",
    "gena_lm_meanD",
    "gena_lm_meanG",
    "hyena_dna",
    "hyena_dna_meanmean",
    "hyena_dna_meanD",
    "hyena_dna_meanG",
]
TSS_ROWS = [
    "tss_nt_v2",
    "tss_nt_v2_meanmean",
    "tss_nt_v2_meanD",
    "tss_nt_v2_meanG",
]
ENFORMER_ROWS = [
    "enformer_tss_4mer",
    "enformer_trunk_global",
    "enformer_trunk_center",
    "enformer_tracks_center",
]


def _latest_metrics(metrics: list[dict]) -> dict[tuple[str, str], dict]:
    latest: dict[tuple[str, str], dict] = {}
    for run in metrics:
        if run.get("model") != "logistic_probe" or run.get("task") != "family5":
            continue
        key = (run.get("encoder") or run.get("feature_source"), "family5")
        latest[key] = run
    return latest


def _latest_regression(metrics: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for run in metrics:
        if run.get("model") == "kmer_baseline_4":
            latest["kmer"] = run
            continue
        if run.get("model") != "linear_probe":
            continue
        dataset = run.get("dataset") or "dataset.parquet"
        latest[dataset] = run
    return latest


def _regression_dataset(label: str) -> str | None:
    if label == "kmer":
        return None
    if label == "dnabert2":
        return "dataset.parquet"
    if label == "nt_v2":
        return "dataset_nt_v2.parquet"
    return f"dataset_{label}.parquet"


def _kappa_from_confusion(label: str) -> float | None:
    path = DATA / f"confusion_5way_{label}.json"
    if not path.exists():
        return None
    cm = np.asarray(json.loads(path.read_text())["matrix"], dtype=np.float64)
    total = cm.sum()
    if total == 0:
        return None
    observed = np.trace(cm) / total
    expected = (cm.sum(axis=0) * cm.sum(axis=1)).sum() / (total * total)
    if expected == 1.0:
        return None
    return float((observed - expected) / (1.0 - expected))


def _fmt(value, digits=4):
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _row(label: str, by_metric: dict, by_regression: dict) -> str:
    metric = by_metric.get((label, "family5"))
    kappa = metric.get("test_kappa") if metric else None
    if kappa is None:
        kappa = _kappa_from_confusion(label)
    reg_dataset = _regression_dataset(label)
    reg = by_regression.get("kmer") if label == "kmer" else (
        by_regression.get(reg_dataset) if reg_dataset else None
    )
    return (
        f"| `{label}` "
        f"| {_fmt(metric.get('test_macro_f1') if metric else None)} "
        f"| {_fmt(kappa)} "
        f"| {_fmt(metric.get('test_accuracy') if metric else None)} "
        f"| {_fmt(reg.get('test_r2_macro') if reg else None)} |\n"
    )


def main():
    metrics = json.loads((DATA / "metrics.json").read_text())
    by_metric = _latest_metrics(metrics)
    by_regression = _latest_regression(metrics)
    out = DATA / "family5_table.md"

    with out.open("w") as f:
        f.write("# Family5 Model Expansion Table\n\n")
        f.write("Main DNA encoder comparison. Binary tasks are legacy/appendix only.\n\n")
        f.write("| Feature source | 5-way F1 | 5-way kappa | 5-way acc | Ridge R2 |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for label in MAIN_ROWS:
            f.write(_row(label, by_metric, by_regression))

        f.write("\n## TSS Self-Supervised Encoder Ablation\n\n")
        f.write("| Feature source | 5-way F1 | 5-way kappa | 5-way acc | Ridge R2 |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for label in TSS_ROWS:
            f.write(_row(label, by_metric, by_regression))

        f.write("\n## Enformer Supervised Sequence-To-Function Comparator\n\n")
        f.write("| Feature source | 5-way F1 | 5-way kappa | 5-way acc | Ridge R2 |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for label in ENFORMER_ROWS:
            f.write(_row(label, by_metric, by_regression))

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
