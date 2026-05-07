"""Build the Ridge-to-GenePT regression results table.

Reads the latest regression metrics from ``data/metrics.json`` and writes a
compact paper-facing table to ``data/regression_table.md``.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"

ROWS = [
    "kmer",
    "dnabert2",
    "dnabert2_meanmean",
    "dnabert2_meanD",
    "dnabert2_meanG",
    "dnabert2_maxmean",
    "dnabert2_clsmean",
    "nt_v2",
    "nt_v2_meanmean",
    "nt_v2_meanD",
    "nt_v2_meanG",
    "nt_v2_maxmean",
    "nt_v2_clsmean",
    "gena_lm",
    "gena_lm_meanmean",
    "gena_lm_meanD",
    "gena_lm_meanG",
    "gena_lm_maxmean",
    "gena_lm_clsmean",
    "hyena_dna",
    "hyena_dna_meanmean",
    "hyena_dna_meanD",
    "hyena_dna_meanG",
    "hyena_dna_maxmean",
    "hyena_dna_clsmean",
]

TSS_ROWS = [
    "tss_nt_v2",
    "tss_nt_v2_meanmean",
    "tss_nt_v2_meanD",
    "tss_nt_v2_meanG",
    "tss_nt_v2_maxmean",
    "tss_nt_v2_clsmean",
]

ENFORMER_ROWS = [
    "enformer_tss_4mer",
    "enformer_trunk_global",
    "enformer_trunk_center",
    "enformer_tracks_center",
]


def _dataset_name(label: str) -> str | None:
    if label == "kmer":
        return None
    if label == "dnabert2":
        return "dataset.parquet"
    if label == "nt_v2":
        return "dataset_nt_v2.parquet"
    return f"dataset_{label}.parquet"


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


def _fmt(value, digits: int = 4) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _delta(value, baseline) -> str:
    if value is None or baseline is None:
        return "-"
    return f"{float(value) - float(baseline):+.4f}"


def _row(label: str, by_regression: dict[str, dict], baseline_r2: float | None) -> str:
    dataset = _dataset_name(label)
    run = by_regression.get("kmer") if label == "kmer" else by_regression.get(dataset or "")
    r2 = run.get("test_r2_macro") if run else None
    return (
        f"| `{label}` "
        f"| {_fmt(r2)} "
        f"| {_delta(r2, baseline_r2)} "
        f"| {_fmt(run.get('test_mean_cosine') if run else None)} "
        f"| {_fmt(run.get('test_median_cosine') if run else None)} "
        f"| {_fmt(run.get('alpha') if run else None, digits=3)} |\n"
    )


def main() -> None:
    metrics = json.loads((DATA / "metrics.json").read_text())
    by_regression = _latest_regression(metrics)
    baseline = by_regression.get("kmer", {}).get("test_r2_macro")
    out = DATA / "regression_table.md"

    with out.open("w") as f:
        f.write("# Ridge-To-GenePT Regression Table\n\n")
        f.write(
            "Secondary cross-modal probe: frozen sequence features are mapped "
            "to 1536-d GenePT summary embeddings with Ridge regression. "
            "`Delta vs 4-mer` is computed against the CDS 4-mer baseline.\n\n"
        )
        f.write("| Feature source | Ridge R2 | Delta vs 4-mer | Mean cosine | Median cosine | Alpha |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for label in ROWS:
            f.write(_row(label, by_regression, baseline))

        f.write("\n## TSS Self-Supervised Encoder Ablation\n\n")
        f.write("| Feature source | Ridge R2 | Delta vs 4-mer | Mean cosine | Median cosine | Alpha |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for label in TSS_ROWS:
            f.write(_row(label, by_regression, baseline))

        f.write("\n## Enformer Supervised Sequence-To-Function Comparator\n\n")
        f.write("| Feature source | Ridge R2 | Delta vs 4-mer | Mean cosine | Median cosine | Alpha |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for label in ENFORMER_ROWS:
            f.write(_row(label, by_regression, baseline))

        scored = [
            (label, by_regression.get("kmer") if label == "kmer" else by_regression.get(_dataset_name(label) or ""))
            for label in ROWS + TSS_ROWS + ENFORMER_ROWS
        ]
        scored = [(label, run) for label, run in scored if run and run.get("test_r2_macro") is not None]
        if scored:
            best_label, best_run = max(scored, key=lambda item: item[1]["test_r2_macro"])
            f.write("\n## Best Observed Regression Cell\n\n")
            f.write("| Feature source | Ridge R2 | Mean cosine |\n")
            f.write("|---|---:|---:|\n")
            f.write(
                f"| `{best_label}` | {_fmt(best_run['test_r2_macro'])} "
                f"| {_fmt(best_run.get('test_mean_cosine'))} |\n"
            )

    print(f"wrote {out}")


if __name__ == "__main__":
    main()
