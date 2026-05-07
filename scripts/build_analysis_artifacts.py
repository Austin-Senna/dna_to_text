"""Build cached paper analysis tables and figures.

This is a reproducibility wrapper over already-computed artifacts. It reads
metrics, confusion matrices, split metadata, and dataset parquets; it never
runs encoders or probes.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_loader.model_registry import ENCODER_SPECS
from data_loader.pooling_aggregator import POOLING_VARIANTS

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"

PAPER_ENCODERS = ("dnabert2", "nt_v2", "gena_lm", "hyena_dna")
POOLING_ORDER = ("base", "meanmean", "meanD", "meanG", "maxmean", "clsmean")
LEGACY_TASKS = ("tf_vs_gpcr", "tf_vs_kinase")
ENFORMER_SOURCES = (
    "enformer_tss_4mer",
    "enformer_trunk_global",
    "enformer_trunk_center",
    "enformer_tracks_center",
)
CONTEXT_ABLATION_SPECS = (
    {
        "label": "CDS 4-mer",
        "context": "CDS",
        "model_group": "composition",
        "family5_feature_source": "kmer",
        "regression_dataset": "kmer",
        "regression_feature_source": "kmer",
    },
    {
        "label": "CDS NT-v2 meanD",
        "context": "CDS",
        "model_group": "self-supervised encoder",
        "family5_feature_source": "nt_v2_meanD",
        "regression_dataset": "dataset_nt_v2_meanD.parquet",
        "regression_feature_source": "nt_v2_meanD",
    },
    {
        "label": "TSS 4-mer",
        "context": "TSS window",
        "model_group": "composition",
        "family5_feature_source": "enformer_tss_4mer",
        "regression_dataset": "dataset_enformer_tss_4mer.parquet",
        "regression_feature_source": "enformer_tss_4mer",
    },
    {
        "label": "TSS NT-v2 meanmean",
        "context": "TSS window",
        "model_group": "self-supervised encoder",
        "family5_feature_source": "tss_nt_v2_meanmean",
        "regression_dataset": "dataset_tss_nt_v2_meanmean.parquet",
        "regression_feature_source": "tss_nt_v2_meanmean",
    },
    {
        "label": "Enformer trunk",
        "context": "TSS window",
        "model_group": "supervised comparator",
        "family5_feature_source": "enformer_trunk_global",
        "regression_dataset": "dataset_enformer_trunk_center.parquet",
        "regression_feature_source": "enformer_trunk_center",
    },
)
DATASET_CANDIDATES = (
    DATA / "dataset_nt_v2_meanD.parquet",
    DATA / "dataset_hyena_dna_meanG.parquet",
    DATA / "dataset_gena_lm.parquet",
    DATA / "dataset.parquet",
)
FAMILY_ORDER = ("tf", "gpcr", "kinase", "ion", "immune")
FAMILY_LABELS = {
    "tf": "TF",
    "gpcr": "GPCR",
    "kinase": "Kinase",
    "ion": "Ion channel",
    "immune": "Immune receptor",
}
_KAPPA_SUMMARY_CACHE: dict[tuple[str, str], float] | None = None


def _timestamp(run: dict) -> str:
    return str(run.get("timestamp") or "")


def _keep_latest(index: dict, key, run: dict) -> None:
    if key not in index or _timestamp(run) >= _timestamp(index[key]):
        index[key] = run


def latest_logistic_runs(metrics: list[dict]) -> dict[tuple[str, str, bool], dict]:
    latest: dict[tuple[str, str, bool], dict] = {}
    for run in metrics:
        if run.get("model") != "logistic_probe":
            continue
        encoder = str(run.get("encoder") or run.get("feature_source") or "")
        task = str(run.get("task") or "")
        shuffled = bool(run.get("shuffled_labels", False))
        _keep_latest(latest, (encoder, task, shuffled), run)
    return latest


def latest_regression_runs(metrics: list[dict]) -> dict[str, dict]:
    latest: dict[str, dict] = {}
    for run in metrics:
        if run.get("model") == "kmer_baseline_4":
            _keep_latest(latest, "kmer", run)
            continue
        if run.get("model") != "linear_probe":
            continue
        dataset = str(run.get("dataset") or "dataset.parquet")
        _keep_latest(latest, dataset, run)
    return latest


def encoder_for_feature(feature_source: str, encoder_names: Iterable[str] = PAPER_ENCODERS) -> str | None:
    for encoder in sorted(encoder_names, key=len, reverse=True):
        if feature_source == encoder or feature_source.startswith(f"{encoder}_"):
            return encoder
    return None


def pooling_for_feature(feature_source: str, encoder_names: Iterable[str] = PAPER_ENCODERS) -> str | None:
    encoder = encoder_for_feature(feature_source, encoder_names)
    if encoder is None:
        return None
    if feature_source == encoder:
        return "base"
    return feature_source.removeprefix(f"{encoder}_")


def dataset_for_feature(feature_source: str) -> str | None:
    if feature_source == "kmer":
        return None
    if feature_source == "dnabert2":
        return "dataset.parquet"
    if feature_source == "nt_v2":
        return "dataset_nt_v2.parquet"
    if feature_source in ("shuffled", "length"):
        return None
    return f"dataset_{feature_source}.parquet"


def feature_from_dataset(dataset: str) -> str | None:
    if dataset == "dataset.parquet":
        return "dnabert2"
    if dataset == "dataset_nt_v2.parquet":
        return "nt_v2"
    if dataset.startswith("dataset_") and dataset.endswith(".parquet"):
        return dataset.removeprefix("dataset_").removesuffix(".parquet")
    return None


def _kappa_from_confusion(feature_source: str) -> float | None:
    path = DATA / f"confusion_5way_{feature_source}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    matrix = np.asarray(payload["matrix"], dtype=np.float64)
    total = matrix.sum()
    if total <= 0:
        return None
    observed = np.trace(matrix) / total
    expected = (matrix.sum(axis=0) * matrix.sum(axis=1)).sum() / (total * total)
    if expected == 1.0:
        return None
    return float((observed - expected) / (1.0 - expected))


def _clean_markdown_cell(value: str) -> str:
    value = value.strip()
    value = value.replace("`", "").replace("*", "")
    value = re.sub(r"\s*\(.*?\)\s*", "", value).strip()
    return value


def parse_kappa_summary(text: str) -> dict[tuple[str, str], float]:
    values: dict[tuple[str, str], float] = {}
    current_task: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("## "):
            current_task = stripped.removeprefix("## ").strip()
            continue
        if current_task is None or not stripped.startswith("|"):
            continue
        cells = [_clean_markdown_cell(cell) for cell in stripped.strip("|").split("|")]
        if len(cells) < 4 or cells[0] in ("Dataset", "---"):
            continue
        label = cells[0]
        if label == "shuffled-label":
            label = "shuffled"
        try:
            values[(current_task, label)] = float(cells[3].replace("+", ""))
        except ValueError:
            continue
    return values


def _kappa_from_summary(task: str, feature_source: str) -> float | None:
    global _KAPPA_SUMMARY_CACHE
    if _KAPPA_SUMMARY_CACHE is None:
        path = DATA / "kappa_summary.md"
        _KAPPA_SUMMARY_CACHE = parse_kappa_summary(path.read_text()) if path.exists() else {}
    return _KAPPA_SUMMARY_CACHE.get((task, feature_source))


def _metric_value(run: dict | None, key: str) -> float | None:
    if not run or run.get(key) is None:
        return None
    return float(run[key])


def _family5_record(feature_source: str, run: dict, task: str = "family5") -> dict:
    kappa = _metric_value(run, "test_kappa")
    if kappa is None and task == "family5":
        kappa = _kappa_from_confusion(feature_source)
    if kappa is None:
        kappa = _kappa_from_summary(task, feature_source)
    encoder = encoder_for_feature(feature_source)
    return {
        "encoder": encoder or feature_source,
        "pooling": pooling_for_feature(feature_source) or "baseline",
        "feature_source": feature_source,
        "test_macro_f1": _metric_value(run, "test_macro_f1"),
        "test_kappa": kappa,
        "test_accuracy": _metric_value(run, "test_accuracy"),
        "C": _metric_value(run, "C"),
        "timestamp": run.get("timestamp"),
    }


def best_family5_rows(
    latest_logistic: dict[tuple[str, str, bool], dict],
    encoder_names: Iterable[str] = PAPER_ENCODERS,
) -> pd.DataFrame:
    rows: list[dict] = []
    for encoder in encoder_names:
        candidates = []
        for (feature_source, task, shuffled), run in latest_logistic.items():
            if task != "family5" or shuffled:
                continue
            if encoder_for_feature(feature_source, encoder_names) == encoder:
                candidates.append((feature_source, run))
        if not candidates:
            continue
        best_feature, best_run = max(
            candidates,
            key=lambda item: _metric_value(item[1], "test_macro_f1") or float("-inf"),
        )
        rows.append(_family5_record(best_feature, best_run))
    return _ordered_encoder_frame(rows)


def main_family5_table(latest_logistic: dict[tuple[str, str, bool], dict]) -> pd.DataFrame:
    rows = []
    kmer = latest_logistic.get(("kmer", "family5", False))
    if kmer:
        rows.append(_family5_record("kmer", kmer))
    rows.extend(best_family5_rows(latest_logistic).to_dict("records"))
    shuffled = latest_logistic.get(("shuffled", "family5", True))
    if shuffled:
        row = _family5_record("shuffled", shuffled)
        row["pooling"] = "anti_baseline"
        rows.append(row)
    return pd.DataFrame(rows)


def pooling_sweep_family5_table(latest_logistic: dict[tuple[str, str, bool], dict]) -> pd.DataFrame:
    rows = []
    for (feature_source, task, shuffled), run in latest_logistic.items():
        if task != "family5" or shuffled:
            continue
        encoder = encoder_for_feature(feature_source)
        if encoder is None:
            continue
        rows.append(_family5_record(feature_source, run))
    return _ordered_encoder_frame(rows)


def _regression_record(feature_source: str, run: dict, baseline_r2: float | None) -> dict:
    encoder = encoder_for_feature(feature_source)
    r2 = _metric_value(run, "test_r2_macro")
    return {
        "encoder": encoder or feature_source,
        "pooling": pooling_for_feature(feature_source) or "baseline",
        "feature_source": feature_source,
        "test_r2_macro": r2,
        "delta_vs_4mer": None if r2 is None or baseline_r2 is None else r2 - baseline_r2,
        "test_mean_cosine": _metric_value(run, "test_mean_cosine"),
        "test_median_cosine": _metric_value(run, "test_median_cosine"),
        "alpha": _metric_value(run, "alpha"),
        "timestamp": run.get("timestamp"),
    }


def regression_full_table(latest_regression: dict[str, dict]) -> pd.DataFrame:
    rows = []
    baseline_r2 = _metric_value(latest_regression.get("kmer"), "test_r2_macro")
    if "kmer" in latest_regression:
        rows.append(_regression_record("kmer", latest_regression["kmer"], baseline_r2))
    for dataset, run in latest_regression.items():
        if dataset == "kmer":
            continue
        feature = feature_from_dataset(dataset)
        if feature is None:
            continue
        rows.append(_regression_record(feature, run, baseline_r2))
    return _ordered_encoder_frame(rows)


def main_regression_table(latest_regression: dict[str, dict]) -> pd.DataFrame:
    full = regression_full_table(latest_regression)
    rows = []
    kmer = full[full["feature_source"] == "kmer"]
    if not kmer.empty:
        rows.append(kmer.iloc[0].to_dict())
    for encoder in PAPER_ENCODERS:
        candidates = full[full["encoder"] == encoder]
        if candidates.empty:
            continue
        idx = candidates["test_r2_macro"].astype(float).idxmax()
        rows.append(full.loc[idx].to_dict())
    return pd.DataFrame(rows)


def combined_model_summary_table(
    family5: pd.DataFrame,
    regression: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for label in ("kmer", *PAPER_ENCODERS, "shuffled"):
        fam = family5[family5["encoder"] == label]
        reg = regression[regression["encoder"] == label]
        rows.append(
            {
                "encoder": label,
                "family5_feature_source": None if fam.empty else fam.iloc[0]["feature_source"],
                "family5_macro_f1": None if fam.empty else fam.iloc[0]["test_macro_f1"],
                "family5_kappa": None if fam.empty else fam.iloc[0]["test_kappa"],
                "family5_accuracy": None if fam.empty else fam.iloc[0]["test_accuracy"],
                "regression_feature_source": None if reg.empty else reg.iloc[0]["feature_source"],
                "ridge_r2_macro": None if reg.empty else reg.iloc[0]["test_r2_macro"],
                "ridge_delta_vs_4mer": None if reg.empty else reg.iloc[0]["delta_vs_4mer"],
            }
        )
    return pd.DataFrame(rows).dropna(how="all", subset=["family5_feature_source", "regression_feature_source"])


def _regression_run_for_dataset(latest_regression: dict[str, dict], dataset: str) -> dict | None:
    if dataset in latest_regression:
        return latest_regression[dataset]
    data_prefixed = f"data/{dataset}"
    if data_prefixed in latest_regression:
        return latest_regression[data_prefixed]
    return None


def context_ablation_table(
    latest_logistic: dict[tuple[str, str, bool], dict],
    latest_regression: dict[str, dict],
) -> pd.DataFrame:
    rows = []
    for spec in CONTEXT_ABLATION_SPECS:
        family_feature = spec["family5_feature_source"]
        family_run = latest_logistic.get((family_feature, "family5", False))
        regression_run = _regression_run_for_dataset(latest_regression, spec["regression_dataset"])
        rows.append(
            {
                "label": spec["label"],
                "context": spec["context"],
                "model_group": spec["model_group"],
                "family5_feature_source": family_feature,
                "regression_feature_source": spec["regression_feature_source"],
                "family5_macro_f1": _metric_value(family_run, "test_macro_f1"),
                "family5_kappa": None if family_run is None else _family5_record(family_feature, family_run)["test_kappa"],
                "family5_accuracy": _metric_value(family_run, "test_accuracy"),
                "ridge_r2_macro": _metric_value(regression_run, "test_r2_macro"),
            }
        )
    return pd.DataFrame(rows)


def legacy_binary_appendix_table(latest_logistic: dict[tuple[str, str, bool], dict]) -> pd.DataFrame:
    rows = []
    for (feature_source, task, shuffled), run in latest_logistic.items():
        if task not in LEGACY_TASKS:
            continue
        row = _family5_record(feature_source, run, task)
        row["task"] = task
        row["shuffled_labels"] = shuffled
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    task_rank = {task: i for i, task in enumerate(LEGACY_TASKS)}
    df["_task_rank"] = df["task"].map(task_rank)
    df["_encoder_rank"] = df["encoder"].map(_encoder_rank).fillna(999)
    df["_pool_rank"] = df["pooling"].map(_pooling_rank).fillna(999)
    return df.sort_values(["_task_rank", "_encoder_rank", "_pool_rank", "feature_source"]).drop(
        columns=["_task_rank", "_encoder_rank", "_pool_rank"]
    ).reset_index(drop=True)


def split_lookup_from_splits(splits: dict) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for split in ("train", "val", "test"):
        ids = splits.get(split, [])
        if not isinstance(ids, list):
            continue
        for ensembl_id in ids:
            lookup[str(ensembl_id)] = split
    return lookup


def dataset_composition_table() -> pd.DataFrame:
    dataset = next((path for path in DATASET_CANDIDATES if path.exists()), None)
    if dataset is None:
        return pd.DataFrame(columns=["family", "total", "train", "val", "test"])
    df = pd.read_parquet(dataset, columns=["ensembl_id", "family"])
    splits_path = DATA / "splits.json"
    split_lookup: dict[str, str] = {}
    if splits_path.exists():
        split_lookup = split_lookup_from_splits(json.loads(splits_path.read_text()))
    counts = []
    for family in FAMILY_ORDER:
        fam = df[df["family"] == family]
        row = {"family": family, "total": int(len(fam))}
        for split in ("train", "val", "test"):
            row[split] = int(sum(split_lookup.get(str(eid)) == split for eid in fam["ensembl_id"]))
        counts.append(row)
    return pd.DataFrame(counts)


def missing_cells_table(
    latest_logistic: dict[tuple[str, str, bool], dict],
    latest_regression: dict[str, dict],
    registered_feature_sources: Iterable[str],
    enformer_sources: Iterable[str] = ENFORMER_SOURCES,
) -> pd.DataFrame:
    rows = []
    for feature in list(registered_feature_sources) + list(enformer_sources):
        if feature in ("kmer", "shuffled", "length"):
            continue
        if (feature, "family5", False) not in latest_logistic:
            rows.append({"artifact": "family5", "feature_source": feature, "reason": "missing cached logistic metric"})
        dataset = dataset_for_feature(feature)
        if dataset and dataset not in latest_regression:
            rows.append({"artifact": "regression", "feature_source": feature, "reason": "missing cached Ridge metric"})
    return pd.DataFrame(rows, columns=["artifact", "feature_source", "reason"])


def registered_feature_sources() -> list[str]:
    sources = []
    for encoder in PAPER_ENCODERS:
        sources.append(encoder)
        sources.extend(f"{encoder}_{variant}" for variant in POOLING_VARIANTS)
    return sources


def _encoder_rank(encoder: str) -> int:
    order = ("kmer", *PAPER_ENCODERS, "shuffled", "length")
    try:
        return order.index(encoder)
    except ValueError:
        return 999


def _pooling_rank(pooling: str) -> int:
    try:
        return POOLING_ORDER.index(pooling)
    except ValueError:
        return 999


def _ordered_encoder_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["_encoder_rank"] = df["encoder"].map(_encoder_rank).fillna(999)
    df["_pool_rank"] = df["pooling"].map(_pooling_rank).fillna(999)
    return df.sort_values(["_encoder_rank", "_pool_rank", "feature_source"]).drop(
        columns=["_encoder_rank", "_pool_rank"]
    ).reset_index(drop=True)


def _check_write(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists; pass --overwrite to replace it")


def _format_cell(value) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _to_markdown(df: pd.DataFrame) -> str:
    columns = list(df.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_format_cell(row[col]) for col in columns) + " |")
    return "\n".join(lines) + "\n"


def write_table(
    df: pd.DataFrame,
    name: str,
    tables_dir: Path,
    *,
    title: str,
    description: str,
    overwrite: bool,
) -> tuple[Path, Path]:
    tables_dir.mkdir(parents=True, exist_ok=True)
    csv_path = tables_dir / f"{name}.csv"
    md_path = tables_dir / f"{name}.md"
    _check_write(csv_path, overwrite)
    _check_write(md_path, overwrite)
    df.to_csv(csv_path, index=False)
    md_path.write_text(f"# {title}\n\n{description}\n\n{_to_markdown(df)}")
    return csv_path, md_path


def _savefig(fig, path: Path, overwrite: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    _check_write(path, overwrite)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _label_series(df: pd.DataFrame) -> list[str]:
    return [str(v) for v in df["feature_source"].tolist()]


def plot_family5_bar(family5: pd.DataFrame, metric: str, out: Path, overwrite: bool) -> Path | None:
    df = family5.dropna(subset=[metric])
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = _label_series(df)
    values = df[metric].astype(float).to_numpy()
    colours = ["#808080" if label in ("kmer", "shuffled") else "#2f6f9f" for label in labels]
    ax.bar(range(len(labels)), values, color=colours)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, max(1.0, float(values.max()) * 1.12))
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"Family5 {metric.replace('_', ' ')}")
    for i, value in enumerate(values):
        ax.text(i, value + 0.015, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return _savefig(fig, out, overwrite)


def plot_ridge_r2(regression: pd.DataFrame, out: Path, overwrite: bool) -> Path | None:
    df = regression.dropna(subset=["test_r2_macro"])
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = _label_series(df)
    values = df["test_r2_macro"].astype(float).to_numpy()
    colours = ["#808080" if label == "kmer" else "#6b4c9a" for label in labels]
    ax.bar(range(len(labels)), values, color=colours)
    kmer = df[df["feature_source"] == "kmer"]
    if not kmer.empty:
        ax.axhline(float(kmer.iloc[0]["test_r2_macro"]), color="#444444", linestyle="--", linewidth=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Ridge macro R2")
    ax.set_title("Ridge-to-GenePT regression")
    for i, value in enumerate(values):
        ax.text(i, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    return _savefig(fig, out, overwrite)


def plot_tradeoff(combined: pd.DataFrame, out: Path, overwrite: bool) -> Path | None:
    df = combined.dropna(subset=["family5_macro_f1", "ridge_r2_macro"])
    if df.empty:
        return None
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    for _, row in df.iterrows():
        label = row["encoder"]
        colour = "#808080" if label == "kmer" else "#3b7f5c"
        ax.scatter(row["ridge_r2_macro"], row["family5_macro_f1"], s=55, color=colour)
        ax.text(row["ridge_r2_macro"] + 0.002, row["family5_macro_f1"], str(label), fontsize=8, va="center")
    ax.set_xlabel("Ridge macro R2")
    ax.set_ylabel("Family5 macro-F1")
    ax.set_title("Family classification vs cross-modal regression")
    fig.tight_layout()
    return _savefig(fig, out, overwrite)


def plot_context_ablation(context: pd.DataFrame, out: Path, overwrite: bool) -> Path | None:
    required = ["family5_macro_f1", "ridge_r2_macro"]
    df = context.dropna(subset=required)
    if df.empty:
        return None
    colours = {
        "composition": "#7a7a7a",
        "self-supervised encoder": "#2f6f9f",
        "supervised comparator": "#3b7f5c",
    }
    y = np.arange(len(df))
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.8), sharey=True)
    panels = (
        (axes[0], "family5_macro_f1", "Family5 macro-F1", 1.0),
        (axes[1], "ridge_r2_macro", "Ridge macro R2", 0.24),
    )
    for ax, metric, title, xmax in panels:
        values = df[metric].astype(float).to_numpy()
        bar_colours = [colours[group] for group in df["model_group"]]
        ax.barh(y, values, color=bar_colours)
        ax.set_xlim(0, xmax)
        ax.set_xlabel(title)
        ax.set_title(title)
        for i, value in enumerate(values):
            ax.text(value + xmax * 0.015, i, f"{value:.3f}", va="center", ha="left", fontsize=8)
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df["label"].tolist())
    axes[0].invert_yaxis()
    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color=color, markersize=8, label=label)
        for label, color in colours.items()
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Coding sequence vs TSS context")
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    return _savefig(fig, out, overwrite)


def plot_pooling_heatmap(pooling: pd.DataFrame, out: Path, overwrite: bool) -> Path | None:
    df = pooling.dropna(subset=["test_macro_f1"])
    if df.empty:
        return None
    pivot = df.pivot_table(index="encoder", columns="pooling", values="test_macro_f1", aggfunc="max")
    rows = [encoder for encoder in PAPER_ENCODERS if encoder in pivot.index]
    cols = [pool for pool in POOLING_ORDER if pool in pivot.columns]
    if not rows or not cols:
        return None
    values = pivot.loc[rows, cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    im = ax.imshow(values, cmap="YlGnBu", vmin=np.nanmin(values), vmax=np.nanmax(values))
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticklabels(rows)
    ax.set_title("Family5 macro-F1 by encoder and pooling")
    for i in range(len(rows)):
        for j in range(len(cols)):
            value = values[i, j]
            if np.isnan(value):
                continue
            ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("macro-F1")
    fig.tight_layout()
    return _savefig(fig, out, overwrite)


def plot_confusion_best_family5(family5: pd.DataFrame, out: Path, overwrite: bool) -> Path | None:
    encoder_rows = family5[~family5["feature_source"].isin(["kmer", "shuffled"])]
    encoder_rows = encoder_rows.dropna(subset=["test_macro_f1"])
    if encoder_rows.empty:
        return None
    best = encoder_rows.loc[encoder_rows["test_macro_f1"].astype(float).idxmax()]
    feature = best["feature_source"]
    path = DATA / f"confusion_5way_{feature}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    classes = payload["classes"]
    matrix = np.asarray(payload["matrix"], dtype=int)
    order = [classes.index(family) for family in FAMILY_ORDER if family in classes]
    matrix = matrix[np.ix_(order, order)]
    labels = [FAMILY_LABELS[FAMILY_ORDER[i]] for i in range(len(order))]
    normalised = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(normalised, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted family")
    ax.set_ylabel("True family")
    ax.set_title(f"Best family5 confusion matrix: {feature}")
    for i in range(len(labels)):
        for j in range(len(labels)):
            frac = normalised[i, j]
            colour = "white" if frac > 0.55 else "black"
            ax.text(j, i, f"{matrix[i, j]}\n{frac:.2f}", ha="center", va="center", fontsize=8, color=colour)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalised fraction")
    fig.tight_layout()
    return _savefig(fig, out, overwrite)


def dataset_path_for_feature(feature_source: str) -> Path | None:
    if feature_source == "dnabert2":
        return DATA / "dataset.parquet"
    if feature_source == "nt_v2":
        return DATA / "dataset_nt_v2.parquet"
    dataset = dataset_for_feature(feature_source)
    if dataset is None:
        return None
    return DATA / dataset


def _umap_coords(X: np.ndarray) -> np.ndarray:
    import umap

    return umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X)


def _plot_umap(df: pd.DataFrame, title: str, out: Path, overwrite: bool) -> Path:
    X = np.stack(df["x"].values).astype(np.float32)
    coords = _umap_coords(X)
    families = df["family"].to_numpy()
    palette = {
        "tf": "#1f77b4",
        "gpcr": "#d62728",
        "kinase": "#2ca02c",
        "ion": "#9467bd",
        "immune": "#ff7f0e",
    }
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    for family in FAMILY_ORDER:
        mask = families == family
        if not mask.any():
            continue
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, alpha=0.65, c=palette[family], label=f"{family} (n={int(mask.sum())})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8, markerscale=2.0, framealpha=0.85)
    fig.tight_layout()
    return _savefig(fig, out, overwrite)


def plot_umap_best_family5(family5: pd.DataFrame, out: Path, overwrite: bool) -> Path | None:
    encoder_rows = family5[~family5["feature_source"].isin(["kmer", "shuffled"])]
    encoder_rows = encoder_rows.dropna(subset=["test_macro_f1"])
    if encoder_rows.empty:
        return None
    best = encoder_rows.loc[encoder_rows["test_macro_f1"].astype(float).idxmax()]
    feature = best["feature_source"]
    dataset_path = dataset_path_for_feature(feature)
    if dataset_path is None or not dataset_path.exists():
        return None
    df = pd.read_parquet(dataset_path)
    return _plot_umap(df, f"UMAP of best family5 feature: {feature}", out, overwrite)


def plot_umap_dnabert2_compare(out: Path, overwrite: bool) -> Path | None:
    before_path = DATA / "dataset.parquet"
    after_path = DATA / "dataset_dnabert2_meanmean.parquet"
    if not before_path.exists() or not after_path.exists():
        return None
    before = pd.read_parquet(before_path).sort_values("ensembl_id").reset_index(drop=True)
    after = pd.read_parquet(after_path).sort_values("ensembl_id").reset_index(drop=True)
    if not (before["ensembl_id"].to_numpy() == after["ensembl_id"].to_numpy()).all():
        return None
    coords_before = _umap_coords(np.stack(before["x"].values).astype(np.float32))
    coords_after = _umap_coords(np.stack(after["x"].values).astype(np.float32))
    families = before["family"].to_numpy()
    palette = {
        "tf": "#1f77b4",
        "gpcr": "#d62728",
        "kinase": "#2ca02c",
        "ion": "#9467bd",
        "immune": "#ff7f0e",
    }
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, coords, title in (
        (axes[0], coords_before, "DNABERT-2 before tokenisation fix"),
        (axes[1], coords_after, "DNABERT-2 after tokenisation fix"),
    ):
        for family in FAMILY_ORDER:
            mask = families == family
            ax.scatter(coords[mask, 0], coords[mask, 1], s=4, alpha=0.6, c=palette[family], label=f"{family} (n={int(mask.sum())})")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP-1")
        ax.set_ylabel("UMAP-2")
        ax.set_title(title)
    axes[1].legend(loc="upper right", fontsize=8, markerscale=2.0, framealpha=0.85)
    fig.suptitle("DNABERT-2 boundary-token effect")
    fig.tight_layout()
    return _savefig(fig, out, overwrite)


def _write_manifest(
    out_dir: Path,
    *,
    tables: list[Path],
    figures: list[Path],
    missing: pd.DataFrame,
    skipped: list[str],
    overwrite: bool,
) -> Path:
    manifest_path = out_dir / "manifest.json"
    _check_write(manifest_path, overwrite)
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bundle": "paper",
        "inputs": {
            "metrics": str(DATA / "metrics.json"),
            "splits": str(DATA / "splits.json"),
            "data_dir": str(DATA),
        },
        "tables": [str(path) for path in tables],
        "figures": [str(path) for path in figures],
        "missing_cells": missing.to_dict("records"),
        "skipped": skipped,
    }
    manifest_path.write_text(json.dumps(payload, indent=2))
    return manifest_path


def build_analysis_artifacts(
    out_dir: Path,
    *,
    skip_umap: bool = False,
    overwrite: bool = False,
) -> dict[str, list[Path] | Path]:
    metrics = json.loads((DATA / "metrics.json").read_text())
    latest_logistic = latest_logistic_runs(metrics)
    latest_regression = latest_regression_runs(metrics)

    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    tables: list[Path] = []
    figures: list[Path] = []
    skipped: list[str] = []

    composition = dataset_composition_table()
    family5 = main_family5_table(latest_logistic)
    regression = main_regression_table(latest_regression)
    combined = combined_model_summary_table(family5, regression)
    context_ablation = context_ablation_table(latest_logistic, latest_regression)
    pooling = pooling_sweep_family5_table(latest_logistic)
    regression_full = regression_full_table(latest_regression)
    legacy = legacy_binary_appendix_table(latest_logistic)
    missing = missing_cells_table(latest_logistic, latest_regression, registered_feature_sources(), ENFORMER_SOURCES)

    table_specs = [
        ("dataset_composition", composition, "Dataset Composition", "Family totals and split counts."),
        ("main_family5", family5, "Main Family5", "Best cached family5 cell per encoder plus baselines."),
        ("main_regression", regression, "Main Regression", "Best cached Ridge-to-GenePT cell per encoder plus 4-mer baseline."),
        ("combined_model_summary", combined, "Combined Model Summary", "Family5 and Ridge headline metrics side by side."),
        (
            "context_ablation",
            context_ablation,
            "Context Ablation",
            "CDS, TSS, and Enformer context comparison for family5 and Ridge probes.",
        ),
        ("pooling_sweep_family5", pooling, "Pooling Sweep Family5", "All cached encoder-pooling family5 cells."),
        ("regression_full", regression_full, "Regression Full", "All cached Ridge cells with delta versus 4-mer."),
        ("legacy_binary_appendix", legacy, "Legacy Binary Appendix", "Cached binary-task results retained for appendix use."),
        ("missing_cells", missing, "Missing Cells", "Registered cells without cached metrics yet."),
    ]
    for name, df, title, description in table_specs:
        csv_path, md_path = write_table(df, name, tables_dir, title=title, description=description, overwrite=overwrite)
        tables.extend([csv_path, md_path])

    figure_specs = [
        ("family5_macro_f1.png", lambda: plot_family5_bar(family5, "test_macro_f1", figures_dir / "family5_macro_f1.png", overwrite)),
        ("family5_kappa.png", lambda: plot_family5_bar(family5, "test_kappa", figures_dir / "family5_kappa.png", overwrite)),
        ("ridge_r2.png", lambda: plot_ridge_r2(regression, figures_dir / "ridge_r2.png", overwrite)),
        ("model_tradeoff_f1_vs_r2.png", lambda: plot_tradeoff(combined, figures_dir / "model_tradeoff_f1_vs_r2.png", overwrite)),
        (
            "context_ablation_cds_tss_enformer.png",
            lambda: plot_context_ablation(
                context_ablation,
                figures_dir / "context_ablation_cds_tss_enformer.png",
                overwrite,
            ),
        ),
        ("pooling_heatmap_family5.png", lambda: plot_pooling_heatmap(pooling, figures_dir / "pooling_heatmap_family5.png", overwrite)),
        ("confusion_best_family5.png", lambda: plot_confusion_best_family5(family5, figures_dir / "confusion_best_family5.png", overwrite)),
    ]
    for name, make_figure in figure_specs:
        path = make_figure()
        if path is not None:
            figures.append(path)
        else:
            skipped.append(name)

    if skip_umap:
        skipped.extend(["umap_best_family5.png", "umap_dnabert2_tokenisation_compare.png"])
    else:
        for name, make_figure in (
            ("umap_best_family5.png", lambda: plot_umap_best_family5(family5, figures_dir / "umap_best_family5.png", overwrite)),
            ("umap_dnabert2_tokenisation_compare.png", lambda: plot_umap_dnabert2_compare(figures_dir / "umap_dnabert2_tokenisation_compare.png", overwrite)),
        ):
            path = make_figure()
            if path is not None:
                figures.append(path)
            else:
                skipped.append(name)

    manifest = _write_manifest(out_dir, tables=tables, figures=figures, missing=missing, skipped=skipped, overwrite=overwrite)
    return {"tables": tables, "figures": figures, "manifest": manifest}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("analysis"))
    parser.add_argument("--bundle", choices=["paper"], default="paper")
    parser.add_argument("--skip-umap", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_analysis_artifacts(args.out, skip_umap=args.skip_umap, overwrite=args.overwrite)
    print(f"wrote {len(result['tables'])} table files")
    print(f"wrote {len(result['figures'])} figure files")
    print(f"wrote manifest -> {result['manifest']}")


if __name__ == "__main__":
    main()
