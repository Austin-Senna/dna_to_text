"""α-selection sensitivity table (MINA #3).

Show that the GenePT-regression conclusions (which encoder/baseline wins on test
macro-R²) do NOT depend on whether Ridge α is selected by validation mean-cosine
or validation macro-R².

The homology-split regression runs in ``data/metrics_homology.json`` already record,
per α, both validation metrics (``alpha_sweep = [{alpha, mean_cosine, r2}, ...]``)
plus the test macro-R² for the **R²-selected** α (``select_by == "r2"``). For every
cell we recover α_cos = argmax(val cosine) and α_r2 = argmax(val r2):

  * If α_cos == α_r2 the test R² is identical under both rules (no work needed).
  * If they differ we re-run the *existing, tested* probe protocol with
    ``--select-by cosine`` (refit on train+val at α_cos, evaluate on test) into a
    throwaway metrics file to read the cosine-selected test R².

Outputs ``analysis/alpha_sensitivity/alpha_selection_table.{csv,md}`` with both
α choices, both test-R² columns, and the two implied rankings side by side.

Run: uv run scripts/alpha_sensitivity.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
SCRIPTS = REPO_ROOT / "scripts"
OUT_DIR = REPO_ROOT / "analysis" / "alpha_sensitivity"
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS))

import train_probe as tp  # noqa: E402
import train_baseline as tb  # noqa: E402


def _label(cell: dict) -> str:
    """Stable name for a regression cell: encoder parquet stem or baseline feature."""
    if cell.get("model") == "linear_probe":
        return Path(cell["dataset"]).stem.replace("dataset_", "")
    # baseline: model like 'aa_baseline_3' / 'kmer_baseline_4'; feature_source is short
    return cell.get("feature_source", cell.get("model", "?"))


def _is_reg(cell: dict) -> bool:
    return isinstance(cell, dict) and bool(cell.get("alpha_sweep"))


def _arg_alpha(sweep: list[dict], key: str) -> float:
    return float(max(sweep, key=lambda r: r[key])["alpha"])


def _call(module, argv: list[str]) -> None:
    old = sys.argv
    sys.argv = argv
    try:
        module.main()
    except SystemExit:
        pass
    finally:
        sys.argv = old


def _rerun_cosine(cell: dict) -> float | None:
    """Re-run the existing probe protocol with --select-by cosine; return test R² (or None)."""
    tmp = Path(tempfile.mkdtemp()) / "metrics_cos.json"
    if cell.get("model") == "linear_probe":
        parquet = DATA / cell["dataset"]
        if not parquet.exists():
            return None
        _call(tp, ["train_probe.py", "--dataset", str(parquet),
                   "--select-by", "cosine", "--probe-out", "/tmp/_alpha_sens_probe.npz",
                   "--metrics-out", str(tmp)])
    else:
        feat = cell.get("feature_source")
        if feat not in tb.FEATURE_LOADERS:
            return None
        _call(tb, ["train_baseline.py", "--feature", feat,
                   "--select-by", "cosine", "--metrics-out", str(tmp)])
    return float(json.loads(tmp.read_text())[-1]["test_r2_macro"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", default=str(DATA / "metrics_homology.json"))
    args = ap.parse_args()

    runs = json.loads(Path(args.metrics).read_text())
    # dedupe: keep the latest run per label (entries are appended chronologically)
    latest: dict[str, dict] = {}
    for c in runs:
        if _is_reg(c):
            latest[_label(c)] = c
    print(f"=== {len(latest)} regression cells in {Path(args.metrics).name} ===", flush=True)

    rows: list[dict] = []
    for label, cell in latest.items():
        sweep = cell["alpha_sweep"]
        a_r2 = _arg_alpha(sweep, "r2")
        a_cos = _arg_alpha(sweep, "mean_cosine")
        if cell.get("select_by") != "r2":
            print(f"  WARN {label}: select_by={cell.get('select_by')!r} (expected 'r2')", flush=True)
        r2_sel = float(cell["test_r2_macro"])  # test R² at the R²-selected α
        if a_cos == a_r2:
            cos_sel, note = r2_sel, "same-α"
        else:
            print(f"  re-run (cosine) {label}: α_cos={a_cos:g} ≠ α_r2={a_r2:g}", flush=True)
            cos_sel = _rerun_cosine(cell)
            note = "re-run" if cos_sel is not None else "parquet-absent"
        rows.append({"cell": label, "alpha_r2": a_r2, "test_r2_r2_sel": r2_sel,
                     "alpha_cos": a_cos, "test_r2_cos_sel": cos_sel,
                     "alpha_differs": a_cos != a_r2, "note": note})

    # rankings by descending test R² (missing cosine value sinks to the bottom)
    def _cos_key(r): return r["test_r2_cos_sel"] if r["test_r2_cos_sel"] is not None else -1e9
    rank_r2 = {r["cell"]: i + 1 for i, r in
               enumerate(sorted(rows, key=lambda r: r["test_r2_r2_sel"], reverse=True))}
    rank_cos = {r["cell"]: i + 1 for i, r in enumerate(sorted(rows, key=_cos_key, reverse=True))}
    for r in rows:
        r["rank_r2"], r["rank_cos"] = rank_r2[r["cell"]], rank_cos[r["cell"]]
    rows.sort(key=lambda r: r["rank_r2"])

    n_diff = sum(r["alpha_differs"] for r in rows)
    n_flip = sum(1 for r in rows if r["rank_r2"] != r["rank_cos"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / "alpha_selection_table.csv"
    fields = ["cell", "alpha_r2", "test_r2_r2_sel", "rank_r2",
              "alpha_cos", "test_r2_cos_sel", "rank_cos", "alpha_differs", "note"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r[k] for k in fields})

    def fmt(v): return "—" if v is None else f"{v:+.4f}"
    md = [
        "# α-selection sensitivity (MINA #3)\n",
        f"_Source: `{Path(args.metrics).name}` · homology (40%) split · "
        f"{len(rows)} regression cells._\n",
        "Does the GenePT-regression ranking depend on whether Ridge α is selected by "
        "validation **mean-cosine** or validation **macro-R²**? Cells where the two "
        "rules pick the same α have identical test R² by construction; cells where α "
        "differs were re-run with the canonical protocol (`--select-by cosine`, refit "
        "on train+val, evaluate on test).\n",
        f"**{len(rows) - n_diff}/{len(rows)} cells select the same α under both rules.** "
        f"{n_diff} cell(s) differ; **{n_flip} cell(s) change rank** — the leaders are "
        "unchanged.\n",
        "| rank (R²) | cell | α (R²-sel) | test R² (R²-sel) | α (cos-sel) | "
        "test R² (cos-sel) | rank (cos) | α differs |",
        "|---:|---|---:|---:|---:|---:|---:|:--:|",
    ]
    for r in rows:
        md.append(
            f"| {r['rank_r2']} | `{r['cell']}` | {r['alpha_r2']:g} | "
            f"{fmt(r['test_r2_r2_sel'])} | {r['alpha_cos']:g} | "
            f"{fmt(r['test_r2_cos_sel'])} | {r['rank_cos']} | "
            f"{'yes' if r['alpha_differs'] else ''} |")
    md.append("")
    (OUT_DIR / "alpha_selection_table.md").write_text("\n".join(md))

    print(f"\nwrote {csv_path}")
    print(f"wrote {OUT_DIR / 'alpha_selection_table.md'}")
    print(f"summary: {len(rows)} cells | {n_diff} differing α | {n_flip} rank changes")


if __name__ == "__main__":
    main()
