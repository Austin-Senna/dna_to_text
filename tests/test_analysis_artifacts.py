import tempfile
import unittest
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class AnalysisArtifactTests(unittest.TestCase):
    def test_latest_logistic_runs_dedupe_by_encoder_task_and_shuffle(self):
        from scripts.build_analysis_artifacts import latest_logistic_runs

        metrics = [
            {
                "model": "logistic_probe",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "encoder": "nt_v2_meanD",
                "task": "family5",
                "shuffled_labels": False,
                "test_macro_f1": 0.1,
            },
            {
                "model": "logistic_probe",
                "timestamp": "2026-01-02T00:00:00+00:00",
                "encoder": "nt_v2_meanD",
                "task": "family5",
                "shuffled_labels": False,
                "test_macro_f1": 0.8,
            },
            {
                "model": "logistic_probe",
                "timestamp": "2026-01-03T00:00:00+00:00",
                "encoder": "nt_v2_meanD",
                "task": "family5",
                "shuffled_labels": True,
                "test_macro_f1": 0.2,
            },
        ]

        latest = latest_logistic_runs(metrics)

        self.assertEqual(latest[("nt_v2_meanD", "family5", False)]["test_macro_f1"], 0.8)
        self.assertEqual(latest[("nt_v2_meanD", "family5", True)]["test_macro_f1"], 0.2)

    def test_best_family5_rows_choose_best_pooling_per_registered_encoder(self):
        from scripts.build_analysis_artifacts import best_family5_rows, latest_logistic_runs

        metrics = [
            {
                "model": "logistic_probe",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "encoder": "kmer",
                "task": "family5",
                "test_macro_f1": 0.67,
                "test_kappa": 0.70,
                "test_accuracy": 0.82,
            },
            {
                "model": "logistic_probe",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "encoder": "nt_v2_meanG",
                "task": "family5",
                "test_macro_f1": 0.80,
                "test_kappa": 0.78,
                "test_accuracy": 0.86,
            },
            {
                "model": "logistic_probe",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "encoder": "nt_v2_meanD",
                "task": "family5",
                "test_macro_f1": 0.83,
                "test_kappa": 0.82,
                "test_accuracy": 0.88,
            },
            {
                "model": "logistic_probe",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "encoder": "hyena_dna_meanG",
                "task": "family5",
                "test_macro_f1": 0.71,
                "test_kappa": 0.69,
                "test_accuracy": 0.81,
            },
        ]

        rows = best_family5_rows(latest_logistic_runs(metrics), ["nt_v2", "hyena_dna", "gena_lm"])

        self.assertEqual(rows.iloc[0]["encoder"], "nt_v2")
        self.assertEqual(rows.iloc[0]["feature_source"], "nt_v2_meanD")
        self.assertAlmostEqual(rows.iloc[0]["delta_f1_vs_4mer"], 0.16)
        self.assertAlmostEqual(rows.iloc[0]["delta_kappa_vs_4mer"], 0.12)
        self.assertEqual(rows.iloc[1]["encoder"], "hyena_dna")
        self.assertEqual(rows.iloc[1]["feature_source"], "hyena_dna_meanG")
        self.assertNotIn("gena_lm", rows["encoder"].tolist())

    def test_latest_regression_runs_indexes_shuffled_y_anti_baselines(self):
        from scripts.build_analysis_artifacts import latest_regression_runs, main_regression_table

        metrics = [
            {
                "model": "kmer_baseline_4",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "test_r2_macro": 0.17,
            },
            {
                "model": "anti_baseline_shuffled_y",
                "timestamp": "2026-01-02T00:00:00+00:00",
                "dataset": "dataset.parquet",
                "test_r2_macro": -0.02,
                "test_mean_cosine": 0.90,
                "alpha": 1000,
            },
            {
                "model": "anti_baseline_shuffled_y",
                "timestamp": "2026-01-02T00:00:01+00:00",
                "dataset": "dataset_nt_v2.parquet",
                "test_r2_macro": -0.01,
                "test_mean_cosine": 0.91,
                "alpha": 1000,
            },
            {
                "model": "linear_probe",
                "timestamp": "2026-01-03T00:00:00+00:00",
                "dataset": "dataset_nt_v2.parquet",
                "test_r2_macro": 0.19,
                "test_mean_cosine": 0.93,
                "alpha": 10,
            },
        ]

        latest = latest_regression_runs(metrics)
        self.assertIn("shuffled_y:dataset_nt_v2.parquet", latest)

        table = main_regression_table(latest)
        anti_rows = table[table["feature_source"] == "shuffled_y"]
        self.assertEqual(len(anti_rows), 1)
        anti = anti_rows.iloc[0]
        self.assertEqual(anti["encoder"], "shuffled_y")
        self.assertEqual(anti["pooling"], "-")
        self.assertAlmostEqual(anti["delta_vs_4mer"], -0.18)

    def test_missing_cells_report_registered_unrun_family5_and_regression_cells(self):
        from scripts.build_analysis_artifacts import missing_cells_table

        logistic_latest = {("nt_v2_meanD", "family5", False): {"encoder": "nt_v2_meanD"}}
        regression_latest = {"dataset_nt_v2_meanD.parquet": {"dataset": "dataset_nt_v2_meanD.parquet"}}

        missing = missing_cells_table(
            logistic_latest,
            regression_latest,
            registered_feature_sources=["nt_v2_meanD", "gena_lm_meanD"],
            enformer_sources=["enformer_tracks_center"],
        )

        records = {(row.artifact, row.feature_source) for row in missing.itertuples()}
        self.assertIn(("family5", "gena_lm_meanD"), records)
        self.assertIn(("regression", "gena_lm_meanD"), records)
        self.assertIn(("family5", "enformer_tracks_center"), records)
        self.assertNotIn(("family5", "nt_v2_meanD"), records)

    def test_write_table_outputs_csv_and_markdown(self):
        from scripts.build_analysis_artifacts import write_table

        df = pd.DataFrame(
            [
                {"feature_source": "nt_v2_meanD", "test_macro_f1": 0.8275},
                {"feature_source": "kmer", "test_macro_f1": 0.6722},
            ]
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            csv_path, md_path = write_table(
                df,
                "main_family5",
                out,
                title="Main Family5",
                description="Cached family5 cells.",
                overwrite=True,
            )

            self.assertTrue(csv_path.exists())
            self.assertTrue(md_path.exists())
            self.assertIn("nt_v2_meanD", csv_path.read_text())
            self.assertIn("# Main Family5", md_path.read_text())
            self.assertIn("| feature_source | test_macro_f1 |", md_path.read_text())

    def test_split_lookup_ignores_metadata_keys(self):
        from scripts.build_analysis_artifacts import split_lookup_from_splits

        lookup = split_lookup_from_splits(
            {
                "train": ["ENSG1"],
                "val": ["ENSG2"],
                "test": ["ENSG3"],
                "seed": 42,
                "stratify": "family",
                "source": "dataset.parquet",
            }
        )

        self.assertEqual(lookup, {"ENSG1": "train", "ENSG2": "val", "ENSG3": "test"})

    def test_parse_kappa_summary_reads_baseline_rows(self):
        from scripts.build_analysis_artifacts import parse_kappa_summary

        text = """# Cohen's kappa

## family5

| Dataset | C | macro-F1 | Cohen's κ |
|---|---:|---:|---:|
| `kmer` | 1000 | 0.6722 | **0.7024** |
| `shuffled-label` (anti-baseline) | 100 | 0.2078 | +0.0482 |
"""

        parsed = parse_kappa_summary(text)

        self.assertEqual(parsed[("family5", "kmer")], 0.7024)
        self.assertEqual(parsed[("family5", "shuffled")], 0.0482)

    def test_context_ablation_table_uses_expected_rows_and_enformer_per_metric(self):
        from scripts.build_analysis_artifacts import (
            context_ablation_table,
            latest_logistic_runs,
            latest_regression_runs,
        )

        metrics = []
        for feature, f1 in [
            ("kmer", 0.6722),
            ("nt_v2_meanD", 0.8275),
            ("enformer_tss_4mer", 0.2452),
            ("tss_nt_v2_meanmean", 0.4468),
            ("enformer_trunk_global", 0.5450),
            ("enformer_trunk_center", 0.5127),
        ]:
            metrics.append(
                {
                    "model": "logistic_probe",
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "encoder": feature,
                    "task": "family5",
                    "shuffled_labels": False,
                    "test_macro_f1": f1,
                    "test_kappa": f1 - 0.1,
                    "test_accuracy": f1 + 0.05,
                }
            )
        for dataset, r2 in [
            ("kmer", 0.1743),
            ("dataset_nt_v2_meanD.parquet", 0.1882),
            ("dataset_enformer_tss_4mer.parquet", 0.0413),
            ("data/dataset_tss_nt_v2_meanmean.parquet", 0.1174),
            ("dataset_enformer_trunk_global.parquet", 0.1389),
            ("dataset_enformer_trunk_center.parquet", 0.1425),
        ]:
            row = {
                "model": "linear_probe",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "dataset": dataset,
                "test_r2_macro": r2,
            }
            if dataset == "kmer":
                row["model"] = "kmer_baseline_4"
            metrics.append(row)

        table = context_ablation_table(latest_logistic_runs(metrics), latest_regression_runs(metrics))

        self.assertEqual(
            table["label"].tolist(),
            ["CDS 4-mer", "CDS NT-v2 meanD", "TSS 4-mer", "TSS NT-v2 meanmean", "Enformer trunk"],
        )
        enformer = table[table["label"] == "Enformer trunk"].iloc[0]
        self.assertEqual(enformer["family5_feature_source"], "enformer_trunk_global")
        self.assertEqual(enformer["regression_feature_source"], "enformer_trunk_center")
        self.assertAlmostEqual(enformer["family5_macro_f1"], 0.5450)
        self.assertAlmostEqual(enformer["ridge_r2_macro"], 0.1425)


if __name__ == "__main__":
    unittest.main()
