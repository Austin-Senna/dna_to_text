import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from linear_trainer.selection import select_by_val, val_score


class SelectionTests(unittest.TestCase):
    def test_classification_selects_on_validation_not_test(self):
        a = {"C_sweep": [{"C": 1, "macro_f1": 0.70}], "test_macro_f1": 0.90}
        b = {"C_sweep": [{"C": 1, "macro_f1": 0.60}, {"C": 10, "macro_f1": 0.75}],
             "test_macro_f1": 0.80}
        self.assertIs(select_by_val([a, b]), b)

    def test_regression_uses_r2_unless_selected_by_cosine(self):
        rec = {"alpha_sweep": [{"alpha": 1, "r2": 0.05, "mean_cosine": 0.9},
                               {"alpha": 10, "r2": 0.07, "mean_cosine": 0.8}]}
        self.assertEqual(val_score(rec), 0.07)
        self.assertEqual(val_score({**rec, "select_by": "cosine"}), 0.9)

    def test_legacy_cosine_only_sweep_scores_on_cosine(self):
        rec = {"alpha_sweep": [{"alpha": 1, "mean_cosine": 0.91},
                               {"alpha": 10, "mean_cosine": 0.93}]}
        self.assertEqual(val_score(rec), 0.93)

    def test_ties_keep_first_run(self):
        a = {"C_sweep": [{"C": 1, "macro_f1": 0.5}]}
        b = {"C_sweep": [{"C": 1, "macro_f1": 0.5}]}
        self.assertIs(select_by_val([a, b]), a)

    def test_missing_sweep_raises(self):
        with self.assertRaises(KeyError):
            val_score({"test_macro_f1": 0.5})


if __name__ == "__main__":
    unittest.main()
