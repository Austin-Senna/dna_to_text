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

    def test_ties_without_an_order_keep_first_run(self):
        a = {"C_sweep": [{"C": 1, "macro_f1": 0.5}]}
        b = {"C_sweep": [{"C": 1, "macro_f1": 0.5}]}
        self.assertIs(select_by_val([a, b]), a)

    def test_missing_sweep_raises(self):
        with self.assertRaises(KeyError):
            val_score({"test_macro_f1": 0.5})


if __name__ == "__main__":
    unittest.main()


# --- G14: explicit candidate pools; ties by pool order, not record order ------

def _cls(src, val):
    return {"feature_source": src, "task": "family5", "C_sweep": [{"C": 1.0, "macro_f1": val}]}


def test_a_decoy_pool_never_joins_the_headline_pick():
    import headline_cells as hc
    recs = {"tss_dnabert2_meanmean": _cls("tss_dnabert2_meanmean", 0.5),
            "tss_dnabert2_tssanchored": _cls("tss_dnabert2_tssanchored", 0.9),
            "tss_dnabert2_centermean": _cls("tss_dnabert2_centermean", 0.8)}
    assert hc.best(recs, "tss_dnabert2_") == "tss_dnabert2_meanmean"


def test_tied_pools_pick_the_same_cell_in_any_record_order():
    import headline_cells as hc
    a, b = _cls("nt_v2_meanD", 0.6), _cls("nt_v2_meanmean", 0.6)
    forward = hc.best({"nt_v2_meanD": a, "nt_v2_meanmean": b}, "nt_v2_")
    backward = hc.best({"nt_v2_meanmean": b, "nt_v2_meanD": a}, "nt_v2_")
    assert forward == backward == "nt_v2_meanmean"


def test_the_table_builder_ignores_a_decoy_pool(monkeypatch):
    import build_paper_tables as bpt
    monkeypatch.setitem(bpt.CLS, "tss_dnabert2_tssanchored", _cls("tss_dnabert2_tssanchored", 0.99))
    pool, _ = bpt.cls_best_pool("dnabert2", "TSS")
    assert pool != "tssanchored"


def _decoys():
    decoys = []
    for enc in ("dnabert2", "nt_v2", "gena_lm", "hyena_dna"):
        for src in (f"{enc}_tssanchored", f"tss_{enc}_tssanchored", f"tss_{enc}_centermean"):
            decoys.append({**_cls(src, 0.99), "test_macro_f1": 0.99})
            ds = f"dataset_{src}.parquet"
            decoys.append({"model": "linear_probe", "dataset": ds, "test_r2_macro": 0.99,
                           "alpha_sweep": [{"alpha": 1.0, "r2": 0.99, "mean_cosine": 0.9}]})
    return decoys


def test_builder_accessors_ignore_decoy_pools():
    import build_paper_tables as bpt
    import build_result_figures as brf
    m, decoyed = bpt.M, bpt.M + _decoys()
    for enc in bpt.ENCODERS:
        for tss in (False, True):
            assert bpt._best_f1_family5(decoyed, enc, tss) == bpt._best_f1_family5(m, enc, tss)
            assert brf._best_cls(decoyed, enc, tss) == brf._best_cls(m, enc, tss)
            assert brf._best_reg_enc_ctx(decoyed, enc, tss) == brf._best_reg_enc_ctx(m, enc, tss)
        assert bpt._reg_r2_tss(decoyed, enc) == bpt._reg_r2_tss(m, enc)
        assert bpt._cls_f1(decoyed, enc) == bpt._cls_f1(m, enc)
        assert bpt._reg_r2(decoyed, enc) == bpt._reg_r2(m, enc)
        assert brf._best_reg_enc(decoyed, enc) == brf._best_reg_enc(m, enc)


def test_tables_and_headline_pick_the_same_pools():
    import build_paper_tables as bpt
    import headline_cells as hc
    for enc in hc.ENCODERS:
        assert f"{enc}_{bpt.cls_best_pool(enc)[0]}" == hc.CLS_BEST[enc]
        assert f"{enc}_{bpt.reg_best_pool(enc)[0]}" == hc.REG_BEST[enc]
        assert f"tss_{enc}_{bpt.cls_best_pool(enc, 'TSS')[0]}" == hc.CLS_BEST_TSS[enc]
        assert f"tss_{enc}_{bpt.reg_best_pool(enc, 'TSS')[0]}" == hc.REG_BEST_TSS[enc]
    assert (bpt.BEST_DNA_CLS, bpt.BEST_DNA_REG) == (hc.BEST_DNA_CLS, hc.BEST_DNA_REG)
