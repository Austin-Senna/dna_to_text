import sys
import unittest

import pytest
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


def _by_source(arm="cds", val=None):
    """One family5 record per candidate pool of every encoder, plus the k-mers."""
    from data_loader.model_registry import encoder_pools
    from linear_trainer import records as R
    val = val or {}
    prefix = "tss_" if arm == "tss" else ""
    srcs = [f"{prefix}{e}_{p}" for e in R.ENCODERS for p in encoder_pools(e, arm.upper())]
    srcs += list(R.NT_KMERS) + list(R.AA_KMERS)
    return {s: _cls(s, val.get(s, 0.5 + 0.001 * i)) for i, s in enumerate(srcs)}


DECOYS = ("tssanchored", "centermean", "chunk4mergc", "chunk6mer")


@pytest.mark.parametrize("arm", ["cds", "tss"])
def test_the_records_picks_ignore_decoy_pools(arm):
    from linear_trainer import records as R
    by = _by_source(arm)
    prefix = "tss_" if arm == "tss" else ""
    decoyed = {**by, **{f"{prefix}{e}_{d}": _cls(f"{prefix}{e}_{d}", 0.99)
                        for e in R.ENCODERS for d in DECOYS}}
    for enc in R.ENCODERS:
        assert R.best_pool(decoyed, enc, arm) == R.best_pool(by, enc, arm)
    assert R.best_encoder(decoyed, arm) == R.best_encoder(by, arm)


def test_the_records_picks_do_not_depend_on_record_order():
    from linear_trainer import records as R
    by = _by_source(val={"nt_v2_meanD": 0.9, "nt_v2_meanmean": 0.9, "aa2": 0.8, "aa3": 0.8})
    rev = dict(reversed(list(by.items())))
    assert R.best_pool(by, "nt_v2", "cds") == R.best_pool(rev, "nt_v2", "cds")
    assert R.best_aa(by) == R.best_aa(rev)
    assert R.best_encoder(by, "cds") == R.best_encoder(rev, "cds")


def test_the_records_picks_never_read_a_test_metric():
    from linear_trainer import records as R
    by = _by_source(val={"kmer": 0.6, "kmer6": 0.5, "aa1": 0.4, "aa2": 0.5, "aa3": 0.45})
    flipped = {s: {**r, "test_macro_f1": 1.0 - r["C_sweep"][0]["macro_f1"]} for s, r in by.items()}
    assert (R.best_nt_kmer(flipped), R.best_aa(flipped)) == (R.best_nt_kmer(by), R.best_aa(by)) == ("kmer", "aa2")
    assert R.best_encoder(flipped, "cds") == R.best_encoder(by, "cds")


@pytest.mark.parametrize("script", ["build_paper_tables.py", "build_result_figures.py",
                                    "build_statistics.py", "build_umap_compare.py"])
def test_the_builders_have_no_pick_of_their_own(script):
    """Every pick in a builder goes through linear_trainer.records (G14), which
    refuses a partial candidate set."""
    src = (ROOT / "scripts" / script).read_text()
    assert "select_pool" not in src and "val_score" not in src and "select_by_val" not in src
    assert "sorted(" not in src or script == "build_statistics.py"   # stats sorts keys, not scores


def _decoys():
    decoys = []
    for enc in ("dnabert2", "nt_v2", "gena_lm", "hyena_dna"):
        for src in (f"{enc}_tssanchored", f"tss_{enc}_tssanchored", f"tss_{enc}_centermean"):
            decoys.append({**_cls(src, 0.99), "test_macro_f1": 0.99})
            ds = f"dataset_{src}.parquet"
            decoys.append({"model": "linear_probe", "dataset": ds, "test_r2_macro": 0.99,
                           "alpha_sweep": [{"alpha": 1.0, "r2": 0.99, "mean_cosine": 0.9}]})
    return decoys


def test_the_poster_accessors_ignore_decoy_pools():
    import poster_may_records as brf      # the May accessors, frozen for the poster
    m, decoyed = brf.M, brf.M + _decoys()
    for enc in ("dnabert2", "nt_v2", "gena_lm", "hyena_dna"):
        for tss in (False, True):
            assert brf._best_cls(decoyed, enc, tss) == brf._best_cls(m, enc, tss)
            assert brf._best_reg_enc_ctx(decoyed, enc, tss) == brf._best_reg_enc_ctx(m, enc, tss)
        assert brf._best_reg_enc(decoyed, enc) == brf._best_reg_enc(m, enc)
