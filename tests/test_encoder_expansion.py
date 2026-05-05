import unittest

import numpy as np


class EncoderRegistryTests(unittest.TestCase):
    def test_main_encoder_registry_exposes_new_models_and_family5_matrix(self):
        from data_loader.model_registry import (
            family5_feature_sources,
            get_encoder_spec,
            main_encoder_names,
        )

        self.assertEqual(
            main_encoder_names(),
            ("dnabert2", "nt_v2", "gena_lm", "hyena_dna", "caduceus_ps"),
        )
        self.assertEqual(get_encoder_spec("gena_lm").model_kind, "self_supervised_encoder")
        self.assertEqual(get_encoder_spec("hyena_dna").max_content_tokens, 8192)
        self.assertEqual(get_encoder_spec("caduceus_ps").cache_name, "caduceus_ps")
        self.assertNotIn("length", family5_feature_sources())
        self.assertIn("kmer", family5_feature_sources())
        self.assertIn("shuffled", family5_feature_sources())

    def test_pooling_variants_are_filtered_by_available_reductions(self):
        from data_loader.pooling_aggregator import available_variants, aggregate

        mean = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        max_ = np.asarray([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32)
        without_cls = {"mean": mean, "max": max_}

        self.assertEqual(
            available_variants(without_cls),
            ("meanmean", "maxmean", "meanD", "meanG"),
        )
        self.assertNotIn("clsmean", available_variants(without_cls))
        np.testing.assert_allclose(
            aggregate(without_cls, "meanD"),
            np.asarray([1.0, 2.0, 3.0, 4.0, 2.0, 3.0], dtype=np.float32),
        )

        with_cls = {**without_cls, "cls": np.asarray([[9.0, 8.0]], dtype=np.float32)}
        self.assertIn("clsmean", available_variants(with_cls))

    def test_multi_pool_allows_tokenizers_without_cls(self):
        import torch

        from data_loader.multi_pool import embed_sequence_multi_pool

        class Tokenizer:
            cls_token_id = None
            bos_token_id = None
            sep_token_id = None
            eos_token_id = None

            def __call__(self, seq, add_special_tokens=False, return_tensors=None):
                return {"input_ids": [0, 1, 2, 3]}

        class Model:
            def __call__(self, input_ids, attention_mask=None):
                hidden = torch.arange(input_ids.shape[1] * 2, dtype=torch.float32).reshape(
                    1, input_ids.shape[1], 2
                )
                return type("Out", (), {"last_hidden_state": hidden})

        reductions = embed_sequence_multi_pool(
            "ACGT", Model(), Tokenizer(), "cpu", max_content_tokens=4, stride=1
        )

        self.assertEqual(set(reductions), {"mean", "max"})

    def test_multi_pool_allows_models_without_attention_mask_argument(self):
        import torch

        from data_loader.multi_pool import embed_sequence_multi_pool

        class Tokenizer:
            cls_token_id = 10
            bos_token_id = None
            sep_token_id = None
            eos_token_id = None

            def __call__(self, seq, add_special_tokens=False, return_tensors=None):
                return {"input_ids": [0, 1]}

        class Model:
            def __call__(self, input_ids):
                hidden = torch.ones((1, input_ids.shape[1], 3), dtype=torch.float32)
                return type("Out", (), {"last_hidden_state": hidden})

        reductions = embed_sequence_multi_pool(
            "AC", Model(), Tokenizer(), "cpu", max_content_tokens=2, stride=1
        )

        self.assertEqual(reductions["mean"].shape, (1, 3))

    def test_gena_lm_loader_installs_old_transformers_pruning_shims(self):
        import transformers.pytorch_utils as pytorch_utils

        from data_loader.gena_lm_encoder import _install_transformers_shims

        old_find = getattr(pytorch_utils, "find_pruneable_heads_and_indices", None)
        old_prune = getattr(pytorch_utils, "prune_linear_layer", None)
        try:
            if hasattr(pytorch_utils, "find_pruneable_heads_and_indices"):
                delattr(pytorch_utils, "find_pruneable_heads_and_indices")
            if hasattr(pytorch_utils, "prune_linear_layer"):
                delattr(pytorch_utils, "prune_linear_layer")

            _install_transformers_shims()

            self.assertTrue(hasattr(pytorch_utils, "find_pruneable_heads_and_indices"))
            self.assertTrue(hasattr(pytorch_utils, "prune_linear_layer"))
        finally:
            if old_find is not None:
                setattr(pytorch_utils, "find_pruneable_heads_and_indices", old_find)
            if old_prune is not None:
                setattr(pytorch_utils, "prune_linear_layer", old_prune)

    def test_gena_lm_loader_accepts_legacy_attention_mask_device_arg(self):
        import torch
        from transformers.modeling_utils import ModuleUtilsMixin

        from data_loader.gena_lm_encoder import _install_transformers_shims

        _install_transformers_shims()

        class Dummy(ModuleUtilsMixin):
            dtype = torch.float32
            config = type("Config", (), {"is_decoder": False})()

        mask = torch.ones((1, 2), dtype=torch.long)
        extended = Dummy().get_extended_attention_mask(mask, (1, 2), torch.device("cpu"))

        self.assertEqual(extended.dtype, torch.float32)
        self.assertEqual(extended.shape, (1, 1, 1, 2))

    def test_gena_lm_loader_installs_head_mask_shim(self):
        from transformers.modeling_utils import ModuleUtilsMixin

        from data_loader.gena_lm_encoder import _install_transformers_shims

        old_get_head_mask = getattr(ModuleUtilsMixin, "get_head_mask", None)
        try:
            if hasattr(ModuleUtilsMixin, "get_head_mask"):
                delattr(ModuleUtilsMixin, "get_head_mask")

            _install_transformers_shims()

            self.assertEqual(ModuleUtilsMixin().get_head_mask(None, 3), [None, None, None])
        finally:
            if old_get_head_mask is not None:
                setattr(ModuleUtilsMixin, "get_head_mask", old_get_head_mask)

    def test_gena_lm_loader_selects_backbone_when_model_is_masked_lm(self):
        from data_loader.gena_lm_encoder import _select_backbone

        backbone = object()
        wrapper = type("Wrapper", (), {"bert": backbone})()

        self.assertIs(_select_backbone(wrapper), backbone)
        self.assertIs(_select_backbone(backbone), backbone)


class EnformerWindowTests(unittest.TestCase):
    def test_tss_window_is_centered_on_strand_specific_tss(self):
        from data_loader.enformer_windows import centered_window

        self.assertEqual(
            centered_window(seq_region_name="1", start=1000, end=2000, strand=1, length=11),
            ("1", 995, 1005),
        )
        self.assertEqual(
            centered_window(seq_region_name="2", start=1000, end=2000, strand=-1, length=11),
            ("2", 1995, 2005),
        )

    def test_tss_window_clips_to_start_of_chromosome(self):
        from data_loader.enformer_windows import centered_window

        self.assertEqual(
            centered_window(seq_region_name="X", start=3, end=50, strand=1, length=10),
            ("X", 1, 10),
        )


class FeatureTests(unittest.TestCase):
    def test_generic_four_mer_feature_handles_any_sequence(self):
        from kmer_baseline import featurize_sequence

        features = featurize_sequence("AAAAC")
        self.assertEqual(features.shape, (256,))
        self.assertAlmostEqual(float(features.sum()), 1.0)


class RegressionTableTests(unittest.TestCase):
    def test_regression_table_dataset_names_match_probe_outputs(self):
        from scripts.build_regression_table import _dataset_name

        self.assertIsNone(_dataset_name("kmer"))
        self.assertEqual(_dataset_name("dnabert2"), "dataset.parquet")
        self.assertEqual(_dataset_name("nt_v2"), "dataset_nt_v2.parquet")
        self.assertEqual(_dataset_name("gena_lm_meanD"), "dataset_gena_lm_meanD.parquet")
        self.assertEqual(_dataset_name("hyena_dna_meanG"), "dataset_hyena_dna_meanG.parquet")


if __name__ == "__main__":
    unittest.main()
