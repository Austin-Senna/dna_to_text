"""Central registry for encoder expansion experiments.

Each spec pins the model revision the features were extracted with (G20) and
says whether the encoder gets boundary tokens. HyenaDNA gets DNA only: it was
never trained with a CLS token, and as a causal model a CLS at position 0 sits
in the receptive field of every position (the constant Mean-CLS vector, G3), so
it has no clsmean or specialmean pools and the CDS grid has 22 configs.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from data_loader.pooling_aggregator import POOLING_VARIANTS

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "data"


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    display_name: str
    model_name: str
    model_kind: str
    cache_name: str
    dataset_stem: str
    loader_module: str
    max_content_tokens: int
    stride: int
    revision: str                  # full hub commit SHA (G20)
    boundary_tokens: bool = True   # CLS/SEP per chunk as the model was trained
    pooling_variants: tuple[str, ...] = POOLING_VARIANTS

    @property
    def base_dataset_path(self) -> Path:
        return DATA / f"dataset_{self.dataset_stem}.parquet"

    @property
    def chunk_dir(self) -> Path:
        """CDS per-chunk reductions with meta records (G19); the May caches
        without meta live in ``chunk_reductions_<name>`` and are not read."""
        return DATA / f"chunk_reductions_v2_{self.cache_name}"

    @property
    def tss_chunk_dir(self) -> Path:
        """TSS per-chunk reductions over the canonical-TSS windows (Phase 1B)."""
        return DATA / f"tss_chunk_reductions_v2_{self.cache_name}"

    def variant_dataset_path(self, variant: str) -> Path:
        return DATA / f"dataset_{self.dataset_stem}_{variant}.parquet"


ENCODER_SPECS: dict[str, EncoderSpec] = {
    "dnabert2": EncoderSpec(
        name="dnabert2",
        display_name="DNABERT-2",
        model_name="zhihan1996/DNABERT-2-117M",
        model_kind="self_supervised_encoder",
        cache_name="dnabert2",
        dataset_stem="dnabert2",
        loader_module="data_loader.encoder_runner",
        max_content_tokens=510,
        stride=64,
        revision="7bce263b15377fc15361f52cfab88f8b586abda0",
    ),
    "nt_v2": EncoderSpec(
        name="nt_v2",
        display_name="NT-v2 100M multi-species",
        model_name="InstaDeepAI/nucleotide-transformer-v2-100m-multi-species",
        model_kind="self_supervised_encoder",
        cache_name="nt_v2",
        dataset_stem="nt_v2",
        loader_module="data_loader.nt_v2_encoder",
        max_content_tokens=998,
        stride=64,
        revision="f34324c6fde36a4f635f0f1f06cac5d25acd6798",
    ),
    "gena_lm": EncoderSpec(
        name="gena_lm",
        display_name="GENA-LM base",
        model_name="AIRI-Institute/gena-lm-bert-base-t2t",
        model_kind="self_supervised_encoder",
        cache_name="gena_lm",
        dataset_stem="gena_lm",
        loader_module="data_loader.gena_lm_encoder",
        max_content_tokens=510,
        stride=64,
        revision="4f1352bd4e820f1dba341047f54ad7795e083bf2",
    ),
    "hyena_dna": EncoderSpec(
        name="hyena_dna",
        display_name="HyenaDNA large",
        model_name="LongSafari/hyenadna-large-1m-seqlen-hf",
        model_kind="self_supervised_encoder",
        cache_name="hyena_dna",
        dataset_stem="hyena_dna",
        loader_module="data_loader.hyena_dna_encoder",
        max_content_tokens=8192,
        stride=512,
        revision="0a629abf9c7f85b4ec9aa6a1aefa3adcf1907446",
        boundary_tokens=False,
        pooling_variants=("meanmean", "maxmean", "meanD", "meanG"),
    ),
}

# Enformer is not a chunked encoder, but its revision is pinned the same way.
# Hub main has been this commit since June 2022, so the May features used it.
ENFORMER_MODEL = "EleutherAI/enformer-official-rough"
ENFORMER_REVISION = "affe5713ae9017460706a44108289b13c5fee16c"
# One forward pass gives both readouts: the whole output window is the standard
# Enformer feature; the central 16 bins are the E5 counterpart to TSS-Anchored.
ENFORMER_READOUTS = {"whole": "trunk_global", "e5_centre": "trunk_center"}

# The TSS whole-window grid never had specialmean (the TSS pool builder predates
# it): 5 pools, 4 for HyenaDNA. centermean is a TSS-only E5 template, not a pool.
TSS_EXCLUDED_POOLS = ("specialmean",)

BASE_DATASET_ALIASES = {
    "dnabert2": DATA / "dataset.parquet",
    "nt_v2": DATA / "dataset_nt_v2.parquet",
}


def main_encoder_names() -> tuple[str, ...]:
    return tuple(ENCODER_SPECS)


def get_encoder_spec(name: str) -> EncoderSpec:
    try:
        return ENCODER_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"unknown encoder: {name!r}") from exc


def encoder_pools(encoder: str, ctx: str = "CDS") -> tuple[str, ...]:
    """The pools that compete in an encoder's validation pick, CDS or TSS."""
    pools = get_encoder_spec(encoder).pooling_variants
    if ctx == "CDS":
        return pools
    if ctx == "TSS":
        return tuple(p for p in pools if p not in TSS_EXCLUDED_POOLS)
    raise ValueError(f"ctx must be 'CDS' or 'TSS', got {ctx!r}")


def dataset_paths(include_variants: bool = True) -> dict[str, Path]:
    paths = dict(BASE_DATASET_ALIASES)
    for spec in ENCODER_SPECS.values():
        paths.setdefault(spec.name, spec.base_dataset_path)
        if include_variants:
            for variant in spec.pooling_variants:
                paths[f"{spec.name}_{variant}"] = spec.variant_dataset_path(variant)
    return paths


def family5_feature_sources() -> tuple[str, ...]:
    sources: list[str] = []
    for spec in ENCODER_SPECS.values():
        sources.append(spec.name)
        sources.extend(f"{spec.name}_{variant}" for variant in spec.pooling_variants)
    sources.extend(["kmer", "shuffled"])
    return tuple(sources)
