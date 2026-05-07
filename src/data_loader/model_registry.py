"""Central registry for encoder expansion experiments."""
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
    pooling_variants: tuple[str, ...] = POOLING_VARIANTS

    @property
    def base_dataset_path(self) -> Path:
        return DATA / f"dataset_{self.dataset_stem}.parquet"

    @property
    def chunk_dir(self) -> Path:
        return DATA / f"chunk_reductions_{self.cache_name}"

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
    ),
}

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
