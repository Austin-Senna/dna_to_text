"""Run NT-v2 100M multi-species over CDS sequences with chunk-and-mean-pool aggregation.

Parallel to encoder_runner.py (DNABERT-2). Kept as a separate module rather than
a shared generic encoder because the two models differ in tokenisation details,
context length, and load-time quirks — and there are only two of them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
import transformers.pytorch_utils as _tf_pytorch_utils
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, PreTrainedModel
from transformers.models.esm.configuration_esm import EsmConfig

MODEL_NAME = "InstaDeepAI/nucleotide-transformer-v2-100m-multi-species"


# NT-v2's remote code was written against an older transformers API. Two
# compatibility shims are needed to load it under transformers 5.x:
#
#   1. The remote code imports pruning helpers that no longer exist. Pruning is
#      never called during inference, so raise-on-call stubs satisfy the import.
#   2. transformers 5.x expects `self.all_tied_weights_keys` to exist before
#      init_weights runs; the remote model calls init_weights inside __init__
#      before that attribute would be set. Add an empty class-level default.
def _install_transformers_shims() -> None:
    for name in ("find_pruneable_heads_and_indices", "prune_linear_layer"):
        if not hasattr(_tf_pytorch_utils, name):
            def _stub(*args, _name=name, **kwargs):
                raise NotImplementedError(f"{_name} stub: pruning unsupported")
            setattr(_tf_pytorch_utils, name, _stub)
    if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
        PreTrainedModel.all_tied_weights_keys = {}
    if not hasattr(PreTrainedModel, "get_head_mask"):
        def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is None:
                return [None] * num_hidden_layers
            raise NotImplementedError("get_head_mask shim: non-None head_mask not supported")
        PreTrainedModel.get_head_mask = _get_head_mask


_install_transformers_shims()


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device: str | None = None):
    if device is None:
        device = _auto_device()

    snapshot_dir = Path(snapshot_download(MODEL_NAME))
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, trust_remote_code=True)

    # NT-v2 uses Gated Linear Units in the FFN (intermediate dim doubled), which
    # the stock transformers EsmModel does not support. The repo ships custom
    # code wired up via auto_map for AutoModelForMaskedLM only, so we load that
    # and take its .esm backbone. The saved config also predates a couple of
    # fields newer transformers expects on EsmConfig (is_decoder,
    # add_cross_attention); fill those in.
    #
    # from_config + manual state-dict load bypasses from_pretrained's newer
    # checks (all_tied_weights_keys etc.) that the remote code hasn't been
    # updated for. strict=False absorbs the MLM head keys we don't need.
    config = AutoConfig.from_pretrained(snapshot_dir, trust_remote_code=True)
    for key, value in EsmConfig().to_dict().items():
        if not hasattr(config, key):
            setattr(config, key, value)
    masked_lm = AutoModelForMaskedLM.from_config(config, trust_remote_code=True)
    state_dict = torch.load(
        snapshot_dir / "pytorch_model.bin", map_location="cpu", weights_only=True
    )
    masked_lm.load_state_dict(state_dict, strict=False)

    model = masked_lm.esm if hasattr(masked_lm, "esm") else masked_lm.base_model
    model.to(device).eval()
    return model, tokenizer, device


def _chunk_ids(ids: list[int], max_tokens: int, stride: int) -> list[list[int]]:
    if len(ids) <= max_tokens:
        return [ids]
    step = max_tokens - stride
    chunks = []
    for start in range(0, len(ids), step):
        chunk = ids[start : start + max_tokens]
        if not chunk:
            break
        chunks.append(chunk)
        if start + max_tokens >= len(ids):
            break
    return chunks


@torch.inference_mode()
def embed_sequence(
    seq: str,
    model,
    tokenizer,
    device: str,
    max_tokens: int = 1000,
    stride: int = 64,
) -> np.ndarray:
    """Tokenize (6-mer non-overlapping), window into chunks, mean-pool tokens, mean across chunks."""
    enc = tokenizer(seq, add_special_tokens=False, return_tensors=None)
    ids = enc["input_ids"]
    chunks = _chunk_ids(ids, max_tokens, stride)

    chunk_vecs = []
    for chunk in chunks:
        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out.last_hidden_state
        vec = hidden.mean(dim=1).squeeze(0).float().cpu().numpy()
        chunk_vecs.append(vec)

    return np.mean(np.stack(chunk_vecs, axis=0), axis=0)


def embed_all(
    cds: dict[str, str],
    cache_dir: str | Path,
    device: str | None = None,
) -> dict[str, np.ndarray]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, np.ndarray] = {}
    pending: list[tuple[str, str]] = []
    for eid, seq in cds.items():
        cache_file = cache_dir / f"{eid}.npy"
        if cache_file.exists():
            out[eid] = np.load(cache_file)
        else:
            pending.append((eid, seq))

    if not pending:
        return out

    model, tokenizer, device = load_model(device)
    print(f"  encoding pending sequences: {len(pending)} on {device}")
    for eid, seq in tqdm(pending, desc="NT-v2 embed"):
        vec = embed_sequence(seq, model, tokenizer, device)
        np.save(cache_dir / f"{eid}.npy", vec)
        out[eid] = vec
    return out
