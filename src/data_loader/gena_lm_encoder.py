"""GENA-LM base loader for frozen CDS embedding extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
import transformers.pytorch_utils as _tf_pytorch_utils
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_utils import ModuleUtilsMixin

MODEL_NAME = "AIRI-Institute/gena-lm-bert-base-t2t"


def _install_transformers_shims() -> None:
    """Support GENA-LM remote code written for older transformers releases."""
    for name in ("find_pruneable_heads_and_indices", "prune_linear_layer"):
        if not hasattr(_tf_pytorch_utils, name):
            def _stub(*args, _name=name, **kwargs):
                raise NotImplementedError(f"{_name} shim: pruning unsupported")
            setattr(_tf_pytorch_utils, name, _stub)

    original = ModuleUtilsMixin.get_extended_attention_mask
    if not getattr(original, "_dna_to_text_legacy_device_shim", False):
        def _legacy_get_extended_attention_mask(self, attention_mask, input_shape, dtype=None):
            if isinstance(dtype, torch.device):
                dtype = None
            return original(self, attention_mask, input_shape, dtype=dtype)

        _legacy_get_extended_attention_mask._dna_to_text_legacy_device_shim = True
        ModuleUtilsMixin.get_extended_attention_mask = _legacy_get_extended_attention_mask

    if not hasattr(ModuleUtilsMixin, "get_head_mask"):
        def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
            if head_mask is None:
                return [None] * num_hidden_layers
            raise NotImplementedError("get_head_mask shim: non-None head_mask not supported")
        ModuleUtilsMixin.get_head_mask = _get_head_mask


def _select_backbone(model):
    return model.bert if hasattr(model, "bert") else model


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device: str | None = None):
    if device is None:
        device = _auto_device()
    _install_transformers_shims()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = _select_backbone(model)
    # GENA-LM's custom modeling registers `token_type_ids` as a non-persistent
    # buffer; on some PyTorch builds it loads with uninitialised memory values
    # (e.g. 299667744) which crash the token_type_embeddings lookup with
    # `IndexError: index out of range in self` (type_vocab_size=2). Zero it
    # explicitly so the embedding auto-generation path works again.
    if hasattr(model.embeddings, "token_type_ids"):
        model.embeddings.token_type_ids = torch.zeros_like(
            model.embeddings.token_type_ids
        )
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
    max_tokens: int = 512,
    stride: int = 64,
) -> np.ndarray:
    enc = tokenizer(seq, add_special_tokens=True, return_tensors=None)
    ids = enc["input_ids"]
    chunks = _chunk_ids(ids, max_tokens, stride)

    chunk_vecs = []
    for chunk in chunks:
        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
        chunk_vecs.append(hidden.mean(dim=1).squeeze(0).float().cpu().numpy())

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
    for eid, seq in tqdm(pending, desc="GENA-LM embed"):
        vec = embed_sequence(seq, model, tokenizer, device)
        np.save(cache_dir / f"{eid}.npy", vec)
        out[eid] = vec
    return out
