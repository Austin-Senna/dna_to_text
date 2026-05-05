"""Caduceus-PS loader for frozen CDS embedding extraction."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

MODEL_NAME = "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(device: str | None = None):
    if device is None:
        device = _auto_device()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
    except Exception:
        model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME, trust_remote_code=True)
        for attr in ("caduceus", "backbone", "base_model"):
            if hasattr(model, attr):
                model = getattr(model, attr)
                break
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


def _hidden_from_output(out):
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state
    if isinstance(out, tuple):
        return out[0]
    raise TypeError(f"cannot find hidden states in Caduceus output {type(out).__name__}")


@torch.inference_mode()
def embed_sequence(
    seq: str,
    model,
    tokenizer,
    device: str,
    max_tokens: int = 8192,
    stride: int = 512,
) -> np.ndarray:
    enc = tokenizer(seq, add_special_tokens=True, return_tensors=None)
    ids = enc["input_ids"]
    chunks = _chunk_ids(ids, max_tokens, stride)

    chunk_vecs = []
    for chunk in chunks:
        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        except TypeError:
            out = model(input_ids=input_ids)
        hidden = _hidden_from_output(out)
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
    for eid, seq in tqdm(pending, desc="Caduceus-PS embed"):
        vec = embed_sequence(seq, model, tokenizer, device)
        np.save(cache_dir / f"{eid}.npy", vec)
        out[eid] = vec
    return out
