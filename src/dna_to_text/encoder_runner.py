"""Run DNABERT-2 over CDS sequences with chunk-and-mean-pool aggregation."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "zhihan1996/DNABERT-2-117M"


def load_model(device: str | None = None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)
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
    """Tokenize, window into chunks, mean-pool tokens per chunk, mean across chunks."""
    enc = tokenizer(seq, add_special_tokens=False, return_tensors=None)
    ids = enc["input_ids"]
    chunks = _chunk_ids(ids, max_tokens, stride)

    chunk_vecs = []
    for chunk in chunks:
        input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
        # mean over token dim
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
    for eid, seq in tqdm(pending, desc="DNABERT-2 embed"):
        vec = embed_sequence(seq, model, tokenizer, device)
        np.save(cache_dir / f"{eid}.npy", vec)
        out[eid] = vec
    return out
