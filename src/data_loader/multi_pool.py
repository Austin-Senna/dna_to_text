"""Per-chunk reductions for the Phase 4b pooling sweep.

Forward pass once per chunk; capture three per-chunk reductions in the same
pass so disk + compute are amortised across all five pooling variants:

    mean : per-dim mean over content tokens (excludes special tokens)
    max  : per-dim max  over content tokens (excludes special tokens)
    cls  : the model's CLS-token representation (position 0)

Tokenisation includes special tokens so position 0 IS the trained CLS
representation. Content tokens are positions 1..-2 (excluding CLS at 0
and SEP at -1).

Output per gene: an .npz with three (n_chunks, d) arrays.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm


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
def embed_sequence_multi_pool(
    seq: str,
    model,
    tokenizer,
    device: str,
    max_content_tokens: int,
    stride: int,
) -> dict[str, np.ndarray]:
    """Tokenise the sequence, chunk into content windows, run forward with CLS+SEP
    wrapped per chunk, and return per-chunk reductions.

    Returns: {"mean": (n_chunks, d), "max": (n_chunks, d), "cls": (n_chunks, d)}.
    """
    # CLS/BOS at start is optional. BERT-family tokenizers expose a trained
    # summary position, but some long-context DNA tokenizers do not. If absent,
    # we still cache mean/max reductions and downstream code skips clsmean.
    cls_id = tokenizer.cls_token_id
    if cls_id is None:
        cls_id = tokenizer.bos_token_id
    sep_id = tokenizer.sep_token_id
    if sep_id is None:
        sep_id = tokenizer.eos_token_id
    has_cls = cls_id is not None
    has_sep = sep_id is not None

    enc = tokenizer(seq, add_special_tokens=False, return_tensors=None)
    content_ids = enc["input_ids"]
    chunks = _chunk_ids(content_ids, max_content_tokens, stride)

    mean_per_chunk: list[np.ndarray] = []
    max_per_chunk: list[np.ndarray] = []
    cls_per_chunk: list[np.ndarray] = []

    for chunk_content in chunks:
        chunk_ids = ([cls_id] if has_cls else []) + chunk_content + ([sep_id] if has_sep else [])
        input_ids = torch.tensor([chunk_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        try:
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        except TypeError:
            out = model(input_ids=input_ids)
        hidden = out[0] if isinstance(out, tuple) else out.last_hidden_state
        # hidden: (1, n_tokens, d). Position 0 is CLS; position -1 is SEP only
        # if we appended one. Slice content accordingly.
        cls_vec = hidden[:, 0, :].squeeze(0) if has_cls else None
        content_start = 1 if has_cls else 0
        content_end = -1 if has_sep else hidden.shape[1]
        content = hidden[:, content_start:content_end, :]
        mean_vec = content.mean(dim=1).squeeze(0)
        max_vec = content.max(dim=1).values.squeeze(0)

        mean_per_chunk.append(mean_vec.float().cpu().numpy())
        max_per_chunk.append(max_vec.float().cpu().numpy())
        if cls_vec is not None:
            cls_per_chunk.append(cls_vec.float().cpu().numpy())

    reductions = {
        "mean": np.stack(mean_per_chunk, axis=0),
        "max":  np.stack(max_per_chunk, axis=0),
    }
    if cls_per_chunk:
        reductions["cls"] = np.stack(cls_per_chunk, axis=0)
    return reductions


def embed_all_multi_pool(
    cds: dict[str, str],
    load_model_fn: Callable,
    cache_dir: str | Path,
    max_content_tokens: int,
    stride: int,
    device: str | None = None,
    desc: str = "multi-pool embed",
) -> dict[str, dict[str, np.ndarray]]:
    """Run multi-pool extraction over all CDS, caching one .npz per gene.

    `load_model_fn` is the encoder's existing `load_model(device)` returning
    (model, tokenizer, device). Loaded only if there are pending sequences.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, dict[str, np.ndarray]] = {}
    pending: list[tuple[str, str]] = []
    for eid, seq in cds.items():
        cache_file = cache_dir / f"{eid}.npz"
        if cache_file.exists():
            with np.load(cache_file) as data:
                out[eid] = {k: data[k] for k in ("mean", "max", "cls") if k in data.files}
        else:
            pending.append((eid, seq))

    if not pending:
        return out

    model, tokenizer, device = load_model_fn(device)
    print(f"  encoding pending sequences: {len(pending)} on {device}")
    for eid, seq in tqdm(pending, desc=desc):
        red = embed_sequence_multi_pool(
            seq, model, tokenizer, device,
            max_content_tokens=max_content_tokens, stride=stride,
        )
        np.savez(cache_dir / f"{eid}.npz", **red)
        out[eid] = red
    return out
