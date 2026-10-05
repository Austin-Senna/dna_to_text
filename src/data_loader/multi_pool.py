"""Per-chunk reductions for the Phase 4b pooling sweep.

Forward pass once per chunk; capture up to four per-chunk reductions in the
same pass so disk + compute are amortised across all pooling variants:

    mean : per-dim mean over content tokens (excludes special tokens)
    special_mean : per-dim mean over every model token (includes specials)
    max  : per-dim max  over content tokens (excludes special tokens)
    cls  : the model's CLS-token representation (position 0)

Boundary tokens come from the encoder spec, not from whatever the tokenizer
declares: with ``boundary_tokens`` each chunk is wrapped in the tokenizer's
CLS and, where the tokenizer has one, SEP (position 0 IS the trained CLS
representation; content is 1..-2, or 1..end for NT-v2, whose tokenizer has no SEP);
without them (HyenaDNA) the model sees DNA tokens only, and only ``mean`` and
``max`` are stored (``special_mean`` would equal ``mean``, and there is no CLS).

Output per gene: an .npz of (n_chunks, d) arrays plus a ``meta`` record
(``cache_meta``); a file built from other inputs, code, model, card or torch
build is refused.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from tqdm import tqdm

from data_loader.cache_meta import (RUNTIME_KEYS, StaleCache, read_meta, read_npz, runtime_stamp,
                                    sha256_text, write_npz)

REDUCTION_KEYS = ("mean", "special_mean", "max", "cls")


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
    *,
    boundary_tokens: bool,
) -> dict[str, np.ndarray]:
    """Tokenise the sequence, chunk into content windows, run forward per chunk
    (wrapped in CLS/SEP only when ``boundary_tokens``), and return per-chunk
    reductions: {"mean", "max"} plus "special_mean" and "cls" with boundary tokens.
    """
    cls_id = sep_id = None
    if boundary_tokens:
        cls_id = tokenizer.cls_token_id
        if cls_id is None:
            cls_id = tokenizer.bos_token_id
        sep_id = tokenizer.sep_token_id
        if sep_id is None:
            sep_id = tokenizer.eos_token_id
        if cls_id is None:
            raise ValueError("boundary_tokens=True but the tokenizer has no CLS/BOS token")
    has_cls = cls_id is not None
    has_sep = sep_id is not None

    enc = tokenizer(seq, add_special_tokens=False, return_tensors=None)
    content_ids = enc["input_ids"]
    chunks = _chunk_ids(content_ids, max_content_tokens, stride)

    mean_per_chunk: list[np.ndarray] = []
    special_mean_per_chunk: list[np.ndarray] = []
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
        special_mean_vec = hidden.mean(dim=1).squeeze(0)
        mean_vec = content.mean(dim=1).squeeze(0)
        max_vec = content.max(dim=1).values.squeeze(0)

        mean_per_chunk.append(mean_vec.float().cpu().numpy())
        special_mean_per_chunk.append(special_mean_vec.float().cpu().numpy())
        max_per_chunk.append(max_vec.float().cpu().numpy())
        if cls_vec is not None:
            cls_per_chunk.append(cls_vec.float().cpu().numpy())

    reductions = {
        "mean": np.stack(mean_per_chunk, axis=0),
        "max":  np.stack(max_per_chunk, axis=0),
    }
    if boundary_tokens:
        reductions["special_mean"] = np.stack(special_mean_per_chunk, axis=0)
    if cls_per_chunk:
        reductions["cls"] = np.stack(cls_per_chunk, axis=0)
    return reductions


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def extraction_meta(spec, seq: str, device: str | None, runtime: dict | None = None) -> dict:
    """What one gene's reductions depend on: the encoder, its revision and
    chunking, the device, the card and torch build (``runtime_stamp``), and the
    exact input sequence (G19)."""
    return {"encoder": spec.name, "model": spec.model_name, "revision": spec.revision,
            "boundary_tokens": bool(spec.boundary_tokens),
            "max_content_tokens": int(spec.max_content_tokens), "stride": int(spec.stride),
            "device": None if device is None else device.split(":")[0],
            **(runtime or dict.fromkeys(RUNTIME_KEYS)),
            "input_sha256": sha256_text(seq)}


def embed_all_multi_pool(
    seqs: dict[str, str],
    load_model_fn: Callable,
    cache_dir: str | Path,
    spec,
    device: str | None = None,
    desc: str = "multi-pool embed",
    collect: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    """Run multi-pool extraction over every sequence, caching one .npz per gene.

    ``spec`` is the encoder's ``EncoderSpec`` (chunking, boundary tokens,
    revision). A cached file is reused only if its meta matches exactly; any
    other file raises ``StaleCache`` (G19). `load_model_fn` is the encoder's
    `load_model(device)` returning (model, tokenizer, device), loaded only if
    something is pending.

    `collect=False` skips retaining the per-gene arrays in the returned dict
    (values become None) so long-window runs do not accumulate GB of reductions
    in RAM; the per-gene .npz cache is still written. `len(out)` stays correct.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = device or _auto_device()  # resolved first: the device is part of the meta
    runtime = runtime_stamp(device)

    out: dict[str, dict[str, np.ndarray]] = {}
    pending: list[tuple[str, str, dict]] = []
    for eid, seq in seqs.items():
        meta = extraction_meta(spec, seq, device, runtime)
        cached = read_npz(cache_dir / f"{eid}.npz", meta)
        if cached is None:
            pending.append((eid, seq, meta))
        else:
            out[eid] = cached if collect else None

    if not pending:
        return out

    model, tokenizer, device = load_model_fn(device)
    print(f"  encoding pending sequences: {len(pending)} on {device}")
    for eid, seq, meta in tqdm(pending, desc=desc):
        red = embed_sequence_multi_pool(
            seq, model, tokenizer, device,
            max_content_tokens=spec.max_content_tokens, stride=spec.stride,
            boundary_tokens=spec.boundary_tokens,
        )
        write_npz(cache_dir / f"{eid}.npz", red, meta)
        out[eid] = red if collect else None
    return out


def load_reductions(cache_dir: str | Path, spec, seqs: dict[str, str]) -> dict[str, dict[str, np.ndarray]]:
    """Every gene's cached reductions, each checked against its current input.

    Pool builders read through this, so a parquet can only be built from
    reductions of these exact sequences (windows from the manifest, CDS from
    the sequence cache) and this encoder revision (G19), all from one device,
    card and torch build, which may be another machine's. Missing genes raise.
    """
    run_keys = ("device", *RUNTIME_KEYS)
    out: dict[str, dict[str, np.ndarray]] = {}
    missing: list[str] = []
    devices: set = set()
    for eid, seq in seqs.items():
        path = Path(cache_dir) / f"{eid}.npz"
        cached = read_npz(path, extraction_meta(spec, seq, None), ignore=run_keys)
        if cached is None:
            missing.append(eid)
            continue
        out[eid] = cached
        meta = read_meta(path)
        if meta.get("device_name") is None:
            raise StaleCache(f"{path} is unstamped: no record of the card or torch build it ran on")
        devices.add(tuple(meta.get(k) for k in run_keys))
    if missing:
        raise RuntimeError(f"{spec.name}: no reductions in {cache_dir} for {len(missing)} genes: "
                           f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    if len(devices) > 1:
        raise StaleCache(f"{spec.name}: reductions in {cache_dir} mix runs {sorted(devices, key=str)}")
    return out
