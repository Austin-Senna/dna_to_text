"""Run DNABERT-2 over CDS sequences with chunk-and-mean-pool aggregation."""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, BertConfig as HFBertConfig
from transformers import dynamic_module_utils

MODEL_NAME = "zhihan1996/DNABERT-2-117M"
MODEL_REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
OPTIONAL_REMOTE_IMPORTS = {"flash_attn_triton"}


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@contextmanager
def _ignore_optional_remote_imports(names: set[str]):
    """Skip optional relative imports that old remote model code marks as hard deps."""
    original = dynamic_module_utils.get_relative_imports

    def patched(module_file: str | Path) -> list[str]:
        imports = original(module_file)
        return [name for name in imports if name not in names]

    dynamic_module_utils.get_relative_imports = patched
    try:
        yield
    finally:
        dynamic_module_utils.get_relative_imports = original


def load_model(device: str | None = None):
    if device is None:
        device = _auto_device()

    snapshot_dir = Path(snapshot_download(MODEL_NAME, revision=MODEL_REVISION))

    with _ignore_optional_remote_imports(OPTIONAL_REMOTE_IMPORTS):
        config = AutoConfig.from_pretrained(
            snapshot_dir,
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            snapshot_dir,
            trust_remote_code=True,
        )
        defaults = HFBertConfig().to_dict()
        for key, value in defaults.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        if getattr(config, "pad_token_id", None) is None:
            config.pad_token_id = tokenizer.pad_token_id
        masked_lm = AutoModelForMaskedLM.from_config(
            config,
            trust_remote_code=True,
        )
    weights_path = snapshot_dir / "pytorch_model.bin"
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    masked_lm.load_state_dict(state_dict)
    model = masked_lm.bert

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
    print(f"  encoding pending sequences: {len(pending)} on {device}")
    for eid, seq in tqdm(pending, desc="DNABERT-2 embed"):
        vec = embed_sequence(seq, model, tokenizer, device)
        np.save(cache_dir / f"{eid}.npy", vec)
        out[eid] = vec
    return out
