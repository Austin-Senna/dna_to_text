"""Enformer feature extraction for supervised sequence-to-function comparison."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

MODEL_NAME = "EleutherAI/enformer-official-rough"
ENFORMER_LENGTH = 196_608

_BASE_TO_INDEX = np.full(256, 4, dtype=np.int64)
for _base, _idx in {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}.items():
    _BASE_TO_INDEX[ord(_base)] = _idx
    _BASE_TO_INDEX[ord(_base.lower())] = _idx
del _base, _idx


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sequence_to_indices(seq: str, length: int = ENFORMER_LENGTH) -> np.ndarray:
    """Map ACGTN to Enformer's integer input format, center-cropping or padding."""
    seq = seq.upper()
    if len(seq) > length:
        start = (len(seq) - length) // 2
        seq = seq[start : start + length]
    elif len(seq) < length:
        pad_left = (length - len(seq)) // 2
        pad_right = length - len(seq) - pad_left
        seq = ("N" * pad_left) + seq + ("N" * pad_right)
    return _BASE_TO_INDEX[np.frombuffer(seq.encode("ascii"), dtype=np.uint8)]


def _center_slice(arr, center_bins: int):
    n = arr.shape[0]
    start = max(0, (n - center_bins) // 2)
    end = min(n, start + center_bins)
    return arr[start:end]


def load_model(device: str | None = None):
    if device is None:
        device = _auto_device()
    try:
        from enformer_pytorch import from_pretrained
        from enformer_pytorch.modeling_enformer import Enformer
    except ImportError as exc:
        raise ImportError(
            "Enformer extraction requires the optional dependency "
            "`enformer-pytorch`. Install it with `uv pip install enformer-pytorch`."
        ) from exc
    # enformer-pytorch 0.8.x does not call PreTrainedModel.post_init(), but
    # Transformers 5 expects this map to exist while loading checkpoint shards.
    if not hasattr(Enformer, "all_tied_weights_keys"):
        Enformer.all_tied_weights_keys = {}
    model = from_pretrained(MODEL_NAME)
    model.to(device).eval()
    return model, device


@torch.inference_mode()
def extract_features(
    seq: str,
    model,
    device: str,
    center_bins: int = 16,
) -> dict[str, np.ndarray]:
    idx = sequence_to_indices(seq)
    seq_tensor = torch.tensor(idx, dtype=torch.long, device=device).unsqueeze(0)
    output, embeddings = model(seq_tensor, return_embeddings=True)
    emb = embeddings.squeeze(0).float().cpu().numpy()

    human = output["human"] if isinstance(output, dict) else output
    tracks = human.squeeze(0).float().cpu().numpy()

    return {
        "trunk_global": emb.mean(axis=0).astype(np.float32),
        "trunk_center": _center_slice(emb, center_bins).mean(axis=0).astype(np.float32),
        "tracks_center": _center_slice(tracks, center_bins).mean(axis=0).astype(np.float32),
    }


def embed_all_enformer(
    windows: dict[str, str],
    cache_dir: str | Path,
    device: str | None = None,
    center_bins: int = 16,
) -> dict[str, dict[str, np.ndarray]]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, dict[str, np.ndarray]] = {}
    pending: list[tuple[str, str]] = []
    for eid, seq in windows.items():
        cache_file = cache_dir / f"{eid}.npz"
        if cache_file.exists():
            with np.load(cache_file) as data:
                out[eid] = {k: data[k] for k in data.files}
        else:
            pending.append((eid, seq))
    if not pending:
        return out

    model, device = load_model(device)
    print(f"  extracting Enformer features for {len(pending)} genes on {device}")
    for eid, seq in tqdm(pending, desc="Enformer features"):
        features = extract_features(seq, model, device, center_bins=center_bins)
        np.savez(cache_dir / f"{eid}.npz", **features)
        out[eid] = features
    return out
