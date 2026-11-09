"""Enformer feature extraction for supervised sequence-to-function comparison.

Input is exactly one canonical-TSS window (196,608 bp, gene orientation, TSS at
index 98,304); anything else is refused rather than cropped or padded. One
forward pass gives every readout (``model_registry.ENFORMER_READOUTS``):

- ``trunk_global``: the mean over all 896 output bins. Enformer crops its trunk
  to ``target_length=896`` bins of 128 bp (enformer-pytorch ``crop_final``,
  checked Sept 29, 2026), so this averages the central 114,688 bp; the outer
  40,960 bp on each side are context the model attends to, not averaged bins.
  "Whole window" in the paper must mean the whole output window (G11).
- ``trunk_center``: the central ``center_bins`` (16 x 128 = 2,048 bp around
  the TSS), the E5 counterpart to TSS-Anchored.
- ``tracks_center``: the human head over those central bins.
Per-gene caches carry a meta record and are refused if built differently (G19);
``load_enformer_features`` reads a finished cache (possibly another machine's)
without the model.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from data_loader.cache_meta import (RUNTIME_KEYS, StaleCache, read_meta, read_npz, runtime_stamp,
                                    sha256_text, write_npz)
from data_loader.model_registry import ENFORMER_MODEL, ENFORMER_REVISION

MODEL_NAME = ENFORMER_MODEL
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
    """Map ACGTN (other IUPAC codes -> N) to Enformer's integer input format.

    The window must already be exactly ``length``: a silent crop or symmetric
    pad would move the TSS off the centre bins (G4).
    """
    if len(seq) != length:
        raise ValueError(f"Enformer input must be exactly {length} bp, got {len(seq)}")
    return _BASE_TO_INDEX[np.frombuffer(seq.upper().encode("ascii"), dtype=np.uint8)]


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
    model = from_pretrained(MODEL_NAME, revision=ENFORMER_REVISION)
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


def enformer_meta(seq: str, center_bins: int, device: str | None,
                  runtime: dict | None = None) -> dict:
    return {"model": MODEL_NAME, "revision": ENFORMER_REVISION, "center_bins": int(center_bins),
            "device": None if device is None else device.split(":")[0],
            **(runtime or dict.fromkeys(RUNTIME_KEYS)), "input_sha256": sha256_text(seq)}


def embed_all_enformer(
    windows: dict[str, str],
    cache_dir: str | Path,
    device: str | None = None,
    center_bins: int = 16,
) -> dict[str, dict[str, np.ndarray]]:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    device = device or _auto_device()
    runtime = runtime_stamp(device)

    out: dict[str, dict[str, np.ndarray]] = {}
    pending: list[tuple[str, str, dict]] = []
    for eid, seq in windows.items():
        meta = enformer_meta(seq, center_bins, device, runtime)
        cached = read_npz(cache_dir / f"{eid}.npz", meta)
        if cached is None:
            pending.append((eid, seq, meta))
        else:
            out[eid] = cached
    if not pending:
        return out

    model, device = load_model(device)
    print(f"  extracting Enformer features for {len(pending)} genes on {device}")
    for eid, seq, meta in tqdm(pending, desc="Enformer features"):
        features = extract_features(seq, model, device, center_bins=center_bins)
        write_npz(cache_dir / f"{eid}.npz", features, meta)
        out[eid] = features
    return out


def load_enformer_features(cache_dir: str | Path, windows: dict[str, str],
                           center_bins: int = 16) -> dict[str, dict[str, np.ndarray]]:
    """Every gene's cached features, each checked against its current window.

    For building datasets from a finished cache, e.g. one extracted on AWS:
    the device, card and torch build are not compared with this machine, but a
    cache that mixes them is refused. Missing genes raise.
    """
    run_keys = ("device", *RUNTIME_KEYS)
    out: dict[str, dict[str, np.ndarray]] = {}
    missing: list[str] = []
    runs: set = set()
    for eid, seq in windows.items():
        path = Path(cache_dir) / f"{eid}.npz"
        cached = read_npz(path, enformer_meta(seq, center_bins, None), ignore=run_keys)
        if cached is None:
            missing.append(eid)
            continue
        out[eid] = cached
        meta = read_meta(path)
        if meta.get("device_name") is None:
            raise StaleCache(f"{path} is unstamped: no record of the card or torch build it ran on")
        runs.add(tuple(meta.get(k) for k in run_keys))
    if missing:
        raise RuntimeError(f"Enformer: no features in {cache_dir} for {len(missing)} genes: "
                           f"{missing[:5]}{'...' if len(missing) > 5 else ''}")
    if len(runs) > 1:
        raise StaleCache(f"Enformer: features in {cache_dir} mix runs {sorted(runs, key=str)}")
    return out
