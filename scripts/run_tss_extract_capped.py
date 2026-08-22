"""VRAM-capped launcher for run_tss_multi_pool_extract.py (MLCB E5, camera-ready).

Caps this process to a fraction of GPU memory BEFORE the encoder loads, so a
memory spike raises a caught Python OOM (and the run resumes from the per-gene
.npz cache) instead of taking down the driver / WSL. All args after the script
name are forwarded to run_tss_multi_pool_extract.py unchanged.

Usage:
    uv run scripts/run_tss_extract_capped.py --encoder dnabert2 --device cuda
    MEM_FRACTION=0.5 uv run scripts/run_tss_extract_capped.py --encoder hyena_dna --device cuda
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path

os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:128"
)

import torch

FRACTION = float(os.environ.get("MEM_FRACTION", "0.55"))
if torch.cuda.is_available():
    torch.cuda.set_per_process_memory_fraction(FRACTION, 0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[capped] GPU cap = {FRACTION:.2f} x {total_gb:.1f} GB = {FRACTION * total_gb:.1f} GB")

runpy.run_path(
    str(Path(__file__).resolve().parent / "run_tss_multi_pool_extract.py"),
    run_name="__main__",
)
