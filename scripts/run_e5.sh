#!/usr/bin/env bash
# E5 (TSS-anchored pooling) reproduction driver — MLCB camera-ready.
#
# Runs the CPU-only steps 2-8 of the E5 chain in order (see the "Scripts &
# reproduction" table in analysis/tss_overlap/center_chunk_finding.md). Each
# script defaults to all four encoders (dnabert2, nt_v2, gena_lm, hyena_dna)
# and both splits; none writes tracked data/metrics.json.
#
# PREREQUISITE (step 1, GPU, not run here): the per-chunk reduction cache
# data/tss_chunk_reductions_<enc>/ must exist. Build it (GPU-gated) with:
#   uv run scripts/run_tss_extract_capped.py --encoder <enc> ...
# hyena_dna is the WSL hard-hang risk — run it capped and last.
#
# Usage: bash scripts/run_e5.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== 2/8 center-chunk diagnostic =="
uv run scripts/check_tss_center_chunk.py

echo "== 3/8 build TSS-anchored datasets (--anchor nearestcenter) =="
uv run scripts/build_tss_anchored_datasets.py

echo "== 4/8 probe family5 on anchored (both splits) =="
uv run scripts/probe_tss_anchored.py

echo "== 5/8 build chunk-matched composition baselines =="
uv run scripts/build_tss_composition_baseline.py

echo "== 6/8 probe composition baselines (both splits) =="
uv run scripts/probe_tss_composition.py

echo "== 7/8 bootstrap CIs on anchored macro-F1 =="
uv run scripts/bootstrap_tss_anchored.py

echo "== 8/8 paired bootstrap: anchored vs best global pool =="
uv run scripts/paired_tss_anchored.py

echo "== E5 chain complete =="
