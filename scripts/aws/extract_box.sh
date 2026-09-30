#!/usr/bin/env bash
# Phase 3 GPU extraction on one CUDA box (written for an AWS g5, A10G 24 GB).
#
# Run from a clone of this repo at the extraction commit, with the gitignored
# inputs in place: data/sequences/ (checked against data/cds_manifest.tsv),
# data/tss_windows_e115/ (checked against data/tss_windows.tsv), and the Hugging
# Face snapshots at the revisions pinned in src/data_loader/model_registry.py,
# each also recorded as the snapshot's refs/main (the pre-fix code loads main).
# Models load offline, so nothing is fetched mid-run. One process uses the GPU
# at a time. PyTorch's numeric defaults are kept (as in the May runs) and
# recorded in the provenance file. Paths may be relative to the caller.
#
#   extract_box.sh provenance OUT
#       Record the commit, lock file, GPU, driver and torch build in OUT/provenance.json.
#   extract_box.sh pilot OLD_CLONE PILOT_DIR OUT
#       The Phase 3 gate. OLD_CLONE is a separate clone at 095dcf5 with its own
#       venv, data/sequences and data/enformer_windows; PILOT_DIR holds what
#       `compare_extraction_caches.py pick` wrote; OUT must not exist yet.
#       Preflight (pins, input checksums, same python/torch in both venvs),
#       then the old and new code on the pilot genes, the new code twice on the
#       repeat genes, and the same-box checks: Leg B and the repeat, bit-exact.
#       HyenaDNA's new inputs drop CLS/SEP by design, so it is checked for G3
#       and measured against the old code instead.
#   extract_box.sh full OUT [--shutdown]
#       Every production cache into its default *_v2 dir, each checked by census.
#       --shutdown powers the box off when the script exits, pass, fail or
#       Ctrl-C (cancel a pending power-off with `sudo shutdown -c`).
set -euo pipefail

usage() { awk 'NR == 1 { next } /^#/ { sub(/^# ?/, ""); print; next } { exit }' "$0"; exit 2; }
abspath() { realpath -m -- "$1"; }

cmd=${1:-}; shift || true
case $cmd in  # resolve the caller's relative paths before leaving their directory
  provenance) [[ $# -eq 1 ]] || usage; set -- "$(abspath "$1")" ;;
  pilot) [[ $# -eq 3 ]] || usage; set -- "$(abspath "$1")" "$(abspath "$2")" "$(abspath "$3")" ;;
  full)
    [[ ($# -eq 1 || ($# -eq 2 && $2 == --shutdown)) && $1 != -* ]] || usage
    set -- "$(abspath "$1")" "${@:2}" ;;
  *) usage ;;
esac

cd "$(dirname "$0")/../.."
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
ENCODERS=(dnabert2 nt_v2 gena_lm hyena_dna)
BOUNDARY_ENCODERS=(dnabert2 nt_v2 gena_lm)  # HyenaDNA's new inputs differ by design
ESM_SIZES=(150m 650m)
COMPARE=scripts/compare_extraction_caches.py

run_step() {  # run_step OUT NAME N_GENES CMD... : log, time, and record one step
  local out=$1 name=$2 n=$3; shift 3
  mkdir -p "$out/logs"
  local t0=$SECONDS
  echo "=== $name ($n genes): $*"
  "$@" 2>&1 | tee "$out/logs/$name.log"
  printf '%s\t%s\t%s\n' "$name" "$n" "$((SECONDS - t0))" >> "$out/timings.tsv"
}

provenance() {  # provenance CLONE JSON : the clone's commit and venv, this box's GPU
  local clone=$1 json=$2
  mkdir -p "$(dirname "$json")"
  (cd "$clone" && uv run --frozen python - "$json" <<'PY'
import hashlib, json, platform, subprocess, sys, urllib.request
from datetime import datetime, timezone
from pathlib import Path
import torch
import data_loader

def sh(*cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()

def instance_type():
    try:  # IMDSv2, best effort; absent off EC2
        req = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                     headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
        token = urllib.request.urlopen(req, timeout=2).read().decode()
        req = urllib.request.Request("http://169.254.169.254/latest/meta-data/instance-type",
                                     headers={"X-aws-ec2-metadata-token": token})
        return urllib.request.urlopen(req, timeout=2).read().decode()
    except Exception:
        return None

gpu = sh("nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader").split(", ")
record = {
    "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "git_sha": sh("git", "rev-parse", "HEAD"),
    "git_dirty": bool(sh("git", "status", "--porcelain", "--untracked-files=no")),
    # The venv imports this clone's src/, not another clone's (editable installs are per clone).
    "imports_own_code": Path(data_loader.__file__).resolve().is_relative_to(Path.cwd().resolve()),
    "uv_lock_sha256": hashlib.sha256(open("uv.lock", "rb").read()).hexdigest(),
    "instance_type": instance_type(),
    "gpu": gpu[0], "driver": gpu[1],
    "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
    "float32_matmul_precision": torch.get_float32_matmul_precision(),
    "cudnn_deterministic": torch.backends.cudnn.deterministic,
    "cudnn_benchmark": torch.backends.cudnn.benchmark,
}
json.dump(record, open(sys.argv[1], "w"), indent=1)
print(json.dumps(record, indent=1))
PY
  )
}

preflight() {  # preflight OLD PILOT OUT : nothing runs unless the premise of Leg B holds
  local old=$1 pilot=$2 out=$3
  for f in pilot_genes.tsv pilot_template.parquet repeat_template.parquet \
           pilot_old_inputs.sha256 pilot_new_inputs.sha256; do
    [[ -f $pilot/$f ]] || { echo "missing $pilot/$f" >&2; exit 1; }
  done
  [[ $(git -C "$old" rev-parse HEAD) == 095dcf5* ]] || { echo "$old is not at 095dcf5" >&2; exit 1; }
  # Every input either clone reads for the pilot genes, byte for byte as picked locally;
  # a missing file would make the old code fetch it from Ensembl REST (a later release).
  (cd "$old" && sha256sum --check --quiet --strict "$pilot/pilot_old_inputs.sha256")
  sha256sum --check --quiet --strict "$pilot/pilot_new_inputs.sha256"
  # The old loaders take refs/main; the new ones the pinned revision. Both must be the pin.
  uv run --frozen python - <<'PY'
from pathlib import Path
from huggingface_hub.constants import HF_HUB_CACHE
import hashlib
import torch
from data_loader.model_registry import (ENFORMER_MODEL, ENFORMER_REVISION, ESM2_CHECKPOINT_SHA256,
                                        get_encoder_spec)

pins = {s.model_name: s.revision for s in map(get_encoder_spec, ("dnabert2", "nt_v2", "gena_lm", "hyena_dna"))}
pins[ENFORMER_MODEL] = ENFORMER_REVISION
bad = []
for model, rev in pins.items():
    repo = Path(HF_HUB_CACHE) / f"models--{model.replace('/', '--')}"
    ref = repo / "refs" / "main"
    main = ref.read_text().strip() if ref.exists() else None
    if main != rev or not (repo / "snapshots" / rev).is_dir():
        bad.append(f"{model}: refs/main={main}, pinned={rev}")
for name, sha in ESM2_CHECKPOINT_SHA256.items():  # both clones' ESM-2 steps read this file
    ckpt = Path(torch.hub.get_dir()) / "checkpoints" / f"{name}.pt"
    h = hashlib.sha256()
    with open(ckpt, "rb") if ckpt.exists() else open("/dev/null", "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != sha:
        bad.append(f"{ckpt.name}: {'missing' if not ckpt.exists() else h.hexdigest()}, pinned={sha}")
if bad:
    raise SystemExit("inputs are not the pinned ones:\n  " + "\n  ".join(bad))
print(f"HF cache: refs/main is the pinned revision for all {len(pins)} models; ESM-2 checkpoints pinned")
PY
  provenance "$PWD" "$out/provenance.json"
  provenance "$old" "$out/provenance_old.json"
  uv run --frozen python - "$out/provenance.json" "$out/provenance_old.json" <<'PY'
import json, sys
new, old = (json.load(open(p)) for p in sys.argv[1:])
bad = [k for k in ("python", "torch", "cuda", "cudnn", "uv_lock_sha256") if new[k] != old[k]]
bad += [f"{side} clone has uncommitted tracked changes" for side, p in (("new", new), ("old", old)) if p["git_dirty"]]
bad += [f"{side} venv imports another clone's code" for side, p in (("new", new), ("old", old))
        if not p["imports_own_code"]]
if bad:
    raise SystemExit(f"preflight: {bad}")
print("old and new venvs: same python, torch, CUDA, cuDNN and lock file; both clones clean")
PY
}

compare() {  # compare OUT LABEL A B ARGS... : a failed comparison is recorded, not fatal
  local out=$1 label=$2 a=$3 b=$4; shift 4
  uv run --frozen "$COMPARE" compare "$a" "$b" --label "$label" --csv "$out/compare/$label.csv" "$@" \
    | tee -a "$out/pilot_report.txt" || echo "$label" >> "$out/failed.txt"
}

hyena_checks() {  # G3 on HyenaDNA's new caches, and no boundary-token arrays in them
  local out=$1; shift
  uv run --frozen python - "$@" <<'PY' | tee -a "$out/pilot_report.txt" || echo hyena_checks >> "$out/failed.txt"
import sys
from pathlib import Path
import numpy as np
from linear_trainer.sources import check_not_constant

for d in sys.argv[1:]:
    pooled = []
    for f in sorted(Path(d).glob("*.npz")):
        with np.load(f, allow_pickle=False) as z:
            keys = set(z.files) - {"meta"}
            if keys != {"mean", "max"}:
                raise SystemExit(f"{f}: arrays {sorted(keys)}, expected mean and max only")
            pooled.append(z["mean"].mean(axis=0))
    check_not_constant(np.vstack(pooled), d)
    print(f"[{Path(d).name}] {len(pooled)} genes: mean/max only, meanmean not constant (G3)")
PY
}

pilot() {
  local old=$1 pilot=$2 out=$3
  [[ ! -e $out ]] || { echo "$out exists: a pilot writes into a fresh directory only" >&2; exit 1; }
  mkdir -p "$out/compare"
  preflight "$old" "$pilot" "$out"
  local tpl=$pilot/pilot_template.parquet rep=$pilot/repeat_template.parquet
  local n n_rep
  n=$(($(wc -l < "$pilot/pilot_genes.tsv") - 1))
  n_rep=$(uv run --frozen python -c "import pandas as pd, sys; print(len(pd.read_parquet(sys.argv[1])))" "$rep")

  # Old code path (095dcf5): its CDS script can only write data/chunk_reductions_<enc>.
  for e in "${ENCODERS[@]}"; do
    [[ ! -e $old/data/chunk_reductions_$e ]] || { echo "$old/data/chunk_reductions_$e exists" >&2; exit 1; }
    run_step "$out" "old_cds_$e" "$n" bash -c "cd '$old' && uv run --frozen scripts/run_multi_pool_extract.py \
      --encoder $e --gene-table '$tpl' --device cuda"
    mv "$old/data/chunk_reductions_$e" "$out/old_cds_$e"
    run_step "$out" "old_tss_$e" "$n" bash -c "cd '$old' && uv run --frozen scripts/run_tss_multi_pool_extract.py \
      --encoder $e --template-dataset '$tpl' --window-cache data/enformer_windows \
      --cache-dir '$out/old_tss_$e' --device cuda"
  done
  run_step "$out" old_enformer "$n" bash -c "cd '$old' && uv run --frozen scripts/run_enformer_features.py \
    --template-dataset '$tpl' --window-cache data/enformer_windows --feature-cache '$out/old_enformer' --device cuda"
  # The old script always rewrites its tracked datasets with the pilot's 50 rows; put them back
  # so the throwaway old clone stays clean for a rerun.
  git -C "$old" restore -- data/dataset_enformer_tss_4mer.parquet data/dataset_enformer_trunk_global.parquet \
    data/dataset_enformer_trunk_center.parquet data/dataset_enformer_tracks_center.parquet
  for s in "${ESM_SIZES[@]}"; do
    run_step "$out" "old_esm2_$s" "$n" bash -c "cd '$old' && uv run --frozen scripts/run_esm2.py \
      --size $s --template '$tpl' --out-cache '$out/old_esm2_$s' --device cuda --fp16"
  done

  # New code path (this clone), same genes; ESM-2 in fp16 to match, and fp32 for production timing.
  for e in "${ENCODERS[@]}"; do
    run_step "$out" "new_cds_$e" "$n" uv run --frozen scripts/run_multi_pool_extract.py \
      --encoder "$e" --gene-table "$tpl" --cache-dir "$out/new_cds_$e" --device cuda
    run_step "$out" "new_tss_$e" "$n" uv run --frozen scripts/run_tss_multi_pool_extract.py \
      --encoder "$e" --gene-table "$tpl" --cache-dir "$out/new_tss_$e" --device cuda
  done
  run_step "$out" new_enformer "$n" uv run --frozen scripts/run_enformer_features.py \
    --template-dataset "$tpl" --feature-cache "$out/new_enformer" --device cuda --no-datasets
  for s in "${ESM_SIZES[@]}"; do
    run_step "$out" "new_esm2_${s}_fp16" "$n" uv run --frozen scripts/run_esm2.py \
      --size "$s" --template "$tpl" --out-cache "$out/new_esm2_${s}_fp16" --device cuda --fp16
    run_step "$out" "new_esm2_${s}_fp32" "$n" uv run --frozen scripts/run_esm2.py \
      --size "$s" --template "$tpl" --out-cache "$out/new_esm2_${s}_fp32" --device cuda
  done

  # Repeat: the new code twice on the repeat genes, every path Leg B gates plus fp32 ESM-2.
  for r in 1 2; do
    for e in "${ENCODERS[@]}"; do
      run_step "$out" "rep${r}_cds_$e" "$n_rep" uv run --frozen scripts/run_multi_pool_extract.py \
        --encoder "$e" --gene-table "$rep" --cache-dir "$out/rep${r}_cds_$e" --device cuda
      run_step "$out" "rep${r}_tss_$e" "$n_rep" uv run --frozen scripts/run_tss_multi_pool_extract.py \
        --encoder "$e" --gene-table "$rep" --cache-dir "$out/rep${r}_tss_$e" --device cuda
    done
    run_step "$out" "rep${r}_enformer" "$n_rep" uv run --frozen scripts/run_enformer_features.py \
      --template-dataset "$rep" --feature-cache "$out/rep${r}_enformer" --device cuda --no-datasets
    for p in fp16 fp32; do
      local precision=()
      [[ $p == fp32 ]] || precision=(--fp16)
      run_step "$out" "rep${r}_esm2_150m_$p" "$n_rep" uv run --frozen scripts/run_esm2.py --size 150m \
        --template "$rep" --out-cache "$out/rep${r}_esm2_150m_$p" --device cuda "${precision[@]}"
    done
  done

  # Same-box comparisons, bit-exact.
  for e in "${ENCODERS[@]}"; do
    compare "$out" "repeat_cds_$e" "$out/rep1_cds_$e" "$out/rep2_cds_$e" --exact
    compare "$out" "repeat_tss_$e" "$out/rep1_tss_$e" "$out/rep2_tss_$e" --exact
  done
  compare "$out" repeat_enformer "$out/rep1_enformer" "$out/rep2_enformer" --exact
  for p in fp16 fp32; do
    compare "$out" "repeat_esm2_150m_$p" "$out/rep1_esm2_150m_$p" "$out/rep2_esm2_150m_$p" --exact
  done
  for e in "${BOUNDARY_ENCODERS[@]}"; do
    compare "$out" "legB_cds_$e" "$out/new_cds_$e" "$out/old_cds_$e" --exact
    compare "$out" "legB_tss_$e" "$out/new_tss_$e" "$out/old_tss_$e" --exact
  done
  compare "$out" legB_enformer "$out/new_enformer" "$out/old_enformer" --exact
  for s in "${ESM_SIZES[@]}"; do
    compare "$out" "legB_esm2_$s" "$out/new_esm2_${s}_fp16" "$out/old_esm2_$s" --exact
  done
  hyena_checks "$out" "$out/new_cds_hyena_dna" "$out/new_tss_hyena_dna"

  # Measurements, not gates: HyenaDNA without CLS/SEP, and fp32 vs fp16 ESM-2.
  for x in cds tss; do
    compare "$out" "measure_hyena_${x}_new_vs_old" "$out/new_${x}_hyena_dna" "$out/old_${x}_hyena_dna" \
      --keys mean,max --max-rel-l2 0 --min-cos 1 --report-only
  done
  for s in "${ESM_SIZES[@]}"; do
    compare "$out" "measure_esm2_${s}_fp32_vs_fp16" "$out/new_esm2_${s}_fp32" "$out/new_esm2_${s}_fp16" \
      --max-rel-l2 0 --min-cos 1 --report-only
  done

  if [[ -s $out/failed.txt ]]; then
    echo "PILOT FAILED: $(tr '\n' ' ' < "$out/failed.txt")"; exit 1
  fi
  echo "PILOT same-box checks passed; pull $out back for Leg A/A'."
}

full() {
  local out=$1
  mkdir -p "$out"
  provenance "$PWD" "$out/provenance.json"
  uv run --frozen python - "$out/provenance.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
if p["git_dirty"] or not p["imports_own_code"]:
    raise SystemExit("full: the clone has uncommitted tracked changes or imports another clone's code")
PY
  for e in "${ENCODERS[@]}"; do
    run_step "$out" "cds_$e" all uv run --frozen scripts/run_multi_pool_extract.py --encoder "$e" --device cuda
    uv run --frozen "$COMPARE" census "data/chunk_reductions_v2_$e"
  done
  for e in "${ENCODERS[@]}"; do
    run_step "$out" "tss_$e" all uv run --frozen scripts/run_tss_multi_pool_extract.py --encoder "$e" --device cuda
    uv run --frozen "$COMPARE" census "data/tss_chunk_reductions_v2_$e"
  done
  run_step "$out" enformer all uv run --frozen scripts/run_enformer_features.py --device cuda --no-datasets
  uv run --frozen "$COMPARE" census data/enformer_features_v2
  for s in "${ESM_SIZES[@]}"; do
    run_step "$out" "esm2_$s" all uv run --frozen scripts/run_esm2.py --size "$s" --device cuda
    uv run --frozen "$COMPARE" census "data/esm2_${s}_embeddings_v2"
  done
  echo "FULL extraction done; every cache passed census."
}

case $cmd in
  provenance) provenance "$PWD" "$1/provenance.json" ;;
  pilot) pilot "$@" ;;
  full)
    if [[ ${2:-} == --shutdown ]]; then trap 'sudo shutdown -h +1' EXIT; fi
    full "$1" ;;
esac
