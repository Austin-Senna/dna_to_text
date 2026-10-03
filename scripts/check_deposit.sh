#!/usr/bin/env bash
# Clean-room check of the Zenodo deposit (Phase 7, G30): the tarballs plus the
# committed source must rebuild every feature parquet the canonical records read.
#
# 1. SHA256SUMS verifies the tarballs as they will be uploaded.
# 2. `git archive HEAD` gives a copy of the source with no untracked files, and
#    the tarballs are unpacked into it.
# 3. The Stage 4 builders run there, from the deposited caches only (their own
#    G19 checks refuse a stale or foreign cache file), and overwrite the
#    parquets git ships.
# 4. Every file a record stamps (parquets, stored predictions, GenePT targets)
#    must match its hash under the copy.
#
# Imports resolve to the copy (PYTHONPATH ahead of the editable install), so no
# builder reads this checkout's data/. CPU only, no network (HF tokenizers
# come from the local cache).
#
# Run: scripts/check_deposit.sh outputs/deposit [workdir]   (about 15 GB free in workdir)
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
deposit="$(cd "$1" && pwd)"
clean="$(mktemp -d "${2:-${TMPDIR:-/tmp}}/mina-deposit-check.XXXXXX")"
py="${repo}/.venv/bin/python"
echo "deposit ${deposit}; clean copy ${clean} at $(git -C "${repo}" rev-parse --short HEAD)"

(cd "${deposit}" && sha256sum --check --quiet SHA256SUMS)
git -C "${repo}" archive HEAD | tar -x -C "${clean}"
# Only what SHA256SUMS lists: a stray tarball in the directory is not unpacked.
while read -r _ name; do
    case "${name}" in *.tar.gz) tar -xzf "${deposit}/${name}" -C "${clean}" ;; esac
done < "${deposit}/SHA256SUMS"

export PYTHONPATH="${clean}/src:${clean}/scripts"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "${clean}"
"${py}" - "${clean}" <<'PY'
import sys
import data_loader, linear_trainer
for m in (data_loader, linear_trainer):
    assert m.__file__.startswith(sys.argv[1]), f"{m.__name__} imports from {m.__file__}, not the copy"
PY

for e in dnabert2 nt_v2 gena_lm hyena_dna; do
    "${py}" scripts/build_pooling_datasets.py --encoder "${e}"
    "${py}" scripts/build_tss_pooling_datasets.py --encoder "${e}"
done
"${py}" scripts/build_tss_anchored_datasets.py
"${py}" scripts/build_tss_composition_baseline.py
"${py}" scripts/run_enformer_features.py --from-cache
"${py}" scripts/build_esm2_datasets.py --size 150m
"${py}" scripts/build_esm2_datasets.py --size 650m

"${py}" scripts/build_deposit.py --verify "${clean}"
echo "deposit check passed; remove ${clean} when done"
