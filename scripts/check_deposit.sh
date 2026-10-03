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
#    must match its hash under the copy, both before the builders run (what
#    the source tarball and the deposit ship) and after (what they rebuild).
#
# The check runs on the commit that packed the deposit, from a clean tree.
# Imports resolve to the copy (PYTHONPATH ahead of the editable install), so no
# builder reads this checkout's data/. CPU only; the composition features
# fetch the encoders' tokenizers at their pinned revisions.
#
# Run: scripts/check_deposit.sh outputs/deposit [workdir]   (about 15 GB free in workdir)
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
deposit="$(cd "$1" && pwd)"
py="${repo}/.venv/bin/python"

(cd "${deposit}" && sha256sum --check --quiet SHA256SUMS)
packed="$("${py}" -c 'import json, sys; print(json.load(open(sys.argv[1]))["source_git_sha"])' \
    "${deposit}/deposit_manifest.json")"
if [ "${packed}" != "$(git -C "${repo}" rev-parse HEAD)" ]; then
    echo "refusing: the deposit was packed at ${packed}, HEAD is $(git -C "${repo}" rev-parse HEAD)" >&2
    exit 1
fi
if [ -n "$(git -C "${repo}" status --porcelain -- src scripts data)" ]; then
    echo "refusing: uncommitted changes under src, scripts or data (the .venv imports them)" >&2
    exit 1
fi
clean="$(mktemp -d "${2:-${TMPDIR:-/tmp}}/mina-deposit-check.XXXXXX")"
echo "deposit ${deposit}; clean copy ${clean} at $(git -C "${repo}" rev-parse --short HEAD)"
git -C "${repo}" archive HEAD | tar -x -C "${clean}"
# Only what SHA256SUMS lists: a stray tarball in the directory is not unpacked.
while read -r _ name; do
    case "${name}" in *.tar.gz) tar -xzf "${deposit}/${name}" -C "${clean}" ;; esac
done < "${deposit}/SHA256SUMS"

export PYTHONPATH="${clean}/src:${clean}/scripts"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
cd "${clean}"
"${py}" - "${clean}" <<'PY'
import sys
import data_loader, linear_trainer
for m in (data_loader, linear_trainer):
    assert m.__file__.startswith(sys.argv[1]), f"{m.__name__} imports from {m.__file__}, not the copy"
PY

"${py}" scripts/build_deposit.py --verify "${clean}"   # as shipped
for e in dnabert2 nt_v2 gena_lm hyena_dna; do
    "${py}" scripts/build_pooling_datasets.py --encoder "${e}"
    "${py}" scripts/build_tss_pooling_datasets.py --encoder "${e}"
done
"${py}" scripts/build_tss_anchored_datasets.py
"${py}" scripts/build_tss_composition_baseline.py
"${py}" scripts/run_enformer_features.py --from-cache
"${py}" scripts/build_esm2_datasets.py --size 150m
"${py}" scripts/build_esm2_datasets.py --size 650m

"${py}" scripts/build_deposit.py --verify "${clean}"   # as rebuilt
echo "deposit check passed; remove ${clean} when done"
