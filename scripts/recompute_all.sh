#!/usr/bin/env bash
# The camera-ready recompute (Phase 4): every probe cell, then the null bands.
#
# Pins every BLAS/OpenMP pool to one thread before Python starts (thread count
# changes lbfgs results; measurements_2026-09.md §4), refuses a dirty tree, and
# records the commit. Records land in data/v2/, predictions in
# outputs/predictions/v2/ (data/v2 itself is exempt from the dirty check, so a
# run can resume). A rerun resumes cells already recorded at the same
# commit; records from another commit are refused, never mixed.
#
# Run: scripts/recompute_all.sh [main|null|all]   (default: all)
set -euo pipefail
cd "$(dirname "$0")/.."

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

# Plain commands and an assignment, so set -e stops the run when git can't read
# the tree (no .git, "dubious ownership"); a failure inside [ ... ] would pass.
git rev-parse --verify HEAD > /dev/null
dirty="$(git status --porcelain -- src scripts data tests pyproject.toml uv.lock ':!data/v2')"
if [ -n "${dirty}" ]; then
    echo "refusing: uncommitted changes under src, scripts, data, tests or the lockfile" >&2
    git status --short -- src scripts data tests pyproject.toml uv.lock ':!data/v2' >&2
    exit 1
fi

group="${1:-all}"
echo "recompute at $(git rev-parse HEAD), group ${group}, started $(date -u +%FT%TZ)"
uv run --frozen scripts/recompute_all.py --group "${group}"
echo "finished $(date -u +%FT%TZ)"
