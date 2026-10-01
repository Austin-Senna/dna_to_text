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
# Run: scripts/recompute_all.sh [main|null|all] [runner options]   (default: all)
#
# Parallel: launch N shards from one commit, e.g. `scripts/recompute_all.sh all
# --shard 0/8` ... `--shard 7/8` (they share records files under a lock), then
# finish with a plain `scripts/recompute_all.sh all`: it runs nothing new, checks
# that every cell has exactly one record, runs G1, and writes
# data/v2/run_complete.json, which build_statistics.py requires.
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
shift || true
echo "recompute at $(git rev-parse HEAD), group ${group} $*, started $(date -u +%FT%TZ)"
uv run --frozen scripts/recompute_all.py --group "${group}" "$@"
# A whole-manifest pass (group all; no --shard, --only, --dry-run or --skip-g1)
# ends with the independent reimplementation (Rule 3): it refits the headline
# cells without the pipeline's code and writes reproduction.json next to the
# records, which build_statistics.py requires.
out_dir="data/v2"
whole=1
prev=""
for arg in "$@"; do
    case "${arg}" in
        --shard|--shard=*|--only|--only=*|--dry-run|--skip-g1) whole=0 ;;
        --out-dir=*) out_dir="${arg#--out-dir=}" ;;
    esac
    if [ "${prev}" = "--out-dir" ]; then out_dir="${arg}"; fi
    prev="${arg}"
done
if [ "${group}" = "all" ] && [ "${whole}" = 1 ]; then
    uv run --frozen scripts/reproduce_headline.py --records "${out_dir}" --jobs 10
fi
echo "finished $(date -u +%FT%TZ)"
