#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TRACKS="${TRACKS:-atomic_ops,main}"
MODEL="${MODEL:-qwen3_5_9b}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-data/evaluation/infer/${MODEL}}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"

cmd=(
  "${REPO_ROOT}/scripts/score_benchmark_tracks.sh"
  --tracks "${TRACKS}"
  --predictions-root "${PREDICTIONS_ROOT}"
  --progress-every "${PROGRESS_EVERY}"
)
if [[ "${RESUME:-true}" == "true" ]]; then
  cmd+=(--resume)
fi

exec "${cmd[@]}"
