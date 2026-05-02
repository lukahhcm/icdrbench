#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TRACKS="${TRACKS:-atomic_ops,main}"
BENCHMARK_DIR="${BENCHMARK_DIR:-}"
MODEL="${MODEL:-qwen3.6-plus}"
BASE_URL="${BASE_URL:-http://123.57.212.178:3333/v1}"
PREDICTIONS_ROOT="${PREDICTIONS_ROOT:-data/evaluation/infer/qwen3_6_plus}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/evaluation/score/qwen3_6_plus}"
PREDICTION_INSTANCE_FIELD="${PREDICTION_INSTANCE_FIELD:-instance_id}"
PREDICTION_STATUS_FIELD="${PREDICTION_STATUS_FIELD:-}"
PREDICTION_TEXT_FIELD="${PREDICTION_TEXT_FIELD:-}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"

cmd=(
  "${REPO_ROOT}/scripts/score_benchmark_tracks.sh"
  --tracks "${TRACKS}"
  --predictions-root "${PREDICTIONS_ROOT}"
  --output-root "${OUTPUT_ROOT}"
  --model "${MODEL}"
  --base-url "${BASE_URL}"
  --prediction-instance-field "${PREDICTION_INSTANCE_FIELD}"
  --progress-every "${PROGRESS_EVERY}"
)

if [[ -n "${BENCHMARK_DIR}" ]]; then
  cmd+=(--benchmark-dir "${BENCHMARK_DIR}")
fi
if [[ -n "${PREDICTION_STATUS_FIELD}" ]]; then
  cmd+=(--prediction-status-field "${PREDICTION_STATUS_FIELD}")
fi
if [[ -n "${PREDICTION_TEXT_FIELD}" ]]; then
  cmd+=(--prediction-text-field "${PREDICTION_TEXT_FIELD}")
fi
if [[ "${RESUME:-true}" == "true" ]]; then
  cmd+=(--resume)
fi

exec "${cmd[@]}"
