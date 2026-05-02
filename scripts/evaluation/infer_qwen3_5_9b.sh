#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TRACKS="${TRACKS:-atomic_ops,main,order_sensitivity}"
EVAL_ROOT="${EVAL_ROOT:-data/benchmark}"
MODEL="${MODEL:-qwen3_5_9b}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8904/v1}"
API_KEY="${API_KEY:-EMPTY}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/evaluation/infer/${MODEL}}"
PROMPT_VARIANT_INDICES="${PROMPT_VARIANT_INDICES:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_INPUT_CHARS="${MAX_INPUT_CHARS:-0}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-0}"
CONCURRENCY="${CONCURRENCY:-16}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"

cmd=(
  "${REPO_ROOT}/scripts/infer_benchmark_tracks.sh"
  --tracks "${TRACKS}"
  --eval-root "${EVAL_ROOT}"
  --model "${MODEL}"
  --base-url "${BASE_URL}"
  --api-key "${API_KEY}"
  --output-root "${OUTPUT_ROOT}"
  --prompt-variant-indices "${PROMPT_VARIANT_INDICES}"
  --max-samples "${MAX_SAMPLES}"
  --max-input-chars "${MAX_INPUT_CHARS}"
  --temperature "${TEMPERATURE}"
  --max-tokens "${MAX_TOKENS}"
  --concurrency "${CONCURRENCY}"
  --progress-every "${PROGRESS_EVERY}"
)

if [[ "${RESUME:-true}" == "true" ]]; then
  cmd+=(--resume)
fi

exec "${cmd[@]}"
