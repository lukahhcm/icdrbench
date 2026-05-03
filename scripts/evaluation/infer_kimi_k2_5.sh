#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

TRACKS="${TRACKS:-atomic_ops,main,order_sensitivity}"
EVAL_ROOT="${EVAL_ROOT:-data/benchmark}"
MODEL="${MODEL:-kimi-k2.5}"
BASE_URL="${BASE_URL:-http://123.57.212.178:3333/v1}"
API_KEY="${API_KEY:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/evaluation/infer/kimi_k2_5}"
PROMPT_VARIANT_INDICES="${PROMPT_VARIANT_INDICES:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_INPUT_CHARS="${MAX_INPUT_CHARS:-0}"
TEMPERATURE="${TEMPERATURE:-1}"
TOP_P="${TOP_P:-0.95}"
MAX_TOKENS="${MAX_TOKENS:-0}"
CONCURRENCY="${CONCURRENCY:-10}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"

cmd=(
  "${REPO_ROOT}/scripts/infer_benchmark_tracks.sh"
  --tracks "${TRACKS}"
  --eval-root "${EVAL_ROOT}"
  --model "${MODEL}"
  --base-url "${BASE_URL}"
  --output-root "${OUTPUT_ROOT}"
  --prompt-variant-indices "${PROMPT_VARIANT_INDICES}"
  --max-samples "${MAX_SAMPLES}"
  --max-input-chars "${MAX_INPUT_CHARS}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-tokens "${MAX_TOKENS}"
  --concurrency "${CONCURRENCY}"
  --progress-every "${PROGRESS_EVERY}"
)

if [[ -n "${API_KEY}" ]]; then
  cmd+=(--api-key "${API_KEY}")
fi
if [[ "${RESUME:-true}" == "true" ]]; then
  cmd+=(--resume)
fi

exec "${cmd[@]}"
