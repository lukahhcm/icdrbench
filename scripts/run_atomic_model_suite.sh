#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_atomic_model_suite.sh [infer|score|both]

Batch-run a fixed suite of models on the primary CDR-Bench tracks.
All frequently edited parameters live in the config block at the top of this script.

Modes:
  infer   Run model inference only
  score   Score previously saved predictions only
  both    Run inference and then scoring

Examples:
  ./scripts/run_atomic_model_suite.sh infer
  ./scripts/run_atomic_model_suite.sh score
  ./scripts/run_atomic_model_suite.sh both
EOF
}

MODE="${1:-infer}"
if [[ "$MODE" == "-h" || "$MODE" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$MODE" != "infer" && "$MODE" != "score" && "$MODE" != "both" ]]; then
  echo "Unsupported mode: $MODE" >&2
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

###############################################################################
# Config: edit this block when you want to change datasets, tracks, or models.
###############################################################################

# Tracks:
#   atomic_ops
#   main
# Keep atomic_ops for the first sweep; add main later if needed.
TRACKS="atomic_ops"

# If your benchmark files are not stored in this repo's default data/ tree,
# set them here to absolute paths.
EVAL_ROOT="${EVAL_ROOT:-data/benchmark_prompts}"
BENCHMARK_DIR="${BENCHMARK_DIR:-data/benchmark}"

# Output roots. The script will create one subdirectory per model slug.
INFERENCE_ROOT="${INFERENCE_ROOT:-data/inference_runs}"
SCORE_ROOT="${SCORE_ROOT:-data/score_runs}"

# Inference behavior.
PROMPT_VARIANT_INDICES="${PROMPT_VARIANT_INDICES:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
PROGRESS_EVERY="${PROGRESS_EVERY:-20}"
RESUME_INFER="${RESUME_INFER:-true}"
RESUME_SCORE="${RESUME_SCORE:-true}"

# Shared gateway settings. Leave empty if you want to set per-model overrides below.
DEFAULT_BASE_URL="${DEFAULT_BASE_URL:-}"
DEFAULT_API_KEY="${DEFAULT_API_KEY:-}"

# Model suite.
# One line per model:
#   slug|model_name|base_url|api_key
#
# Rules:
# - slug controls the output directory name
# - model_name is the API model string
# - base_url/api_key can be left empty to inherit DEFAULT_BASE_URL/DEFAULT_API_KEY
# - to add a model, just append one new line
MODELS=(
  "gpt_5_4|gpt-5.4||"
  "gemini_3_1_pro|gemini-3.1-pro-preview||"
  "qwen3_6_plus|qwen3.6-plus||"
)

###############################################################################

parse_model_entry() {
  local entry="$1"
  IFS='|' read -r MODEL_SLUG_VALUE MODEL_NAME_VALUE MODEL_BASE_URL_VALUE MODEL_API_KEY_VALUE <<< "$entry"
}

run_infer_for_model() {
  local entry="$1"
  parse_model_entry "$entry"
  local model_name="$MODEL_NAME_VALUE"
  local model_slug="$MODEL_SLUG_VALUE"
  local base_url="${MODEL_BASE_URL_VALUE:-$DEFAULT_BASE_URL}"
  local api_key="${MODEL_API_KEY_VALUE:-$DEFAULT_API_KEY}"

  if [[ -z "$model_name" ]]; then
    echo "Missing model_name for model entry: $entry" >&2
    exit 1
  fi
  if [[ -z "$model_slug" ]]; then
    echo "Missing model_slug for model entry: $entry" >&2
    exit 1
  fi

  local cmd=(
    ./scripts/infer_benchmark_tracks.sh
    --tracks "$TRACKS"
    --eval-root "$EVAL_ROOT"
    --model "$model_name"
    --output-root "$INFERENCE_ROOT/$model_slug"
    --prompt-variant-indices "$PROMPT_VARIANT_INDICES"
    --max-samples "$MAX_SAMPLES"
    --temperature "$TEMPERATURE"
    --max-tokens "$MAX_TOKENS"
    --progress-every "$PROGRESS_EVERY"
  )
  if [[ -n "$base_url" ]]; then
    cmd+=(--base-url "$base_url")
  fi
  if [[ -n "$api_key" ]]; then
    cmd+=(--api-key "$api_key")
  fi
  if [[ "$RESUME_INFER" == "true" ]]; then
    cmd+=(--resume)
  fi

  echo "[suite] mode=infer model=$model_name slug=$model_slug tracks=$TRACKS"
  "${cmd[@]}"
}

run_score_for_model() {
  local entry="$1"
  parse_model_entry "$entry"
  local model_name="$MODEL_NAME_VALUE"
  local model_slug="$MODEL_SLUG_VALUE"
  local base_url="${MODEL_BASE_URL_VALUE:-$DEFAULT_BASE_URL}"

  local cmd=(
    ./scripts/score_benchmark_tracks.sh
    --tracks "$TRACKS"
    --benchmark-dir "$BENCHMARK_DIR"
    --predictions-root "$INFERENCE_ROOT/$model_slug"
    --output-root "$SCORE_ROOT/$model_slug"
    --model "$model_name"
    --progress-every "$PROGRESS_EVERY"
  )
  if [[ -n "$base_url" ]]; then
    cmd+=(--base-url "$base_url")
  fi
  if [[ "$RESUME_SCORE" == "true" ]]; then
    cmd+=(--resume)
  fi

  echo "[suite] mode=score model=$model_name slug=$model_slug tracks=$TRACKS"
  "${cmd[@]}"
}

for entry in "${MODELS[@]}"; do
  case "$MODE" in
    infer)
      run_infer_for_model "$entry"
      ;;
    score)
      run_score_for_model "$entry"
      ;;
    both)
      run_infer_for_model "$entry"
      run_score_for_model "$entry"
      ;;
  esac
done

echo "[complete] mode=$MODE tracks=$TRACKS"
