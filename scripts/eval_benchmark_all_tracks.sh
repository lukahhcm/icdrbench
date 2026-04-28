#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  eval_benchmark_all_tracks.sh [options]

Run CDR-Bench evaluation across one or more tracks using any OpenAI-compatible API,
including a local vLLM server. By default this script covers:

  1. atomic_ops
  2. main
  3. order_sensitivity

Modes:
  1. predict + score (default)
  2. predict only
  3. score only from existing prediction files

Options:
  --benchmark-dir <path>             Benchmark JSONL root. Default: data/benchmark
  --eval-root <path>                 Prompt-eval root. Default: data/benchmark_prompts
  --output-root <path>               Output root. Default: data/eval_runs/all_tracks
  --predictions-root <path>          Existing predictions root for score-only mode. Default: --output-root
  --tracks <csv>                     Comma-separated tracks. Default: atomic_ops,main,order_sensitivity
  --model <name>                     API model name. Required unless --score-only
  --base-url <url>                   OpenAI-compatible API base URL
  --api-key <key>                    API key. For local vLLM you can use EMPTY
  --prompt-variant-index <int>       Which prompt variant to use. Default: 0
  --max-samples <int>                Optional cap for smoke tests. Default: 0 (all)
  --temperature <float>              Default: 0.0
  --max-tokens <int>                 Default: 4096
  --resume                           Resume predictions from existing per-track files
  --predict-only                     Only run inference, skip scoring
  --score-only                       Only score existing prediction files
  --prediction-instance-field <key>  Score-only override. Default: instance_id
  --prediction-status-field <key>    Score-only override for prediction status field
  --prediction-text-field <key>      Score-only override for prediction clean-text field
  -h, --help                         Show this help

Examples:
  ./scripts/eval_benchmark_all_tracks.sh \
    --model gpt-5.4 \
    --base-url http://123.57.212.178:3333/v1 \
    --output-root data/eval_runs/gpt54_all

  ./scripts/eval_benchmark_all_tracks.sh \
    --model local-model \
    --base-url http://127.0.0.1:8000/v1 \
    --api-key EMPTY \
    --output-root data/eval_runs/local_model_all

  ./scripts/eval_benchmark_all_tracks.sh \
    --score-only \
    --predictions-root /path/to/predictions_root \
    --output-root data/eval_runs/scored_server_predictions
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="$REPO_ROOT/.venv-ops/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

BENCHMARK_DIR="data/benchmark"
EVAL_ROOT="data/benchmark_prompts"
OUTPUT_ROOT="data/eval_runs/all_tracks"
PREDICTIONS_ROOT=""
TRACKS_CSV="atomic_ops,main,order_sensitivity"
MODEL=""
BASE_URL=""
API_KEY=""
PROMPT_VARIANT_INDEX="0"
MAX_SAMPLES="0"
TEMPERATURE="0.0"
MAX_TOKENS="4096"
RESUME="false"
PREDICT_ONLY="false"
SCORE_ONLY="false"
PREDICTION_INSTANCE_FIELD="instance_id"
PREDICTION_STATUS_FIELD=""
PREDICTION_TEXT_FIELD=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark-dir)
      BENCHMARK_DIR="$2"
      shift 2
      ;;
    --eval-root)
      EVAL_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --predictions-root)
      PREDICTIONS_ROOT="$2"
      shift 2
      ;;
    --tracks)
      TRACKS_CSV="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --base-url)
      BASE_URL="$2"
      shift 2
      ;;
    --api-key)
      API_KEY="$2"
      shift 2
      ;;
    --prompt-variant-index)
      PROMPT_VARIANT_INDEX="$2"
      shift 2
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --temperature)
      TEMPERATURE="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --resume)
      RESUME="true"
      shift 1
      ;;
    --predict-only)
      PREDICT_ONLY="true"
      shift 1
      ;;
    --score-only)
      SCORE_ONLY="true"
      shift 1
      ;;
    --prediction-instance-field)
      PREDICTION_INSTANCE_FIELD="$2"
      shift 2
      ;;
    --prediction-status-field)
      PREDICTION_STATUS_FIELD="$2"
      shift 2
      ;;
    --prediction-text-field)
      PREDICTION_TEXT_FIELD="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$PREDICT_ONLY" == "true" && "$SCORE_ONLY" == "true" ]]; then
  echo "--predict-only and --score-only cannot be used together." >&2
  exit 1
fi

if [[ "$SCORE_ONLY" != "true" && -z "$MODEL" ]]; then
  echo "--model is required unless --score-only is used." >&2
  exit 1
fi

if [[ -z "$PREDICTIONS_ROOT" ]]; then
  PREDICTIONS_ROOT="$OUTPUT_ROOT"
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
IFS=',' read -r -a TRACKS <<< "$TRACKS_CSV"

track_benchmark_path() {
  case "$1" in
    atomic_ops)
      printf '%s\n' "$BENCHMARK_DIR/atomic_ops.jsonl"
      ;;
    main)
      printf '%s\n' "$BENCHMARK_DIR/main.jsonl"
      ;;
    order_sensitivity)
      printf '%s\n' "$BENCHMARK_DIR/order_sensitivity.jsonl"
      ;;
    *)
      echo "Unsupported track: $1" >&2
      exit 1
      ;;
  esac
}

track_eval_path() {
  case "$1" in
    atomic_ops)
      printf '%s\n' "$EVAL_ROOT/atomic_ops/eval/atomic_ops.jsonl"
      ;;
    main)
      printf '%s\n' "$EVAL_ROOT/main/eval/main.jsonl"
      ;;
    order_sensitivity)
      printf '%s\n' "$EVAL_ROOT/order_sensitivity/eval/order_sensitivity.jsonl"
      ;;
    *)
      echo "Unsupported track: $1" >&2
      exit 1
      ;;
  esac
}

for track in "${TRACKS[@]}"; do
  benchmark_path="$(track_benchmark_path "$track")"
  eval_path="$(track_eval_path "$track")"
  track_output_dir="$OUTPUT_ROOT/$track"
  track_predictions_dir="$PREDICTIONS_ROOT/$track"
  predictions_path="$track_predictions_dir/predictions.jsonl"

  mkdir -p "$track_output_dir"

  if [[ "$SCORE_ONLY" != "true" ]]; then
    if [[ ! -f "$eval_path" ]]; then
      echo "Missing eval file for track=$track: $eval_path" >&2
      exit 1
    fi
    predict_cmd=(
      "$PYTHON_BIN" -m cdrbench.eval.run_benchmark_eval predict
      --eval-path "$eval_path"
      --output-path "$track_output_dir/predictions.jsonl"
      --model "$MODEL"
      --prompt-variant-index "$PROMPT_VARIANT_INDEX"
      --max-samples "$MAX_SAMPLES"
      --temperature "$TEMPERATURE"
      --max-tokens "$MAX_TOKENS"
    )
    if [[ -n "$BASE_URL" ]]; then
      predict_cmd+=(--base-url "$BASE_URL")
    fi
    if [[ -n "$API_KEY" ]]; then
      predict_cmd+=(--api-key "$API_KEY")
    fi
    if [[ "$RESUME" == "true" ]]; then
      predict_cmd+=(--resume)
    fi
    echo "[run] track=$track step=predict output_dir=$track_output_dir"
    "${predict_cmd[@]}"
  fi

  if [[ "$PREDICT_ONLY" != "true" ]]; then
    if [[ ! -f "$benchmark_path" ]]; then
      echo "Missing benchmark file for track=$track: $benchmark_path" >&2
      exit 1
    fi
    if [[ "$SCORE_ONLY" == "true" && ! -f "$predictions_path" ]]; then
      echo "Missing predictions file for track=$track: $predictions_path" >&2
      exit 1
    fi
    if [[ "$SCORE_ONLY" != "true" ]]; then
      predictions_path="$track_output_dir/predictions.jsonl"
    fi
    score_cmd=(
      "$PYTHON_BIN" -m cdrbench.eval.run_benchmark_eval score
      --predictions-path "$predictions_path"
      --benchmark-path "$benchmark_path"
      --output-dir "$track_output_dir/scored"
      --prediction-instance-field "$PREDICTION_INSTANCE_FIELD"
    )
    if [[ -n "$MODEL" ]]; then
      score_cmd+=(--model "$MODEL")
    fi
    if [[ -n "$BASE_URL" ]]; then
      score_cmd+=(--base-url "$BASE_URL")
    fi
    if [[ -n "$PREDICTION_STATUS_FIELD" ]]; then
      score_cmd+=(--prediction-status-field "$PREDICTION_STATUS_FIELD")
    fi
    if [[ -n "$PREDICTION_TEXT_FIELD" ]]; then
      score_cmd+=(--prediction-text-field "$PREDICTION_TEXT_FIELD")
    fi
    echo "[run] track=$track step=score output_dir=$track_output_dir/scored"
    "${score_cmd[@]}"
  fi

  echo "[done] track=$track output_dir=$track_output_dir"
done

echo "[complete] evaluation finished for tracks: ${TRACKS[*]}"
echo "[complete] outputs rooted at: $OUTPUT_ROOT"
