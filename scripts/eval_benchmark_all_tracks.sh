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
  --eval-root <path>                 Final self-contained benchmark root. Default: data/benchmark
  --infer-root <path>                Inference output root. Default: data/evaluation/infer/all_tracks
  --output-root <path>               Legacy alias: same as --infer-root
  --predictions-root <path>          Existing predictions root for score-only mode. Default: --infer-root
  --tracks <csv>                     Comma-separated tracks. Default: atomic_ops,main,order_sensitivity
  --model <name>                     API model name. Required unless --score-only
  --base-url <url>                   OpenAI-compatible API base URL
  --api-key <key>                    API key. For local vLLM you can use EMPTY
  --prompt-variant-index <int>       Which prompt variant to use. Default: 0
  --max-samples <int>                Optional cap for smoke tests. Default: 0 (all)
  --temperature <float>              Default: 0.0
  --max-tokens <int>                 Default: 0 (use model/server default)
  --concurrency <int>                Request concurrency. Default: 1
  --resume                           Resume predictions from existing per-track files
  --predict-only                     Only run inference, skip scoring
  --score-only                       Only score existing prediction files
  -h, --help                         Show this help

Examples:
  ./scripts/eval_benchmark_all_tracks.sh \
    --model gpt-5.4 \
    --base-url http://123.57.212.178:3333/v1 \
    --infer-root data/evaluation/infer/gpt54_all

  ./scripts/eval_benchmark_all_tracks.sh \
    --model local-model \
    --base-url http://127.0.0.1:8000/v1 \
    --api-key EMPTY \
    --infer-root data/evaluation/infer/local_model_all

  ./scripts/eval_benchmark_all_tracks.sh \
    --score-only \
    --predictions-root /path/to/predictions_root
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="$REPO_ROOT/.venv-ops/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

EVAL_ROOT="data/benchmark"
INFER_ROOT="data/evaluation/infer/all_tracks"
PREDICTIONS_ROOT=""
TRACKS_CSV="atomic_ops,main,order_sensitivity"
MODEL=""
BASE_URL=""
API_KEY=""
PROMPT_VARIANT_INDEX="0"
MAX_SAMPLES="0"
TEMPERATURE="0.0"
MAX_TOKENS="0"
CONCURRENCY="1"
RESUME="false"
PREDICT_ONLY="false"
SCORE_ONLY="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --eval-root)
      EVAL_ROOT="$2"
      shift 2
      ;;
    --infer-root)
      INFER_ROOT="$2"
      shift 2
      ;;
    --output-root)
      INFER_ROOT="$2"
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
    --concurrency)
      CONCURRENCY="$2"
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
  PREDICTIONS_ROOT="$INFER_ROOT"
fi

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
IFS=',' read -r -a TRACKS <<< "$TRACKS_CSV"

track_benchmark_path() {
  case "$1" in
    atomic_ops)
      printf '%s\n' "$BENCHMARK_DIR/atomic_ops/atomic_ops.jsonl"
      ;;
    main)
      printf '%s\n' "$BENCHMARK_DIR/main/main.jsonl"
      ;;
    order_sensitivity)
      printf '%s\n' "$BENCHMARK_DIR/order_sensitivity/order_sensitivity.jsonl"
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
      printf '%s\n' "$EVAL_ROOT/atomic_ops/atomic_ops.jsonl"
      ;;
    main)
      printf '%s\n' "$EVAL_ROOT/main/main.jsonl"
      ;;
    order_sensitivity)
      printf '%s\n' "$EVAL_ROOT/order_sensitivity/order_sensitivity.jsonl"
      ;;
    *)
      echo "Unsupported track: $1" >&2
      exit 1
      ;;
  esac
}

for track in "${TRACKS[@]}"; do
  eval_path="$(track_eval_path "$track")"
  track_infer_dir="$INFER_ROOT/$track"
  track_predictions_dir="$PREDICTIONS_ROOT/$track"
  predictions_path="$track_predictions_dir/predictions.jsonl"

  mkdir -p "$track_infer_dir"

  if [[ "$SCORE_ONLY" != "true" ]]; then
    if [[ ! -f "$eval_path" ]]; then
      echo "Missing eval file for track=$track: $eval_path" >&2
      exit 1
    fi
    predict_cmd=(
      "$PYTHON_BIN" -m cdrbench.eval.run_benchmark_eval predict
      --eval-path "$eval_path"
      --output-path "$track_infer_dir/predictions.jsonl"
      --model "$MODEL"
      --prompt-variant-index "$PROMPT_VARIANT_INDEX"
        --max-samples "$MAX_SAMPLES"
        --temperature "$TEMPERATURE"
        --max-tokens "$MAX_TOKENS"
        --concurrency "$CONCURRENCY"
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
    echo "[run] track=$track step=predict output_dir=$track_infer_dir"
    "${predict_cmd[@]}"
  fi

  if [[ "$PREDICT_ONLY" != "true" ]]; then
    if [[ "$SCORE_ONLY" == "true" && ! -f "$predictions_path" ]]; then
      echo "Missing predictions file for track=$track: $predictions_path" >&2
      exit 1
    fi
    if [[ "$SCORE_ONLY" != "true" ]]; then
      predictions_path="$track_infer_dir/predictions.jsonl"
    fi
    score_cmd=(
      "$PYTHON_BIN" -m cdrbench.eval.run_benchmark_eval score
      --predictions-path "$predictions_path"
      --output-dir "$(dirname "$predictions_path")"
      --progress-every 20
    )
    score_output_dir="$(dirname "$predictions_path")"
    echo "[run] track=$track step=score output_dir=$score_output_dir"
    "${score_cmd[@]}"
  fi

  echo "[done] track=$track infer_dir=$track_infer_dir"
done

echo "[complete] evaluation finished for tracks: ${TRACKS[*]}"
echo "[complete] inference rooted at: $INFER_ROOT"
echo "[complete] reports written under predictions roots"
