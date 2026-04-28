#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  score_benchmark_tracks.sh [options]

Score previously saved raw inference outputs for one or more CDR-Bench tracks.
By default this script targets the two primary tracks:

  1. atomic_ops
  2. main

This step is metric-only. It does not call the model API.

Options:
  --benchmark-dir <path>              Benchmark JSONL root. Default: data/benchmark
  --predictions-root <path>           Inference root. Default: data/inference_runs
  --output-root <path>                Score output root. Default: data/score_runs
  --tracks <csv>                      Comma-separated tracks. Default: atomic_ops,main
  --model <name>                      Optional model label for reports
  --base-url <url>                    Optional base-url label for reports
  --prediction-instance-field <key>   Default: instance_id
  --prediction-status-field <key>     Optional override for prediction status field
  --prediction-text-field <key>       Optional override for prediction clean-text field
  --progress-every <int>              Default: 20
  --resume                            Resume scoring from existing output-dir files
  -h, --help                          Show this help

Examples:
  ./scripts/score_benchmark_tracks.sh \
    --predictions-root data/inference_runs/gpt54 \
    --output-root data/score_runs/gpt54 \
    --model gpt-5.4

  ./scripts/score_benchmark_tracks.sh \
    --tracks atomic_ops,main \
    --predictions-root data/inference_runs/local_model \
    --output-root data/score_runs/local_model \
    --model local-model
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
PREDICTIONS_ROOT="data/inference_runs"
OUTPUT_ROOT="data/score_runs"
TRACKS_CSV="atomic_ops,main"
MODEL=""
BASE_URL=""
PREDICTION_INSTANCE_FIELD="instance_id"
PREDICTION_STATUS_FIELD=""
PREDICTION_TEXT_FIELD=""
PROGRESS_EVERY="20"
RESUME="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --benchmark-dir)
      BENCHMARK_DIR="$2"
      shift 2
      ;;
    --predictions-root)
      PREDICTIONS_ROOT="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
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
    --progress-every)
      PROGRESS_EVERY="$2"
      shift 2
      ;;
    --resume)
      RESUME="true"
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

for track in "${TRACKS[@]}"; do
  benchmark_path="$(track_benchmark_path "$track")"
  predictions_path="$PREDICTIONS_ROOT/$track/predictions.jsonl"
  output_dir="$OUTPUT_ROOT/$track"
  mkdir -p "$output_dir"

  if [[ ! -f "$benchmark_path" ]]; then
    echo "Missing benchmark file for track=$track: $benchmark_path" >&2
    exit 1
  fi
  if [[ ! -f "$predictions_path" ]]; then
    echo "Missing predictions file for track=$track: $predictions_path" >&2
    exit 1
  fi

  cmd=(
    "$PYTHON_BIN" -m cdrbench.eval.run_benchmark_eval score
    --predictions-path "$predictions_path"
    --benchmark-path "$benchmark_path"
    --output-dir "$output_dir"
    --prediction-instance-field "$PREDICTION_INSTANCE_FIELD"
    --progress-every "$PROGRESS_EVERY"
  )
  if [[ -n "$MODEL" ]]; then
    cmd+=(--model "$MODEL")
  fi
  if [[ -n "$BASE_URL" ]]; then
    cmd+=(--base-url "$BASE_URL")
  fi
  if [[ -n "$PREDICTION_STATUS_FIELD" ]]; then
    cmd+=(--prediction-status-field "$PREDICTION_STATUS_FIELD")
  fi
  if [[ -n "$PREDICTION_TEXT_FIELD" ]]; then
    cmd+=(--prediction-text-field "$PREDICTION_TEXT_FIELD")
  fi
  if [[ "$RESUME" == "true" ]]; then
    cmd+=(--resume)
  fi

  echo "[run] track=$track step=score output_dir=$output_dir"
  "${cmd[@]}"
  echo "[done] track=$track scored=$output_dir"
done

echo "[complete] scoring finished for tracks: ${TRACKS[*]}"
echo "[complete] outputs rooted at: $OUTPUT_ROOT"
