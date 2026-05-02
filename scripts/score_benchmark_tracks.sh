#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  score_benchmark_tracks.sh [options]

Score previously saved inference outputs for one or more CDR-Bench tracks.
By default this script targets the two primary tracks:

  1. atomic_ops
  2. main

This step only reads predictions, computes metrics, and writes reports next to `predictions.jsonl`.

Options:
  --predictions-root <path>           Inference root. Default: data/evaluation/infer
  --tracks <csv>                      Comma-separated tracks. Default: atomic_ops,main
  --progress-every <int>              Default: 20
  --resume                            Resume scoring from existing report files in the same directory
  -h, --help                          Show this help

Examples:
  ./scripts/score_benchmark_tracks.sh \
    --predictions-root data/evaluation/infer/gpt54

  ./scripts/score_benchmark_tracks.sh \
    --tracks atomic_ops,main \
    --predictions-root data/evaluation/infer/local_model
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="$REPO_ROOT/.venv-ops/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

PREDICTIONS_ROOT="data/evaluation/infer"
TRACKS_CSV="atomic_ops,main"
PROGRESS_EVERY="20"
RESUME="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --predictions-root)
      PREDICTIONS_ROOT="$2"
      shift 2
      ;;
    --tracks)
      TRACKS_CSV="$2"
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

for track in "${TRACKS[@]}"; do
  predictions_path="$PREDICTIONS_ROOT/$track/predictions.jsonl"
  output_dir="$PREDICTIONS_ROOT/$track"
  mkdir -p "$output_dir"

  if [[ ! -f "$predictions_path" ]]; then
    echo "Missing predictions file for track=$track: $predictions_path" >&2
    exit 1
  fi

  cmd=(
    "$PYTHON_BIN" -m cdrbench.eval.run_benchmark_eval score
    --predictions-path "$predictions_path"
    --output-dir "$output_dir"
    --progress-every "$PROGRESS_EVERY"
  )
  if [[ "$RESUME" == "true" ]]; then
    cmd+=(--resume)
  fi

  echo "[run] track=$track step=score output_dir=$output_dir"
  "${cmd[@]}"
  echo "[done] track=$track scored=$output_dir"
done

echo "[complete] scoring finished for tracks: ${TRACKS[*]}"
echo "[complete] reports written under: $PREDICTIONS_ROOT"
