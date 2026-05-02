#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  build_engineering_order_subset.sh [options]

Build a smaller engineering subset from the full order-sensitivity benchmark.

Default behavior:
  - read from data/benchmark_full/order_sensitivity
  - use selection summaries from data/processed/benchmark_instances
  - write to data/benchmark/order_sensitivity
  - keep the strongest 1 order family per recipe
  - keep up to 5 full groups per retained family
  - each retained group keeps front/middle/end together

Options:
  --source-dir <path>                Full order-sensitivity benchmark directory. Default: data/benchmark_full/order_sensitivity
  --processed-summary-dir <path>     Benchmark-instance summary directory. Default: data/processed/benchmark_instances
  --output-dir <path>                Output subset directory. Default: data/benchmark/order_sensitivity
  --groups-per-family <int>          Max groups kept per retained family. Default: 5
  -h, --help                         Show this help
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="$REPO_ROOT/.venv-ops/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

SOURCE_DIR="data/benchmark_full/order_sensitivity"
PROCESSED_SUMMARY_DIR="data/processed/benchmark_instances"
OUTPUT_DIR="data/benchmark/order_sensitivity"
GROUPS_PER_FAMILY="5"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --processed-summary-dir)
      PROCESSED_SUMMARY_DIR="$2"
      shift 2
      ;;
    --groups-per-family)
      GROUPS_PER_FAMILY="$2"
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

export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON_BIN" -m cdrbench.prepare_data.build_engineering_order_subset \
  --source-dir "$SOURCE_DIR" \
  --processed-summary-dir "$PROCESSED_SUMMARY_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --groups-per-family "$GROUPS_PER_FAMILY"
