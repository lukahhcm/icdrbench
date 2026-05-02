#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  build_engineering_atomic_subset.sh [options]

Build a smaller engineering subset from the full atomic benchmark.

Default behavior:
  - read from data/benchmark_full/atomic_ops
  - use selection summaries from data/processed/benchmark_instances when available
  - write to data/benchmark/atomic_ops
  - keep up to 6 rows per operator
  - try to keep KEEP/DROP balanced when both exist

Options:
  --source-dir <path>                Full atomic benchmark directory. Default: data/benchmark_full/atomic_ops
  --processed-summary-dir <path>     Benchmark-instance summary directory. Default: data/processed/benchmark_instances
  --output-dir <path>                Output subset directory. Default: data/benchmark/atomic_ops
  --rows-per-operator <int>          Max rows kept per operator. Default: 6
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

SOURCE_DIR="data/benchmark_full/atomic_ops"
PROCESSED_SUMMARY_DIR="data/processed/benchmark_instances"
OUTPUT_DIR="data/benchmark/atomic_ops"
ROWS_PER_OPERATOR="6"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-dir)
      SOURCE_DIR="$2"
      shift 2
      ;;
    --processed-summary-dir)
      PROCESSED_SUMMARY_DIR="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --rows-per-operator)
      ROWS_PER_OPERATOR="$2"
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

"$PYTHON_BIN" -m cdrbench.prepare_data.build_engineering_atomic_subset \
  --source-dir "$SOURCE_DIR" \
  --processed-summary-dir "$PROCESSED_SUMMARY_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --rows-per-operator "$ROWS_PER_OPERATOR"
