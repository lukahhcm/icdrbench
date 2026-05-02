#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  build_engineering_main_subset.sh [options]

Build a smaller engineering subset from the full main benchmark.

Default behavior:
  - read from data/benchmark_full/main
  - write to data/benchmark/main
  - keep at most 1 clean-only, 1 filter-then-clean, and 1 clean-then-filter variant per recipe
  - keep up to 10 rows per retained variant

Options:
  --source-dir <path>                Full main benchmark directory. Default: data/benchmark_full/main
  --output-dir <path>                Output subset directory. Default: data/benchmark/main
  --rows-per-variant <int>           Max rows kept per retained variant. Default: 10
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

SOURCE_DIR="data/benchmark_full/main"
OUTPUT_DIR="data/benchmark/main"
ROWS_PER_VARIANT="10"

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
    --rows-per-variant)
      ROWS_PER_VARIANT="$2"
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

"$PYTHON_BIN" -m cdrbench.prepare_data.build_engineering_main_subset \
  --source-dir "$SOURCE_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --rows-per-variant "$ROWS_PER_VARIANT"
