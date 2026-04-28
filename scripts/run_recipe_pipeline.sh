#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/run_recipe_pipeline.sh [options]

Run the full CDR-Bench recipe pipeline end-to-end under a configurable text-length cap:

  1. Filter the domain-filtered corpus to <= max-text-length
  2. Mine recipe families from tagged records under the same length cap
  3. Materialize deterministic recipe libraries
  4. Materialize benchmark instances and references
  5. Seed old prompt caches when available
  6. Run the prompt pipeline to build eval-ready prompt tracks

Options:
  --max-text-length <int>                 Default: 10000
  --source-filtered-path <path>           Default: data/processed/domain_filtered/all.jsonl
  --filtered-output-dir <path>            Default: data/processed/domain_filtered
  --tagged-dir <path>                     Default: data/processed/domain_tags
  --recipe-mining-output-dir <path>       Default: data/processed/recipe_mining
  --recipe-library-output-dir <path>      Default: data/processed/recipe_library
  --benchmark-output-dir <path>           Default: data/benchmark
  --prompt-output-root <path>             Default: data/benchmark_prompts
  --seed-prompt-cache-root <path>         Default: data/benchmark_prompts
  --tracks <csv>                          Default: atomic_ops,main,order_sensitivity
  --prompt-source <llm|template>          Default: llm
  --model <name>                          Prompt-generation model override
  --base-url <url>                        Prompt-generation base URL override
  --api-key <key>                         Prompt-generation API key override
  --judge-model <name>                    Judge model override
  --judge-base-url <url>                  Judge base URL override
  --judge-api-key <key>                   Judge API key override
  --python-bin <path>                     Default: .venv-ops/bin/python or python3
  --skip-prompt-pipeline                  Stop after benchmark materialization
  --no-prompt-resume                      Disable recipe-level prompt cache reuse
  -h, --help                              Show this help

Examples:
  ./scripts/run_recipe_pipeline.sh

  ./scripts/run_recipe_pipeline.sh \
    --model gpt-5.4 \
    --base-url http://123.57.212.178:3333/v1
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

MAX_TEXT_LENGTH="10000"
SOURCE_FILTERED_PATH="data/processed/domain_filtered/all.jsonl"
FILTERED_OUTPUT_DIR="data/processed/domain_filtered"
TAGGED_DIR="data/processed/domain_tags"
RECIPE_MINING_OUTPUT_DIR="data/processed/recipe_mining"
RECIPE_LIBRARY_OUTPUT_DIR="data/processed/recipe_library"
BENCHMARK_OUTPUT_DIR="data/benchmark"
PROMPT_OUTPUT_ROOT="data/benchmark_prompts"
SEED_PROMPT_CACHE_ROOT="data/benchmark_prompts"
TRACKS_CSV="atomic_ops,main,order_sensitivity"
PROMPT_SOURCE="llm"
SKIP_PROMPT_PIPELINE="false"
PROMPT_RESUME="true"

MODEL=""
BASE_URL=""
API_KEY=""
JUDGE_MODEL=""
JUDGE_BASE_URL=""
JUDGE_API_KEY=""

PYTHON_BIN="$REPO_ROOT/.venv-ops/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --max-text-length)
      MAX_TEXT_LENGTH="$2"
      shift 2
      ;;
    --source-filtered-path)
      SOURCE_FILTERED_PATH="$2"
      shift 2
      ;;
    --filtered-output-dir)
      FILTERED_OUTPUT_DIR="$2"
      shift 2
      ;;
    --tagged-dir)
      TAGGED_DIR="$2"
      shift 2
      ;;
    --recipe-mining-output-dir)
      RECIPE_MINING_OUTPUT_DIR="$2"
      shift 2
      ;;
    --recipe-library-output-dir)
      RECIPE_LIBRARY_OUTPUT_DIR="$2"
      shift 2
      ;;
    --benchmark-output-dir)
      BENCHMARK_OUTPUT_DIR="$2"
      shift 2
      ;;
    --prompt-output-root)
      PROMPT_OUTPUT_ROOT="$2"
      shift 2
      ;;
    --seed-prompt-cache-root)
      SEED_PROMPT_CACHE_ROOT="$2"
      shift 2
      ;;
    --tracks)
      TRACKS_CSV="$2"
      shift 2
      ;;
    --prompt-source)
      PROMPT_SOURCE="$2"
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
    --judge-model)
      JUDGE_MODEL="$2"
      shift 2
      ;;
    --judge-base-url)
      JUDGE_BASE_URL="$2"
      shift 2
      ;;
    --judge-api-key)
      JUDGE_API_KEY="$2"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --skip-prompt-pipeline)
      SKIP_PROMPT_PIPELINE="true"
      shift 1
      ;;
    --no-prompt-resume)
      PROMPT_RESUME="false"
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

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "Required file not found: $path" >&2
    exit 1
  fi
}

seed_prompt_cache_for_track() {
  local track="$1"
  local src_dir="$SEED_PROMPT_CACHE_ROOT/$track"
  local dst_dir="$PROMPT_OUTPUT_ROOT/$track"
  local cache_name="recipe_prompt_library_cache.jsonl"
  local src_path="$src_dir/$cache_name"
  local dst_path="$dst_dir/$cache_name"

  mkdir -p "$dst_dir"
  if [[ -f "$src_path" && ! -f "$dst_path" ]]; then
    cp "$src_path" "$dst_path"
    echo "[seed-cache] track=$track copied $src_path -> $dst_path"
  fi
}

echo "[stage 1/5] filter domain-filtered corpus to <= ${MAX_TEXT_LENGTH} chars"
require_file "$SOURCE_FILTERED_PATH"
"$PYTHON_BIN" -m cdrbench.prepare_data.filter_domain_filtered_by_length \
  --input-path "$SOURCE_FILTERED_PATH" \
  --output-dir "$FILTERED_OUTPUT_DIR" \
  --max-text-length "$MAX_TEXT_LENGTH"

echo "[stage 2/5] mine recipe families"
"$PYTHON_BIN" -m cdrbench.prepare_data.mine_domain_workflows \
  --tagged-dir "$TAGGED_DIR" \
  --domains-config configs/domains.yaml \
  --output-dir "$RECIPE_MINING_OUTPUT_DIR" \
  --domain-field assigned_domain \
  --min-active-mappers 2 \
  --min-support 5 \
  --min-workflow-support 5 \
  --min-support-ratio 0.02 \
  --min-combo-len 2 \
  --max-combo-len 5 \
  --top-k 50 \
  --max-families-per-domain 6 \
  --max-workflows-per-family 8 \
  --max-text-length "$MAX_TEXT_LENGTH"

echo "[stage 3/5] materialize deterministic recipe libraries"
"$PYTHON_BIN" -m cdrbench.prepare_data.materialize_domain_workflows \
  --workflow-mining-dir "$RECIPE_MINING_OUTPUT_DIR" \
  --filtered-path "$FILTERED_OUTPUT_DIR/all.jsonl" \
  --output-dir "$RECIPE_LIBRARY_OUTPUT_DIR"

echo "[stage 4/5] materialize benchmark instances and deterministic references"
"$PYTHON_BIN" -m cdrbench.prepare_data.materialize_benchmark_instances \
  --workflow-library-dir "$RECIPE_LIBRARY_OUTPUT_DIR" \
  --filtered-path "$FILTERED_OUTPUT_DIR/all.jsonl" \
  --output-dir "$BENCHMARK_OUTPUT_DIR" \
  --target-drop-rate 0.5 \
  --max-candidate-records 0 \
  --max-input-chars "$MAX_TEXT_LENGTH" \
  --min-positive-ratio-threshold 0.001 \
  --zero-ratio-threshold-policy min-positive \
  --max-instances-per-variant 50 \
  --max-order-groups-per-family 30 \
  --max-atomic-candidate-records 1000 \
  --max-atomic-instances-per-op 10 \
  --min-keep 5 \
  --min-drop 5 \
  --min-order-sensitive-groups 5 \
  --min-atomic-keep 3 \
  --min-atomic-drop 3

if [[ "$SKIP_PROMPT_PIPELINE" == "true" ]]; then
  echo "[complete] recipe data pipeline finished without prompt generation"
  echo "[complete] benchmark_dir=$BENCHMARK_OUTPUT_DIR"
  exit 0
fi

echo "[stage 5/5] build prompt libraries and eval-ready prompt tracks"
for track in "${TRACKS[@]}"; do
  case "$track" in
    atomic_ops|main|order_sensitivity)
      seed_prompt_cache_for_track "$track"
      ;;
    *)
      echo "Unsupported track in --tracks: $track" >&2
      exit 1
      ;;
  esac
done

prompt_cmd=(
  ./scripts/run_prompt_pipeline_all_tracks.sh
  --benchmark-dir "$BENCHMARK_OUTPUT_DIR"
  --output-root "$PROMPT_OUTPUT_ROOT"
  --prompt-source "$PROMPT_SOURCE"
  --tracks "$TRACKS_CSV"
  --python-bin "$PYTHON_BIN"
)
if [[ "$PROMPT_RESUME" != "true" ]]; then
  prompt_cmd+=(--no-resume)
fi
if [[ -n "$MODEL" ]]; then
  prompt_cmd+=(--model "$MODEL")
fi
if [[ -n "$BASE_URL" ]]; then
  prompt_cmd+=(--base-url "$BASE_URL")
fi
if [[ -n "$API_KEY" ]]; then
  prompt_cmd+=(--api-key "$API_KEY")
fi
if [[ -n "$JUDGE_MODEL" ]]; then
  prompt_cmd+=(--judge-model "$JUDGE_MODEL")
fi
if [[ -n "$JUDGE_BASE_URL" ]]; then
  prompt_cmd+=(--judge-base-url "$JUDGE_BASE_URL")
fi
if [[ -n "$JUDGE_API_KEY" ]]; then
  prompt_cmd+=(--judge-api-key "$JUDGE_API_KEY")
fi
"${prompt_cmd[@]}"

echo "[complete] recipe pipeline finished"
echo "[complete] filtered_dir=$FILTERED_OUTPUT_DIR"
echo "[complete] recipe_mining_dir=$RECIPE_MINING_OUTPUT_DIR"
echo "[complete] recipe_library_dir=$RECIPE_LIBRARY_OUTPUT_DIR"
echo "[complete] benchmark_dir=$BENCHMARK_OUTPUT_DIR"
echo "[complete] prompt_output_root=$PROMPT_OUTPUT_ROOT"
