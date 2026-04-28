#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_prompt_pipeline_all_tracks.sh [options]

This script runs the full prompt pipeline sequentially for:
  1. atomic_ops
  2. main
  3. order_sensitivity

For each track it will:
  1. generate recipe-level prompt candidates
  2. judge/check the accepted prompt pool
  3. build eval-ready sample files by combining prompts with benchmark rows
  4. check the eval output summary

Options:
  --benchmark-dir <path>                  Benchmark JSONL directory. Default: data/benchmark
  --output-root <path>                    Root output directory. Default: data/benchmark_prompts
  --prompt-config <path>                  Prompt config YAML. Default: configs/workflow_prompting.yaml
  --prompt-source <llm|template>          Prompt source. Default: llm
  --variants-per-recipe <int>             Style presets requested per recipe. Default: 11
  --candidates-per-style <int>            Candidate prompts generated per style. Default: 3
  --prompt-variants-per-sample <int>      Distinct styles sampled per eval sample. Default: 3
  --prompt-sampling-seed <int>            Deterministic sampling seed. Default: 0
  --min-prompt-variants-per-sample <int>  Minimum distinct accepted styles required. Default: 3
  --tracks <csv>                          Comma-separated subset. Default: atomic_ops,main,order_sensitivity
  --model <name>                          Generation model override
  --base-url <url>                        Generation base URL override
  --api-key <key>                         Generation API key override
  --judge-model <name>                    Judge model override
  --judge-base-url <url>                  Judge base URL override
  --judge-api-key <key>                   Judge API key override
  --python-bin <path>                     Python executable. Default: .venv-ops/bin/python or python3
  --no-resume                             Disable recipe-level resume cache reuse
  -h, --help                              Show this help

Examples:
  ./scripts/run_prompt_pipeline_all_tracks.sh

  ./scripts/run_prompt_pipeline_all_tracks.sh \
    --prompt-source llm \
    --model qwen-plus \
    --base-url https://dashscope.aliyuncs.com/compatible-mode/v1
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

BENCHMARK_DIR="data/benchmark"
OUTPUT_ROOT="data/benchmark_prompts"
PROMPT_CONFIG="configs/workflow_prompting.yaml"
PROMPT_SOURCE="llm"
VARIANTS_PER_RECIPE=11
CANDIDATES_PER_STYLE=3
PROMPT_VARIANTS_PER_SAMPLE=3
PROMPT_SAMPLING_SEED=0
MIN_PROMPT_VARIANTS_PER_SAMPLE=3
TRACKS_CSV="atomic_ops,main,order_sensitivity"
RESUME="true"

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
    --benchmark-dir)
      BENCHMARK_DIR="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --prompt-config)
      PROMPT_CONFIG="$2"
      shift 2
      ;;
    --prompt-source)
      PROMPT_SOURCE="$2"
      shift 2
      ;;
    --variants-per-recipe|--variants-per-workflow)
      VARIANTS_PER_RECIPE="$2"
      shift 2
      ;;
    --candidates-per-style)
      CANDIDATES_PER_STYLE="$2"
      shift 2
      ;;
    --prompt-variants-per-sample)
      PROMPT_VARIANTS_PER_SAMPLE="$2"
      shift 2
      ;;
    --prompt-sampling-seed)
      PROMPT_SAMPLING_SEED="$2"
      shift 2
      ;;
    --min-prompt-variants-per-sample)
      MIN_PROMPT_VARIANTS_PER_SAMPLE="$2"
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
    --no-resume)
      RESUME="false"
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

track_input_path() {
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

check_generation_summary() {
  local summary_path="$1"
  local track="$2"
  "$PYTHON_BIN" - "$summary_path" "$track" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
track = sys.argv[2]
if not summary_path.exists():
    raise SystemExit(f"Missing generation summary: {summary_path}")

rows = [json.loads(line) for line in summary_path.open('r', encoding='utf-8') if line.strip()]
entry = next((row for row in rows if row.get('track') == track), None)
if entry is None:
    raise SystemExit(f"No generation summary row found for track={track} in {summary_path}")

accepted_recipe_count = int(entry.get('accepted_recipe_count', entry.get('accepted_workflow_count', 0)) or 0)
accepted_candidate_count = int(entry.get('accepted_candidate_count', 0) or 0)
print(
    f"[check:generation] {track}: "
    f"input_rows={entry.get('input_rows', 0)} "
    f"recipes={entry.get('recipe_count', entry.get('workflow_count', 0))} "
    f"accepted_recipes={accepted_recipe_count} "
    f"accepted_candidates={accepted_candidate_count} "
    f"skipped_rows={entry.get('skipped_rows', 0)}",
    flush=True,
)
if accepted_recipe_count <= 0 or accepted_candidate_count <= 0:
    raise SystemExit(f"Prompt pool check failed for {track}: no accepted prompts were saved.")
PY
}

check_eval_summary() {
  local summary_path="$1"
  local track="$2"
  "$PYTHON_BIN" - "$summary_path" "$track" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
track = sys.argv[2]
if not summary_path.exists():
    raise SystemExit(f"Missing eval summary: {summary_path}")

rows = [json.loads(line) for line in summary_path.open('r', encoding='utf-8') if line.strip()]
entry = next((row for row in rows if row.get('track') == track), None)
if entry is None:
    raise SystemExit(f"No eval summary row found for track={track} in {summary_path}")

kept_rows = int(entry.get('kept_rows', 0) or 0)
print(
    f"[check:eval] {track}: "
    f"input_rows={entry.get('input_rows', 0)} "
    f"kept_rows={kept_rows} "
    f"missing_pool_rows={entry.get('missing_pool_rows', 0)} "
    f"insufficient_style_rows={entry.get('insufficient_style_rows', 0)}",
    flush=True,
)
if kept_rows <= 0:
    raise SystemExit(f"Eval build check failed for {track}: no eval rows were produced.")
PY
}

run_track() {
  local track="$1"
  local benchmark_input
  benchmark_input="$(track_input_path "$track")"
  require_file "$benchmark_input"
  require_file "$PROMPT_CONFIG"

  local track_output_dir="$OUTPUT_ROOT/$track"
  local cache_path="$track_output_dir/recipe_prompt_library_cache.jsonl"
  local library_path="$track_output_dir/recipe_prompt_library.jsonl"
  local eval_output_dir="$track_output_dir/eval"

  mkdir -p "$track_output_dir" "$eval_output_dir"

  echo "[run] track=$track step=generate+judge output_dir=$track_output_dir"
  local generate_cmd=(
    "$PYTHON_BIN" -m cdrbench.prompting.generate_workflow_prompt_library
    --benchmark-dir "$BENCHMARK_DIR"
    --output-dir "$track_output_dir"
    --prompt-config "$PROMPT_CONFIG"
    --prompt-source "$PROMPT_SOURCE"
    --tracks "$track"
    --variants-per-recipe "$VARIANTS_PER_RECIPE"
    --candidates-per-style "$CANDIDATES_PER_STYLE"
    --cache-path "$cache_path"
  )
  if [[ "$RESUME" == "true" ]]; then
    generate_cmd+=(--resume)
  fi
  if [[ -n "$MODEL" ]]; then
    generate_cmd+=(--model "$MODEL")
  fi
  if [[ -n "$BASE_URL" ]]; then
    generate_cmd+=(--base-url "$BASE_URL")
  fi
  if [[ -n "$API_KEY" ]]; then
    generate_cmd+=(--api-key "$API_KEY")
  fi
  if [[ -n "$JUDGE_MODEL" ]]; then
    generate_cmd+=(--judge-model "$JUDGE_MODEL")
  fi
  if [[ -n "$JUDGE_BASE_URL" ]]; then
    generate_cmd+=(--judge-base-url "$JUDGE_BASE_URL")
  fi
  if [[ -n "$JUDGE_API_KEY" ]]; then
    generate_cmd+=(--judge-api-key "$JUDGE_API_KEY")
  fi
  "${generate_cmd[@]}"

  check_generation_summary "$track_output_dir/prompt_generation_summary.jsonl" "$track"
  require_file "$library_path"

  echo "[run] track=$track step=build-eval output_dir=$eval_output_dir"
  local build_cmd=(
    "$PYTHON_BIN" -m cdrbench.prompting.build_eval_prompt_tracks
    --benchmark-dir "$BENCHMARK_DIR"
    --prompt-library "$library_path"
    --output-dir "$eval_output_dir"
    --tracks "$track"
    --prompt-variants-per-sample "$PROMPT_VARIANTS_PER_SAMPLE"
    --prompt-sampling-seed "$PROMPT_SAMPLING_SEED"
    --min-prompt-variants-per-sample "$MIN_PROMPT_VARIANTS_PER_SAMPLE"
  )
  "${build_cmd[@]}"

  check_eval_summary "$eval_output_dir/prompt_eval_build_summary.jsonl" "$track"
  echo "[done] track=$track library=$library_path eval_dir=$eval_output_dir"
}

for track in "${TRACKS[@]}"; do
  case "$track" in
    atomic_ops|main|order_sensitivity)
      run_track "$track"
      ;;
    *)
      echo "Unsupported track in --tracks: $track" >&2
      exit 1
      ;;
  esac
done

echo "[complete] prompt pipeline finished for tracks: ${TRACKS[*]}"
echo "[complete] outputs rooted at: $OUTPUT_ROOT"
