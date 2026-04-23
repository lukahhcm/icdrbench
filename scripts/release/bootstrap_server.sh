#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a server to run this project by pulling code from GitHub and curated JSONL data from Hugging Face.
#
# Example:
#   HF_TOKEN=hf_xxx ./scripts/release/bootstrap_server.sh \
#     --github-url https://github.com/yourname/cdrbench.git \
#     --hf-dataset yourname/cdrbench-raw \
#     --workdir ~/work/cdrbench

usage() {
  cat <<'EOF'
Usage:
  bootstrap_server.sh --github-url <url> --hf-dataset <namespace/name> [--workdir <dir>] [--branch <name>] [--run-probe]

Required:
  --github-url      GitHub repository URL
  --hf-dataset      Hugging Face dataset repo id, e.g. yourname/cdrbench-raw

Optional:
  --workdir         Local workspace directory (default: ~/work/cdrbench)
  --branch          Branch to checkout (default: main)
  --run-probe       Immediately run Data-Juicer per-op probe after setup

Environment:
  HF_TOKEN          Optional. Required for private HF datasets.
  CDRBENCH_DATA_JUICER_ROOT / ICDRBENCH_DATA_JUICER_ROOT / DATA_JUICER_ROOT
                    Optional. Path to a Data-Juicer repo checkout. Needed for --run-probe
                    when Data-Juicer is not located at ./data-juicer
EOF
}

GITHUB_URL=""
HF_DATASET=""
WORKDIR="$HOME/work/cdrbench"
BRANCH="main"
DATA_SUBDIR="raw"
RUN_PROBE="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --github-url)
      GITHUB_URL="$2"
      shift 2
      ;;
    --hf-dataset)
      HF_DATASET="$2"
      shift 2
      ;;
    --workdir)
      WORKDIR="$2"
      shift 2
      ;;
    --branch)
      BRANCH="$2"
      shift 2
      ;;
    --run-probe)
      RUN_PROBE="true"
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

if [[ -z "$GITHUB_URL" || -z "$HF_DATASET" ]]; then
  echo "--github-url and --hf-dataset are required" >&2
  usage
  exit 1
fi

command -v git >/dev/null 2>&1 || { echo "git not found" >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 not found" >&2; exit 1; }
command -v uv >/dev/null 2>&1 || { echo "uv not found" >&2; exit 1; }

mkdir -p "$WORKDIR"
cd "$WORKDIR"

PROJECT_DIR="$WORKDIR/repo"
if [[ ! -d "$PROJECT_DIR/.git" ]]; then
  git clone "$GITHUB_URL" "$PROJECT_DIR"
fi

cd "$PROJECT_DIR"
git fetch --all --prune
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"

uv venv .venv-ops --python 3.11
uv pip install --python .venv-ops/bin/python -e .
uv pip install --python .venv-ops/bin/python -U huggingface_hub py-data-juicer

.venv-ops/bin/python scripts/release/download_hf_jsonl.py \
  --repo-id "$HF_DATASET" \
  --repo-root "$PROJECT_DIR"

if [[ "$RUN_PROBE" == "true" ]]; then
  PYTHONPATH=src .venv-ops/bin/python scripts/prepare_data/run_dj_per_op_probe.py \
    --execute \
    --resume \
    --np 2 \
    --mapper-max-records 1000 \
    --filter-max-records 5000 \
    --summary-csv data/processed/dj_per_op_probe/summary_full.csv
fi

echo "Server bootstrap complete."
echo "Project dir: $PROJECT_DIR"
echo "Now you can run, for example:"
echo "  PYTHONPATH=src .venv-ops/bin/python scripts/prepare_data/tag_and_assign_domains.py --resume"
echo "  PYTHONPATH=src .venv-ops/bin/python scripts/prepare_data/run_dj_per_op_probe.py --execute --resume --summary-csv data/processed/dj_per_op_probe/summary_full.csv"
