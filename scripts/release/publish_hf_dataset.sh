#!/usr/bin/env bash
set -euo pipefail

# Publish curated JSONL files to a Hugging Face dataset repository.
#
# Example:
#   HF_TOKEN=hf_xxx ./scripts/release/publish_hf_dataset.sh \
#     --repo-id yourname/cdrbench-raw

usage() {
  cat <<'EOF'
Usage:
  publish_hf_dataset.sh --repo-id <namespace/name> [--manifest <path>] [--target-subdir <name>] [--commit-message <msg>] [--delete-extra]

Required:
  --repo-id         Hugging Face dataset repo id, e.g. yourname/cdrbench-raw

Optional:
  --manifest        File list manifest (default: configs/release_jsonl_manifest.txt)
  --target-subdir   Subdirectory inside dataset repo (default: raw)
  --commit-message  Git commit message (default: sync data snapshot)
  --delete-extra    Delete files in target-subdir that are not in manifest

Environment:
  HF_TOKEN          Hugging Face access token with write permission
EOF
}

REPO_ID=""
MANIFEST="configs/release_jsonl_manifest.txt"
TARGET_SUBDIR="raw"
COMMIT_MESSAGE="sync data snapshot"
DELETE_EXTRA="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-id)
      REPO_ID="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --target-subdir)
      TARGET_SUBDIR="$2"
      shift 2
      ;;
    --commit-message)
      COMMIT_MESSAGE="$2"
      shift 2
      ;;
    --delete-extra)
      DELETE_EXTRA="true"
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

if [[ -z "$REPO_ID" ]]; then
  echo "--repo-id is required" >&2
  usage
  exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "HF_TOKEN is not set" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON_BIN="$REPO_ROOT/.venv-ops/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

PY_ARGS=(
  --repo-id "$REPO_ID"
  --repo-root "$REPO_ROOT"
  --manifest "$MANIFEST"
  --target-subdir "$TARGET_SUBDIR"
  --commit-message "$COMMIT_MESSAGE"
)

if [[ "$DELETE_EXTRA" == "true" ]]; then
  PY_ARGS+=(--delete-extra)
fi

PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
  "$PYTHON_BIN" -m cdrbench.release.publish_hf_jsonl "${PY_ARGS[@]}"
