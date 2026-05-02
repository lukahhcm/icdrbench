#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper around scripts/start_vllm.sh.
# Legacy usage:
#   bash scripts/serve_vllm_openai.sh <model_path> <model_name> [tp_size] [gpu_ids] [port]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="${1:-/mnt/workspace/shared/qwen/Qwen/Qwen3.5-9B}"
MODEL_NAME="${2:-qwen3_5_9b}"
TP_SIZE="${3:-1}"
GPU_IDS="${4:-3}"
PORT="${5:-8904}"

export VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-True}"

exec "${SCRIPT_DIR}/start_vllm.sh" "${MODEL_PATH}" "${MODEL_NAME}" "${PORT}" "${GPU_IDS}" "${TP_SIZE}"
