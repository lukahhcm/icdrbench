#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PATH="${MODEL_PATH:-/mnt/workspace/shared/qwen/Qwen/Qwen3.5-4B}"
MODEL_NAME="${MODEL_NAME:-qwen3_5_4b}"
PORT="${PORT:-8903}"
GPU_IDS="${GPU_IDS:-2}"
TP_SIZE="${TP_SIZE:-1}"
VLLM_USE_MODELSCOPE="${VLLM_USE_MODELSCOPE:-True}" \
  "${SCRIPT_DIR}/../start_vllm.sh" "${MODEL_PATH}" "${MODEL_NAME}" "${PORT}" "${GPU_IDS}" "${TP_SIZE}"
