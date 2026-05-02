#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8901}"
PID_FILE="/tmp/vllm_${PORT}.pid"

if [[ -f "${PID_FILE}" ]]; then
  PID="$(cat "${PID_FILE}")"
  echo "[stop_vllm] Stopping vLLM PID=${PID} (port=${PORT})"
  kill "${PID}" 2>/dev/null || true
  rm -f "${PID_FILE}"
else
  echo "[stop_vllm] No PID file found for port ${PORT}"
fi
