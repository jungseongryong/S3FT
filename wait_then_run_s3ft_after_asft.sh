#!/usr/bin/env bash
set -euo pipefail

ASFT_PID="${ASFT_PID:-286084}"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-60}"

echo "[$(date -Is)] Waiting for ASFT PID ${ASFT_PID}"
while kill -0 "${ASFT_PID}" 2>/dev/null; do
  sleep "${CHECK_INTERVAL_SECONDS}"
done

echo "[$(date -Is)] ASFT PID ${ASFT_PID} finished; starting S3FT"
exec bash -x /workspace/ASFT/run_medmcqa_10k_s3ft_paper.sh
