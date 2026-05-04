#!/usr/bin/env bash
set -euo pipefail

WAIT_PID="${1:-}"
if [[ -z "${WAIT_PID}" ]]; then
  echo "Usage: $0 <pid-to-wait-for>" >&2
  exit 2
fi

cd /workspace/ASFT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface/}"
export TOKENIZERS_PARALLELISM=false

PYTHON="${PYTHON:-/tmp/adasd-venv/bin/python}"
MODEL="/workspace/ASFT/output/qwen3_4b_medmcqa_10k_paper/dft"
TEST_DATA_DIR="/workspace/ASFT/eval/medeval/test_data"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="/workspace/ASFT/eval_results/qwen3_4b_medmcqa_10k_dft_med_eval_rerun_${STAMP}.json"
LOG="/workspace/ASFT/logs/eval_qwen3_4b_medmcqa_10k_dft_rerun_${STAMP}.out"

echo "[$(date -Is)] Waiting for PID ${WAIT_PID} before DFT re-eval" | tee -a "${LOG}"
while kill -0 "${WAIT_PID}" 2>/dev/null; do
  sleep 60
done

echo "[$(date -Is)] START DFT re-eval" | tee -a "${LOG}"
"${PYTHON}" eval/medeval/run_med_eval.py \
  --model "${MODEL}" \
  --test_data_dir "${TEST_DATA_DIR}" \
  --output_json "${OUT}" \
  2>&1 | tee -a "${LOG}"
echo "[$(date -Is)] DONE DFT re-eval output=${OUT}" | tee -a "${LOG}"
