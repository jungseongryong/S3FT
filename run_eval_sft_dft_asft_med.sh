#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ASFT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface/}"
export TOKENIZERS_PARALLELISM=false

PYTHON="${PYTHON:-/tmp/adasd-venv/bin/python}"
RUN_ROOT="/workspace/ASFT/output/qwen3_4b_medmcqa_10k_paper"
TEST_DATA_DIR="/workspace/ASFT/eval/medeval/test_data"
RESULT_DIR="/workspace/ASFT/eval_results"
LOG_DIR="/workspace/ASFT/logs"

mkdir -p "${RESULT_DIR}" "${LOG_DIR}"

run_eval() {
  local mode="$1"
  local model_dir="${RUN_ROOT}/${mode}"
  local output_json="${RESULT_DIR}/qwen3_4b_medmcqa_10k_${mode}_med_eval.json"
  local log_file="${LOG_DIR}/eval_qwen3_4b_medmcqa_10k_${mode}.out"

  echo "[$(date -Is)] START eval mode=${mode} model=${model_dir}" | tee -a "${log_file}"
  "${PYTHON}" eval/medeval/run_med_eval.py \
    --model "${model_dir}" \
    --test_data_dir "${TEST_DATA_DIR}" \
    --output_json "${output_json}" \
    2>&1 | tee -a "${log_file}"
  echo "[$(date -Is)] DONE eval mode=${mode} output=${output_json}" | tee -a "${log_file}"
}

run_eval sft
run_eval dft
run_eval asft

