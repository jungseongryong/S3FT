#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ASFT

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export TOKENIZERS_PARALLELISM=false

PYTHON="${PYTHON:-/tmp/adasd-venv/bin/python}"
MODEL="${MODEL:-Qwen/Qwen3-4B}"
DATA_PATH="${DATA_PATH:-/workspace/ASFT/data/train_medmcqa_alpaca_10k.jsonl}"
RUN_ROOT="${RUN_ROOT:-/workspace/ASFT/output/qwen3_4b_medmcqa_10k_paper}"

MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-512}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-3}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
KL_WEIGHT="${KL_WEIGHT:-0.05}"
PRECISION="${PRECISION:-bf16}"

mkdir -p /workspace/ASFT/logs "${RUN_ROOT}"

run_one() {
  local mode="$1"
  local output_dir="${RUN_ROOT}/${mode}"
  local log_file="/workspace/ASFT/logs/train_qwen3_4b_medmcqa_10k_${mode}_paper.out"

  echo "[$(date -Is)] START mode=${mode}"
  "${PYTHON}" train_v2.py \
    --model_name_or_path "${MODEL}" \
    --mode "${mode}" \
    --data_path "${DATA_PATH}" \
    --output_dir "${output_dir}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --global_batch_size "${GLOBAL_BATCH_SIZE}" \
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    --learning_rate "${LEARNING_RATE}" \
    --kl_weight "${KL_WEIGHT}" \
    --precision "${PRECISION}" \
    --gradient_checkpointing True \
    > "${log_file}" 2>&1
  echo "[$(date -Is)] DONE mode=${mode}"
}

run_one dft
run_one asft

echo "[$(date -Is)] DFT+ASFT DONE"
echo "Outputs: ${RUN_ROOT}"
