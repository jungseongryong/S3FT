#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ASFT

PYTHON="${PYTHON:-/workspace/ASFT/.venv_asft/bin/python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/workspace/models/meta-llama-Llama-2-7b-hf}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/.cache/huggingface/hub}"
export TOKENIZERS_PARALLELISM=false

DATA_PATH="${DATA_PATH:-/workspace/ASFT/data/train_medmcqa_alpaca_10k.jsonl}"
OUT_ROOT="${OUT_ROOT:-/workspace/ASFT/output/llama2_7b_medmcqa_10k_lora_paper}"
LOG_DIR="${LOG_DIR:-/workspace/ASFT/logs}"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

COMMON_ARGS=(
  --model_name_or_path "${MODEL_NAME_OR_PATH}"
  --data_path "${DATA_PATH}"
  --model_max_length 512
  --global_batch_size 64
  --per_device_train_batch_size 4
  --num_train_epochs 3
  --learning_rate 5e-4
  --precision bf16
  --gradient_checkpointing True
  --use_lora True
  --lora_r 8
  --lora_alpha 16
  --lora_dropout 0.05
)

run_mode() {
  local mode="$1"
  local out_dir="$2"
  local log_file="$3"
  shift 3

  echo "[$(date -Is)] START ${mode} -> ${out_dir}" | tee "${log_file}"
  set +e
  "${PYTHON}" train_v2.py \
    "${COMMON_ARGS[@]}" \
    --mode "${mode}" \
    --output_dir "${out_dir}" \
    "$@" \
    2>&1 | tee -a "${log_file}"
  local status="${PIPESTATUS[0]}"
  set -e
  echo "[$(date -Is)] EXIT ${mode} status=${status}" | tee -a "${log_file}"
  if [[ "${status}" -ne 0 ]]; then
    return "${status}"
  fi
  echo "[$(date -Is)] DONE ${mode}" | tee -a "${log_file}"
}

run_mode sft \
  "${OUT_ROOT}/sft" \
  "${LOG_DIR}/train_llama2_7b_medmcqa_10k_lora_sft.out"

run_mode dft \
  "${OUT_ROOT}/dft" \
  "${LOG_DIR}/train_llama2_7b_medmcqa_10k_lora_dft.out"

run_mode asft \
  "${OUT_ROOT}/asft" \
  "${LOG_DIR}/train_llama2_7b_medmcqa_10k_lora_asft.out" \
  --kl_weight 0.03 \
  --alpha 0.1

run_mode s3ft \
  "${OUT_ROOT}/s3ft_base_rkl_beta0.3" \
  "${LOG_DIR}/train_llama2_7b_medmcqa_10k_lora_s3ft_base_rkl_beta0.3.out" \
  --s3ft_teacher base \
  --s3ft_kl_direction rkl \
  --s3ft_beta 0.3 \
  --s3ft_tau 1.0
