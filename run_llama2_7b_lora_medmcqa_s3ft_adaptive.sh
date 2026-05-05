#!/usr/bin/env bash
set -euo pipefail

cd /workspace/ASFT

PYTHON="${PYTHON:-/workspace/ASFT/.venv_asft/bin/python}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-/workspace/models/meta-llama-Llama-2-7b-hf}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/workspace/.cache/huggingface/hub}"
export TOKENIZERS_PARALLELISM=false

SIZE="${SIZE:-10k}"
DATA_PATH="${DATA_PATH:-/workspace/ASFT/data/train_medmcqa_alpaca_${SIZE}.jsonl}"
OUT_ROOT="${OUT_ROOT:-/workspace/ASFT/output/llama2_7b_medmcqa_lora_paper}"
LOG_DIR="${LOG_DIR:-/workspace/ASFT/logs}"
S3FT_TEACHER="${S3FT_TEACHER:-base}"
S3FT_KL_DIRECTION="${S3FT_KL_DIRECTION:-fkl}"
S3FT_BETA="${S3FT_BETA:-0.3}"
S3FT_TAU="${S3FT_TAU:-1.0}"
RUN_SUFFIX="${RUN_SUFFIX:-softdecay}"

RUN_NAME="${SIZE}_s3ft_adaptive_${RUN_SUFFIX}_${S3FT_TEACHER}_${S3FT_KL_DIRECTION}_beta${S3FT_BETA}"
OUT_DIR="${OUT_ROOT}/${RUN_NAME}"
LOG_FILE="${LOG_DIR}/train_llama2_7b_medmcqa_${RUN_NAME}.out"

mkdir -p "${OUT_ROOT}" "${LOG_DIR}"

echo "[$(date -Is)] START llama2 medmcqa ${RUN_NAME} -> ${OUT_DIR}" | tee "${LOG_FILE}"
set +e
"${PYTHON}" train_v2.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --data_path "${DATA_PATH}" \
  --model_max_length 512 \
  --global_batch_size 64 \
  --per_device_train_batch_size 4 \
  --num_train_epochs 3 \
  --learning_rate 5e-4 \
  --precision bf16 \
  --gradient_checkpointing True \
  --use_lora True \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --mode s3ft_adaptive \
  --s3ft_teacher "${S3FT_TEACHER}" \
  --s3ft_kl_direction "${S3FT_KL_DIRECTION}" \
  --s3ft_beta "${S3FT_BETA}" \
  --s3ft_tau "${S3FT_TAU}" \
  --output_dir "${OUT_DIR}" \
  2>&1 | tee -a "${LOG_FILE}"
status="${PIPESTATUS[0]}"
set -e
echo "[$(date -Is)] EXIT llama2 medmcqa ${RUN_NAME} status=${status}" | tee -a "${LOG_FILE}"
if [[ "${status}" -ne 0 ]]; then
  exit "${status}"
fi
echo "[$(date -Is)] DONE llama2 medmcqa ${RUN_NAME}" | tee -a "${LOG_FILE}"
