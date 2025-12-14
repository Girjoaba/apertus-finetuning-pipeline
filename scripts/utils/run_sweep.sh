#!/usr/bin/env bash
set -euo pipefail

echo "WANDB_SWEEP_ID=${WANDB_SWEEP_ID:-}"
echo "WANDB_RUN_ID=${WANDB_RUN_ID:-}"

RUN_ID="${WANDB_RUN_ID:-manual}"
OUTDIR="output/apertus_lora/sweeps/${RUN_ID}"
RUN_NAME="ap-medqa-sweep-${RUN_ID}"

python sft_train.py \
  --config configs/sft_lora_8B.yaml \
  --output_dir "${OUTDIR}" \
  --run_name "${RUN_NAME}" \
  "$@"
