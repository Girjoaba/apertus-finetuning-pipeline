#!/bin/bash
#SBATCH --job-name=LSAIE_lora
#SBATCH --partition=normal
#SBATCH --account=large-sc-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/finetune_apertus_lora8B_%j.out
#SBATCH --error=logs/finetune_apertus_lora8B_%j.err
#SBATCH --array=1-50%10

set -eo pipefail

# Assume you submit the job from the root of your repo
if [ -f "$SLURM_SUBMIT_DIR/config.sh" ]; then
    source "$SLURM_SUBMIT_DIR/config.sh"
else
    echo "Error: config.sh not found in $SLURM_SUBMIT_DIR"
    exit 1
fi

SWEEP_ID="lsae/apertus-finetune/sf84g2tt"

CMD='
set -eo pipefail

echo "[srun] host=$(hostname) user=${USER:-unset} uid=$(id -u)"
cd '"$PROJECT_DIR"'

# Force W&B dirs INSIDE the runtime that runs wandb
export WANDB_DIR="/iopsstor/scratch/cscs/${USER}/wandb"
export WANDB_CACHE_DIR="/iopsstor/scratch/cscs/${USER}/wandb_cache"
export WANDB_CONFIG_DIR="/iopsstor/scratch/cscs/${USER}/wandb_config"

mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR"

echo "WANDB_DIR=$WANDB_DIR"
env | grep -E "^WANDB_" | sort

wandb agent --count 1 '"$SWEEP_ID"'
'


srun \
    --environment=$ENVIRONMENT \
    bash -c "$CMD"
