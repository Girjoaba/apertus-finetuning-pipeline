#!/bin/bash
#SBATCH --job-name=LSAIE_lora_improved
#SBATCH --partition=normal
#SBATCH --account=large-sc-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/finetune_lora8B_improved_%j.out
#SBATCH --error=logs/finetune_lora8B_improved_%j.err

# =============================================================================
# Improved 8B LoRA Training Script
# Changes from original:
#   - Longer time limit (3h vs 1h) for more training steps
#   - Uses improved config with better hyperparameters
# =============================================================================

set -eo pipefail

# Load project configuration
if [ -f "$SLURM_SUBMIT_DIR/config.sh" ]; then
    source "$SLURM_SUBMIT_DIR/config.sh"
else
    echo "Error: config.sh not found in $SLURM_SUBMIT_DIR"
    exit 1
fi

# Print job info
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Start time: $(date)"
echo "Config: configs/sft_lora_8B_improved.yaml"
echo "=============================================="

CMD="
    echo \"[srun] rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID\"

    cd $PROJECT_DIR

    # Run training with improved config
    python sft_train.py --config configs/sft_lora_8B_improved.yaml
"

srun \
    --environment=$ENVIRONMENT \
    bash -c "$CMD"

echo "=============================================="
echo "End time: $(date)"
echo "=============================================="
