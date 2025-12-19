#!/bin/bash
#SBATCH --job-name=eval_8B_lora_improved
#SBATCH --output=logs/eval_8B_lora_improved_%j.out
#SBATCH --error=logs/eval_8B_lora_improved_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --partition=normal
#SBATCH --account=large-sc-2

set -eo pipefail

# =============================================================================
# Evaluate Apertus-8B + LoRA Improved (local) on MedQA
# =============================================================================

# Source project configuration
if [ -f "$SLURM_SUBMIT_DIR/config.sh" ]; then
    source "$SLURM_SUBMIT_DIR/config.sh"
else
    echo "Error: config.sh not found in $SLURM_SUBMIT_DIR"
    exit 1
fi

# LoRA adapter path (relative to project dir)
ADAPTER="./output/apertus_lora_8B_improved"

# HuggingFace cache settings
export HF_HOME="${HF_HOME:-$SCRATCH_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

echo "============================================================"
echo "MedQA Evaluation - Apertus-8B + LoRA Improved (local)"
echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Nodes:       ${SLURM_JOB_NUM_NODES}"
echo "GPUs:        ${SLURM_GPUS_PER_NODE} per node"
echo "Adapter:     ${ADAPTER}"
echo "Project dir: ${PROJECT_DIR}"
echo "Environment: ${ENVIRONMENT}"
echo "============================================================"

CMD="
    cd $PROJECT_DIR
    python scripts/evaluate_finetuned_8B_improved.py --adapter ${ADAPTER}
"

srun --environment=$ENVIRONMENT \
    bash -c "$CMD"

echo "============================================================"
echo "Evaluation complete!"
echo "Results: results/apertus_8b_lora_improved_test.json"
echo "============================================================"
