#!/bin/bash
#SBATCH --job-name=eval_base_8B
#SBATCH --partition=normal
#SBATCH --account=large-sc-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_base_8B_%j.out
#SBATCH --error=logs/eval_base_8B_%j.err

set -eo pipefail

# Source project configuration
if [ -f "$SLURM_SUBMIT_DIR/config.sh" ]; then
    source "$SLURM_SUBMIT_DIR/config.sh"
else
    echo "Error: config.sh not found in $SLURM_SUBMIT_DIR"
    exit 1
fi

# Network settings
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export FI_PROVIDER="cxi"
export OMP_NUM_THREADS=16

# HuggingFace cache settings (from config.sh)
export HF_HOME="${HF_HOME:-$SCRATCH_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

echo "=============================================="
echo "MedQA Test Evaluation - Base Apertus-8B"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: 1"
echo "Project dir: $PROJECT_DIR"
echo "HF cache: $HF_HOME"
echo "=============================================="

# Run evaluation script
CMD="
    cd $PROJECT_DIR
    python scripts/evaluate_base_8B.py \
        --batch_size 8 \
        --output results/base_8B_test.json \
        --log_file logs/eval_base_8B_detailed.log
"

srun --environment=$ENVIRONMENT \
    bash -c "$CMD"

echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: results/base_8B_test.json"
echo "Detailed log: logs/eval_base_8B_detailed.log"
echo "=============================================="
