#!/bin/bash
#SBATCH --job-name=eval_finetuned_70B
#SBATCH --partition=normal
#SBATCH --account=large-sc-2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_finetuned_70B_%j.out
#SBATCH --error=logs/eval_finetuned_70B_%j.err
#SBATCH --exclusive

set -eo pipefail

# Source project configuration
if [ -f "$SLURM_SUBMIT_DIR/config.sh" ]; then
    source "$SLURM_SUBMIT_DIR/config.sh"
else
    echo "Error: config.sh not found in $SLURM_SUBMIT_DIR"
    exit 1
fi

# Network and distributed settings
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export FI_PROVIDER="cxi"
export NCCL_NET_GDR_LEVEL=3
export OMP_NUM_THREADS=16

# HuggingFace cache settings (from config.sh)
export HF_HOME="${HF_HOME:-$SCRATCH_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

echo "=============================================="
echo "MedQA Test Evaluation - Finetuned Apertus-70B (LoRA)"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NUM_NODES"
echo "GPUs per node: 4"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * 4))"
echo "Project dir: $PROJECT_DIR"
echo "HF cache: $HF_HOME"
echo "Adapter: LSAIE-TEAM/apertus-medqa-lora-70B-tuned-parameters"
echo "=============================================="

# Run evaluation script
# Using srun with container environment
# Must cd to PROJECT_DIR inside the container as it mounts at different path
# HuggingFace authentication is handled by the Python script
# (reads token from ~/.cache/huggingface/token)
CMD="
    cd $PROJECT_DIR
    python scripts/evaluate_finetuned_70B.py \
        --batch_size 4 \
        --output results/finetuned_70B_test.json
"

srun --environment=$ENVIRONMENT \
    bash -c "$CMD"

echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: results/finetuned_70B_test.json"
echo "=============================================="
