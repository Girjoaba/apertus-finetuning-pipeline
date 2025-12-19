#!/bin/bash
#SBATCH --job-name=eval_70B_lora_hf
#SBATCH --output=logs/eval_70B_lora_hf_%j.out
#SBATCH --error=logs/eval_70B_lora_hf_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --account=large-sc-2
#SBATCH --exclusive

set -eo pipefail

# =============================================================================
# Evaluate Apertus-70B + LSAIE-TEAM LoRA on MedQA
# =============================================================================

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

# HuggingFace cache settings
export HF_HOME="${HF_HOME:-$SCRATCH_DIR/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"

echo "============================================================"
echo "MedQA Evaluation - Apertus-70B + LoRA (HuggingFace)"
echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Nodes:       ${SLURM_JOB_NUM_NODES}"
echo "GPUs:        ${SLURM_GPUS_PER_NODE} per node"
echo "Total GPUs:  $((SLURM_JOB_NUM_NODES * 4))"
echo "Base Model:  swiss-ai/Apertus-70B-Instruct-2509"
echo "Adapter:     LSAIE-TEAM/Apertus-70B-Instruct-LoRA"
echo "Project dir: ${PROJECT_DIR}"
echo "Environment: ${ENVIRONMENT}"
echo "============================================================"

CMD="
    cd $PROJECT_DIR
    python scripts/evaluate_finetuned_70B_hf.py
"

srun --environment=$ENVIRONMENT \
    bash -c "$CMD"

echo "============================================================"
echo "Evaluation complete!"
echo "Results: results/apertus_70b_lora_hf_test.json"
echo "============================================================"
