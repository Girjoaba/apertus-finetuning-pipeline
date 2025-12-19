#!/bin/bash
#SBATCH --job-name=eval_model_template
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --partition=normal
#SBATCH --account=a-a]$

# =============================================================================
# MedQA Evaluation SLURM Template
# =============================================================================
# 
# This is a template for running model evaluation on the Alps cluster.
# Copy this file and modify the CONFIGURATION section for your model.
#
# Usage:
#   1. Copy this file: cp eval_template.sbatch eval_mymodel.sbatch
#   2. Modify the CONFIGURATION section below
#   3. Submit: sbatch eval_mymodel.sbatch
#
# GPU Requirements:
#   - 8B models:  1 node, 1-4 GPUs (~16GB VRAM each)
#   - 70B models: 4 nodes, 16 GPUs total (~80GB VRAM each)
#
# =============================================================================

# =============================================================================
# CONFIGURATION - Modify these for your model
# =============================================================================

# Base model: HuggingFace model ID or local path
BASE_MODEL="swiss-ai/Apertus-8B-Instruct-2509"

# LoRA adapter: HuggingFace ID, local path, or leave empty for base model only
ADAPTER=""

# Display name for results (optional, leave empty for auto-generated)
MODEL_NAME=""

# Output file (optional, leave empty for auto-generated)
OUTPUT_FILE=""

# Batch size (reduce if you get out-of-memory errors)
BATCH_SIZE=8

# Maximum samples (leave empty for all 1273 test samples)
MAX_SAMPLES=""

# Maximum tokens to generate per question
MAX_NEW_TOKENS=128

# =============================================================================
# END CONFIGURATION - Don't modify below unless you know what you're doing
# =============================================================================

# Project directory
PROJECT_DIR="/users/lbaumberger/scratch/lsai-project/apertus-finetuning-pipeline"

# Container image
CONTAINER_IMAGE="${PROJECT_DIR}/apertus_finetune.toml"

echo "============================================================"
echo "MedQA Model Evaluation"
echo "============================================================"
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Nodes:       ${SLURM_JOB_NUM_NODES}"
echo "GPUs:        ${SLURM_GPUS_PER_NODE} per node"
echo "Base Model:  ${BASE_MODEL}"
echo "Adapter:     ${ADAPTER:-None}"
echo "Model Name:  ${MODEL_NAME:-Auto}"
echo "Batch Size:  ${BATCH_SIZE}"
echo "Max Samples: ${MAX_SAMPLES:-All}"
echo "============================================================"

# Build command with optional arguments
CMD="cd ${PROJECT_DIR} && python scripts/evaluate_template.py"
CMD="${CMD} --base_model ${BASE_MODEL}"
CMD="${CMD} --batch_size ${BATCH_SIZE}"
CMD="${CMD} --max_new_tokens ${MAX_NEW_TOKENS}"

# Add optional arguments if set
if [ -n "${ADAPTER}" ]; then
    CMD="${CMD} --adapter ${ADAPTER}"
fi

if [ -n "${MODEL_NAME}" ]; then
    CMD="${CMD} --model_name '${MODEL_NAME}'"
fi

if [ -n "${OUTPUT_FILE}" ]; then
    CMD="${CMD} --output ${OUTPUT_FILE}"
fi

if [ -n "${MAX_SAMPLES}" ]; then
    CMD="${CMD} --max_samples ${MAX_SAMPLES}"
fi

echo "Running command:"
echo "${CMD}"
echo "============================================================"

# Run with container
srun --container-image="${CONTAINER_IMAGE}" \
     bash -c "${CMD}"

echo "============================================================"
echo "Evaluation complete!"
echo "Check results in the results/ directory"
echo "============================================================"
