#!/bin/bash
# config.sh

# ------- dynamic variables
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------- user defined
export SCRATCH_DIR="/iopsstor/scratch/cscs/$USER"   
export TRITON_CACHE_DIR="$SCRATCH_DIR/triton_cache"

export ENVIRONMENT="my_finetune"  # <-- put your env from .edf here

export WANDB_API_KEY="" # <-- your w&b key here 
export WANDB_DIR="/iopsstor/scratch/cscs/$USER/wandb"
export WANDB_PROJECT="apertus-finetune"

export HF_HOME="/iopsstor/scratch/cscs/$USER/.cache/huggingface"
export TRANSFORMERS_CACHE="/iopsstor/scratch/cscs/$USER/.cache/huggingface"
export HF_DATASETS_CACHE="/iopsstor/scratch/cscs/$USER/.cache/huggingface/datasets"

# Create necessary directories
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/results"
mkdir -p "$PROJECT_DIR/output"
