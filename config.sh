#!/bin/bash
# config.sh

# ------- dynamic variables
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ------- user defined
export SCRATCH_DIR="/iopsstor/scratch/cscs/$USER"   
export TRITON_CACHE_DIR="$SCRATCH_DIR/triton_cache"

export ENVIRONMENT="my_finetune"  # <-- put your env from .edf here




# Create necessary directories
mkdir -p "$TRITON_CACHE_DIR"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/results"
mkdir -p "$PROJECT_DIR/output"
