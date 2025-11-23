#!/bin/bash
set -e  # Exit immediately if a command fails

# ------------------------
# Loading Config
# ------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Start searching from the script's directory
SEARCH_DIR="$SCRIPT_DIR"
CONFIG_FOUND=false

# Walk up the directory tree looking for config.sh
while [ "$SEARCH_DIR" != "/" ]; do
    if [ -f "$SEARCH_DIR/config.sh" ]; then
        source "$SEARCH_DIR/config.sh"
        CONFIG_FOUND=true
        break
    fi
    # Move up one level
    SEARCH_DIR="$(dirname "$SEARCH_DIR")"
done

if [ "$CONFIG_FOUND" = false ]; then
    echo "Error: config.sh not found in any parent directory of $SCRIPT_DIR"
    exit 1
fi


# ------------------------
# Running Script
# ------------------------

RESULTS_DIR="$PROJECT_DIR/results"
OUTPUT_DIR="$PROJECT_DIR/output/apertus_lora"


if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Warning: LoRA output directory $OUTPUT_DIR does not exist yet."
fi

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Starting Evaluation in $(hostname)"
echo "Project Directory: $PROJECT_DIR"
echo "=========================================="

# Evaluate BASE Model
echo "[1/2] Evaluating BASE Model..."
lm_eval --model hf \
    --model_args pretrained=swiss-ai/Apertus-8B-Instruct-2509,trust_remote_code=True,dtype=bfloat16 \
    --tasks medqa_4options \
    --device cuda:0 \
    --batch_size auto > "$RESULTS_DIR/log_base.txt" 2>&1

# Extract Accuracy
BASE_ACC=$(grep '|medqa_4options|' "$RESULTS_DIR/log_base.txt" | grep -oP '0\.\d+' | head -n 1)
echo ">> Base Model Accuracy: $BASE_ACC"

# Evaluate LoRA Model
echo "[2/2] Evaluating LoRA Model..."
lm_eval --model hf \
    --model_args pretrained=swiss-ai/Apertus-8B-Instruct-2509,peft=$OUTPUT_DIR,trust_remote_code=True,dtype=bfloat16 \
    --tasks medqa_4options \
    --device cuda:0 \
    --batch_size auto > "$RESULTS_DIR/log_lora.txt" 2>&1

# Extract Accuracy
LORA_ACC=$(grep '|medqa_4options|' "$RESULTS_DIR/log_lora.txt" | grep -oP '0\.\d+' | head -n 1)
echo ">> LoRA Model Accuracy: $LORA_ACC"

# Save Comparison
echo "Saving results..."
echo 'Model,Accuracy,Task' > "$RESULTS_DIR/comparison.csv"
echo "Base,$BASE_ACC,MedQA" >> "$RESULTS_DIR/comparison.csv"
echo "LoRA,$LORA_ACC,MedQA" >> "$RESULTS_DIR/comparison.csv"

echo "------------------------------------------"
cat "$RESULTS_DIR/comparison.csv"
echo "------------------------------------------"