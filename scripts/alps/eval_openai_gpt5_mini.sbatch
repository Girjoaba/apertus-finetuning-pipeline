#!/bin/bash
#SBATCH --job-name=eval_gpt5_mini
#SBATCH --partition=normal
#SBATCH --account=large-sc-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_gpt5_mini_%j.out
#SBATCH --error=logs/eval_gpt5_mini_%j.err

set -eo pipefail

# Source project configuration
if [ -f "$SLURM_SUBMIT_DIR/config.sh" ]; then
    source "$SLURM_SUBMIT_DIR/config.sh"
else
    echo "Error: config.sh not found in $SLURM_SUBMIT_DIR"
    exit 1
fi

# OpenAI API key
if [ -z "$OPENAI_API_KEY" ]; then
    if [ -f "$HOME/.openai_api_key" ]; then
        export OPENAI_API_KEY=$(cat "$HOME/.openai_api_key")
        echo "Loaded OPENAI_API_KEY from ~/.openai_api_key"
    else
        echo "Error: OPENAI_API_KEY not set"
        echo "Set it with: export OPENAI_API_KEY='sk-...' before running sbatch"
        echo "Or save to ~/.openai_api_key"
        exit 1
    fi
fi

echo "=============================================="
echo "MedQA Test Evaluation - OpenAI GPT-5-mini"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Project dir: $PROJECT_DIR"
echo "=============================================="

# Run evaluation script
# Note: unset SSL_CERT_FILE to fix SSL issues in container
CMD="
    cd $PROJECT_DIR
    unset SSL_CERT_FILE
    pip install --quiet openai tiktoken
    python scripts/evaluate_openai.py \
        --model gpt-5-mini \
        --requests_per_minute 100 \
        --output results/openai_gpt5_mini_test.json \
        --log_file logs/eval_openai_gpt5_mini_detailed.log
"

srun --environment=$ENVIRONMENT \
    bash -c "$CMD"

echo "=============================================="
echo "Evaluation complete!"
echo "Results saved to: results/openai_gpt5_mini_test.json"
echo "Detailed log: logs/eval_openai_gpt5_mini_detailed.log"
echo "=============================================="
