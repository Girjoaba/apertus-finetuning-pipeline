#!/bin/bash
#SBATCH --job-name=LSAIE_test_models
#SBATCH --partition=normal
#SBATCH --account=large-sc-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/test_models_%j.out
#SBATCH --error=logs/test_models_%j.err

# Assume you submit the job from the root of your repo
if [ -f "$SLURM_SUBMIT_DIR/config.sh" ]; then
    source "$SLURM_SUBMIT_DIR/config.sh"
else
    echo "Error: config.sh not found in $SLURM_SUBMIT_DIR"
    exit 1
fi

# Arguments can be passed via sbatch: sbatch test_models.sbatch --models 8b_lora 8b_full
EXTRA_ARGS="${@}"

CMD="
    cd $PROJECT_DIR
    python scripts/utils/test_models.py \
        --wandb-project apertus-finetune \
        --device cuda:0 \
        $EXTRA_ARGS
"

srun \
    --environment=$ENVIRONMENT \
    bash -c "$CMD"
