#!/bin/bash
#SBATCH --job-name=LSAIE_lora_eval_medqa
#SBATCH --partition=normal
#SBATCH --account=large-sc-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:20:00
#SBATCH --output=logs/eval_apertus_medqa_%j.out
#SBATCH --error=logs/eval_apertus_medqa_%j.err


# Assume you submit the job from the root of your repo
if [ -f "$SLURM_SUBMIT_DIR/config.sh" ]; then
    source "$SLURM_SUBMIT_DIR/config.sh"
else
    echo "Error: config.sh not found in $SLURM_SUBMIT_DIR"
    exit 1
fi


CMD="
    cd $PROJECT_DIR/scripts/utils
    ./run_eval.sh
"

srun \
    --environment=$ENVIRONMENT \
    bash -c "$CMD"