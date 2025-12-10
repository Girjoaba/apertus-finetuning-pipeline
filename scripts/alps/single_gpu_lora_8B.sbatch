#!/bin/bash
#SBATCH --job-name=LSAIE_lora
#SBATCH --partition=normal
#SBATCH --account=large-sc-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=logs/finetune_apertus_lora8B_%j.out
#SBATCH --error=logs/finetune_apertus_lora8B_%j.err

set -eo pipefail

# Assume you submit the job from the root of your repo
if [ -f "$SLURM_SUBMIT_DIR/config.sh" ]; then
    source "$SLURM_SUBMIT_DIR/config.sh"
else
    echo "Error: config.sh not found in $SLURM_SUBMIT_DIR"
    exit 1
fi

CMD="
    # print current environment variables
    echo \"[srun] rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID\"

    cd $PROJECT_DIR

    # run the script
    python sft_train.py --config configs/sft_lora_8B.yaml
"

srun \
    --environment=$ENVIRONMENT \
    bash -c "$CMD"
