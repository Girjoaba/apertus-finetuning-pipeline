#!/bin/bash
#SBATCH --job-name=LSAIE_lora
#SBATCH --partition=normal
#SBATCH --account=large-sc-2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --time=00:10:00
#SBATCH --output=output/output_4_%j.out
#SBATCH --error=output/output_4_%j.err
#SBATCH --environment=/capstor/store/cscs/ethz/large-sc-2/environment/ngc_pt_jan.toml
# Stop the script if a command fails or if an undefined variable is used
set -eo pipefail
# The sbatch script is executed by only one node.
echo "[sbatch-master] running on $(hostname)"
echo "[sbatch-master] SLURM_NODELIST: $SLURM_NODELIST"
echo "[sbatch-master] SLURM_NNODES: $SLURM_NNODES"
echo "[sbatch-master] SLURM_NODEID: $SLURM_NODEID"
echo "[sbatch-master] define some env vars that will be passed to the compute nodes"

# The defined environment vars will be shared with the other compute nodes.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)  
export MASTER_PORT=12345   # Choose an unused port
export FOOBAR=666
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))
echo "[sbatch-master] execute command on compute nodes"

# ---- Triton cache off NFS ----
export TRITON_CACHE_DIR=/iopsstor/scratch/cscs/$USER/triton_cache
mkdir -p "$TRITON_CACHE_DIR"
echo "TRITON_CACHE_DIR = $TRITON_CACHE_DIR"

PROJECT_DIR="/users/$USER/scratch/apertus-finetuning-pipeline"

# The command that will run on each process
CMD="
    # print current environment variables
    echo \"[srun] rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID\"

    cd $PROJECT_DIR

    # run the script
    python sft_train.py \
    --config configs/sft_lora.yaml \
    --model_name_or_path swiss-ai/Apertus-8B-Instruct-2509 \
"
# Submits the CMD to all the processes on all the nodes.
srun bash -c "$CMD"
echo "[sbatch-master] task finished"