#!/bin/bash
#SBATCH --job-name=LSAIE_lora_70B_full_param
#SBATCH --partition=normal
#SBATCH --account=large-sc-2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/finetune_apertus_full_param70B_%j.out
#SBATCH --error=logs/finetune_apertus_full_param70B_%j.err
#SBATCH --exclusive

set -eo pipefail

if [ -f "$SLURM_SUBMIT_DIR/config.sh" ]; then
    source "$SLURM_SUBMIT_DIR/config.sh"
else
    echo "Error: config.sh not found in $SLURM_SUBMIT_DIR"
    exit 1
fi

export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)  
export MASTER_PORT=12345   # Choose an unused port
export FOOBAR=666
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))

export WANDB_CACHE_DIR=/iopsstor/scratch/cscs/$USER/wandb_cache
export WANDB_DIR=/iopsstor/scratch/cscs/$USER/wandb
mkdir -p $WANDB_CACHE_DIR
mkdir -p $WANDB_DIR

export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn
export FI_PROVIDER="cxi"
export NCCL_NET_GDR_LEVEL=3
export OMP_NUM_THREADS=16

echo "Head node: $MASTER_ADDR"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * 4))"

CMD="
    echo \"[srun] rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID\"

    cd $PROJECT_DIR
    
    torchrun \
    --nnodes="${SLURM_NNODES}" \
    --node_rank=\$SLURM_NODEID \
    --nproc_per_node=4 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    sft_train.py --config configs/sft_lora_70B.yaml
"

srun \
    --environment=$ENVIRONMENT \
    bash -c "$CMD"
