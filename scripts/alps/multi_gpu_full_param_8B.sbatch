#!/bin/bash
#SBATCH --job-name=LSAIE_lora
#SBATCH --partition=normal
#SBATCH --account=large-sc-2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/finetune_apertus_full_param8B_%j.out
#SBATCH --error=logs/finetune_apertus_full_param8B_%j.err
#SBATCH --exclusive

set -eo pipefail

# Assume you submit the job from the root of your repo
if [ -f "$SLURM_SUBMIT_DIR/config.sh" ]; then
    source "$SLURM_SUBMIT_DIR/config.sh"
else
    echo "Error: config.sh not found in $SLURM_SUBMIT_DIR"
    exit 1
fi

# The defined environment vars will be shared with the other compute nodes.
export MASTER_ADDR=$(scontrol show hostname "$SLURM_NODELIST" | head -n1)  
export MASTER_PORT=12345   # Choose an unused port
export FOOBAR=666
export WORLD_SIZE=$(( SLURM_NNODES * SLURM_NTASKS_PER_NODE ))


# NCCL optimizations
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=hsn     # High Speed Network interface
export FI_PROVIDER="cxi"          # Cray Slingshot specific provider
export NCCL_NET_GDR_LEVEL=3       # GPUDirect RDMA
export OMP_NUM_THREADS=16

# The sbatch script is executed by only one node.
echo "Head node: $MASTER_ADDR"
echo "Total GPUs: $((SLURM_JOB_NUM_NODES * 4))"

CMD="
    echo \"[srun] rank=\$SLURM_PROCID host=\$(hostname) noderank=\$SLURM_NODEID localrank=\$SLURM_LOCALID\"

    cd $PROJECT_DIR

    # run the script
    torchrun \
    --nnodes="${SLURM_NNODES}" \
    --node_rank=\$SLURM_NODEID \
    --nproc_per_node=4 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    sft_train.py --config configs/sft_full_8B.yaml
"

srun \
    --environment=$ENVIRONMENT \
    bash -c "$CMD"
