#!/bin/bash
#SBATCH -C gpu
#SBATCH --account=m4992_g
#SBATCH --qos=debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --time=30:00
#SBATCH --image=nersc/pytorch:25.02.01
#SBATCH --module=gpu,nccl-plugin
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

# --- Rendezvous setup for torchrun ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=8
export PYTHONNOUSERSITE=1

# NERSC recommends torchrun for launching Lightning training.
# When launched with torchrun the Trainer should use
# strategy="ddp" (or "fsdp", "deepspeed", etc.) and set
# devices="auto" / num_nodes to let Lightning pick up the
# environment set by torchrun.
WORK_DIR=$SLURM_SUBMIT_DIR/lightning

srun shifter \
  torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc-per-node=$SLURM_GPUS_PER_NODE \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  "$WORK_DIR/train_lightning.py"
