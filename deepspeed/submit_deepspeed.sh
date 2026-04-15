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
#SBATCH --output=logs/rubriq_%j.out
#SBATCH --error=logs/rubriq_%j.err

# --- Rendezvous setup for torchrun ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export OMP_NUM_THREADS=8

# NERSC recommends torchrun over the deepspeed launcher.
# torchrun handles node discovery via SLURM, so no hostfile is needed.
# The DeepSpeed config is passed inside the training script via
# deepspeed.initialize(config="ds_config.json") rather than on the
# command line.
srun shifter \
  torchrun \
  --nnodes=$SLURM_JOB_NUM_NODES \
  --nproc-per-node=$SLURM_GPUS_PER_NODE \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  train_deepspeed.py --deepspeed_config ds_config.json
