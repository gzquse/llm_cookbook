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

# NERSC recommends srun for Horovod (one MPI rank per GPU).
# Horovod uses MPI (via NCCL) for all-reduce communication, so each
# SLURM task maps to one Horovod rank / one GPU.
export OMP_NUM_THREADS=8

srun shifter \
  python train_horovod.py --epochs 5
