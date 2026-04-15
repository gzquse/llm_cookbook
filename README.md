# Distributed Training Examples for NERSC Perlmutter

Example SLURM job scripts and training code for four distributed training
frameworks on Perlmutter, following
[NERSC's launcher documentation](https://docs.nersc.gov/machinelearning/launchers/)
and [training library guidance](https://docs.nersc.gov/machinelearning/training/).

All examples use the **Shifter** container `nersc/pytorch:25.02.01` with the
NCCL plugin for optimized GPU communication.

## Quick reference

| Framework | Launcher | Tasks per node | When to use |
|-----------|----------|----------------|-------------|
| **DeepSpeed** | `torchrun` | 1 | Very large models needing ZeRO memory optimization, CPU offloading, or pipeline parallelism |
| **Lightning** | `torchrun` | 1 | Rapid prototyping with automatic DDP/FSDP/DeepSpeed strategy switching |
| **Megatron** | `torchrun` | 1 | Billion-parameter transformer LLMs requiring tensor + pipeline parallelism |
| **Horovod** | `srun` | 4 (one per GPU) | TensorFlow compatibility or legacy MPI-based workflows |

## Directory layout

```
deepspeed/
  submit_deepspeed.sh   # SLURM batch script
  ds_config.json        # ZeRO Stage-2 config
  train_deepspeed.py    # Training script

lightning/
  submit_lightning.sh   # SLURM batch script
  train_lightning.py    # LightningModule + Trainer

megatron/
  submit_megatron.sh    # SLURM batch script (calls Megatron's pretrain_gpt.py)
  train_megatron.py     # Conceptual overview / minimal demo

horovod/
  submit_horovod.sh     # SLURM batch script
  train_horovod.py      # Training script with hvd.DistributedOptimizer
```

## Usage

1. **Edit paths and resource requests** -- Replace placeholder paths
   (`/path/to/...`, `<num_nodes>`) with your actual data/model paths and
   desired node counts.

2. **Submit**:
   ```bash
   cd deepspeed && sbatch submit_deepspeed.sh
   cd lightning && sbatch submit_lightning.sh
   cd megatron  && sbatch submit_megatron.sh
   cd horovod   && sbatch submit_horovod.sh
   ```

3. **Monitor**: `squeue -u $USER` or check the SLURM output file.

## Key NERSC recommendations

- **torchrun over framework-native launchers** -- NERSC recommends `torchrun`
  for DeepSpeed, Lightning, and Megatron instead of each framework's own CLI
  launcher.  `torchrun` integrates cleanly with SLURM and removes the need
  for hostfiles.

- **srun for Horovod** -- Horovod uses MPI semantics (one rank per GPU), so
  `srun` is the natural launcher with `--ntasks-per-node=4`.

- **Shifter + NCCL plugin** -- All examples request the `gpu,nccl-plugin`
  module for high-performance inter-node communication.

- **HF_HOME on $SCRATCH** -- When using Hugging Face models/tokenizers, set
  `export HF_HOME=$SCRATCH/cache/huggingface` to avoid filesystem issues on
  the home directory.

## References

- [NERSC Training Launchers](https://docs.nersc.gov/machinelearning/launchers/)
- [NERSC Training Libraries](https://docs.nersc.gov/machinelearning/training/)
- [NERSC Distributed Training](https://docs.nersc.gov/machinelearning/distributed-training/)
- [DeepSpeed Documentation](https://www.deepspeed.ai/)
- [Lightning Documentation](https://lightning.ai/docs/pytorch/stable/)
- [Megatron-LM Quick Start](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html)
- [Horovod Documentation](https://horovod.readthedocs.io/)
