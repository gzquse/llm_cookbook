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

# --- Paths (adjust to your setup) ---
MEGATRON_DIR=/path/to/Megatron-LM
VOCAB_FILE=/path/to/gpt2-vocab.json
MERGE_FILE=/path/to/gpt2-merges.txt
DATA_PATH=/path/to/my-gpt2_text_document

GPUS_PER_NODE=$SLURM_GPUS_PER_NODE
NUM_NODES=$SLURM_JOB_NUM_NODES
WORLD_SIZE=$(( GPUS_PER_NODE * NUM_NODES ))

# --- Parallelism configuration ---
# With 8 GPUs total (2 nodes x 4 GPUs), one possible layout:
#   TP=2  (tensor-parallel within a node)
#   PP=2  (pipeline-parallel across groups)
#   DP=2  (data-parallel replicas = WORLD_SIZE / TP / PP)
TP=2
PP=2

# --- Model hyperparameters (GPT-2 small as a demo) ---
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTENTION_HEADS=12
SEQ_LENGTH=1024
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=32

srun shifter \
  torchrun \
  --nnodes=$NUM_NODES \
  --nproc-per-node=$GPUS_PER_NODE \
  --rdzv-backend=c10d \
  --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
  $MEGATRON_DIR/pretrain_gpt.py \
    --tensor-model-parallel-size $TP \
    --pipeline-model-parallel-size $PP \
    --num-layers $NUM_LAYERS \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $SEQ_LENGTH \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --train-iters 500 \
    --lr 6e-4 \
    --min-lr 6e-5 \
    --lr-decay-style cosine \
    --lr-warmup-iters 50 \
    --weight-decay 0.1 \
    --clip-grad 1.0 \
    --fp16 \
    --log-interval 10 \
    --save-interval 200 \
    --eval-interval 100 \
    --eval-iters 10 \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --data-path $DATA_PATH \
    --split 98,2,0 \
    --save /path/to/checkpoints \
    --load /path/to/checkpoints
