"""
Minimal Megatron-LM GPT pre-training example for NERSC Perlmutter.

For production work you would normally use Megatron's own pretrain_gpt.py
(shown in submit_megatron.sh).  This file illustrates the key Megatron
concepts in a self-contained script for learning purposes.

Prerequisites:
  - Megatron-LM installed (pip install megatron-core, or clone the repo)
  - A tokenized dataset produced by Megatron's tools/preprocess_data.py
  - GPT-2 vocabulary and merge files

See https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/index.html
for a full quick-start guide.
"""

import os
import torch


def print_rank0(msg):
    if int(os.environ.get("RANK", 0)) == 0:
        print(msg)


def main():
    # Megatron handles distributed init, model creation, and training
    # internally.  The canonical entry point is:
    #
    #   from megatron.training import pretrain
    #   pretrain(
    #       train_valid_test_datasets_provider,
    #       model_provider,
    #       ModelType.encoder_or_decoder,
    #       forward_step,
    #       args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
    #   )
    #
    # Because Megatron's pretrain() expects its own argparse namespace and
    # data pipeline, production users should invoke pretrain_gpt.py directly
    # (as shown in submit_megatron.sh).  Below is a sketch of the pieces.

    print_rank0("=== Megatron-LM workflow overview ===")
    print_rank0("")
    print_rank0("1. Preprocess data:")
    print_rank0("     python tools/preprocess_data.py \\")
    print_rank0("       --input raw_text.jsonl \\")
    print_rank0("       --output-prefix my-gpt2 \\")
    print_rank0("       --vocab-file gpt2-vocab.json \\")
    print_rank0("       --merge-file gpt2-merges.txt \\")
    print_rank0("       --tokenizer-type GPT2BPETokenizer \\")
    print_rank0("       --workers 32")
    print_rank0("")
    print_rank0("2. Launch training (see submit_megatron.sh):")
    print_rank0("     torchrun ... pretrain_gpt.py \\")
    print_rank0("       --tensor-model-parallel-size 2 \\")
    print_rank0("       --pipeline-model-parallel-size 2 \\")
    print_rank0("       ...")
    print_rank0("")
    print_rank0("3. Key parallelism dimensions:")
    print_rank0("     TP  -- splits attention heads / MLP across GPUs")
    print_rank0("     PP  -- splits layers into pipeline stages")
    print_rank0("     DP  -- replicates the model for data parallelism")
    print_rank0("     DP = world_size / (TP * PP)")

    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    print_rank0(f"\nCurrent world size: {world_size}")
    print_rank0(f"Rank 0 device: cuda:{torch.cuda.current_device()}")


if __name__ == "__main__":
    main()
