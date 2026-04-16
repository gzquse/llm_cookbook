"""
Distributed training example for NERSC Perlmutter using PyTorch DDP.

Launch with torchrun (see submit_horovod.sh).  Each torchrun worker is one
DDP rank pinned to one GPU.  DDP uses NCCL for gradient all-reduce.
"""

import argparse
import os

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


class SimpleModel(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_dummy_dataset(num_samples=10000, input_dim=784, num_classes=10):
    x = torch.randn(num_samples, input_dim)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(x, y)


def main():
    args = get_args()

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)

    dataset = make_dummy_dataset()
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4
    )

    model = SimpleModel().cuda()
    model = DDP(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * world_size)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        sampler.set_epoch(epoch)
        total_loss = 0.0
        for step, (data, target) in enumerate(loader):
            data = data.cuda()
            target = target.cuda()

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if rank == 0:
            avg_loss = total_loss / (step + 1)
            print(f"Epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.4f}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
