"""
Horovod distributed training example for NERSC Perlmutter.

Launch with srun (see submit_horovod.sh).  Each srun task is one Horovod
rank pinned to one GPU.  Horovod uses NCCL under the hood for all-reduce.
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

import horovod.torch as hvd


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

    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    dataset = make_dummy_dataset()
    sampler = DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, num_workers=4
    )

    model = SimpleModel().cuda()

    # Scale learning rate by number of workers (linear scaling rule)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr * hvd.size())

    # Wrap optimizer with Horovod's DistributedOptimizer for gradient
    # all-reduce across ranks
    optimizer = hvd.DistributedOptimizer(
        optimizer, named_parameters=model.named_parameters()
    )

    # Broadcast initial parameters from rank 0 so all ranks start in sync
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

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

        if hvd.rank() == 0:
            avg_loss = total_loss / (step + 1)
            print(f"Epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.4f}")


if __name__ == "__main__":
    main()
