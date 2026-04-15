"""
DeepSpeed training example for NERSC Perlmutter.

Launch with torchrun (see submit_deepspeed.sh).  The DeepSpeed config
(ds_config.json) is loaded inside deepspeed.initialize() so that torchrun
can be used as the launcher instead of the deepspeed CLI.
"""

import argparse
import torch
import torch.nn as nn
import deepspeed
from torch.utils.data import DataLoader, TensorDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--epochs", type=int, default=3)
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

    deepspeed.init_distributed()

    model = SimpleModel()
    dataset = make_dummy_dataset()

    model_engine, optimizer, trainloader, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        config=args.deepspeed_config,
    )

    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model_engine.train()
        total_loss = 0.0
        for step, (data, target) in enumerate(trainloader):
            data = data.to(model_engine.local_rank)
            target = target.to(model_engine.local_rank)

            output = model_engine(data)
            loss = criterion(output, target)

            model_engine.backward(loss)
            model_engine.step()

            total_loss += loss.item()

        if model_engine.local_rank == 0:
            avg_loss = total_loss / (step + 1)
            print(f"Epoch {epoch + 1}/{args.epochs}  loss={avg_loss:.4f}")


if __name__ == "__main__":
    main()
