"""
PyTorch Lightning training example for NERSC Perlmutter.

Launch with torchrun (see submit_lightning.sh).  The Trainer is configured
for DDP by default; switch to FSDP or DeepSpeed by changing the strategy.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split

import lightning as L


class LitClassifier(L.LightningModule):
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=10, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_acc", acc, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


class DummyDataModule(L.LightningDataModule):
    def __init__(self, num_samples=10000, input_dim=784, num_classes=10,
                 batch_size=64):
        super().__init__()
        self.num_samples = num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.batch_size = batch_size

    def setup(self, stage=None):
        x = torch.randn(self.num_samples, self.input_dim)
        y = torch.randint(0, self.num_classes, (self.num_samples,))
        dataset = TensorDataset(x, y)
        self.train_ds, self.val_ds = random_split(dataset, [0.8, 0.2])

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size,
                          shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                          num_workers=4)


def main():
    dm = DummyDataModule()
    model = LitClassifier()

    trainer = L.Trainer(
        max_epochs=5,
        accelerator="gpu",
        devices="auto",
        num_nodes=int(os.environ.get("SLURM_JOB_NUM_NODES", 1)),
        strategy="ddp",            # swap to "fsdp" or "deepspeed_stage_2" etc.
        precision="16-mixed",
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
