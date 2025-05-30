"""Module for creating Torch Datasets to train on."""

import os

import lightning as pl
from torch.utils.data import DataLoader

from sigkit.datasets.procedural import ProceduralDataset


class SigKitDataModule(pl.LightningDataModule):
    """LightningDataModule for SigKit datasets."""

    def __init__(
        self,
        train_dataset: ProceduralDataset,
        val_dataset: ProceduralDataset,
        batch_size: int = 64,
        num_workers: int = os.cpu_count() // 2,
    ):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = len(self.train_dataset.modems)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
