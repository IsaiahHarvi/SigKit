"""Training Module for the SigKitClassifier."""

from typing import Dict, List

import click
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from sigkit.datasets.procedural import ProceduralDataset
from sigkit.models.DataModule import SigKitDataModule
from sigkit.models.Module import SigKitClassifier
from sigkit.modem.base import Modem
from sigkit.modem.psk import PSK


@click.command()
@click.option(
    "--batch-size",
    default=32,
    type=int,
    show_default=True,
    help="Batch size for training",
)
@click.option("--lr", default=1e-3, type=float, show_default=True, help="Learning rate")
@click.option(
    "--max-epochs",
    default=10000,
    type=int,
    show_default=True,
    help="Maximum number of epochs if not using early stop",
)
def train(batch_size: int, lr: float, max_epochs: int):
    """Train the SigKitClassifier on SigKit datasets."""
    mapping_list: List[Dict[Modem, List[int]]] = [{PSK: [2, 4, 8, 16, 32, 64]}]
    train_ds = ProceduralDataset(mapping_list)
    val_ds = ProceduralDataset(mapping_list, val=True, seed=42)

    dm = SigKitDataModule(
        train_dataset=train_ds, val_dataset=val_ds, batch_size=batch_size
    )

    model = SigKitClassifier(num_classes=dm.num_classes, lr=lr)

    logger = WandbLogger(project="SigKit")
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[
            ModelCheckpoint(
                monitor="val_acc", save_top_k=1, mode="min", filename="best.ckpt"
            ),
            EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose=True),
        ],
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    train()
