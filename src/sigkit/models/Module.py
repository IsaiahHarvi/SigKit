"""Module for constructing the Neural Network."""

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0


class SigKitClassifier(pl.LightningModule):
    """LightningModule with parameterized backbones for signal classification."""

    def __init__(self, num_classes: int, lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        backbone = efficientnet_b0(weights=None)
        backbone.classifier = nn.Identity()

        # replace the first Conv2d to accept 2-channel I/Q inputs
        old_stem = backbone.features[0][0]
        new_stem = nn.Conv2d(
            in_channels=2,
            out_channels=old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            bias=(old_stem.bias is not None),
        )
        nn.init.kaiming_normal_(new_stem.weight, nonlinearity="relu")
        if new_stem.bias is not None:
            nn.init.zeros_(new_stem.bias)
        backbone.features[0][0] = new_stem

        self.backbone = backbone
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(1280, num_classes))

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(2)  # (B, 2, 1, 4096) required bc 2D -- i hate this though
        features = self.backbone(x)
        return self.head(features)

    def training_step(self, batch, batch_idx):
        signals, labels = batch
        logits = self(signals)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        signals, labels = batch
        logits = self(signals)
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
