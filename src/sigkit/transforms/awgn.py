"""Module for AWGN Torch Transform."""

import torch
from torch import nn


class ApplyAWGN(nn.Module):
    """Additive White Gaussian Noise Torch Transform."""

    def __init__(self, snr_db: float):
        super().__init__()
        self.snr_db = snr_db

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies AWGN to the Tensor to reach the target SNR.

        Expects a (2, N) shaped tensor of I & Q channels
        Returns the signal with AWGN applied to the target snr_db.
        """
        sig_power = (x.pow(2).sum(dim=0)).mean()
        snr_lin = 10.0 ** (self.snr_db / 10.0)
        noise_power = sig_power / snr_lin

        noise = torch.sqrt(noise_power / 2.0) * torch.randn_like(x)

        return x + noise
