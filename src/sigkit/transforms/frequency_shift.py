"""Module for FrequencyShift Torch Transform."""

from typing import Tuple, Union

import numpy as np
import torch
from torch import nn

from sigkit.core.base import SigKitError


class ApplyFrequencyShift(nn.Module):
    """Apply a constant or random frequency offset to a 1D complex64 torch.Tensor.

    Args:
        freq_offset:
            - If a single float or int: apply that fixed frequency (in Hz).
            - If a tuple/list of two floats: (min_freq, max_freq), pick
              random uniform from [min_freq, max_freq] per call.
        sample_rate:
            - Sampling rate of the signal (in samples per second).
    """

    def __init__(
        self,
        freq_offset: Union[float, Tuple[float, float]],
        sample_rate: float,
    ):
        super().__init__()
        if isinstance(freq_offset, (int, float)):
            self.min_f = float(freq_offset)
            self.max_f = float(freq_offset)
        elif (
            isinstance(freq_offset, (tuple, list))
            and len(freq_offset) == 2
            and all(isinstance(f, (int, float)) for f in freq_offset)
        ):
            self.min_f = float(freq_offset[0])
            self.max_f = float(freq_offset[1])
        else:
            raise SigKitError(
                f"ApplyFrequencyShift: freq_offset must be a single number or a tuple"
                f" of two numbers, got {freq_offset}"
            )

        if sample_rate <= 0:
            raise SigKitError(
                f"ApplyFrequencyShift: sample_rate must be positive, got {sample_rate}"
            )
        self.sample_rate = float(sample_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the frequency shift to the Tensor."""
        if x.dtype != torch.complex64 or x.ndim != 1:
            raise SigKitError(
                f"ApplyFrequencyShift expects a 1D tensor of dtype=torch.complex64,"
                f" got {x.shape=}, {x.dtype=}"
            )

        if self.min_f == self.max_f:
            f = self.min_f
        else:
            r = torch.rand(1).item()
            f = self.min_f + (self.max_f - self.min_f) * r

        n = torch.arange(x.shape[0], device=x.device, dtype=torch.float32)
        t = n / self.sample_rate

        ang = 2 * np.pi * f * t
        real = torch.cos(ang)
        imag = torch.sin(ang)
        phase = torch.complex(real, imag)

        return x * phase
