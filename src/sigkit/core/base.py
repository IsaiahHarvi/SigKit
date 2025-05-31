"""Abstract base classes and common exceptions for SigKit."""

from dataclasses import dataclass, field

import numpy as np
import torch


class SigKitError(Exception):
    """Base exception for Sigkit-specific errors."""


@dataclass
class Signal:
    """A class for a complex waveform.

    Parameters:
        samples: ndarray of shape (N) containing complex64 values, defaults to 4096.
        sample_rate: in Hz
        carrier_frequency: in Hz.
    """

    samples: np.ndarray = field(
        default_factory=lambda: np.zeros(4096, dtype=np.complex64)
    )
    sample_rate: float = 1.0
    carrier_frequency: float = 0.0

    def __post_init__(self):
        if self.samples.dtype != np.complex64:
            raise SigKitError("Signal samples must be np.ndarray[complex64]")

    def to_tensor(self) -> torch.Tensor:
        """Convert the samples parameter to a PyTorch Tensor.

        Convert into a float32 tensor of shape (2, N),
        where row 0 = real (I) and row 1 = imag (Q).
        Note that for our training pipeline, N should be 4096.
        """
        return torch.from_numpy(
            np.stack([np.real(self.samples), np.imag(self.samples)], axis=0).astype(
                np.float32
            )
        )

    def to_baseband(self) -> "Signal":
        """Convert the Signal to baseband by removing the carrier frequency.

        If the carrier frequency is 0, the method returns.
        """
        if self.carrier_frequency == 0.0:
            return self

        t = np.arange(self.samples.size) / self.sample_rate
        baseband_samples = (
            self.samples * np.exp(-1j * 2 * np.pi * self.carrier_frequency * t)
        ).astype(np.complex64)

        return Signal(
            samples=baseband_samples,
            sample_rate=self.sample_rate,
            carrier_frequency=0.0,
        )
