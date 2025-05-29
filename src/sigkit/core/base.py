import torch
import numpy as np

from dataclasses import dataclass, field


class SigKitError(Exception):
    """Base exception for Sigkit-specific errors."""


@dataclass
class Signal:
    """
    A container for a complex waveform.
        - samples: ndarray of shape (N) containing complex64 values, defaults to 4096.
        - sample_rate: in Hz
        - center_freq: in Hz
    """

    samples: np.ndarray = field(
        default_factory=lambda: np.zeros(4096, dtype=np.complex64)
    )
    sample_rate: float = 1.0
    center_freq: float = 0.0

    def to_tensor(self) -> torch.Tensor:
        """
        Convert into a float32 tensor of shape (2, 4096),
        where row 0 = real (I) and row 1 = imag (Q).
        """
        return torch.from_numpy(
            np.stack([np.real(self.samples), np.imag(self.samples)], axis=0).astype(
                np.float32
            )
        )

