from abc import ABC, abstractmethod

import numpy as np

from sigkit.core.base import Signal


class Modem(ABC):
    """
    Abstract base class for all modulators/demodulators.
    """

    @abstractmethod
    def modulate(self, bits: np.ndarray) -> Signal:
        """
        bits: shape (..., n_bits), dtype {0,1}
        returns a Signal with samples.shape == (..., 2, n_samples)
        """
        raise NotImplementedError

    @abstractmethod
    def demodulate(self, signal: Signal | np.ndarray) -> np.ndarray:
        """
        signal.samples: shape (..., 2, n_samples)
        returns bit‐probabilities or hard bits, shape (..., n_bits)
        """
        raise NotImplementedError
