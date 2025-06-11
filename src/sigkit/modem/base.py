"""ABC Module for the Modem Package."""

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from sigkit.core.base import SigKitError, Signal


class Modem(ABC):
    """Abstract base class for all modulators/demodulators."""

    def __init__(
        self,
        sample_rate: int,
        symbol_rate: int,
        n_components: int,
        cf: Optional[float] = 0.0,
    ):
        """Initialize the common parameters for any Modem.

        Sets the arguments as members of the class and checks their validity.

        Args:
            sample_rate: Sampling rate in Hz. Must be a positive int and
                satisfy the Nyquist criterion for the symbol rate.
            symbol_rate: Symbol (baud) rate. Must be a positive int and
                exactly divide sample_rate.
            n_components: Number of constellation points, must be a power of two.
            cf: Carrier frequency in Hz. If given, must be in [0, sample_rate/2].

        Raises:
            ValueError: If any argument is out of bounds or of the wrong type.
        """
        if sample_rate < (2 * symbol_rate):
            raise SigKitError(
                f"sample_rate ({sample_rate}) must be ≥ 2 × symbol_rate ({symbol_rate})"
            )
        if sample_rate % symbol_rate != 0:
            raise SigKitError(
                f"{sample_rate=} must be an integer multiple of {symbol_rate=}"
            )

        if cf != 0.0:
            if not isinstance(cf, (int, float)) or cf < 0:
                raise SigKitError(f"cf must be a non‐negative number, got {cf}")
            if cf > (sample_rate / 2):
                raise SigKitError(
                    f"cf ({cf}) must not exceed Nyquist frequency ({sample_rate / 2})"
                )

        if not isinstance(n_components, int) or n_components < 2:
            raise ValueError(f"n_components must be ≥ 2, got {n_components}")
        # check power of two
        if (n_components & (n_components - 1)) != 0:
            raise ValueError(f"n_components must be a power of two, got {n_components}")

        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.cf = cf
        self.sps: int = sample_rate // symbol_rate  # samples per symbol
        self.bits_per_symbol = int(np.log2(n_components))
        self.n_components = n_components

    def __label__(self) -> str:
        return f"{self.n_components}-{self.__class__.__name__}"

    @abstractmethod
    def modulate(self, bits: np.ndarray) -> Signal:
        """ABC Method for modulating bits.

        bits: shape (..., n_bits), dtype {0,1}
        returns a Signal with samples.shape == (..., 2, n_samples).
        """
        raise NotImplementedError

    @abstractmethod
    def demodulate(self, signal: Signal | np.ndarray) -> np.ndarray:
        """ABC Method for demodulating a Signal.

        signal.samples: shape (..., 2, n_samples)
        returns bit‐probabilities or hard bits, shape (..., n_bits).
        """
        raise NotImplementedError

    def extract_symbols(self, signal: Signal | np.ndarray) -> np.ndarray:
        """Generic symbol extractor.

        - Removes carrier (if any),
        - Downsamples one sample per symbol at midpoint,
        - Finds nearest constellation point.
        """
        x = signal.samples if isinstance(signal, Signal) else signal
        if self.cf:
            t = np.arange(x.size) / self.sample_rate
            x = x * np.exp(-1j * 2 * np.pi * self.cf * t)
        idx = np.arange(self.sps // 2, x.size, self.sps)
        sym_samples = x[idx]
        dists = np.abs(sym_samples[:, None] - self.constellation[None, :])
        return dists.argmin(axis=1)
