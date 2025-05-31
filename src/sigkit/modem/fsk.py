"""Phase Shift Keying Module."""

import numpy as np

from sigkit.core.base import SigKitError, Signal
from sigkit.modem.base import Modem


class FSK(Modem):
    """FSK Modem for modulating and demodulating bits."""

    def __init__(
        self
    ):
        """N-FSK Modem.

        Args:
            todo
        """
        pass

    def modulate(self, bits: np.ndarray) -> Signal:
        """Modulate bits with FSK.

        Args:
            bits: 1D array of 0 | 1, length multiple of log2(n_components)

        Returns:
            Signal: containing complex64 samples
        """
        if bits.ndim != 1 or bits.size % self.bits_per_symbol != 0:
            raise SigKitError(
                f"Number of bits must be a multiple of {self.bits_per_symbol}"
            )

        samples = np.array([0+0j]*10, dtype=np.complex64)
        return Signal(
            samples=samples.astype(np.complex64),
            sample_rate=self.sample_rate,
            carrier_frequency=self.cf,
        )

    def demodulate(self, signal: Signal | np.ndarray) -> np.ndarray:
        """Map received FSK samples to bits.

        Args:
            signal: Signal containing modulated complex samples.

        Returns:
            1D array of bits.
        """
        x = signal.samples if isinstance(signal, Signal) else signal
        if not x.dtype == np.complex64:
            raise SigKitError("Demodulate expects samples to be of type np.complex64.")

        bits_matrix = np.zeros(10, dtype=np.uint8)
        return bits_matrix.ravel()
