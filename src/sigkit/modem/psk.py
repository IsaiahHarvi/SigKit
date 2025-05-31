"""Phase Shift Keying Module."""

import numpy as np

from sigkit.core.base import SigKitError, Signal
from sigkit.modem.base import Modem


class PSK(Modem):
    """PSK Modem for modulating and demodulating bits."""

    def __init__(
        self, sample_rate: int, symbol_rate: int, n_components, cf: float = 0.0
    ):
        """N-PSK Modem.

        Args:
            sample_rate: Sampling rate of the waveform
            symbol_rate: Symbol rate, used to calculate samples per symbol
            n_components: Number of PSK points (e.g. 2, 4, 8, 16..)
            cf: Carrier frequency
        """
        super().__init__(sample_rate, symbol_rate, n_components, cf)
        bin_indicies = np.arange(n_components)
        indicies = bin_indicies ^ (bin_indicies >> 1)
        phases = 2 * np.pi * indicies / n_components
        self.constellation = np.exp(1j * phases)

    def modulate(self, bits: np.ndarray) -> Signal:
        """Modulate bits with PSK.

        Args:
            bits: 1D array of 0 | 1, length multiple of log2(n_components)

        Returns:
            Signal: containing complex64 samples
        """
        if bits.ndim != 1 or bits.size % self.bits_per_symbol != 0:
            raise SigKitError(
                f"Number of bits must be a multiple of {self.bits_per_symbol}"
            )
        bits = bits.reshape(-1, self.bits_per_symbol)
        weights = 1 << np.arange(self.bits_per_symbol - 1, -1, -1)
        bin_indicies = bits.dot(weights)
        indicies = bin_indicies ^ (bin_indicies >> 1)

        baseband = self.constellation[indicies]
        samples = np.repeat(baseband, self.sps)

        if self.cf != 0.0:
            t = np.arange(samples.size) / self.sample_rate
            samples *= np.exp(1j * 2 * np.pi * self.cf * t)

        return Signal(
            samples=samples.astype(np.complex64),
            sample_rate=self.sample_rate,
            carrier_frequency=self.cf,
        )

    def demodulate(self, signal: Signal | np.ndarray) -> np.ndarray:
        """Map received PSK samples to bits.

        Args:
            signal: Signal containing modulated complex samples.

        Returns:
            1D array of bits.
        """
        x = signal.samples if isinstance(signal, Signal) else signal
        if not x.dtype == np.complex64:
            raise SigKitError("Demodulate expects samples to be of type np.complex64.")

        indices = self.extract_symbols(x)
        bin_indices = indices.copy()
        shift = indices >> 1
        while np.any(shift):
            bin_indices ^= shift
            shift >>= 1

        bits_matrix = (
            (bin_indices[:, None] & (1 << np.arange(self.bits_per_symbol - 1, -1, -1)))
            > 0
        ).astype(np.uint8)
        return bits_matrix.ravel()
