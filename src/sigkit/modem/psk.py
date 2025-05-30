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
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.n_components = n_components
        self.cf = cf

        self.bits_per_symbol = int(np.log2(n_components))
        self.constellation = np.exp(
            1j
            * (2 * np.pi)
            * (np.array([i ^ (i >> 1) for i in range(n_components)]) / n_components)
        ).astype(np.complex64)

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

        symbols = bits.reshape(-1, self.bits_per_symbol)
        baseband = self.constellation[
            symbols.dot(1 << np.arange(self.bits_per_symbol)[::-1])
        ]
        samples = np.repeat(baseband, self.sps)
        if self.cf:
            t = np.arange(samples.size) / self.sample_rate
            samples *= np.exp(1j * (2 * np.pi) * self.cf * t)

        return Signal(
            samples=samples, sample_rate=self.sample_rate, carrier_frequency=self.cf
        )

    def demodulate(self, signal: Signal | np.ndarray) -> np.ndarray:
        """Map received PSK samples to bits.

        Args:
            signal: Signal containing modulated complex samples.

        Returns:
            1D array of bits.
        """
        if isinstance(signal, Signal):
            x = signal.samples
        else:
            x = signal

        if self.cf:
            t = np.arange(x.size) / self.sample_rate
            x *= np.exp(-1j * 2 * np.pi * self.cf * t)

        symbols = x[np.arange(self.sps // 2, x.size, self.sps)]
        dists = np.abs(symbols[:, None] - self.constellation[None, :])
        bits = (
            (
                dists.argmin(axis=1)[:, None]
                & (1 << np.arange(self.bits_per_symbol)[::-1])
            )
            > 0
        ).astype(np.uint8)
        return bits.ravel()
