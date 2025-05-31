"""Frequency Shift Module utilized for impairments."""

import numpy as np

from sigkit.core.base import SigKitError, Signal
from sigkit.impairments.base import Impairment


class FrequencyShift(Impairment):
    """Shift a baseband Signal in frequency by a constant offset.

    Args:
        freq_offset: Frequency offset in Hz. Positive shifts up, negative shifts down.

    Example:
        >>> imp = FrequencyShift(freq_offset=1e3)
        >>> shifted = imp.apply(signal)
    """

    def __init__(self, freq_offset: float):
        if not isinstance(freq_offset, (int, float)):
            raise SigKitError(f"freq_offset must be a number, got {type(freq_offset)}")
        self.freq_offset = float(freq_offset)

    def apply(self, signal: Signal) -> Signal:
        x: np.ndarray = signal.samples
        if x.dtype != np.complex64:
            raise SigKitError(
                "FrequencyShift impairment expects samples to be of type np.complex64."
            )

        t = np.arange(x.size) / signal.sample_rate
        shifted = x * np.exp(1j * 2 * np.pi * self.freq_offset * t)
        cf = signal.carrier_frequency + self.freq_offset

        return Signal(
            samples=shifted.astype(np.complex64),
            sample_rate=signal.sample_rate,
            carrier_frequency=cf,
        )
