"""AWGN Method for the Impairments Module."""

import numpy as np

from sigkit.core.base import SigKitError, Signal
from sigkit.impairments.base import Impairment


class AWGN(Impairment):
    """Apply Additive White Gaussian Noise to a Signal."""

    def __init__(self, snr_db: float):
        self.snr_db = snr_db

    def apply(self, signal: Signal) -> Signal:
        """Applies AWGN to the samples of input Signal and returns a new Signal.

        Expects a Signal object with an np.ndarray of np.complex64 samples.
        Returns the Signal with AWGN applied to the target snr_db.
        """
        x: np.ndarray = signal.samples
        if not x.dtype == np.complex64:
            raise SigKitError(
                "AWGN impairment expects samples to be of type np.complex64."
            )

        sig_power = np.mean(np.abs(x) ** 2)
        snr_lin = 10.0 ** (self.snr_db / 10.0)
        noise_power = sig_power / snr_lin

        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)
        )

        return Signal(
            samples=(x + noise).astype(np.complex64),
            sample_rate=signal.sample_rate,
            carrier_frequency=signal.carrier_frequency,
        )
