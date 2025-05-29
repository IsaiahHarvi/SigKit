import numpy as np

from sigkit.core.base import Signal
from sigkit.impairments.base import Impairment


class AWGN(Impairment):
    """Apply Additive White Gaussian Noise to a Signal"""

    def __init__(self, snr_db: float):
        self.snr_db = snr_db

    def apply(self, signal: Signal) -> Signal:
        """
        Expects a Signal object with an np.ndarray of np.complex64 samples.
        Returns the Signal with AWGN applied to the target snr_db.
        """
        x: np.ndarray = signal.samples
        sig_power = np.mean(np.abs(x) ** 2)
        snr_lin = 10.0 ** (self.snr_db / 10.0)
        noise_power = sig_power / snr_lin

        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(*x.shape) + 1j * np.random.randn(*x.shape)
        )

        return Signal(
            samples=x + noise,
            sample_rate=signal.sample_rate,
            center_freq=signal.center_freq,
        )
