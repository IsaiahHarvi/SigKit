import numpy as np
import pytest

from sigkit.core.base import Signal
from sigkit.impairments.awgn import AWGN
from sigkit.metrics.integrity import estimate_snr


@pytest.mark.parametrize("snr_db", [0, 10, 20, 30])
def test_awgn_np(snr_db):
    awgn = AWGN(snr_db=snr_db)
    signal = Signal(
        samples=(np.exp(1j * 2 * np.pi * np.arange(4096) / 4096).astype(np.complex64)),
        sample_rate=1e6,
        carrier_frequency=0.0,
    )
    noisy = awgn.apply(signal)
    measured = estimate_snr(signal.samples, noisy.samples)

    # allow ±1 dB tolerance
    assert (
        abs(measured - snr_db) < 1.0
    ), f"measured {measured:.2f} dB != target {snr_db} dB"
