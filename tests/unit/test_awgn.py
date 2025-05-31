import pytest
import numpy as np
import torch

from sigkit.metrics.integrity import estimate_snr
from sigkit.transforms.awgn import ApplyAWGN
from sigkit.impairments.awgn import AWGN
from sigkit.core.base import Signal


@pytest.mark.parametrize("snr_db", [0, 10, 20, 30])
def test_awgn_torch(snr_db):
    awgn = ApplyAWGN(snr_db=snr_db)

    theta = 2 * torch.pi * torch.arange(4096, dtype=torch.float32) / 4096
    x = torch.stack([torch.cos(theta), torch.sin(theta)], dim=0)  # (2, 4096)

    y = awgn(x)
    assert y.shape == x.shape, "Output shape should match input shape"

    measured = estimate_snr(x, y)
    assert abs(measured - snr_db) < 1.0, f"measured {measured:.2f} dB != target {snr_db} dB"


@pytest.mark.parametrize("snr_db", [0, 10, 20, 30])
def test_awgn_np(snr_db):
    awgn = AWGN(snr_db=snr_db)

    samples = (
        np.exp(1j * 2 * np.pi * np.arange(4096) / 4096)
        .astype(np.complex64)
    )
    signal = Signal(
        samples=samples,
        sample_rate=1e6,
        carrier_frequency=0.0,
    )

    noisy = awgn.apply(signal)
    measured = estimate_snr(signal.samples, noisy.samples)

    assert abs(measured - snr_db) < 1.0, f"measured {measured:.2f} dB != target {snr_db} dB"

