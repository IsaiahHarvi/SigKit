import pytest
import torch

from sigkit.metrics.integrity import estimate_snr
from sigkit.transforms.awgn import ApplyAWGN


@pytest.mark.parametrize("snr_db", [0, 10, 20, 30])
def test_awgn_torch(snr_db):
    awgn = ApplyAWGN(snr_db=snr_db)
    theta = 2 * torch.pi * torch.arange(4096, dtype=torch.float32) / 4096

    x = torch.stack([torch.cos(theta), torch.sin(theta)], dim=0)  # shape (2, 4096)
    y = awgn(x)
    assert y.shape == x.shape
    measured = estimate_snr(x, y)

    # allow ±1 dB tolerance
    assert abs(measured - snr_db) < 1.0, (
        f"measured {measured:.2f}dB != target {snr_db}dB"
    )
