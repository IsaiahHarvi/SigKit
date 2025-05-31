import math
import numpy as np
import pytest
import torch

from sigkit.core.base import Signal
from sigkit.impairments.phase_shift import PhaseShift
from sigkit.transforms.phase_shift import ApplyPhaseShift



@pytest.mark.parametrize("phase_offset", [0.0, math.pi / 4, -math.pi / 2])
def test_numpy_phase_shift_fixed_offset(phase_offset):
    N = 8
    samples = np.exp(1j * 2 * np.pi * np.arange(N) / N).astype(np.complex64)
    sig = Signal(samples=samples, sample_rate=1e3, carrier_frequency=0.0)

    imp = PhaseShift(phase_offset=phase_offset)
    out = imp.apply(sig)

    expected = (samples * np.exp(1j * phase_offset)).astype(np.complex64)

    assert np.allclose(np.real(out.samples), np.real(expected), atol=1e-6)
    assert np.allclose(np.imag(out.samples), np.imag(expected), atol=1e-6)
    assert out.carrier_frequency == pytest.approx(0.0)


@pytest.mark.parametrize("phase_range", [(-np.pi, np.pi), (0.0, np.pi / 2)])
def test_numpy_phase_shift_random_in_range(phase_range):
    np.random.seed(42)

    N = 16
    samples = np.ones(N, dtype=np.complex64)
    sig = Signal(samples=samples, sample_rate=1e3, carrier_frequency=0.0)

    imp = PhaseShift(phase_offset=phase_range)
    out = imp.apply(sig)

    first_phase = np.arctan2(np.imag(out.samples[0]), np.real(out.samples[0]))

    if first_phase < -np.pi:
        first_phase += 2 * np.pi
    elif first_phase > np.pi:
        first_phase -= 2 * np.pi

    min_ph, max_ph = phase_range
    assert min_ph <= first_phase <= max_ph, f"phase {first_phase} not in [{min_ph}, {max_ph}]"


@pytest.mark.parametrize("phase_offset", [0.0, math.pi / 4, -math.pi / 2])
def test_torch_phase_shift_fixed_offset(phase_offset):
    N = 4096
    theta = 2 * math.pi * torch.arange(N, dtype=torch.float32) / N
    x = torch.stack([torch.cos(theta), torch.sin(theta)], dim=0)  # shape: (2, 4096)

    imp = ApplyPhaseShift(phase_offset=phase_offset)
    y = imp(x)

    samples_np = (
        (x[0].numpy() + 1j * x[1].numpy()) * np.exp(1j * phase_offset)
    ).astype(np.complex64)
    expected_I = np.real(samples_np).astype(np.float32)
    expected_Q = np.imag(samples_np).astype(np.float32)

    assert y.shape == x.shape
    assert y.dtype == torch.float32
    assert torch.allclose(y[0], torch.from_numpy(expected_I), atol=1e-6)
    assert torch.allclose(y[1], torch.from_numpy(expected_Q), atol=1e-6)


@pytest.mark.parametrize("phase_range", [(-np.pi, np.pi), (0.0, np.pi / 2)])
def test_torch_phase_shift_random_in_range(phase_range):
    torch.manual_seed(1234)

    N = 4096
    x = torch.zeros(2, N, dtype=torch.float32)
    x[0] = 1.0  # I = 1
    x[1] = 0.0  # Q = 0

    imp = ApplyPhaseShift(phase_offset=phase_range)
    y = imp(x)

    first_I = y[0, 0].item()
    first_Q = y[1, 0].item()
    first_phase = math.atan2(first_Q, first_I)

    if first_phase < -np.pi:
        first_phase += 2 * np.pi
    elif first_phase > np.pi:
        first_phase -= 2 * np.pi

    min_ph, max_ph = phase_range
    assert min_ph <= first_phase <= max_ph, f"phase {first_phase} not in [{min_ph}, {max_ph}]"

