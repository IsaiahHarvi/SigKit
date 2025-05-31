import numpy as np
import pytest

from sigkit.core.base import SigKitError, Signal
from sigkit.impairments.frequency_shift import FrequencyShift


def test_frequency_shift_basic_tone():
    N = 256
    fs_hz = 1e3
    samples = np.ones(N, dtype=np.complex64)
    sig = Signal(samples=samples, sample_rate=fs_hz, carrier_frequency=0.0)

    f_offset = 100.0
    fs_imp = FrequencyShift(freq_offset=f_offset)
    shifted_sig = fs_imp.apply(sig)

    assert shifted_sig.carrier_frequency == pytest.approx(f_offset, rel=1e-6)

    n = np.arange(N)
    expected = np.exp(1j * 2 * np.pi * f_offset * n / fs_hz).astype(np.complex64)

    assert np.allclose(
        np.real(shifted_sig.samples), np.real(expected), atol=1e-5
    )
    assert np.allclose(
        np.imag(shifted_sig.samples), np.imag(expected), atol=1e-5
    )


def test_frequency_shift_negative_offset():
    N = 128
    fs_hz = 8e2
    samples = np.ones(N, dtype=np.complex64)
    sig = Signal(samples=samples, sample_rate=fs_hz, carrier_frequency=0.0)

    f_offset = -200.0
    fs_imp = FrequencyShift(freq_offset=f_offset)
    shifted_sig = fs_imp.apply(sig)

    assert shifted_sig.carrier_frequency == pytest.approx(f_offset, rel=1e-6)

    n = np.arange(N)
    expected = np.exp(1j * 2 * np.pi * f_offset * n / fs_hz).astype(np.complex64)

    assert np.allclose(
        shifted_sig.samples, expected, atol=1e-5
    )

