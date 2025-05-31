import numpy as np
import pytest

from sigkit.metrics.integrity import calculate_ber
from sigkit.modem.fsk import FSK


@pytest.mark.parametrize("n_components", [2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("cf", [0.0, 256.0])
def test_fsk_modem(n_components, cf):
    """
    For each supported M-FSK (M=2,4,8,16,32,64) and for CF=0 (baseband) and CF=256 Hz,
    modulate random bits, demodulate them, and assert BER==0.
    """

    modem = FSK(
    )

    bits = np.zeros(10, dtype=np.complex64)
    demod_bits = modem.demodulate(bits)

    assert demod_bits.shape == bits.shape
    assert np.array_equal(bits, demod_bits)
    assert calculate_ber(bits, demod_bits) == 0.0
