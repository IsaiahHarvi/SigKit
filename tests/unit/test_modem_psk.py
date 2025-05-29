import numpy as np
import pytest

from sigkit.metrics.integrity import calculate_ber
from sigkit.modem.psk import PSK


@pytest.mark.parametrize("n_components", [2, 4, 8, 16, 32, 64])
@pytest.mark.parametrize("cf", [0.0, 256.0])
def test_psk_modem(n_components, cf):
    """
    For each supported M-PSK (M=2,4,8,16,32,64) and for CF=0 (baseband) and CF=256 Hz,
    modulate random bits, demodulate them, and assert BER==0.
    """
    sample_rate = 1024
    symbol_rate = 32

    modem = PSK(
        sample_rate=sample_rate,
        symbol_rate=symbol_rate,
        n_components=n_components,
        cf=cf,
    )

    num_symbols = 100
    bits = np.random.randint(
        0, 2, size=num_symbols * modem.bits_per_symbol, dtype=np.uint8
    )
    signal = modem.modulate(bits)
    demod_bits = modem.demodulate(signal)

    assert demod_bits.shape == bits.shape
    assert np.array_equal(bits, demod_bits)
    assert calculate_ber(bits, demod_bits) == 0.0
