"""Frequency Shift Keying Module."""

import numpy as np

from sigkit.core.base import SigKitError, Signal
from sigkit.modem.base import Modem


class FSK(Modem):
    """FSK Modem for modulating and demodulating bits."""

    def __init__(
        self, sample_rate: int, symbol_rate: int, n_components, cf: float = 0.0
    ):
        """N-FSK Modem.

        Args:
            sample_rate: Sampling rate of the waveform
            symbol_rate: Symbol rate, used to calculate samples per symbol
            n_components: Number of FSK tones (e.g. 2, 4, 8, 16..)
            cf: Carrier frequency
        """
        super().__init__(sample_rate, symbol_rate, n_components, cf)

        if n_components > self.sps:
            raise SigKitError(
                f"samples_per_symbol ({self.sps}) must be ≥ {n_components=})"
            )

        tones = [(cf + (i * symbol_rate)) for i in range(n_components)]
        self.tones = np.array(tones, dtype=np.float32)

    def modulate(self, bits: np.ndarray) -> Signal:
        """Modulate bits with FSK.

        Args:
            bits: 1D array of 0 | 1, length multiple of log2(n_components)

        Returns:
            Signal: containing complex64 samples
        """
        if bits.ndim != 1 or bits.size % self.bits_per_symbol != 0:
            raise SigKitError(
                f"Number of bits must be a multiple of {self.bits_per_symbol}"
            )

        bits = bits.reshape(-1, self.bits_per_symbol)
        num_symbols = bits.shape[0]
        num_samples = num_symbols * self.sps

        symbol_tones = np.zeros(num_symbols, dtype=np.float32)
        for i in range(num_symbols):
            symbol_index = 0
            for bit in bits[i]:
                symbol_index = (symbol_index << 1) | int(bit)

            symbol_tone = self.tones[symbol_index]
            symbol_tones[i] = symbol_tone

        samples = np.zeros(num_samples, dtype=np.complex64)
        for i in range(num_symbols):
            symbol_tone = symbol_tones[i]
            base = (2.0 * np.pi * symbol_tone) / self.sample_rate

            phase = base * np.arange(self.sps)
            i_samples = np.cos(phase)
            q_samples = np.sin(phase)
            chunk = i_samples + (1j * q_samples)

            start = self.sps * i
            end = self.sps * (i + 1)
            samples[start:end] = chunk.astype(np.complex64)

        return Signal(
            samples=np.array(samples, dtype=np.complex64),
            sample_rate=self.sample_rate,
            carrier_frequency=self.cf,
        )

    def demodulate(self, signal: Signal | np.ndarray) -> np.ndarray:
        """Map received FSK samples to bits.

        Args:
            signal: Signal containing modulated complex samples.

        Returns:
            1D array of bits.
        """
        samples = signal.samples if isinstance(signal, Signal) else signal
        if not samples.dtype == np.complex64:
            raise SigKitError("Demodulate expects samples to be of type np.complex64.")

        bins = np.zeros(len(self.tones), dtype=int)
        for i, tone in enumerate(self.tones):
            tone_mod = tone % self.sample_rate
            bin_index = int(round(tone_mod * self.sps / self.sample_rate))
            bins[i] = bin_index

        num_symbols = samples.size // self.sps
        symbols = samples.reshape(num_symbols, self.sps)
        spectrum = np.fft.fft(symbols, axis=1)
        magnitudes = np.abs(spectrum[:, bins])
        symbol_indices = np.argmax(magnitudes, axis=1)

        num_bits = num_symbols * self.bits_per_symbol
        output = np.zeros(num_bits, dtype=np.uint8)
        for symbol_index in range(num_symbols):
            symbol = symbol_indices[symbol_index]
            for bit_pos in range(self.bits_per_symbol):
                sample_index = (symbol_index * self.bits_per_symbol) + bit_pos
                shift = self.bits_per_symbol - bit_pos - 1
                output[sample_index] = (symbol >> shift) & 1

        return output
