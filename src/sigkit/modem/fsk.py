"""Frequency Shift Keying Module."""

import math

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

        tones = [
            (cf + (i * symbol_rate)) for i in range(n_components)
        ]
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

        symbol_tones = []
        for symbol_bits in bits:
            symbol_index = 0
            for bit in symbol_bits:
                symbol_index = (symbol_index << 1) | int(bit)

            symbol_tone = self.tones[symbol_index]
            symbol_tones.append(symbol_tone)

        samples = []
        for symbol_tone in symbol_tones:
            base = (2. * np.pi * symbol_tone) / self.sample_rate
            for n in range(self.sps):
                phase = base * n
                i = math.cos(phase)
                q = math.sin(phase)
                sample = i + (1j * q)
                samples.append(sample)

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
        symbol_indices = []
        for i in range(num_symbols):
            start = i * self.sps
            end = start + self.sps
            chunk = samples[start:end]

            spectrum = np.fft.fft(chunk)
            magnitudes = np.abs(spectrum[bins])
            best_tone_index = int(np.argmax(magnitudes))
            symbol_indices.append(best_tone_index)


        output = []
        for i in symbol_indices:
            for bit_pos in range(self.bits_per_symbol - 1, -1, -1):
                output.append((i >> bit_pos) & 1)

        return np.array(output, dtype=np.uint8)
