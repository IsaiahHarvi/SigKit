"""Contains matplotlib visualizations for signal metrics."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from sigkit.core.base import Signal


def plot_constellation(signal: Signal, ax=None, s: int = 20):
    """Plot the constellation diagram of a Signal.

    Args:
        signal: Signal object containing complex samples.
        ax: Optional matplotlib Axes to plot on.
        s: Marker size.
    """
    samples = signal.samples
    real = np.real(samples)
    imag = np.imag(samples)
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(real, imag, s=s)
    ax.set_xlabel("In-phase")
    ax.set_ylabel("Quadrature")
    ax.set_title("Constellation Diagram")
    ax.grid(True)
    return ax


def plot_time(
    signal: Signal,
    ax=None,
    one_symbol: bool = False,
    symbol_rate: Optional[float] = None,
) -> plt.Axes:
    """Plot the real (I) and imaginary (Q) parts of a Signal over time.

    Args:
        signal: Signal object containing complex samples.
        ax: Optional matplotlib Axes to plot on.
        one_symbol: If True, only plot the first symbol period.
        symbol_rate: Symbol rate in Hz; required if one_symbol=True.

    Returns:
        The matplotlib Axes containing the plot.
    """
    samples = signal.samples
    fs = signal.sample_rate
    # full time axis
    t_full = np.arange(samples.size) / fs
    i_full = np.real(samples)
    q_full = np.imag(samples)

    if one_symbol:
        if symbol_rate is None:
            raise ValueError("symbol_rate must be provided when one_symbol=True")
        # compute samples per symbol
        sps = int(fs / symbol_rate)
        t = t_full[:sps]
        i = i_full[:sps]
        q = q_full[:sps]
    else:
        t, i, q = t_full, i_full, q_full

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(t, i, label="I")
    ax.plot(t, q, label="Q")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.set_title("Time-domain Signal")
    return ax


def plot_frequency(signal: Signal, ax=None):
    """Plot the magnitude spectrum of a Signal using FFT.

    Args:
        signal: Signal object containing complex samples.
        ax: Optional matplotlib Axes to plot on.
    """
    samples = signal.samples
    N = samples.size
    fs = signal.sample_rate
    X = np.fft.fftshift(np.fft.fft(samples))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1 / fs))
    mag = np.abs(X)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(freqs, mag)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Frequency Spectrum")
    return ax


def plot_psd(signal: Signal, ax=None, nfft=1024):
    """Plot the Power Spectral Density (PSD) of a Signal.

    Args:
        signal: Signal object containing complex samples.
        ax: Optional matplotlib Axes to plot on.
        nfft: Number of FFT points.
    """
    samples = signal.samples
    fs = signal.sample_rate
    if ax is None:
        fig, ax = plt.subplots()
    # Matplotlib PSD (Welch-like) method
    ax.psd(samples, NFFT=nfft, Fs=fs, scale_by_freq=True)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Power Spectral Density")
    return ax


def plot_spectrogram(signal: Signal, ax=None, nfft=256, noverlap=128, cmap="viridis"):
    """Plot the spectrogram of a Signal using Matplotlib's specgram.

    Args:
        signal: Signal object containing complex samples.
        ax: Optional matplotlib Axes to plot on.
        nfft: Number of FFT points.
        noverlap: Number of overlapping points.
        cmap: Colormap to use.
    """
    samples = signal.samples
    fs = signal.sample_rate
    if ax is None:
        fig, ax = plt.subplots()
    Pxx, freqs, bins, im = ax.specgram(
        samples, NFFT=nfft, Fs=fs, noverlap=noverlap, cmap=cmap, scale="dB"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram (dB)")
    return ax
