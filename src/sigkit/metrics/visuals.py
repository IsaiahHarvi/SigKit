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

    Returns:
        The matplotlib Axes containing the plot.
    """
    if signal.carrier_frequency != 0.0:
        signal = signal.to_baseband()
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
    n_samples: Optional[int] = None,
    ax=None,
) -> plt.Axes:
    """Plot the real (I) part of a Signal over time.

    Args:
        signal: Signal object containing complex samples.
        n_samples: Optional number of samples to plot. If None, plots all.
        ax: Optional matplotlib Axes to plot on.

    Returns:
        The matplotlib Axes containing the plot.
    """
    t = np.arange(signal.samples.size) / signal.sample_rate
    i = np.real(signal.samples)

    if n_samples is not None:
        t = t[:n_samples]
        i = i[:n_samples]

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(t, i, label="Real")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.set_title("Time-domain Signal")
    plt.tight_layout()
    return ax


def plot_phase(
    signal: Signal, ax=None, unwrap: bool = False, n_samples: Optional[int] = None
) -> plt.Axes:
    """Plot the instantaneous phase of a Signal over time.

    Args:
        signal: Signal object containing complex samples.
        ax: Optional matplotlib Axes to plot on.
        unwrap: If True, unwrap the phase to show continuous variation.
        n_samples: Optional number of samples to plot. If None, plots all.

    Returns:
        The matplotlib Axes containing the plot.
    """
    phi = np.angle(signal.samples)
    if unwrap:
        phi = np.unwrap(phi)
    t = np.arange(phi.size) / signal.sample_rate

    if n_samples is not None:
        t = t[:n_samples]
        phi = phi[:n_samples]

    if ax is None:
        fig, ax = plt.subplots()

    ax.plot(t, phi)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Phase (rad)")
    ax.set_title("Instantaneous Phase")
    return ax


def plot_frequency(signal: Signal, ax=None):
    """Plot the magnitude spectrum of a Signal using FFT.

    Args:
        signal: Signal object containing complex samples.
        ax: Optional matplotlib Axes to plot on.

    Returns:
        The matplotlib Axes containing the plot.
    """
    N = signal.samples.size
    fs = signal.sample_rate
    X = np.fft.fftshift(np.fft.fft(signal.samples))
    freqs = np.fft.fftshift(np.fft.fftfreq(N, d=1 / fs))
    mag = np.abs(X)
    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(freqs, mag)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Frequency Spectrum")
    plt.tight_layout()
    return ax


def plot_psd(signal: Signal, ax=None, nfft=1024):
    """Plot the Power Spectral Density (PSD) of a Signal.

    Args:
        signal: Signal object containing complex samples.
        ax: Optional matplotlib Axes to plot on.
        nfft: Number of FFT points.

    Returns:
        The matplotlib Axes containing the plot.
    """
    fs = signal.sample_rate
    if ax is None:
        fig, ax = plt.subplots()
    # Matplotlib PSD (Welch-like) method
    ax.psd(signal.samples, NFFT=nfft, Fs=fs, scale_by_freq=True)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD")
    ax.set_title("Power Spectral Density")
    plt.tight_layout()
    return ax


def plot_spectrogram(signal: Signal, ax=None, nfft=2048, noverlap=None, cmap="viridis"):
    """Plot the spectrogram of a Signal using Matplotlib's specgram.

    Args:
        signal: Signal object containing complex samples.
        ax: Optional matplotlib Axes to plot on.
        nfft: Number of FFT points.
        noverlap: Number of overlapping points.
        cmap: Colormap to use.

    Returns:
        The matplotlib Axes containing the plot.
    """
    fs = signal.sample_rate
    if ax is None:
        fig, ax = plt.subplots()
    Pxx, freqs, bins, im = ax.specgram(
        signal.samples, NFFT=nfft, Fs=fs, noverlap=noverlap, cmap=cmap, scale="dB"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title("Spectrogram (dB)")
    plt.tight_layout()
    return ax
