"""Methods for computing signal integrity metrics like SNR and BER."""

from typing import Union

import numpy as np
import torch


def estimate_snr(
    clean: Union[np.ndarray, torch.Tensor],
    noisy: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute SNR (dB) between clean and noisy signals."""
    if isinstance(clean, torch.Tensor) and isinstance(noisy, torch.Tensor):
        return _estimate_snr_torch(clean, noisy)
    if isinstance(clean, np.ndarray) and isinstance(noisy, np.ndarray):
        return _estimate_snr_np(clean, noisy)

    raise ValueError(f"Type mismatch: got {type(clean)} vs {type(noisy)}")


def _estimate_snr_np(clean: np.ndarray, noisy: np.ndarray) -> float:
    """NumPy implementation of SNR in dB."""
    sig_power = np.mean(np.abs(clean) ** 2)
    noise_power = np.mean(np.abs(noisy - clean) ** 2)
    return 10 * np.log10(sig_power / noise_power)


def _estimate_snr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """PyTorch implementation of SNR in dB.

    Expects real-imag stacked: shape (2, N).
    """
    sig_power = (x.pow(2).sum(dim=0)).mean().item()
    noise_power = ((y - x).pow(2).sum(dim=0)).mean().item()
    return 10 * torch.log10(torch.tensor(sig_power / noise_power)).item()


def calculate_ber(
    bits: Union[np.ndarray, torch.Tensor],
    truth_bits: Union[np.ndarray, torch.Tensor],
) -> float:
    """Compute bit-error rate (fraction of mismatches)."""
    # If torch, convert to NumPy
    if isinstance(truth_bits, torch.Tensor):
        truth_bits = truth_bits.cpu().numpy()
    if isinstance(bits, torch.Tensor):
        bits = bits.cpu().numpy()

    errors = np.count_nonzero(truth_bits != bits)
    return errors / truth_bits.size
