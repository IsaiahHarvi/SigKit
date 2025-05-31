"""Module for PhaseShift Torch Transform."""

import math
from typing import Tuple

import torch
from torch import nn

from sigkit.core.base import SigKitError


class ApplyPhaseShift(nn.Module):
    """Apply a constant or random phase offset to a (2, N) I/Q torch.Tensor.

    Args:
        phase_offset:
            - If float or int: apply that fixed phase (radians).
            - If tuple of two numbers (min_phase, max_phase):
              pick a random phase uniformly in each forward() call.
    """

    def __init__(self, phase_offset: float | Tuple[float, float]):
        super().__init__()
        if isinstance(phase_offset, (int, float)):
            self.min_phi = float(phase_offset)
            self.max_phi = float(phase_offset)
        elif (
            isinstance(phase_offset, (tuple, list))
            and len(phase_offset) == 2
            and all(isinstance(p, (int, float)) for p in phase_offset)
        ):
            self.min_phi = float(phase_offset[0])
            self.max_phi = float(phase_offset[1])
        else:
            raise SigKitError(
                "ApplyPhaseShift: phase_offset must be a number or 2‐tuple range,"
                f"got {phase_offset!r}"
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies a PhaseShift to the Tensor.

        Args:
            x: torch.Tensor of shape (2, N), dtype=torch.float32, where
               row 0 = I, row 1 = Q.

        Returns:
            A new torch.Tensor of shape (2, N), dtype=torch.float32, with a phase
            rotation by phi, where phi is either the fixed value or
            a random sample in [min_phi, max_phi].
        """
        if (x.ndim != 2) or (x.shape[0] != 2) or (x.dtype != torch.float32):
            raise SigKitError(f"PhaseShiftTorch expects shape (2, N), got {x.shape=}")

        if self.min_phi == self.max_phi:  # fixed
            phi = self.min_phi
        else:
            r = torch.rand(1).item()
            phi = self.min_phi + (self.max_phi - self.min_phi) * r

        c = math.cos(phi)
        s = math.sin(phi)

        # (2, N) row 0 = I, row 1 = Q
        return torch.stack([(x[0] * c - x[1] * s), (x[0] * s + x[1] * c)], dim=0)

