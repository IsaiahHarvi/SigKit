"""Phase Shift Module utilized for impairments."""

from typing import Tuple

import numpy as np

from sigkit.core.base import SigKitError, Signal
from sigkit.impairments.base import Impairment


class PhaseShift(Impairment):
    """Apply a constant or random phase offset to a baseband Signal.

    Args:
        phase_offset:
            - If float: apply a fixed phase (radians).
            - If tuple of two numbers (min_phase, max_phase):
              pick a random phase (per call) uniformly in [min_phase, max_phase].
    """

    def __init__(self, phase_offset: float | Tuple[float, float] = (-np.pi, np.pi)):
        if isinstance(phase_offset, (int, float)):
            self.phase_range = (float(phase_offset), float(phase_offset))
        elif (
            isinstance(phase_offset, (tuple, list))
            and len(phase_offset) == 2
            and all(isinstance(p, (int, float)) for p in phase_offset)
        ):
            self.phase_range = (float(phase_offset[0]), float(phase_offset[1]))
        else:
            raise SigKitError(
                "phase_offset must be a number or a 2‐tuple/list of numbers, "
                f"got {phase_offset!r}"
            )

    def apply(self, signal: Signal) -> Signal:
        x: np.ndarray = signal.samples
        if x.dtype != np.complex64:
            raise SigKitError(
                "PhaseShift impairment expects samples to be np.complex64."
            )

        min_ph, max_ph = self.phase_range
        if min_ph == max_ph:  # fixed offset
            phi = min_ph
        else:
            phi = float(np.random.uniform(min_ph, max_ph))

        phase_factor = np.exp(1j * phi).astype(np.complex64)
        shifted = (x * phase_factor).astype(np.complex64)

        return Signal(
            samples=shifted,
            sample_rate=signal.sample_rate,
            carrier_frequency=signal.carrier_frequency,
        )
