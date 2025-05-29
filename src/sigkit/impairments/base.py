"""ABC Module for the Impairment package."""

from abc import ABC, abstractmethod

from sigkit.core.base import Signal


class Impairment(ABC):
    """Base for all numpy signal impairments."""

    @abstractmethod
    def apply(self, signal: Signal) -> Signal:
        """ABC Method for applying an impairment to an input Signal.

        Given a Signal object with np.complex64 samples,
        return a new Signal with the impairment applied.
        """

    def __call__(self, signal: Signal) -> Signal:
        return self.apply(signal)
