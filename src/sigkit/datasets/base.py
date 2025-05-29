"""ABC Module for the Dataset package."""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch
from torch.utils.data import Dataset

from sigkit.core.base import Signal


class SignalDataset(Dataset, ABC):
    """Base class for datasets that return (Signal, label) pairs."""

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Signal, Union[int, torch.Tensor]]:
        """ABC Method for retrieving items from the dataset.

        Return:
            Signal: the input waveform container
            label:  an integer or Tensor of target bits/classes.
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples in the dataset."""
        raise NotImplementedError
