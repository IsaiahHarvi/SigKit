import torch
from abc import ABC, abstractmethod
from typing import Tuple, Union
from core.base import Signal
from typing import Tuple, Union
from torch.utils.data import Dataset


class SignalDataset(Dataset, ABC):
    """
    Base class for datasets that return (Signal, label) pairs.
    """

    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Signal, Union[int, torch.Tensor]]:
        """
        Return:
          - Signal: the input waveform container
          - label:  an integer or Tensor of target bits/classes
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Number of samples in the dataset."""
        raise NotImplementedError

