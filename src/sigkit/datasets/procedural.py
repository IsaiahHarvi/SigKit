import numpy as np
import random
from typing import List, Dict, Type, Tuple, Optional
import torch
from torch.utils.data import Dataset

from sigkit.core.base import Signal
from sigkit.modem.base import Modem
from sigkit.models.utils import CLASS_MAP


class ProceduralDataset(Dataset):
    """
    Procedural map-style dataset generating an (effectively) infinite stream of symbols.

    Args:
        mapping_list: List of dicts mapping a Modem subclass to list of constellation sizes.
            e.g. [{PSK: [2,4,8,16]}, {QAM: [4,16,64]}]
        sample_rate: Sampling rate (Hz) for all modems.
        symbol_rate: Symbol rate (Hz) for all modems.

    Behavior:
        - Instantiates one modem per (ModemClass, constellation) entry.
        - __getitem__ ignores idx and returns a random (Signal, symbol_idx).
        - __len__ returns length (default a very large number to emulate infinite).
    """

    def __init__(
        self,
        mapping_list: List[Dict[Type[Modem], List[int]]],
        sample_rate: int = 1024,
        symbol_rate: int = 32,
        length: int = (2**31 - 1),
        seed: Optional[int] = None,
    ):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # instantiate modem instances
        self.modems: List[Tuple[Modem, str]] = []
        for mapping in mapping_list:
            if not isinstance(mapping, dict):
                raise ValueError(
                    "Each mapping must be dict {ModemClass: [constellations]}"
                )
            for modem_cls, consts in mapping.items():
                if not issubclass(modem_cls, Modem):
                    raise ValueError(f"Key must be Modem subclass, got {modem_cls}")
                for M in consts:
                    if not isinstance(M, int) or M < 2:
                        raise ValueError(f"Constellation size must be int>=2, got {M}")

                    modem = modem_cls(
                        sample_rate=sample_rate,
                        symbol_rate=symbol_rate,
                        n_components=M,  # further validation in modem subclasses
                        cf=0.0,
                    )
                    cls_idx = f"{M}-{modem_cls.__name__}"
                    self.modems.append((modem, cls_idx))
        if not self.modems:
            raise ValueError("No modem instances created; check mapping_list")
        self.length = length

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        modem, cls_idx = random.choice(self.modems)

        if (4096 % modem.sps) != 0:
            raise ValueError(f"Desired length 4096 not divisible by {modem.sps=}")

        num_symbols = 4096 // modem.sps
        bits = np.random.randint(
            0, 2, size=(num_symbols * modem.bits_per_symbol,), dtype=np.uint8
        )

        signal: Signal = modem.modulate(bits)
        if signal.samples.size != 4096:
            raise AssertionError(
                f"Generated waveform length {signal.samples.size} != 4096"
            )
        return signal.to_tensor(), CLASS_MAP[cls_idx]
