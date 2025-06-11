"""Utility file for SigKit Training and Inference processes."""

from sigkit.core.base import SigKitError

PSK_CLASS_MAP = {
    "2-PSK": 0,
    "4-PSK": 1,
    "8-PSK": 2,
    "16-PSK": 3,
}

FSK_CLASS_MAP = {
    "2-FSK": 0,
    "4-FSK": 1,
    "8-FSK": 2,
    "16-FSK": 3,
}

CLASS_MAP = {
    "2-PSK": 0,
    "4-PSK": 1,
    "8-PSK": 2,
    "16-PSK": 3,
    "2-FSK": 4,
    "4-FSK": 5,
    "8-FSK": 6,
    "16-FSK": 7,
}


def get_class_name(class_idx: int) -> str:
    """Returns the class name for a given class index.

    Args:
        class_idx (int): The class index.

    Returns:
        str: The class name.
    """
    rev_class_map = {v: k for k, v in CLASS_MAP.items()}
    if class_idx < 0 or class_idx >= len(rev_class_map):
        raise SigKitError(f"Class index {class_idx} is out of bounds.")
    return rev_class_map[class_idx]


def get_class_index(class_name: str) -> int:
    """Returns the class index for a given class name.

    Args:
        class_name (str): The class name.

    Returns:
        int: The class index.
    """
    if class_name not in CLASS_MAP:
        raise SigKitError(f"Class name '{class_name}' not found in CLASS_MAP.")
    return CLASS_MAP[class_name]
