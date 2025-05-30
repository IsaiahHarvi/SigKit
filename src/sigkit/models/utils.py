"""Utility file for SigKit Training and Inference processes."""

PSK_CLASS_MAP = {
    0: "2-PSK",
    1: "4-PSK",
    2: "8-PSK",
    3: "16-PSK",
    4: "32-PSK",
    5: "64-PSK",
}

CLASS_MAP = {name: idx for idx, name in PSK_CLASS_MAP.items()}
