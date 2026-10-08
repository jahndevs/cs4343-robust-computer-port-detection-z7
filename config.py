"""Shared paths and settings used by all scripts."""
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data.yaml"
DATASET = ROOT / "prepared"
RUNS = ROOT / "runs"
CLASSES = ["displayport", "ethernet", "hdmi", "usb-a", "usb-c", "vga"]


def pick_device():
    """CUDA GPU if available, then Apple Silicon GPU, then CPU."""
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
