"""Test: device selection logic."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from check_environment import select_device
from train import pick_device
import torch


def test_select_device_cuda():
    assert select_device(True, False) == "cuda"
    assert select_device(True, True) == "cuda"


def test_select_device_mps():
    assert select_device(False, True) == "mps"
    assert pick_device("") in ("cuda:0", "mps", "cpu")


def test_select_device_cpu():
    assert select_device(False, False) == "cpu"


def test_mps_available():
    # Only check type, not force mps
    assert isinstance(torch.backends.mps.is_available(), bool)
