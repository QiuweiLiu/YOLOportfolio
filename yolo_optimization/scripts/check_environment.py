#!/usr/bin/env python
"""环境检查脚本 — Phase 0/1 验证。

检查项(需求文档 §5):
- Python / PyTorch / Ultralytics / OpenCV 版本
- CUDA / MPS 可用性
- 当前推荐 device
- 系统架构
- 必要目录是否存在

用法:
    python yolo_optimization/scripts/check_environment.py
"""

from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path

# 项目根 = 本脚本的 ../../..(yolo_optimization/scripts/check_environment.py → 根)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

REQUIRED_DIRS = [
    "yolo_optimization/configs",
    "yolo_optimization/scripts",
    "yolo_optimization/results",
    "yolo_optimization/reports",
    "yolo_optimization/tests",
    "data/raw",
    "experiments",
    "outputs",
]

# 最少依赖,import 失败时单独报错而不是让整个脚本崩掉
def _module_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "unknown")
    except ImportError:
        return "NOT INSTALLED"


def select_device(cuda_ok: bool, mps_ok: bool) -> str:
    """按需求文档 §1 的顺序:cuda → mps → cpu。"""
    if cuda_ok:
        return "cuda"
    if mps_ok:
        return "mps"
    return "cpu"


def main() -> int:
    print("=" * 60)
    print("YOLO Optimization Portfolio — Environment Report")
    print("=" * 60)

    # --- 系统与解释器 ---
    print(f"\n[System]")
    print(f"  platform : {platform.platform()}")
    print(f"  machine  : {platform.machine()}")
    print(f"  python   : {sys.version.split()[0]} ({sys.executable})")

    # --- 包版本 ---
    print(f"\n[Packages]")
    torch_version = _module_version("torch")
    print(f"  torch        : {torch_version}")
    print(f"  ultralytics  : {_module_version('ultralytics')}")
    print(f"  opencv       : {_module_version('cv2')}")
    print(f"  numpy        : {_module_version('numpy')}")

    # --- 设备 ---
    print(f"\n[Device]")
    cuda_ok = False
    mps_ok = False
    if torch_version != "NOT INSTALLED":
        import torch

        cuda_ok = torch.cuda.is_available()
        mps_ok = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()
    print(f"  CUDA available : {cuda_ok}")
    print(f"  MPS  available : {mps_ok}")
    if torch_version != "NOT INSTALLED":
        print(f"  recommended    : {select_device(cuda_ok, mps_ok)}")

    # --- 磁盘 ---
    total, used, free = shutil.disk_usage(PROJECT_ROOT)
    print(f"\n[Disk @ {PROJECT_ROOT}]")
    print(f"  total {total / 1e9:.1f} GB | free {free / 1e9:.1f} GB")

    # --- 目录 ---
    print(f"\n[Directories]")
    missing = []
    for d in REQUIRED_DIRS:
        p = PROJECT_ROOT / d
        ok = p.is_dir()
        print(f"  {'OK ' if ok else 'MISS'} {d}")
        if not ok:
            missing.append(d)

    print()
    if missing:
        print(f"[WARN] 缺失目录: {missing}")
    else:
        print("[OK] 所有必要目录存在")
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())