"""Test: metrics JSON I/O."""

import json
import tempfile
from pathlib import Path


def test_evaluation_json_loadable():
    p = Path("yolo_optimization/results/experiments/baseline_v1/evaluation.json")
    if not p.exists():
        # 如果 baseline 未跑,找 exp003
        p = Path("yolo_optimization/results/experiments/exp003_v3/evaluation.json")
        if not p.exists():
            return
    data = json.loads(p.read_text())
    assert "mAP50" in data
    assert "precision" in data
    assert "recall" in data


def test_metrics_relative_path():
    p = Path("yolo_optimization/results/experiments/baseline_v1/metrics.json")
    if not p.exists():
        p = Path("yolo_optimization/results/experiments/exp003_v3/metrics.json")
        if not p.exists():
            return
    data = json.loads(p.read_text())
    if "best_weights" in data:
        assert not data["best_weights"].startswith("/Users/"), "不应含绝对路径"
        assert not data["best_weights"].startswith("/home/")


def test_comparison_json():
    p = Path("yolo_optimization/results/comparison/comparison.json")
    if not p.exists():
        return
    data = json.loads(p.read_text())
    assert "comparison_table" in data
