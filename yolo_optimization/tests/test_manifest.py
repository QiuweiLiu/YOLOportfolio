"""Test: frozen manifest round-trip."""

import json
from pathlib import Path

MANIFEST = Path("yolo_optimization/data_manifests/task_v1.json")


def test_manifest_exists():
    assert MANIFEST.exists(), "task_v1.json 不存在"
    data = json.loads(MANIFEST.read_text())
    assert "splits" in data
    assert "class_mapping" in data
    assert "class_names" in data


def test_manifest_splits_fixed():
    data = json.loads(MANIFEST.read_text())
    # train/val/test 应固定为 640/80/80
    assert data["n_images_per_split"]["train"] == 640
    assert data["n_images_per_split"]["val"] == 80
    assert data["n_images_per_split"]["test"] == 80


def test_manifest_class_consistency():
    data = json.loads(MANIFEST.read_text())
    assert data["n_classes"] == len(data["class_names"])
    assert data["n_classes"] == len(data["class_mapping"])


def test_manifest_id_uniqueness():
    data = json.loads(MANIFEST.read_text())
    all_ids = []
    for ids in data["splits"].values():
        all_ids.extend(ids)
    assert len(all_ids) == len(set(all_ids)), "image_id 在不同 split 中重复"
