"""Integration test: prepare_dataset from manifest."""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np


def _make_image(path: Path, w=64, h=64):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def test_prepare_from_manifest():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from prepare_dataset import main as prepare_main
    import subprocess

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ann_path = tmp / "annotations.json"
        img_dir = tmp / "images"
        out_dir = tmp / "processed"
        manifest_path = tmp / "manifest.json"

        # Synthetic COCO: 6 images, 2 categories
        coco = {
            "images": [
                {"id": i, "file_name": f"img_{i}.jpg", "width": 64, "height": 64}
                for i in range(6)
            ],
            "annotations": [
                {"id": 1, "image_id": 0, "category_id": 0, "bbox": [5, 5, 20, 20], "area": 400},
                {"id": 2, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 20], "area": 400},
                {"id": 3, "image_id": 2, "category_id": 0, "bbox": [5, 5, 20, 20], "area": 400},
                {"id": 4, "image_id": 3, "category_id": 1, "bbox": [5, 5, 20, 20], "area": 400},
                {"id": 5, "image_id": 4, "category_id": 0, "bbox": [5, 5, 20, 20], "area": 400},
                {"id": 6, "image_id": 5, "category_id": 1, "bbox": [5, 5, 20, 20], "area": 400},
            ],
            "categories": [
                {"id": 0, "name": "cat"},
                {"id": 1, "name": "dog"},
            ],
        }
        ann_path.write_text(json.dumps(coco))
        for i in range(6):
            _make_image(img_dir / f"img_{i}.jpg")

        # Manifest: fixed splits
        manifest = {
            "task": "test_task",
            "data_name": "test_out",
            "seed": 42,
            "class_mapping": {"0": 0, "1": 1},
            "class_names": {"0": "cat", "1": "dog"},
            "splits": {"train": [0, 1, 2, 3], "val": [4], "test": [5]},
            "n_classes": 2,
        }
        manifest_path.write_text(json.dumps(manifest))

        # Run prepare_dataset with manifest
        sys_argv = [
            "prepare_dataset.py",
            "--annotations", str(ann_path),
            "--images-dir", str(img_dir),
            "--output-dir", str(out_dir),
            "--name", "test_out",
            "--manifest", str(manifest_path),
        ]
        import sys
        old_argv = sys.argv
        sys.argv = sys_argv
        try:
            ret = prepare_main()
        finally:
            sys.argv = old_argv
        assert ret == 0

        # Verify outputs exist
        assert (out_dir / "test_out" / "images" / "train").exists()
        assert (out_dir / "test_out" / "labels" / "train").exists()
        assert (out_dir / "test_out" / "dataset.yaml").exists()
        assert (out_dir / "test_out" / "metadata.json").exists()

        # Verify split IDs unchanged
        meta = json.loads((out_dir / "test_out" / "metadata.json").read_text())
        assert meta["splits"]["train"] == 4
        assert meta["splits"]["val"] == 1
        assert meta["splits"]["test"] == 1
        assert meta["subset_actual"] == 6
        assert meta["n_categories"] == 2
