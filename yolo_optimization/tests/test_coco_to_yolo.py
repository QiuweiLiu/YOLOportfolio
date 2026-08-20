"""Test: COCO → YOLO conversion."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_dataset import write_yolo_txt


def test_bbox_normalization():
    im = {"file_name": "batch_1/000001.jpg", "width": 640, "height": 480}
    ann = {"category_id": 0, "bbox": [0, 0, 320, 240], "area": 76800}
    # category 0 -> remapped 0
    out = Path("/tmp/test_labels")
    out.mkdir(parents=True, exist_ok=True)
    write_yolo_txt(im, [ann], {0: 0}, {0: "cat"}, out)
    lines = (out / "batch_1" / "000001.txt").read_text().strip().splitlines()
    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "0"
    cx, cy, bw, bh = map(float, parts[1:])
    assert 0 < cx < 1 and 0 < cy < 1


def test_out_of_bounds_filtered():
    im = {"file_name": "batch_1/000002.jpg", "width": 640, "height": 480}
    # bbox 完全越界 (会被过滤)
    ann = {"category_id": 0, "bbox": [1000, 1000, 100, 100], "area": 10000}
    out = Path("/tmp/test_labels2")
    out.mkdir(parents=True, exist_ok=True)
    write_yolo_txt(im, [ann], {0: 0}, {0: "cat"}, out)
    lines = (out / "batch_1" / "000002.txt").read_text().strip()
    assert lines == ""


def test_remap_consistency():
    # 原始 category 5 映射到 new 2
    im = {"file_name": "a.jpg", "width": 100, "height": 100}
    ann = {"category_id": 5, "bbox": [10, 10, 20, 20], "area": 400}
    out = Path("/tmp/test_labels3")
    out.mkdir(parents=True, exist_ok=True)
    write_yolo_txt(im, [ann], {5: 2}, {2: "new_cls"}, out)
    line = (out / "a.txt").read_text().strip()
    assert line.startswith("2 ")
