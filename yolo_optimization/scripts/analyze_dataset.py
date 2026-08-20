#!/usr/bin/env python
"""数据集分析脚本(Phase 3):COCO JSON → 统计 + 图表。

覆盖需求文档 §7:
- 类别分布: 每类 image count / object count / percentage
- bbox 尺寸分布: small / medium / large(COCO 风格: area < 32²、32²~96²、> 96²)
- bbox aspect ratio 分布
- 每图目标数: mean / median / 分布直方图
- 类别不平衡: 少数类识别(实例数 < max(10, 5% of max))

输出(results/dataset_analysis/):
- stats.json     数值事实(唯一数字来源)
- *.png          图表(类别分布/面积/长宽比/每图目标数)

用法:
    python yolo_optimization/scripts/analyze_dataset.py \
        --annotations data/raw/taco/annotations.json \
        --output results/dataset_analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("analyze_dataset")

COCO_SMALL_AREA = 32 * 32    # < 1024 px²
COCO_LARGE_AREA = 96 * 96    # > 9216 px²


def collect(coco: dict) -> dict:
    """提取分析所需数据: 每图类别列表、bbox 面积、长宽比。"""
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    # 先构建 image_id -> annotations 映射,避免 O(n×m) 扫描
    anns_by_image: dict[int, list] = {}
    for a in coco["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)
    per_img: list[list[str]] = []
    areas: list[float] = []
    ratios: list[float] = []
    for im in coco["images"]:
        anns = anns_by_image.get(im["id"], [])
        per_img.append([cat_names[a["category_id"]] for a in anns])
        for a in anns:
            if a.get("area", 0) <= 0 or "bbox" not in a:
                continue
            areas.append(a["area"])
            w, h = a["bbox"][2], a["bbox"][3]
            if h > 0:
                ratios.append(w / h)
    return {"cat_names": cat_names, "per_img": per_img, "areas": areas, "ratios": ratios}


def compute_stats(coco: dict) -> tuple[dict, dict]:
    data = collect(coco)
    n_imgs = len(coco["images"])
    n_anns = len(coco["annotations"])
    log.info("images=%d annotations=%d", n_imgs, n_anns)

    cat_counts: Counter = Counter(a["category_id"] for a in coco["annotations"])
    cat_img: Counter = Counter()
    for objs in data["per_img"]:
        cat_img.update(set(objs))
    cat_names = data["cat_names"]

    class_stats = {}
    for cid, name in sorted(cat_names.items()):
        class_stats[name] = {
            "category_id": cid,
            "image_count": int(cat_img[name]),
            "object_count": int(cat_counts[cid]),
            "object_percentage": round(100 * cat_counts[cid] / max(1, n_anns), 2),
        }

    sizes = {"small": 0, "medium": 0, "large": 0}
    for a in data["areas"]:
        if a < COCO_SMALL_AREA:
            sizes["small"] += 1
        elif a > COCO_LARGE_AREA:
            sizes["large"] += 1
        else:
            sizes["medium"] += 1
    n_area = max(1, len(data["areas"]))
    size_pct = {k: round(100 * v / n_area, 2) for k, v in sizes.items()}

    obj_arr = np.array([len(o) for o in data["per_img"]])
    if obj_arr.size == 0:
        obj_arr = np.array([0])
    ratios_arr = np.array(data["ratios"]) if data["ratios"] else np.array([0.0])

    max_obj = max(cat_counts.values(), default=0)
    minority = sorted(
        cat_names[c] for c, cnt in cat_counts.items() if cnt < max(10, 0.05 * max_obj)
    )

    stats = {
        "n_images": n_imgs,
        "n_annotations": n_anns,
        "images_with_objects": int(np.count_nonzero(obj_arr)),
        "class_stats": class_stats,
        "objects_per_image": {
            "mean": round(float(obj_arr.mean()), 2),
            "median": round(float(np.median(obj_arr)), 2),
            "min": int(obj_arr.min()),
            "max": int(obj_arr.max()),
        },
        "bbox_size_counts": sizes,
        "bbox_size_percentage": size_pct,
        "aspect_ratio": {
            "mean": round(float(ratios_arr.mean()), 3),
            "median": round(float(np.median(ratios_arr)), 3),
        },
        "minority_classes": minority,
    }
    return stats, data


def plot_stats(stats: dict, data: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(stats["class_stats"].keys())
    obj = [stats["class_stats"][n]["object_count"] for n in names]

    # 1) 类别实例数
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(names, obj, color="#4C72B0")
    ax.set_xlabel("class")
    ax.set_ylabel("object count")
    ax.set_title("Per-class object count")
    plt.xticks(rotation=75, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "class_distribution.png", dpi=120)
    plt.close(fig)

    # 2) bbox 面积分布(log10)
    areas = np.clip(np.array(data["areas"]), 1, None)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.log10(areas), bins=40)
    ax.set_xlabel("log10(bbox area)")
    ax.set_title("Bounding box area distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "bbox_area.png", dpi=120)
    plt.close(fig)

    # 3) aspect ratio
    ratios = np.clip(np.array(data["ratios"]), 0, 5)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(ratios, bins=40)
    ax.set_xlabel("aspect ratio (w/h)")
    ax.set_title("bbox aspect ratio distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "aspect_ratio.png", dpi=120)
    plt.close(fig)

    # 4) 每图目标数
    per_img = [len(o) for o in data["per_img"]]
    n_bins = min(50, max(10, max(per_img) if per_img else 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(per_img, bins=n_bins)
    ax.set_xlabel("objects per image")
    ax.set_title("objects per image distribution")
    fig.tight_layout()
    fig.savefig(out_dir / "objects_per_image.png", dpi=120)
    plt.close(fig)

    log.info("plots written to %s", out_dir)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--annotations", required=True, type=Path, help="COCO annotations.json")
    ap.add_argument("--output", required=True, type=Path, help="输出目录(如 results/dataset_analysis)")
    args = ap.parse_args()

    if not args.annotations.exists():
        log.error("annotations not found: %s", args.annotations)
        return 1
    with open(args.annotations, encoding="utf-8") as f:
        coco = json.load(f)

    stats, data = compute_stats(coco)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    plot_stats(stats, data, args.output)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())