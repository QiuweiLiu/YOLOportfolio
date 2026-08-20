#!/usr/bin/env python
"""数据集准备脚本:COCO JSON → YOLO 格式,支持分层抽样与固定切分。

TACO 等 COCO 格式数据集 → Ultralytics YOLO 训练目录:

    data/processed/<name>/
    ├── images/{train,val,test}/
    ├── labels/{train,val,test}/*.txt
    ├── dataset.yaml
    └── metadata.json        # 数据版本信息:seed/子集/每类实例数/切分记录

用法:
    python yolo_optimization/scripts/prepare_dataset.py \
        --annotations data/raw/taco/annotations.json \
        --images-dir data/raw/taco/images \
        --output-dir data/processed \
        --name taco \
        --subset 800 --seed 42

约束:
- 抽样按类别分布层次化,保证少数类在子集中可复现保留
- 切分 train/val/test 比例记录到 metadata.json,任何实验必须引用该版本
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("prepare_dataset")


def load_coco(annotations_path: Path) -> dict:
    """读取 COCO JSON。"""
    if not annotations_path.exists():
        raise FileNotFoundError(f"annotations not found: {annotations_path}")
    with open(annotations_path, encoding="utf-8") as f:
        coco = json.load(f)
    log.info("COCO loaded: %d images, %d annotations, %d categories",
             len(coco["images"]), len(coco["annotations"]), len(coco["categories"]))
    return coco


def build_index(coco: dict) -> dict:
    """image_id → annotations;category_id → name。"""
    anns_by_img: dict[int, list] = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)
    cat_names = {c["id"]: c["name"] for c in coco["categories"]}
    return {"anns_by_img": dict(anns_by_img), "cat_names": cat_names}


def filter_by_min_instances(coco: dict, idx: dict, min_n: int) -> tuple[list[dict], dict, dict, dict, int]:
    """按每类实例数下限过滤类别;返回 (保留图, 过滤后 anns_by_img, old->new 类号重映射, 新类名->名, 丢弃图数)。"""
    counts: Counter = Counter()
    for anns in idx["anns_by_img"].values():
        counts.update(a["category_id"] for a in anns)
    keep = sorted(cid for cid, n in counts.items() if n >= min_n)
    remap = {cid: new for new, cid in enumerate(keep)}
    new_names = {remap[cid]: idx["cat_names"][cid] for cid in keep}
    anns_by_img_new: dict[int, list] = {}
    kept_imgs, dropped = [], 0
    for im in coco["images"]:
        anns = [a for a in idx["anns_by_img"].get(im["id"], []) if a["category_id"] in remap]
        if anns:
            kept_imgs.append(im)
            anns_by_img_new[im["id"]] = anns
        else:
            dropped += 1
    log.info("min_instances=%d: keep %d classes, %d images (drop %d unlabeled)",
             min_n, len(remap), len(kept_imgs), dropped)
    return kept_imgs, anns_by_img_new, remap, new_names, dropped


def instance_signature(anns: list[dict]) -> str:
    """图片的类别分布指纹,用于分层抽样(保证少数类图片入选)。"""
    counts = Counter(ann["category_id"] for ann in anns)
    return json.dumps(sorted(counts.items()), sort_keys=True)


def select_subset(images: list[dict], anns_by_img: dict, n: int | None,
                  seed: int) -> list[dict]:
    """按 stratum(类别分布指纹)分层抽样 n 张;n 为空或 >= 总数时返回全部。"""
    if n is None or n >= len(images):
        return images
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = defaultdict(list)
    for im in images:
        buckets[instance_signature(anns_by_img.get(im["id"], []))].append(im)
    selected: list[dict] = []
    per_bucket = max(1, int(n / len(buckets)))
    for items in buckets.values():
        rng.shuffle(items)
        selected.extend(items[:per_bucket])
    rng.shuffle(selected)
    if len(selected) > n:
        selected = selected[:n]
    elif len(selected) < n:
        rest = [im for im in images if im not in selected]
        rng.shuffle(rest)
        selected.extend(rest[: n - len(selected)])
    log.info("subset: %d / %d images (n=%s, seed=%d)", len(selected), len(images), n, seed)
    return selected


def copy_with_retry(src: Path, dst: Path, retries: int = 5, wait: float = 30.0) -> None:
    """图片复制带重试(本地 FileProvider/云同步偶发换出时 read 会阻塞或 TimeoutError)。"""
    for attempt in range(1, retries + 1):
        try:
            shutil.copy2(src, dst)
            return
        except OSError as e:
            if attempt == retries:
                raise
            log.warning("copy failed (%s), retry %d/%d after %.0fs: %s",
                        type(e).__name__, attempt, retries, wait, src.name)
            time.sleep(wait)


def write_yolo_txt(im: dict, anns: list, remap: dict, cat_id_to_name: dict,
                   labels_dir: Path) -> list[str]:
    """单图标注 → YOLO 归一化 txt(category_id 经 remap 重映射);返回该图出现的类别名。"""
    w, h = float(im["width"]), float(im["height"])
    lines: list[str] = []
    classes: list[str] = []
    for ann in anns:
        if ann.get("area", 0) <= 0 or "bbox" not in ann:
            continue
        x, y, bw, bh = (float(v) for v in ann["bbox"])
        cx, cy, bw_n, bh_n = (x + bw / 2) / w, (y + bh / 2) / h, bw / w, bh / h
        if not all(0 < v <= 1 for v in (cx, cy, bw_n, bh_n)):  # 越界框丢弃
            continue
        lines.append(f"{remap[ann['category_id']]} {cx:.6f} {cy:.6f} {bw_n:.6f} {bh_n:.6f}")
        classes.append(cat_id_to_name[remap[ann["category_id"]]])
    txt = labels_dir / Path(im["file_name"]).with_suffix(".txt")
    txt.parent.mkdir(parents=True, exist_ok=True)
    txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return classes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--annotations", required=True, type=Path, help="COCO annotations.json")
    ap.add_argument("--images-dir", required=True, type=Path, help="图片目录(相对 file_name 的根)")
    ap.add_argument("--output-dir", required=True, type=Path, help="输出根,如 data/processed")
    ap.add_argument("--name", required=True, help="数据集名(子目录与 dataset.yaml 用)")
    ap.add_argument("--subset", type=int, default=None, help="抽样图片数(默认全量)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--splits", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                    help="train/val/test 比例(和为 1)")
    ap.add_argument("--min-instances", type=int, default=None,
                    help="类别实例数下限(过滤低频类,缺省不过滤)")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="frozen task manifest JSON(使用固定切分,忽略 --subset/--splits/--min-instances)")
    ap.add_argument("--manifest-out", type=Path, default=None,
                    help="保存切分 manifest 到 JSON(用于后续实验复现同一 split)")
    args = ap.parse_args()

    if any(not 0 <= s <= 1 for s in args.splits):
        log.error("split fractions must be in [0,1]")
        return 1
    if abs(sum(args.splits) - 1.0) > 1e-9:
        log.error("splits must sum to 1.0, got %s", args.splits)
        return 1

    coco = load_coco(args.annotations)
    idx = build_index(coco)

    # --- 读取 manifest 或生成新切分 ---
    if args.manifest:
        log.info("loading manifest: %s", args.manifest)
        with open(args.manifest) as f:
            man = json.load(f)
        remap = {int(k): int(v) for k, v in man["class_mapping"].items()}
        idx["cat_names"] = {int(k): v for k, v in man["class_names"].items()}
        # 按 manifest 中的 ID 过滤图片
        id_to_im = {im["id"]: im for im in coco["images"]}
        split_map = []
        for split_name in ("train", "val", "test"):
            imgs = [id_to_im[iid] for iid in man["splits"][split_name] if iid in id_to_im]
            # 过滤掉不在映射中的标注
            anns = {im["id"]: [a for a in idx["anns_by_img"].get(im["id"], [])
                              if a["category_id"] in remap] for im in imgs}
            split_map.append((split_name, imgs))
        log.info("manifest loaded: %d classes, splits %s",
                 len(remap), {name: len(imgs) for name, imgs in split_map})
    else:
        remap = None
        if args.min_instances:
            coco_images, anns_by_img, remap, cat_names, dropped = \
                filter_by_min_instances(coco, idx, args.min_instances)
            idx = {"anns_by_img": anns_by_img, "cat_names": cat_names}
            images = select_subset(coco_images, idx["anns_by_img"], args.subset, args.seed)
        else:
            images = select_subset(coco["images"], idx["anns_by_img"], args.subset, args.seed)

        rng = random.Random(args.seed)
        rng.shuffle(images)
        n_tr = int(len(images) * args.splits[0])
        n_va = int(len(images) * args.splits[1])
        split_map = [("train", images[:n_tr]),
                     ("val", images[n_tr:n_tr + n_va]),
                     ("test", images[n_tr + n_va:])]

        # 保存 manifest 如果指定
        if args.manifest_out:
            man = {
                "task": args.name,
                "seed": args.seed,
                "min_instances": args.min_instances,
                "n_classes": len(idx["cat_names"]),
                "class_mapping": {str(k): str(v) for k, v in (remap or {c: c for c in idx["cat_names"]}).items()},
                "class_names": {str(k): v for k, v in idx["cat_names"].items()},
                "splits": {name: [im["id"] for im in imgs] for name, imgs in split_map},
                "n_images_per_split": {name: len(imgs) for name, imgs in split_map},
            }
            args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
            with open(args.manifest_out, "w") as f:
                json.dump(man, f, indent=2, ensure_ascii=False)
            log.info("manifest saved -> %s", args.manifest_out)

    out_root = args.output_dir / args.name
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    (out_root / "labels").mkdir(parents=True, exist_ok=True)

    class_counter: Counter = Counter()
    copies = missing = 0
    for split_name, imgs in split_map:
        img_out = out_root / "images" / split_name
        lbl_out = out_root / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for im in imgs:
            src = args.images_dir / im["file_name"]
            if not src.exists():
                missing += 1
                log.warning("missing image: %s", src)
                continue
            dst = img_out / im["file_name"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            copy_with_retry(src, dst)
            copies += 1
            classes = write_yolo_txt(im, idx["anns_by_img"].get(im["id"], []),
                                     remap if remap else {c: c for c in idx["cat_names"]},
                                     idx["cat_names"], lbl_out)
            class_counter.update(classes)

    if copies == 0:
        log.error("no images copied — check --images-dir layout")
        return 1
    if missing:
        log.warning("%d images missing from source dir", missing)

    cat_ids = sorted(idx["cat_names"])
    names = {i: idx["cat_names"][cid] for i, cid in enumerate(cat_ids)}
    # 使用 repo-relative path 而非绝对路径
    try:
        rel_path = out_root.relative_to(Path.cwd())
    except ValueError:
        rel_path = out_root.resolve()
    (out_root / "dataset.yaml").write_text(yaml.safe_dump({
        "path": str(rel_path),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }, sort_keys=False, allow_unicode=True), encoding="utf-8")

    meta = {
        "name": args.name,
        "seed": args.seed,
        "subset_requested": args.subset,
        "subset_actual": len(images),
        "min_instances": args.min_instances,
        "splits": {k: len(v) for k, v in split_map},
        "n_categories": len(cat_ids),
        "class_names": names,
        "per_class_instances": dict(sorted(class_counter.items())),
        "n_images_copied": copies,
        "n_images_missing": missing,
    }
    (out_root / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("done → %s", out_root)
    log.info("per-class instances: %s", dict(sorted(class_counter.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())