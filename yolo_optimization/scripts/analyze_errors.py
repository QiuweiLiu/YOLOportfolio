#!/usr/bin/env python
"""FP/FN 错误分析脚本。

对 frozen val/test 集运行模型推理,用 IoU matching 分析:
- 每类 TP/FP/FN
- 目标尺寸分析(small/medium/large)
- 高置信度 FP / 低置信度 FN
- 生成 FP/FN 案例图

用法:
    python yolo_optimization/scripts/analyze_errors.py \
        --weights yolo_optimization/results/experiments/baseline_v1/weights/best.pt \
        --data yolo_optimization/data_manifests/task_v1.json \
        --output yolo_optimization/results/error_analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.ops import scale_boxes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("analyze_errors")

COLORS = {
    "TP": (0, 255, 0),    # green (BGR)
    "FP": (0, 0, 255),    # red (BGR) - OpenCV uses BGR
    "FN": (255, 0, 0),    # blue (BGR)
}


def compute_iou(box1, box2):
    """IoU between two boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_predictions(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, iou_thresh=0.5):
    """Greedy class-aware matching between GT and predictions.

    Rules:
      - Only same class + IoU >= thresh + GT not yet matched => TP
      - Otherwise prediction => FP (includes wrong-class high IoU)
      - Unmatched GT => FN
      - Duplicate predictions on same GT: first (highest conf) TP, rest FP

    Returns:
        tp_list: list of (original_pred_idx, gt_idx, iou)
        fp_list: list of original unmatched pred indices
        fn_list: list of unmatched gt indices
    """
    matched_gt = set()
    tp_list, fp_list = [], []

    # Sort by confidence but keep original indices
    order = np.argsort(-np.asarray(pred_scores)) if pred_scores else np.array([], dtype=int)

    for original_idx in order:
        p_box = pred_boxes[original_idx]
        p_cls = pred_classes[original_idx]
        best_iou, best_gt = 0.0, -1
        for g_idx, (g_box, g_cls) in enumerate(zip(gt_boxes, gt_classes)):
            if g_idx in matched_gt:
                continue
            if g_cls != p_cls:
                continue  # class-aware: only same class can be TP
            iou = compute_iou(p_box, g_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = g_idx

        if best_iou >= iou_thresh and best_gt >= 0:
            matched_gt.add(best_gt)
            tp_list.append((int(original_idx), best_gt, best_iou))
        else:
            fp_list.append(int(original_idx))

    fn_list = [g_idx for g_idx in range(len(gt_boxes)) if g_idx not in matched_gt]
    return tp_list, fp_list, fn_list


def get_size_category(bbox_area, img_area):
    """Classify object size by bbox_area / image_area ratio."""
    ratio = bbox_area / img_area if img_area > 0 else 0
    if ratio < 0.01:
        return "tiny (<1%)"
    elif ratio < 0.05:
        return "small (1-5%)"
    else:
        return "large (>5%)"


def draw_boxes(img, gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores,
               tp_list, fp_list, fn_list, class_names):
    """Draw GT/TP/FP/FN boxes on image for visualization."""
    h, w = img.shape[:2]

    # Draw FN (ground truth not matched)
    for fn_idx in fn_list:
        x1, y1, x2, y2 = map(int, gt_boxes[fn_idx])
        label = f"FN:{class_names.get(gt_classes[fn_idx], str(gt_classes[fn_idx]))}"
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS["FN"], 2)
        cv2.putText(img, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["FN"], 1)

    # Draw TP and FP
    for tp in tp_list:
        p_idx, g_idx, iou = tp
        if p_idx >= len(pred_boxes):
            continue
        x1, y1, x2, y2 = map(int, pred_boxes[p_idx])
        label = f"TP:{class_names.get(pred_classes[p_idx], str(pred_classes[p_idx]))}:{pred_scores[p_idx]:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS["TP"], 2)
        cv2.putText(img, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["TP"], 1)

    for fp_idx in fp_list:
        if fp_idx >= len(pred_boxes):
            continue
        x1, y1, x2, y2 = map(int, pred_boxes[fp_idx])
        label = f"FP:{class_names.get(pred_classes[fp_idx], str(pred_classes[fp_idx]))}:{pred_scores[fp_idx]:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), COLORS["FP"], 2)
        cv2.putText(img, label, (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS["FP"], 1)

    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True, type=Path, help="best.pt")
    ap.add_argument("--data", required=True, type=Path, help="dataset.yaml 或 manifest JSON")
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--conf", type=float, default=0.001, help="检测置信度阈值(默认 0.001 尽量不漏)")
    ap.add_argument("--iou", type=float, default=0.5, help="matching IoU 阈值")
    ap.add_argument("--output", type=Path, default=Path("yolo_optimization/results/error_analysis"),
                    help="输出目录")
    ap.add_argument("--max-examples", type=int, default=5, help="每类 FP/FN 最大案例图数")
    args = ap.parse_args()

    if not args.weights.exists():
        log.error("weights not found: %s", args.weights)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)

    # 加载模型
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(args.weights))
    class_names = model.names

    # 加载数据集 — 从 manifest 获取图片路径 (优先 data_name)
    if args.data.suffix == ".json":
        man = json.loads(args.data.read_text())
        task_name = man["task"]
        data_name = man.get("data_name", man.get("task", "task_v1"))
        data_dir = man.get("data_processed", f"data/processed/{data_name}")
        image_dir = Path(f"{data_dir}/images/{args.split}")
        image_paths = sorted(image_dir.rglob("*.[jJ][pP][gG]")) + sorted(image_dir.rglob("*.[pP][nN][gG]"))
        log.info("loaded %d images from %s", len(image_paths), image_dir)
    else:
        # 从 dataset.yaml 推断
        import yaml
        with open(args.data) as f:
            cfg = yaml.safe_load(f)
        image_dir = Path(cfg.get(args.split, cfg.get("val", "")))
        image_paths = sorted(image_dir.rglob("*.[jJ][pP][gG]")) + sorted(image_dir.rglob("*.[pP][nN][gG]"))
        task_name = args.data.stem

    if not image_paths:
        log.error("no images found in %s", image_dir)
        return 1

    log.info("running inference on %d images...", len(image_paths))
    results = model.predict(
        list(image_paths),
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.5,
        device=device,
        save=False,
        save_txt=False,
        max_det=300,
        verbose=False,
    )

    # 统计
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0})
    size_stats = Counter()
    high_conf_fps = []
    missed_small = []
    example_images = {"fp": [], "fn": []}

    for r in results:
        img_path = Path(r.path)
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # 获取 GT — 从 labels 目录加载
        gt_boxes, gt_classes = [], []
        lbl_path = str(img_path).replace("/images/", "/labels/").replace(".jpg", ".txt").replace(".JPG", ".txt")
        # 也可能是 .png 图片
        if not Path(lbl_path).exists():
            lbl_path = lbl_path.replace(".png", ".txt")
        if Path(lbl_path).exists():
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls = int(parts[0])
                        cx, cy, bw, bh = map(float, parts[1:5])
                        x1 = (cx - bw / 2) * img_w
                        y1 = (cy - bh / 2) * img_h
                        x2 = (cx + bw / 2) * img_w
                        y2 = (cy + bh / 2) * img_h
                        gt_boxes.append([x1, y1, x2, y2])
                        gt_classes.append(cls)

        # 获取 predictions
        pred_boxes, pred_classes, pred_scores = [], [], []
        if r.boxes is not None:
            for box in r.boxes:
                xyxy = box.xyxy[0].tolist()
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                pred_boxes.append(xyxy)
                pred_classes.append(cls)
                pred_scores.append(conf)

        # Matching
        tp_list, fp_list, fn_list = match_predictions(
            gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, args.iou)

        # 统计 per-class
        for g_idx in range(len(gt_classes)):
            per_class[gt_classes[g_idx]]["support"] += 1
        for tp in tp_list:
            p_idx, g_idx, iou = tp
            per_class[gt_classes[g_idx]]["tp"] += 1
        for fp_idx in fp_list:
            per_class[pred_classes[fp_idx]]["fp"] += 1
        for fn_idx in fn_list:
            per_class[gt_classes[fn_idx]]["fn"] += 1

        # 目标尺寸分析
        for g_idx, g_box in enumerate(gt_boxes):
            bbox_area = (g_box[2] - g_box[0]) * (g_box[3] - g_box[1])
            img_area = img_h * img_w
            cat = get_size_category(bbox_area, img_area)
            size_stats[cat] += 1
            if g_idx in fn_list:
                missed_small.append((cat, g_idx, img_path.name))

        # 收集 high-confidence FP
        for fp_idx in fp_list:
            if pred_scores[fp_idx] > 0.5:
                high_conf_fps.append({
                    "image": img_path.name,
                    "class": class_names.get(pred_classes[fp_idx], str(pred_classes[fp_idx])),
                    "confidence": pred_scores[fp_idx],
                })

        # 生成案例图
        if len(example_images["fp"]) < args.max_examples and fp_list:
            vis = draw_boxes(img.copy(), gt_boxes, gt_classes,
                             pred_boxes, pred_classes, pred_scores,
                             tp_list, fp_list, fn_list, class_names)
            out_path = args.output / f"fp_example_{len(example_images['fp']):02d}.jpg"
            cv2.imwrite(str(out_path), vis)
            example_images["fp"].append(str(out_path.name))

        if len(example_images["fn"]) < args.max_examples and fn_list:
            vis = draw_boxes(img.copy(), gt_boxes, gt_classes,
                             pred_boxes, pred_classes, pred_scores,
                             tp_list, fp_list, fn_list, class_names)
            out_path = args.output / f"fn_example_{len(example_images['fn']):02d}.jpg"
            cv2.imwrite(str(out_path), vis)
            example_images["fn"].append(str(out_path.name))

    # 整理输出
    n_total = sum(v["support"] for v in per_class.values())
    n_tp = sum(v["tp"] for v in per_class.values())
    n_fp = sum(v["fp"] for v in per_class.values())
    n_fn = sum(v["fn"] for v in per_class.values())

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(Path.cwd()))
        except ValueError:
            return str(p)
    report = {
        "model": _rel(Path(args.weights)),
        "dataset": _rel(Path(args.data)),
        "split": args.split,
        "imgsz": args.imgsz,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "n_images": len(results),
        "n_ground_truth": n_total,
        "n_true_positives": n_tp,
        "n_false_positives": n_fp,
        "n_false_negatives": n_fn,
        "per_class": {},
        "object_size_distribution": dict(size_stats.most_common()),
        "high_confidence_false_positives": high_conf_fps[:20],
        "example_images": example_images,
    }

    for cls_id, stats in per_class.items():
        name = class_names.get(cls_id, str(cls_id))
        p = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0.0
        r = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0.0
        report["per_class"][name] = {
            "support": stats["support"],
            "tp": stats["tp"],
            "fp": stats["fp"],
            "fn": stats["fn"],
            "precision": round(p, 4),
            "recall": round(r, 4),
        }

    report_path = args.output / "error_analysis.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    log.info("error analysis -> %s", report_path)
    log.info("TP=%d  FP=%d  FN=%d  total GT=%d", n_tp, n_fp, n_fn, n_total)
    log.info("example images: %s", example_images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())