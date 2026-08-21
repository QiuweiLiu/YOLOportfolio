#!/usr/bin/env python
"""推理 CLI: 用 best.pt 对单图/目录/视频进行检测。

输出:
  - 标注图片/视频
  - detections.json (每帧检测结果)

用法:
    python yolo_optimization/scripts/inference.py \
        --weights yolo_optimization/results/experiments/baseline_v1/weights/best.pt \
        --source path/to/image.jpg \
        --output outputs/demo

    python yolo_optimization/scripts/inference.py \
        --weights path/to/best.pt \
        --source path/to/video.mp4 \
        --output outputs/demo
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("inference")


def process_image(model, img_path: Path, output: Path, conf: float, iou: float,
                  class_names: dict | None, save_img: bool) -> list[dict]:
    results = model.predict(str(img_path), conf=conf, iou=iou, save=False, save_txt=False)
    detections = []
    for r in results:
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                detections.append({
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "confidence": round(float(box.conf[0].item()), 4),
                    "class_id": int(box.cls[0].item()),
                    "class_name": class_names.get(int(box.cls[0].item()), "?") if class_names else "?",
                })
        if save_img:
            plot = r.plot()
            out_path = output / f"{img_path.stem}_pred.jpg"
            cv2.imwrite(str(out_path), plot)
            log.info("saved: %s", out_path)
    return detections


def process_video(model, video_path: Path, output: Path, conf: float, iou: float,
                  class_names: dict | None, max_frames: int = 0) -> list[dict]:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_video = None
    all_detections = []
    frame_idx = 0

    # Create writer upfront (not dependent on detections)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = output / f"{video_path.stem}_pred.mp4"
    out_video = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    if not out_video.isOpened():
        log.warning("failed to open video writer: %s", out_path)
        out_video = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if max_frames and frame_idx >= max_frames:
            break

        results = model.predict(frame, conf=conf, iou=iou, save=False, save_txt=False)
        r = results[0]
        # Single plot call per frame
        plot_img = r.plot()
        if out_video is not None:
            # Ensure plot size matches writer (Ultralytics may change size with imgsz)
            if plot_img.shape[1] != w or plot_img.shape[0] != h:
                plot_img = cv2.resize(plot_img, (w, h))
            out_video.write(plot_img)

        frame_dets = []
        if r.boxes is not None:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                frame_dets.append({
                    "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "confidence": round(float(box.conf[0].item()), 4),
                    "class_id": int(box.cls[0].item()),
                    "class_name": class_names.get(int(box.cls[0].item()), "?") if class_names else "?",
                })
        if frame_dets:
            all_detections.append({"frame": frame_idx, "detections": frame_dets})
        frame_idx += 1

    cap.release()
    if out_video is not None:
        out_video.release()
        log.info("video saved: %s", out_path)
    log.info("video done: %d frames, %d with detections", frame_idx, len(all_detections))
    return all_detections


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True, type=Path, help="best.pt")
    ap.add_argument("--source", required=True, type=Path, help="图片/目录/视频路径")
    ap.add_argument("--output", type=Path, default=Path("outputs/demo"), help="输出目录")
    ap.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU 阈值")
    ap.add_argument("--save-img", dest="save_img", action="store_true", help="保存标注图片")
    ap.add_argument("--no-save-img", dest="save_img", action="store_false", help="不保存标注图片")
    ap.set_defaults(save_img=True)
    ap.add_argument("--max-frames", type=int, default=0, help="视频最大帧数(0=全部)")
    args = ap.parse_args()

    if not args.weights.exists():
        log.error("weights not found: %s", args.weights)
        return 1
    if not args.source.exists():
        log.error("source not found: %s", args.source)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = YOLO(str(args.weights))
    class_names = model.names

    if args.source.is_dir():
        detections = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG"):
            for p in sorted(args.source.glob(ext)):
                dets = process_image(model, p, args.output, args.conf, args.iou,
                                     class_names, args.save_img)
                detections.append({"file": p.name, "detections": dets})
    elif args.source.suffix.lower() in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
        detections = process_video(model, args.source, args.output, args.conf, args.iou,
                                   class_names, args.max_frames)
    else:
        detections = process_image(model, args.source, args.output, args.conf, args.iou,
                                   class_names, args.save_img)

    det_path = args.output / "detections.json"
    with det_path.open("w") as f:
        json.dump(detections, f, indent=2, ensure_ascii=False)
    log.info("detections -> %s", det_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())