#!/usr/bin/env python
"""评估脚本:用 best.pt 在 frozen val/test 上计算最终指标。

输出:
  <run_dir>/evaluation.json
    - precision / recall / mAP50 / mAP50-95
    - per-class metrics
    - model path
    - dataset version
    - split

用法:
    python yolo_optimization/scripts/evaluate.py \
        --weights yolo_optimization/results/experiments/baseline_v1/weights/best.pt \
        --data yolo_optimization/data_manifests/task_v1.json \
        --split val
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("evaluate")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--weights", required=True, type=Path, help="best.pt 路径")
    ap.add_argument("--data", required=True, type=Path, help="dataset.yaml 或任务 manifest JSON 路径")
    ap.add_argument("--split", default="val", choices=["val", "test"], help="评估集")
    ap.add_argument("--imgsz", type=int, default=416)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="", help="cuda:0 / mps / cpu; 缺省自动选择")
    ap.add_argument("--output", type=Path, default=None, help="evaluation.json 输出路径(缺省: weights 所在目录)")
    args = ap.parse_args()

    if not args.weights.exists():
        log.error("weights not found: %s", args.weights)
        return 1

    # 确定 data 来源: manifest JSON 或 dataset.yaml
    data_arg = str(args.data)
    if args.data.suffix == ".json":
        # 从 manifest 生成 dataset.yaml
        man = json.loads(args.data.read_text())
        yaml_path = args.data.with_suffix(".yaml")
        import yaml
        yaml.dump({
            "path": str(yaml_path.parent.resolve()),
            args.split: f"data/processed/{man['task']}/images/{args.split}",
            "names": {int(k): v for k, v in man["class_names"].items()},
        }, yaml_path.open("w"), sort_keys=False, allow_unicode=True)
        data_arg = str(yaml_path)
        log.info("generated dataset.yaml from manifest: %s", yaml_path)

    device = args.device
    if not device:
        import torch
        if torch.cuda.is_available():
            device = "cuda:0"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    log.info("device=%s | weights=%s | data=%s | split=%s | imgsz=%d",
             device, args.weights, data_arg, args.split, args.imgsz)

    model = YOLO(str(args.weights))
    results = model.val(
        data=data_arg,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        plots=True,
        save_json=True,
        save_txt=False,
        save_conf=False,
    )

    run_dir = results.save_dir
    log.info("val done -> %s", run_dir)

    # 从 results 提取指标
    eval_data = {
        "model": str(args.weights),
        "dataset": str(args.data),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "mAP50": float(results.box.map50),
        "mAP50-95": float(results.box.map),
        "per_class": {},
        "confusion_matrix": str(run_dir / "confusion_matrix.png"),
    }

    # per-class metrics
    try:
        for i, name in results.names.items():
            cls_data = results.box.class_result(i)
            if cls_data is not None:
                eval_data["per_class"][name] = {
                    "precision": float(cls_data[0]),
                    "recall": float(cls_data[1]),
                    "mAP50": float(cls_data[2]),
                    "mAP50-95": float(cls_data[3]),
                }
    except (AttributeError, IndexError, TypeError):
        log.warning("per-class metrics not available from this Ultralytics version")

    output = args.output or run_dir
    output.mkdir(parents=True, exist_ok=True)
    ev_path = output / "evaluation.json"
    with ev_path.open("w") as f:
        json.dump(eval_data, f, indent=2, ensure_ascii=False, default=str)
    log.info("evaluation written -> %s", ev_path)
    log.info("mAP50=%.4f  mAP50-95=%.4f  P=%.4f  R=%.4f",
             eval_data["mAP50"], eval_data["mAP50-95"],
             eval_data["precision"], eval_data["recall"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())