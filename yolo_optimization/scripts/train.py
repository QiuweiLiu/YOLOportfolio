#!/usr/bin/env python
"""YOLO 训练脚本(baseline 与对比实验共用)。

device 自动选择: cuda -> mps -> cpu(显式 --device 可覆盖)。
训练输出统一落入 <project>/<name>/ (git 不跟踪),best.pt 与训练曲线由
Ultralytics 生成;训练完成后把 best 指标沉淀为 metrics.json(数值事实源)。

用法:
    python yolo_optimization/scripts/train.py --data data/processed/taco/dataset.yaml \
        --model yolov8n.pt --epochs 30 --name baseline_v1 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")

DEFAULT_PROJECT = Path(__file__).resolve().parents[1] / "results" / "experiments"


def pick_device(requested: str) -> str:
    """cuda -> mps -> cpu 自动选择; 显式请求时原样返回。"""
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def write_metrics(dir_: Path, metrics: dict) -> None:
    with (dir_ / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)
    log.info("metrics written -> %s/metrics.json", dir_)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=Path("data/processed/taco/dataset.yaml"), help="dataset.yaml")
    ap.add_argument("--model", default="yolov8n.pt", help="权重/模型名(默认 yolov8n.pt,也可指定 .pt 路径)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1, help="<=0 时用 Ultralytics 自动 batch")
    ap.add_argument("--device", default="", help="cuda:0 / mps / cpu; 缺省自动选择")
    ap.add_argument("--project", type=Path, default=DEFAULT_PROJECT)
    ap.add_argument("--name", default="exp", help="实验名(输出子目录)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=30, help="早停 epoch 数")
    ap.add_argument("--fraction", type=float, default=1.0, help="每 epoch 用数据比例(调试用)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--plots", action="store_true", default=False, help="生成训练曲线图")
    ap.add_argument("--cache", default=False, help="'disk' 或 True 缓存数据集(默认不缓存)")
    ap.add_argument("--mosaic", type=float, default=1.0, help="mosaic 增强强度(0 关闭)")
    ap.add_argument("--close-mosaic", type=int, default=10, help="最后 N epoch 关闭 mosaic")
    ap.add_argument("--cls", type=float, default=0.5, help="分类 loss 权重")
    ap.add_argument("--resume", action="store_true", help="从 <project>/<name>/weights/last.pt 继续")
    ap.add_argument("--amp", type=int, default=1, help="AMP 开关(1 开,0 关;MPS 上如需可关)")
    args = ap.parse_args()

    if not args.data.exists():
        log.error("dataset.yaml not found: %s", args.data)
        return 1

    device = pick_device(args.device)
    if device == "mps" and args.amp != 1:
        log.info("MPS 训练关闭 AMP(规避精度/SVD 算子)")
    log.info("device=%s | model=%s | data=%s | epochs=%d | imgsz=%d | batch=%d | seed=%d",
             device, args.model, args.data, args.epochs, args.imgsz, args.batch, args.seed)

    model = YOLO(args.model)
    results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        seed=args.seed,
        patience=args.patience,
        fraction=args.fraction,
        workers=args.workers,
        lr0=args.lr0,
        cls=args.cls,
        project=str(args.project),
        name=args.name,
        exist_ok=True,
        plots=args.plots,
        cache=args.cache,
        mosaic=args.mosaic,
        close_mosaic=args.close_mosaic,
        resume=args.resume,
        amp=bool(args.amp),
        val=True,
    )

    if results is None:
        log.error("train finished without results (resume?) — metrics not written")
        return 1

    run_dir = results.save_dir
    log.info("train done -> %s", run_dir)
    metrics: dict = {"device": device, "args": vars(args) | {"device": device}}
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            last = rows[-1]
            for key in ("epoch", "metrics/precision(B)", "metrics/recall(B)",
                        "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
                if key in last:
                    metrics[f"last_{key}"] = last[key]
        else:
            log.warning("results.csv empty — metrics not extracted")
    else:
        log.warning("results.csv missing — metrics not extracted")

    # 使用 repo-relative 路径
    cwd = Path.cwd()
    try:
        rel_weights = run_dir.relative_to(cwd) / "weights" / "best.pt"
    except ValueError:
        rel_weights = run_dir / "weights" / "best.pt"
    metrics["best_weights"] = str(rel_weights)
    write_metrics(run_dir, metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())