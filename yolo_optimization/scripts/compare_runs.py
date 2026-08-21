#!/usr/bin/env python
"""实验对比工具:读取多个 evaluation.json 生成对比表。

用法:
    python yolo_optimization/scripts/compare_runs.py \
        --runs baseline=path/to/baseline/evaluation.json \
                   exp_a=path/to/exp_a/evaluation.json \
        --output yolo_optimization/results/comparison/comparison.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("compare_runs")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", nargs="+", required=True,
                    help="name=path 对, 如 baseline=path/eval.json exp_a=path/eval.json")
    ap.add_argument("--baseline", type=str, default=None,
                    help="baseline 实验名(用于计算提升, 如 baseline)")
    ap.add_argument("--output", type=Path, default=Path("yolo_optimization/results/comparison/comparison.json"),
                    help="输出路径")
    args = ap.parse_args()
    if args.baseline and args.baseline not in [r.split("=", 1)[0] for r in args.runs]:
        log.error("baseline '%s' not found in --runs", args.baseline)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)

    runs = {}
    for item in args.runs:
        if "=" not in item:
            log.error("invalid format (need name=path): %s", item)
            return 1
        name, path = item.split("=", 1)
        p = Path(path)
        if not p.exists():
            log.error("evaluation.json not found: %s", p)
            return 1
        with open(p) as f:
            runs[name] = json.load(f)

    if not runs:
        log.error("no valid runs provided")
        return 1

    # 主指标对比表
    summary = []
    for name, data in runs.items():
        summary.append({
            "experiment": name,
            "precision": data.get("precision"),
            "recall": data.get("recall"),
            "mAP50": data.get("mAP50"),
            "mAP50-95": data.get("mAP50-95"),
            "model": data.get("model"),
            "split": data.get("split"),
            "dataset": data.get("dataset"),
        })

    # 按 mAP50 排序
    valid = [s for s in summary if s["mAP50"] is not None]
    valid.sort(key=lambda x: x["mAP50"], reverse=True)

    # Per-class 对比
    per_class = {}
    for name, data in runs.items():
        if "per_class" in data:
            for cls_name, cls_stats in data["per_class"].items():
                if cls_name not in per_class:
                    per_class[cls_name] = {}
                per_class[cls_name][name] = cls_stats

    output = {
        "comparison_table": summary,
        "sorted_by_mAP50": valid,
        "per_class_comparison": per_class,
        "best_experiment": valid[0]["experiment"] if valid else None,
        "best_mAP50": valid[0]["mAP50"] if valid else None,
    }

    # 差值 — 必须显式 baseline
    if args.baseline:
        baseline_entry = next((s for s in summary if s["experiment"] == args.baseline), None)
        best_entry = valid[0] if valid else None
        if baseline_entry and best_entry:
            output["improvement"] = {
                "baseline": baseline_entry["experiment"],
                "best": best_entry["experiment"],
                "mAP50_delta": round(best_entry["mAP50"] - baseline_entry["mAP50"], 4) if best_entry["mAP50"] and baseline_entry["mAP50"] else None,
                "mAP50_change_pct": round((best_entry["mAP50"] - baseline_entry["mAP50"]) / baseline_entry["mAP50"] * 100, 1)
                if best_entry["mAP50"] and baseline_entry["mAP50"] and baseline_entry["mAP50"] > 0 else None,
            }
    elif len(valid) >= 2:
        # 向后兼容: 未指定 baseline 时仍计算但警告
        log.warning("no --baseline specified, improvement relative to worst run")
        baseline = valid[-1]
        best = valid[0]
        output["improvement"] = {
            "baseline": baseline["experiment"],
            "best": best["experiment"],
            "mAP50_delta": round(best["mAP50"] - baseline["mAP50"], 4) if best["mAP50"] and baseline["mAP50"] else None,
            "mAP50_change_pct": round((best["mAP50"] - baseline["mAP50"]) / baseline["mAP50"] * 100, 1)
            if best["mAP50"] and baseline["mAP50"] and baseline["mAP50"] > 0 else None,
        }

    with args.output.open("w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    log.info("comparison -> %s", args.output)

    print("\n=== Experiment Comparison ===")
    print(f"{'Experiment':<20} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8}")
    print("-" * 60)
    for s in valid:
        print(f"{s['experiment']:<20} {s['mAP50']:>8.4f} {s['mAP50-95']:>10.4f} "
              f"{s['precision']:>10.4f} {s['recall']:>8.4f}")
    if "improvement" in output and output["improvement"]["mAP50_change_pct"]:
        imp = output["improvement"]
        print(f"\nImprovement: {imp['mAP50_change_pct']:.1f}% ({imp['baseline']} → {imp['best']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())