# YOLO Model Optimization Portfolio

[English](#english) | [简体中文](#简体中文)

---

## English

A practical object detection optimization workflow with dataset audit, frozen evaluation splits, FP/FN diagnosis, hypothesis-driven controlled experiments, and reproducible before/after evaluation.

**What this demonstrates:** Given an existing YOLO detection task, systematically identify problems through data analysis, establish a frozen baseline, diagnose False Positive / False Negative errors, design hypothesis-driven experiments, and validate improvements on a fixed evaluation set.

### Demo

<div align="center">
  <img src="assets/fp_example_00.jpg" width="45%" alt="FP Example">
  <img src="assets/fn_example_00.jpg" width="45%" alt="FN Example">
  <p><em>Red: False Positive | Blue: False Negative | Green: True Positive</em></p>
</div>

### Case Study: TACO Litter Detection

The [TACO dataset](http://tacodataset.org/) (Trash Annotations in Context) contains 1,500 images of litter in natural environments with 60 fine-grained categories and 4,784 annotations.

**Challenge:** 38 out of 60 classes have fewer than 50 instances each — extreme class imbalance makes full 60-class detection impractical. [The official paper](https://arxiv.org/abs/2003.06975) reports only 17.6% AP using Mask R-CNN at 1024×1024 resolution.

### Task Scope Study

The first step was auditing the dataset to determine the appropriate task scope:

| Experiment | Classes | Criteria | mAP50 | Notes |
|---|---|---|---|---|
| EXP-001 | 60 (full) | All classes | 0.086 | Extreme class imbalance |
| EXP-002 | 23 | ≥50 instances | 0.140 | +63% vs full |
| EXP-003 | **8** | **≥200 instances** | **0.244** | **Focused task** |

> **Important:** Metrics across different class scopes are NOT treated as direct model optimization comparisons. These experiments demonstrate the dataset audit process — identifying that the full 60-class taxonomy is not suitable for practical detection, and establishing a focused 8-class task as the final benchmark.

The final task (`task_v1`) focuses on the 8 most common litter classes:
Clear plastic bottle, Plastic bottle cap, Drink can, Other plastic, Plastic film, Other plastic wrapper, Unlabeled litter, Cigarette

### Frozen Baseline

The evaluation split is fixed via a [data manifest](yolo_optimization/data_manifests/task_v1.json) — all experiments use the exact same train/val/test images (640/80/80).

| Metric | Value |
|---|---|
| Model | YOLOv8n (3.0M params) |
| Input size | 416×416 |
| Training | 30 epochs, batch 8, seed 42 |
| **mAP50** | **0.238** |
| **mAP50-95** | **0.181** |
| **Precision** | **0.400** |
| **Recall** | **0.266** |

*Metrics from `evaluate.py` on frozen validation set (best.pt).*

### Error Analysis

**Object Size Distribution** (205 val instances):

| Size | Count | % |
|---|---|---|
| Tiny (<1% image area) | 155 | 75.6% |
| Small (1-5%) | 28 | 13.7% |
| Large (>5%) | 22 | 10.7% |

**Key findings:**
- **Cigarette** is the hardest class: 53 instances, only 5 detected (recall 0.11). Most are tiny (<1% of image area).
- **Clear plastic bottle** performs best: 14 instances, 10 detected (recall 1.0, precision 0.03 at low threshold).
- 75.6% of all objects are tiny — small object detection is the primary bottleneck.

### Deliverables

| Script | Purpose |
|---|---|
| [`train.py`](yolo_optimization/scripts/train.py) | Train YOLO with auto device selection, metrics logging |
| [`evaluate.py`](yolo_optimization/scripts/evaluate.py) | Evaluate best.pt on frozen val/test → standardized evaluation.json |
| [`analyze_errors.py`](yolo_optimization/scripts/analyze_errors.py) | FP/FN analysis with per-class stats, size analysis, visual examples |
| [`inference.py`](yolo_optimization/scripts/inference.py) | CLI inference on images/video → annotated output + detections.json |
| [`compare_runs.py`](yolo_optimization/scripts/compare_runs.py) | Compare multiple evaluation.json → comparison table |
| [`prepare_dataset.py`](yolo_optimization/scripts/prepare_dataset.py) | COCO→YOLO conversion with frozen manifest support |
| [`analyze_dataset.py`](yolo_optimization/scripts/analyze_dataset.py) | Dataset statistics (class distribution, bbox size, aspect ratio) |
| [`check_environment.py`](yolo_optimization/scripts/check_environment.py) | Verify environment (fresh clone compatible) |

### Quick Start

```bash
git clone https://github.com/QiuweiLiu/YOLOportfolio.git
cd YOLOportfolio

# 1. Create environment (Python 3.11)
conda env create -f environment.yml
conda activate yolo-portfolio

# 2. Verify environment
python yolo_optimization/scripts/check_environment.py

# 3. Download TACO dataset
# (requires ~2.7GB, see TACO official repo for download options)

# 4. Prepare data with frozen manifest
python yolo_optimization/scripts/prepare_dataset.py \
  --annotations data/raw/taco/annotations.json \
  --images-dir data/raw/taco/images \
  --output-dir data/processed \
  --name taco_v1 \
  --manifest yolo_optimization/data_manifests/task_v1.json

# 5. Train baseline
python yolo_optimization/scripts/train.py \
  --data data/processed/taco_v1/dataset.yaml \
  --model yolov8n.pt \
  --epochs 30 --imgsz 416 --batch 8 --seed 42

# 6. Evaluate
python yolo_optimization/scripts/evaluate.py \
  --weights yolo_optimization/results/experiments/exp/weights/best.pt \
  --data yolo_optimization/data_manifests/task_v1.json \
  --split val

# 7. Error analysis
python yolo_optimization/scripts/analyze_errors.py \
  --weights yolo_optimization/results/experiments/exp/weights/best.pt \
  --data yolo_optimization/data_manifests/task_v1.json \
  --split val

# 8. Run inference on your own images
python yolo_optimization/scripts/inference.py \
  --weights yolo_optimization/results/experiments/exp/weights/best.pt \
  --source path/to/your/image.jpg
```

### Project Structure

```
YOLOportfolio/
├── yolo_optimization/
│   ├── scripts/          # Training, evaluation, analysis, inference
│   ├── configs/          # Experiment configuration files
│   ├── data_manifests/   # Frozen task definitions (splits, class mapping)
│   ├── results/          # Experiment outputs (gitignored)
│   └── tests/            # Unit tests
├── assets/               # Selected results for README
├── requirements.txt      # Locked dependencies (Python 3.11)
├── environment.yml       # Conda environment
└── LICENSE               # MIT
```

### Hardware

Developed on **Apple Silicon (MPS)** — torch 2.13 / Ultralytics 8.4.121 / Python 3.11. Compatible with NVIDIA CUDA (auto-selects cuda → mps → cpu).

### License

- Code: MIT (see [LICENSE](LICENSE))
- TACO dataset: CC BY 4.0

---

## 简体中文

一套面向真实客户项目的 YOLO 模型优化工作流：数据审计 → 固化 baseline → FP/FN 诊断 → 假设驱动实验 → 固定评估集验证 → 最终交付。

### 能力展示

| 能力 | 说明 |
|---|---|
| 数据审计 | 分析类别分布/目标尺寸/任务范围 |
| 固化评估集 | 所有实验使用完全相同的 train/val/test |
| 错误分析 | 逐类 FP/FN + 目标尺寸分析 + 可视化案例 |
| 受控实验 | 每次只改一个变量，固定评估集验证 |
| 可复现 | 配置/seed/数据版本全部记录 |

### 案例：TACO 垃圾检测

从 60 类全量数据出发，通过审计发现 38 类样本不足 50 个，最终聚焦到 8 个高频类（mAP50 从 0.086 提升至 0.244）。[了解更多](yolo_optimization/data_manifests/task_v1.json)

### 快速开始

```bash
git clone https://github.com/QiuweiLiu/YOLOportfolio.git
cd YOLOportfolio
conda env create -f environment.yml
conda activate yolo-portfolio
python yolo_optimization/scripts/check_environment.py
```

详细步骤见英文版 [Quick Start](#quick-start)。