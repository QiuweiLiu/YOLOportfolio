# YOLO 模型优化 Portfolio

[English](README.md) | [简体中文](README.zh-CN.md)

一套面向真实客户项目的 YOLO 模型优化工作流：数据审计 → 固化 baseline → FP/FN 诊断 → 假设驱动实验 → 固定评估集验证 → 最终交付。

**核心能力：** 给定一个已有 YOLO 检测任务，系统地分析数据、建立固化 baseline、诊断错误、设计受控实验并验证优化是否真实有效。

### 演示

<div align="center">
  <img src="assets/hero_demo2.jpg" width="80%" alt="均衡案例">
  <p><em>均衡案例：绿 TP / 红 FP / 蓝 FN — 1 TP, 0 FP, 0 FN (val/000083.jpg, best.pt 640px)</em></p>
  <img src="assets/fp_example_00.jpg" width="45%" alt="FP 案例">
  <img src="assets/fn_example_00.jpg" width="45%" alt="FN 案例">
  <p><em>红：误检 | 蓝：漏检 | 绿：正确检出</em></p>
</div>

### 视频演示

<video src="assets/video_demo.mp4" controls width="100%"></video>

*5 秒演示：TACO 验证集 8 类垃圾检测（YOLOv8n, 640px）。输入：`outputs/demo_input.mp4` → 输出：`assets/video_demo.mp4` via `inference.py`。*

> **视频位置：** 仓库内 [`assets/video_demo.mp4`](assets/video_demo.mp4)（GitHub 直接播放），本地 `outputs/demo/demo_short_pred.mp4`。

### 案例背景：TACO 垃圾检测

[TACO 数据集](http://tacodataset.org/) 包含 1,500 张自然环境垃圾照片，60 个细粒度类别，4,784 个标注。

**挑战：** 60 个类别中 38 类样本不足 50 个——极端长尾使得 60 类全量检测不现实。[官方论文](https://arxiv.org/abs/2003.06975)使用 Mask R-CNN 在 1024×1024 分辨率上也仅得 17.6% AP。

### 任务范围研究

| 范围 | 类别数 | 筛选标准 | mAP50 | 解读 |
|---|---|---|---|---|
| 全量分类 | 60 | 全部 | 0.086 | 严重长尾 / 范围研究 |
| 过滤范围 | 23 | ≥50 实例 | 0.140 | 仅作范围研究 |
| 聚焦任务 | **8** | **≥200 实例** | **0.244** | 选定任务 |

> 不同类别范围的 mAP 不作为直接的模型优化对比。此研究说明为何 60 类全量不适合，并确定 8 个高频类作为最终业务任务。

最终任务（task_v1）聚焦 8 个常见垃圾类别：
塑料瓶、瓶盖、易拉罐、其他塑料、塑料薄膜、其他塑料包装、未标记垃圾、烟蒂

### 固化 Baseline

所有实验共享同一份 [数据清单](yolo_optimization/data_manifests/task_v1.json)（train/val/test = 640/80/80）。

| 指标 | 值 |
|---|---|
| 模型 | YOLOv8n (3.0M 参数) |
| 输入 | 416×416 |
| 训练 | 30 epochs, batch 8, seed 42 |
| **mAP50** | **0.238** |
| **mAP50-95** | **0.181** |
| **Precision** | **0.400** |
| **Recall** | **0.266** |

*基于 best.pt 在固化验证集上的评估结果。*

### 错误分析

基于阈值的匹配（IoU ≥ 0.5, conf ≥ 0.001）在固化验证集上 — 与 Ultralytics AP 验证指标不同。
诊断用 TP/FP/FN 统计用于失败分析，可能与 COCO 风格的 AP 评估不同。

**目标尺寸分布**（204 个验证实例）：

| 尺寸 | 数量 | 占比 |
|---|---|---|
| 极小 (<1% 图面积) | 154 | 75.5% |
| 小 (1-5%) | 28 | 13.7% |
| 大 (>5%) | 22 | 10.8% |

**关键发现：**
- **烟蒂**最难：53 个实例，5 TP / 48 FN（召回 0.09，精度 0.00），多数为极小目标
- **透明塑料瓶**最好：13 个实例，12 TP / 1 FN（召回 0.92，精度 0.04）
- 75.5% 目标为极小 — 小目标检测是主要瓶颈

### 优化成果

相同任务（8类）+ 相同数据划分上的受控对比：

| 实验 | 假设 | 变更 | mAP50 | 提升 |
|---|---|---|---|---|
| Baseline | — | — | 0.238 | — |
| **EXP-A** | **提高分辨率改善小目标** | **640px** | **0.315** | **+32%** |
| EXP-B | 关闭 mosaic | mosaic 0 | 0.270 | +13% |
| EXP-C | 类加权 | cls 1.0 | 0.263 | +10% |
| EXP-D | 延长训练 | 60 epochs | 0.270 | +13% |

### 交付物

面向客户，你将获得：

- 训练好的 YOLO 权重（`best.pt`）
- 可复现的训练配置（`baseline_v1.yaml`）
- 评估报告（`evaluation.json`）
- FP/FN 失败分析（`error_analysis.json` + 10 张案例图）
- 推理脚本（`inference.py` — 单图/目录/视频）
- 优化建议（基于错误分析）

### 脚本

| 脚本 | 用途 |
|---|---|
| [`train.py`](yolo_optimization/scripts/train.py) | 训练 YOLO，自动设备选择 |
| [`evaluate.py`](yolo_optimization/scripts/evaluate.py) | 评估 best.pt → evaluation.json |
| [`analyze_errors.py`](yolo_optimization/scripts/analyze_errors.py) | FP/FN 分析 |
| [`inference.py`](yolo_optimization/scripts/inference.py) | 推理 CLI |
| [`compare_runs.py`](yolo_optimization/scripts/compare_runs.py) | 多实验对比 |
| [`prepare_dataset.py`](yolo_optimization/scripts/prepare_dataset.py) | COCO→YOLO 转换 |
| [`analyze_dataset.py`](yolo_optimization/scripts/analyze_dataset.py) | 数据集统计 |
| [`check_environment.py`](yolo_optimization/scripts/check_environment.py) | 环境检查 |

### 快速开始

```bash
git clone https://github.com/QiuweiLiu/YOLOportfolio.git
cd YOLOportfolio

# 1. 创建环境 (Python 3.11)
conda env create -f environment.yml
conda activate yolo-portfolio

# 2. 验证环境
python yolo_optimization/scripts/check_environment.py

# 3. 下载 TACO 数据集
# (约 2.7GB, 见 TACO 官方仓库)

# 4. 准备数据（固化清单）
python yolo_optimization/scripts/prepare_dataset.py \
  --annotations data/raw/taco/annotations.json \
  --images-dir data/raw/taco/images \
  --output-dir data/processed \
  --name taco_v1 \
  --manifest yolo_optimization/data_manifests/task_v1.json

# 5. 训练
python yolo_optimization/scripts/train.py \
  --data data/processed/taco_v1/dataset.yaml \
  --model yolov8n.pt \
  --epochs 30 --imgsz 416 --batch 8 --seed 42

# 6. 评估
python yolo_optimization/scripts/evaluate.py \
  --weights yolo_optimization/results/experiments/exp/weights/best.pt \
  --data yolo_optimization/data_manifests/task_v1.json \
  --split val

# 7. 错误分析
python yolo_optimization/scripts/analyze_errors.py \
  --weights yolo_optimization/results/experiments/exp/weights/best.pt \
  --data yolo_optimization/data_manifests/task_v1.json \
  --split val

# 8. 推理
python yolo_optimization/scripts/inference.py \
  --weights yolo_optimization/results/experiments/exp/weights/best.pt \
  --source path/to/your/image.jpg
```

### 项目结构

```
YOLOportfolio/
├── yolo_optimization/
│   ├── scripts/          # 训练、评估、分析、推理
│   ├── configs/          # 实验配置
│   ├── data_manifests/   # 固化任务定义
│   ├── results/          # 实验输出（gitignored）
│   └── tests/            # 单元测试
├── assets/               # 展示用精选结果
├── requirements.txt      # 锁定依赖
└── LICENSE               # MIT
```

### 硬件

Apple Silicon (MPS) 开发，兼容 NVIDIA CUDA（自动选择 cuda → mps → cpu）。

### 许可证

- 代码：MIT
- TACO 数据集：CC BY 4.0
