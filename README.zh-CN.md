# YOLO 模型优化 Portfolio

[English](README.md) | 简体中文

一套面向真实客户项目的 YOLO 模型优化工作流：数据审计 → 固化 baseline → FP/FN 诊断 → 假设驱动实验 → 固定评估集验证 → 最终交付。

## 能力展示

- **数据审计**：分析类别分布/目标尺寸/任务范围合理性
- **固化评估集**：所有实验使用完全相同的 train/val/test
- **错误分析**：逐类 FP/FN + 目标尺寸分析 + 可视化案例
- **受控实验**：每次只改一个变量，固定评估集验证
- **可复现**：所有配置/seed/数据版本记录完整

## 案例背景：TACO 垃圾检测

[TACO 数据集](http://tacodataset.org/)（Trash Annotations in Context）包含 1,500 张自然环境垃圾照片，60 个细粒度类别，4,784 个标注。

**挑战**：60 个类别中 38 类样本不足 50 个——极端长尾使得 60 类全量检测不现实。[官方论文](https://arxiv.org/abs/2003.06975)使用 Mask R-CNN 在 1024×1024 分辨率上也仅得 17.6% AP。

### 任务范围研究

| 范围 | 类别数 | 筛选标准 | mAP50 | 解读 |
|---|---|---|---|---|
| 全量分类 | 60 | 全部 | 0.086 | 严重长尾 / 范围研究 |
| 过滤范围 | 23 | ≥50 实例 | 0.140 | 仅作范围研究 |
| 聚焦任务 | **8** | **≥200 实例** | **0.244** | 选定任务 |

> **重要：** 不同类别范围的 mAP 不作为直接的模型优化对比。以上实验展示的是数据集审计过程——识别出 60 类全量不适合实际检测，确定 8 个高频类作为最终业务任务。

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

*基于 best.pt 在固化验证集上的评估结果。*

### 优化成果

在相同任务（8类）+ 相同数据划分上的受控对比：

| 实验 | 假设 | 变更 | mAP50 | 提升 |
|---|---|---|---|---|
| Baseline | — | — | 0.238 | — |
| EXP-A | 提高分辨率改善小目标 | 640px | **0.315** | **+32%** |
| EXP-B | 关闭 mosaic | mosaic 0 | 0.270 | +13% |
| EXP-C | 类加权 | cls 1.0 | 0.263 | +10% |
| EXP-D | 延长训练 | 60 epochs | 0.270 | +13% |

### 快速开始

```bash
git clone https://github.com/QiuweiLiu/YOLOportfolio.git
cd YOLOportfolio
conda env create -f environment.yml
conda activate yolo-portfolio
python yolo_optimization/scripts/check_environment.py
```

后续：准备数据 / 训练 / 评估 / 分析 见 [英文版 README](README.md#quick-start)。

### 硬件

Apple Silicon (MPS) 开发，兼容 NVIDIA CUDA（自动选择 cuda → mps → cpu）。

### 许可证

- 代码：MIT
- TACO 数据集：CC BY 4.0