你现在需要在我的 Mac 上从零创建一个用于远程兼职/自由职业接单展示的 Computer Vision Portfolio 项目。

项目目标不是做论文创新，也不是堆复杂框架，而是模拟真实客户常见的 YOLO 模型优化任务：

已有目标检测 baseline  
→ 数据集分析  
→ baseline 训练与评估  
→ False Positive / False Negative 分析  
→ 目标尺寸和类别问题分析  
→ 少量 controlled optimization experiments  
→ before / after 对比  
→ 最终模型与报告交付

这个项目未来会公开放在 GitHub，并用于 Upwork、Freelancer、电鸭、程序员客栈等平台投递 YOLO / Computer Vision / Video Analytics 类短期项目。

因此，请优先考虑：

- 工程规范
- 可复现性
- 真实实验
- 客户可读性
- 简单可靠
- Mac Apple Silicon 兼容
- 后续可以迁移到 NVIDIA CUDA 环境

不要为了显得高级而引入不必要的复杂设计。

---

# 1. 本机环境约束

开发机器：

- Apple Silicon Mac
- 优先使用 Python 3.11
- PyTorch
- Ultralytics YOLO
- OpenCV
- Apple MPS GPU

训练设备优先自动判断：

1. 如果 MPS 可用，使用 `mps`
2. 如果未来运行在 CUDA 环境，支持 `cuda`
3. 否则退回 `cpu`

不要把代码写死为 CUDA。

实现统一的 device detection，例如：

```python
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
```

同时注意 Apple MPS 可能存在算子兼容性问题。

如果某些操作 MPS 不支持，应：
- 优先寻找兼容写法
- 必要时允许对应步骤回退 CPU
- 在 README 中如实记录
- 不要偷偷忽略错误

---

# 2. 数据集选择

请先选择一个：

- 公开
- License 清晰
- 体量较小
- 目标检测任务
- 适合 YOLO
- 最好包含一定的小目标、类别不平衡或有一定优化空间
- Mac 上可以较快训练

的数据集。

建议规模：

大约 500–3000 张图片。

不要一上来用完整 COCO。

如果候选数据集有多个，请先比较：

- 数据规模
- 类别数
- License
- 下载方式
- 是否已有 YOLO 格式
- 是否适合快速实验
- 是否具有真实的 error analysis 空间

然后选择一个最合适的。

在正式下载/训练之前，先告诉我选择理由。

---

# 3. 模型选择

第一版使用轻量模型。

优先：

- 当前 Ultralytics 官方推荐的 nano 级 YOLO 模型

不要使用 large / xlarge。

目标是快速实验，而不是追求 benchmark SOTA。

所有模型版本必须根据当前已安装 Ultralytics 实际支持情况确认，不要凭记忆猜接口。

---

# 4. 项目结构

请建立清晰的工程目录，大致如下：

```text
CV-Engineering-Portfolio/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── yolo_optimization/
│   │
│   ├── README.md
│   │
│   ├── configs/
│   │   ├── baseline.yaml
│   │   └── experiments/
│   │
│   ├── scripts/
│   │   ├── check_environment.py
│   │   ├── prepare_dataset.py
│   │   ├── analyze_dataset.py
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   ├── analyze_errors.py
│   │   ├── visualize_failures.py
│   │   ├── compare_runs.py
│   │   └── inference.py
│   │
│   ├── results/
│   │   ├── baseline/
│   │   ├── experiments/
│   │   ├── fp_examples/
│   │   ├── fn_examples/
│   │   └── comparison/
│   │
│   ├── reports/
│   │   └── optimization_report.md
│   │
│   └── tests/
│
└── docs/
```

可以根据实际需要调整，但不要把所有功能塞进一个脚本或 Notebook。

Notebook 可以辅助分析，但不能成为主运行入口。

---

# 5. 环境验证

正式开始项目前，先完成环境检测脚本。

至少检查：

- Python version
- PyTorch version
- Ultralytics version
- OpenCV version
- CUDA availability
- MPS availability
- 当前 device
- 当前系统架构
- 必要目录是否存在

运行后打印清晰报告。

只有环境确认正常以后再继续。

---

# 6. Baseline

首先建立一个严格固定的 baseline。

baseline 必须记录：

- 模型版本
- dataset version
- random seed
- image size
- batch size
- epochs
- optimizer
- learning rate
- augmentation
- device
- Ultralytics version
- PyTorch version

至少保存：

- best weights
- last weights
- Precision
- Recall
- mAP50
- mAP50-95
- per-class metrics
- confusion matrix
- training curves
- validation predictions

Baseline 不允许在看到结果之后偷偷修改配置。

---

# 7. 数据集分析

在训练之前实现 dataset analysis。

至少分析：

## 类别分布
每个类别：
- image count
- object count
- percentage

## Bounding box 尺寸分布

按照目标面积分析：

- small
- medium
- large

可以采用 COCO 风格定义，或者根据该数据集尺寸建立合理定义。

如果自定义阈值，要说明理由。

## Bounding box aspect ratio

统计长宽比。

## 每张图目标数量

分析：
- mean
- median
- distribution

## 类别不平衡

识别明显少数类。

最终输出图表到：

`results/dataset_analysis/`

---

# 8. Error Analysis

这是整个 Portfolio 最重要的部分。

不要只显示 mAP。

实现真正的：

## False Positive analysis

找出：
- high-confidence FP
- 每类别 FP
- 典型 FP 图片

## False Negative analysis

找出：
- 每类别 FN
- small-object FN
- low-confidence misses
- 典型 FN 图片

## IoU / confidence analysis

分析：
- confidence distribution
- IoU distribution
- confidence threshold 对 Precision / Recall 的影响

## Object size performance

分别计算或尽可能分析：

- small object
- medium object
- large object

的检测表现。

## Per-class analysis

找出：
- 最差类别
- 最好类别
- 容易混淆类别

生成典型失败案例图片。

图片中清楚标记：

- Ground Truth
- Prediction
- FP
- FN
- confidence

不要只输出 JSON。

最终形成一个：

`baseline_error_analysis.md`

里面自动总结发现。

---

# 9. Controlled Experiments

不要进行大规模 hyperparameter search。

只进行少量、有明确假设的实验。

建议总共 3–5 组。

例如：

### Experiment A — Image Resolution

Baseline:
640

Experiment:
更高 resolution，例如 768 或 800

假设：
如果小目标较多，提高输入分辨率可能提高 small-object recall。

### Experiment B — Augmentation

根据数据集实际问题调整：

- scale
- translate
- mosaic
- hsv
- flip

不要无脑增加 augmentation。

### Experiment C — Training Hyperparameters

例如：
- learning rate
- patience
- epochs
- weight decay

一次只改变少量变量。

### Experiment D — Small Object Strategy

只有当 dataset analysis 真的证明存在 small-object 问题时再做。

---

# 10. 实验原则

每个 experiment 都必须：

1. 提出 Hypothesis
2. 写明 Compared with baseline
3. 尽量只修改一个主要因素
4. 使用相同 validation set
5. 记录完整 config
6. 保存训练输出
7. 给出结果
8. 分析结果是否支持假设

不要：

- 随机乱调参数
- 根据结果事后修改 hypothesis
- 只保留表现好的实验
- 删除失败实验

失败实验也可以保留，因为 Portfolio 想证明的是工程判断能力。

---

# 11. 比较结果

自动生成一个 summary table：

```text
Experiment | Precision | Recall | mAP50 | mAP50-95 | Notes
Baseline
Experiment A
Experiment B
Experiment C
Best
```

所有数字必须从真实实验结果中自动读取。

禁止手动编造。

同时输出：

- metrics comparison chart
- training curve comparison
- per-class comparison
- FP/FN comparison
- small-object comparison（如果适用）

---

# 12. 最终优化报告

自动生成：

`reports/optimization_report.md`

结构：

## Executive Summary

简洁说明：

- baseline 问题
- 主要发现
- 哪些优化有效
- 哪些无效
- 最终推荐配置

## Dataset Analysis

## Baseline Results

## Error Analysis

## Optimization Experiments

## Before vs After

## Remaining Failure Cases

## Recommended Production Configuration

## Deliverables

语言要偏工程和客户沟通，不要写成论文。

---

# 13. README

README 是非常重要的交付物。

它主要面向潜在 freelance 客户。

README 首页需要让别人 30 秒内看懂：

> 我拿到一个已有 YOLO 项目以后，能够系统地找出问题、做实验并优化模型。

README 建议结构：

# YOLO Model Optimization Portfolio

一句话简介：

Practical YOLO optimization workflow including dataset analysis, FP/FN diagnosis, controlled experiments, and before/after evaluation.

## Demo

放真实检测图片或 GIF。

## Problem

## Workflow

可以展示：

```text
Dataset
  ↓
Baseline
  ↓
Error Analysis
  ↓
Hypothesis
  ↓
Controlled Experiments
  ↓
Best Configuration
  ↓
Final Evaluation
```

## Dataset

说明：
- 来源
- License
- 数量
- 类别

## Baseline

## Error Analysis

重点展示 FP/FN 图片。

## Experiments

## Results

放 before / after 表。

## Deliverables

例如：

- optimized weights
- reproducible training config
- evaluation results
- FP/FN analysis
- inference script
- optimization report

## Quick Start

## Project Structure

## Reproducibility

## Hardware

明确说明：

本项目开发和小规模实验在 Apple Silicon / MPS 环境运行，但代码同时兼容 CUDA。

---

# 14. Inference Demo

实现一个简单 inference CLI。

例如：

```bash
python yolo_optimization/scripts/inference.py \
  --weights path/to/best.pt \
  --source path/to/image_or_video
```

支持：

- image
- directory
- video

输出：

- annotated result
- detection JSON 或 CSV

至少包含：

- class
- confidence
- bounding box

---

# 15. 代码质量

要求：

- 清晰函数划分
- type hints
- argparse
- logging
- config 驱动
- 异常处理
- 不硬编码绝对路径
- 不硬编码我的用户名
- 不依赖本机特定目录
- random seed 可配置
- 所有关键命令 README 中可复现

不要过度抽象。

---

# 16. 测试原则

每完成一个模块：

先运行。

确认能工作。

再继续。

至少测试：

- import
- config parsing
- device selection
- dataset loading
- metrics parsing
- inference

不要一次写完几十个文件再统一测试。

如果出现错误：

1. 阅读实际报错
2. 查官方文档
3. 定位原因
4. 修改
5. 重新测试

不要靠猜接口。

---

# 17. 依赖原则

使用当前官方稳定方案。

对于：

- Ultralytics
- PyTorch
- MPS
- YOLO API

如果不确定接口，请优先查看当前官方文档或已经安装包的实际 API。

不要因为记忆中的旧版本接口而写代码。

---

# 18. Git 原则

项目初始化 Git。

建立清晰 commit。

例如：

```text
chore: initialize CV portfolio structure
feat: add environment validation
feat: add dataset preparation
feat: add baseline training
feat: add dataset analysis
feat: add FP/FN error analysis
feat: add controlled experiments
docs: add optimization report and README
```

不要一个 commit 提交整个项目。

不要提交：

- dataset 原始大文件
- 巨大 cache
- 临时文件
- Python cache
- 无必要的大模型

weights 如果太大，应使用 GitHub Release 或明确说明下载方式。

---

# 19. 禁止事项

严格禁止：

1. 伪造 Precision / Recall / mAP。
2. 伪造实验结果。
3. 伪造训练截图。
4. 假装没有运行过的代码已经验证。
5. 为了 README 好看而编结果。
6. 偷偷删除失败实验。
7. 一上来跑大量 GPU 实验。
8. 使用大型模型。
9. 把项目做成过度复杂的 MLOps 系统。
10. 把所有逻辑塞进 Notebook。
11. 不验证代码就宣称完成。
12. 更改或破坏机器现有 Python 环境。

如果需要创建环境，优先建立独立 virtual environment / conda environment。

---

# 20. 执行顺序

严格按照下面顺序执行：

Phase 0
- 检查当前机器环境
- 检查磁盘
- 检查 Python
- 检查 MPS
- 选择项目目录

Phase 1
- 创建项目结构
- 创建独立 Python 环境
- 安装最低必要依赖
- environment smoke test

Phase 2
- 调研并选择数据集
- 检查 License
- 下载小规模数据
- dataset validation

Phase 3
- Dataset analysis

Phase 4
- Baseline
- 先使用极小 epoch 做 smoke test
- 确认 pipeline 没问题

Phase 5
- 正式 baseline

Phase 6
- Error analysis

Phase 7
- 提出 3–5 个 optimization hypotheses

注意：

不要直接跑实验。

先把：

- baseline findings
- error analysis
- proposed experiments

汇报给我。

只有设计合理以后再继续正式 optimization experiments。

Phase 8
- Controlled experiments

Phase 9
- Final comparison

Phase 10
- README
- report
- Git cleanup

---

# 21. 最终验收标准

这个项目完成后，我应该能够做到：

```bash
git clone <repo>
cd CV-Engineering-Portfolio
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

然后按照 README：

- 下载/准备数据
- 跑 baseline
- evaluate
- error analysis
- inference

关键流程应该能够复现。

同时，一个不懂我研究背景的 freelance 客户打开 GitHub 首页后，在 30–60 秒内应该能看懂：

1. 我会 YOLO。
2. 我不是只会调用训练命令。
3. 我会分析 FP/FN。
4. 我会根据问题设计优化实验。
5. 我会比较 before / after。
6. 我会交付可运行代码。
7. 我能处理真实 Computer Vision 项目。

---

现在先执行 Phase 0–Phase 2。

不要直接开始长时间训练。

先检查环境、确定技术栈、选择数据集、建立工程结构，并向我汇报：

1. 当前 Mac 环境情况
2. MPS 是否正常
3. 推荐数据集
4. 数据集 License
5. 项目目录结构
6. 当前依赖方案
7. 下一步准备怎么做
8. 可能存在的风险

确认基础 pipeline 没问题后，再进入 baseline。