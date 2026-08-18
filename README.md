# YOLO Model Optimization Portfolio

Practical YOLO optimization workflow including dataset analysis, FP/FN diagnosis, controlled experiments, and before/after evaluation.

> 本仓库是一个面向真实客户的 YOLO 模型优化工作流展示项目:拿到一个目标检测任务后,系统地分析数据、建立 baseline、诊断 False Positive / False Negative、设计受控实验、并交付可复现的优化结果。

## Demo

> ⏳ 待补充:真实检测对比图 / 演示视频(Phase 10 填入)

## Problem

客户项目中的 YOLO 模型往往存在这些问题:

- 只有 mAP,不知道**错在哪**:哪些类别最差、哪些是 FP、哪些是小目标没检到
- 数据本身的问题(类别不平衡、小目标占比、标注噪声)没有被量化
- 优化靠"加大 epoch / 调参",没有假设,改完也不知道为什么有效

本项目用一套可复现的流程回答:**问题出在哪里、什么方案有效、提升多少**。

## Workflow

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

> ⏳ 待定(Phase 2 选定后填写:来源 / License / 数量 / 类别)

## Baseline

> ⏳ 待定(Phase 5 填写真实指标)

## Error Analysis

> ⏳ 待定(Phase 6 填写 FP/FN 案例图)

## Experiments

> ⏳ 待定(Phase 7–8 填写)

## Results

> ⏳ 待定(Phase 9 填写 before/after 对比表)

## Deliverables

- [x] 可复现训练/评估/分析脚本(`yolo_optimization/scripts/`)
- [x] 环境检测脚本与锁定依赖(`check_environment.py` / `requirements.txt` / `environment.yml`)
- [ ] optimized weights(实验完成后)
- [ ] FP/FN 错误分析报告与案例图
- [ ] 优化对比报告(`yolo_optimization/reports/`)

## Quick Start

```bash
git clone <repo-url>
cd YOLOportfolio

# 1. 创建环境(Python 3.11)
conda env create -f environment.yml
conda activate yolo-portfolio

# 2. 验证环境
python yolo_optimization/scripts/check_environment.py

# 3. 准备数据 / 训练 / 评估 / 分析(详见对应脚本 --help)
```

## Project Structure

```text
YOLOportfolio/
├── data/raw/                  # 原始数据集(只读,git 不跟踪)
├── yolo_optimization/         # 主代码
│   ├── configs/               # 训练配置(baseline.yaml 等)
│   ├── scripts/               # 环境检查/数据下载/分析/训练/评估/错误分析/推理
│   ├── results/               # 结果产物(baseline/错误案例/对比)
│   ├── reports/               # 交付报告
│   └── tests/
├── docs/                      # 长期设计文档
├── requirements.txt           # 锁定依赖(Python 3.11)
├── environment.yml            # conda 环境复现
└── LICENSE
```

## Reproducibility

- 正式实验: 每个实验单独目录,记录 config / random seed / 数据版本 / 命令,数值统一写入 `metrics` 文件,全部数字从真实结果自动读取。
- 权重与数据集不提交仓库,按 README/脚本方式下载。
- 依赖版本锁定于 `requirements.txt`。

## Hardware

本项目开发与小规模实验在 **Apple Silicon (MPS)** 环境运行(torch 2.13 / Python 3.11),代码同时兼容 **NVIDIA CUDA** 环境(自动选择 cuda → mps → cpu)。

## License

- 仓库代码: MIT(见 LICENSE)
- 数据集 License: 以各数据集官方声明为准(选定后在本 README 注明)