# PROJECT — YOLO Optimization Portfolio

> 最后更新:2026-08-18(初始化)

## 目标

在 Mac (Apple Silicon) 上从零构建一个用于远程兼职/自由职业接单展示的 **Computer Vision Portfolio 项目**，公开放在 GitHub，用于 Upwork、Freelancer、电鸭、程序员客栈等平台投递 YOLO/Computer Vision/Video Analytics 短期项目。

定位不是论文创新或复杂框架，而是模拟真实客户常见的 YOLO 模型优化任务：

```
已有 baseline → 数据集分析 → baseline 训练评估 → FP/FN 诊断
→ 目标尺寸/类别问题分析 → 少量受控优化实验 → before/after 对比 → 交付
```

## 范围

- 数据集调研与选择(公开、License 清晰、500–3000 张、有真实优化空间)
- 数据集分析(类别分布、bbox 尺寸、长宽比、每图目标数、类别不平衡)
- 严格固定的 baseline(完整记录 config,禁止事后改配置)
- FP/FN Error Analysis(每类 FP/FN、小目标 FN、IoU/confidence 分析、分尺寸性能、失败案例图)
- 3–5 组受控优化实验(每组有假设、单变量为主、相同验证集)
- before/after 对比(数字全部从真实实验结果自动读取)
- 交付:README、optimization_report.md、baseline_error_analysis.md、inference CLI、可复现代码

## 非目标(明确排除)

- 不做 benchmark SOTA 追求;不用 large/xlarge 模型
- 不做大规模 hyperparameter search
- 不做过度复杂的 MLOps 系统
- 不把所有逻辑塞进 Notebook

## 硬性约束(红线)

1. 禁止伪造 Precision/Recall/mAP/截图/结果;未验证的代码不得宣称完成
2. 禁止删除失败实验;禁止事后修改 hypothesis 或 baseline 配置
3. 不损坏机器现有 Python 环境(一律独立 venv)
4. 不硬编码用户名/绝对路径/本机目录
5. 所有关键命令可复现(README 可复现)
6. 不确定接口时查官方文档或已装包实际 API,不凭记忆猜接口

## 环境

- Apple Silicon Mac (arm64),macOS 26.2,RAM 8GB
- 训练设备自动判断:mps 优先(cuda 环境时自动切 cuda,否则 cpu);MPS 不支持的算子回退 CPU 并如实记录
- Python 3.11(独立 venv);PyTorch + Ultralytics YOLO(nano 级)+ OpenCV
- 代码必须同时兼容 CUDA 环境,不写死

## 架构概览

```
YOLOportfolio/
├── .project/                  # Project OS 控制面(本目录)
├── AGENTS.md                 # 项目级指令
├── docs/                     # 长期设计文档
├── data/raw/                 # 原始数据(只读)
├── experiments/              # 正式实验(EXP-xxx + EXPERIMENT_LEDGER.jsonl)
├── outputs/                  # 临时程序输出(非事实源)
├── .scratch/                 # 临时分析与草稿(任务后清理/迁移)
└── yolo_optimization/        # 主代码包(客户可见交付结构,按需求文档)
    ├── configs/              # baseline.yaml + experiments/
    ├── scripts/              # check_environment / prepare / analyze / train / evaluate / ...
    ├── results/              # baseline/ fp_examples/ fn_examples/ comparison/ dataset_analysis/
    ├── reports/              # optimization_report.md、baseline_error_analysis.md
    └── tests/
```

设计决策详见 `DECISIONS.md`。当前真实状态见 `STATE.md`,执行计划见 `PLAN.md`。