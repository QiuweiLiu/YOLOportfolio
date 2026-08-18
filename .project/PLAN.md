# PLAN.md — 唯一当前执行计划

> 创建:2026-08-18 | 当前进行中:Phase 0–2 规划待用户确认

依据需求文档 `Codex Prompt — YOLO Optimization Portfolio.md` 的分阶段顺序(Phase 0–10)。
每个阶段都有:产出 / 验收标准 / 用户确认点。正式实验(Phase 4+)必须登记 `experiments/EXPERIMENT_LEDGER.jsonl` 并记录 commit、config、seed、数据版本、命令、硬件。

## 阶段总览

| Phase | 内容 | 产出 | 状态 |
|---|---|---|---|
| 0 | 环境验证与方案 | 环境报告、Python 3.11 来源 | ✅ 完成(2026-08-18) |
| 1 | 工程结构 + 独立环境(环境部分) | 目录、.gitignore、requirements.txt、environment.yml、check_environment.py、smoke | 🔄 结构/环境完成;LICENSE/README 骨架、Git 首次提交待做 |
| 2 | 数据集调研与下载 | 候选比较表、选定数据 + License + 下载脚本、data/raw 校验 | ⏳ 下一步 |
| 3 | 数据集分析 | analyze_dataset 脚本 + results/dataset_analysis/ 图表 | — |
| 4 | Baseline smoke | 极小 epoch 验证 pipeline(仅"能跑通") | — |
| 5 | 正式 baseline | 权重 + 全指标 + per-class + confusion matrix + 训练曲线 | — |
| 6 | Error analysis | FP/FN 案例图、IoU/conf 分析、分尺寸性能 + baseline_error_analysis.md | — |
| 7 | 假设设计 | baseline findings + error analysis + 3–5 个实验假设报告 | — |
| 8 | 受控实验 3–5 组 | EXP-xxx 目录、metrics.csv、RESULT.md | — |
| 9 | 对比 | comparison 表(自动读结果)+ 图表 + before/after | — |
| 10 | 交付 | README、optimization_report.md、Git cleanup、可选 demo | — |

> 用户确认点:C2(Phase 2 数据集)、C3(Phase 4 正式跑前)、C4(Phase 7 实验设计)。

## Phase 0 — 环境验证(下一步执行)

1. `df -h` 检查磁盘余量;确认数据下载与训练缓存空间。
2. 确认 Python 3.11 来源方案(候选:conda/miniforge、uv、pyenv —— 待用户指定或批准其一)。
3. 创建独立 venv(`python3.11 -m venv .venv`)。
4. 写 `yolo_optimization/scripts/check_environment.py`(文档 §5 要求):Python/PyTorch/Ultralytics/OpenCV 版本、CUDA/MPS device、系统架构、必要目录;打印报告;写 requirements.txt。
5. 原生跑通最小 smoke:import + device detection + 一次 1-sample 前向。
6. 只安装最低必要依赖(文档 §17:不破坏现有环境,独立环境优先)。

依赖方案(最低集,Phase 1 写入 requirements.txt):
`torch`、`ultralytics`、`opencv-python`、`numpy`、`pyyaml`、`pandas`、`matplotlib`。版本以安装后实际为准并记录。

## Phase 1 — 工程结构

- 已建目录骨架(2026-08-18 初始化时创建)。
- 补充:`.gitignore`、`LICENSE`、根 `README.md` 骨架(最终版 Phase 10)、`requirements.txt`。
- Git:按功能提交(初始化 → check_environment → prepare/analyze → train/eval → error analysis → experiments → docs),禁止一个 commit 全项目。

## Phase 2 — 数据集调研(候选矩阵,下载前汇报)

评估维度:规模(500–3000 张)、类别数、License、下载方式、是否已有 YOLO 格式、是否适合快速实验、是否有真实 error analysis 空间(小目标/类别不平衡)。

初始候选(Phase 2 逐一核实官方来源与 License 文本,并最终向用户推荐一个):

| 候选 | 预估规模 | 特点 | 待核实点 |
|---|---|---|---|
| Roboflow「Aquarium」系 | ~600–800 张 | 小/密集目标、现成 YOLO 格式 | 需 Roboflow API key;License 以页面为准 |
| TACO(垃圾检测) | ~1500 张 | 多类别、噪声标注、CC BY 4.0 | 需自行转 YOLO 格式;类别多是否合适 |
| PASCAL VOC 2012 | ~11000 张(可抽子集) | 经典、标注质量高 | License 仅限研究,商用需注意;规模子集化 |
| VisDrone-DET | 千–万级(可抽) | 典型小目标/无人机视角 | License 学术用途限制,接单展示有争议 |
| Google Open Images 子集 | 可抽 500–3000 | 类别多、CC 图片为主 | 需按图级 License 筛选,转换成本较高 |

选择标准优先级:License 清晰(可公开展示)> 700–2000 张 > 有真实优化空间 > 转换成本低。选定后向用户汇报理由与 License(文档 §2 要求),确认后再下载。

## 实验框架预规划(Phase 7 定稿,此处仅记方向)

候选假设(基于 error analysis 结果筛 3–5 个,单变量为主,相同验证集):
- 分辨率(A/768–800 对 640):小目标 recall 假设
- augmentation(B/mosaic、scale、translate 等定向调整)
- 训练超参(C:lr、epochs、weight decay,一次只动少量)
- 小目标策略(D:仅在 dataset analysis 证明确有 small-object 问题才做)
失败实验保留不删,数字一律自动读真实结果。

## 风险与缓解

- 8GB RAM:batch/resolution 受限 → 小 batch + 梯度累积;正式实验前 smoke。
- MPS 算子兼容:优先兼容写法,必要时回退 CPU 并记 README(不静默吞错误)。
- Roboflow 需要账号/API key:候选评估时作为成本计入,备选免账号来源。
- 系统 Python 3.9 干扰:全部操作走 `.venv/`,脚本用 `python -m` 方式执行。

## 当前任务列表

1. [x] Phase 0:环境验证(磁盘/conda 修复/check_environment.py/smoke)
2. [x] Phase 1a:requirements.txt(锁实际版本)+ environment.yml + .gitignore
3. [ ] Phase 1b:LICENSE 选型、README 骨架、Git 首次按功能提交(待用户点头)
4. [ ] Phase 2:数据集候选核实与推荐汇报(需批准联网调研与下载)