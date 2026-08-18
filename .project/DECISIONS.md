# DECISIONS.md — 决策记录

> 追加记录重要决策:决策、理由、依据、状态。新决策 append,不改写历史。

---

## D1 — 项目根目录沿用当前目录(2026-08-18,已批准)

- **决策**: 不使用文档建议名 `CV-Engineering-Portfolio` 新建子目录,直接以 `/Users/liuqiuwei/Documents/YOLOportfolio` 为仓库根;README/H1 仍按文档用 "YOLO Model Optimization Portfolio" 标题(用户确认初始化时未要求改名)。
- **理由**: 目录已存在且是 git repo;改名只在将来决定。
- **状态**: 已执行(初始化即生效)。

## D2 — 控制面目录与交付结构融合

- **决策**: 顶层同时建立 Project OS 控制面目录(`.project/`、`docs/`、`data/raw/`、`experiments/`、`outputs/`、`.scratch/`)与需求文档要求的交付结构(`yolo_optimization/` 内含 configs/scripts/results/reports/tests);不另建 `src/`(避免与 `yolo_optimization/` 双写)。
- **理由**: 需求文档 §4 明确给出客户可见结构(交付物以它为准);Project OS 服务于项目治理;`src/` 语义与 `yolo_optimization/` 重复,遵循"不双写、不过度设计"。
- **影响**: 代码统一归 `yolo_optimization/`;正式实验归 `experiments/EXP-xxx`;程序临时输出归 `outputs/`;results 内保留需求文档指定子目录。

## D3 — Python 环境策略:独立 venv,Python 3.11

- **决策**: 在项目内建 `.venv`(若机器无 3.11 解释器,先经确认安装 3.11 来源再建);永不修改系统 Python(实测 3.9.13)。
- **理由**: 文档 §17/§19-12(不破坏现有环境);需求明确 Python 3.11。
- **状态**: 待 Phase 0 执行;来源方案(conda/uv/pyenv)待用户确认。

## D4 — 正式实验登记制

- **决策**: 每个正式实验唯一 `EXP-xx_名称` 目录,含 config、metrics.csv、RESULT.md;`experiments/EXPERIMENT_LEDGER.jsonl` 一行一实验;数值事实只写 metrics 文件, RESULT.md 解释不复制数值。
- **理由**: 文档 §6/§10/§11(可复现、禁止编造、禁止删失败实验);与 Project OS 正式实验规范一致。
- **状态**: 生效(Phase 4 起强制)。

## D5 — 修复 ~/.condarc 失效频道(2026-08-18,用户已批准)

- **决策**: 从 `~/.condarc` 删除 TUNA 已下线频道 `https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/`(实测 404);保留 conda-forge、TUNA main、defaults 与原 strict 优先级。
- **理由**: 该频道的存在使任何 conda 操作直接失败;删除是最小修复(用户确认后执行)。
- **状态**: 已修复并验证。

## D6 — Python 3.11 来源采用已装 miniconda(2026-08-18,用户已批准)

- **决策**: 不装 uv/pyenv/brew(本机均未装),直接用已有 `/opt/miniconda3` 创建 `yolo-portfolio` 环境(python=3.11.15,conda-forge);依赖经该环境 pip 安装并锁定于 `requirements.txt`(实际版本)。
- **理由**: 零新增工具、不破坏 base 3.13 与系统 3.9、README 复现可用 `conda env create -f environment.yml`。
- **状态**: 已执行(2026-08-18)。

## D7 — device 自动选择顺序:cuda → mps → cpu

- **决策**: 设备选择遵循需求文档 §1 代码示例(order: cuda → mps → cpu),`check_environment.py` 已实现;本机实测 MPS=True。
- **理由**: 需求文档为最高权威,示例代码即此顺序;Mac 上 cuda 永远不可达,顺序不影响本机行为,但保证未来 CUDA 环境直接生效。
- **状态**: 已实现。

## 待决策(TBD)

- 数据集最终选择 + License 确认——Phase 2 汇报。
- LICENSE 文件选型(MIT 等)——Phase 1 收尾确认。