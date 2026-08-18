# STATE.md — 当前真实状态

> 更新:2026-08-18(Phase 0 完成,Phase 1 结构就绪)

## 当前阶段

- ✅ **Phase 0 环境验证完成**;Phase 1 核心(独立环境 + 依赖 + check_environment.py)完成。
- 下一步:**Phase 2 数据集调研**(候选核实 + License 汇报 + 用户确认后下载)。

## 已验证事实(2026-08-18 实测)

| 项 | 结果 |
|---|---|
| 平台 | macOS 26.2 (Build 25C56),arm64 |
| RAM | 8.0 GB(偏小,影响 batch/resolution 上限) |
| 磁盘 | 245 GB 总量,剩余 ~12.4 GB(`${PROJECT_ROOT}` 所在卷) |
| 系统 Python | 3.9.13(未动) |
| **conda 环境** | `yolo-portfolio` = **Python 3.11.15**(/opt/miniconda3/envs/yolo-portfolio) |
| torch / ultralytics | **2.13.0** / **8.4.121** |
| opencv / numpy / pandas / matplotlib / pyyaml | 5.0.0.93 / 2.4.6 / 3.0.5 / 3.11.1 / 6.0.3 |
| **MPS** | ✅ **可用且前向已验证**(yolov8n 640 合成图 smoke:model load 77s 含下载、mps predict 8.8s 首次含内核初始化、cpu 0.18s、0 detections 符合随机图预期) |
| CUDA | 不可用(本机无 GPU,代码兼容)cuda → mps → cpu 选择顺序遵循需求文档 §1 |
| 依赖健康 | `pip check` 无冲突;全部 import OK |
| Git | main 分支仍零提交(待 Phase 1 收尾统一提交) |

## 已修复的环境问题

- `~/.condarc` 曾引用 TUNA 已下线频道 `pkgs/free`(404,conda 任何操作失败)→ 已删该行,保留 TUNA main + conda-forge + defaults(用户已批准,D5)。

## 待验证/待确认

1. **数据集选择 + License(Phase 2,汇报后由用户批准下载)**
2. 磁盘余量:12.4 GB,数据集 ≤3 GB 可行,但训练产物(多组实验权重 ~6 MB/个 + 结果图)要留意;必要时清理 pip 缓存。

## 已登记问题/风险

- R1 8GB RAM:MPS 训练 batch 受限 → 计划采用小 batch + 可能梯度累积;正式训练前 smoke。
- R2 磁盘 12.4 GB: 下载数据集前核对体积;Ultralytics 权重缓存默认在 `~/Library/Application Support/Ultralytics`,注意不误放项目外。
- R3 MPS 算子兼容(文档要求回退 CPU 并如实记录)——前向已验证可用,训练收敛性待 Phase 4 smoke 确认。

## 下一步(见 PLAN.md)

Phase 2 数据集调研与选择 → 用户确认 License 与下载方式 → 下载 → validate → Phase 3 数据分析。