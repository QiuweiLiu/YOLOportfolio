# HANDOFF — 可恢复快照

> 覆盖更新,不追加日志。最后更新:2026-08-18

## 当前状态

`in_progress` — Phase 0/1a 完成,等待用户批准 Phase 1b(提交)+ Phase 2(数据集调研,联网)。

## 本轮已完成(2026-08-18)

1. **Phase 0 环境验证** ✅
   - 修复 `~/.condarc`(删除失效 TUNA `pkgs/free` 频道,已批准)
   - 创建 conda 环境 `yolo-portfolio`(Python 3.11.15,conda-forge)
   - 安装并锁定依赖:torch 2.13.0 / ultralytics 8.4.121 / opencv-python 5.0.0.93 / numpy 2.4.6 / pandas 3.0.5 / matplotlib 3.11.1 / pyyaml 6.0.3
   - `check_environment.py` 运行通过:arm64、MPS ✅(recommended=mps)、目录全 OK、`pip check` 无冲突
   - **MPS 前向 smoke**: yolov8n.pt(已下载至 Ultralytics 默认缓存)合成图 predict MPS 8.8s 首次(含内核初始化)/ CPU 0.18s / 0 detections(随机图符合预期)
2. **Phase 1 配套**: `requirements.txt`(锁版本)、`environment.yml`(conda 复现)、`.gitignore`、`AGENTS.md`、控制面文档更新(D5–D7)。
3. Git 仍零提交(等 Phase 1b 一次性按功能提交或用户指示)。

## 待用户确认(恢复条件)

- C-A: 是否执行 Phase 1b —— LICENSE 选型(MIT 建议)+ 根 README 骨架 + 首次 git 提交(分 2–3 个 commit)
- C-B: **Phase 2 数据集调研授权**(需联网核实候选 License/下载方式,详见 PLAN.md 候选矩阵)

## 下一步(获批准后)

```text
Phase 1b: LICENSE + README 骨架 + git 首次提交(多 commit)
Phase 2: 候选核实(License 原文/规模/格式/YOLO 转换成本) → 推荐汇报 → 批准 → 下载至 data/raw → validate
Phase 3: analyze_dataset.py + 图表
```

## 关键路径与风险提醒(此时点)

- 磁盘余量 12.4 GB:数据集 ≤3GB 优先;下载前核对体积。
- 8GB RAM:MPS 训练 baseline 用小 batch;正式训练前 Phase 4 smoke。
- Roboflow 候选需 API key(计入成本),备选免账号来源。
- 所有实验数字必须真实自动读取;失败实验保留(项目红线)。

## 关键文件清单

- 控制面:`.project/{PROJECT,STATE,PLAN,DECISIONS,HANDOFF}.md`
- 环境:`environment.yml`、`requirements.txt`
- 代码:`yolo_optimization/scripts/check_environment.py`(第一个已验证脚本)
- 需求文档:根目录 `Codex Prompt — YOLO Optimization Portfolio.md`