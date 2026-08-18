# YOLO Optimization Portfolio — 项目级指令

叠加在全局 `~/.config/opencode/AGENTS.md` 之上,面向本仓库。

## 控制面

- 读取顺序:`.project/PROJECT.md` → `.project/STATE.md` → `.project/PLAN.md` → `.project/HANDOFF.md`
- 状态、计划、决策、交接只写入 `.project/` 下对应文件;不建立平行 canonical 文档。
- 每轮涉及项目状态的工作结束后,覆盖更新 `HANDOFF.md`。

## 项目纪律(承接需求文档红线)

1. 禁止伪造任何指标/实验/截图;未验证的代码不得宣称完成;失败实验保留。
2. 正式实验必须登记 `experiments/EXPERIMENT_LEDGER.jsonl`,结果入 `experiments/EXP-xxx/`,数字只写 `metrics.csv/json`,RESULT.md 不复制数值。
3. 环境:全部操作使用项目独立 venv(Python 3.11);不修改系统 Python(3.9.13)。
4. 设备自动判断 mps → cuda → cpu;MPS 不支持算子回退 CPU 并如实记录,不静默忽略。
5. 不猜接口:Ultralytics/PyTorch API 一律查已装版本实际接口或官方文档。
6. 不硬编码用户名/绝对路径/本机目录。
7. 交付物:README 面向 freelance 客户;报告偏工程语言,不写论文腔。
8. 下载数据集、联网调研、高算力训练前,先向用户汇报方案得到确认。