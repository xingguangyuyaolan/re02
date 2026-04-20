# evaluate.py 源码导读（当前版本）

对应文件：`evaluate.py`

更新时间：2026-04-20

## 1. 文件职责

`evaluate.py` 用于对训练完成模型进行离线评测，输出：

1. 回合级评测明细
2. 汇总统计
3. 评测图表

## 2. 关键流程

1. 解析 `--run-dir` 或 `--model`
2. 加载训练时保存的 `config.json`
3. 重建与训练一致的模型结构
4. 加载 checkpoint，强制 `epsilon=0` 贪心评估
5. 重建环境并执行评测
6. 写出 `evaluation/<timestamp>/` 工件

## 3. 当前版本对齐点

### 3.1 模型结构对齐

评测会恢复注意力相关开关，避免“训练结构与评测结构不一致”。

### 3.2 环境参数对齐

评测会恢复关键环境参数（边界、碰撞、重置、安全约束等），保证训练-评测口径一致。

### 3.3 兼容旧 run

若历史 run 配置字段不完整，评测会按默认值回退。

## 4. 常用命令

```bash
python evaluate.py --run-dir artifacts/qmix/<run_name> --episodes 10 --max-steps 500
python evaluate.py --model artifacts/qmix/<run_name>/best_model.pt --episodes 10
python evaluate.py --run-dir artifacts/qmix/<run_name> --device cpu
```

## 5. 输出文件

评测目录下生成：

1. `episode_metrics.jsonl`
2. `summary.json`
3. `eval_performance.png`
4. `eval_task_metrics.png`
