# 多无人机覆盖强化学习项目（ROS2 Jazzy + Gazebo Harmonic）

更新时间：2026-04-20

## 项目简介

本项目用于研究“基于注意力机制的多智能体强化学习无人机路径覆盖方法”。
核心能力：

- Gazebo 多无人机并行仿真环境（4 UAV）
- Attention-QMIX 训练与评测闭环
- 课程学习（Stage1 -> Stage2 -> Stage3）
- 自动日志、指标、图表、模型工件输出

## 当前代码结构

- 训练入口：`train.py`
- 评测入口：`evaluate.py`
- 课程脚本：`train_curriculum_p0.sh`、`src/scripts/train_curriculum_p0.sh`
- 算法实现：`src/scripts/attention_qmix.py`
- 环境封装：`src/scripts/gazebo_pettingzoo_env.py`
- 桥接脚本：`setup_bridges.sh`、`src/scripts/setup_bridges.sh`
- 配置目录：`configs/`
- 文档目录：`docs/`
- 输出目录：`artifacts/qmix/`

## 快速开始

### 1) 启动仿真世界

```bash
gz sim -s src/worlds/mazes/stage_1_open.sdf
```

### 2) 启动 ROS2/Gazebo 桥接

```bash
bash src/scripts/setup_bridges.sh
```

### 3) 启动训练

```bash
python3 train.py --config configs/p0_stage1_open.json
```

指定设备：

```bash
python3 train.py --config configs/p0_stage1_open.json --device cuda
```

从 checkpoint 继续训练：

```bash
python3 train.py --config configs/p0_stage1_open.json --resume artifacts/qmix/<run_name>/best_model.pt
```

### 4) 课程学习（推荐）

```bash
bash train_curriculum_p0.sh
```

默认依次使用：

- `configs/p0_stage1_open.json`
- `configs/p0_stage2_easy.json`
- `configs/p0_stage3_maze1.json`

说明：

- 跨阶段 resume 时会创建新 run 目录。
- 跨阶段 resume 时会加载网络权重，但重置训练计数与 epsilon 到当前阶段配置值。

### 5) 运行评测

```bash
python3 evaluate.py --run-dir artifacts/qmix/<run_name> --episodes 10 --max-steps 500
```

也可直接指定模型：

```bash
python3 evaluate.py --model artifacts/qmix/<run_name>/best_model.pt --episodes 10
```

## 训练与评测输出

每个 run 目录下常见文件：

- `metrics.jsonl`：逐回合训练指标
- `summary.json`：训练汇总
- `best_model.pt`、`latest_model.pt`、`final_model.pt`
- `checkpoints/`：周期 checkpoint
- `plots/learning_curves.png`
- `plots/task_metrics.png`
- `plots/benchmark_metrics.png`
- `logs/runtime.log`、`logs/errors.log`
- `evaluation/<timestamp>/summary.json`
- `evaluation/<timestamp>/episode_metrics.jsonl`
- `evaluation/<timestamp>/eval_performance.png`
- `evaluation/<timestamp>/eval_task_metrics.png`

## 当前任务定义（对齐代码）

- 动作：离散动作映射为 4 维连续控制量
- 回合终止：达到 `max_steps` 或全部 agent 结束
- 关键指标：
  - `coverage_rate`
  - `full_coverage_success`
  - `collision_rate`
  - `out_of_bounds_rate`
  - `repeated_coverage_rate`
  - `overlap_rate`

## 常见问题

1. 训练启动即失败：先检查世界是否运行、桥接是否常驻。
2. loss 长时间为空：通常是 replay buffer 未达到 batch 大小。
3. 训练和评测差异大：优先使用 `--run-dir` 评测，确保结构和环境参数自动恢复。

## 相关文档

- `docs/训练日志诊断手册.md`
- `docs/train_py_源码导读.md`
- `docs/attention_qmix_源码导读.md`
- `docs/gazebo_pettingzoo_env_源码导读.md`
- `docs/evaluate_py_源码导读.md`
