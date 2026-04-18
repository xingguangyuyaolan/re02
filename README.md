# 多无人机区域覆盖仿真（基于 ROS2 + Gazebo）

## 项目介绍

本项目基于 ROS2 Jazzy 与 Gazebo Harmonic，搭建了一个多无人机（Multi-UAV）协同路径覆盖仿真环境。
目标是为“基于注意力机制的多智能体强化学习路径覆盖”研究提供仿真基础：
- 在迷宫场景中部署多个 UAV 代理
- 每个 UAV 具有独立的 ROS2 话题命名空间（cmd_vel/odom/tf）
- 支持通过 ROS2 节点发布控制指令、获取观测，并为后续强化学习训练提供接口

## 功能特点

- **多 UAV 模型**：`src/models/uav1`、`src/models/uav2`、`src/models/uav3`、`src/models/uav4` 等模型目录已配置独立控制/观测话题。
- **迷宫场景**：`src/worlds/mazes/maze_*.sdf` 已支持多 UAV 同时 spawn，并可在 Gazebo 中直接运行。
- **ROS2/Gazebo 桥接**：`src/scripts/setup_bridges.sh` 用于建立 Gazebo 与 ROS2 之间的通信（topic bridge）。

## 使用方法

### 1) 运行仿真场景

```bash
gz sim src/worlds/mazes/maze_1.sdf
```

新增了由易到难的训练场景（保留原有 `maze_1.sdf`）：

- `src/worlds/mazes/stage_1_open.sdf`：仅起点与终点，无障碍。
- `src/worlds/mazes/stage_2_easy.sdf`：少量简单障碍。
- `src/worlds/mazes/stage_3_intermediate.sdf`：中等数量障碍与绕行。
- `src/worlds/mazes/stage_4_challenging.sdf`：高密度障碍，路径更曲折。
- `src/worlds/mazes/stage_5_maze_like.sdf`：类迷宫结构，难度最高。

例如从最简单场景开始：

```bash
gz sim src/worlds/mazes/stage_1_open.sdf
```

### 2) 启动 ROS2 <-> Gazebo 桥接

```bash
bash src/scripts/setup_bridges.sh
```

### 3) 训练 QMIX 多无人机区域覆盖策略

本仓库已将 QMIX 训练核心适配到 Gazebo 多无人机环境（CTDE）。

```bash
python3 train.py
```

指定训练设备（`auto`/`cpu`/`cuda`，默认 `auto`）：

```bash
python3 train.py --device cuda
```

使用可复现实验配置（含 NaN 防护参数与固定种子）：

```bash
python3 train.py --config configs/repro_nan_fix.json
```

使用阶段化训练配置（例如 stage_1 开始）：

```bash
python3 train.py --config configs/stage_1_open_train.json
```

训练结束后会在对应 run 目录下自动生成图表文件：

- `plots/learning_curves.png`：reward、loss、epsilon、episode_steps 折线图
- `plots/task_metrics.png`：coverage_rate、repeated_coverage_rate、overlap_rate、collision_rate、out_of_bounds_rate、coverage_completion_time、full_coverage_success 折线图
- `plots/benchmark_metrics.png`：环境步进耗时、训练更新耗时、每回合更新次数

也可以对历史训练结果单独重新生成图表：

```bash
python3 plot_metrics.py --run-dir artifacts/qmix/<run_name>
```

说明：`--config` 支持简写文件名，例如 `stage_1_open_train.json`，程序会自动在 `configs/` 下查找。

从 checkpoint 继续训练：

```bash
python3 train.py --config configs/repro_nan_fix.json --resume latest
```

或指定具体 checkpoint 文件：

```bash
python3 train.py --resume artifacts/qmix/<run_name>/latest_model.pt
```

评估模型并输出评估阶段图表：

```bash
python3 evaluate.py --run-dir artifacts/qmix/<run_name> --episodes 10 --max-steps 300
```

评估结果会写入对应训练目录下的 `evaluation/<timestamp>/`，包含：

- `episode_metrics.jsonl`：逐回合评估指标
- `summary.json`：评估汇总
- `eval_performance.png`：reward、steps、coverage_rate、full_coverage_success 图表
- `eval_task_metrics.png`：collision_rate、out_of_bounds_rate、repeated_coverage_rate、overlap_rate、coverage_completion_time 图表

训练脚本入口：`train.py`
核心算法实现：`src/scripts/attention_qmix.py`
环境封装：`src/scripts/gazebo_pettingzoo_env.py`

说明：
- 动作为离散动作集合（内部映射为 `cmd_vel` 连续控制）。
- 奖励由新区域覆盖奖励、重复覆盖惩罚、协同分区奖励、重叠惩罚、碰撞惩罚与步长惩罚组成，支持区域全覆盖学习。
- 回合在覆盖率达到阈值、全部无人机失效，或达到 `max_steps` 时结束。
- 训练阶段包含数值保护与诊断输出：观测 NaN/Inf 清洗、Q/TD 裁剪、非有限损失与梯度告警。
- 训练过程会自动保存到 `artifacts/qmix/<run_name>/`：
	- `metrics.jsonl`：逐回合 reward/loss/epsilon 日志
		- 额外包含简短 benchmark 字段：
			- `benchmark_env_step_time_sec`
			- `benchmark_train_update_time_sec`
			- `benchmark_train_updates`
		- 额外包含任务指标：
			- `coverage_rate`
			- `repeated_coverage_rate`
			- `overlap_rate`
			- `collision_rate`
			- `out_of_bounds_rate`
			- `coverage_completion_time`
			- `full_coverage_success`
	- `latest_model.pt`：最新模型
	- `best_model.pt`：最佳 reward 模型
	- `final_model.pt`：训练结束最终模型
	- `checkpoints/episode_XXXX.pt`：周期性 checkpoint
	- `logs/runtime.log`：完整运行日志（配置、诊断、告警）
	- `logs/errors.log`：仅错误与异常日志（含堆栈）
	- `config.json` 与 `summary.json`：配置快照与汇总统计
		- `summary.json` 额外包含 benchmark 均值统计（env/train/update）与任务指标均值
	- 支持 resume：`--resume latest` 或 `--resume <checkpoint_path>`

### 4) 验证多 UAV 话题

```bash
ros2 topic list | grep uav
```

## 目录说明

- `src/models/uav1`, `src/models/uav2`, `src/models/uav3`, `src/models/uav4`：独立 UAV 模型目录，包含 `model.sdf`/`model.config`。
- `src/worlds/mazes/maze_*.sdf`：原有迷宫场景（保留不变）。
- `src/worlds/mazes/stage_*.sdf`：新增难度递进场景（从无障碍到类迷宫）。
- `train.py`：QMIX 训练入口。
- `src/scripts/attention_qmix.py`：适配 Gazebo 多无人机环境的 QMIX 训练核心。
- `src/scripts/gazebo_pettingzoo_env.py`：Gazebo + ROS2 的并行多智能体环境封装。
- `src/scripts/setup_bridges.sh`：启动 Gazebo/ROS2 通信桥接。
