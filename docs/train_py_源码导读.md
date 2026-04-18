# train.py 源码导读

对应源码：
- [train.py](train.py)

作用定位：
- 这是训练系统装配入口。
- 负责解析命令行、加载配置、构建环境、构建 QMIX 参数，并调用训练主循环。

---

## 1. 调用总览

主函数入口：
- [train.py](train.py#L94)

调用链：
1. 解析参数 [_parse_args](train.py#L33)
2. 读取 JSON 配置 [_load_json_config](train.py#L50)
3. 合并命令行与配置值 [_get_value](train.py#L86)
4. 构造 QMIXConfig [train.py](train.py#L114)
5. 初始化日志 [train.py](train.py#L136)
6. 设置全局随机种子 [_set_global_seed](train.py#L27)
7. 构建环境 GazeboMultiUAVParallelEnv [train.py](train.py#L140)
8. 调用训练主循环 train_attention_qmix [train.py](train.py#L181)

---

## 2. 关键函数逐段说明

### 2.1 _set_global_seed
- 位置：[train.py](train.py#L27)
- 输入：seed(int)
- 输出：无
- 行为：同步设置 random、numpy、torch 的种子
- 常见坑：
  - 若只设 numpy 不设 torch，会出现“看似固定种子但模型行为仍漂移”。

### 2.2 _parse_args
- 位置：[train.py](train.py#L33)
- 输入：命令行
- 输出：argparse 命名空间
- 关键参数：
  - --config
  - --resume
  - --episodes / --max-steps
  - --device(auto/cpu/cuda)
- 常见坑：
  - 命令行参数会作为默认值，被配置文件覆盖，排查时要看合并后的最终日志配置行。

### 2.3 _load_json_config
- 位置：[train.py](train.py#L50)
- 输入：path
- 输出：dict
- 逻辑：
  - 允许传完整路径、无后缀名、仅文件名
  - 自动尝试在 configs 目录下查找
- 常见坑：
  - 配置名拼写错误时会尝试多个候选，最终抛 FileNotFoundError。

### 2.4 _get_value
- 位置：[train.py](train.py#L86)
- 输入：cfg_dict, key, fallback
- 输出：配置值或默认值
- 作用：统一“配置优先，参数兜底”。

### 2.5 _default_run_name
- 位置：[train.py](train.py#L90)
- 输入：seed
- 输出：run_name 字符串
- 作用：生成带时间戳的实验目录名。

### 2.6 main
- 位置：[train.py](train.py#L94)
- 输入：无（内部读取 args）
- 输出：无（副作用是训练并写工件）

分段逻辑：
1. 读配置并提取环境参数
2. 组装 QMIXConfig（训练超参与稳定性参数）
3. 启用日志系统
4. 设置随机种子
5. 创建 Gazebo 并行多智能体环境
6. 打印最终生效训练配置
7. 调用 train_attention_qmix 启动训练

常见坑：
- world_name 与 Gazebo 实际 world 不一致，后续 reset 可能失败。
- 误以为 device=cuda 一定生效，实际会在算法层做自动回退。

---

## 3. 与其他模块的数据边界

输入到环境构造器：
- uav_names, lidar_size, max_steps, goal_xyz, arena 边界、reset 稳定化参数等
- 对应调用位置 [train.py](train.py#L140)

输入到算法配置：
- batch_size, lr, gamma, epsilon, clip 参数、输出目录参数
- 对应调用位置 [train.py](train.py#L114)

训练结果输出：
- artifacts/qmix/<run_name>/ 下的模型、日志、summary
- 实际写入逻辑在 [src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L506)

---

## 4. 快速排查点

1. 配置是否真正生效
- 看 runtime 的 TrainConfig 行

2. 设备是否按预期
- 看训练日志中的 compute device 输出

3. 训练是否启动
- 若 main 后立即中断，多半是环境 reset 或桥接问题

4. resume 是否生效
- 看 summary 的 resumed 字段与 start_episode

---

## 5. 推荐阅读顺序

1. [train.py](train.py#L94)
2. [src/scripts/gazebo_pettingzoo_env.py](src/scripts/gazebo_pettingzoo_env.py#L495)
3. [src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L506)
4. [evaluate.py](evaluate.py#L114)
