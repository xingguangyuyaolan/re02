# evaluate.py 源码导读

对应源码：
- [evaluate.py](evaluate.py)

作用定位：
- 离线评估训练好的模型。
- 读取 checkpoint，构造与训练一致的策略网络，在 epsilon=0 下做纯贪心评估。

---

## 1. 调用总览

主函数入口：
- [evaluate.py](evaluate.py#L114)

调用链：
1. 参数解析 [_parse_args](evaluate.py#L47)
2. 解析模型路径 [_resolve_model_path](evaluate.py#L91)
3. 加载运行配置 [_load_run_config](evaluate.py#L107)
4. 根据配置重建 QMIXForUAV
5. 加载 checkpoint 并强制 epsilon=0
6. 构建环境并执行评估回合
7. 输出汇总指标

---

## 2. 关键函数逐段说明

### 2.1 _parse_args
- 位置：[evaluate.py](evaluate.py#L47)
- 输入：命令行
- 输出：argparse 命名空间
- 关键参数：
  - --run-dir 或 --model（二选一）
  - --episodes
  - --max-steps
  - --device
  - --goal-x/--goal-y/--goal-z
  - --world

常见坑：
- run-dir 和 model 同时传会冲突，脚本已用 mutually exclusive group 限制。

### 2.2 _resolve_model_path
- 位置：[evaluate.py](evaluate.py#L91)
- 输入：args
- 输出：model_path, config_path
- 逻辑：
  - 给 run-dir 时优先 best_model.pt，否则 final_model.pt
- 常见坑：
  - run-dir 不完整时会找不到模型文件。

### 2.3 _load_run_config
- 位置：[evaluate.py](evaluate.py#L107)
- 输入：config_path
- 输出：dict 或 None
- 作用：读取训练阶段保存的结构与配置快照。

### 2.4 main
- 位置：[evaluate.py](evaluate.py#L114)
- 输入：无（内部读取 args）
- 输出：控制台评估摘要

分段逻辑：
1. 解析路径与配置
2. 读取模型结构参数（n_agents/obs_dim/state_dim/action_table）
3. 构建 QMIXConfig，强制 epsilon=0
4. 实例化 QMIXForUAV 并 load_checkpoint
5. 创建环境并按回合评估
6. 汇总平均 reward、成功率、平均步数

---

## 3. 张量与数据流

每回合循环里关键数据：
- obs_n: [n_agents, obs_dim]
- last_onehot_a: [n_agents, action_dim]
- avail_a_n: [n_agents, action_dim]
- a_n: [n_agents]
- actions_cont: [n_agents, 4]

关键调用：
- 观测转矩阵 [_obs_dict_to_matrix](src/scripts/attention_qmix.py#L452)
- 动作选择 choose_action [src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L301)
- 离散到连续映射 discrete_to_continuous [src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L324)

---

## 4. 评估指标含义

输出摘要在 [evaluate.py](evaluate.py#L244)：
- Mean reward: 平均回报
- Success rate: 至少有一个 agent 到达目标的回合占比
- Mean steps: 平均回合长度

注意：
- 当前成功判据是任意 agent reached_goal 为 true，不是全体到达。

---

## 5. 常见坑

1. 评估 world 与训练 world 不一致
- 可能造成指标不可比。

2. 误把评估当训练
- evaluate 不会更新参数，始终推理模式。

3. 忘记 epsilon=0 的意义
- 这是去探索化评估，指标更稳定但不代表探索能力。

4. 缺少 config.json 时使用默认结构
- 如果训练结构与默认不同，评估可能偏差。

---

## 6. 推荐阅读顺序

1. [evaluate.py](evaluate.py#L114)
2. [src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L274)
3. [src/scripts/gazebo_pettingzoo_env.py](src/scripts/gazebo_pettingzoo_env.py#L552)
