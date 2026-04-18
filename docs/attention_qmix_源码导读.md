# Attention-QMIX 源码导读（按函数逐段）

对应源码：
- [src/scripts/attention_qmix.py](src/scripts/attention_qmix.py)

本文目标：
- 让你可以独立读懂训练全流程
- 看到每个函数就知道输入、输出、张量形状
- 提前规避最常见的训练坑与数值坑

---

## 1. 全局调用链（先建立脑图）

训练入口最终会调用 [train_attention_qmix](src/scripts/attention_qmix.py#L506)。

核心链路：
1. 环境返回观测字典 obs_dict
2. [_obs_dict_to_matrix](src/scripts/attention_qmix.py#L452) 转成 obs_n
3. [_state_from_obs_matrix](src/scripts/attention_qmix.py#L456) 转成全局状态 s
4. [QMIXForUAV.choose_action](src/scripts/attention_qmix.py#L301) 选离散动作 a_n
5. [QMIXForUAV.discrete_to_continuous](src/scripts/attention_qmix.py#L324) 映射连续控制
6. 环境 step 后把 transition 存入 [EpisodeReplayBuffer.store_transition](src/scripts/attention_qmix.py#L139)
7. 条件满足时调用 [QMIXForUAV.train_step](src/scripts/attention_qmix.py#L337)
8. 回合结束补最后时刻到 [EpisodeReplayBuffer.store_last_step](src/scripts/attention_qmix.py#L150)
9. 周期保存模型与日志

---

## 2. 基础工具函数

### 2.1 _resolve_compute_device
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L20)
- 输入：device_name，字符串，支持 auto/cpu/cuda
- 输出：torch.device
- 作用：
  - auto: 自动探测 CUDA 可用性
  - cuda: 请求 CUDA，若探测失败自动降级到 CPU
  - cpu: 强制 CPU
- 常见坑：
  - 显卡驱动或 CUDA 版本不兼容会在首个 CUDA 张量分配时崩，此函数已用 probe 防止“配置看起来可用但运行崩溃”。

### 2.2 orthogonal_init
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L45)
- 输入：layer, gain
- 输出：无（原地初始化）
- 作用：线性层或 GRU 参数做正交初始化
- 常见坑：
  - 如果你替换网络结构，新增层未初始化，初期训练方差会更大。

---

## 3. 个体价值网络 QNetworkRNN

### 3.1 __init__
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L54)
- 输入：
  - input_dim: 每个 agent 的输入维度
  - hidden_dim: GRU 隐状态维度
  - action_dim: 离散动作数
- 输出：网络实例
- 内部模块：
  - fc1: Linear(input_dim, hidden_dim)
  - rnn: GRUCell(hidden_dim, hidden_dim)
  - fc2: Linear(hidden_dim, action_dim)

### 3.2 forward
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L65)
- 输入：inputs，形状通常是 [B*n_agents, input_dim] 或 [n_agents, input_dim]
- 输出：Q 值，形状 [B*n_agents, action_dim] 或 [n_agents, action_dim]
- 状态：self.rnn_hidden 会在时间展开中持续更新
- 常见坑：
  - rnn_hidden 与输入 batch 维不匹配会报错，代码在 choose_action 里做了保护重置。
  - 训练和执行共用同一个 eval_q，若不保存恢复 hidden，会污染在线执行时序；本项目在 train_step 已保护。

---

## 4. 混合网络 QMixNet

### 4.1 __init__
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L72)
- 输入：
  - n_agents
  - state_dim
  - batch_size
  - qmix_hidden_dim
  - hyper_hidden_dim
  - hyper_layers_num
- 输出：网络实例
- 关键点：
  - 超网络根据全局状态 s 生成 w1/w2
  - 对 w1/w2 取 abs，保证单调性约束（各 agent 局部 Q 增大不会使 Q_tot 下降）

### 4.2 forward
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L101)
- 输入：
  - q: 形状 [B, T, n_agents]
  - s: 形状 [B, T, state_dim]
- 输出：q_total，形状 [B, T, 1]
- 关键中间形状：
  - q reshape -> [-1, 1, n_agents]
  - w1 -> [-1, n_agents, qmix_hidden_dim]
  - hidden -> [-1, 1, qmix_hidden_dim]
  - w2 -> [-1, qmix_hidden_dim, 1]
  - q_total -> [B, T, 1]
- 常见坑：
  - 该实现依赖 config.batch_size 进行最终 view，训练时 sample 的 batch 大小必须固定等于配置值。

---

## 5. 回放缓存 EpisodeReplayBuffer

### 5.1 __init__
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L116)
- 输入：n_agents, obs_dim, state_dim, action_dim, episode_limit, buffer_size, batch_size
- 输出：缓存实例
- 内部数据布局：
  - obs_n: [buffer_size, episode_limit+1, n_agents, obs_dim]
  - s: [buffer_size, episode_limit+1, state_dim]
  - avail_a_n: [buffer_size, episode_limit+1, n_agents, action_dim]
  - last_onehot_a_n: [buffer_size, episode_limit+1, n_agents, action_dim]
  - a_n: [buffer_size, episode_limit, n_agents]
  - r: [buffer_size, episode_limit, 1]
  - dw: [buffer_size, episode_limit, 1]
  - active: [buffer_size, episode_limit, 1]

### 5.2 store_transition
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L139)
- 输入：单步 transition
- 输出：无
- 关键点：last_onehot_a_n 写在 episode_step+1 位置，符合 RNN 输入“下一时刻使用上一动作”语义。

### 5.3 store_last_step
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L150)
- 输入：最终时刻 obs/state/avail
- 输出：无
- 关键点：写入 t+1 对齐信息，并记录 episode_len。

### 5.4 sample
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L159)
- 输入：无
- 输出：
  - batch: 字典张量
  - max_episode_len: 当前批次最大有效长度
- 形状约定：
  - 序列类键返回 [B, max_len(+1), ...]
  - 动作/奖励等返回 [B, max_len, ...]
- 常见坑：
  - 如果 current_size < batch_size，外部必须跳过训练；本项目在 train_step 入口已处理。

---

## 6. 配置结构 QMIXConfig

- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L174)
- 作用：统一管理训练超参数、稳定性参数、保存参数
- 最关键字段：
  - 强化学习：gamma, epsilon, epsilon_min, epsilon_decay_steps
  - 训练：batch_size, lr, target_update_freq
  - 稳定性：obs_clip, q_clip, td_clip, grad_clip_norm
  - 工程：output_root, run_name, checkpoint_interval, resume_path, device
- 常见坑：
  - epsilon_decay_steps 过小会过早贪心，卡在次优策略。
  - q_clip/td_clip 太小会抑制学习，太大又可能数值爆炸。

---

## 7. 训练器 QMIXForUAV

### 7.1 __init__
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L211)
- 输入：n_agents, obs_dim, state_dim, action_table, config
- 输出：训练器实例
- 内部创建：
  - eval_q / target_q
  - eval_mix / target_mix
  - optimizer
  - replay_buffer
- 输入维拼接规则：
  - 基础 obs_dim
  - 若 add_last_action=True，额外 + action_dim
  - 若 add_agent_id=True，额外 + n_agents
- 当前项目默认维度：
  - obs_dim = 44
  - action_dim = 13
  - n_agents = 4
  - input_dim = 44 + 13 + 4 = 61
- 常见坑：
  - 改动 add_last_action/add_agent_id 后，旧 checkpoint 可能无法直接加载。

### 7.2 build_checkpoint
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L258)
- 输入：episode, total_steps, best_reward
- 输出：可被 torch.save 的字典
- 包含内容：网络参数、优化器状态、epsilon、统计量、配置快照。

### 7.3 load_checkpoint
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L274)
- 输入：checkpoint_path
- 输出：episode/total_steps/best_reward 元信息
- 常见坑：
  - 仅加载权重但不加载 optimizer 会改变恢复训练轨迹；本函数已支持 optimizer 恢复。

### 7.4 _sanitize_tensor
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L295)
- 输入：tensor, clip_abs
- 输出：清洗后 tensor
- 作用：处理 NaN/Inf 并可选裁剪，避免损失非有限。

### 7.5 choose_action
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L301)
- 输入：
  - obs_n: [n_agents, obs_dim]
  - last_onehot_a_n: [n_agents, action_dim]
  - avail_a_n: [n_agents, action_dim]
  - epsilon: 标量
- 输出：a_n，形状 [n_agents]，每个元素是离散动作索引
- 逻辑：
  - epsilon 概率随机动作
  - 否则前向 eval_q，mask 不可行动作后 argmax
- 常见坑：
  - avail_a_n 全 0 会导致随机采样失败；当前环境全部可行，所以默认全 1。
  - hidden 维度不一致时会重置，避免 batch size 变化导致 GRU 报错。

### 7.6 discrete_to_continuous
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L324)
- 输入：a_n, [n_agents]
- 输出：连续动作 [n_agents, 4]
- 作用：查表 action_table，把离散策略落地到速度指令。

### 7.7 get_inputs
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L327)
- 输入：
  - batch
  - max_episode_len
- 输出：拼接后输入，形状 [B, max_len+1, n_agents, input_dim]
- 常见坑：
  - agent_id 的 repeat 维度如果写错，会造成拼接失败或隐式广播错误。

### 7.8 train_step（全文件最关键）
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L337)
- 输入：total_steps
- 输出：loss(float) 或 None（跳过更新）

分段解释：
1. 采样与设备迁移
- 从 buffer 采样 batch，所有键搬到 device。

2. 输入清洗
- 对 obs/state/reward/input 做 sanitize 与 clip。

3. 时间展开前向
- t=0..T-1 逐时刻跑 eval_q 与 target_q。
- q_evals, q_targets 初始形状 [B, T, n_agents, action_dim]。

4. Double Q 目标动作选择
- 用 eval_q(next) 选 argmax 动作，用 target_q 取该动作价值。
- 得到 q_targets: [B, T, n_agents]。

5. 当前动作价值抽取
- 根据 a_n 从 q_evals gather 得 q_evals: [B, T, n_agents]。

6. QMIX 混合
- q_total_eval = eval_mix(q_evals, s_t) -> [B, T, 1]
- q_total_target = target_mix(q_targets, s_{t+1}) -> [B, T, 1]

7. TD 目标与损失
- targets = r + gamma*(1-dw)*q_total_target
- td_error = q_total_eval - targets
- masked_td_error = td_error * active
- loss = sum(masked_td_error^2) / sum(active)

8. 非有限保护与优化
- 非有限 loss 跳过
- 反向传播 + grad clip
- 非有限 grad 跳过

9. 目标网络软周期更新
- 每 target_update_freq 步将 eval 参数拷到 target。

10. epsilon 衰减与 hidden 恢复
- epsilon = max(epsilon_min, epsilon - decay)
- 恢复 acting_hidden，避免训练过程污染在线选择动作状态。

常见坑总表：
- 维度坑：B、T、n_agents、action_dim 的顺序错一位，训练会静默劣化。
- mask 坑：avail_a_n 时序错位会把 next 动作 mask 用到 current。
- 数值坑：不做 sanitize 时，激光异常值可快速引爆 TD。
- 状态坑：不恢复 acting_hidden 会让行为策略时间上下文错乱。

---

## 8. 观测与状态转换工具

### 8.1 _flatten_agent_obs
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L442)
- 输入：单 agent 字典，含 pose 与 lidar
- 输出：一维向量 [obs_dim]
- 当前默认 obs_dim = 8 + 36 = 44
- 常见坑：
  - lidar NaN/Inf 如果不替换，会直接把网络输出推到非有限。

### 8.2 _obs_dict_to_matrix
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L452)
- 输入：obs_dict, agents
- 输出：obs_n，形状 [n_agents, obs_dim]

### 8.3 _state_from_obs_matrix
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L456)
- 输入：obs_matrix [n_agents, obs_dim]
- 输出：全局状态 s [state_dim]
- 当前默认 state_dim = 4*44 = 176

---

## 9. I/O 与工件函数

### 9.1 _prepare_run_dirs
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L460)
- 输入：cfg
- 输出：路径字典（run_dir, checkpoint_dir, metrics_path 等）

### 9.2 _write_json / _append_jsonl / _read_jsonl
- 位置：
  - [src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L476)
  - [src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L481)
  - [src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L486)
- 作用：配置快照、逐回合指标日志、恢复历史

### 9.3 _save_checkpoint
- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L502)
- 作用：统一保存入口

---

## 10. 主训练函数 train_attention_qmix

- 位置：[src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L506)
- 输入：
  - env
  - n_episodes
  - max_steps
  - config
  - resume_path
- 输出：无（产出模型和日志文件）

逐段说明：
1. 从环境自动推导维度
- obs_dim 从空间结构推导
- n_agents 从 env.agents 推导
- state_dim = n_agents * obs_dim

2. 定义离散动作表 action_table
- 13 个动作，每个动作是 [vx, vy, vz, yaw_rate]

3. 构建 trainer 与输出目录
- 保存 config.json 快照

4. resume 逻辑
- 支持 latest 或指定 checkpoint
- 恢复 episode、step、best_reward

5. 回合循环
- reset 环境，准备 obs_n、s、last_onehot
- 每步：choose_action -> step -> 组装 team_reward -> store_transition
- 若缓存足够：train_step

6. 回合结束
- store_last_step
- 写 metrics.jsonl
- 保存 latest/best/checkpoint

7. 异常保护
- 中断或异常时也会在 finally 保存 latest/final 与 summary

8. summary 汇总
- 奖励统计、损失统计、benchmark 时间统计

常见坑：
- team_reward 用平均值，如果你未来做异质 agent，需要重新定义聚合方式。
- dw 的定义是 done 且非 max_steps 截断；如果你改终止逻辑，需同步检查 TD 目标是否正确。
- 长回合 + 高仿真耗时会导致每回合训练更新次数波动较大，注意观察 benchmark_train_updates。

---

## 11. 一页速查：关键张量形状

默认参数下：
- n_agents = 4
- obs_dim = 44
- state_dim = 176
- action_dim = 13
- input_dim = 61（44 + 13 + 4）
- batch_size = 32（示例配置）

执行时（choose_action）：
- obs_n: [4, 44]
- last_onehot_a_n: [4, 13]
- avail_a_n: [4, 13]
- q_value: [4, 13]
- a_n: [4]

训练时（train_step）：
- inputs: [32, T+1, 4, 61]
- q_evals(before gather): [32, T, 4, 13]
- q_targets(after double q gather): [32, T, 4]
- q_evals(after gather): [32, T, 4]
- q_total_eval: [32, T, 1]
- q_total_target: [32, T, 1]
- td_error: [32, T, 1]

---

## 12. 修改建议（按你后续最可能动的点）

1) 想让路径更稳：先改奖励
- 看 [src/scripts/gazebo_pettingzoo_env.py](src/scripts/gazebo_pettingzoo_env.py#L552)
- 先小幅调 step_penalty 和 progress_reward_scale，避免一次改太多项。

2) 想提速：先减仿真耗时
- 看 metrics.jsonl 里的 benchmark_env_step_time_sec。
- 一般仿真是瓶颈，非神经网络。

3) 想提升收敛稳定：优先改 epsilon_decay_steps、lr、grad_clip_norm
- 这三项对早中期学习曲线影响最大。

4) 想换算法结构：
- 最安全路径是保留环境和缓存接口，只替换 QMIXForUAV 内网络与 train_step。

---

## 13. 读码顺序建议（最省时间）

1. [src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L506) 先看主循环
2. [src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L301) 再看动作选择
3. [src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L337) 再看训练更新
4. [src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L71) 看混合网络
5. [src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L115) 最后看缓存细节

照这个顺序，你会从业务流程到数学细节逐层收敛，不会迷失在实现细节里。
