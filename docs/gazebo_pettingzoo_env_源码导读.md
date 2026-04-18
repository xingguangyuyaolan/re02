# gazebo_pettingzoo_env.py 源码导读

对应源码：
- [src/scripts/gazebo_pettingzoo_env.py](src/scripts/gazebo_pettingzoo_env.py)

作用定位：
- 把 Gazebo + ROS2 多无人机仿真封装成 PettingZoo ParallelEnv。
- 为训练器提供标准 reset/step 接口。

---

## 1. 结构总览

入口类：
- [GazeboMultiUAVParallelEnv](src/scripts/gazebo_pettingzoo_env.py#L48)

核心函数：
- 初始化 [src/scripts/gazebo_pettingzoo_env.py](src/scripts/gazebo_pettingzoo_env.py#L51)
- reset [src/scripts/gazebo_pettingzoo_env.py](src/scripts/gazebo_pettingzoo_env.py#L495)
- step [src/scripts/gazebo_pettingzoo_env.py](src/scripts/gazebo_pettingzoo_env.py#L552)
- 工厂函数 make_env [src/scripts/gazebo_pettingzoo_env.py](src/scripts/gazebo_pettingzoo_env.py#L646)

---

## 2. 初始化阶段（__init__）

位置：
- [src/scripts/gazebo_pettingzoo_env.py](src/scripts/gazebo_pettingzoo_env.py#L51)

输入：
- uav_names, lidar_size
- 动作边界 max_xy_vel/max_z_vel/max_yaw_rate
- 回合控制 max_steps
- 奖励参数 goal_xyz/goal_radius/step_penalty/progress_reward_scale
- 安全参数 min_height/max_height/collision_lidar_threshold
- reset 稳定化参数 reset_timeout/reset_position_tolerance 等

输出：
- 环境实例（副作用：创建 ROS2 节点、订阅与发布器）

关键内部状态：
- action_spaces: 每个 agent 一个 Box(4,)
- observation_spaces: 每个 agent 一个 Dict
  - pose: (8,)
  - lidar: (lidar_size,)
- 缓存：_last_odom, _last_lidar, _prev_goal_dist, _dones

常见坑：
- world_name 与实际 Gazebo world 不一致会导致 reset 服务失败。
- lidar_size 与桥接消息维度不一致时会触发截断或补齐逻辑。

---

## 3. ROS 通信与数据接收

回调函数：
- odom 回调 [_make_odom_cb](src/scripts/gazebo_pettingzoo_env.py#L193)
- lidar 回调 [_make_lidar_cb](src/scripts/gazebo_pettingzoo_env.py#L199)

自旋函数：
- [_spin_once](src/scripts/gazebo_pettingzoo_env.py#L205)
- [_spin_for](src/scripts/gazebo_pettingzoo_env.py#L208)

目的：
- 从 ROS2 回调管线拉取最新传感数据，保障 step/reset 读到新观测。

常见坑：
- spin 时间太短会导致观测延迟，表现为策略动作与状态错位。

---

## 4. 可观测性与启动校验

关键函数：
- 话题检查 [_check_required_topics](src/scripts/gazebo_pettingzoo_env.py#L243)
- 传感可用判断 [_has_agent_sensor_data](src/scripts/gazebo_pettingzoo_env.py#L276)
- 等待传感数据 [_wait_for_sensor_data](src/scripts/gazebo_pettingzoo_env.py#L279)

作用：
- 在训练开始前确认 odom/lidar/cmd_vel 关键 topic 存在。
- 避免“训练在空数据上跑”。

---

## 5. 观测构建

核心函数：
- [_collect_observation](src/scripts/gazebo_pettingzoo_env.py#L294)

输入：
- 内部缓存的 Odometry 和 LaserScan

输出：
- obs 字典：
  - obs[agent]["pose"] 形状 (8,)
  - obs[agent]["lidar"] 形状 (lidar_size,)

处理细节：
- 从四元数转 yaw
- lidar 的 NaN/Inf 处理
- lidar 长度不匹配时自动补齐或裁剪

常见坑：
- 如果桥接没起来，pose 会是零向量、lidar 会是 inf 填充。

---

## 6. reset 机制（最关键）

主入口：
- [reset](src/scripts/gazebo_pettingzoo_env.py#L495)

关联函数：
- [_reset_simulation](src/scripts/gazebo_pettingzoo_env.py#L416)
- [_reset_simulation_via_gz](src/scripts/gazebo_pettingzoo_env.py#L439)
- [_wait_for_reset_stabilization](src/scripts/gazebo_pettingzoo_env.py#L383)
- [_validate_reset_positions](src/scripts/gazebo_pettingzoo_env.py#L406)

流程：
1. 先发零速度，抑制残余运动
2. 调用 world reset（ROS service 优先，失败回退 gz service）
3. 清空旧传感缓存
4. 等待新传感到达
5. 检查 reset 后漂移是否超阈值
6. 初始化每个 agent 的前一时刻目标距离

常见坑：
- reset 失败会直接抛 RuntimeError，中断训练，避免旧状态污染新回合。

---

## 7. step 机制与奖励设计

主入口：
- [step](src/scripts/gazebo_pettingzoo_env.py#L552)

输入：
- actions 字典，actions[agent] 形状 (4,)

输出：
- obs, rewards, dones, infos
- 其中 dones 额外包含 __all__

奖励组成：
- 进度奖励：按距离目标变近程度给分
- 步长惩罚：每步固定扣分
- 碰撞惩罚：碰撞额外扣分
- 越界惩罚：超出 arena 扣分
- 到达奖励：到达目标给额外正奖

终止条件：
- 单体 done: 碰撞或到达目标或越界
- 回合 done: 全体 done 或达到 max_steps

info 关键字段：
- distance_to_goal
- reached_goal, collided, out_of_bounds
- startup_grace_active
- altitude_violation

常见坑：
- startup_grace_steps 太小会导致起步抖动被误判碰撞。
- min_height_enforce_steps 太小会导致起飞阶段频繁触发低空违规。

---

## 8. 渲染与关闭

- render [src/scripts/gazebo_pettingzoo_env.py](src/scripts/gazebo_pettingzoo_env.py#L636)
- close [src/scripts/gazebo_pettingzoo_env.py](src/scripts/gazebo_pettingzoo_env.py#L640)

说明：
- 渲染由 Gazebo 自身负责。
- close 负责销毁 ROS2 节点。

---

## 9. 你最该关注的排查点

1. reset 报错
- 优先看 [src/scripts/gazebo_pettingzoo_env.py](src/scripts/gazebo_pettingzoo_env.py#L495)

2. 观测异常
- 看 [_collect_observation](src/scripts/gazebo_pettingzoo_env.py#L294)

3. 奖励不合理
- 看 [step](src/scripts/gazebo_pettingzoo_env.py#L552)

4. 训练早停或高负奖励
- 先排除 spawn 风险告警与 topic 不全问题

---

## 10. 推荐阅读顺序

1. [src/scripts/gazebo_pettingzoo_env.py](src/scripts/gazebo_pettingzoo_env.py#L51)
2. [src/scripts/gazebo_pettingzoo_env.py](src/scripts/gazebo_pettingzoo_env.py#L495)
3. [src/scripts/gazebo_pettingzoo_env.py](src/scripts/gazebo_pettingzoo_env.py#L552)
4. [src/scripts/attention_qmix.py](src/scripts/attention_qmix.py#L506)
