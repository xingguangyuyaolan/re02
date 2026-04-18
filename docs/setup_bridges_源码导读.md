# setup_bridges.sh 源码导读

对应源码：
- 顶层包装脚本 [setup_bridges.sh](setup_bridges.sh)
- 实际桥接脚本 [src/scripts/setup_bridges.sh](src/scripts/setup_bridges.sh)

作用定位：
- 建立 ROS2 与 Gazebo 之间的 topic bridge。
- 让训练环境能发布 cmd_vel，并接收 odom 与 lidar。

---

## 1. 顶层脚本作用

文件：
- [setup_bridges.sh](setup_bridges.sh)

逻辑：
1. set -e（任意命令失败即退出）
2. 解析脚本所在目录
3. exec 到真实脚本 [src/scripts/setup_bridges.sh](src/scripts/setup_bridges.sh)

价值：
- 不论你在项目根目录哪个位置执行，都能正确跳转到真实脚本。

---

## 2. 实际桥接脚本流程

文件：
- [src/scripts/setup_bridges.sh](src/scripts/setup_bridges.sh)

关键结构：
- UAV_NAMES 数组 [src/scripts/setup_bridges.sh](src/scripts/setup_bridges.sh#L3)
- 循环启动每个 UAV 的三类桥接 [src/scripts/setup_bridges.sh](src/scripts/setup_bridges.sh#L6)
- trap 捕获退出信号并清理子进程 [src/scripts/setup_bridges.sh](src/scripts/setup_bridges.sh#L31)
- wait 阻塞保持脚本常驻 [src/scripts/setup_bridges.sh](src/scripts/setup_bridges.sh#L33)

---

## 3. 三类桥接逐条解释

### 3.1 cmd_vel 桥接
- 位置：[src/scripts/setup_bridges.sh](src/scripts/setup_bridges.sh#L8)
- 方向：ROS2 -> Gazebo
- 作用：训练器发布速度命令，驱动 UAV 运动

### 3.2 odom 桥接
- 位置：[src/scripts/setup_bridges.sh](src/scripts/setup_bridges.sh#L14)
- 方向：Gazebo -> ROS2
- 作用：把 /model/uavX/odometry 重映射到 /uavX/odom

### 3.3 lidar 桥接
- 位置：[src/scripts/setup_bridges.sh](src/scripts/setup_bridges.sh#L20)
- 方向：Gazebo -> ROS2
- 作用：提供障碍距离观测

---

## 4. 与环境代码的接口对齐

环境订阅与发布约定在：
- 发布 /uavX/cmd_vel [src/scripts/gazebo_pettingzoo_env.py](src/scripts/gazebo_pettingzoo_env.py#L185)
- 订阅 /uavX/odom [src/scripts/gazebo_pettingzoo_env.py](src/scripts/gazebo_pettingzoo_env.py#L188)
- 订阅 /uavX/lidar [src/scripts/gazebo_pettingzoo_env.py](src/scripts/gazebo_pettingzoo_env.py#L191)

只要 bridge 脚本跑通，环境即可在 reset/step 中拿到有效传感数据。

---

## 5. 常见坑

1. 脚本没有常驻
- 现象：训练刚开始就提示缺 topic 或传感全空。

2. 只起了部分 UAV 桥接
- 现象：某些 agent 无 odom/lidar，训练早停或奖励异常。

3. 中断时未清理桥接进程
- 现象：重启桥接后端口或连接状态异常。
- 当前脚本已用 trap + kill 做清理。

4. topic 命名不一致
- 现象：环境检查 topic 失败。
- 需确保桥接命名与环境订阅路径完全一致。

---

## 6. 验收标准

桥接成功后，训练前应满足：
1. 环境 topic 检查通过
2. reset 后能收到 odom/lidar
3. metrics 中 benchmark_env_step_time_sec 正常且非零

若任一失败，优先回看本脚本与环境订阅路径是否对齐。
