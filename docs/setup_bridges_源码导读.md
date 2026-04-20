# setup_bridges.sh 源码导读（当前版本）

对应文件：

- 顶层入口：`setup_bridges.sh`
- 实际实现：`src/scripts/setup_bridges.sh`

更新时间：2026-04-20

## 1. 文件职责

建立 ROS2 与 Gazebo 的话题桥接，使训练环境可以：

1. 发布 UAV 控制命令
2. 接收里程计与激光数据

## 2. 脚本结构

1. 顶层脚本仅做路径转发
2. 实际脚本按 UAV 列表循环启动桥接
3. 使用 `trap` 在退出时清理子进程
4. 使用 `wait` 保持脚本常驻

## 3. 桥接类型

每个 UAV 通常桥接：

1. `cmd_vel`（ROS -> Gazebo）
2. `odom`（Gazebo -> ROS）
3. `lidar`（Gazebo -> ROS）

## 4. 常见问题

1. 桥接脚本未常驻，导致训练读不到传感器。
2. 部分 UAV 桥接失败，导致个别 agent 长期异常。
3. 旧进程未清理，重启后桥接状态异常。

## 5. 验收建议

1. 训练前确认相关 topic 存在。
2. reset 后能快速收到 odom/lidar。
3. 环境不再出现“无传感数据”早停。
