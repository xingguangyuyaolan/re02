# train.py 源码导读（当前版本）

对应文件：`train.py`

更新时间：2026-04-20

## 1. 文件职责

`train.py` 是训练装配入口，负责：

1. 解析参数与配置
2. 构建 `QMIXConfig`
3. 初始化日志
4. 构建 Gazebo 多智能体环境
5. 启动 `train_attention_qmix`

## 2. 关键流程

1. 读取 `--config`（支持在 `configs/` 中解析）
2. 合并命令行参数与配置项
3. 处理 `resume` 逻辑
4. 组装环境参数和算法参数
5. 调用训练主循环

## 3. 重点更新点

### 3.1 跨阶段 run_name 解析

当前实现区分：

- 同阶段续训：复用原 run 目录
- 跨阶段迁移：创建新 run 目录

目的：避免 Stage2 写入 Stage1 目录。

### 3.2 训练配置覆盖

`train.py` 负责把配置中的关键字段完整传递到：

1. `QMIXConfig`
2. `GazeboMultiUAVParallelEnv`

包括碰撞终止策略、边界惩罚、高度约束、重置稳定参数等。

## 4. 调试建议

1. 看启动日志确认最终生效配置。
2. 重点核对 `run_name/output_root/resume` 三元组。
3. 对跨阶段训练，确认目录与 epsilon 是否符合预期。
