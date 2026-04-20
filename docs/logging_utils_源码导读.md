# logging_utils.py 源码导读（当前版本）

对应文件：`src/scripts/logging_utils.py`

更新时间：2026-04-20

## 1. 文件职责

统一项目日志体系，提供：

1. 运行日志（全量）
2. 错误日志（仅 ERROR+）
3. 控制台日志
4. 全局与线程异常落盘

## 2. 主要接口

`setup_project_logging(output_root, run_name, console_level=INFO)`

返回日志路径字典，供训练入口打印与记录。

## 3. 设计要点

1. 每次初始化会清理旧 handler，避免重复输出。
2. 将未捕获异常统一写入错误日志。
3. 训练/环境/算法模块共用一套 root logger。

## 4. 作用价值

1. 长时训练可回溯。
2. 多模块问题能在同一时间轴定位。
3. 适合配合 `metrics.jsonl` 做故障复盘。
