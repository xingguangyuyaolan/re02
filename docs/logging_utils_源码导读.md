# logging_utils.py 源码导读

对应源码：
- [src/scripts/logging_utils.py](src/scripts/logging_utils.py)

作用定位：
- 统一训练过程日志输出。
- 同时写 runtime 全量日志与 errors 错误日志，并接管全局异常钩子。

---

## 1. 函数总览

- [_build_formatter](src/scripts/logging_utils.py#L10)
- [setup_project_logging](src/scripts/logging_utils.py#L17)

---

## 2. _build_formatter

位置：
- [src/scripts/logging_utils.py](src/scripts/logging_utils.py#L10)

输入：
- 无

输出：
- logging.Formatter

作用：
- 统一日志格式：时间、级别、logger 名称、消息文本。

---

## 3. setup_project_logging

位置：
- [src/scripts/logging_utils.py](src/scripts/logging_utils.py#L17)

输入：
- output_root
- run_name
- console_level

输出：
- 包含 run_dir/logs_dir/runtime_log_path/error_log_path 的字典

分段逻辑：
1. 计算目录并创建 logs 目录
2. 获取 root logger 并设为 DEBUG
3. 清空旧 handler，避免重复打印
4. 添加 runtime 文件 handler（DEBUG 全量）
5. 添加 error 文件 handler（ERROR 及以上）
6. 添加控制台 handler（默认 INFO）
7. 注册 sys.excepthook 捕获未处理异常
8. 注册 threading.excepthook 捕获线程异常
9. 打印 logging initialized 启动日志并返回路径

常见坑：
- 多次调用若不清空旧 handler，会出现日志重复输出；代码已处理。
- 若只看控制台可能漏掉 DEBUG 细节，深度排查要看 runtime.log。

---

## 4. 异常捕获设计

全局异常：
- 由 sys.excepthook 捕获，记录完整堆栈。

线程异常：
- 由 threading.excepthook 捕获，记录线程名与堆栈。

价值：
- 避免训练中出现“终端看到了异常，但日志文件里没有”的情况。

---

## 5. 与训练模块的关系

调用位置：
- [train.py](train.py#L136)

运行效果：
- 训练和环境模块都写到同一套日志体系，便于跨模块排查。

配合阅读：
- [docs/训练日志诊断手册.md](docs/训练日志诊断手册.md)
