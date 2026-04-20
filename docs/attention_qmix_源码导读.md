# attention_qmix.py 源码导读（当前版本）

对应文件：`src/scripts/attention_qmix.py`

更新时间：2026-04-20

## 1. 文件职责

该文件实现 Attention-QMIX 训练核心，包含：

1. 网络结构（个体 Q 网络 + 混合网络 + 可选注意力）
2. 回放缓存（按 episode 存储）
3. 训练器（动作选择、训练更新、保存恢复）
4. 训练主循环与指标记录

## 2. 核心模块

### 2.1 QMIXConfig

集中管理训练超参数、注意力开关、保存路径与早停参数。

### 2.2 QMIXForUAV

训练器主类，关键职责：

1. 构建/同步 eval 与 target 网络
2. 处理 epsilon-greedy 动作选择
3. 执行 train_step
4. 保存/加载 checkpoint

### 2.3 train_attention_qmix

主训练循环，负责：

1. reset/step 与数据采样
2. 触发训练更新
3. 写入 `metrics.jsonl`、`summary.json`
4. 输出训练图表

## 3. 注意力相关能力

当前支持：

1. 自注意力（agent 内特征聚焦）
2. 跨智能体注意力（协同关系建模）
3. 混合网络注意力（联合价值融合增强）

这些开关由配置文件控制，并可被评测脚本恢复。

## 4. 训练稳定性设计

包括但不限于：

1. 数值清洗与裁剪
2. 梯度裁剪
3. target 网络周期更新
4. NaN 跳过计数
5. 训练诊断指标输出

## 5. 恢复逻辑说明

当前版本显式区分：

1. 同阶段 resume：恢复完整训练状态
2. 跨阶段迁移：加载权重，重置 episode/step/epsilon

这是课程学习正确衔接的关键。
