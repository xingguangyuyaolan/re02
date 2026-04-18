# 训练过程分析报告

**运行信息**：
- 世界：`maze_1.sdf`（迷宫场景）
- 完成进度：280/300 episode（中断原因：reset service超时）
- 训练时间：实际运行时间约2.5小时
- 总交互步数：87,142步

---

## 一、核心问题诊断

### 1. **任务学习失败** ⚠️⚠️⚠️

**现象**：
```
训练末期（Episode 280）：
  - coverage_rate = 0.0175 (仅1.75%覆盖率，目标85%)
  - collision_rate = 1.0 (100%碰撞)
  - reward = -145.98 (强烈负奖励)
  - full_coverage_success = False

评估结果（10个episode）：
  - 全部失败，平均覆盖率 1.5%
  - 100%碰撞，平均重叠率 80%+
```

**根本原因分析**：

#### 问题1.1：迷宫复杂度过高 
```
stage_1_open (开放环境) → 成功范例（你之前跑过）
maze_1.sdf (含障碍迷宫) → 难度跳跃式增加
```
- **开放环境**：无障碍，UAV可自由飞行，本质上是"空旷环境探索"
- **迷宫环境**：狭窄走廊、死胡同、障碍物，需要**路径规划**能力
- **学习曲线**：没有过渡阶段
  - Episode 1-8：buffer未满，无训练，随机探索即碰撞
  - Episode 9开始学习，但环境太复杂，agent无法找到有效策略

#### 问题1.2：奖励函数与任务不匹配
```
Coverage Reward: +1.2 (访问新格子)
Revisit Penalty: -0.08 (重复访问)
Collision Penalty: -1.5
Step Penalty: -0.01

每个episode ~193步（到Episode 280）×4个agent = 772个动作
如果100%碰撞：772 × (-1.5) = -1158（仅碰撞惩罚）
加上step_penalty：772 × (-0.01) = -7.72
实际reward ≈ -150 左右
```

**问题**：
- 碰撞惩罚（-1.5）远高于覆盖奖励（+1.2/cell）
- 在迷宫中，随机探索必然持续碰撞 → 训练信号压倒性负向
- Agent学到的唯一有效"策略"是**避免移动**（不动=无碰撞）

#### 问题1.3：无人机在迷宫中无法有效导航
```
LIDAR观测：36个激光束，扫描范围有限
Observation：[x, y, z, yaw, vx, vy, vz, yaw_rate] + 36个激光 + 5个coverage特征
= 56维观察空间

问题：
- 激光扫描在密集迷宫中容易过载（太多障碍）
- RNN hidden state (64维) + Attention mechanism 可能不足以建立"我在迷宫的哪个位置"的认知
- 没有显式的位置保持或边界检测
```

---

### 2. **超参数与训练配置不适配**

| 参数 | 当前值 | 诊断 |
|------|--------|------|
| `batch_size` | 8 | ✓ 合理（小batch快速学习） |
| `lr` | 0.0005 | △ 可能过高（loss波动大） |
| `epsilon_decay_steps` | 30000 | ✗ **太快衰减**（5000步后ε<0.5） |
| `training_interval` | 每3步 | △ 可接受但不够频繁 |
| `coverage_target_ratio` | 0.85 | ✗ **对maze_1过高**（应0.2-0.4） |
| `coverage_cell_size` | 1.0m | △ 对迷宫尺寸可能过大 |

**Learning Rate 问题**：
```python
Loss曲线（Episode 9-50）：
  - Episode 9: loss=218.84 → 5.96
  - Episode 16: loss=1177.76（尖峰）
  - Episode 31: loss=4053.46（极端尖峰）
  - Episode 50: loss=2341.59（仍在波动）

原因：lr=0.0005太高 + Q-value在迷宫中尺度差异大
→ 梯度更新剧烈 → loss不稳定
```

**Epsilon衰减太快**：
```
epsilon_decay_steps=30000
decay_per_step = (1.0 - 0.05) / 30000 = 0.0000316667

按照"每3步训练一次"：
- Episode 9 (train_step 1): ε ≈ 1.0
- Episode 100 (train_step ~1500): ε ≈ 0.5
- Episode 200 (train_step ~3000): ε ≈ 0.1

在未学到任何有效策略的情况下，epsilon快速降低，导致exploration不足
```

**Coverage目标过高**：
```
maze_1.sdf：迷宫世界，即使最优策略也难以达成85%覆盖
- 应该改为：0.2-0.4（初期目标先是"存活+有限探索"）
- 分阶段目标：阶段1→20%, 阶段2→40%, 阶段3→85%
```

---

### 3. **环境稳定性问题**

**症状**：Episode 280时reset service超时导致训练中断

```
原因链：
1. maze_1.sdf 场景复杂（多障碍）
2. 无人机频繁碰撞 → 状态不稳定
3. Reset需要将4个UAV还原到初始位置 → 约需2-5秒
4. 长时间运行（2.5小时+），ROS/Gazebo服务累积延迟
5. reset_timeout=2.0s不足 → 服务调用超时

当前配置已修复：
  ✓ reset_timeout: 5.0 (增加到5秒)
  ✓ 服务动态发现 + 分级重试
  ✓ Reset失败自动二次尝试
```

---

### 4. **Loss波动失控** ⚠️

观测：Episode 31 loss=4053.46, Episode 36 loss=5000+

```
原因分析：

在迷宫中，Q-value估计极度不稳定：
1. Replay buffer中的轨迹都是"碰撞轨迹"
2. TD-target = r + γ * Q_target(s', a')
   - r 大部分是 -1.5 (碰撞) 或 -0.01 (step)
   - Q_target(s') 完全未学到，初始随机
3. 当某条轨迹偶然有低碰撞序列时，Q估计会剧烈变化
4. QMIX mixing network放大了这种不稳定性
   （因为mixing权重对所有agent的贡献做了非线性组合）

症状：
- Loss从几十到几千，表现为"梯度崩溃"
- nan_skip_count=0（还没到NaN，但接近）
```

---

## 二、改进措施（优先级排序）

### **P0（必做）：改变难度进阶**

**方案**：不直接用maze_1.sdf训练，改用分阶段策略

```json
{
  "阶段1": {
    "世界": "stage_1_open.sdf",
    "coverage_target_ratio": 0.85,
    "episodes": 100
  },
  "阶段2": {
    "世界": "stage_2_easy.sdf",        // 轻度障碍
    "coverage_target_ratio": 0.6,
    "episodes": 100,
    "init_checkpoint": "stage_1_best"  // 从stage_1最优模型初始化
  },
  "阶段3": {
    "世界": "maze_1.sdf",
    "coverage_target_ratio": 0.4,      // 目标大幅降低
    "episodes": 200,
    "init_checkpoint": "stage_2_best"
  }
}
```

**预期效果**：
- Stage 1：验证算法在简单环境有效 ✓ (已做)
- Stage 2：学到障碍回避 + 部分覆盖
- Stage 3：在迷宫中实现limited coverage

---

### **P1（重要）：调整超参数**

#### 1. 降低 Learning Rate
```json
"lr": 0.0001         // 降低10倍（从0.0005）
"grad_clip_norm": 5.0  // 更激进的梯度剪裁（从10）
"td_clip": 500        // 限制TD误差尺度（从1000）
```

#### 2. 放慢Epsilon衰减
```json
"epsilon_decay_steps": 100000  // 增加3倍（从30000）
// 这样即使在100000步训练也不会完全关闭exploration
```

#### 3. 调整覆盖奖励比例
```json
"coverage_new_cell_reward": 2.0,      // +67%（从1.2）
"revisit_cell_penalty": 0.02,         // -75%（从0.08，减少惩罚）
"overlap_cell_penalty": 0.05,         // -67%（从0.15）
"out_of_bounds_penalty": 1.0          // -50%（从2.0）
```

**目的**：相对提高exploration奖励，减少碰撞作为主导信号

#### 4. 减少Step Penalty
```json
"step_penalty": 0.001   // -90%（从0.01，减少时间压力）
```

---

### **P2（优化）：改进Environment设计**

#### 1. 引入"导向性观测"
```python
# 在gazebo_pettingzoo_env.py中：
# 新增特征：
{
  "wall_distance": 最近墙壁距离,
  "corridor_width": 当前走廊宽度,
  "visited_cells_nearby": 附近已覆盖的格子比例
}
```

#### 2. 激进的碰撞超时
```json
"min_height_enforce_steps": 20,    // 更早执行高度检查
"collision_lidar_threshold": 0.3,  // 从0.25增大（更早检测）
"startup_grace_steps": 5           // 减少起始宽度（从12）
```

#### 3. 覆盖网格适配迷宫
```json
"coverage_cell_size": 0.5,  // 更细的网格（从1.0），捕捉更多细节
"local_coverage_radius": 2   // 增加局部视野（从1）
```

---

### **P3（架构优化）：考虑RNN+Attention容量**

当前：
- RNN hidden: 64维
- Attention: 对4个agent的Q值做attention

可考虑：
```python
"rnn_hidden_dim": 128,          // 翻倍（从64）
"qmix_hidden_dim": 64,           // 增加（从32）
"hyper_hidden_dim": 128,         // 增加（从64）
```

**理由**：迷宫场景需要更强的状态记忆能力

---

### **P4（故障排查）：监控与诊断**

在train.py中添加诊断日志：

```python
if episode % 10 == 0:
    LOGGER.info(
        f"[Diagnosis] Episode {episode}: "
        f"avg_loss={np.mean(recent_losses):.2f}, "
        f"avg_reward={np.mean(recent_rewards):.2f}, "
        f"collision_rate={collision_rate:.3f}, "
        f"coverage_rate={coverage_rate:.4f}, "
        f"q_eval_max={q_eval_max:.2f}, "
        f"q_target_max={q_target_max:.2f}, "
        f"buffer_size={trainer.replay_buffer.current_size}/{cfg.batch_size}"
    )
```

---

## 三、问题排查清单

- [ ] **验证LIDAR是否在迷宫中正常工作** → 检查早期episode的激光扫描数据
- [ ] **确认UAV初始位置** → maze_1.sdf的spawn_point是否与stage_1相同
- [ ] **检查碰撞判定边界** → collision_lidar_threshold=0.25是否过松
- [ ] **监测ROS bridge延迟** → 见 `/tmp/gz_bridges.log` 中的内存/CPU使用

---

## 四、立即可执行的修复（15分钟）

1. **新建 [configs/maze_1_train_v2.json](configs/maze_1_train_v2.json)**：
   ```json
   {
     "world_name": "maze_1.sdf",
     "coverage_target_ratio": 0.2,      // 从0.85降到0.2
     "lr": 0.0001,                      // 从0.0005降到0.0001
     "epsilon_decay_steps": 100000,     // 从30000增到100000
     "coverage_new_cell_reward": 2.0,   // 从1.2增到2.0
     "step_penalty": 0.001,             // 从0.01降到0.001
     "reset_timeout": 5.0,
     "coverage_cell_size": 0.5,         // 从1.0降到0.5
     "episodes": 200
   }
   ```

2. **重跑训练**：
   ```bash
   bash src/scripts/train_headless.sh src/worlds/mazes/maze_1.sdf configs/maze_1_train_v2.json
   ```

3. **监控输出**：
   ```bash
   tail -f artifacts/qmix/*/logs/runtime.log | grep -E "Episode|coverage_rate|collision"
   ```

---

## 五、总结

| 问题 | 严重性 | 根本原因 | 修复难度 |
|------|--------|---------|---------|
| 任务失败 | 极高 | 迷宫过难 + 奖励失衡 | 低（改参数） |
| Loss波动 | 高 | LR过高 + 状态空间复杂 | 低（降LR） |
| 环境不稳定 | 中 | Reset timeout不足 | 已修复 |
| 学习缓慢 | 中 | Epsilon一次衰减 | 低（调参数） |

**核心建议**：不是算法有问题，而是**任务难度与学习阶段不匹配**。按P0→P1→P2顺序逐步执行改进。
