# 知识点 05：推理 PP 与训练 PP 的差异

> 核心问题：vLLM 是推理框架，它的 PP 跟训练 PP 有什么本质差异？

---

## 1. 核心差异总览

| 维度 | 训练 PP | 推理 PP |
|------|---------|---------|
| **Backward** | 有 | **无** |
| **调度** | micro-batch 填充 | **continuous batching** |
| **气泡处理** | 1F1B / Interleaved | **async scheduling** |
| **KV cache** | 不涉及 | **每个 stage 各自维护** |
| **通信内容** | activation + gradient | 只 activation |
| **请求生命周期** | 一个 epoch 内固定 | 动态进出 |
| **目标** | 训练吞吐 | 推理延迟 + 吞吐 |

---

## 2. 差异一：没有 Backward

### 训练场景

```
forward → loss → backward → 梯度更新
PP 需要协调 forward 和 backward 两个方向的数据流
```

### 推理场景

```
forward → logits → sample → token
PP 只有 forward 一个方向
```

**对 PP 实现的影响**：

- 没有 backward 调度复杂性（1F1B / Interleaved 1F1B 都失去意义）
- 没有 gradient 通信（PP 间只传 activation）
- 内存模型简化（不需要保留激活用于 backward）

---

## 3. 差异二：Continuous Batching

### 训练场景的 batch

训练一个 batch 是固定的：

```
epoch 开始 → batch_1 → batch_2 → ... → batch_N → epoch 结束
```

每个 batch 内 micro_batch 数量固定，PP 调度可以预先规划。

### 推理场景的 batch

推理请求是**流式到达**的：

```
时刻 t=0: 1 个请求
时刻 t=1: +2 个请求（共 3 个）
时刻 t=2: -1 个请求（完成 1 个，剩 2 个）
```

batch 在动态变化。

### Continuous Batching 思路

```
vLLM 的 PP 流水线持续运转：
- 每个 step，scheduler 从等待队列挑若干请求
- 把它们的 token 组成 micro_batch 喂给 stage 0
- stage 间流水线持续工作
- 已完成的请求退出，新请求加入
```

**对 PP 实现的影响**：

- 没有"一个 batch 跑完"的概念
- stage 必须持续接收新 micro_batch（不能停下来）
- scheduler 是核心：决定哪些请求参与下一步

### Continuous Batching 怎么消除气泡？

回忆 GPipe 推理场景的尾部气泡：

```
时间:        1    2    3    4    5    6    7
Stage 0:   |m1F|m2F|m3F|m4F| ★  | ★  | ★  |   ← 尾部气泡
Stage 1:   | ★ |m1F|m2F|m3F|m4F| ★  | ★  |
Stage 2:   | ★ | ★ |m1F|m2F|m3F|m4F| ★  |
Stage 3:   | ★ | ★ | ★ |m1F|m2F|m3F|m4F|
```

Continuous Batching 在 t=5 时立刻塞入新 micro_batch m5, m6, m7：

```
时间:        1    2    3    4    5    6    7
Stage 0:   |m1F|m2F|m3F|m4F|m5F|m6F|m7F|   ← 气泡被消除
Stage 1:   | ★ |m1F|m2F|m3F|m4F|m5F|m6F|
Stage 2:   | ★ | ★ |m1F|m2F|m3F|m4F|m5F|
Stage 3:   | ★ | ★ | ★ |m1F|m2F|m3F|m4F|
```

**关键**：只要有新请求，气泡就被消除。代价是冷启动延迟（首个 token）。

---

## 4. 差异三：KV Cache 各自维护

### 训练场景

训练时**没有 KV cache 概念**：每层 forward 重新计算 attention，不缓存。

### 推理场景

推理必须用 KV cache（避免重复计算历史 token 的 K/V）。

**KV cache 跟层绑定**：

```
Stage 0 维护：layer 0-15 的 KV cache
Stage 1 维护：layer 16-31 的 KV cache
...
```

**对 PP 通信的影响**：

- **KV cache 不在 stage 间传递**（它是 stage 私有的）
- PP 间只传 activation（hidden_states）
- 每个 stage 自己管理 KV cache 的扩容、驱逐、prefill/decode

**好处**：

- 通信量小
- KV cache 显存压力被均匀分摊到所有 stage

**坏处**：

- KV cache 的复杂逻辑要在每个 stage 都实现（vLLM 通过 ModelRunner 统一管理）

---

## 5. 差异四：Async Scheduling

### 训练场景的同步

训练每个 step 必须同步：
- forward 完 → 算 loss
- backward 完 → all-reduce 梯度
- 优化器更新参数

PP 各 stage 之间有强同步依赖。

### 推理场景的异步机会

推理没有梯度同步，可以异步：

```
传统 PP：
Stage 0 forward(t=1) → send → Stage 1 forward(t=1) → ...
所有 stage 在同一 step 内强同步

Async PP：
Stage 0 forward(t=1) → send → Stage 1 forward(t=1)
                              ↓ Stage 0 同时开始 forward(t=2)
Stage 0 不等 Stage 1 完成，连续处理下一批
```

**Async Scheduling 的本质**：每个 stage 像独立的工人，stage 间通过队列连接，不互相等待。

**对 PP 实现的影响**：

- `IntermediateTensors` 不再阻塞等待
- 用 double buffering 隐藏通信延迟
- scheduler 可以更激进地塞请求

vLLM 的 async-scheduling（#39074 系列）就是为此设计。

---

## 6. 差异五：请求生命周期

### 训练场景

所有请求（数据集样本）的生命周期一致：训练开始到训练结束。

### 推理场景

每个请求有独立的生命周期：

```
请求 A: prefill → decode → decode → ... → 完成
请求 B:           prefill → decode → ... → 完成
请求 C:                       prefill → ... → 完成
```

不同请求可能处于不同阶段（prefill / decode / finished）。

**对 PP 的影响**：

- 每个 stage 必须维护所有正在处理的请求的状态
- KV cache 要按请求维度管理
- scheduler 要协调"哪些 stage 有哪些请求"

---

## 7. 推理 PP 的特殊挑战

基于以上差异，推理 PP 有几个训练 PP 不会遇到的挑战：

### 7.1 跨 stage 状态同步

```
当请求 A 在 Stage 3 完成（生成 EOS）：
- Stage 3 知道 A 完成了
- 但 Stage 0/1/2 还在处理 A 的某些 token
- 怎么通知它们停止？

vLLM 的方案：通过 scheduler 协调，每个 step 调度时告知所有 stage
```

### 7.2 MTP（推测解码）的状态广播

```
MTP 在 last rank 生成 draft token
但其他 rank 也需要知道哪些 token 被 accept
怎么把 last rank 的状态广播给其他 rank？

vLLM 上游：dist.broadcast
vllm-ascend：scheduler 进程同步（紧急方案）
```

这是 [L4/09_PP+MTP.md](../L4-进阶场景/09_PP+MTP.md) 的核心话题。

### 7.3 Scheduler 跨进程协调

vLLM V1 架构：
```
EngineCore 进程（跑 scheduler）
   ↓ RPC
Worker 进程 × N（每个跑一个 PP rank）
```

scheduler 跨进程驱动所有 worker，PP rank 之间还要点对点通信。

---

## 8. 总结对比

| 维度 | 训练 PP | 推理 PP |
|------|---------|---------|
| 调度算法 | 1F1B / Interleaved | continuous batching |
| 气泡消除 | 调度算法本身 | continuous batching + async |
| 状态同步 | 梯度 all-reduce | scheduler 协调 + broadcast |
| 内存管理 | activation 队列 | KV cache per stage |
| 复杂度集中点 | PP 调度器 | scheduler + model runner |

---

## 9. 学习提示

理解推理 PP 的关键：

1. **不要套用训练 PP 的所有概念**：1F1B / Interleaved 1F1B 在推理中作用有限
2. **关注 scheduler**：推理 PP 的核心难点在调度协调
3. **关注 KV cache**：是推理 PP 内存和通信的核心
4. **关注 continuous batching**：是推理 PP 消除气泡的关键

---

**下一步**：[../L3-vLLM实现/06_四大核心抽象.md](../L3-vLLM实现/06_四大核心抽象.md) — 进入 vLLM 源码，看 PP 的四大核心抽象。
