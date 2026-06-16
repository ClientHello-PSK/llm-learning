# 知识点 03：GPipe 调度详解

> 核心问题：GPipe 怎么调度 forward/backward？气泡具体长什么样？

---

## 1. GPipe 的核心思想

**All-Forward → All-Backward**：

```
阶段一：所有 micro-batch 依次做完 forward
阶段二：所有 micro-batch 依次做完 backward
```

最简单直接的 PP 调度，由 Google 在 2019 年提出。

---

## 2. 完整时序图（PP=4, micro_batch=4）

> 详细彩色图见 [../diagrams/gpipe_timeline.md](../diagrams/gpipe_timeline.md)

```
时间:        1    2    3    4    5    6    7    8    9   10   11   12   13   14
Stage 0:   |m1F|m2F|m3F|m4F|        气泡 (7格)        |m4B|m3B|m2B|m1B|
Stage 1:   | ★ |m1F|m2F|m3F|m4F|     气泡 (5格)    |m4B|m3B|m2B|m1B| ★ |
Stage 2:   | ★ | ★ |m1F|m2F|m3F|m4F|  气泡 (3格) |m4B|m3B|m2B|m1B| ★ | ★ |
Stage 3:   | ★ | ★ | ★ |m1F|m2F|m3F|m4F|m4B|m3B|m2B|m1B| ★ | ★ | ★ |
                          ↑
                     Forward 完成后立刻 Backward
```

---

## 3. 关键观察

### 3.1 Forward 阶段（t=1 到 t=7）

- m1 最先进入流水线，t=4 走完所有 stage
- m4 最后进入，t=7 走完所有 stage
- Forward 阶段总长 = PP + micro_batch - 1 = 4 + 4 - 1 = 7

### 3.2 Backward 阶段（t=8 到 t=14）

- Backward 从 S3 开始（最后的 stage 先 backward）
- **顺序是 LIFO**：m4 最先 backward（因为 m4 最后完成 forward，激活还在 S3 内存里）
- m1 最后 backward

### 3.3 为什么 Backward 是 LIFO？

考虑 m1 在 t=4 完成 forward（在 S3），它的 activation 必须保留到 backward 时使用。
此时 m2, m3, m4 还没 forward，S3 内存还能装下它们的 activation。
所以 GPipe 让所有 forward 跑完（t=7），此时 S3 装着 m1/m2/m3/m4 全部激活。
backward 必须从最后一个 forward（m4）开始，因为 m4 是最近被放入激活队列的。

### 3.4 气泡分布

- **Stage 0**：气泡在中间（7 格）——前向等后面处理完，反向等梯度传回
- **Stage 3**：气泡在两端（前 3 格 + 后 3 格）——启动晚，结束早
- 中间 stage 的气泡居中

气泡呈"沙漏"形态：两端 stage 气泡分散，中间 stage 气泡集中。

---

## 4. 气泡公式推导

### 4.1 总时间

```
Forward 阶段 = PP + micro_batch - 1
Backward 阶段 = PP + micro_batch - 1
总时间 = 2 × (PP + micro_batch - 1)
```

### 4.2 每个 stage 的实际工作时间

每个 stage 处理所有 micro_batch：
```
工作时间 = micro_batch × 2（每个 micro_batch 一次 forward + 一次 backward）
```

### 4.3 气泡占比

```
气泡 = 总时间 - 工作时间
     = 2 × (PP + micro_batch - 1) - 2 × micro_batch
     = 2 × (PP - 1)

气泡占比 = 气泡 / 总时间
        = 2(PP-1) / 2(PP + micro_batch - 1)
        = (PP - 1) / (PP + micro_batch - 1)
```

### 4.4 代入验证

PP=4, micro_batch=4:
```
气泡占比 = 3 / 7 ≈ 43% ✓
```

PP=4, micro_batch=16:
```
气泡占比 = 3 / 19 ≈ 16%
```

**结论**：micro_batch 越大，气泡占比越小，但单 micro_batch 显存占用也越小（更多激活在内存中累积）。

---

## 5. GPipe 的优缺点

### 优点

| 优点 | 说明 |
|------|------|
| 实现简单 | 调度逻辑直接（all-F then all-B） |
| 通信模式清晰 | forward 只 send，backward 只 send 反向 |
| 跟 continuous batching 兼容 | 推理场景主流 |

### 缺点

| 缺点 | 说明 |
|------|------|
| 气泡大 | 中间 stage 有大段空闲 |
| 内存占用高 | 所有 micro_batch 的激活都要保留到 backward |
| 不适合大 micro_batch 数 | 内存吃不消 |

---

## 6. 内存峰值分析

GPipe 的关键问题：内存。

**Stage N-1 的激活累积**：

```
t=4: S3 收到 m1 activation
t=5: S3 收到 m2 activation
t=6: S3 收到 m3 activation
t=7: S3 收到 m4 activation
此时 S3 内存装着 4 个 activation
```

直到 t=8 开始 backward，才能释放 m4 的 activation。

**内存峰值 ≈ micro_batch × activation_size**

这是 GPipe 最大的痛点，1F1B 调度就是为解决这个问题而生（见下一篇）。

---

## 7. GPipe 在推理场景的应用

vLLM 推理场景**只有 forward，没有 backward**，所以 GPipe 时序图简化为：

```
时间:        1    2    3    4    5    6    7
Stage 0:   |m1F|m2F|m3F|m4F| ★  | ★  | ★  |
Stage 1:   | ★ |m1F|m2F|m3F|m4F| ★  | ★  |
Stage 2:   | ★ | ★ |m1F|m2F|m3F|m4F| ★  |
Stage 3:   | ★ | ★ | ★ |m1F|m2F|m3F|m4F|
```

**气泡集中在尾部**：所有 micro_batch 跑完后，前面的 stage 空闲。

vLLM 通过 **continuous batching** 消除这个尾部气泡：持续有新请求进来，stage 不会空闲。

---

## 8. 小结

- GPipe = all-forward → all-backward，最简单的 PP 调度
- 气泡占比公式：`(PP-1) / (PP+micro_batch-1)`
- 内存峰值：`micro_batch × activation_size`
- Backward 顺序是 LIFO（最后 forward 的最先 backward）
- 推理场景只有 forward，气泡在尾部，用 continuous batching 消除

---

**下一步**：[04_1F1B调度详解.md](04_1F1B调度详解.md) — 怎么改进 GPipe 的内存问题？
