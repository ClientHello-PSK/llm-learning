# 知识点 10：PP + Async Scheduling

> 核心问题：传统 PP 有 stage 间等待的气泡，async-scheduling 怎么消除？

---

## 1. 传统 PP 的同步瓶颈

### 1.1 同步模式

```
Step N:
    Stage 0: forward ─→ send
                        ↓
    Stage 1:           recv → forward → send
                                       ↓
    Stage 2:                          recv → forward → send
                                                      ↓
    Stage 3:                                         recv → forward

所有 stage 必须按顺序、强同步执行。
```

### 1.2 气泡来源

```
Step N 完成后，Stage 0 必须等：
- Stage 1/2/3 处理完
- 接收新的 input
才能开始 Step N+1
```

这导致 Stage 0 大段空闲。

---

## 2. Async Scheduling 的核心思想

**每个 stage 像独立工人**：不互相等待，通过队列连接。

```
Step N+1 开始时：
    Stage 0 不等 Stage 1 完成 Step N
    而是直接开始处理新的 micro_batch

Stage 间通过"队列"传递数据
队列有 buffer，可以缓存 1-2 个 micro_batch
```

类似 CPU 流水线的乱序执行。

---

## 3. 双缓冲（Double Buffering）

实现 async 的关键技术。

### 3.1 单缓冲的问题

```
Stage 0: |forward A|send A|   等   |forward B|send B|
                              ↑
                          Stage 1 没接收完，Stage 0 不能 send B
```

### 3.2 双缓冲方案

```
准备两个 buffer：buffer_0, buffer_1

Step N:
  Stage 0: forward A → 写入 buffer_0
  Stage 1: 从 buffer_1（之前的数据）读
  
Step N+1:
  Stage 0: forward B → 写入 buffer_1
  Stage 1: 从 buffer_0（A 的数据）读
```

写入和读取交替使用不同 buffer，互不阻塞。

---

## 4. vLLM 的 async-scheduling 实现

### 4.1 整体架构

```
EngineCore 进程
    ↓
scheduler（不停调度）
    ↓ RPC
Worker 进程 × N（每个一个 PP rank）
    ↓
每个 worker 内部：
    - input_queue（接收 scheduler 调度）
    - output_queue（输出结果）
    - PP 通信 buffer
```

### 4.2 关键设计

| 组件 | 作用 |
|------|------|
| EngineCore | 主进程，跑 scheduler |
| RPC | EngineCore 跟 worker 通信 |
| AsyncScheduler | 不等所有 worker 完成，连续调度 |
| IntermediateTensors buffer | PP stage 间的双缓冲 |

### 4.3 跟传统同步的对比

**传统同步**：

```python
for step in steps:
    schedule = scheduler.schedule()
    outputs = workers.execute(schedule)  # 等所有 worker 完成
    scheduler.update(outputs)
```

**Async**：

```python
# 持续调度，不等
while True:
    if has_new_schedule:
        workers.submit_async(new_schedule)
    
    if has_completed_output:
        output = workers.get_output()
        scheduler.update(output)
```

---

## 5. Async 怎么消除气泡？

### 5.1 传统 PP 的尾部气泡

```
时间:    1    2    3    4    5    6    7
Stage 0: |m1F|m2F|m3F|m4F| ★  | ★  | ★  |   ← 尾部气泡
Stage 3: | ★ | ★ | ★ |m1F|m2F|m3F|m4F|
```

### 5.2 Async PP 的流水

```
时间:    1    2    3    4    5    6    7
Stage 0: |m1F|m2F|m3F|m4F|m5F|m6F|m7F|   ← 连续处理
Stage 1: | ★ |m1F|m2F|m3F|m4F|m5F|m6F|
Stage 2: | ★ | ★ |m1F|m2F|m3F|m4F|m5F|
Stage 3: | ★ | ★ | ★ |m1F|m2F|m3F|m4F|
```

Stage 0 一直在 forward，不等 Stage 3。

### 5.3 气泡去哪了？

气泡转移到：
- **流水线启动期**（最初的 PP-1 个 step）
- **流水线排空期**（最后 PP-1 个 step）

但只要请求源源不断，启动期和排空期都可以被吸收。

---

## 6. Async 跟 PP+MTP 的关系

### 6.1 协同优势

async-scheduling + broadcast（vllm#39704）是黄金组合：

```
Stage 3 完成 draft + 验证
    ↓
broadcast 给其他 rank（不通过 scheduler）
    ↓
其他 rank 异步接收 + 更新状态
    ↓
所有 rank 同时进入 Step N+1
```

### 6.2 跟 scheduler 同步的冲突

如果 PP+MTP 用 scheduler 同步：

```
Stage 3 完成后，必须通过 scheduler 通知其他 rank
    ↓
scheduler 必须等所有 rank 反馈
    ↓
async-scheduling 失效（又变回同步）
```

这就是为什么 reviewer 强烈推荐 broadcast 方案。

---

## 7. Async 的挑战

### 7.1 状态一致性

每个 rank 异步处理，怎么保证状态一致？

**方案**：
- 用 broadcast 同步关键状态（如采样结果）
- 每个 rank 独立维护 KV cache，但通过 broadcast 保证 token 序列一致

### 7.2 错误处理

某个 rank 出错时，怎么传播？

**方案**：
- 通过 EngineCore 协调
- 出错 rank 通过 RPC 通知 EngineCore
- EngineCore 通知所有 rank 回滚

### 7.3 调试复杂

异步执行让 debug 困难：

- 时序错乱难复现
- 多 rank 协同调试

**方案**：
- 详细日志（每 step 记录每个 rank 状态）
- 可回放的执行 trace

---

## 8. vLLM 的 async-scheduling 演进

### 8.1 PR 系列

- **vllm#39074 系列**：async-scheduling 基础架构
- **vllm#39704**：PP+MTP 跟 async 配合
- **vllm-ascend**：适配中

### 8.2 当前状态

| 平台 | async-scheduling |
|------|-----------------|
| vLLM 上游 | 已支持 |
| vLLM V1 架构 | 基础设计 |
| vllm-ascend | 适配中 |

---

## 9. 性能对比

### 9.1 同步 PP vs Async PP

| 场景 | 同步 PP | Async PP |
|------|---------|----------|
| 单条 prompt | 慢（气泡大） | 略快 |
| 高 QPS | 慢（互相等待） | 快（持续流水） |
| 长序列 | 慢 | 快 |
| 短序列 | 接近 | 接近 |

### 9.2 实测提升

在大规模部署场景，async-scheduling 通常带来 **20-50% 的吞吐提升**。

---

## 10. 小结

- Async-scheduling 让每个 stage 像独立工人，通过队列协作
- 双缓冲是关键技术（避免读写冲突）
- 消除气泡的本质：把气泡从"中间"转移到"启动期和排空期"
- 跟 PP+MTP 配合：必须用 broadcast 方案，否则破坏 async
- vLLM 上游已支持，vllm-ascend 适配中

---

**下一步**：[11_PP组合并行.md](11_PP组合并行.md) — PP 怎么跟 TP、EP 组合？
