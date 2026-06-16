# 知识点 04：推理 TP 与训练 TP 的差异

> 核心问题：vLLM 是推理框架，它的 TP 跟训练（Megatron）TP 有什么本质差异？

---

## 1. 核心差异总览

| 维度 | 训练 TP | 推理 TP |
|------|---------|---------|
| **Backward** | 有 all-reduce 梯度 | **无** |
| **激活内存** | 要保留用于 backward | **不用** |
| **通信频率** | forward + backward 各 all-reduce | 只 forward all-reduce |
| **KV cache** | 不涉及 | **每 rank 自己维护** |
| **通信内容** | activation + gradient | 只 activation |
| **同步模式** | 强同步 | async scheduling 可异步 |
| **目标** | 训练吞吐 | 推理延迟 + 吞吐 |

---

## 2. 差异一：没有 Backward

### 训练场景

训练每个 step 的 TP 通信：

```
forward:
   fc2/o_proj 后 all-reduce（合并输出）
backward:
   fc2/o_proj 后 all-reduce（合并梯度）
   embedding 层 all-reduce（梯度同步）
```

每个 Linear 块两次 all-reduce。

### 推理场景

推理只有 forward：

```
forward:
   fc2/o_proj 后 all-reduce
```

每 Linear 块一次 all-reduce。

**对实现的影响**：

- backward 路径不需要实现
- 内存不需要保留中间激活用于 backward
- communication_op.py 不需要 backward 相关的 reduce-scatter 等

---

## 3. 差异二：激活内存

### 训练场景

为了 backward，每层的中间激活都要保存：

```
forward 时记录：X (input), h (post-activation), attn_out, ...
backward 时反推：根据这些算梯度
```

TP 不改变这点：激活内存仍然要按层保存。

### 推理场景

推理不需要 backward，激活可以即用即弃：

```
forward 一层 → 拿到输出 → 立刻丢弃输入
不需要保存中间激活
```

**对 TP 实现的影响**：

- 不需要 activation checkpointing
- 内存占用更少
- TP rank 的中间激活可以直接复用 buffer

---

## 4. 差异三：KV Cache 各自维护

### 训练场景

训练时**没有 KV cache 概念**：每层 forward 重新计算 attention。

### 推理场景

推理必须用 KV cache。

**KV cache 跟 head 绑定**：

```
TP=8 的 Llama 70B (Q=64, KV=8):

GPU 0: 维护 head 0 的 KV cache
GPU 1: 维护 head 1 的 KV cache
...
GPU 7: 维护 head 7 的 KV cache
```

**对 TP 通信的影响**：

- **KV cache 不在 TP rank 间传递**（私有）
- TP 间只传 activation（forward 输入和最后 all-reduce）

**好处**：

- KV cache 显存压力被均匀分摊到所有 TP rank
- 不引入额外通信

---

## 5. 差异四：Continuous Batching

### 训练场景的 batch

训练一个 batch 固定：

```
固定 num_micro_batch，TP 调度可以预先规划
```

### 推理场景

请求流式到达，batch 在动态变化：

```
时刻 t=0: 1 个请求，TP rank 各算自己 head
时刻 t=1: +2 个请求
时刻 t=2: -1 个请求
```

**对 TP 实现的影响**：

- 每个 TP rank 必须独立处理 batch 的变化
- attention metadata（start/end offsets、KV cache 位置）每个 rank 都要 broadcast 一致
- vLLM 用 `broadcast_tensor_dict` 从 rank 0 广播 attention metadata

---

## 6. 差异五：Attention Metadata 同步

### 训练场景

attention metadata 是 batch 级别的固定值，不需要同步。

### 推理场景

vLLM 在每一步 forward 时，scheduler 算出的 attention metadata 必须广播给所有 TP rank：

```python
# vllm/v1/worker/gpu_worker.py 简化
def execute_model(model_input):
    # model_input 包含 attention_metadata
    # 用 broadcast_tensor_dict 让所有 TP rank 同步
    
    if not is_driver:
        model_input = broadcast_tensor_dict(src=0)  # 从 driver rank 接收
    ...
    output = model.forward(model_input)
```

**关键**：

- 只有 driver rank（通常 tp_rank=0）跟 scheduler 通信
- 其他 rank 从 driver rank 接收 metadata
- 这是 TP 在 vLLM V1 中的核心同步点

---

## 7. 差异六：Async Scheduling

### 训练场景

每层 forward + backward 都同步，强同步是必须的。

### 推理场景

vLLM 的 async-scheduling（V1）允许 forward 异步执行：

```
传统 TP：
   rank 0: forward(L1) → all-reduce(L1) → forward(L2) → all-reduce(L2)...
   rank 1: forward(L1) → all-reduce(L1) → forward(L2) → all-reduce(L2)...
   两个 rank 在每个 all-reduce 都同步

Async scheduling：
   rank 0: forward(L1) → send / forward(L2 of step k-1) 同时进行
   rank 1: forward(L1) → wait / forward(L2 of step k-1) 同时进行
```

通信和计算 overlap，减少气泡。

---

## 8. 推理 TP 的特殊挑战

### 8.1 Custom All-Reduce 优化

推理对 all-reduce 延迟极敏感：

- 训练：all-reduce 占总时间 10-20%
- 推理：all-reduce 占总时间 30-50%（小 batch 下）

vLLM 提供了 **custom all-reduce kernel**（不依赖 NCCL）：

- 利用 NVLink / NVSwitch 直接 shared memory
- 单节点内比 NCCL 快 1.5-2x
- vLLM V1 默认开启

详见 [../L3-vLLM实现/06_AllReduce实现.md](../L3-vLLM实现/06_AllReduce实现.md)。

### 8.2 LM Head 的 TP 处理

最后一层的 lm_head（logits 计算）：

- 词表大（如 128k），`[hidden, vocab_size]` 权重很大
- vLLM 用 Column Parallel：每 rank 算自己那段 vocab 的 logits
- 配合 `ParallelLMHead` 和 `gather_output` 选项
- 实际通常用 `tensor_model_parallel_gather` 把所有 rank 的 logits 合并

### 8.3 EP 与 TP 的混合

MoE 模型下，expert 部分常用 EP 而非 TP：

- attention 部分仍然 TP（head 切分）
- expert 部分用 EP（expert 切分）
- vLLM 的 MoE 实现支持 TP+EP 混合

详见 [../L4-进阶场景/08_TP组合并行.md](../L4-进阶场景/08_TP组合并行.md)。

---

## 9. 总结对比

| 维度 | 训练 TP（Megatron） | 推理 TP（vLLM） |
|------|---------------------|-----------------|
| 通信次数/层 | 2（fwd + bwd） | 1（仅 fwd） |
| 中间激活 | 必须保存 | 可即用即弃 |
| KV cache | 无 | per-rank 维护 |
| Metadata 同步 | 不需要 | broadcast_tensor_dict |
| All-reduce 实现 | NCCL | NCCL + custom kernel |
| 同步模式 | 强同步 | 可 async overlap |

---

## 10. 学习提示

理解推理 TP 的关键：

1. **不要套用训练 TP 的所有概念**：1F1B、ring-pipe 等训练概念在推理中不适用
2. **关注 KV cache**：是推理 TP 的核心内存
3. **关注 attention metadata 同步**：是推理 TP 在 vLLM 中的同步难点
4. **关注 custom all-reduce**：是推理 TP 性能优化的关键

---

**下一步**：[../L3-vLLM实现/05_vLLM_TP核心抽象.md](../L3-vLLM实现/05_vLLM_TP核心抽象.md) — 进入 vLLM 源码。
