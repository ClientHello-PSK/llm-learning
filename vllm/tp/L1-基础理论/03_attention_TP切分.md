# 知识点 03：Attention 的 TP 切分

> 核心问题：Multi-Head Attention 在 TP 下怎么切？GQA / MQA 怎么办？

---

## 1. MHA 在 TP 下的切分

### 1.1 出发点

Multi-Head Attention：

```
Q = X × W_q    shape: [M, num_heads × head_dim]
K = X × W_k    shape: [M, num_kv_heads × head_dim]
V = X × W_v    shape: [M, num_kv_heads × head_dim]

attention(Q, K, V) → [M, num_heads × head_dim]

Y = attn_out × W_o    shape: [M, hidden_dim]
```

### 1.2 TP 切分思路

按 head 切：

```
TP=p，把 num_heads 分成 p 组，每组 num_heads/p 个 head

tp_rank i 处理：
    Q_i = X × W_q_i        shape: [M, (num_heads/p) × head_dim]
    K_i = X × W_k_i
    V_i = X × W_v_i
    
    attention_i(Q_i, K_i, V_i)   ← 每个 rank 独立算自己的 head
    
    attn_out_i: [M, (num_heads/p) × head_dim]
    
Y_i = attn_out_i × W_o_i   ← Row Parallel，沿 head 维切 W_o
```

最后 all-reduce 合并：

```
Y = all_reduce(Y_i)
```

### 1.3 关键观察

**Attention 的 Q/K/V 是 Column Parallel（按 head 切），output projection 是 Row Parallel**。

跟 MLP 的 fc1 (Column) → fc2 (Row) 配对完全一样，通信只发生在 output proj 之后。

---

## 2. 为什么 attention 内部不需要通信？

Attention 的 head 之间是**完全独立**的：

```
每个 head 各自算自己的 attention scores、softmax、weighted sum
head 之间不交互
```

TP rank i 只算自己的 num_heads/p 个 head，**不需要与其他 rank 同步**。

这是 TP 切 attention 的根本优势：head 数量天然适合并行切分。

---

## 3. GQA / MQA 的麻烦

### 3.1 GQA 是什么

Grouped Query Attention（GQA）：

```
普通 MHA:     Q 有 N heads, K/V 各有 N heads      (1:1)
MQA:         Q 有 N heads, K/V 各有 1 head         (N:1)
GQA:         Q 有 N heads, K/V 各有 N/g heads      (1:g)
```

例：Llama 70B 用 GQA，Q=64 heads，K/V=8 heads（g=8）。

### 3.2 TP 切 GQA 的问题

简单按 head 切 Q 不行，因为：

```
tp_rank i 应该处理 head [i, i+1, ..., i+N/p-1]

但 K/V 只有 N/g heads，怎么分？
```

### 3.3 解决方案：复制 KV head

如果 `num_kv_heads < tp_size`，**复制 KV head**。

例：Q=64, K=V=8, TP=8

```
每张卡分配 64/8 = 8 个 Q head
每张卡分配 8/8 = 1 个 KV head（但需要 8 个）
   → 每个 KV head 复制 8 份

每个 rank 处理：
    Q: 8 heads
    K: 1 head (但被复制使用 8 次)
    V: 1 head
```

**代价**：K/V 权重在多个 rank 上重复，但 attention 计算仍然正确。

vLLM 中 `QKVParallelLinear` 处理这个：

```python
# vllm/model_executor/layers/linear.py:966-973
tp_size = get_tensor_model_parallel_world_size()
self.num_heads = divide(self.total_num_heads, tp_size)
if tp_size >= self.total_num_kv_heads:
    # KV head 数 <= TP size: 每 rank 拿 1 个 KV head
    # 但通过 num_kv_head_replicas 标记这个 head 在多少 rank 上有副本
    self.num_kv_heads = 1
    self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
else:
    # KV head 数 > TP size: 每 rank 均分 KV head
    self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
    self.num_kv_head_replicas = 1
```

**关键**：当 `tp_size >= total_num_kv_heads` 时，**每 rank 的 `num_kv_heads` 设为 1**（而不是 `total_num_kv_heads`）。这意味着每 rank 只持有 1 个 KV head 的权重，但因为该 KV head 在多个 rank 上有副本（`num_kv_head_replicas` 份），整体仍能覆盖所有 KV head。

### 3.4 限制条件

- **必须 `num_heads % tp_size == 0`**：Q head 要能均分
- **如果 `num_kv_heads >= tp_size`**：必须 `num_kv_heads % tp_size == 0`（KV head 均分，无复制）
- **如果 `num_kv_heads < tp_size`**：必须 `tp_size % num_kv_heads == 0`（KV head 复制整数次）

例：Llama 70B (Q=64, KV=8)：

| TP size | 可行？ | 每 rank Q head | 每 rank KV head |
|---------|--------|----------------|-----------------|
| 1 | ✓ | 64 | 8 |
| 2 | ✓ | 32 | 4 |
| 4 | ✓ | 16 | 2 |
| 8 | ✓ | 8 | 1（复制） |
| 16 | ✗ | - | - |

---

## 4. Attention 的通信总结

| 阶段 | 通信 | 说明 |
|------|------|------|
| Q/K/V 投影 | 无 | Column Parallel，输入 X 各卡都有 |
| attention 计算 | 无 | head 之间独立 |
| output 投影 | **all-reduce** | Row Parallel |
| KV cache 写入 | 无 | 每 rank 写自己的 KV 分片 |

**整个 attention 块只有 output proj 后一次 all-reduce**。

---

## 5. KV cache 在 TP 下的分片

每张卡维护自己 num_kv_heads/p（或复制后的）head 的 KV cache：

```
GPU 0:  KV cache for head [0..n/p-1]
GPU 1:  KV cache for head [n/p..2n/p-1]
...
```

**关键**：KV cache 不在 TP rank 间共享。

对 GQA，KV cache 大小被 g 倍降低（跟 Q head 比）。

---

## 6. MLA（DeepSeek）的特殊处理

DeepSeek V3/V4 用 MLA（Multi-head Latent Attention），跟标准 MHA 差异较大：

- Q/K/V 投影后是低秩压缩的 latent
- 实际 attention 用解压后的 head

TP 下，MLA 的处理方式：

- latent 部分按 TP 切
- decode 阶段的吸收矩阵、解压等需要特殊处理

vLLM 用 `MLAAttention`（`vllm/model_executor/layers/mla.py`）单独实现，TP 切分逻辑跟标准 attention 不完全一样。

---

## 7. Attention TP 的实际效果（Llama 70B）

```
num_heads = 64, num_kv_heads = 8, head_dim = 128, hidden_dim = 8192

完整 weights:
    qkv_proj:  [8192, (64+8+8)*128] = [8192, 10240]  ≈ 168MB
    o_proj:    [8192, 8192]                            ≈ 134MB

TP=8:
    qkv_proj:  [8192, 10240/8] = [8192, 1280]         ≈ 21MB
    o_proj:    [8192/8, 8192] = [1024, 8192]          ≈ 17MB
    
    每 rank 的 attention head: 8 Q + 1 KV (num_kv_head_replicas=1，每 rank 1 个 KV head)
```

注意 q_proj 的切分：64/8 = 8 heads/rank。KV 共 8 个、TP=8 时正好每 rank 拿 1 个 KV head（`num_kv_heads=1, num_kv_head_replicas=1`，无需复制）。如果 TP=16，则每 rank 也拿 1 个 KV head，但每 KV head 被复制 2 次（`num_kv_head_replicas=2`）。

---

## 8. 常见错误

### 8.1 误以为 attention 内部有 all-reduce

**错**：attention head 是独立的，attention 计算本身不通信。

通信只发生在 `o_proj` 之后（all-reduce）和 `qkv_proj` 之前的可能 broadcast（如果 X 不是 replicated）。

### 8.2 误以为 K/V 在 TP rank 间共享

**错**：每个 TP rank 维护自己的 K/V 分片，独立计算 attention。

跨 TP rank 共享 K/V 的方案叫 **cross-attention KV sharing** 或 **Disaggregated Attention**，是研究性的，vLLM 默认不做。

### 8.3 误以为 TP 一定要能整除 num_kv_heads

**错**：`num_kv_heads < tp_size` 时通过复制解决，只要 `tp_size % num_kv_heads == 0` 即可。

---

## 9. 小结

- attention TP 切分：Q/K/V 按 head 切（Column），o_proj 沿 head 维切（Row）
- attention 内部**不通信**（head 独立）
- GQA / MQA：当 KV head 数 < TP size，复制 KV head
- 限制：`num_heads % tp_size == 0`

---

**下一步**：[../L2-推理TP特化/04_推理TP与训练TP差异.md](../L2-推理TP特化/04_推理TP与训练TP差异.md) — 推理场景下 TP 有什么特殊。
