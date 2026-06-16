# 知识点 05：vLLM TP 的核心抽象与 Linear 实现

> 核心问题：vLLM 怎么实现 TP？ColumnParallelLinear / RowParallelLinear 的源码细节？
> 源码位置：`vllm/distributed/parallel_state.py`、`vllm/model_executor/layers/linear.py`、`vllm/distributed/communication_op.py`

---

## 1. TP 核心抽象总览

| 抽象 | 类型 | 职责 |
|------|------|------|
| `GroupCoordinator` | 通用进程组 | 跟 PP 共用同一个类，通过 `_TP` 全局变量区分 |
| `ColumnParallelLinear` | Linear 层 | Column Parallel：切输出维度 |
| `RowParallelLinear` | Linear 层 | Row Parallel：切输入维度 |
| `MergedColumnParallelLinear` | Linear 层 | 多个 Column 合并（如 gate + up） |
| `QKVParallelLinear` | Linear 层 | Q/K/V 合并，处理 GQA |
| `ReplicatedLinear` | Linear 层 | 不切，每张卡完整副本 |
| `VocabParallelEmbedding` | Embedding 层 | 词表维切分 |
| `tensor_model_parallel_all_reduce` | 通信函数 | TP 组 all-reduce |

**重要**：vLLM **没有专门的 TP 类**（跟 PP 一样），TP 也用 `GroupCoordinator`，只是通过 `_TP` 全局变量区分。

---

## 2. GroupCoordinator（TP 复用）

### 2.1 全局变量

源码：`vllm/distributed/parallel_state.py:1331`

```python
_TP: GroupCoordinator | None = None

def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP
```

### 2.2 TP 组的初始化

源码：`vllm/distributed/parallel_state.py:1758-1772`

```python
# Build the tensor model-parallel groups.
global _TP
assert _TP is None, "tensor model parallel group is already initialized"
group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
group_ranks = [x.tolist() for x in group_ranks]

# message queue broadcaster is only used in tensor model parallel group
_TP = init_model_parallel_group(
    group_ranks,
    get_world_group().local_rank,
    backend,
    use_message_queue_broadcaster=True,  # 仅 TP 用消息队列广播
    group_name="tp",
)
```

**关键点**：

- TP 组开 `use_message_queue_broadcaster=True`（PP、EP、外部 DP 不开；DCP 和 inner_dp_world 也开）
- 原因：TP/DCP 通信频繁，用消息队列（shm_broadcast）加速 metadata 广播

### 2.3 layout 顺序

源码：`vllm/distributed/parallel_state.py:1740`（注释）和 `1749-1755`

```
layout 顺序是: ExternalDP x DP x PP x PCP x TP
```

```python
all_ranks = torch.arange(world_size).reshape(
    -1,
    data_parallel_size,
    pipeline_model_parallel_size,
    prefill_context_model_parallel_size,
    tensor_model_parallel_size,
)
```

**TP 是最内层**，意味着同一 TP 组的 ranks 是相邻的（同节点内），符合"TP 不跨节点"的要求。

---

## 3. ColumnParallelLinear 源码细节

源码：`vllm/model_executor/layers/linear.py:394`

### 3.1 关键属性

```python
class ColumnParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, ...):
        self.tp_rank = get_tensor_model_parallel_rank() if not disable_tp else 0
        self.tp_size = get_tensor_model_parallel_world_size() if not disable_tp else 1
        self.input_size_per_partition = input_size          # K 不变
        self.output_size_per_partition = divide(output_size, self.tp_size)  # N/p
        self.output_partition_sizes = [self.output_size_per_partition]
        ...
        self.gather_output = gather_output  # 是否在 forward 后 all-gather
```

### 3.2 forward（核心）

源码：`vllm/model_executor/layers/linear.py:549-561`

```python
def forward(self, input_):
    bias = self.bias if not self.skip_bias_add else None
    
    # 直接做 GEMM，输入 input_ 是完整的（每张卡都有）
    output_parallel = self.quant_method.apply(self, input_, bias)
    
    if self.gather_output and self.tp_size > 1:
        # all-gather 沿最后一维（output 维）
        output = tensor_model_parallel_all_gather(output_parallel)
    else:
        output = output_parallel
    
    if not self.return_bias:
        return output
    output_bias = self.bias if self.skip_bias_add else None
    return output, output_bias
```

### 3.3 关键点

- 输入 `input_` 完整（不切）
- 输出 `output_parallel` 是分片
- 如果 `gather_output=True`：all-gather 合并（通信）
- 如果 `gather_output=False`：保持分片（不通信，给下游 Row Parallel）

### 3.4 偏置处理

bias shape `[N]` → 每张卡用 `[N/p]`。

vLLM 在权重加载时按 `tp_rank` 切分（`set_weight_attrs`），forward 时直接用本地 bias。

---

## 4. RowParallelLinear 源码细节

源码：`vllm/model_executor/layers/linear.py:1307`

### 4.1 关键属性

```python
class RowParallelLinear(LinearBase):
    def __init__(self, input_size, output_size, ..., input_is_parallel=True, reduce_results=True):
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, self.tp_size)  # K/p
        self.output_size_per_partition = output_size                      # N 不变
        ...
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
```

### 4.2 forward（核心）

源码：`vllm/model_executor/layers/linear.py:1445-1468`

```python
def forward(self, input_):
    if self.input_is_parallel:
        # 输入已经是分片（来自上游 ColumnParallel）
        input_parallel = input_
    else:
        # 输入是完整的，自己切
        split_input = split_tensor_along_last_dim(input_, num_partitions=self.tp_size)
        input_parallel = split_input[self.tp_rank].contiguous()
    
    # bias 只在 rank 0 加（避免 all-reduce 后重复加）
    bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
    
    # GEMM
    output_parallel = self.quant_method.apply(self, input_parallel, bias_)
    
    if self.reduce_results and self.tp_size > 1:
        # all-reduce 把每张卡的部分和加起来
        output = tensor_model_parallel_all_reduce(output_parallel)
    else:
        output = output_parallel
    
    return output
```

### 4.3 关键点

- **bias 只在 rank 0 加**（`tp_rank > 0` 时为 None）
- 原因：all-reduce 会对所有 rank 求和，如果每张卡都加 bias，最后会加 N 次
- `input_is_parallel=True` 是默认值，假设输入来自上游 ColumnParallelLinear（已分片）

### 4.4 为什么 bias 只在 rank 0 加？

数学上：

```
正确结果: Y = sum_i (X_i × A_i) + b

如果每 rank 都加 b:
all_reduce(output_i + b) = sum_i (output_i + b) = sum_i (output_i) + p × b

只在 rank 0 加 b:
all_reduce(output_0 + b, output_1, ..., output_{p-1}) 
    = (output_0 + b) + sum_{i>0}(output_i)
    = sum_i(output_i) + b    ✓
```

---

## 5. MergedColumnParallelLinear

源码：`vllm/model_executor/layers/linear.py:577`

### 5.1 用途

把多个 Column 合并成一个权重，典型场景：

```
gate_proj + up_proj → gate_up_proj
```

Llama 的 MLP：

```
传统做法:
    h_gate = SiLU(X × W_gate)        # [M, 4K]
    h_up = X × W_up                  # [M, 4K]
    h = h_gate * h_up                # [M, 4K]
    Y = h × W_down                   # [M, K]

合并做法（vLLM 默认）:
    gate_up = concat([W_gate, W_up], dim=1)  # [K, 8K]
    h = SiLU(X × gate_up[:, :4K]) * (X × gate_up[:, 4K:])  # 一次 GEMM
    Y = h × W_down
```

### 5.2 TP 切分

MergedColumnParallelLinear 继承 ColumnParallelLinear，切分逻辑类似，但每个 sub-matrix 独立切：

```python
# linear.py:618-619
self.tp_size = get_tensor_model_parallel_world_size()
self.tp_rank = get_tensor_model_parallel_rank()

# 每个 output_sizes[i] 都切 tp_size 份
self.output_partition_sizes = [
    divide(output_size, self.tp_size) for output_size in self.output_sizes
]
```

权重 layout：每 rank 拿到 `[gate_段 | up_段]`，每段 size 是 `output_sizes[i]/tp_size`。

---

## 6. QKVParallelLinear

源码：`vllm/model_executor/layers/linear.py:914`

### 6.1 关键属性

```python
class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self, hidden_size, head_size, total_num_heads, 
                 total_num_kv_heads=None, ...):
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = total_num_heads
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        
        # Q head 数切分
        self.num_heads = divide(self.total_num_heads, tp_size)
        
        # KV head 处理（GQA）
        if tp_size >= self.total_num_kv_heads:
            # KV head 数 <= TP size: 每 rank 持有 1 个 KV head 的权重
            # num_kv_head_replicas 表示这个 KV head 在多少 rank 上有副本
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(tp_size, self.total_num_kv_heads)
        else:
            # KV head 数 > TP size: 每 rank 均分 KV head
            self.num_kv_heads = divide(self.total_num_kv_heads, tp_size)
            self.num_kv_head_replicas = 1
```

源码精确位置：`vllm/model_executor/layers/linear.py:966-973`。

### 6.2 输出大小

源码：`vllm/model_executor/layers/linear.py:975-978`

```python
output_size = (
    self.num_heads * self.head_size
    + self.num_kv_heads * self.head_size
    + self.num_kv_heads * self.v_head_size
)
```

注意：**输出公式不乘 `num_kv_head_replicas`**。当 `tp_size >= total_num_kv_heads` 时，`num_kv_heads=1`，每 rank 只持有一个 KV head 的权重（但通过 replicas 标记这个 head 在多少 rank 上重复出现）。

例：Llama 70B，total_num_heads=64, total_num_kv_heads=8, head_size=128, TP=8：

```
num_heads = 64/8 = 8
进入 if 分支 (8 >= 8)
num_kv_heads = 1
num_kv_head_replicas = 8/8 = 1

输出 = (8 + 1 + 1) × 128 = 10 × 128 = 1280
（与完整 qkv_proj [8192, 10240] 切 8 份 = [8192, 1280] 一致 ✓）
```

例：Llama 70B，TP=4：

```
num_heads = 64/4 = 16
进入 else 分支 (4 < 8)
num_kv_heads = 8/4 = 2
num_kv_head_replicas = 1

输出 = (16 + 2 + 2) × 128 = 20 × 128 = 2560
（与完整 10240 切 4 份 = 2560 一致 ✓）
```

例：MQA 模型（Q=64, KV=1），TP=8：

```
num_heads = 64/8 = 8
进入 if 分支 (8 >= 1)
num_kv_heads = 1
num_kv_head_replicas = 8/1 = 8   ← 每 rank 持有这个 KV head 的副本

输出 = (8 + 1 + 1) × 128 = 10 × 128 = 1280
（这个 KV head 权重在 8 个 rank 上各有一份，是冗余的）
```

---

## 7. VocabParallelEmbedding

源码：`vllm/model_executor/layers/vocab_parallel_embedding.py:253`

### 7.1 切分方式

词表维度（dim=0）按 TP 切：

```python
self.tp_size = get_tensor_model_parallel_world_size()
self.num_embeddings_per_partition = divide(num_embeddings, self.tp_size)
self.weight = Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
```

### 7.2 forward

```python
def forward(self, input_):
    if self.tp_size > 1:
        # 输入 token id 减去偏移，映射到本 rank 的局部 id
        input_mask = input_ < self.vocab_start_index
        masked_input = input_ - self.vocab_start_index
        masked_input[input_mask] = 0
    else:
        masked_input = input_
    
    output_parallel = F.embedding(masked_input, self.weight)
    
    # 不属于本 rank 的 token，置零
    if self.tp_size > 1:
        output_parallel[input_mask, :] = 0.
    
    # all-reduce 把所有 rank 的结果加起来
    output = tensor_model_parallel_all_reduce(output_parallel) if self.tp_size > 1 else output_parallel
    return output
```

**为什么用 all-reduce 而不是 all-gather**：

- 每个 token 只属于一个 TP rank（其他 rank 把它置零）
- all-reduce 后，所有 rank 都得到正确结果
- 跟 Column + Row 配对类似：embedding 是 lookup（行选择），等效 Row Parallel

### 7.3 lm_head 的并行

最后一层的 logits 计算（lm_head）用 `ParallelLMHead`：

- 默认 ColumnParallel：每 rank 算自己那段 vocab 的 logits
- 推理通常用 `tensor_model_parallel_gather` 把所有 rank 的 logits 合并

---

## 8. 关键代码位置（已验证）

| 抽象/方法 | 文件 | 行号 |
|----------|------|------|
| `_TP` 全局变量 | `vllm/distributed/parallel_state.py` | L1331 |
| `get_tp_group` | 同上 | L1334 |
| TP 组初始化 | 同上 | L1758-1772 |
| layout 注释（ExternalDP×DP×PP×PCP×TP） | 同上 | L1740 |
| `tensor_model_parallel_all_reduce` | `vllm/distributed/communication_op.py` | L12 |
| `tensor_model_parallel_all_gather` | 同上 | L17 |
| `tensor_model_parallel_reduce_scatter` | 同上 | L24 |
| `tensor_model_parallel_gather` | 同上 | L31 |
| `broadcast_tensor_dict` | 同上 | L38 |
| `ColumnParallelLinear` | `vllm/model_executor/layers/linear.py` | L394 |
| `ColumnParallelLinear.forward` | 同上 | L549 |
| `RowParallelLinear` | 同上 | L1307 |
| `RowParallelLinear.forward` | 同上 | L1444 |
| `MergedColumnParallelLinear` | 同上 | L577 |
| `QKVParallelLinear` | 同上 | L914 |
| `QKVParallelLinear.__init__`（GQA 处理） | 同上 | L961-972 |
| `ReplicatedLinear` | 同上 | L289 |
| `VocabParallelEmbedding` | `vllm/model_executor/layers/vocab_parallel_embedding.py` | L253 |

---

## 9. 常见误用提醒

| 误用 | 正确 |
|------|------|
| `class TensorParallelGroup`（不存在） | 用 `GroupCoordinator` |
| `tp.broadcast()`（不存在专用） | 用 `tp.broadcast_tensor_dict()` |
| RowParallelLinear 默认每 rank 都加 bias | **只在 rank 0 加** |
| `ColumnParallelLinear(gather_output=True)` 一定通信 | 只在 `tp_size > 1` 时才通信 |
| Q/K/V 三个独立 ColumnParallelLinear | 用 `QKVParallelLinear` 合并 |

---

## 10. 小结

- vLLM **没有专门的 TP 类**，TP 复用 `GroupCoordinator`，通过 `_TP` 区分
- TP 组独享 `use_message_queue_broadcaster=True`（shm_broadcast 加速）
- layout 顺序：`ExternalDP × DP × PP × PCP × TP`，TP 是最内层（同节点）
- ColumnParallelLinear：切输出，可选 gather；RowParallelLinear：切输入，all-reduce
- QKVParallelLinear 处理 GQA：KV head 数 < TP size 时复制

---

**下一步**：[06_AllReduce实现.md](06_AllReduce实现.md) — all-reduce 怎么实现？custom all-reduce 是什么？
