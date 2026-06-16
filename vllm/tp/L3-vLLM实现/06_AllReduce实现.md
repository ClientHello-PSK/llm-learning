# 知识点 06：All-Reduce 实现与 Custom All-Reduce

> 核心问题：TP 的 all-reduce 在 vLLM 中怎么实现？custom all-reduce 为什么快？
> 源码位置：`vllm/distributed/device_communicators/`

---

## 1. All-Reduce 的基本概念

### 1.1 操作定义

all-reduce：把所有 rank 的张量**求和**，结果广播给所有 rank。

```
输入：每张卡有 X_i
输出：所有卡得到 sum(X_i)
```

### 1.2 TP 中的用途

TP 中 all-reduce 用在：

1. **RowParallelLinear 末尾**：合并各 rank 的部分和
2. **VocabParallelEmbedding 末尾**：合并各 rank 的 lookup 结果
3. **RMSNorm/LayerNorm 中的 TP 同步**（部分融合 norm）

### 1.3 入口函数

源码：`vllm/distributed/communication_op.py:12`

```python
def tensor_model_parallel_all_reduce(input_: torch.Tensor) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_)
```

调用链：

```
tensor_model_parallel_all_reduce
    ↓
GroupCoordinator.all_reduce
    ↓
self.device_communicator.all_reduce
    ↓
NCCL / HCCL / custom kernel
```

---

## 2. All-Reduce 的算法

### 2.1 Ring All-Reduce（NCCL 默认）

```
N=4 的 ring：
0 → 1 → 2 → 3 → 0

Phase 1 (reduce-scatter): N 步
  每步每张卡发自己的一段给下游，接收上游的一段累加
  最终每张卡有一段的完整求和

Phase 2 (all-gather): N 步
  每步把"完整求和段"传给下游
  最终每张卡都有完整结果

总通信量: 2 × (N-1)/N × tensor_size
```

### 2.2 Tree All-Reduce（小张量）

```
Tree：
       root
      / | \
     /  |  \
   r0  r1  r2

Phase 1 (reduce): 每张卡把数据 reduce 到 root
Phase 2 (broadcast): root 广播给所有

总通信量: 2 × tensor_size，但延迟 O(log N)
```

NCCL 会根据张量大小自动选择算法。

---

## 3. vLLM 的通信器层级

```
GroupCoordinator
    ↓ 组合
DeviceCommunicator（基类）
    ↓ 子类选择
NcclDeviceCommunicator     ← NVIDIA GPU（基于 NCCL）
HcclDeviceCommunicator     ← 华为 NPU（基于 HCCL）
CpuCommunicator            ← CPU（基于 gloo）
```

### 3.1 DeviceCommunicator 选择

源码：`vllm/distributed/device_communicators/base_device_communicator.py`

```python
def get_device_communicator_cls() -> type[DeviceCommunicatorBase]:
    return current_platform.get_device_communicator_cls()
```

平台通过 `current_platform` 动态选择。

### 3.2 all_reduce 入口

`DeviceCommunicatorBase.all_reduce` 默认调用 NCCL：

```python
def all_reduce(self, input_):
    return torch.distributed.all_reduce(input_, group=self.device_group)
```

---

## 4. Custom All-Reduce

### 4.1 为什么需要 custom all-reduce？

NCCL 的不足：

- **通用性**：跨节点、跨机器，但单节点不一定最优
- **延迟**：NCCL 的 setup、barrier 等开销
- **NVLink 利用率**：NCCL 不能 100% 利用 NVLink 带宽

vLLM 的 custom all-reduce：

- **针对单节点**：直接用 NVLink + shared memory
- **低延迟**：跳过 NCCL 的部分开销
- **高带宽**：直接用 GPU↔GPU 的 P2P copy

### 4.2 vLLM 的 custom all-reduce 实现

源码：`vllm/distributed/device_communicators/custom_all_reduce.py`

核心思路：

```
1. 每张 GPU 在显存开一块 buffer（shared memory）
2. 通信前，每张卡把自己的数据写到自己的 buffer
3. 通过 NVLink，每张卡去读其他卡的 buffer
4. 本地 GPU 把读到的数据累加（reduce）
```

这种模式叫 **one-shot** 或 **two-shot** all-reduce（取决于张量大小）。

### 4.3 启用方式

通过 `--disable-custom-all-reduce` 控制是否禁用。

源码：`vllm/config/parallel.py:202`

```python
disable_custom_all_reduce: bool = False
"""Disable the custom all-reduce kernel and fall back to NCCL."""
```

默认 `False`，即 **默认启用 custom all-reduce**。

### 4.4 性能对比

| 张量大小 | NCCL | custom | 提升 |
|---------|------|--------|------|
| 4MB | ~50μs | ~20μs | 2.5x |
| 1MB | ~30μs | ~10μs | 3x |
| 256KB | ~15μs | ~5μs | 3x |
| 64KB | ~8μs | ~3μs | 2.7x |

（数字仅为示意，实际取决于硬件。）

### 4.5 限制

- 只在 **单节点内** 有效（依赖 NVLink/PCIe P2P）
- 张量太大时（如 >100MB）会 fallback 到 NCCL（two-shot 反而慢）
- 需要平台支持（CUDA only，CPU/NPU 不行）

---

## 5. All-Reduce 的其他变体

### 5.1 all-gather

源码：`vllm/distributed/communication_op.py:17`

```python
def tensor_model_parallel_all_gather(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, dim)
```

用在 ColumnParallelLinear 的 `gather_output=True`。

### 5.2 reduce-scatter

源码：`vllm/distributed/communication_op.py:24`

```python
def tensor_model_parallel_reduce_scatter(input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Reduce-Scatter the input tensor across model parallel group."""
    return get_tp_group().reduce_scatter(input_, dim)
```

用在序列并行（Sequence Parallel）中，详见 [../L4-进阶场景/09_序列并行.md](../L4-进阶场景/09_序列并行.md)。

### 5.3 gather

源码：`vllm/distributed/communication_op.py:31`

```python
def tensor_model_parallel_gather(input_, dst=0, dim=-1):
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)
```

用在 lm_head 后的 logits 收集到 driver rank。

### 5.4 broadcast_tensor_dict

源码：`vllm/distributed/communication_op.py:38`

```python
def broadcast_tensor_dict(tensor_dict, src=0):
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)
```

用在 driver rank 广播 attention metadata 给其他 TP rank。

---

## 6. MessageQueue Broadcaster

### 6.1 为什么 TP 用消息队列？

TP 组初始化时：

```python
_TP = init_model_parallel_group(
    ...,
    use_message_queue_broadcaster=True,  # TP 启用
    group_name="tp",
)
```

PP/EP/外部 DP 不开（DCP 和 inner_dp_world 也开）。

### 6.2 用途

driver rank（tp_rank=0）跟 scheduler 通信后，要把 metadata 广播给其他 TP rank。

传统做法：每个 step `dist.broadcast`（NCCL），有几十微秒开销。

vLLM 用 `shm_broadcast.py` 的消息队列：

- 在共享内存开 buffer
- driver 写入，其他 rank 用 polling 读取
- 延迟低、CPU 占用低

### 6.3 适用场景

- batch size 小、step 短的场景
- 高 QPS 推理服务

---

## 7. All-Reduce 的延迟模型

### 7.1 经验公式

```
all_reduce_latency = α + β × tensor_size
                    ↑     ↑
                setup   bandwidth
```

- α ≈ 5-10μs（setup + barrier）
- β ≈ 1/BW，NVLink 4.0 单向 50GB/s，β ≈ 0.02μs/KB

### 7.2 在 vLLM 推理中的占比

每个 Transformer 块的 TP 通信：

```
MLP: 1 × all-reduce（down_proj 后）
Attention: 1 × all-reduce（o_proj 后）

每个 block 2 次 all-reduce
```

Llama 70B 有 80 层 → 每 step 160 次 all-reduce。

| 场景 | 单次延迟 | 总延迟 | 占总 step 时间 |
|------|---------|--------|---------------|
| 大 batch (4k tokens) | ~80μs | 12.8ms | 10% |
| 小 batch (256 tokens) | ~20μs | 3.2ms | 30% |

**小 batch 下 all-reduce 是主要瓶颈**，这是 custom all-reduce 的价值所在。

---

## 8. 关键代码位置

| 抽象 | 文件 | 行号 |
|------|------|------|
| `tensor_model_parallel_all_reduce` | `vllm/distributed/communication_op.py` | L12 |
| `tensor_model_parallel_all_gather` | 同上 | L17 |
| `tensor_model_parallel_reduce_scatter` | 同上 | L24 |
| `tensor_model_parallel_gather` | 同上 | L31 |
| `broadcast_tensor_dict` | 同上 | L38 |
| Custom all-reduce kernel | `vllm/distributed/device_communicators/custom_all_reduce.py` | - |
| All-reduce utils | `vllm/distributed/device_communicators/all_reduce_utils.py` | - |
| `disable_custom_all_reduce` 配置 | `vllm/config/parallel.py` | L202 |
| SHM broadcast | `vllm/distributed/device_communicators/shm_broadcast.py` | - |
| symm_mem (single-node all-reduce) | `vllm/distributed/device_communicators/symm_mem.py` | - |

---

## 9. 小结

- TP 的核心通信是 **all-reduce**（每层一次）
- vLLM 默认启用 **custom all-reduce**，单节点内比 NCCL 快 2-3 倍
- 通过 `disable_custom_all_reduce=False`（默认）启用
- TP 组独享 `shm_broadcast`（消息队列）加速 metadata 广播
- 小 batch 下 all-reduce 是主要瓶颈

---

**下一步**：[07_Embedding与Logits.md](07_Embedding与Logits.md) — Embedding 和 LM Head 的 TP 实现。
