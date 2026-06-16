# 知识点 02：进程组 Layout

> 核心问题：vLLM 怎么把 DP/PP/TP/EP 排进 world_size 个 rank？
> 源码位置：`vllm/distributed/parallel_state.py:1740`（layout 注释）

---

## 1. Layout 的核心注释

源码：`vllm/distributed/parallel_state.py:1740`

```python
# the layout order is: ExternalDP x DP x PP x PCP x TP
# ExternalDP is the data parallel group that is not part of the model,
# every dp rank can generate independently (in verl integration).
# DP is the data parallel group that is part of the model,
# all the ranks in the same DP group should generate simultaneously,
# i.e. the `generate` call in the same DP group should be called together,
# otherwise it will cause deadlock.
```

5 个维度（从外到内）：

```
ExternalDP × DP × PP × PCP × TP
```

---

## 2. 各维度的含义

### 2.1 ExternalDP（最外层）

- **不在 vLLM 模型内**，是外部多进程/多容器
- 每个 ExternalDP rank 是独立的 vLLM 实例
- 典型场景：verl 训练框架集成、K8s 多 pod 部署

### 2.2 DP（vLLM 内部 DP）

- vLLM 模型内的数据并行
- **同 DP 组的所有 rank 必须同时 generate**（否则死锁）
- 这是 vLLM DP 跟 ExternalDP 的关键区别

### 2.3 PP（流水线并行）

- 切层
- 同 PP 组的 rank 沿流水线传递 activation

### 2.4 PCP（Prefill Context Parallel）

- prefill 阶段的 sequence 切分
- 主要用于长序列 prefill 跨节点

### 2.5 TP（最内层）

- 切矩阵
- 同 TP 组的 rank 相邻（同节点内）

---

## 3. 为什么是这个顺序？

### 3.1 TP 在最内层

源码：

```python
all_ranks = torch.arange(world_size).reshape(
    -1,
    data_parallel_size,          # 维 1：DP
    pipeline_model_parallel_size, # 维 2：PP
    prefill_context_model_parallel_size,  # 维 3：PCP
    tensor_model_parallel_size,   # 维 4：TP（最内）
)
```

TP 在最内层意味着**同一 TP 组的 rank 是相邻的**：

```
world_size = 8, TP=2:

ranks [0, 1] 是一组 TP
ranks [2, 3] 是一组 TP
ranks [4, 5] 是一组 TP
ranks [6, 7] 是一组 TP
```

相邻 = 同节点 = NVLink/HCCS 互联。这符合"TP 不跨节点"的物理要求。

### 3.2 PP 在 TP 外

PP 跨 stage 需要 send/recv，可以在节点间走（PP 不依赖高速互联）。

PP 在 TP 外意味着：同 PP 组的不同 stage，其 TP rank 是相邻编号的：

```
TP=2, PP=2:

PP rank 0: ranks [0, 1]
PP rank 1: ranks [2, 3]

PP 组（同 TP rank 跨 stage）:
   PP rank 0 的 TP rank 0 = global rank 0
   PP rank 1 的 TP rank 0 = global rank 2
```

### 3.3 DP 在 PP 外

DP 是模型副本，每个副本包含完整 PP+TP 结构：

```
TP=2, PP=2, DP=2 (8 卡):

DP rank 0:
   PP rank 0: ranks [0, 1]   (TP)
   PP rank 1: ranks [2, 3]   (TP)

DP rank 1:
   PP rank 0: ranks [4, 5]   (TP)
   PP rank 1: ranks [6, 7]   (TP)
```

### 3.4 ExternalDP 在 DP 外

ExternalDP 跟 DP 的区别：

- ExternalDP：不在 vLLM 进程组内，独立进程
- DP：vLLM 进程组内，同组需要同步 generate

---

## 4. 进程组的生成

源码：`vllm/distributed/parallel_state.py:1833-1848`

```python
global _DP
assert _DP is None, "data parallel group is already initialized"
group_ranks = all_ranks.transpose(1, 4).reshape(-1, data_parallel_size).unbind(0)
group_ranks = [x.tolist() for x in group_ranks]
if enable_elastic_ep:
    _DP = _init_stateless_group(...)
else:
    _DP = init_model_parallel_group(
        group_ranks, get_world_group().local_rank, backend, group_name="dp"
    )
```

### 4.1 transpose 操作的含义

`all_ranks.transpose(1, 4)` 把 DP 维（维 1）和 TP 维（维 4）交换：

```
原: [ExternalDP, DP, PP, PCP, TP]
后: [ExternalDP, TP, PP, PCP, DP]
```

然后 reshape 到 2D，最后一维是 DP → unbind 得到所有 DP 组。

### 4.2 例：DP=2, PP=2, TP=2 (8 卡)

```
all_ranks (reshape 到 [1, 2, 2, 1, 2]):
[
  [  # ExternalDP 维（只有 1）
    [  # DP 维
      [[0, 1]],   # PP 维 0, PCP 维 0, TP 维 [0, 1]
      [[2, 3]],   # PP 维 1, PCP 维 0, TP 维 [0, 1]
    ],
    [  # DP 维 1
      [[4, 5]],
      [[6, 7]],
    ]
  ]
]

transpose(1, 4) → 交换 DP 和 TP 维
reshape 到 [4, 2] → 每个 DP 组大小 2
unbind(0) → 4 个 DP 组：
   组 0: [0, 4]   ← DP rank 0 在 ExternalDP 0, PP 0, TP 0 = rank 0
                       ExternalDP 0, PP 0, TP 1 = rank 1? 
                       不对，应该是同 TP rank 跨 DP
```

具体 transpose 后的逻辑较复杂，关键是：**DP 组的成员是同 TP rank、跨 DP rank 的 ranks**。

---

## 5. 每张卡属于哪些组？

例：world_size=8, TP=2, PP=2, DP=2

每张卡 global rank ∈ [0, 7]，每张卡属于：

| 组类型 | 卡 0 的组成员 | 卡 4 的组成员 |
|--------|--------------|--------------|
| TP 组 | [0, 1] | [4, 5] |
| PP 组 | [0, 2] | [4, 6] |
| DP 组 | [0, 4] | [0, 4] |
| World 组 | [0..7] | [0..7] |

- TP 组：跨 TP rank（同 DP、同 PP）
- PP 组：跨 PP rank（同 DP、同 TP rank）
- DP 组：跨 DP rank（同 PP、同 TP rank）

---

## 6. EP 组的特殊性

源码：`vllm/distributed/parallel_state.py:1850-1876`

EP 组的 layout：

```python
group_ranks = (
    all_ranks.transpose(1, 2)  # 交换 DP 和 PP 维
    .reshape(
        -1,
        data_parallel_size
        * prefill_context_model_parallel_size
        * tensor_model_parallel_size,
    )
    .unbind(0)
)
```

EP 组大小 = `DP × PCP × TP`，意味着 **EP 跨 DP、PCP、TP rank**。

**含义**：在 MoE 模型下，DP 增加会自然变成 EP 增加（expert 在更大范围内切分）。

---

## 7. external_dp 与 hybrid_lb

源码：`vllm/config/parallel.py:146-159`

```python
data_parallel_external_lb: bool = False
"""Whether to use "external" DP LB mode. Applies only to online serving
and when data_parallel_size > 0. ..."""

data_parallel_hybrid_lb: bool = False
"""Whether to use "hybrid" DP LB mode. ... Enables running an AsyncLLM
and API server on a "per-node" basis where vLLM load balances
between local data parallel ranks, but an external LB balances
between vLLM nodes/replicas."""
```

### 7.1 三种 LB 模式

| 模式 | 含义 |
|------|------|
| 默认 | 单个 vLLM 实例，内部 DP rank LB |
| external_lb | 每个 DP rank 独立 vLLM 实例，外部 LB 路由 |
| hybrid_lb | 节点内 LB + 节点间 external LB |

详见 [../L4-进阶场景/07_DP_external_lb.md](../L4-进阶场景/07_DP_external_lb.md)。

---

## 8. 多维并行的实例

### 8.1 8 卡典型配置

```
TP=8, DP=1:       单副本，全用于 TP
TP=4, DP=2:       两副本，每副本 TP=4
TP=2, DP=4:       四副本，每副本 TP=2
TP=1, DP=8:       八副本（纯 DP）
```

### 8.2 16 卡 DeepSeek V4 案例

```
PP=2, EP=4, TP=2, DP=1:
   - 单副本
   - 跨 2 节点（PP 跨节点）
   - 节点内 TP=2
   - EP=4 跨剩余卡

PP=2, EP=4, TP=2, DP=2:
   - 双副本
   - 每副本同上
```

---

## 9. layout 的物理意义

```
最外层 ─────────────────► 最内层
ExternalDP   DP    PP   PCP   TP
   │         │     │     │    │
跨进程    跨副本  跨stage  跨seq 跨矩阵
   │         │     │     │    │
独立进程  同步生成  send/recv all2all all-reduce
   │         │     │     │    │
无通信    几乎无    低带宽  中带宽 高带宽
```

**带宽需求从外到内递增**，所以高速互联（NVLink）给最内层（TP），低速互联（以太网）给最外层（ExternalDP）。

---

## 10. 关键代码位置

| 内容 | 文件 | 行号 |
|------|------|------|
| layout 注释 | `vllm/distributed/parallel_state.py` | L1740 |
| `all_ranks` reshape | 同上 | L1749-1755 |
| TP 组创建 | 同上 | L1758-1772 |
| PP 组创建 | 同上 | L1816-1831 |
| DP 组创建 | 同上 | L1833-1848 |
| EP 组创建 | 同上 | L1850-1876 |
| `data_parallel_size` 配置 | `vllm/config/parallel.py` | L126 |
| `data_parallel_external_lb` | 同上 | L146 |
| `data_parallel_hybrid_lb` | 同上 | L153 |

---

## 11. 小结

- layout 顺序：`ExternalDP × DP × PP × PCP × TP`
- TP 最内层（保证同节点、低延迟）
- DP 外层（不需要高速互联，跨副本独立）
- EP 组跨 DP+PCP+TP（MoE 时 DP 增加 = EP 增加）
- external_lb / hybrid_lb 是不同部署模式

---

**下一步**：[../L2-推理DP特化/03_推理DP与训练DP差异.md](../L2-推理DP特化/03_推理DP与训练DP差异.md) — 推理 DP 跟训练 DP 有什么本质差异？
