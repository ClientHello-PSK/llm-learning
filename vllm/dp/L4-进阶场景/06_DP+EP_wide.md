# 知识点 06：DP + EP Wide-EP 部署

> 核心问题：什么是 wide-EP？为什么 DeepSeek/Kimi 等大 MoE 模型要用这种部署？

---

## 1. Wide-EP 是什么

Wide-EP = **Wide Expert Parallelism**（宽专家并行）

含义：把 EP size 设置得非常大（如 32、64、128），让每个 expert 在很多 rank 上独立存在。

```
普通 EP：
   EP=8, 256 experts
   每 rank 32 experts

Wide-EP：
   EP=64, 256 experts
   每 rank 4 experts
   （expert 分得很细，跨很多 rank）
```

---

## 2. 为什么需要 Wide-EP

### 2.1 MoE 模型的显存压力

DeepSeek V4 671B 的 expert 权重：

```
256 experts × [hidden × intermediate] = 256 × [2048, 7168] × 2bytes
            ≈ 7.4 TB
```

远超任何单机显存（A100 80GB × 8 = 640GB）。

### 2.2 EP 切分降低单卡显存

```
EP=8:  每 rank 32 experts, 920 GB / 8 = 115 GB/rank  (仍超)
EP=64: 每 rank 4 experts, 920 GB / 64 = 14 GB/rank   (可接受)
```

Wide-EP 让单卡 expert 权重显存可控。

### 2.3 跟 TP 配合

attention 部分用 TP（按 head 切），expert 部分用 EP（按 expert 切）：

```
TP=2 (单节点 NVLink)
EP=64 (跨 32 节点 × 2 TP)

每节点：2 个 NPU
   - attention：TP=2 切 head
   - expert：EP=64 切 expert
```

---

## 3. DP 在 Wide-EP 中的角色

源码：`vllm/config/parallel.py:127-128`

```python
data_parallel_size: int = Field(default=1, ge=1)
"""Number of data parallel groups. MoE layers will be sharded according to
the product of the tensor parallel size and data parallel size."""
```

**MoE 层按 TP × DP 切分**。

所以 wide-EP 部署通常用：

```
EP_effective = TP × DP

例：TP=2, DP=32
   EP_effective = 64
```

**含义**：在 wide-EP 模式下，DP 增加就是 EP 增加。

---

## 4. Wide-EP 的部署拓扑

### 4.1 跨节点 wide-EP

```
节点 1: rank 0, 1 (TP=2)
节点 2: rank 2, 3
节点 3: rank 4, 5
...
节点 32: rank 62, 63

EP_effective = 64（跨 32 节点 × TP=2）
```

### 4.2 K8s 部署（每 pod 一个 rank）

源码：`vllm/config/parallel.py:147-152`

```python
data_parallel_external_lb: bool = False
"""This is useful for a "one-pod-per-rank"
wide-EP setup in Kubernetes. Supported only for MoE deployments"""
```

```
Pod 1: rank 0
Pod 2: rank 1
...
Pod 64: rank 63

外部 LB 路由请求
```

---

## 5. Wide-EP 的通信开销

### 5.1 All-to-all 延迟

EP size 增加 → all-to-all 延迟增加：

```
EP=8:    all-to-all ~50μs (单节点)
EP=32:   all-to-all ~200μs (跨节点)
EP=64:   all-to-all ~400μs (跨多节点)
EP=128:  all-to-all ~800μs
```

### 5.2 跨节点带宽

普通以太网（100Gbps）：

- 单次 all-to-all 64MB → 5ms
- 远高于单节点 NVLink（200GB/s）的 0.3ms

需要 RoCEv2 / InfiniBand 等高速网络。

### 5.3 vLLM 的优化

源码：`vllm/config/parallel.py:185-195`

```python
all2all_backend: All2AllBackend = "allgather_reducescatter"
"""All2All backend for MoE expert parallel communication. Available options:

- "allgather_reducescatter": All2all based on allgather and reducescatter
- "deepep_high_throughput": Use deepep high-throughput kernels
- "deepep_low_latency": Use deepep low-latency kernels
- "mori_high_throughput": MoRI EP with InterNodeV1 for multi-node
- "mori_low_latency": MoRI EP with InterNodeV1LL for multi-node
- "nixl_ep": Use nixl-ep kernels
- "flashinfer_nvlink_two_sided": ...
- "flashinfer_nvlink_one_sided": ...
"""
```

不同后端针对不同场景：

- **deepep_high_throughput**：高吞吐，适合大 batch
- **deepep_low_latency**：低延迟，适合小 batch / 实时
- **mori_high_throughput / mori_low_latency**：跨节点优化
- **nixl_ep**：基于 NIXL 的 P2P

---

## 6. Wide-EP 的 batch 处理

### 6.1 小 batch 的麻烦

EP 大时，单 rank 的 token 数少：

```
batch=64, EP=64
   每 rank 平均 1 个 token
```

expert 计算的 GEMM 退化成 GEMV，效率低。

### 6.2 解决：大 batch

Wide-EP 适合大 batch 推理：

```
batch=2048, EP=64
   每 rank 平均 32 个 token
   expert GEMM 高效
```

DeepSeek 服务部署：通常 batch 数千 token，用 wide-EP。

### 6.3 decode 阶段的特殊处理

decode 时 batch 中每个请求只产生 1 token：

```
batch=2048 请求, decode 阶段:
   每 step 2048 个 token
   EP=64, 每 rank 平均 32 token（路由不均的话可能更少）
```

vLLM 的 deepep_low_latency 专门优化 decode。

---

## 7. DeepEP 优化

### 7.1 DeepEP 是什么

DeepSeek 开源的高性能 EP 通信库：

- **High Throughput 模式**：批处理优先，大 batch 友好
- **Low Latency 模式**：延迟优先，小 batch / decode 友好

### 7.2 跟标准 NCCL all-to-all 的区别

| 维度 | NCCL all-to-all | DeepEP |
|------|----------------|--------|
| 内核 | 通用 | MoE 专用 |
| Token routing | 普通 | 预计算 dispatch plan |
| Buffer 管理 | 用户 | 内部优化 |
| 跨节点 | TCP/IB | RDMA + 优化 |

### 7.3 vLLM 集成

通过 `--all2all-backend deepep_high_throughput` 或 `deepep_low_latency` 启用。

---

## 8. Wide-EP 的调度挑战

### 8.1 全局 batch 同步

所有 EP rank 必须**同时** forward（all-to-all 同步要求）。

源码：`vllm/distributed/parallel_state.py:1745-1746`（死锁注释）

### 8.2 Token routing 平衡

token 路由由 model 决定，可能不均匀：

```
某 step 大量 token 都路由到 expert 5
expert 5 所在 rank 负载重，其他 rank 空闲
```

解决：

- **expert load balancing**（训练时优化 routing）
- **EPLB（Expert Parallelism Load Balancing）**：动态调整 expert 位置

源码：`vllm/config/parallel.py:171`

```python
enable_eplb: bool = False
"""Enable expert parallelism load balancing for MoE layers."""
```

### 8.3 弹性 EP

源码：`vllm/config/parallel.py:205`

```python
enable_elastic_ep: bool = False
"""Enable elastic expert parallelism with stateless NCCL groups for DP/EP."""
```

允许动态扩缩容 EP size（增减节点时）。

---

## 9. 实际部署示例

### 9.1 DeepSeek V4 671B 单区域

```
节点数: 32
每节点 NPU: 2
TP=2 (节点内), EP=64 (跨节点)

PP=4 (跨 stage), DP=2 (双副本)
```

### 9.2 K8s wide-EP

```
64 pods, 每 pod 一个 NPU + 一个 vLLM 实例
TP=1 (单 pod), EP=64 (跨 pod)

external_lb=True
K8s service 路由请求
```

---

## 10. Wide-EP 的成本

| 维度 | 成本 |
|------|------|
| 网络带宽 | 高（all-to-all 跨节点） |
| 节点数 | 多（EP size 大） |
| Expert 切分 | 每节点 expert 数少 |
| 通信延迟 | all-to-all ~ms 级 |
| 同步开销 | 每 step 所有 rank 同步 |

---

## 11. 适用场景

| 场景 | 是否适合 wide-EP |
|------|------------------|
| 大 MoE 模型（DeepSeek V4、Kimi K2） | **适合** |
| Dense 模型 | 不适合（用普通 DP） |
| 小 batch / 实时 | 用 low_latency 模式 |
| 大 batch / 吞吐 | 用 high_throughput 模式 |
| 跨节点网络 | 需要 RoCEv2 / IB |

---

## 12. 关键代码位置

| 内容 | 文件 | 行号 |
|------|------|------|
| MoE 按 TP × DP 切分注释 | `vllm/config/parallel.py` | L127-128 |
| `data_parallel_external_lb` | 同上 | L146 |
| `all2all_backend` | 同上 | L185 |
| `enable_eplb` | 同上 | L171 |
| `enable_elastic_ep` | 同上 | L205 |
| All-to-all 实现 | `vllm/distributed/device_communicators/all2all.py` | - |
| 弹性 EP | `vllm/distributed/elastic_ep/` | - |
| EPLB | `vllm/distributed/eplb/` | - |

---

## 13. 小结

- Wide-EP：EP size 很大（64+），expert 分得很细
- MoE 层按 TP × DP 切分 → DP 增加 = EP 增加
- K8s 部署：每 pod 一个 rank + external_lb
- 通信开销大，需要 DeepEP 等优化
- 适合大 MoE 模型 + 大 batch 场景

---

**下一步**：[07_DP_external_lb.md](07_DP_external_lb.md) — external_lb / hybrid_lb 模式详解。
