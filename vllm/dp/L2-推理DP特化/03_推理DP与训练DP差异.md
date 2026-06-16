# 知识点 03：推理 DP 与训练 DP 的差异

> 核心问题：vLLM 是推理框架，它的 DP 跟训练 DP（DDP/FSDP）有什么本质差异？

---

## 1. 核心差异总览

| 维度 | 训练 DP（DDP/FSDP） | 推理 DP（vLLM） |
|------|---------------------|-----------------|
| **通信内容** | 梯度 all-reduce | **几乎无通信** |
| **通信时机** | 每 step backward 后 | **无** |
| **权重同步** | 必须（参数一致性） | 加载时一次同步 |
| **MoE 角色** | expert 复制 | **expert 切分**（借道 EP） |
| **同步模式** | 强同步（每 step） | 异步独立 |
| **请求路由** | 数据并行（dataset 切分） | **请求路由（每请求）** |
| **目标** | 训练吞吐（吞吐+收敛） | 推理吞吐 + 延迟 |

---

## 2. 差异一：无梯度通信

### 训练场景

DDP（Distributed Data Parallel）的核心：

```
每个 step:
   1. forward
   2. backward（产生梯度）
   3. all-reduce 梯度（DP rank 间通信）
   4. optimizer step

通信量: 模型参数大小 × 2（梯度 + 可能的参数同步）
```

### 推理场景

vLLM 推理 DP：

```
每个请求:
   1. 路由到某 DP rank
   2. forward
   3. 返回结果

通信: 0
```

**根本原因**：推理不更新参数，不需要梯度同步。

---

## 3. 差异二：权重不动态同步

### 训练场景

DDP 要求每 step 后所有 rank 权重一致：

```
所有 rank 都从相同参数起步
forward + backward + 梯度 all-reduce + 优化器更新
所有 rank 同时更新，保证权重一致
```

如果不一致，训练就发散了。

### 推理场景

推理权重是 **加载后静态**：

```
启动时加载权重 → 每个 DP rank 加载同一份权重
之后再也不变
```

**vLLM 的实现**：

- 启动时广播权重（broadcast from rank 0）
- 或者每个 rank 独立加载（更省内存）

**特殊情况：在线学习 / 持续训练**：需要同步权重，但 vLLM 默认不支持。

---

## 4. 差异三：MoE 中的角色反转

### 训练 DP + MoE

训练 DP 下，MoE 的 expert 通常是 **完整复制**：

```
DP rank 0: 完整 expert 0, 1, 2, ...
DP rank 1: 完整 expert 0, 1, 2, ...
DP rank 2: 完整 expert 0, 1, 2, ...
```

每个 DP rank 都有所有 expert。

### 推理 DP + MoE

源码：`vllm/config/parallel.py:127-128`

```python
data_parallel_size: int = Field(default=1, ge=1)
"""Number of data parallel groups. MoE layers will be sharded according to
the product of the tensor parallel size and data parallel size."""
```

**注释明确**：MoE 层按 **TP × DP** 切分。

```
Llama 4 / DeepSeek V4 MoE 部署:
   TP=2, DP=4 → expert 跨 8 个 rank 切分（= EP=8）
   
   expert 0: rank 0
   expert 1: rank 1
   ...
   expert 7: rank 7
   
   每 rank 一个 expert（或一组 expert）
```

**含义**：推理 DP 跟 EP 共享 layout，DP 增加 = EP 增加 = expert 分得更细。

### 4.1 为什么可以这么做？

训练需要每个 DP rank 算完整梯度（梯度 all-reduce 需要完整 expert）。

推理不需要梯度，每个 token 只路由到一个 expert。所以 expert 跨 DP rank 切分是可行的（token 通过 all-to-all 路由到对应 rank）。

### 4.2 这就是 wide-EP 部署

详见 [../L4-进阶场景/06_DP+EP_wide.md](../L4-进阶场景/06_DP+EP_wide.md)。

---

## 5. 差异四：请求路由 vs 数据切分

### 训练场景

数据并行：dataset 切分成 N 份，每 DP rank 处理一份。

```
dataset [0..1M] → DP=4:
   rank 0: [0..250k)
   rank 1: [250k..500k)
   rank 2: [500k..750k)
   rank 3: [750k..1M)
```

每个 rank 处理固定数据，**预先确定**。

### 推理场景

请求**动态到达**：

```
时刻 t=0: 请求 A 到达
   scheduler 决定: A → DP rank 0

时刻 t=1: 请求 B 到达
   scheduler 决定: B → DP rank 2

时刻 t=2: 请求 C 到达
   scheduler 决定: C → DP rank 1
```

**关键**：路由是**实时决策**，需要负载均衡。

### 5.1 vLLM 的路由策略

vLLM 的默认路由（单实例多 DP rank）：

- scheduler 维护全局请求队列
- 每个 DP rank 有独立的 KV cache / batch
- scheduler 根据负载、KV cache 占用、batch size 决定路由

### 5.2 外部 LB 模式

多容器/多 pod 部署时，外部 LB（如 nginx、Kubernetes Service）做路由：

- 轮询 / 加权轮询
- 最少连接数
- 延迟优先

---

## 6. 差异五：MoE 下的 DP 同步

### 非 MoE 模型

DP rank 完全独立，无同步：

```
rank 0: forward 请求 A
rank 1: forward 请求 B
两者完全独立，可以并行
```

### MoE 模型

DP rank（作为 EP rank）需要 all-to-all 通信：

```
rank 0: token 1 需要expert 3（在 rank 3）
        token 2 需要expert 0（在 rank 0）

all-to-all:
   rank 0 把 token 1 发给 rank 3
   rank 0 接收其他 rank 发来的、需要 expert 0 的 token
   每 rank 算自己 expert
   再 all-to-all 把结果发回
```

**同步要求**：同 EP（=DP）组的 rank 必须**同步 forward**，否则 all-to-all 死锁。

### 6.1 vLLM 的注释

源码：`vllm/distributed/parallel_state.py:1745-1746`

```
# all the ranks in the same DP group should generate simultaneously,
# i.e. the `generate` call in the same DP group should be called together,
# otherwise it will cause deadlock.
```

**含义**：MoE 模型下，DP 组的死锁风险。

### 6.2 怎么避免死锁

- scheduler 同步驱动所有 DP rank
- 每个 step 所有 DP rank 同时 forward
- async scheduling 下需要小心（这就是 vLLM 区分 ExternalDP 和 DP 的原因）

---

## 7. 差异六：KV cache 的存储

### 训练场景

训练时**没有 KV cache 概念**。

### 推理场景

每个 DP rank 维护自己处理的请求的 KV cache：

```
rank 0: 请求 [A, B, C] 的 KV cache
rank 1: 请求 [D, E] 的 KV cache
```

**关键**：

- KV cache **不在 DP rank 间共享**
- 请求路由到某个 rank 后，整个生命周期都在那个 rank
- KV cache 大小 = 单副本的 KV cache size

---

## 8. 总结对比

| 维度 | 训练 DP（DDP/FSDP） | 推理 DP（vLLM） |
|------|---------------------|-----------------|
| 通信 | 每 step 梯度 all-reduce | 无（非 MoE） / all-to-all（MoE） |
| 权重 | 每 step 同步 | 加载时同步 |
| MoE 角色 | expert 复制 | **expert 切分（借道 EP）** |
| 请求路由 | 数据集切分 | 实时路由 |
| 同步模式 | 强同步 | 异步独立（非 MoE） |
| KV cache | 无 | per-rank 维护 |

---

## 9. 学习提示

理解推理 DP 的关键：

1. **不要套用 DDP/FSDP 的所有概念**：DDP 的梯度 all-reduce、FSDP 的参数 sharding 都不适用
2. **关注 MoE 的角色反转**：DP 在 MoE 下变成 EP，是最重要的差异
3. **关注请求路由**：是推理 DP 调度的核心
4. **关注 KV cache 隔离**：是推理 DP 内存模型的核心

---

**下一步**：[../L3-vLLM实现/04_DP架构与启动.md](../L3-vLLM实现/04_DP架构与启动.md) — vLLM 怎么启动 DP 模式？
