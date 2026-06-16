# 知识点 07：External LB 与 Hybrid LB

> 核心问题：vLLM 的 external_lb 和 hybrid_lb 模式是什么？什么时候用？
> 源码位置：`vllm/config/parallel.py`、`vllm/entrypoints/openai/`

---

## 1. 三种 DP LB 模式总览

| 模式 | 命令行 | 路由 | 节点内 | 节点间 |
|------|--------|------|--------|--------|
| 默认 | `--data-parallel-size N` | vLLM 内部 | vLLM LB | - |
| **external_lb** | `--data-parallel-external-lb` | 外部 LB | 外部 LB | 外部 LB |
| **hybrid_lb** | `--data-parallel-hybrid-lb` | 节点内 vLLM + 节点间外部 | vLLM LB | 外部 LB |

---

## 2. 默认模式

### 2.1 拓扑

```
单进程 vLLM CLI
   ├─ DP rank 0 (EngineCore + Worker)
   ├─ DP rank 1 (EngineCore + Worker)
   ├─ ...
   └─ DP rank N-1 (EngineCore + Worker)

API server (in 主进程)
   ↓ 接收 HTTP 请求
   ↓ 路由到 DP rank X
```

### 2.2 路由实现

vLLM 主进程的 API server 收到请求后：

```python
# 简化伪代码
async def generate(request):
    # 选 DP rank
    dp_rank = select_dp_rank()  # round-robin / least-loaded
    # 发送给对应的 EngineCore
    response = await engines[dp_rank].generate(request)
    return response
```

### 2.3 适用

- 单节点部署
- 简单易用
- DP rank 数较少

---

## 3. External LB 模式

### 3.1 拓扑

```
外部 LB（nginx / K8s Service / HAProxy）
   ↓
   ├─ Pod 1: vLLM 实例 1 (DP rank 0)
   ├─ Pod 2: vLLM 实例 2 (DP rank 1)
   ├─ ...
   └─ Pod N: vLLM 实例 N (DP rank N-1)

每个 Pod 是独立的 vLLM 进程/容器，有自己的 API server
```

### 3.2 启动方式

每个 pod 独立启动：

```bash
# Pod 1
VLLM_DP_SIZE=4 VLLM_DP_RANK=0 VLLM_DP_MASTER_IP=... \
vllm serve ... --tensor-parallel-size 2 --data-parallel-external-lb

# Pod 2
VLLM_DP_SIZE=4 VLLM_DP_RANK=1 VLLM_DP_MASTER_IP=... \
vllm serve ... --tensor-parallel-size 2 --data-parallel-external-lb
```

### 3.3 外部 LB 路由

外部 LB（如 nginx）配置：

```nginx
upstream vllm_cluster {
    server pod1:8000;
    server pod2:8000;
    server pod3:8000;
    server pod4:8000;
}

server {
    location /v1/completions {
        proxy_pass http://vllm_cluster;
    }
}
```

或者 K8s Service 自动负载均衡。

### 3.4 适用场景

源码：`vllm/config/parallel.py:147-152`

```python
"""This is useful for a "one-pod-per-rank"
wide-EP setup in Kubernetes. Supported only for MoE deployments"""
```

- **K8s wide-EP 部署**
- 只支持 MoE 模型（non-MoE 用独立 vLLM 实例即可）
- 一个 pod 一个 rank，便于扩缩容

### 3.5 限制

**只支持 MoE 模型**：

源码：`vllm/config/parallel.py:150`

```python
# non-MoE
# models should use independent vLLM instances without --data-parallel-*
# arguments.
```

non-MoE 模型不需要这个模式（直接用独立 vLLM 实例即可，因为没有跨 rank 通信）。

---

## 4. Hybrid LB 模式

### 4.1 拓扑

```
外部 LB（节点间）
   ↓
   ├─ 节点 1: vLLM 实例
   │     ├─ DP rank 0 (本地)
   │     ├─ DP rank 1 (本地)
   │     └─ vLLM 内部 LB 路由
   │
   ├─ 节点 2: vLLM 实例
   │     ├─ DP rank 2 (本地)
   │     ├─ DP rank 3 (本地)
   │     └─ vLLM 内部 LB 路由
   │
   └─ 节点 3: vLLM 实例
         └─ ...
```

### 4.2 设计意图

源码：`vllm/config/parallel.py:154-159`

```python
"""Enables running an AsyncLLM
and API server on a "per-node" basis where vLLM load balances
between local data parallel ranks, but an external LB balances
between vLLM nodes/replicas."""
```

- **节点内**：vLLM 内部 LB（高效，共享内存）
- **节点间**：外部 LB（如 nginx）

### 4.3 启动方式

```bash
# 节点 1
VLLM_DP_SIZE=4 VLLM_DP_RANK=0 VLLM_DP_START_RANK=0 \
vllm serve ... --tensor-parallel-size 2 --data-parallel-hybrid-lb

VLLM_DP_SIZE=4 VLLM_DP_RANK=1 \
vllm serve ... --tensor-parallel-size 2 --data-parallel-hybrid-lb

# 节点 2
VLLM_DP_SIZE=4 VLLM_DP_RANK=2 VLLM_DP_START_RANK=2 \
vllm serve ... --tensor-parallel-size 2 --data-parallel-hybrid-lb
```

### 4.4 适用场景

- 跨节点 wide-EP
- 节点内多 NPU（如每节点 8 NPU，内部 DP=2/4）
- 节点间通过外部 LB（K8s Service）

### 4.5 跟 external_lb 的区别

| 维度 | external_lb | hybrid_lb |
|------|-------------|-----------|
| 节点内 | 每 rank 独立 pod | vLLM 内部 LB |
| 节点间 | 外部 LB | 外部 LB |
| API server 数 | 每 rank 一个 | 每节点一个 |
| 路由延迟 | 高（每次都过外部 LB） | 低（节点内本地） |

---

## 5. External LB 的常见实现

### 5.1 Nginx

```nginx
upstream vllm {
    least_conn;
    server pod1:8000 max_fails=3 fail_timeout=30s;
    server pod2:8000 max_fails=3 fail_timeout=30s;
    ...
}
```

策略：

- **round-robin**：循环（默认）
- **least_conn**：最少连接（推荐）
- **ip_hash**：按 IP hash（保持会话）

### 5.2 Kubernetes Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: vllm-service
spec:
  selector:
    app: vllm
  ports:
  - port: 8000
  type: LoadBalancer
```

K8s 自动负载均衡（默认 iptables / IPVS）。

### 5.3 HAProxy / Envoy

类似 nginx，更专业的 LB 软件。

### 5.4 Service Mesh（Istio）

更复杂，支持服务发现、熔断、重试等。

---

## 6. 路由策略对比

### 6.1 vLLM 内部 LB

**优势**：

- 知道每个 DP rank 的负载（KV cache、batch size）
- 可以做精细的 least-loaded 路由
- 共享内存通信，低延迟

**劣势**：

- 单点 API server，吞吐受限
- 不能跨节点

### 6.2 外部 LB

**优势**：

- 可以横向扩展（多个 LB 实例）
- K8s 集成好
- 跨节点统一管理

**劣势**：

- 不知道 DP rank 内部状态（只能基于连接数）
- HTTP 转发开销

### 6.3 Hybrid LB

**优势**：

- 节点内：vLLM LB 精细控制
- 节点间：外部 LB 水平扩展

**劣势**：

- 部署复杂
- 需要正确划分节点边界

---

## 7. AsyncLLM 在 LB 中的角色

源码：`vllm/config/parallel.py:155-157`

```python
"""Enables running an AsyncLLM
and API server on a "per-node" basis"""
```

**AsyncLLM**：vLLM V1 的异步引擎，支持 async scheduling。

在 hybrid_lb 模式下：

- 每节点一个 AsyncLLM
- 节点内多 DP rank 通过 async scheduling 切换
- 节点间通过外部 LB

---

## 8. 配置参数详解

### 8.1 data_parallel_external_lb

```python
data_parallel_external_lb: bool = False
```

- 默认 `False`
- 启用时：每 rank 独立 vLLM 实例 + 外部 LB
- 隐式启用：当 `--data-parallel-rank` 显式提供时

### 8.2 data_parallel_hybrid_lb

```python
data_parallel_hybrid_lb: bool = False
```

- 默认 `False`
- 启用时：节点内 vLLM LB + 节点间外部 LB
- 需要配合 `--data-parallel-start-rank` 使用

### 8.3 data_parallel_start_rank

源码：`vllm/config/parallel.py:136`

```python
data_parallel_rank_local: int | None = None
"""Local rank of the data parallel group, set only in SPMD mode."""
```

用于 hybrid_lb：节点的起始 rank 编号。

---

## 9. 选 LB 模式的决策

```
单节点？
├─ 是 → 默认模式（vLLM 内部 LB）
└─ 否 → 跨节点
    ├─ MoE wide-EP？
    │   ├─ 是 →
    │   │   ├─ 每节点单 vLLM（多 DP rank）→ hybrid_lb
    │   │   └─ 每 rank 单 pod → external_lb
    │   └─ 否 →
    │       └─ 独立 vLLM 实例 + 外部 LB（无 DP 通信）
```

---

## 10. 监控与故障恢复

### 10.1 健康检查

```bash
# vLLM 健康检查端点
curl http://pod1:8000/health
```

外部 LB 据此剔除故障 pod。

### 10.2 优雅停机

```bash
# vLLM 优雅停机
kill -SIGTERM <pid>
# vLLM 等待当前请求完成，再退出
```

外部 LB 配合健康检查，自动摘除。

### 10.3 KV cache 跨 pod 迁移

实验性功能：`vllm/distributed/kv_transfer/`

允许请求在 pod 间迁移时携带 KV cache（避免重算 prefill）。

---

## 11. 关键代码位置

| 内容 | 文件 | 行号 |
|------|------|------|
| `data_parallel_external_lb` | `vllm/config/parallel.py` | L146 |
| `data_parallel_hybrid_lb` | 同上 | L153 |
| `data_parallel_backend` | 同上 | L144 |
| AsyncLLM | `vllm/v1/engine/async_llm.py` | - |
| API server | `vllm/entrypoints/openai/api_server.py` | - |
| KV transfer（实验） | `vllm/distributed/kv_transfer/` | - |

---

## 12. 小结

- **默认模式**：vLLM 内部 LB，单节点
- **external_lb**：每 rank 独立 vLLM，外部 LB 路由，K8s wide-EP
- **hybrid_lb**：节点内 vLLM LB + 节点间外部 LB
- external_lb 只支持 MoE（non-MoE 用独立 vLLM 实例即可）
- 选模式看部署拓扑（单/多节点）和模型类型（MoE/Dense）

---

**回顾全图**：[../00_知识地图.md](../00_知识地图.md) — DP 学习地图。
