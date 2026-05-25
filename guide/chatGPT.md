# 大模型推理 + 算子开发学习路线

## 一、目标定位

你的目标不是：

* 纯 AI 应用开发
* 纯 CUDA Kernel 开发

而是：

> 推理框架 + Runtime + 算子优化 + 推理系统

对应岗位：

* LLM Inference Engineer
* AI Infra Engineer
* 推理引擎工程师
* 推理优化工程师

核心能力：

```text
模型结构
+ 推理 Runtime
+ KV Cache
+ CUDA/Triton
+ Kernel优化
+ Memory管理
+ 调度系统
+ Serving
```

---

# 二、推荐总路线（核心）

```text
Transformer
    ↓
llama.cpp
    ↓
ggml Runtime
    ↓
CUDA基础
    ↓
核心算子开发
    ↓
FlashAttention
    ↓
vLLM
    ↓
TensorRT-LLM
```

---

# 三、为什么从 llama.cpp 开始

## 推荐项目

* llama.cpp
* ggml

### 项目地址

* llama.cpp
  [https://github.com/ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)

* ggml
  [https://github.com/ggml-org/ggml](https://github.com/ggml-org/ggml)

---

## 为什么最适合“推理 + 算子”

相比 vLLM：

| llama.cpp   | vLLM   |
| ----------- | ------ |
| 结构清晰        | 工程复杂   |
| 容易跟源码       | 调度逻辑重  |
| 推理链路完整      | 分布式复杂  |
| Runtime容易理解 | 初学容易迷失 |

---

## 能学到什么

### 推理链路

```text
用户输入
→ tokenize
→ embedding
→ transformer
→ logits
→ sampling
→ next token
→ KV Cache更新
→ 下一轮decode
```

### Runtime

* graph execution
* tensor layout
* memory allocator
* backend abstraction
* scheduler

### 算子

* matmul
* rmsnorm
* rope
* softmax
* attention

---

# 四、第一阶段：推理链路

## 项目

### llama.cpp

重点函数：

```c
llama_decode_internal()
```

---

## 目标

理解：

### Prefill

```text
一次性处理 prompt
生成 KV Cache
```

### Decode

```text
每次只生成一个 token
复用 KV Cache
```

---

## 必须掌握

| 内容             | 重要度   |
| -------------- | ----- |
| Token Flow     | ★★★★★ |
| KV Cache       | ★★★★★ |
| Sampling       | ★★★★  |
| Prefill/Decode | ★★★★★ |

---

# 五、第二阶段：Runtime（非常重要）

## 项目

### ggml

重点模块：

```text
ggml_tensor
ggml_graph
ggml_backend
ggml_allocator
ggml_scheduler
```

---

## 学习重点

### Tensor

理解：

* shape
* stride
* dtype
* layout

---

### Graph

理解：

```text
op graph
→ execution graph
→ backend dispatch
```

---

### Allocator

理解：

* memory reuse
* tensor lifecycle
* 显存复用
* fragmentation

---

### Backend

理解：

* CPU backend
* CUDA backend
* Metal backend
* Vulkan backend

---

# 六、第三阶段：CUDA 基础

## 必学内容

| 内容                | 重要度   |
| ----------------- | ----- |
| thread/block/grid | ★★★★★ |
| warp              | ★★★★★ |
| shared memory     | ★★★★★ |
| memory coalescing | ★★★★★ |
| tensor core       | ★★★★★ |

---

## 推荐项目

### CUDA Samples

[https://github.com/NVIDIA/cuda-samples](https://github.com/NVIDIA/cuda-samples)

---

# 七、第四阶段：核心算子开发

## 不要一开始就学复杂 Attention

推荐顺序：

```text
MatMul
↓
RMSNorm
↓
RoPE
↓
Softmax
↓
Attention
```

---

# 八、核心算子详解

## 1. MatMul（最重要）

### 本质

```text
LLM = MatMul Machine
```

---

### 必学

* tile
* shared memory
* tensor core
* vectorization
* fp16/bf16/int8/int4

---

### 推荐项目

#### CUTLASS

[https://github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass)

---

## 2. RMSNorm

### 学习点

* reduction
* vectorized load/store
* fused op

---

## 3. RoPE

### 学习点

* rotary embedding
* tensor layout
* cache locality

---

## 4. Softmax

### 学习点

* reduction
* numerical stability
* fusion

---

# 九、第五阶段：Attention 优化

## 推荐路线

```text
naive attention
↓
flash attention
↓
paged attention
```

---

# 十、FlashAttention（核心）

## 项目

FlashAttention

[https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

---

## 核心思想

### Attention 的瓶颈：

不是 FLOPS

而是：

```text
HBM Memory Bandwidth
```

---

## 学习重点

* SRAM tiling
* kernel fusion
* memory movement
* online softmax

---

# 十一、第六阶段：工业推理框架

## 项目

### vLLM

[https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm)

---

## 学习重点

### Continuous Batching

```text
提升 GPU 利用率
```

---

### PagedAttention

```text
KV Cache 虚拟内存化
```

---

### Block Manager

```text
KV Cache allocator
```

---

### Scheduler

```text
请求调度
```

---

# 十二、第七阶段：生产级优化

## 项目

### TensorRT-LLM

[https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)

---

## 学习重点

* graph optimization
* kernel fusion
* quantization
* CUDA graph
* overlap execution

---

# 十三、真正重要的知识（核心）

## 推理优化真正难点

不是：

```text
kernel 能跑
```

而是：

```text
系统整体吞吐最优
```

---

## 真正核心能力

| 内容                  | 重要度   |
| ------------------- | ----- |
| KV Cache            | ★★★★★ |
| Memory Layout       | ★★★★★ |
| Continuous Batching | ★★★★★ |
| Scheduler           | ★★★★★ |
| Kernel Fusion       | ★★★★★ |
| CUDA Graph          | ★★★★  |
| Quantization        | ★★★★  |

---

# 十四、推荐源码阅读顺序

## 第一阶段

### llama.cpp

重点：

```text
llama_decode_internal
```

---

## 第二阶段

### ggml

重点：

```text
ggml_graph_compute
ggml_backend_sched
```

---

## 第三阶段

### FlashAttention

重点：

```text
memory movement
online softmax
tiling
```

---

## 第四阶段

### vLLM

重点：

```text
scheduler
paged attention
block manager
```

---

# 十五、不要踩的坑

## 不要一开始学：

* Agent
* RAG
* RLHF
* LoRA
* Prompt Engineering
* LangChain

这些对“推理 + 算子”帮助有限。

---

# 十六、最终应该形成的能力

真正重要的是：

# “GPU 上 token 流动的空间感”

包括：

```text
token
→ tensor
→ memory layout
→ kernel
→ cache
→ scheduler
→ next token
```

当你能在脑海中动态理解：

* tensor 如何布局
* KV Cache 如何增长
* attention 如何搬运数据
* kernel 如何调度 warp
* scheduler 如何组织 batch

你就真正进入：

# LLM 推理工程领域了。
