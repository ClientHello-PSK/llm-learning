# 知识点 10：vllm-ascend 的 TP 适配

> 核心问题：华为 Ascend NPU 上，TP 实现跟上游 vLLM 有什么差异？

---

## 1. 主要差异概述

| 维度 | 上游 vLLM (CUDA) | vllm-ascend |
|------|-----------------|-------------|
| 通信库 | NCCL | **HCCL**（Huawei Collective Communication Library） |
| custom all-reduce | 默认启用 | **不启用**（用 HCCS / HCCL 替代） |
| Layer 切分 | 标准 Column/Row | 一致（共享代码） |
| GEMM | cuBLAS | **CubeCL** / Ascend C kernel |
| Attention | FlashAttention (CUDA) | **FlashAttention-Ascend** |
| MessageQueue 广播 | shm_broadcast | shm_broadcast（部分支持） |

---

## 2. HCCL 替代 NCCL

### 2.1 HCCL 是什么

HCCL（Huawei Collective Communication Library）：华为为 Ascend NPU 提供的集合通信库，对标 NCCL。

- 支持 all-reduce / all-gather / reduce-scatter / broadcast 等
- 利用 HCCS（Huawei Cache Coherent Switch）互联
- 跟 NCCL API 高度相似（torch.distributed 抽象层透明）

### 2.2 vllm-ascend 的接入

源码：`vllm-ascend/vllm_ascend/distributed/`（如有）

通过 `HcclDeviceCommunicator`（继承 `DeviceCommunicatorBase`），跟 NCCL 走同一个 `GroupCoordinator` 抽象。

### 2.3 性能特征

- HCCS 带宽：~200 GB/s（对标 NVLink 4.0 的 600 GB/s，约为其 1/3）
- all-reduce 延迟：单次 ~30-50μs（NCCL ~10-20μs）
- 大张量表现略逊，小张量差距小

### 2.4 配置

vllm-ascend 启动时，`device_communicator` 通过 `current_platform.get_device_communicator_cls()` 自动选 `HcclDeviceCommunicator`。

---

## 3. Custom All-Reduce 在 Ascend 上的状态

### 3.1 默认不启用

上游 vLLM 的 custom all-reduce 基于 CUDA + NVLink P2P，**Ascend NPU 不能直接用**。

vllm-ascend 处理：

- 通过 `disable_custom_all_reduce=True` 强制禁用
- 或者通过 platform check 跳过 custom all-reduce 初始化
- 退化为 HCCL 实现

### 3.2 vllm-ascend 的探索

vllm-ascend 仓库中有尝试用 Ascend C 实现等价的 fast all-reduce，但成熟度低于 CUDA 版本。

### 3.3 影响

- TP=8（单节点 8 NPU）时，all-reduce 延迟比上游 vLLM 大
- 小 batch 推理性能略差（受 all-reduce 延迟限制）

---

## 4. Linear 层的实现一致性

### 4.1 共享上游代码

vllm-ascend 复用上游 vLLM 的 `linear.py`：

- `ColumnParallelLinear`、`RowParallelLinear`、`QKVParallelLinear` 等都一样
- 只是底层 `quant_method.apply` 走 Ascend C kernel

### 4.2 量化适配

Ascend 上的量化方案（如 W8A8、FP8）用 Ascend C 实现：

- matmul 走 CubeCL（Ascend 的 GEMM 库）
- 量化/反量化用 Ascend 算子

源码：`vllm-ascend/vllm_ascend/quantization/`

### 4.3 cube vs vector 核心

Ascend NPU 有两类核心：

- **Cube 核心**：做矩阵乘法（对标 GPU tensor core）
- **Vector 核心**：做 elementwise / reduce（对标 GPU cuda core）

Linear 层的 GEMM 走 cube，all-reduce 走 vector + 通信。

---

## 5. Attention 的差异

### 5.1 FlashAttention-Ascend

源码：`vllm-ascend/vllm_ascend/attention/`

- 接口跟上游 `Attention` 一致
- 底层用 `torch_npu.fusion.flash_attention` 或自研 kernel
- 支持 PagedAttention（vLLM 的 KV cache 分页机制）

### 5.2 TP 切分逻辑

跟上游一致：

- Q/K/V 按 head 切（GQA 处理同上游）
- attention 内部不通信
- o_proj 后 all-reduce（HCCL）

### 5.3 MLA 的特殊处理

DeepSeek V3/V4 的 MLA 在 Ascend 上有特殊优化：

- 吸收矩阵的预计算
- decode 阶段的特殊 attention pattern

源码：`vllm-ascend/vllm_ascend/lora/ops/` 或 `vllm_ascend/attention/`

---

## 6. 编译优化的差异

### 6.1 vllm-ascend 的 torch.compile

Ascend NPU 通过 `torch_npu` 接入 PyTorch，支持 `torch.compile`：

- `inductor` 后端适配 Ascend
- 部分上游 fusion pass（如 all-reduce + RMSNorm）在 Ascend 上不一定可用

### 6.2 跳过的上游优化

| 上游优化 | Ascend 状态 |
|---------|-------------|
| custom all-reduce | 不启用 |
| symm_mem（NVLink 内 P2P） | 不适用 |
| all-reduce + RMSNorm 融合 | 部分支持 |
| sequence parallelism pass | 部分支持 |

---

## 7. 启动配置

### 7.1 推荐配置

```bash
# 单节点 8 NPU
vllm serve deepseek-ai/DeepSeek-V3 \
    --tensor-parallel-size 8 \
    --disable-custom-all-reduce \  # Ascend 必加
    ...
```

### 7.2 跨节点配置

```bash
# 2 节点 × 8 NPU，TP=8（单节点）, PP=2（跨节点）
vllm serve ... \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 2 \
    --disable-custom-all-reduce
```

跨节点通常用 PP（不用 TP），原因跟 CUDA 一样。

### 7.3 EP 的考虑

MoE 模型在 Ascend 上常用 EP：

- expert 用 EP（all-to-all via HCCL）
- attention 用 TP（all-reduce via HCCL）
- 两者在同一 HCCL communicator 上协调

---

## 8. vllm-ascend 的关键源码位置

| 模块 | 路径 |
|------|------|
| Platform 定义 | `vllm-ascend/vllm_ascend/platform.py` |
| HCCL communicator | `vllm-ascend/vllm_ascend/distributed/`（如有） |
| Attention layer | `vllm-ascend/vllm_ascend/attention/` |
| 量化 | `vllm-ascend/vllm_ascend/quantization/` |
| LoRA | `vllm-ascend/vllm_ascend/lora/` |
| MLP 优化 | `vllm-ascend/vllm_ascend/linear/`（如有） |

具体路径请以最新 vllm-ascend 仓库为准。

---

## 9. 已知问题与限制

### 9.1 all-reduce 性能

- HCCL 比 NCCL 慢，小 batch 推理 TP 通信开销占比大
- 建议小 batch 场景降低 TP size（如 TP=2/4 而非 8）

### 9.2 算子兼容性

- 部分 CUDA 算子需要 Ascend 等价实现
- 新模型上线时可能遇到 `OperatorNotFound` 错误

### 9.3 HCCS 拓扑

- HCCS 比 NVLink 慢，且拓扑依赖具体 NPU 型号
- 跨 die 通信（同节点内不同 die）性能可能差异大

---

## 10. 跟上游 vLLM 的差异总结

```
上游 vLLM (CUDA):
   - 通信：NCCL + custom all-reduce
   - GEMM：cuBLAS / CUTLASS
   - Attention：FlashAttention (CUDA)
   - 编译：torch.compile + inductor

vllm-ascend (NPU):
   - 通信：HCCL（无 custom all-reduce）
   - GEMM：CubeCL / Ascend C kernel
   - Attention：FlashAttention-Ascend
   - 编译：torch_npu + 部分上游 fusion
```

**Linear/Attention 切分逻辑跟上游一致**，只是底层算子和通信库替换。

---

## 11. 小结

- Ascend 用 HCCL 替代 NCCL，all-reduce 延迟略大
- custom all-reduce 在 Ascend 上**默认禁用**
- Linear/Attention 的 TP 切分逻辑跟上游一致（共享代码）
- 跨节点部署时优先 PP（跟上游一致）
- MoE 模型推荐 EP + TP 组合

---

**回顾全图**：[../00_知识地图.md](../00_知识地图.md) — TP 学习地图。
