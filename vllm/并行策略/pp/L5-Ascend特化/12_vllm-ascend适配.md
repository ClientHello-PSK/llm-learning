# 知识点 12：vllm-ascend 的 PP 适配

> 核心问题：vllm-ascend 跟上游 vLLM 在 PP 实现上有哪些差异？
> 相关代码：`vllm_ascend/worker/model_runner_v1.py:255`（NPUModelRunner）、`vllm_ascend/distributed/`

---

## 1. vllm-ascend 的定位

vllm-ascend 是 vLLM 在华为 Ascend NPU 平台的硬件插件。

```
vLLM（上游）
    ↓ 通过 plugin 接口
vllm-ascend（华为维护）
    ↓
Ascend NPU（910B/910C 等）
```

vllm-ascend 不重新实现 vLLM，而是**适配差异**：

- 通信后端：NCCL → HCCL
- model_runner：GPUModelRunner → NPUModelRunner
- 算子：CUDA kernel → Ascend C kernel
- 部分 scheduler 逻辑特化

---

## 2. PP 相关的主要差异

| 维度 | 上游 vLLM | vllm-ascend |
|------|----------|-------------|
| 通信后端 | NCCL | HCCL |
| ModelRunner | `GPUModelRunner` | `NPUModelRunner` |
| PP+MTP 方案 | broadcast（PR 提议中） | scheduler 同步（PR 提议中） |
| async-scheduling | 已支持 | 适配中 |
| 算子 | CUDA | Ascend C |

**重要说明**：PP+MTP 在两边都是 PR 阶段（[vllm#39704](https://github.com/vllm-project/vllm/pull/39704)、[vllm-ascend#10051](https://github.com/vllm-project/vllm-ascend/pull/10051)），**均未合并**。

---

## 3. NPUModelRunner

### 3.1 类定义（已验证）

源码：`vllm_ascend/worker/model_runner_v1.py:255`

```python
class NPUModelRunner(GPUModelRunner):
    """Ascend 特化的 ModelRunner"""
    
    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        super().__init__(vllm_config, device)
        # Ascend 特化的初始化
        ...
```

继承 `GPUModelRunner`（位于 `vllm/v1/worker/gpu_model_runner.py:418`），重写差异部分。

### 3.2 PP+MTP 相关方法（PR 提议，未合并）

PR #10051 **计划添加**以下方法到 NPUModelRunner（**注意：当前主分支不存在**）：

- `_compute_non_last_pp_mtp_accepted_counts`：计算 PP+MTP 的 accept 数
- `_prepare_non_last_pp_mtp_state_update`：准备非 last rank 的状态更新
- `_collect_pp_mtp_readded_token`：收集 PP+MTP 重新加入的 token
- `_pp_mtp_update_req_spec_token_ids`：更新 spec_token_ids 的上下文管理器

**验证状态**：grep 这些方法名在 `vllm_ascend/worker/model_runner_v1.py` 中**找不到任何匹配**，说明 PR 尚未合并。

### 3.3 跟上游的差异点

- 上游用 broadcast 同步（vllm#39704 提议）
- ascend 用 scheduler 同步（#10051 提议）
- ascend 额外实现了辅助方法处理 FlashComm1（sequence parallelism）

---

## 4. 通信后端：HCCL

### 4.1 HCCL 简介

HCCL（Huawei Collective Communication Library）是华为的集合通信库，对标 NCCL。

| 维度 | NCCL | HCCL |
|------|------|------|
| 平台 | NVIDIA GPU | 华为 Ascend NPU |
| 互联 | NVLink / InfiniBand | HCCS / RoCE |
| API | 类 MPI | 兼容 NCCL 风格 |
| 性能 | GPU 优化 | NPU 优化 |

### 4.2 在 vLLM 中的使用

vLLM 通过 `DeviceCommunicator` 抽象层屏蔽差异：

```python
# GroupCoordinator.__init__ 中
device_comm_cls = resolve_obj_by_qualname(
    current_platform.get_device_communicator_cls()
)
# GPU 平台：返回 NcclDeviceCommunicator
# NPU 平台：返回 HcclDeviceCommunicator
```

底层 `torch.distributed.send / recv / broadcast` 在 NPU 平台自动用 HCCL 后端。

### 4.3 HCCL 特化点

```python
# vllm_ascend 中可能有 HCCL 特化
# 例如：特定通信模式的优化、特定数据类型的支持
```

---

## 5. vllm-ascend#10051 详解（PR 内容，未合并）

### 5.1 PR 目标

将上游 vLLM 的 PP+MTP 支持移植到 vllm-ascend。

### 5.2 核心改动文件（PR 提议）

| 文件 | 改动 |
|------|------|
| `vllm_ascend/core/scheduler_profiling_chunk.py` | scheduler 级 PP+MTP 对齐 |
| `vllm_ascend/worker/model_runner_v1.py` | worker-state 更新 + token 广播 |
| 平台 patch | ModelRunnerOutput、EngineCore、ModelConfig |
| Qwen3.5 MTP | forward patch |

### 5.3 scheduler_profiling_chunk.py 的改动（PR 提议）

PR 计划添加 PP+MTP 对齐逻辑，处理 spec_token_ids，跨 PP rank 对齐 batch size。

### 5.4 model_runner_v1.py 的改动（PR 提议）

PR 计划在 NPUModelRunner 中添加 worker state 更新和 token 广播方法。

### 5.5 Qwen3.5 MTP 的特殊处理（PR 提议）

PR 计划 patch Qwen3.5 MTP 的 forward，确保 drafter 在 last PP rank 执行。

---

## 6. vllm-ascend PP 的特化挑战

### 6.1 跟上游对齐的压力

vllm-ascend 必须紧跟上游：

```
上游 vllm#39704 用 broadcast 方案
    ↓
vllm-ascend#10051 用 scheduler 方案（紧急）
    ↓
未来必须迁移到 broadcast（reviewer 建议）
```

迁移过程中需要：
- 重写 PP+MTP 同步逻辑
- 适配 NPUModelRunner
- 测试覆盖

### 6.2 算子级差异

某些 PP 相关的算子在 Ascend 上有特殊实现：

- HCCL broadcast 跟 NCCL broadcast 行为差异
- 异步通信 API 的差异
- dtype 支持的差异（如 FP8）

### 6.3 FlashComm1（sequence parallelism）

Ascend 特有的 sequence parallelism 优化，在 DeepseekV4Model.forward 中已存在（`vllm_ascend/models/deepseek_v4.py:1145-1157`）：

```python
forward_ctx = get_forward_context()
if forward_ctx is not None and forward_ctx.flash_comm_v1_enabled:
    # token 在 TP rank 间分散
    # 需要 all_gather 收集
    h_states_flat = tensor_model_parallel_all_gather(hidden_states.flatten(1), dim=0)
```

这给 PP+MTP 带来额外复杂度（PP 边界的 token 必须先 all_gather 才能 send）。

---

## 7. vllm-ascend 跟上游的对比总结

| 功能 | 上游 vLLM | vllm-ascend |
|------|----------|-------------|
| 基础 PP | ✓（已合并） | ✓（已合并） |
| PP + TP | ✓ | ✓ |
| PP + EP | ✓ | ✓ |
| PP + MTP | broadcast（PR 提议，未合并） | scheduler（PR 提议，未合并） |
| Async scheduling | ✓ | 适配中 |
| FlashComm1 | ✗ | ✓（Ascend 特化） |

---

## 8. 调试 vllm-ascend PP 的技巧

### 8.1 启用 PP 日志

```bash
export VLLM_LOG_LEVEL=DEBUG
# 或在代码中
import logging
logging.getLogger("vllm.distributed").setLevel(logging.DEBUG)
```

### 8.2 检查 rank 分布

```python
from vllm.distributed.parallel_state import get_pp_group
pp = get_pp_group()
print(f"PP rank_in_group: {pp.rank_in_group}, world_size: {pp.world_size}")
print(f"Is first: {pp.is_first_rank}, Is last: {pp.is_last_rank}")
print(f"Ranks: {pp.ranks}")
```

### 8.3 验证 IntermediateTensors

```python
# 在 forward 入口/出口打印 tensor 信息
def forward(self, ...):
    if not get_pp_group().is_first_rank:
        print(f"Received: {intermediate_tensors['hidden_states'].shape}")
    ...
    if not get_pp_group().is_last_rank:
        print(f"Sending: {hidden_states.shape}")
    ...
```

### 8.4 常见错误

| 错误 | 原因 | 解决 |
|------|------|------|
| shape mismatch | 上游下游张量 shape 不一致 | 检查 IntermediateTensors 字段 |
| dtype mismatch | dtype 不一致 | 统一为 bf16 |
| 死锁 | send/recv 不配对 | 检查 first/last rank 分支 |
| HCCL 错误 | 通信后端问题 | 检查 HCCS 互联、HCCL 版本 |
| API 不存在 | 用了 `send_forward` 等不存在方法 | 改用 `send_tensor_dict` |

---

## 9. 学习建议

### 9.1 优先看上游 vLLM

vllm-ascend 是适配，逻辑主体在上游。先看懂上游，再看 ascend 差异。

### 9.2 关注 PR review 讨论

[vllm-ascend#10051](https://github.com/vllm-project/vllm-ascend/pull/10051) 的 review 讨论比代码更有价值：

- reviewer 指出的设计问题
- 作者解释的实现选择
- 跟上游方案的差异讨论

### 9.3 实际跑一次

最快的理解方式：

```bash
# 用 PP=2 跑一次推理
vllm serve deepseek-ai/DeepSeek-V4 \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 2
```

观察实际行为，配合代码理解。

---

## 10. 小结

- vllm-ascend 是 vLLM 在 Ascend NPU 的插件，不重写只适配
- 主要差异：HCCL（vs NCCL）、NPUModelRunner、FlashComm1
- **重要**：PP+MTP 在两边都是 PR 阶段，未合并；文中描述的方法（如 `_compute_non_last_pp_mtp_accepted_counts`）**当前不存在**
- vllm-ascend#10051 用 scheduler 同步方案（紧急），未来迁移到 broadcast（跟上游对齐）
- 学习建议：先看上游，再看 ascend 差异，最后实际跑一次

---

## 全部完成

恭喜！至此已完成 5 层、12 个知识点的 PP 学习。建议回到 [00_知识地图.md](../00_知识地图.md) 复盘，并打开 vLLM 源码对照加深理解。
