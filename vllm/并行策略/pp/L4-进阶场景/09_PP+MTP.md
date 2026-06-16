# 知识点 09：PP + MTP（推测解码）

> 核心问题：PP 跟推测解码（MTP）配合时，最大的难题是什么？怎么解决？
> 相关 PR：[vllm#39704](https://github.com/vllm-project/vllm/pull/39704)、[vllm-ascend#10051](https://github.com/vllm-project/vllm-ascend/pull/10051)
>
> **重要说明**：这两个 PR 在编写本文时（2026-06）都处于 **Open、未合并**状态。本文描述的内容**尚未进入主分支**，是 PR 提议的方案。

---

## 1. MTP 简介

**MTP（Multi-Token Prediction）**：一次预测多个 token（draft），再用主模型验证。

```
传统自回归：
  Token_1 → Token_2 → Token_3 → ... （一次一个）

MTP：
  Token_1, Token_2, ..., Token_k（draft，k 个）
  ↓
  主模型验证 → accept 部分 / reject 部分
  ↓
  接受的 token 加入序列
```

MTP 加速 1.5x-3x，但实现复杂。

---

## 2. PP + MTP 的核心难题

### 2.1 MTP 只在 last rank 工作

```
PP=4 的流水线：
  Stage 0 → Stage 1 → Stage 2 → Stage 3
                                  ↓
                            sampler 采样
                                  ↓
                            MTP draft model 生成 draft
                                  ↓
                            验证 draft → accepted/rejected
```

MTP 的 draft + 验证都在 **last rank** 完成。

### 2.2 但所有 rank 都需要知道结果

```
当 Stage 3 知道 draft token 被 accept：
- Stage 3 知道哪些 token 是有效的
- Stage 0/1/2 仍持有这些 token 的 KV cache 状态
- 它们必须更新自己的状态：
  - 把 accept 的 token 标记为"已处理"
  - 把 reject 的 draft 从 KV cache 滚回
```

**核心矛盾**：状态在 last rank 决定，但需要在所有 rank 同步。

### 2.3 不解决的后果

- KV cache 不一致 → 推理结果错误
- batch state 错乱 → 后续 token 错误
- 严重时 NaN 或崩溃

---

## 3. 两种解决方案

### 3.1 方案 A：scheduler 进程同步

**vllm-ascend#10051 的初版方案**（PR 中提出）。

```
Stage 3 完成 draft + 验证
    ↓
通过 scheduler 进程广播结果
    ↓
scheduler 调度下一步时，告知所有 rank：
  "这些 token 是 accept 的，更新你们的 batch state"
    ↓
所有 rank 接收并更新
```

**优点**：
- 实现相对简单
- 利用现有 scheduler 通道

**缺点**：
- 跟 async-scheduling 不兼容（scheduler 必须等所有 rank 同步）
- 性能损失

### 3.2 方案 B：dist.broadcast

**vllm#39704 上游方案**（PR 中提出）。

```python
# last rank 完成采样后，直接广播给所有 PP rank
recv = torch.empty((num_reqs, width), dtype=torch.int32, device=device)
dist.broadcast(
    sampled_token_ids, 
    src=pp.ranks[-1],  # 或 pp.last_rank（GroupCoordinator.last_rank 属性）
    group=pp.device_group
)
# 所有 rank 都拿到了采样结果
# 各自更新自己的 batch state
```

**优点**：
- 直接、低延迟
- 跟 async-scheduling 兼容（rank 间点对点通信，不走 scheduler）

**缺点**：
- 每个 rank 都要实现状态更新方法
- 代码改动更多

### 3.3 两种方案对比

| 维度 | scheduler 同步（ascend 初版） | broadcast（vllm 上游） |
|------|-------------------------------|----------------------|
| 通信路径 | scheduler → workers | last rank → all ranks |
| 延迟 | 高（多跳） | 低（一跳） |
| async 兼容 | 不兼容 | 兼容 |
| 实现复杂度 | 中 | 高 |
| 长期可维护 | 差 | 好 |

---

## 4. 上游 vllm#39704 提议的核心改动

> 注意：以下内容来自 PR 描述和 review 评论，**PR 未合并**，可能变动。

### 4.1 scheduler 改动

`scheduler` 需要处理 spec_token_ids：

```python
# scheduler 调度时，把 spec_token_ids 也作为输入
model_runner_output.spec_token_ids = ...

# 处理 PP 模式下 batch size 的对齐
# 每个 stage 处理的 token 数必须一致
```

### 4.2 EngineCore 改动

```python
# PP batch_queue 模式下跳过全局 post-step 更新
# 因为更新会在每个 rank 各自做
```

### 4.3 GPUModelRunner 改动（PR 提议）

PR 计划在 GPUModelRunner 中添加：

- `_compute_non_last_pp_mtp_accepted_counts`：计算每个请求 accept 的 token 数
- `_prepare_non_last_pp_mtp_state_update`：准备非 last rank 的状态更新

**注意**：截至本文编写时，这些方法**尚未合入主分支**。

### 4.4 模型层面

DeepSeek MTP 模型需要实现 `SupportsPP` Protocol（已存在于 `vllm/model_executor/models/interfaces.py:616`）：

```python
class SupportsPP(Protocol):
    """Marker for models that support pipeline parallel"""
    ...
```

---

## 5. vllm-ascend#10051 的特化（PR 提议）

> 注意：以下内容来自 PR 描述和 review 评论，**PR 未合并**，可能变动。

### 5.1 核心差异

跟上游相比，vllm-ascend 的初版用 scheduler 同步方案。

### 5.2 涉及的核心文件（PR 改动）

| 文件 | 改动内容 |
|------|---------|
| `vllm_ascend/core/scheduler_profiling_chunk.py` | scheduler 级 PP+MTP 对齐 |
| `vllm_ascend/worker/model_runner_v1.py` | worker-state 更新 + token 广播 |
| 平台 patch | ModelRunnerOutput、EngineCore、ModelConfig |
| Qwen3.5 MTP | forward patch |

### 5.3 计划添加的方法

PR 计划在 `NPUModelRunner`（位于 `vllm_ascend/worker/model_runner_v1.py:255`）中添加：

- `_compute_non_last_pp_mtp_accepted_counts`
- `_prepare_non_last_pp_mtp_state_update`
- `_collect_pp_mtp_readded_token`
- `_pp_mtp_update_req_spec_token_ids`

**重要**：截至本文编写时，这些方法**尚未存在于 vllm-ascend 主分支**。它们是 PR #10051 提议的代码。

### 5.4 Qwen3.5 MTP 的特殊处理

PR 提议 patch Qwen3.5 MTP 的 forward，确保 drafter model 在 last PP rank 执行。

---

## 6. review 中暴露的关键 bug

Gemini Code Assist 和人工 review 指出多个问题（来自 PR 评论区）：

### 6.1 关于 `pp.last_rank`（需要澄清）

PR 初版代码使用了 `pp.last_rank`。Gemini Code Assist 评论说：

> "The attribute `pp.last_rank` is used here, but `GroupCoordinator` does not have a `last_rank` attribute in vLLM."

**实际情况（2026-06 验证）**：当前 vLLM 主分支的 `GroupCoordinator` **确实有** `last_rank` property（`vllm/distributed/parallel_state.py:535`）：

```python
@property
def last_rank(self):
    """Return the global rank of the last process in the group"""
    return self.ranks[-1]
```

可能的情况：
- PR 初版基于较老的 vLLM 版本，那时 `last_rank` 还未添加
- 或 Gemini Code Assist 分析的代码版本有偏差

**结论**：现在 `pp.last_rank` 在主分支可用，无需改成 `pp.ranks[-1]`。

### 6.2 High：get_pp_group 未 import

```python
# llm_base_proposer.py
# ❌ 直接使用
assert get_pp_group().is_last_rank, ...

# ✓ 必须先 import
from vllm.distributed.parallel_state import get_pp_group
```

### 6.3 High：字典查找不安全

```python
# ❌ 直接字典查找，可能 KeyError
req_index = model_runner_output.req_id_to_index[req_id]

# ✓ 用 .get()
req_index = model_runner_output.req_id_to_index.get(req_id)
if req_index is None:
    continue
```

### 6.4 Medium：dtype 不匹配

review 评论指出：
> "The sender broadcasts `sampled_token_ids` with dtype `int32` (from sampler), but the receiver allocates an `int64` buffer."

```python
# 上游 sampler 输出 int32
# 但 receiver 分配 int64 buffer
# → NCCL/HCCL 通信会出错
recv = torch.empty(..., dtype=torch.int32)  # 必须 int32
```

---

## 7. PP+MTP 完整数据流（PR 提议的目标）

```
Step N:
    ┌─────── Stage 0 ───────┐  ┌─────── Stage 1 ───────┐  ...  ┌─────── Stage 3 ───────┐
    │ 处理 token             │→│ 处理 token             │→ ... →│ 处理 token             │
    │ （包括 draft）          │  │ （包括 draft）          │       │ （包括 draft）          │
    │                        │  │                        │       │                        │
    │                        │  │                        │       │ sampler → 主采样结果    │
    │                        │  │                        │       │ MTP drafter → draft    │
    │                        │  │                        │       │ 验证 draft             │
    │                        │  │                        │       │ ↓                      │
    │                        │  │                        │       │ accepted_token_ids     │
    │                        │  │                        │       │ ↓                      │
    │                        │  │                        │       │ broadcast(src=last_rank)│
    │                        │  │                        │       │ ↓ ↓ ↓                  │
    │ 更新 batch state       │←│ 更新 batch state       │← ... ←│ 更新 batch state       │
    │ 更新 KV cache          │  │ 更新 KV cache          │       │ 更新 KV cache          │
    └────────────────────────┘  └────────────────────────┘       └────────────────────────┘
    
    Step N+1 开始（如果有 accept 的 token，会一次性处理多个）
```

---

## 8. 为什么 broadcast 是未来方向？

reviewer `lidenghui1110` 在 PR 评论区说：

> "Compared to use scheduler process to sync draft_token between PP stages in this PR, I would rather to use broadcast following the strategy in async-scheduling in this PR: #39074. For the future maintaining of PP with async-scheduling, I recommend the second one which implemented in vllm."

### 8.1 跟 async-scheduling 的协同

async-scheduling 下，每个 rank 像独立工人，stage 间不互相等待。如果用 scheduler 同步：

```
scheduler 必须等所有 rank 报告状态 → rank 间异步被破坏
```

如果用 broadcast：

```
last rank 完成 → 直接 broadcast → 其他 rank 异步接收 → rank 间异步保持
```

### 8.2 性能优势

scheduler 同步要走多跳（last rank → scheduler → 其他 rank），延迟高。
broadcast 一跳直达，延迟低。

### 8.3 长期维护

vLLM 上游在用 broadcast 方案，vllm-ascend 跟上游对齐后，后续 sync 代码更容易。

vllm-ascend#10051 作者 `gjc0824` 在 PR 评论区回应：

> "The current implementation is the adaptation of your solution to the vllm-ascend solution, which was only for urgent needs. I think the core issue of PP+MTP is to update the metadata of non-last PP rank. The better method is following the broadcast of PP+async_scheduling. However, I may not continue with the follow-up work."

---

## 9. 当前状态（截至 2026-06）

| PR | 状态 | 备注 |
|----|------|------|
| [vllm#39704](https://github.com/vllm-project/vllm/pull/39704) | Open，有 merge conflict | 上游 broadcast 方案 |
| [vllm-ascend#10051](https://github.com/vllm-project/vllm-ascend/pull/10051) | Open，有 merge conflict | Ascend 适配（scheduler 方案） |

**两个 PR 都未合并**。vllm-ascend 作者已表态：当前是紧急方案，未来应该迁移到 broadcast 方案。

---

## 10. 小结

- PP+MTP 核心难题：last rank 决定状态，所有 rank 需要同步
- 两种方案：scheduler 同步（紧急）vs broadcast（长期）
- 上游用 broadcast，Ascend 暂用 scheduler，未来对齐
- **重要**：两个 PR 都未合并，文中描述的代码大部分尚未进入主分支
- 关于 `pp.last_rank`：现在主分支已存在该属性，PR review 中的担忧已不成立
- MTP 模型需要实现 `SupportsPP` Protocol（已存在于 interfaces.py:616）

---

**下一步**：[10_PP+async_scheduling.md](10_PP+async_scheduling.md) — async-scheduling 怎么消除 PP 气泡？
