# `sample_tokens` 流程与分析

> 对应文件：[vllm-ascend/vllm_ascend/worker/model_runner_v1.py:2397-2606](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2397-L2606)
> 关联方法：[propose_draft_token_ids:1686-1915](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1686-L1915)、[_copy_draft_token_ids_to_cpu:1917-1945](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1917-L1945)

## 0. 背景

PP（流水线并行）模式下，worker 被拆成两次调用：

- `execute_model()`：算前向 → 把中间状态打包成 `execute_model_state`
- `sample_tokens()`：拿这个 state 做 sample + drafter + 收尾

`sample_tokens(grammar_output)` 进入后，按以下顺序执行。

### 0.1 不是 PP 专属，是统一收尾入口

更准确地说：`sample_tokens` **不是 PP 专属**，而是 vllm-ascend 在所有 LLM 解码场景（单卡 / TP / PP / async）下的统一收尾入口。依据 [worker_base.py:145](vllm/vllm/vllm/v1/worker/worker_base.py#L145) 的契约——"`execute_model` 返回 None → 必须立即调用 `sample_tokens`"——只要 `execute_model` 走主路径（[L2394](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2394) `return None`），`sample_tokens` 就一定会被调用。区别只在于：

| 场景 | execute_model 返回 | execute_model_state | sample_tokens 行为 |
|---|---|---|---|
| 单卡/TP LLM | None | 已设置 | 走主路径：采样+draft+输出 |
| PP last rank | None | 已设置 | 走主路径 |
| PP 非 final rank | hidden_states | None | 走早退分支 |
| Pooling | output | — | 不被调用 |

PP 模式特殊之处只在于：它多了一条早退分支用来处理"非 final rank 没有 logits 可采"的情况，并额外承担 async 调度下的 sampled token 广播/接收职责。函数本身的"采样+draft+构造输出"主流程在所有 LLM 解码场景下都跑。

### 0.2 在推理流程中的位置（lm_head 之后）

`lm_head`（`compute_logits`）在 `execute_model` 内部执行（[L2394](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2394)），产出的 logits 打包进 `ExecuteModelState`，`execute_model` 返回 None，框架再调 `sample_tokens`。所以 `sample_tokens` **在 lm_head 之后**。

```
┌─────────────────────────────────────────────────────────────────────┐
│  scheduler.schedule_step()                                          │
│  （调度：决定本步跑哪些 req、多少 token、KV cache 如何分配）        │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ SchedulerOutput
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  worker.execute_model(scheduler_output)         ← 前向计算          │
│                                                                     │
│  ┌─────────────────────────────────────────────┐                    │
│  │  embedding                                  │                    │
│  │     ↓                                       │                    │
│  │  transformer layers × N                     │                    │
│  │   ├─ self-attention (读/写 KV cache)        │                    │
│  │   ├─ cross-attention (mm 模型)              │                    │
│  │   └─ MLP / MoE                              │                    │
│  │     ↓                                       │                    │
│  │  hidden_states  ────────────────────────┐  │                    │
│  │     ↓                                   │  │                    │
│  │  ★ lm_head = compute_logits()  ★        │  │  ← lm_head 在这里   │
│  │     ↓                                   │  │                    │
│  │  logits                                 │  │                    │
│  │     ↓                                   │  │                    │
│  │  (PP 非 last rank: send_tensor_dict)    │  │                    │
│  │  (PP last rank:     打包 ExecuteModelState)│                    │
│  └─────────────────────────────────────────────┘                    │
│                                                                     │
│  return None  ← 注意：返回 None 触发框架调用 sample_tokens          │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ ExecuteModelState (含 logits/hidden_states/...)
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  ★ worker.sample_tokens(grammar_output)  ★  ← 本函数：收尾          │
│                                                                     │
│  ┌─── 主路径（LLM 解码场景，execute_model_state 非空）──────────┐   │
│  │  [1] 解包 ExecuteModelState                                  │   │
│  │  [2] grammar bitmask + _sample(logits, spec_decode_metadata) │   │
│  │      ├─ 普通采样 (spec_decode_metadata None)                 │   │
│  │      └─ rejection 采样 (非空，验证上一步 draft)              │   │
│  │  [3] sampling_done_event.record()  (async 时)                │   │
│  │  [4] propose_draft_token_ids (spec_decode 开启时)            │   │
│  │      └─ 5 种 drafter 分支（EAGLE/MTP/dflash 等）             │   │
│  │  [5] _bookkeeping_sync → valid_sampled_token_ids 等          │   │
│  │  [6] draft_token 段（finalize_kv_connector）                 │   │
│  │  [7] 构造 ModelRunnerOutput                                  │   │
│  │  [8] async 状态更新 _update_states_after_model_execute       │   │
│  │  [9] PP 广播 _pp_broadcast_prev_sampled_token_ids            │   │
│  │  [10] 同步/异步双路径返回                                    │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─── 早退路径（PP 非 final rank，execute_model_state 为空）──┐    │
│  │  receive sampled token (async+PP) → 透传 kv_connector → 返回│    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  return ModelRunnerOutput / AsyncGPUModelRunnerOutput               │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ ModelRunnerOutput
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  scheduler 接收 ModelRunnerOutput                                    │
│  ├─ 更新 req 状态（accepted token 数、finished 标记等）             │
│  ├─ 触发下一步调度                                                  │
│  └─ 把 sampled token 通过 IPC 送回 worker（同步模式）               │
│     或 worker 自己已写入 input_batch（async 模式，绕开 IPC）        │
└─────────────────────────────────────────────────────────────────────┘
```

### 0.3 在大模型推理"逻辑阶段"中的位置

| 阶段 | 是否调用 sample_tokens | 说明 |
|---|---|---|
| **Prefill**（首步填 prompt） | ✅ 调用 | execute_model 算完整个 prompt 的前向 + lm_head → sample_tokens 出第一个 token（+ draft） |
| **Decode**（逐 token 生成） | ✅ 调用 | 每一步都是 execute_model → sample_tokens，循环到 EOS |
| **PP 中间 rank 的 forward** | ✅ 调用但走早退 | execute_model 返回 hidden_states（send 给下游），sample_tokens 进早退分支 |
| **Pooling 模型** | ❌ 不调用 | execute_model 直接 return output（pooling 结果），不走 sample_tokens |

### 0.4 一句话定位

`sample_tokens` 处于 **"前向 + lm_head 已完成"** 与 **"下一步调度开始"** 之间，负责把 logits 收敛成最终输出 token 并维护投机解码状态。它**不在 transformer 层内**，也**不动 KV cache**（除了 PD 分离场景的 kv_connector save/put）；它是 worker 侧的"采样 + 起草 + 状态收尾"出口。

具体到代码层面：
- **上界**：[execute_model L2394](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2394) 的 `compute_logits`（lm_head）执行完毕，logits 打包进 `ExecuteModelState`
- **下界**：[sample_tokens L2606](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2606) `return async_output` 或 [L2569](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2569) `return model_runner_output`，把输出交回 scheduler

`lm_head` 是 `execute_model` 的最后一拍，`sample_tokens` 是 `execute_model` 之后的下一拍——两者由框架在同一调度步内连续驱动，靠 `ExecuteModelState` 这个临时载体传递 12 个字段。

---

## 1. 整体流程

```
sample_tokens(grammar_output) 进入
│
├─ [0] PP 非 final rank 的早退路径  (L2405-2421)
│   │ 条件：execute_model_state is None（本 rank 不负责 sample）
│   │ 副作用：
│   │   - async + PP 时调 _pp_receive_prev_sampled_token_ids_to_input_batch()
│   │     让下游（PCP 输入准备）能拿到上一步 sampled token
│   │   - 透传 kv_connector_output（PP + KV transfer 场景）
│   │ 返回：None / EMPTY_MODEL_RUNNER_OUTPUT
│
├─ [1] 解包 execute_model_state  (L2423-2439)
│   │ 拿到 12 个字段，并立即 self.execute_model_state = None 清空
│   │ 关键字段用途：
│   │   - logits                    → [2] 喂 sampler
│   │   - spec_decode_metadata      → [2] 决定普通/rejection 采样
│   │   - hidden_states / aux / sample_hidden_states → [4] 喂 drafter
│   │   - spec_decode_common_attn_metadata           → [4] drafter 自己的 attn meta
│   │   - positions, batch_desc, cudagraph_stats, ec_connector_output → 收尾用
│
├─ [2] 结构化输出掩码 + 采样  (L2442-2451)
│   │ grammar_output 非空时：
│   │   logits → cpu.float() → apply_grammar_bitmask → 回 device 原精度
│   │   （与 GPU 版不同：Ascend 不支持 torch.compile 优化此步）
│   │ 然后 _sample(logits, spec_decode_metadata) → sampler_output
│   │   - spec_decode_metadata None  → 普通采样
│   │   - 非空                         → rejection 采样，验证上一步 draft 接受情况
│
├─ [3] sampling_done_event  (L2453-2458)   ← need_accepted_tokens 时
│   │ 记录 NPU event，供 [8] 异步状态更新时 wait
│
├─ [4] propose_draft_token_ids 闭包  (L2462-2477)
│   │ 捕获 hidden_states / spec_decode_metadata / positions 等上下文，
│   │ 让 padded-batch 用 GPU sampled、ngram 用 CPU sampled 两种时机复用
│   │
│   └─ self.propose_draft_token_ids(...)   (L1686-1915)
│        │
│        ├─ drafter 为 None → 返回 None（外层 self._draft_token_ids = None，
│        │   避免残留上一步 draft；机制是"返回 None 由调用方赋值"，不是函数内第一行置空）
│        │
│        ├─ 5 种 drafter 分支：
│        │   ① AscendNgramProposer / SuffixDecoding → CPU n-gram 匹配
│        │   ② AscendNgramProposerNPU → npu_ngram_spec_decode 算子，异步 D2H num_valid_draft_tokens
│        │   ③ AscendMedusaProposer → 多头预测
│        │   ④ AscendExtractHiddenStatesProposer → 抽 aux_hidden_states
│        │   ⑤ EAGLE / draft_model → 走下面子流程
│        │
│        └─ [EAGLE 子流程] (L1783-1911)
│             ├─ prepare_next_token_ids_cpu / _padded → next_token_ids
│             │   padded 模式：组 [num_reqs, num_spec+1] GPU tensor，rejected=-1
│             │   disable_padded_drafter_batch：CPU list
│             ├─ get_mtp_target_hidden_states() → 重绑 hidden_states
│             │   （DeepSeek V4 MTP 需要 pre-hc_head residual；采样已用过，重绑安全）
│             ├─ spec_decode_metadata None → 全量切片
│             │   否则 → prepare_inputs_padded → token_indices (+ num_rejected_tokens_gpu)
│             ├─ 组装三元组：target_token_ids / target_positions / target_hidden_states
│             └─ drafter._propose(...) → draft_token_ids (GPU tensor)
│   │
│   └─ _copy_draft_token_ids_to_cpu(scheduler_output)  (L1917-1945)
│        ├─ 记录 self._draft_token_req_ids = input_batch.req_ids.copy()
│        │   （供站点 2 按 req_ids 对齐 draft token 到正确 request）
│        └─ 异步 D2H：draft_token_ids_copy_stream 上 wait_stream(default)
│            → copy_(non_blocking=True) → event.record()
│            （站点 2 是否用 GPU .cpu().tolist() 还是 CPU 副本，需在 ModelRunnerOutput
│              构造处单独核验，本函数范围内无法定论）
│
├─ [5] _bookkeeping_sync  (L2479-2493)
│   │ 产出 valid_sampled_token_ids、logprobs_lists、prompt_logprobs_dict、
│   │ req_ids_output_copy、req_id_to_index_output_copy、invalid_req_indices
│   │ → 站点 2 构造 ModelRunnerOutput 的主要数据源
│
├─ [6] draft_token 段  (L2495-2520)
│   │ use_padded_batch → propose_draft_token_ids(sampler_output.sampled_token_ids)  // GPU
│   │ 否则             → propose_draft_token_ids(valid_sampled_token_ids)           // CPU
│   │ spec_decode 开启时 finalize_kv_connector（v0.18 推迟到 draft 跑完再 save/put）
│
├─ [7] 构造 ModelRunnerOutput  (L2522-2533)
│   │ + profiling timing / dynamic_eplb.forward_end / _finalize_dump_data  (L2534-2541)
│
├─ [8] 异步状态更新  (L2543-2550)   ← need_accepted_tokens 时
│   │ 在 global_stream 上 wait_event(sampling_done_event)
│   │ → _update_states_after_model_execute(sampled_token_ids, scheduler_output)
│
├─ [9] PP 广播  (L2555-2557)   ← async + pp.is_last_rank
│   │ _pp_broadcast_prev_sampled_token_ids，绕开 scheduler/engine IPC
│
└─ [10] 双路径返回  (L2559-2606)
    ├─ 同步路径：routed_experts_initialized 时把 routed_experts_cpu 包成
    │   RoutedExpertsLists 挂到 output；返回 ModelRunnerOutput
    └─ 异步路径：构造 AsyncGPUModelRunnerOutput
        - routed_experts 用 buf[:total].clone() + slot_mapping[:total].clone()
          （capturer buffer 下一步会 clear_buffer()、slot_mapping 下一步会被
           _prepare_inputs 覆写，不 clone 会让 copy stream 读到撕裂数据）
        - set_async_sampled_token_ids 把 sampled_token_ids_cpu + ready_event 注入 input_batch
        - 返回 async_output
```

---

## 2. 早退路径 [0] 详解（L2405-2421）

### 2.1 触发条件

`execute_model_state is None`，说明本次 `execute_model` 没有产生需要 sample 的中间状态。典型场景：

- **PP 非 final rank**：见 [L2356-2358](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2356-L2358)，非 last rank 只 `send_tensor_dict` 把 hidden_states 送走，不构造 `ExecuteModelState`（赋值点在 [L2374](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2374)）。
- **Pooling 模型 + 非 final rank**：见 [L2341-2348](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2341-L2348) 的早退 return，根本走不到 L2374。

### 2.2 三个分支

| 分支 | 条件 | 作用 |
|---|---|---|
| ① 接收 sampled token | `use_async_scheduling and pp.world_size > 1 and not skip_pp_pd_broadcast` | 调 `_pp_receive_prev_sampled_token_ids_to_input_batch()`，从 last PP rank 拉回上一步采样结果 |
| ② 纯空返回 | `not kv_connector_output` | 没 KV 数据要传，直接 `return None` |
| ③ 透传 KV | `kv_connector_output` 非空 | `is_empty()` 时返回 `EMPTY_MODEL_RUNNER_OUTPUT`；否则拷一份并挂上 `kv_connector_output`，让 KV transfer 链路继续走 |

### 2.3 原理

#### (1) 为什么非 final rank 也会进 `sample_tokens`

PP 模式下所有 rank 都按相同 API 节奏走（`execute_model` → `sample_tokens`），由框架统一驱动。但只有 last rank 真正算了 logits 并在 [L2374](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2374) 构造了 `ExecuteModelState`；其它 rank 的 forward 只是把 hidden_states 往下游 send，没有需要采样的 logits，所以 `execute_model_state` 保持 None。这段就是给这些"空跑"的 rank 一个干净的收尾点，避免它们走到下面的 `_sample` / drafter 逻辑里去。

#### (2) 为什么要 `skip_pp_pd_broadcast`

```python
skip_pp_pd_broadcast = self.is_kv_producer and pp.world_size > 1
```

`is_kv_producer` 来自 [L478-482](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L478-L482)，标识本 rank 是 KV producer（PD 分离场景下 producer 端）。这类 rank 自己存 KV、不参与 last rank 的 sampled token 广播链路，所以**主动跳过** receive，避免把别的 PP 组的 sampled token 错误地塞进自己的 input_batch。

#### (3) 为什么 async + PP 要在早退路径里 receive

这是这段最关键的设计。在**异步调度**下，scheduler 不再走"last rank 算完 → 通过 engine IPC → 广播给其它 rank"的同步路径，而是各 rank 直接用上一步的 sampled token 准备下一步输入（典型是 PCP —— chunked prefill 的输入准备）。

- 非 final rank 在 async 模式下不会通过 scheduler 拿到 sampled token，所以必须由 last rank 广播过来（[L2555-2557](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2555-L2557) 的 `_pp_broadcast_prev_sampled_token_ids` 是发送端）。
- 这里 [L2411](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2411) 是**接收端**，把 token 直接写进 `input_batch`，让下一步 `_prepare_inputs` / PCP 能直接读到，**绕开 scheduler/engine IPC**，这是 async 调度能并起来跑的关键。

#### (4) 为什么 KV connector_output 还要透传

`kv_connector_output`（如 AscendConnector、PagedKVConnector 等）可能在本 rank 上有未消费完的 KV transfer 状态（save/put 的中间结果）。即便本 rank 不采样，KV pool 的 save/put 流程仍要继续，所以即使走早退路径也得把 `kv_connector_output` 原样带回给上层调用者，避免 KV 状态卡死。注释 `# In case of PP with kv transfer` 说的就是这个。

#### (5) 一句话串联

> "我是 PP 中间 rank，没 logits 可采、没 draft 可生成；但 async 模式下我得从 last rank 拿回上一步的 sampled token 喂给下一步的 PCP 输入准备，顺便把 KV connector 的中间状态透传出去，然后干净退出。"

---

## 3. 关键修正与设计意图

### 3.1 drafter 为 None 时的置空机制

- 不是"函数第一行置空 `self._draft_token_ids`"。
- 实际机制：[L1700-1702](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1700-L1702) 内 `if not self.drafter: draft_token_ids = None`（**局部变量**），函数返回 None；外层闭包 [L2464](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2464) 把 None 赋给 `self._draft_token_ids`。
- `self._draft_token_req_ids` 不在 `propose_draft_token_ids` 里赋值，而在 `_copy_draft_token_ids_to_cpu` [L1927](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1927)。
- 代码中**不存在 `input_fits_in_drafter` 早退分支**。

### 3.2 EAGLE 子流程关键点

- `prepare_next_token_ids_padded` vs `_cpu`：padded 模式组 `[num_reqs, num_spec+1]` GPU tensor（rejected=-1），disable_padded_drafter_batch 走 CPU list。
- `get_mtp_target_hidden_states()`：DeepSeek V4 MTP 需要 pre-hc_head 的 residual，采样已用过 hidden_states，重绑安全。
- `prepare_inputs_padded` 产出 `token_indices` + `num_rejected_tokens_gpu`，drafter 只在采样点喂 token。

### 3.3 async/sync 双路径（[10]）

- 同步路径：`routed_experts_cpu` 直接包成 `RoutedExpertsLists`。
- 异步路径：必须 `.clone()` routing_data 和 slot_mapping，原因见 [L2571-2582](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2571-L2582) 注释 —— capturer buffer 下一步 `clear_buffer()`、slot_mapping 下一步被 `_prepare_inputs` 覆写，不 clone 会让 copy stream 读到撕裂数据。

---

## 4. 仍需下游核验的项

1. **站点 2（ModelRunnerOutput 构造后挂 spec_token_ids）的数据源**：究竟用 `self._draft_token_ids` 的 `.cpu().tolist()` 还是 `self._draft_token_ids_cpu`，需在 LLMEngine/scheduler 链路定位。
2. **`_sample` 内部**：spec_decode_metadata 非空时是否真的走 `self.rejection_sampler`，需读 [L2609+](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2609) 确认。
3. **PP 广播/接收实现**：`_pp_receive_prev_sampled_token_ids_to_input_batch` / `_pp_broadcast_prev_sampled_token_ids` 在 vllm-ascend 内未定义，应在 vllm 核心基类，需去核心源码确认。
4. **`execute_model_state is None` 的精确触发路径**：需顺着 `execute_model` 全流程追一遍才能给穷举清单（pooling 早退 / PP 非 last rank / 首次调用未执行 execute_model 等）。
5. **"NPU event.synchronize() 不可靠"** 这一论断在本函数范围内无注释支撑，标注为待核验。

---

## 5. 分支 ⑤ 深度解析：EAGLE / MTP / dflash / draft_model

### 5.1 触发条件与 drafter 实例

[L1783](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1783) 的 `elif self.speculative_config.use_eagle() or self.speculative_config.uses_draft_model():`

- `use_eagle()` 命中的 method：`eagle`、`eagle3`、`mtp`、`dflash`（[speculative.py:1109-1110](vllm/vllm/vllm/config/speculative.py#L1109-L1110)）
- `uses_draft_model()` 命中的 method：`draft_model`

drafter 实例对照 [__init__.py:42-47](vllm-ascend/vllm_ascend/spec_decode/__init__.py#L42-L47)：

| method | drafter 类 |
|---|---|
| eagle / eagle3 / mtp | `AscendEagleProposer` |
| dflash | `AscendDflashProposer`（继承自 `AscendEagleProposer`） |
| draft_model | `AscendDraftModelProposer` |

### 5.2 子流程总览（L1783-1911）

```
① common_attn_metadata = spec_decode_common_attn_metadata
② sampled_token_ids = valid_sampled_token_ids
③ prepare_next_token_ids (padded 或 cpu)
④ PCP 上下文准备 (long_seq_metadata / num_prefill_reqs / num_decode_reqs)
⑤ get_mtp_target_hidden_states() 重绑 hidden_states
⑥ 构造三元组（target_token_ids / target_positions / target_hidden_states）
   └─ spec_decode_metadata None → 全量切片路径
   └─ 非空 → prepare_inputs_padded 路径
⑦ drafter._propose(...) → draft_token_ids (GPU tensor)
```

### 5.3 步骤详解

#### ① 准备 `next_token_ids`（[L1787-1814](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1787-L1814)）

**padded 模式（默认）**：[L1798-1814](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1798-L1814)
- 输入是 GPU tensor，shape `[num_reqs, num_spec+1]`，rejected token 标为 -1
- 调 `prepare_next_token_ids_padded`（[llm_base_proposer.py:1682](vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py#L1682)）
- 输出：`next_token_ids`（每行最后一个 valid token，全 invalid 时用 `backup_next_token_ids` 从 `request.get_token_id` 取）+ `valid_sampled_tokens_count`

**disable_padded_drafter_batch 模式**：[L1787-1797](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1787-L1797)
- 输入是 CPU `list[list[int]]`
- 调 `prepare_next_token_ids_cpu`，返回 `next_token_ids`（无 count）
- 兼容不支持 padded 的 attention backend，或 MTP-fullgraph + PCP 不兼容时强制走这条

判定点：[L2497-2506](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L2497-L2506) 的 `use_padded_batch`。

#### ② PCP 上下文（[L1816-1828](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1816-L1828)）

PCP = Prefill Context Parallel（长序列切分）。`self.use_cp` 为 True 时：
- `long_seq_metadata`：PCP 长序列元数据
- `input_ids_pcp_full` / `query_start_loc_pcp_full`：PCP 切片下的全量 input_ids 和 query 起止
- `num_prefill_reqs` / `num_decode_reqs`：PCP 下区分 prefill 和 decode 阶段的 req 数

不开启 PCP 时这些全置 None / 0。

#### ③ 重绑 hidden_states（[L1834-1838](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1834-L1838)）

```python
mtp_hidden_states = getattr(
    self.get_model(), "get_mtp_target_hidden_states", lambda: None
)()
if mtp_hidden_states is not None:
    hidden_states = mtp_hidden_states
```

**DeepSeek V4 MTP 特化点**：target 模型若实现了 `get_mtp_target_hidden_states`，返回 **pre-hc_head 的 residual**，覆盖默认 hidden_states。注释（[L1830-1833](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1830-L1833)）说明采样已用过 hidden_states，重绑给 drafter 安全。MTP 头训练时吃 pre-hc_head residual（不是 logits-前那一层），所以喂给 drafter 时必须取对位置。

#### ④ 构造三元组（[L1840-1893](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1840-L1893)）

三元组 `target_token_ids / target_positions / target_hidden_states` 是喂给 drafter 的核心输入。

**路径 A：`spec_decode_metadata is None`（无上一步 draft，全量切片）**[L1841-1858](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1841-L1858)
- 场景：首步 / 上一步全接受 / 非 spec 步
- `token_indices_to_sample = None`（后续在 `_propose` 里 [L668-669](vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py#L668-L669) 默认从 `query_start_loc[1:] - 1` 取）
- `target_token_ids` = `input_ids.gpu[:num_scheduled_tokens]`（或 PCP 的 `input_ids_pcp_full[:num_scheduled_tokens]`）
- `target_positions` = `_get_positions(num_scheduled_tokens)`
- `target_hidden_states`：`use_aux_hidden_state_outputs` 时 cat `aux_hidden_states`，否则 `hidden_states[:num_scheduled_tokens]`

**路径 B：`spec_decode_metadata is not None`（有上一步 draft 要 verify）**[L1859-1893](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1859-L1893)
- 上一步 draft 被 rejection sampler 验证过，部分被拒绝
- 调 `prepare_inputs_padded`（[llm_base_proposer.py:1864](vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py#L1864)），输出三件套：
  - `token_indices`：本步实际要喂给 drafter 的 token 索引
  - `token_indices_to_sample`：每个 req 在 padded batch 里要采样的位置（rejected 当 padding 留着，最后在这里挑出真正要采样的点）
  - `num_rejected_tokens_gpu`：每个 req 上一步被拒绝的 token 数
- `target_token_ids` = `input_ids.gpu[token_indices]`
- `target_positions` = `_get_positions(token_indices)`
- `target_hidden_states` = `hidden_states[token_indices]`（或 aux cat 版本）

**关键差异**：路径 A 喂全量 token，路径 B 只喂未被拒绝的 token（通过 indices 切片）。`num_rejected_tokens_gpu` 是路径 B 独有、传给 drafter 让它在新一轮起草时考虑上一步拒绝情况的关键信号。

#### ⑤ `drafter._propose(...)`（[L1894-1911](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1894-L1911) → [llm_base_proposer.py:643](vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py#L643)）

13 个参数对应上面准备的所有数据：

```python
draft_token_ids = self.drafter._propose(
    target_token_ids=...,           # ④ 三元组之一
    target_positions=...,           # ④ 三元组之一
    target_hidden_states=...,       # ④ 三元组之一
    next_token_ids=...,             # ① 输出
    token_indices_to_sample=...,    # ④ 输出
    common_attn_metadata=...,       # 修改后的 attn meta
    target_model_batch_desc=...,    # 从 execute_model_state 透传
    sampling_metadata=...,          # 从外层闭包
    req_scheduled_tokens=...,       # PCP 相关
    long_seq_metadata=...,          # ② PCP
    num_prefill_reqs=...,           # ② PCP
    num_decode_reqs=...,            # ② PCP
    scheduler_output=...,           # 调度上下文
    num_scheduled_tokens=...,       # 调度上下文
    num_rejected_tokens_gpu=...,    # ④ 路径 B 独有
)
```

`_propose` 内部关键动作：
- eagle3 / dflash 调 `combine_hidden_states` 合并多层 aux hidden states（[L671-682](vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py#L671-L682)）
- `set_inputs_first_pass` 把三元组按 padded batch 摆好（[L684](vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py#L684)）
- 处理 cudagraph dispatch / lora / uniform_decode
- 调 `self.model(...)` 跑 draft 网络，输出 draft_token_ids

### 5.4 核心概念速查

| 概念 | 含义 |
|---|---|
| **padded batch** | 把每个 req 的 (sampled + draft) token 对齐成 `[num_reqs, num_spec+1]` 的 GPU tensor，rejected 用 -1 占位 |
| **disable_padded_drafter_batch** | 关闭 padded，改用 CPU list；兼容性更好但性能差 |
| **spec_decode_metadata** | 上一步 draft 的验证结果（cu_num_draft_tokens 等），None 表示本步无 draft 可验 |
| **token_indices** | 实际要喂给 drafter 的 token 在 input_ids 里的索引 |
| **token_indices_to_sample** | padded batch 里每个 req 真正要采样的位置（剔除 rejected 后） |
| **num_rejected_tokens_gpu** | 每个 req 上一步被 rejection 拒掉的 token 数 |
| **PCP** | Prefill Context Parallel，长 prefill 切分到多步执行 |
| **aux_hidden_states** | target 模型中间层暴露的 hidden states，eagle3/dflash/extract_hidden_states 用 |
| **get_mtp_target_hidden_states** | DeepSeek V4 MTP 特化钩子，返回 pre-hc_head residual |
| **need_accepted_tokens** | async 模式下决定是否走异步状态更新分支 |

### 5.5 难点与学习路径

#### 难点 1：padded vs cpu 两条路

不只是"GPU 还是 CPU"的差异，整个数据流都不同：
- padded：`prepare_next_token_ids_padded` + `prepare_inputs_padded`（一次 triton kernel 同时算 token_indices 和 num_rejected）
- cpu：`prepare_next_token_ids_cpu` + `prepare_inputs`（基于 CPU 的 query_start_loc 重排）

先吃透 padded（默认路径），cpu 路径作为兼容性兜底理解即可。

#### 难点 2：spec_decode_metadata 的两种状态

最容易混淆的点。**None 不代表"没开 spec_decode"，而是"本步没有上一步 draft 要 verify"**：
- 首步：None
- 上一步 draft 全被接受：仍是 None
- 上一步 draft 部分被拒：非 None，走 prepare_inputs_padded

遇到 `if spec_decode_metadata is None` 的分支，理解为"是不是首步 / 全接受步"。

#### 难点 3：PCP（Prefill Context Parallel）

`pcp_size > 1` 时整套逻辑会拐弯：
- `input_ids_pcp_full` 替代 `input_ids.gpu`
- `query_start_loc_pcp_full` 替代 `query_start_loc`
- `num_prefill_reqs` / `num_decode_reqs` 区分阶段
- 路径 B 里要先同步 `common_attn_metadata.query_start_loc` 到 pcp_full 版本（[L1860-1866](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1860-L1866)）

只看解码路径（pcp_size=1）时可暂时跳过这些分支。

#### 难点 4：DeepSeek V4 MTP 的 hidden_states 重绑

代码简短（4 行），但背后的"pre-hc_head residual"概念必须理解：
- target 模型有 `hc_head`，把 residual 转成 logits
- MTP 头训练时吃 residual，不是 logits-前那一层
- 所以要把 hidden_states 替换成 `get_mtp_target_hidden_states()` 的返回值

建议结合 DeepSeek V4 的模型结构图一起看。

#### 难点 5：`_propose` 内部的 set_inputs_first_pass / build_model_inputs

drafter 内部的"组装 padded batch"逻辑，在 [llm_base_proposer.py](vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py) 里。重点关注：
- `set_inputs_first_pass`：把 target 的三元组按 token_indices_to_sample 摆好
- `build_model_inputs_first_pass`：组 drafter 的 input_ids / positions / slot_mapping
- `dummy_run`：cudagraph 捕获时的 dry run

### 5.6 推荐学习路径

```
1. 先理解 spec_decode 的两阶段循环：target forward → draft → verify → 接受/拒绝 → 下一轮
   （把 rejection sampler 的循环逻辑搞清楚）

2. 跑通最简场景：单卡、无 PCP、padded、eagle method、首步（spec_decode_metadata=None）
   只看路径 A，理解三元组怎么来、drafter._propose 怎么调

3. 加上"非首步"：spec_decode_metadata 非空，看 prepare_inputs_padded
   理解 token_indices / token_indices_to_sample / num_rejected_tokens_gpu 三个输出

4. 加上 disable_padded_drafter_batch：对照 cpu 路径
   理解什么时候会强制走 cpu（MTP-fullgraph + PCP 不兼容）

5. 加上 PCP：pcp_size > 1 的所有分支
   理解 input_ids_pcp_full / num_prefill_reqs / num_decode_reqs

6. 加上 MTP 特化：get_mtp_target_hidden_states 的重绑
   理解 pre-hc_head residual 的语义

7. 最后看 _propose 内部：set_inputs_first_pass / build_model_inputs_first_pass
   / dummy_run / cudagraph dispatch
```

### 5.7 关键文件清单

| 文件 | 作用 |
|---|---|
| [model_runner_v1.py:1686-1915](vllm-ascend/vllm_ascend/worker/model_runner_v1.py#L1686-L1915) | `propose_draft_token_ids` 入口，5 个 drafter 分支 |
| [llm_base_proposer.py:643-712](vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py#L643-L712) | `_propose` 主流程 |
| [llm_base_proposer.py:1682-1737](vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py#L1682-L1737) | `prepare_next_token_ids_padded` |
| [llm_base_proposer.py:1739-1863](vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py#L1739-L1863) | `prepare_inputs`（cpu 路径） |
| [llm_base_proposer.py:1864+](vllm-ascend/vllm_ascend/spec_decode/llm_base_proposer.py#L1864) | `prepare_inputs_padded` |
| [eagle_proposer.py](vllm-ascend/vllm_ascend/spec_decode/eagle_proposer.py) | AscendEagleProposer 主体 |
| [dflash_proposer.py](vllm-ascend/vllm_ascend/spec_decode/dflash_proposer.py) | DFlash 覆写部分 |
| [draft_proposer.py](vllm-ascend/vllm_ascend/spec_decode/draft_proposer.py) | draft_model 路线 |

### 5.8 容易踩的坑

1. **`use_eagle()` 把 mtp/dflash 也算进去**：[speculative.py:1110](vllm/vllm/vllm/config/speculative.py#L1110) 的判定不是字面意义，dflash/mtp 都走这个分支，drafter 类才是真正的区分。
2. **路径 A vs 路径 B 的差异不是"首次 vs 后续"**：是"本步有没有上一步 draft 要 verify"。即使跑到第 N 步，如果上一步 draft 全被接受，本步 spec_decode_metadata 仍是 None。
3. **`num_rejected_tokens_gpu` 只在路径 B 出现**：路径 A 这个变量恒为 None，drafter 内部要处理两种情况。
4. **PCP 分支不要硬啃**：先跳过 `pcp_size > 1` 的所有分支，等基础流程通了再回头。
5. **`get_mtp_target_hidden_states` 是模型级钩子**：不是所有 target 模型都有，只有 DeepSeek V4 MTP 等少数模型实现了，其它模型这步是 no-op。
6. **cudagraph dispatch 在 _propose 里**：drafter 跑 draft 网络时也走 cudagraph，注意 `use_cuda_graph` 的判定和 `cudagraph_dispatcher.dispatch` 的逻辑。
