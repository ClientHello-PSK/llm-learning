# 知识点 07：DeepSeek V4 的 PP Forward 实现详解

> 核心问题：DeepSeek V4 怎么用 PP？forward 的每个分支做什么？
>
> 本文以 **vllm-ascend 实现**为线索展开，源码：[vllm_ascend/models/deepseek_v4.py:1103-1169](../../../vllm-ascend/vllm_ascend/models/deepseek_v4.py#L1103-L1169)
>
> 上游 vLLM 也有 DeepseekV4Model，按平台分目录：
> - NVIDIA：[vllm/vllm/models/deepseek_v4/nvidia/model.py:888](../../../vllm/vllm/models/deepseek_v4/nvidia/model.py#L888)（forward L1006）
> - AMD：[vllm/vllm/models/deepseek_v4/amd/model.py:437](../../../vllm/vllm/models/deepseek_v4/amd/model.py#L437)（forward L556）
> - XPU：[vllm/vllm/models/deepseek_v4/xpu/model.py:978](../../../vllm/vllm/models/deepseek_v4/xpu/model.py#L978)
>
> 各平台 **PP 核心逻辑一致**（`is_first_rank` / `is_last_rank` / `IntermediateTensors`），MHC 算子实现不同。

---

## 1. 整体流程图

```
input_ids (only first rank)
    ↓
embed_tokens (only first rank)
    ↓
unsqueeze + repeat(hc_mult) (only first rank)   ← MHC 多流展开
    ↓
┌─── for layer in layers[start_layer:end_layer] ───┐
│    residual = clone(hidden_states)                │
│    y, post, comb = hc_pre(hidden_states, attn参数)│
│    y = norm(y); y = self_attn(y)                  │
│    hidden_states = hc_post(y, residual, ...)      │
│    （同样套路对 FFN 子层）                          │
└───────────────────────────────────────────────────┘
    ↓
save MTP buffer
    ↓
if not is_last_rank:
    return IntermediateTensors({"hidden_states": hidden_states})   ← 继续传下去
    ↓
hc_head(hidden_states)  ← 多流合并
    ↓
norm
    ↓
return hidden_states    ← 最终输出
```

---

## 2. forward() 签名（L1103-1109）

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors:
```

参数说明：
- `input_ids`：first rank 才用到（其他 rank 从 intermediate_tensors 接收）
- `positions`：所有 rank 都需要（位置编码）
- `intermediate_tensors`：非 first rank 接收的输入
- `inputs_embeds`：可选，预计算的 embedding

---

## 3. 入口分支：first rank vs 其他 rank（L1110-1119）

```python
if get_pp_group().is_first_rank:
    if inputs_embeds is not None:
        hidden_states = inputs_embeds
    else:
        hidden_states = self.embed_input_ids(input_ids)
    residual = None
else:
    assert intermediate_tensors is not None
    hidden_states = intermediate_tensors["hidden_states"]
    residual = None
```

### 行为对比

| Rank | 输入来源 | 处理 |
|------|---------|------|
| first | `input_ids` 或 `inputs_embeds` | 调用 embed_tokens 转 embedding |
| 中间 | `intermediate_tensors["hidden_states"]` | 直接接收上一 stage 的多流张量 |
| last | 同中间 | 同上（最后还要 hc_head 合并） |

### 关键点

- `residual = None`：first rank 没有"上一层残差"
- `intermediate_tensors` 必须非空（除非是 first rank），否则 assert 失败

---

## 4. 多流展开（MHC）：只在 first rank（L1133-1134）

```python
if get_pp_group().is_first_rank:
    hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
    # (num_tokens, hidden_dim) → (num_tokens, hc_mult, hidden_dim)
```

### 为什么只在 first rank？

因为多流形态是 **tensor shape 的变化**，一旦展开就贯穿所有层。中间 rank 接收到的张量已经是多流形态了。

### 张量形状演变

```
embed_tokens 输出:   (num_tokens, hidden_dim)             ← 单流
first rank 展开:     (num_tokens, hc_mult, hidden_dim)    ← 多流
所有 decoder layer:  (num_tokens, hc_mult, hidden_dim)    ← 多流贯穿
hc_head 合并:        (num_tokens, hidden_dim)             ← 单流
```

### MHC 跟 PP 的天作之合

MHC 的关键特性：**所有层都保持多流形态，只在 hc_head 一次性合并**。

这意味着：
- PP 段间传递的张量是 (num_tokens, hc_mult, hidden_dim)，**所有段形状一致**
- 合并动作只在 last rank
- PP 实现不需要为 MHC 做任何特殊处理

---

## 5. 层切分：start_layer / end_layer（L1135-1136）

```python
for layer in islice(self.layers, self.start_layer, self.end_layer):
    hidden_states, residual = layer(positions, hidden_states, residual, llama_4_scaling)
```

### 切分逻辑

`start_layer` 和 `end_layer` 在 `__init__` 中通过 `make_layers` 设置（L1044）：

```python
self.start_layer, self.end_layer, self.layers = make_layers(...)
```

**例**：64 层模型，PP=4：

| Rank | start_layer | end_layer | 实际跑的层 |
|------|------------|-----------|----------|
| 0 | 0 | 16 | layer 0-15 |
| 1 | 16 | 32 | layer 16-31 |
| 2 | 32 | 48 | layer 32-47 |
| 3 | 48 | 64 | layer 48-63 |

### 为什么用 islice 而不是直接切片？

`self.layers` 是 `nn.ModuleList`，不能直接切片。`islice` 是迭代器切片。

### 每层只跑自己负责的层

每个 rank 加载的 `self.layers` 是**完整模型的所有层**，但只跑自己负责的段。其他层的权重不加载（通过 weight loader 控制）。

---

## 6. MTP Buffer 保存（L1138-1157）

```python
# Stash pre-hc_head residual for the MTP draft (captured copy_).
# When FlashComm1 (sequence parallelism) is enabled, tokens are
# partitioned across TP ranks via reduce_scatter in each layer's
# row-parallel output projection. We must all_gather here so the
# MTP layers receive the full token set.
from vllm_ascend.ascend_forward_context import get_forward_context

forward_ctx = get_forward_context()
if forward_ctx is not None and forward_ctx.flash_comm_v1_enabled:
    h_states_flat = tensor_model_parallel_all_gather(hidden_states.flatten(1), dim=0)
    pad_size = forward_ctx.pad_size
    if pad_size > 0:
        h_states_flat = h_states_flat[:-pad_size]
    num_tokens = h_states_flat.shape[0]
    self._mtp_hidden_buffer[:num_tokens].copy_(h_states_flat)
else:
    num_tokens = hidden_states.shape[0]
    self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))
```

### 用途

保存 `hidden_states` 给 MTP（推测解码）的 draft model 用。

### FlashComm1 的特殊处理

FlashComm1（sequence parallelism）启用时，token 在 TP rank 间分散，需要 all_gather 收集完整 token。

---

## 7. 出口分支：last rank vs 其他 rank（L1159-1169）

```python
if not get_pp_group().is_last_rank:
    return IntermediateTensors(
        {
            "hidden_states": hidden_states,
        }
    )

# 只有 last rank 才走到这里
hidden_states = self.hc_head(hidden_states, self.hc_head_fn, self.hc_head_scale, self.hc_head_base)
hidden_states = self.norm(hidden_states)
return hidden_states
```

### 行为对比

| Rank | 返回值 | 后续动作 |
|------|--------|---------|
| 中间 | `IntermediateTensors({"hidden_states": ...})` | 上层调用 send_tensor_dict 传给下一 stage |
| last | 单流 hidden_states（已合并 + norm） | 给 sampler 采样 |

### 关键点

- 非 last rank 不做 hc_head（保留多流形态传递）
- 非 last rank 不做 norm（norm 在 last rank 做一次）
- 张量是 **hidden_states 没变**，只是包成 IntermediateTensors

---

## 8. Decoder Layer 内部：PP 无关（L984-）

`DeepseekV2DecoderLayer.forward()` 在 L984 开始，关键代码（hc_pre 在 L972、hc_post 在 L978）：

```python
# DeepseekV2DecoderLayer.forward()
def forward(self, positions, hidden_states, residual, llama_4_scaling):
    # attention 子层
    residual = hidden_states.clone()
    hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_attn_fn, ...)
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states, ...)
    hidden_states = self.hc_post(hidden_states, residual, post, comb)

    # FFN 子层
    residual = hidden_states.clone()
    hidden_states, post, comb = self.hc_pre(hidden_states, self.hc_ffn_fn, ...)
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = self.hc_post(hidden_states, residual, post, comb)

    return hidden_states, residual
```

### 关键观察

- Decoder layer **完全不感知 PP**
- 它只看到多流输入、做多流输出
- 所有 PP 相关逻辑都集中在 `DeepseekV4Model.forward()`

这种设计的好处：**模型层逻辑可复用**，PP 是上层 wrapper。

---

## 9. 完整时序示例：PP=2，单条 prompt

假设 PP=2，输入一条 prompt "Hello world"（3 个 token）：

```
┌─────── Rank 0 (Stage 0) ───────┐  ┌─────── Rank 1 (Stage 1) ───────┐
│                                  │  │                                  │
│ input_ids = [hello, world, eos]  │  │ (等待 intermediate_tensors)      │
│           ↓                      │  │                                  │
│ embed_tokens                     │  │                                  │
│           ↓                      │  │                                  │
│ hidden_states: (3, h)            │  │                                  │
│           ↓                      │  │                                  │
│ unsqueeze + repeat(hc_mult)      │  │                                  │
│           ↓                      │  │                                  │
│ hidden_states: (3, hc_mult, h)   │  │                                  │
│           ↓                      │  │                                  │
│ for layer in layers[0:32]:       │  │                                  │
│     layer.forward(...)           │  │                                  │
│           ↓                      │  │                                  │
│ return IntermediateTensors({     │──│──→ recv: hidden_states           │
│     "hidden_states": hidden_states│  │           ↓                     │
│ })                               │  │ for layer in layers[32:64]:      │
│                                  │  │     layer.forward(...)           │
│                                  │  │           ↓                     │
│                                  │  │ hc_head → norm                   │
│                                  │  │           ↓                     │
│                                  │  │ return hidden_states             │
│                                  │  │           ↓                     │
│                                  │  │ sampler → next token             │
└──────────────────────────────────┘  └──────────────────────────────────┘
```

---

## 10. 关键代码片段汇总（行号已验证）

| 位置 | 代码 | 作用 |
|------|------|------|
| L1103 | `def forward(...)` | forward 函数定义 |
| L1110 | `if get_pp_group().is_first_rank` | first rank 分支 |
| L1114 | `self.embed_input_ids(input_ids)` | first rank 转 embedding |
| L1118 | `intermediate_tensors["hidden_states"]` | 非 first rank 接收 |
| L1133 | `if get_pp_group().is_first_rank` | 多流展开判断 |
| L1134 | `hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)` | 多流展开 |
| L1135 | `islice(self.layers, self.start_layer, self.end_layer)` | 层切分 |
| L1148 | `forward_ctx.flash_comm_v1_enabled` | FlashComm1 检测 |
| L1159 | `if not get_pp_group().is_last_rank` | 非 last rank 返回 |
| L1160 | `IntermediateTensors({"hidden_states": hidden_states})` | 打包传递 |
| L1166 | `self.hc_head(...)` | last rank 多流合并 |
| L1168 | `self.norm(hidden_states)` | last rank 归一化 |

### 相关 MHC 方法位置

| 方法 | 行号 |
|------|------|
| `hc_pre` | L972 |
| `hc_post` | L978 |
| `hc_head` | L1094 |
| `class DeepseekV2DecoderLayer` | L913 |
| `class DeepseekV4Model` | L1007 |

---

## 11. 小结

- forward 的核心：**3 个判断点**（first/last rank、层切分、多流展开）
- 多流展开只在 first rank，因为张量形状变化后贯穿所有层
- 非 last rank 返回 `IntermediateTensors`，由上层调用 `send_tensor_dict` 传输
- Decoder layer 完全不感知 PP，PP 逻辑全在 Model.forward
- MHC 跟 PP 配合好：所有层保持多流形态，形状一致

---

## 12. 上游 vs vllm-ascend 的实现差异

| 维度 | 上游 nvidia（L1006-1057） | vllm-ascend（L1103-1169） |
|------|------------------------|-------------------------|
| 多流展开位置 | forward 开头（L1018）紧跟 embed | 中段（L1133-1134），embed 之后 |
| Decoder 层 MHC | 层内累积 `post_mix`/`res_mix`，最后一次性 `mhc_post_tilelang` | 每层 `hc_pre` + `hc_post` 即时合并 |
| MTP buffer 保存 | last rank 之后才保存（L1045-1046） | 层循环之后、is_last_rank 分支前保存（L1145-1157） |
| 多流合并 | `hc_head_fused_kernel_tilelang`（含 norm） | `self.hc_head` + 显式 `self.norm`（L1166-1168） |
| FlashComm1（sequence parallelism） | ✗ | ✓（L1145-1157，all_gather 还原完整 token） |
| PP 核心逻辑 | `is_first_rank` / `is_last_rank` / `IntermediateTensors` | **完全一致** |

**关键观察**：PP 相关的框架代码（rank 判断、tensor 流转、层切分）在所有平台**几乎一致**，差异集中在 MHC（多流残差）算子和 FlashComm1 这类硬件特化优化上。学 PP 时聚焦通用框架逻辑即可，算子差异是硬件适配范畴。

---

**下一步**：[08_通信机制.md](08_通信机制.md) — stage 间到底怎么 send/recv？用什么通信后端？
