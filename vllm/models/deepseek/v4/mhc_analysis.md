# DeepSeek V4 MHC 机制分析报告（Ascend 版本）

> 基于 `vllm-ascend/vllm_ascend/models/deepseek_v4.py` 及 `vllm-ascend/csrc/torch_binding.cpp`

---

## 1. 概述

MHC（Multi-Head Context）是 DeepSeek V4 的核心创新机制，其本质是**多流残差并行计算**。

传统 Transformer 每层只有一条残差流（residual stream），而 MHC 维护 `hc_mult` 条并行残差流，每条流独立处理信息，在层内通过 `hc_pre` 混合信息、在层尾通过 `hc_post` 写回残差，最终在模型末尾通过 `hc_head` 合并回单流输出。

```
embedding → repeat扩展 → [layer0: hc_pre→attn→hc_post→hc_pre→ffn→hc_post] → ... → hc_head → norm → lm_head
```

---

## 2. 三大算子概览

| 算子 | 位置 | 触发时机 | 职责 |
|------|------|----------|------|
| `hc_pre` | 每个子层开头（attn前/ffn前） | 每层调用2次 | 多流混合，生成子层输入 + 写回指令 |
| `hc_post` | 每个子层结尾（attn后/ffn后） | 每层调用2次 | 将子层结果按写回指令写回多流残差 |
| `hc_head` | 所有层结束后 | 仅调用1次 | 多流合并为单流，送 lm_head |

---

## 3. hc_pre — 多流混合（层内入口）

### 3.1 代码位置

**Python 调用入口：** `vllm_ascend/models/deepseek_v4.py` L835-839

```python
def hc_pre(self, x: torch.Tensor, hc_fn: torch.Tensor, hc_scale: torch.Tensor, hc_base: torch.Tensor):
    y, post, comb = torch.ops._C_ascend.npu_hc_pre(
        x, hc_fn, hc_scale, hc_base, self.hc_mult, self.hc_sinkhorn_iters, self.norm_eps, self.hc_eps
    )
    return y, post, comb
```

**C++ 算子实现：** `vllm-ascend/csrc/torch_binding.cpp`

| 函数 | 行号 | 说明 |
|------|------|------|
| `npu_hc_pre_npu` | L1479-1485 | 默认入口，调用 composite |
| `run_hc_pre_composite` | L1438-1463 | 三步组合实现 |
| `npu_hc_pre_v2_npu` | L1487-1496 | v2 入口，可选融合算子 |
| `run_hc_pre_fusion` | L1465-1477 | 全融合 `aclnnHcPre` |
| 算子注册 | L2573-2589 | `torch.ops._C_ascend.npu_hc_pre` |

### 3.2 输入输出

```
输入：
  x:         (num_tokens, hc_mult, hidden_dim)  ← 多流残差
  hc_fn:     ((2+hc_mult)×hc_mult, hc_mult×h)   ← 可学习权重矩阵
  hc_scale:  (3,)                                ← 缩放系数
  hc_base:   ((2+hc_mult)×hc_mult,)             ← 偏置系数

输出：
  y:         (num_tokens, hidden_dim)            ← 单流，送入 attn/ffn
  post:      (num_tokens, hc_mult)               ← 写回权重，传给 hc_post
  comb:      (num_tokens, hc_mult, hc_mult)      ← 双随机矩阵，传给 hc_post
```

### 3.3 计算原理（composite 三步）

**Step 1：`aclnnHcPreInvRms` — 量尺子**

```cpp
EXEC_NPU_CMD(aclnnHcPreInvRms, x, norm_eps, rsqrt);
// rsqrt: (num_tokens, 1)
// 计算每个token所有流的平均幅度倒数，用于后续归一化
```

**Step 2：`at::linear` — 决策**

```cpp
x_flattened = x_float.flatten(2, -1);           // (tokens, hc_mult × h)
mixes = at::linear(x_flattened, hc_fn);         // (tokens, (2+hc_mult)×hc_mult)
// mixes 包含三块原始系数：
//   [0:hc_mult]         → pre_mix（输入混合参数）
//   [hc_mult:2*hc_mult] → post_mix（写回参数）
//   [2*hc_mult:]        → comb_mix（混合矩阵原始分数）
```

**Step 3：`aclnnHcPreSinkhorn` — 规范化 + 执行混合**

```cpp
EXEC_NPU_CMD(aclnnHcPreSinkhorn, mixes, rsqrt, hc_scale, hc_base, x,
             hc_mult, hc_sinkhorn_iters, hc_eps, y, post, comb_frag);
```

内部做三件事：
1. **Sinkhorn 迭代**：将 comb_mix 原始分数归一化为双随机矩阵（每行每列和均为1），保证信息总量守恒
2. **加权混合**：双随机矩阵 × 多流 → 单流 y（送入 attn/ffn）
3. **计算 post**：根据 post_mix 参数生成各流写回权重

### 3.4 每层有两套参数

每层 decoder 有 attention 和 FFN 两个子层，各自拥有独立的 hc_pre 参数：

```python
# deepseek_v4.py L828-833
self.hc_attn_fn   / self.hc_ffn_fn     # 混合权重矩阵
self.hc_attn_base / self.hc_ffn_base   # 偏置
self.hc_attn_scale / self.hc_ffn_scale  # 缩放系数
```

---

## 4. hc_post — 多流残差写回（层内出口）

### 4.1 代码位置

**Python 调用入口：** `vllm_ascend/models/deepseek_v4.py` L841-845

```python
def hc_post(self, x: torch.Tensor, residual: torch.Tensor, post: torch.Tensor, comb: torch.Tensor):
    y = torch.ops._C_ascend.npu_hc_post(
        x.unsqueeze(dim=0), residual.unsqueeze(dim=0), post.unsqueeze(dim=0), comb.unsqueeze(dim=0)
    )
    return y.squeeze(dim=0)
```

**C++ 算子实现：** `vllm-ascend/csrc/torch_binding.cpp`

| 函数 | 行号 | 说明 |
|------|------|------|
| `npu_hc_post_npu` | L1310-1321 | 调用单一 `aclnnHcPost` |
| `check_hc_post_shape_and_dtype` | L1274-1308 | 形状校验 |
| 算子注册 | L2563-2571 | `torch.ops._C_ascend.npu_hc_post` |

### 4.2 输入输出

```
输入：
  x:         (num_tokens, hidden_dim)            ← attn/ffn 输出，单流
  residual:  (num_tokens, hc_mult, hidden_dim)   ← 层前 clone 的多流快照
  post:      (num_tokens, hc_mult)               ← 来自 hc_pre 的写回权重
  comb:      (num_tokens, hc_mult, hc_mult)      ← 来自 hc_pre 的双随机矩阵

输出：
  out:       (num_tokens, hc_mult, hidden_dim)   ← 更新后的多流残差
```

### 4.3 计算原理

参考实现（`vllm/model_executor/kernels/mhc/torch.py` L94-106）：

```python
# 第一步：用 comb 混合各流残差（层前状态互相交换）
mixed_residual = einsum("...ij,...ih->...jh", comb, residual)
# output[j][h] = Σᵢ comb[i][j] × residual[i][h]
# 即：各流按比例交换旧状态

# 第二步：将单流结果 x 按 post 权重广播到各流
post_term = post × x.unsqueeze(-2)
# 每条流独立决定接收多少 x

# 第三步：相加
out = mixed_residual + post_term
```

### 4.4 设计原理：为什么先 comb 后 post

| | 当前顺序（先 comb 后 post） | 反过来（先 post 后 comb） |
|---|---|---|
| 公式 | `comb×residual + post×x` | `comb×(residual + post×x)` |
| post 控制力 | 精确：各流独立接收 post[i]×x | 被污染：comb 会重新分配 post×x |
| 语义清晰度 | comb 混合旧信息，post 控制新信息 | 两个职责混在一起 |

先 comb 后 post 保证了：**comb 只负责交换历史信息，post 独立控制新信息的接收量。**

---

## 5. hc_head — 多流合并（模型出口）

### 5.1 代码位置

**实现位置：** `vllm_ascend/models/deepseek_v4.py` L957-964（纯 PyTorch，无 NPU 算子）

```python
def hc_head(self, x, hc_fn, hc_scale, hc_base):
    shape, dtype = x.size(), x.dtype
    x = x.flatten(1).float()
    rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + self.norm_eps)
    mixes = torch.nn.functional.linear(x, hc_fn) * rsqrt
    pre = torch.sigmoid(mixes * hc_scale + hc_base) + self.hc_eps
    y = torch.sum(pre.unsqueeze(-1) * x.view(shape), dim=1)
    return y.to(dtype)
```

**调用位置：** `DeepseekV4Model.forward()` L1029

### 5.2 输入输出

```
输入：
  x:         (num_tokens, hc_mult, hidden_dim)  ← 所有层结束后的多流状态
  hc_fn:     (hc_mult, hc_mult × hidden_dim)    ← 可学习打分权重
  hc_scale:  (1,)                                ← 缩放系数
  hc_base:   (hc_mult,)                         ← 偏置

输出：
  y:         (num_tokens, hidden_dim)            ← 单流，送 norm → lm_head
```

### 5.3 计算原理（纯 PyTorch，4步）

```python
# Step 1: 拍平 + float32
x = x.flatten(1).float()              # (tokens, hc_mult × h)

# Step 2: 计算 RMS 归一化因子
rsqrt = torch.rsqrt(x.square().mean(-1, keepdim=True) + norm_eps)  # (tokens, 1)

# Step 3: 给每条流打重要性分数
mixes = linear(x, hc_fn) * rsqrt      # (tokens, hc_mult)
pre = sigmoid(mixes * hc_scale + hc_base) + hc_eps  # 压到(0,1)区间

# Step 4: 按分数加权合并
y = sum(pre.unsqueeze(-1) * x.view(shape), dim=1)  # (tokens, h)
# y = pre[0]×stream0 + pre[1]×stream1 + ... + pre[hc_mult-1]×stream_{hc_mult-1}
```

### 5.4 为什么 hc_head 没有 hc_post

`hc_head` 位于模型最后一层之后，合并后的单流直接进入 `norm` → `lm_head`，没有后续子层，因此不需要写回多流残差。

---

## 6. 完整数据流

### 6.1 模型级流程（DeepseekV4Model.forward，L966-1032）

```
input_ids
    ↓ [仅 PP first rank]
embed_tokens → hidden_states: (num_tokens, hidden_dim)
    ↓ [仅 PP first rank]
unsqueeze + repeat → hidden_states: (num_tokens, hc_mult, hidden_dim)
    ↓
┌─── layer 0 ───────────────────────────────────────────┐
│  residual = clone(hidden_states)         [多流]        │
│  y, post, comb = hc_pre(hidden_states, hc_attn_*)      │
│  y = norm(y)                                           │
│  y = self_attn(y)                        [单流]        │
│  hidden_states = hc_post(y, residual, post, comb) [多流]│
│                                                        │
│  residual = clone(hidden_states)         [多流]        │
│  y, post, comb = hc_pre(hidden_states, hc_ffn_*)       │
│  y = norm(y)                                           │
│  y = mlp(y)                              [单流]        │
│  hidden_states = hc_post(y, residual, post, comb) [多流]│
└────────────────────────────────────────────────────────┘
    ↓ × num_hidden_layers
hc_head(hidden_states) → (num_tokens, hidden_dim)  [单流]
    ↓
norm
    ↓
lm_head → logits
```

### 6.2 单 Decoder 层内的数据结构变化

```
进入层: hidden_states (tokens, hc_mult, h)  多流
    │
    ├── clone → residual (tokens, hc_mult, h)
    │
    ├── hc_pre(attn参数)
    │     输入: (tokens, hc_mult, h)  多流
    │     输出: y (tokens, h)         单流  ← 送入attn
    │           post (tokens, hc_mult)
    │           comb (tokens, hc_mult, hc_mult)
    │
    ├── norm → self_attn → y (tokens, h)  单流
    │
    ├── hc_post
    │     输入: x (tokens, h)              单流
    │           residual (tokens, hc_mult, h)  多流
    │           post, comb (来自hc_pre)
    │     输出: (tokens, hc_mult, h)       多流  ← 残差更新
    │
    ├── clone → residual (tokens, hc_mult, h)
    │
    ├── hc_pre(ffn参数) → 同上，单流送入mlp
    │
    ├── norm → mlp → y (tokens, h)  单流
    │
    └── hc_post → (tokens, hc_mult, h)  多流
        │
        离开层: hidden_states (tokens, hc_mult, h)  多流
```

---

## 7. 三者的逻辑关系

```
              hc_pre                      hc_post                     hc_head
          ┌─────────────┐            ┌──────────────┐           ┌────────────┐
  多流 ──→│ 多流混合     │──→ 单流 ──→│ attn/ffn计算 │──→ 单流 ──→│            │
          │ 生成写回指令  │            │ 写回多流残差  │           │            │
          └─────────────┘            └──────────────┘           │ 多流合并    │
                                                                 │ 为单流     │
  post,comb ─────────────────────────────────────→ hc_post       │            │
                （跨子层传递写回指令）                               └────────────┘
                                                                    模型末尾
                                                                    仅调用1次
```

| 关系 | 说明 |
|------|------|
| hc_pre → hc_post | hc_pre 生成的 `post` 和 `comb` 直接传给同子层的 hc_post |
| hc_post → 下一层 hc_pre | hc_post 输出的多流残差是下一层 hc_pre 的输入 |
| 最后一层 hc_post → hc_head | 所有层结束后，多流残差送入 hc_head 合并 |

---

## 8. 关键参数汇总

### 每层 Decoder 的参数（`DeepseekV2DecoderLayer.__init__`，L828-833）

```python
# attention 子层专用
hc_attn_fn:    ((2+hc_mult)×hc_mult, hc_mult×h)   float32
hc_attn_base:  ((2+hc_mult)×hc_mult,)              float32
hc_attn_scale: (3,)                                 float32

# FFN 子层专用
hc_ffn_fn:     ((2+hc_mult)×hc_mult, hc_mult×h)   float32
hc_ffn_base:   ((2+hc_mult)×hc_mult,)              float32
hc_ffn_scale:  (3,)                                 float32
```

### 模型级参数（`DeepseekV4Model.__init__`，L940-942）

```python
# hc_head 专用（仅 PP last rank）
hc_head_fn:    (hc_mult, hc_mult×h)               float32
hc_head_base:  (hc_mult,)                          float32
hc_head_scale: (1,)                                float32
```

---

## 9. 配置参数来源

来自 `hf_config`（DeepSeek V4 模型配置文件）：

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `hc_mult` | 并行流条数 | 如 4 |
| `hc_sinkhorn_iters` | Sinkhorn 迭代次数 | 如 3 |
| `hc_eps` | sigmoid 下限保护 | 如 1e-4 |
| `rms_norm_eps` | RMS 归一化 epsilon | 如 1e-6 |

---

## 10. PP（Pipeline Parallelism）的影响

| 位置 | PP first rank | PP 中间 rank | PP last rank |
|------|---------------|--------------|--------------|
| embed_tokens + repeat | ✓ | ✗ | ✗ |
| hc_pre / hc_post（每层） | ✓ | ✓ | ✓ |
| hc_head | ✗ | ✗ | ✓ |
| norm | ✗ | ✗ | ✓ |

- `hc_mult` 维度的展开在 **first rank** 完成（L997）
- 中间 rank 接收的 `intermediate_tensors` 已是多流形态
- `hc_head` 合并回单流仅在 **last rank** 执行
