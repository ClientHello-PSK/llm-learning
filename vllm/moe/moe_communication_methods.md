# vLLM-Ascend MoE 四种通信方式详解（ALLGATHER / ALLTOALL / MC2 / FUSED_MC2）

> 本文基于 `vllm-ascend` 源码整理，所有结论均标注源码文件与行号，便于回溯。
> 适用对象：阅读 DeepSeek V4 MoE 在 Ascend NPU 上实现细节的开发者。
> 涉及代码仓库：`vllm-ascend`（`m:\code\llm-learning\vllm\vllm-ascend`）。

---

## 0. 目录

1. [背景：MoE 通信要解决什么问题](#1-背景moe-通信要解决什么问题)
2. [整体调用链与通用骨架](#2-整体调用链与通用骨架)
3. [四种方式的共性：两个策略 + 两个维度](#3-四种方式的共性两个策略--两个维度)
4. [ALLGATHER 详解](#4-allgather-详解)
5. [ALLTOALL 详解](#5-alltoall-详解)
6. [MC2 详解](#6-mc2-详解)
7. [FUSED_MC2 详解](#7-fused_mc2-详解)
8. [触发条件：select_moe_comm_method 决策树](#8-触发条件select_moe_comm_method-决策树)
9. [mc2_tokens_capacity 是怎么算的](#9-mc2_tokens_capacity-是怎么算的)
10. [多维对比表](#10-多维对比表)
11. [本质区别与演进关系](#11-本质区别与演进关系)
12. [选型建议](#12-选型建议)
13. [代码索引](#13-代码索引)
14. [术语表](#14-术语表)

---

## 1. 背景：MoE 通信要解决什么问题

MoE（Mixture of Experts）层里，每个 token 会被路由到 1 个或多个专家。在多卡部署下，专家分布在不同卡上：

- **专家并行（EP, Expert Parallel）**：专家被切分到多张卡，每卡只持有部分专家。某个 token 想去的专家可能在别的卡上。
- **张量并行（TP, Tensor Parallel）**：单个专家的权重在多张卡间切分（如按列切），计算结果需要 AllReduce。
- **数据并行（DP, Data Parallel）**：每卡独立处理一批 token，但 EP 模式下需要先把全量 token 聚齐再路由。

因此，MoE 层在跨卡时必须解决两件事：

| 维度 | 要做的事 | 时机 |
|------|---------|------|
| **TP 维度** | 把单专家的张量并行计算结果合并（AllReduce / ReduceScatter） | 计算前后 |
| **EP 维度** | 把 token 按"它想去的专家所在的卡"分发出去，算完再收回 | 计算前后 |

**四种通信方式（ALLGATHER / ALLTOALL / MC2 / FUSED_MC2）本质上就是 EP 这一层"如何把 token 发出去、算完怎么收回"的不同策略。** 算子层面，它们由 `AscendMoERunner.forward_impl` 在阶段 ③⑤⑦ 调用（见 `fused_moe.py:609-793`）。

---

## 2. 整体调用链与通用骨架

### 2.1 从模型到通信算子的完整调用链

```
deepseek_v4.py: DeepseekV4MoE.forward()
        │  self.experts(hidden_states, router_logits)
        ▼
vllm/.../fused_moe/layer.py: FusedMoE.forward()        # L1318，仅是壳（PluggableLayer）
        │  self.runner.forward()
        ▼
vllm-ascend/.../fused_moe.py: AscendMoERunner.forward_impl()
        │  ★ 真正的实现，8 阶段流水线（见下方）
        ▼
vllm-ascend/.../fused_moe.py: AscendFusedMoE.forward_impl()   # 计算本体（W8A8 等）
        │  ③ moe_comm.prepare()   →  ⑤ quant_method.apply()  →  ⑦ moe_comm.combine/finalize()
        ▼
vllm-ascend/.../moe_comm_method.py: XxxCommImpl.fused_experts()   # ← 四种方式在这里分叉
```

> 来源：`fused_moe.py:609-793`（forward_impl 的 ①~⑧ 注释）、`moe_comm_method.py:122-171`（fused_experts 骨架）。

### 2.2 8 阶段流水线中，通信发生在哪里

`forward_impl`（`fused_moe.py:609-793`）把一次 MoE 拆成 8 个阶段：

| 阶段 | 内容 | 是否通信 |
|------|------|---------|
| ① | 预处理：`moe_layer_index` 回绕 + 强制负载均衡 | 否 |
| ② | **门控流重叠**：路由打分 + 共享专家（与主流异步） | 否（仅路由计算） |
| ③ | **通信准备**：`moe_comm.prepare()` | **是（TP/EP 聚合）** |
| ④ | 主流等待门控流（Event 同步） | 否（流同步） |
| ⑤ | **核心分组矩阵乘**：`quant_method.apply()` | 否（本地计算） |
| ⑥ | EPLB 负载热度收集 | 否 |
| ⑦ | **通信结束**：`moe_comm.combine()` + 规约 | **是（TP/EP 收回合并）** |
| ⑧ | 返回结果（带/不带 Event） | 否 |

**四种通信方式的差异，全部集中在阶段 ③ 和 ⑦。**

---

## 3. 四种方式的共性：两个策略 + 两个维度

所有四种方式都遵循基类 `MoECommMethod`（`moe_comm_method.py:87`）定义的统一接口：

```python
class MoECommMethod:
    def prepare(self, hidden, ...): ...       # ③ 阶段：token 发出去
    def fused_experts(self, hidden, ...): ... # ⑤ 阶段包装：dispatch→MLP→combine
    def finalize(self, output, ...): ...      # ⑦ 阶段：token 收回 + 规约
```

`fused_experts`（`moe_comm_method.py:122-171`）有一个通用三步骨架：

```
token_dispatch(hidden)   →   _apply_mlp(hidden)   →   token_combine(hidden)
```

每种方式通过组合**两个策略对象**来实现差异：

1. **TokenDispatcher**（`token_dispatcher.py`）：负责"路由后重排 token"。
   - `token_dispatch()`：把 token 按路由结果重排（本地）或跨卡发送（EP）。
   - `token_combine()`：把算完的 token 按"原始位置"还原 + 加权求和。

2. **PrepareAndFinalize**（`prepare_finalize.py`）：负责"TP/DP 维度的通信"。
   - `prepare()`：计算前的 AllGather / 填充对齐。
   - `finalize()`：计算后的 AllReduce / ReduceScatter / 去填充。

### 两个正交维度（务必区分，这是理解四种方式的关键）

| 维度 | 策略对象 | 解决问题 | 是否跨卡路由 token |
|------|---------|---------|------------------|
| **维度 A：TP/DP 聚散** | PrepareAndFinalize | 张量并行结果合并 / 数据并行 token 聚齐 | 不改变 token 所属专家，仅聚散 |
| **维度 B：EP 路由** | TokenDispatcher | 把 token 送到"目标专家所在的卡" | 改变 token 分布，按专家路由 |

> 这两个维度是**正交**的：PrepareAndFinalize 管"张量/数据并行聚合"，TokenDispatcher 管"专家路由分发"。四种方式在这两个维度上各有不同实现。

---

## 4. ALLGATHER 详解

> 实现类：`AllGatherCommImpl`（`moe_comm_method.py:185`）
> 策略组合：`TokenDispatcherWithAllGather`（`token_dispatcher.py:350`） + `PrepareAndFinalizeWithAllGather`（`prepare_finalize.py:321`）

### 4.1 核心思想

ALLGATHER 是**最朴素、兼容性最高**的方式：所有卡把 token 全部聚齐（AllGather），每卡都在"全量 token"上跑 MoE，最后用 ReduceScatter 把结果按 rank 切回并求和。

**它不在 EP 维度跨卡路由 token**——每卡都看到所有 token，各自只处理本地专家那部分。本质是"用全量广播 + 分片求和"来规避显式的 token 路由通信。

### 4.2 处理流程

#### Dispatcher（`token_dispatcher.py:350-440`）—— 纯本地

- `token_dispatch()`：调用 `npu_moe_init_routing`（`token_dispatcher.py:402`）。**只做本地重排序**（按专家编号把 token 重新排列成 `[expert0的tokens, expert1的tokens, ...]`），**不跨卡**。
- `token_combine()`：调用 `npu_moe_token_unpermute`（`token_dispatcher.py:429`）。把算完的 token 还原到原始顺序并按路由权重加权求和。

#### PrepareAndFinalize（`prepare_finalize.py:321-520`）—— 通信核心

通信顺序取决于是否开启序列并行（SP）：

**非 SP 模式：**
```
Attn(每卡独立)
  → TP AllReduce          # 张量并行合并
  → DP AllGather          # 聚齐全量 token（每卡现在都有全部 token）
  → MoE 本地计算           # 每卡在全量 token 上跑本地专家
  → DP ReduceScatter      # 按 DP rank 切分并求和
  → TP AllReduce          # 张量并行合并输出
```

**SP 模式（序列并行）：**
```
TP AllGather → Attn → TP ReduceScatter → EP AllGather → MoE → EP ReduceScatter
```

### 4.3 触发条件

ALLGATHER 是**保底/回退方案**，触发条件（`ascend_forward_context.py:274-309` 决策树）：

- EP 未启用（`ep_size == 1`），即没有专家并行；
- 或运行在 310P 芯片（不支持 MC2）；
- 或其它三种方式都不满足时的回退（fallback）。

### 4.4 特点

- ✅ **兼容性最好**：不需要 EP，310P 也能跑。
- ✅ 实现简单，纯集合通信 + 本地重排。
- ❌ **通信量最大**：每卡都要拿到全量 token（AllGather 全广播），带宽消耗随卡数线性增长。
- ❌ 不适合大规模 EP 部署。

---

## 5. ALLTOALL 详解

> 实现类：`AlltoAllCommImpl`（`moe_comm_method.py:235`）
> 策略组合：`TokenDispatcherWithAll2AllV`（`token_dispatcher.py:441`） + `PrepareAndFinalizeWithAll2All`（`prepare_finalize.py:116`）

### 5.1 核心思想

ALLTOALL 是**预填充（prefill）阶段的标准做法**：用原生的 `all_to_all`（点对点全交换）把 token 按目标专家所在的卡分发，计算后再 `all_to_all` 收回。

它和 ALLGATHER 的本质区别：**ALLGATHER 全量广播（每卡拿全部 token），ALLTOALL 只发"该卡专家需要的"那部分 token**——通信量按实际路由需求走，更省带宽，但要求 token 数量足够多以摊薄启动开销。

### 5.2 处理流程

#### PrepareAndFinalize（`prepare_finalize.py:116-220`）

- `prepare()`：先 `pad`（填充对齐到固定块，便于 all_to_all 对齐），再按 TP rank 切分（`prepare_finalize.py` 中填充逻辑）。
- `finalize()`：计算后 `all_gather`（聚齐各 TP rank 结果）+ `unpad`（去填充还原）。

> 注意：这里的 `all_to_all` / `all_gather` 是 **TP 维度**的聚合（见维度 A），把单专家张量并行的结果合并。

#### Dispatcher（`token_dispatcher.py:441-540`）

- `token_dispatch()`：调用 `async_all_to_all`（`token_dispatcher.py:506`），**在 EP 维度跨卡交换 token**——把本卡 token 按"目标专家所在的卡"打包发送，同时接收其它卡发来、目标专家在本卡的 token。
- `token_combine()`：再次 `async_all_to_all` 把算完的结果按原路径送回。

### 5.3 处理流程图

```
[每卡本地 token]
   │
   ├─ prepare: pad + TP 切分
   │
   ├─ token_dispatch: async_all_to_all (EP 维度，按目标专家跨卡分发)
   │        ↓
   │   [本卡收到的、属于本地专家的 token]
   │        ↓
   ├─ _apply_mlp: 本地专家 FFN 计算
   │        ↓
   ├─ token_combine: async_all_to_all (EP 维度，原路收回)
   │        ↓
   └─ finalize: all_gather (TP 合并) + unpad
```

### 5.4 触发条件

ALLTOALL 是**预填充阶段的默认选择**（`ascend_forward_context.py` 决策树）：

- Prefill 阶段且 `num_tokens > 512/rank`（token 数足够多，all_to_all 才划算）；
- A3 芯片 prefill 默认走 ALLTOALL。

### 5.5 特点

- ✅ 通信量比 ALLGATHER 小得多（只发需要的 token）。
- ✅ 适合 token 数大、批处理大的 prefill 场景。
- ❌ 通信与计算**串行**（dispatch 完才能算，算完才能 combine），缺乏重叠。
- ❌ decode 阶段 token 太少（batch×1），all_to_all 启动开销大、不划算，故 decode 一般走 MC2。

---

## 6. MC2 详解

> 实现类：`MC2CommImpl`（`moe_comm_method.py:215`）
> 策略组合：`TokenDispatcherWithMC2`（`token_dispatcher.py:101`） + `PrepareAndFinalizeWithMC2`（`prepare_finalize.py:228`）

### 6.1 核心思想

MC2（MindCluster Communication / 大规模集合通信）是**华为自研的高性能 MoE 通信库**。它的最大卖点是：**把 token 的跨卡分发（dispatch）与专家计算（FFN）在时间上重叠起来**——一边收 token 一边算，通信开销被计算"藏"起来，从而显著降低 decode 阶段的通信等待。

与 ALLTOALL 相比，MC2 在 decode（token 少）场景下通过"通信-计算重叠"获得更低延迟。

### 6.2 处理流程

#### PrepareAndFinalize（`prepare_finalize.py:228-320`）

`PrepareAndFinalizeWithMC2` 继承自 `PrepareAndFinalizeWithAll2All`，但**重写了 `prepare()`**：

- 不再用 `pad`/TP 切分，而是用 **`mc2_mask`** + **`padded_num_tokens`**（`prepare_finalize.py:228` 重写逻辑）。
- `mc2_mask`：标识每个 token 是否参与 MC2 跨卡分发；
- `padded_num_tokens`：对齐到 MC2 库要求的块大小的 token 数。

#### Dispatcher（`token_dispatcher.py:101-340`）—— MC2 核心

- `token_dispatch()`：调用 **`npu_moe_distribute_dispatch_v2`**（`token_dispatcher.py:239`）。
  - 这是华为 MC2 库的算子，**在 EP 维度跨卡分发 token**，同时携带 `mc2_mask` 控制参与度。
  - 与普通 all_to_all 不同，它内部**与下游计算重叠**（dispatch 流水化）。
- `token_combine()`：调用 `npu_moe_distribute_combine`（对称地跨卡收回 + 加权合并）。

### 6.3 处理流程图

```
[每卡本地 token]
   │
   ├─ prepare: 生成 mc2_mask + padded_num_tokens（替换 pad）
   │
   ├─ token_dispatch: npu_moe_distribute_dispatch_v2 (MC2 库，跨卡分发 + 通信/计算重叠)
   │        ↓ ╳ 与 _apply_mlp 在时间上重叠
   ├─ _apply_mlp: 本地专家 FFN 计算
   │        ↓
   ├─ token_combine: npu_moe_distribute_combine (MC2 库，跨卡收回 + 加权合并)
   │        ↓
   └─ finalize
```

### 6.4 触发条件

MC2 主要用于 **decode 阶段**（`ascend_forward_context.py` 决策树）：

- A2 芯片：`num_experts_per_device ≤ 24` 且 `ep_size ≥ 16`；
- A3 芯片 decode 默认（当 token 数 ≤ `mc2_tokens_capacity`，见第 9 节）。

### 6.5 特点

- ✅ **通信与计算重叠**，decode 延迟低。
- ✅ 专为 decode（batch 小、token 少）优化。
- ❌ 依赖华为 MC2 库（310P 不支持）。
- ❌ token 数过大时（prefill）优势不明显，反而被 ALLTOALL 取代。

---

## 7. FUSED_MC2 详解

> 实现类：`FusedMC2CommImpl`（`moe_comm_method.py:259`）
> 策略组合：**复用** `TokenDispatcherWithMC2` + `PrepareAndFinalizeWithMC2`（与 MC2 完全相同），但**重写了 `fused_experts`**（`moe_comm_method.py:285`）

### 7.1 核心思想

FUSED_MC2 是 MC2 的**极致融合版本**。通信策略（dispatcher / prepare_finalize）和 MC2 一模一样，但把"dispatch → MLP 计算 → combine"三步**融合成一个算子**，进一步消除中间结果的落盘与同步开销。

它通过 `VLLM_ASCEND_ENABLE_FUSED_MC2` 环境变量控制融合级别（`envs.py:95-102`）：

| 取值 | 融合算子 | 说明 |
|------|---------|------|
| `0`（默认） | 不启用 FUSED_MC2 | 退回到普通 MC2 |
| `1` | `dispatch_ffn_combine`（`moe_comm_method.py:285-309`） | 把 dispatch + FFN + combine 三步融合成一个算子 |
| `2` | `dispatch_gmm_combine_decode`（W8A8 量化算子） | 仅 decode、投机采样场景的分组矩阵乘融合，更具侵略性 |

### 7.2 处理流程

FUSED_MC2 重写了 `fused_experts`（`moe_comm_method.py:285`），**不再走通用的"dispatch → _apply_mlp → combine"三步骨架**，而是直接调用融合算子：

- `enable_fused_mc2 == 1`：调用 `dispatch_ffn_combine`，一次完成跨卡分发 + FFN + 收回合并；
- `enable_fused_mc2 == 2`：调用 `dispatch_gmm_combine_decode`（见 `w8a8_dynamic.py:309` 附近的融合分组矩阵乘算子）。

```
[每卡本地 token]
   │
   ├─ prepare: mc2_mask + padded_num_tokens（同 MC2）
   │
   ├─ fused_experts: ★直接调用融合算子（dispatch_ffn_combine / dispatch_gmm_combine_decode）
   │     内部一次完成: 跨卡分发 + FFN计算 + 收回合并
   │
   └─ finalize
```

### 7.3 触发条件

FUSED_MC2 是**限制最多、性能最优**的方式（`ascend_forward_context.py:287-309` A3 分支）：

**`enable_fused_mc2 == 1`（dispatch_ffn_combine）需同时满足：**
- A3 芯片；
- 量化方式为 W8A8；
- `ep_size ≤ 32`；
- 非 MTP（Multi-Token Prediction）阶段；
- 未开启动态 EPLB。

**`enable_fused_mc2 == 2`（dispatch_gmm_combine_decode）需同时满足：**
- A3 芯片；
- **仅 decode 阶段**（投机采样场景）；
- W8A8 量化；
- 其它约束同上。

### 7.4 特点

- ✅ **性能最优**：融合算子消除中间落盘与同步，端到端延迟最低。
- ✅ 适合对延迟敏感的大规模生产部署。
- ❌ **限制最多**：仅 A3 + W8A8 + EP≤32 + 非MTP + 非动态EPLB，且=2 还要 decode。
- ❌ 不满足任一约束即自动回退到普通 MC2。

---

## 8. 触发条件：select_moe_comm_method 决策树

> 源码：`ascend_forward_context.py:264-320` 的 `select_moe_comm_method()`，枚举定义 `ascend_forward_context.py:28-31`：
> `ALLGATHER=0, MC2=1, ALLTOALL=2, FUSED_MC2=3`（`MoECommType`）

这是**判断当前一次 MoE 该用哪种通信方式**的总入口。决策按芯片代际分支：

### 8.1 决策伪代码

```
select_moe_comm_method():
    # —— 通用前置判断 ——
    if ep_size == 1 或 芯片是 310P:
        return ALLGATHER                    # ① 保底：无 EP 或 310P

    # —— A3 芯片分支（910B3 / 910C3 等新一代）——
    if is_A3:
        if is_decode and num_tokens <= mc2_tokens_capacity:
            if enable_fused_mc2 in (1, 2) 且满足融合约束:
                return FUSED_MC2            # ② decode + 融合
            else:
                return MC2                  # ③ decode，普通 MC2
        else:  # prefill 或 token 超过阈值
            return ALLTOALL                 # ④ prefill 默认 alltoall

    # —— A2 芯片分支（910B 等）——
    if is_A2:
        if num_experts_per_device <= 24 and ep_size >= 16:
            return MC2                      # ⑤ A2 满足条件走 MC2
        else:
            return ALLTOALL                 # ⑥ 否则 alltoall

    return ALLGATHER                        # ⑦ 最终保底
```

> 完整判断逻辑见 `ascend_forward_context.py:274-309`，其中 A3 的 FUSED_MC2 分支在 `L287-309`，依赖 `fused_mc2_enable`（0/1/2）+ decode/prefill 判定。

### 8.2 决策维度速查

| 维度 | 影响 |
|------|------|
| 芯片代际 | 310P → ALLGATHER；A2 → MC2/ALLTOALL；A3 → 全四种皆可 |
| EP 是否开启 | ep=1 → ALLGATHER |
| 阶段（decode/prefill） | decode → MC2 系；prefill → ALLTOALL |
| token 数 vs mc2_tokens_capacity | 超阈值则 prefill 走 ALLTOALL |
| `enable_fused_mc2` + 约束 | 满足才升级到 FUSED_MC2 |
| 专家分布密度 | A2 的 MC2 需 `专家/卡≤24 & EP≥16` |

---

## 9. mc2_tokens_capacity 是怎么算的

> 源码：`ascend_forward_context.py:198-218`（`set_mc2_tokens_capacity`）

`mc2_tokens_capacity` 是 **decode 与 prefill 的分界阈值**：当单次 MoE 的 `num_tokens ≤ mc2_tokens_capacity` 时判定为 decode 场景，走 MC2/FUSED_MC2；超过则判为 prefill，走 ALLTOALL。

- **不是常量，是计算出来的**，典型值约 **512**（与 EP size、专家数、芯片规格相关）。
- 通过 `set_mc2_tokens_capacity()`（`ascend_forward_context.py:202-214`）在初始化时设置到全局上下文。
- 在 `select_moe_comm_method` 中通过 `get_mc2_tokens_capacity()` 读取并参与比较（`ascend_forward_context.py:266` 附近）。

---

## 10. 多维对比表

| 维度 | ALLGATHER | ALLTOALL | MC2 | FUSED_MC2 |
|------|-----------|----------|-----|-----------|
| **EP 跨卡路由 token** | 否（全广播） | 是（all_to_all） | 是（MC2 库 dispatch） | 是（融合在算子内） |
| **Dispatcher 实现** | `npu_moe_init_routing`（本地重排） | `async_all_to_all`（EP 交换） | `npu_moe_distribute_dispatch_v2`（MC2 库） | 复用 MC2，但融合进算子 |
| **Combine 实现** | `npu_moe_token_unpermute`（本地还原） | `async_all_to_all`（EP 收回） | `npu_moe_distribute_combine`（MC2 库） | 复用 MC2，融合进算子 |
| **PrepareAndFinalize** | `PrepareAndFinalizeWithAllGather` | `PrepareAndFinalizeWithAll2All` | `PrepareAndFinalizeWithMC2`（带 mc2_mask） | 同 MC2 |
| **TP/DP 通信** | AG + RS（全量聚散） | pad + all_gather | mc2_mask + padded_num_tokens | 同 MC2 |
| **通信-计算重叠** | 无 | 无（串行） | **有**（MC2 核心优势） | **有**（且融合算子更彻底） |
| **MLP 计算组织** | 三步骨架 | 三步骨架 | 三步骨架 | **融合算子（dispatch_ffn_combine / dispatch_gmm_combine_decode）** |
| **通信量** | 最大（全广播） | 中（按需） | 中（按需） | 中（按需） |
| **典型阶段** | 兜底 / EP=1 | prefill | decode（A2/A3） | decode（A3 融合） |
| **芯片支持** | 全（含 310P） | A2/A3 | A2/A3（非 310P） | 仅 A3 |
| **约束严格度** | 最低 | 低 | 中 | 最高 |
| **性能** | 最低 | 中（prefill 好） | 高（decode 好） | 最高 |

### 10.1 通信发生位置对比

```
ALLGATHER:  DP AllGather ──→ [本地MoE] ──→ DP ReduceScatter   （全量广播）
ALLTOALL :  EP all_to_all ──→ [本地专家] ──→ EP all_to_all     （按需交换，串行）
MC2      :  EP dispatch ╳──→ [本地专家] ←╳ EP combine          （按需交换，重叠）
FUSED_MC2:  [融合算子：dispatch+FFN+combine 一次完成]          （按需，融合+重叠）
```

---

## 11. 本质区别与演进关系

### 11.1 三个本质维度

1. **"token 怎么到目标专家所在的卡"**
   - ALLGATHER：每卡都拿全量，**不路由**（靠广播规避）。
   - ALLTOALL：`all_to_all` 显式点对点交换。
   - MC2 / FUSED_MC2：MC2 库的 `npu_moe_distribute_dispatch_v2`（优化过的跨卡分发）。

2. **"通信和计算是否重叠"**
   - ALLGATHER / ALLTOALL：**串行**（通信完了才算）。
   - MC2 / FUSED_MC2：**重叠**（通信与 FFN 流水化，通信开销被算力藏起来）。

3. **"三步是否融合成一个算子"**
   - 前三种：dispatch → MLP → combine 三步独立。
   - FUSED_MC2：融为单算子（`dispatch_ffn_combine` / `dispatch_gmm_combine_decode`）。

### 11.2 演进关系（从朴素到极致）

```
ALLGATHER（最朴素，全广播，全场景兜底）
   │  引入 EP 显式路由
   ▼
ALLTOALL（按需点对点交换，省带宽，但串行）
   │  通信-计算重叠 + 华为 MC2 库
   ▼
MC2（decode 优化，通信隐藏，低延迟）
   │  dispatch+FFN+combine 融合成单算子
   ▼
FUSED_MC2（极致融合，性能最优，约束最严）
```

> 一句话概括：**ALLGATHER 是兜底，ALLTOALL 是 prefill 主力，MC2 是 decode 主力，FUSED_MC2 是 MC2 的融合加速版（A3 限定）。**

---

## 12. 选型建议

| 场景 | 推荐方式 | 理由 |
|------|---------|------|
| 310P 芯片 / EP 未开 | ALLGATHER | 唯一可用 / 无需 EP |
| A2 芯片大规模 EP（专家/卡≤24, EP≥16）decode | MC2 | 通信-计算重叠，decode 低延迟 |
| A2 / A3 芯片 prefill（token 多） | ALLTOALL | 大 batch 下 all_to_all 摊薄开销 |
| A3 芯片 decode + W8A8 + EP≤32 + 非MTP/非动态EPLB | **FUSED_MC2** | 融合算子，端到端最优 |
| 不满足 FUSED_MC2 约束的 A3 decode | MC2 | 自动回退 |
| 兜底 / 兜底兜底 | ALLGATHER | 决策树最终回退 |

> 实际部署中，**框架通过 `select_moe_comm_method` 自动决策**，一般无需手动指定；`VLLM_ASCEND_ENABLE_FUSED_MC2`（`envs.py:95-102`）是唯一需要人工调优的开关。

---

## 13. 代码索引

| 关注点 | 文件 | 位置 |
|--------|------|------|
| 四种方式注册入口 | `vllm_ascend/ops/fused_moe/moe_comm_method.py` | `setup_moe_comm_method` L55 |
| 基类 `MoECommMethod` | 同上 | L87 |
| 通用 `fused_experts` 骨架 | 同上 | L122-171 |
| `AllGatherCommImpl` | 同上 | L185 |
| `MC2CommImpl` | 同上 | L215 |
| `AlltoAllCommImpl` | 同上 | L235 |
| `FusedMC2CommImpl`（重写 `fused_experts`） | 同上 | L259, L285 |
| `PrepareAndFinalizeWithAll2All` | `vllm_ascend/ops/fused_moe/prepare_finalize.py` | L116 |
| `PrepareAndFinalizeWithMC2` | 同上 | L228 |
| `PrepareAndFinalizeWithAllGather` | 同上 | L321 |
| `TokenDispatcherWithMC2`（`npu_moe_distribute_dispatch_v2`） | `vllm_ascend/ops/fused_moe/token_dispatcher.py` | L101, L239 |
| `TokenDispatcherWithAllGather`（`npu_moe_init_routing`） | 同上 | L350, L402 |
| `TokenDispatcherWithAll2AllV`（`async_all_to_all`） | 同上 | L441, L506 |
| `select_moe_comm_method` 决策树 | `vllm_ascend/ascend_forward_context.py` | L264-320 |
| `MoECommType` 枚举 | 同上 | L28-31 |
| A3 FUSED_MC2 分支 | 同上 | L287-309 |
| `set_mc2_tokens_capacity` / 阈值计算 | 同上 | L198-218 |
| `VLLM_ASCEND_ENABLE_FUSED_MC2` 环境变量 | `vllm_ascend/envs.py` | L95-102 |
| `AscendMoERunner.forward_impl`（8 阶段） | `vllm_ascend/ops/fused_moe/fused_moe.py` | L609-793 |
| `AscendFusedMoE.forward_impl`（计算本体） | 同上 | L334, L296 |
| W8A8 融合算子 `dispatch_gmm_combine_decode` | `vllm_ascend/ops/fused_moe/w8a8_dynamic.py` | L309 附近 |
| 模型侧入口 `DeepseekV4MoE.forward` | `vllm_ascend/models/deepseek_v4.py` | L453, L479, L482 |
| PluggableLayer 注册（FusedMoE→AscendFusedMoE） | `vllm_ascend/utils.py` | L702, L751 |

---

## 14. 术语表

| 术语 | 含义 |
|------|------|
| **EP**（Expert Parallel） | 专家并行，把不同专家分布到多卡 |
| **TP**（Tensor Parallel） | 张量并行，单专家权重在多卡切分 |
| **DP**（Data Parallel） | 数据并行，每卡独立处理一批 token |
| **SP**（Sequence Parallel） | 序列并行，按 token 维度切分 |
| **MC2** | 华为高性能 MoE 通信库，支持通信-计算重叠 |
| **EPLB**（Expert-Level Load Balance） | 专家级动态负载均衡 |
| **MTP**（Multi-Token Prediction） | 多 token 预测，DeepSeek 的投机采样机制 |
| **AllGather / AllReduce / ReduceScatter** | 标准集合通信原语 |
| **all_to_all** | 点对点全交换，每卡向其它每卡发不同数据 |
| **mc2_mask** | MC2 库的掩码，标识 token 是否参与跨卡分发 |
| **mc2_tokens_capacity** | decode/prefill 分界阈值（典型 ~512），见第 9 节 |
| **dispatch_ffn_combine** | FUSED_MC2 的融合算子（dispatch+FFN+combine 三合一） |
| **dispatch_gmm_combine_decode** | FUSED_MC2=2 的 decode 分组矩阵乘融合算子 |
| **PluggableLayer** | vLLM 后端替换机制，Ascend 用它把 FusedMoE 换成 AscendFusedMoE |
| **npu_moe_distribute_dispatch_v2 / _combine** | MC2 库的跨卡分发/收回算子 |
| **npu_moe_init_routing / token_unpermute** | 本地路由重排/还原算子（ALLGATHER 用） |

---

*本文档基于 `vllm-ascend` 源码整理，源码路径：`m:\code\llm-learning\vllm\vllm-ascend`。所有行号均对应源码当前版本，如源码更新请以实际为准。*
