# 知识点 07：Embedding 与 LM Head 的 TP 实现

> 核心问题：模型首尾的 embedding 和 lm_head 在 TP 下怎么切？
> 源码位置：`vllm/model_executor/layers/vocab_parallel_embedding.py`、`vllm/model_executor/layers/logits_processor.py`

---

## 1. 为什么 Embedding 和 LM Head 特殊？

输入和输出阶段都涉及 **词表维度**（vocab_size）：

```
输入：token_id → embedding lookup（按 vocab_size 索引）
输出：hidden → logits（vocab_size 维度的预测）

vocab_size 通常很大（128k+），权重矩阵巨大
   嵌入矩阵：[vocab_size, hidden_dim]  Llama 70B: [128k, 8192] ≈ 4GB
   LM Head： [hidden_dim, vocab_size] 同样 4GB
```

单卡放不下 → 需要 TP 切分。

---

## 2. VocabParallelEmbedding

源码：`vllm/model_executor/layers/vocab_parallel_embedding.py:253`

### 2.1 切分方式

按词表维度（dim=0）切：

```python
class VocabParallelEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, ...):
        self.tp_size = get_tensor_model_parallel_world_size()
        self.num_embeddings_per_partition = divide(num_embeddings, self.tp_size)
        self.shard = ...  # 当不能整除时用于对齐
        self.vocab_start_index, self.vocab_end_index = ...
        self.weight = Parameter(
            torch.empty(self.num_embeddings_per_partition, embedding_dim)
        )
```

例：vocab_size=128k, TP=8：

```
每 rank 维护 vocab[16k × i : 16k × (i+1)] 的 embedding
权重 shape: [16k, hidden_dim]
```

### 2.2 forward 流程

源码：`vllm/model_executor/layers/vocab_parallel_embedding.py:472-492`

```python
def forward(self, input_):
    if self.tp_size > 1:
        # 1. 用 get_masked_input_and_mask 算出属于本 rank 的 token
        #    返回 masked_input（已减去 start_index，越界处置 0）和 input_mask
        masked_input, input_mask = get_masked_input_and_mask(
            input_,
            self.shard_indices.org_vocab_start_index,
            self.shard_indices.org_vocab_end_index,
            ...,
        )
    else:
        masked_input = input_
    
    # 2. embedding lookup（只查本 rank 的权重）
    output_parallel = self.quant_method.embedding(self, masked_input.long())
    
    # 3. 不属于本 rank 的 token 输出置零
    if self.tp_size > 1:
        output_parallel.masked_fill_(input_mask.unsqueeze(-1), 0)
    
    # 4. all-reduce 把所有 rank 的结果合并（其他 rank 已经置零，等价于正确 lookup）
    output = tensor_model_parallel_all_reduce(output_parallel)
    return output
```

实际源码比这更细致（处理 org_vocab 和 added_vocab 两段），但核心思路就是上述四步。

### 2.3 为什么用 all-reduce 而不是 all-gather？

```
输入: [t1, t2, t3, t4]  4 个 token id
TP=2，每 rank 维护一半 vocab

假设 t1, t3 属于 rank 0，t2, t4 属于 rank 1

Rank 0 lookup：
    [emb(t1), 0, emb(t3), 0]    （t2/t4 不在 rank 0，置零）

Rank 1 lookup：
    [0, emb(t2), 0, emb(t4)]    （t1/t3 不在 rank 1，置零）

all-reduce 求和：
    [emb(t1), emb(t2), emb(t3), emb(t4)]  ✓
```

每张卡都得到完整 embedding（dim 不变），所以是 all-reduce。

### 2.4 与 Column/Row Parallel 的关系

Embedding lookup 本质是 **行选择**（从权重矩阵选若干行）：

- 行选择 ≈ Row Parallel（输入维度 = 行索引）
- 不像 ColumnParallel 那样需要切输出

所以 VocabParallelEmbedding 跟 Row Parallel 类似，forward 后 all-reduce。

---

## 3. ParallelLMHead

源码：`vllm/model_executor/layers/logits_processor.py`

### 3.1 用途

模型最后的 logits 计算：

```
hidden: [M, hidden_dim]    ← 来自最后一个 Transformer block
   ↓ lm_head (matmul)
logits: [M, vocab_size]    ← 每 token 对应 vocab_size 个 token 的预测分数
```

### 3.2 TP 切分

`ParallelLMHead` 用 Column Parallel：

```python
class ParallelLMHead(UnquantizedParallelLinear):
    def __init__(self, num_embeddings, embedding_dim, ...):
        # num_embeddings 是 vocab_size
        # 用 ColumnParallelLinear 切 vocab_size
        ...
```

每 rank 算自己那段 vocab 的 logits：

```
Rank 0: logits[:, 0:vocab/p]
Rank 1: logits[:, vocab/p:2*vocab/p]
...
```

### 3.3 LogitsProcessor 的 TP 处理

源码：`vllm/model_executor/layers/logits_processor.py:123`

```python
class LogitsProcessor(nn.Module):
    def __init__(self, ...):
        ...
        self.tp_size = get_tensor_model_parallel_world_size()
    
    def forward(self, lm_head, hidden_states, embedding_bias=None):
        # 用 ParallelLMHead 做投影
        logits = lm_head.apply_linear(... , bias=embedding_bias)
        # logits shape: [M, vocab/p]
        
        # 如果不 sample，需要 gather 完整 logits
        # 如果 sample（采样），可以只在每 rank sample 自己那段，再 all-gather
        ...
```

### 3.4 Logits 处理策略

**策略 A：gather 全部 logits**

```python
logits = tensor_model_parallel_all_gather(logits_parallel, dim=-1)
# logits: [M, vocab_size]
```

简单但通信量大（vocab_size 大）。

**策略 B：每 rank sample 自己那段**

```
Rank 0 算 logits[:, 0:vocab/p] 的 argmax / topk / multinomial
Rank 1 算 logits[:, vocab/p:2*vocab/p] 的 argmax / topk / multinomial

最后 all-gather + 重新计算全局 argmax / topk
```

通信量小（只传 sample 结果），但实现复杂。

vLLM 默认用策略 A（gather）+ 后续处理，因为 vocab_size 现在普遍 128k+ 但通信仍是小头。

---

## 4. 输入 Embedding 与输出 LM Head 的权重共享

很多模型（如 Llama）**输入 embedding 和输出 lm_head 共享权重**：

```
W_embed = [vocab_size, hidden_dim]   ← 输入 embedding
W_lm_head = W_embed.T                ← 输出 lm_head（同一个权重的转置）
```

vLLM 通过 `tie_word_embeddings` 处理：

```python
# 模型加载时，lm_head 和 embedding 指向同一份权重
self.lm_head.weight = self.model.embed_tokens.weight
```

TP 下，两者都按 vocab 维切，每 rank 拿到对应的 vocab 段。

---

## 5. 完整的 TP 模型 forward 流程

```
输入 token_ids: [batch_size, seq_len]
    ↓
VocabParallelEmbedding（按 vocab 切）
    ↓ all-reduce
hidden: [batch_size, seq_len, hidden_dim]
    ↓
N × Transformer Block:
    ↓
    ├── attention: QKVParallelLinear (Column, gather_output=False)
    │   ↓ attention (每 rank 自己 head)
    │   └── o_proj: RowParallelLinear (all-reduce)
    │
    ├── MLP: MergedColumnParallelLinear (gate+up, Column, gather_output=False)
    │   ↓ activation
    │   └── down_proj: RowParallelLinear (all-reduce)
    ↓
hidden_final: [batch_size, seq_len, hidden_dim]
    ↓
ParallelLMHead (Column, 按 vocab 切)
    ↓ all-gather（或 sample-only）
logits: [batch_size, seq_len, vocab_size]
```

**关键观察**：

- 输入端 VocabParallelEmbedding → all-reduce
- 每个 Transformer block：2 次 all-reduce
- 输出端 ParallelLMHead → all-gather（或直接 sample）

整个 forward 的 TP 通信点清晰可数。

---

## 6. 关键代码位置

| 抽象 | 文件 | 行号 |
|------|------|------|
| `VocabParallelEmbedding` | `vllm/model_executor/layers/vocab_parallel_embedding.py` | L253 |
| Embedding forward (mask + all-reduce) | 同上 | forward 方法 |
| `LogitsProcessor` | `vllm/model_executor/layers/logits_processor.py` | L123 |
| `ParallelLMHead` | `vllm/model_executor/layers/logits_processor.py` | - |
| `tp_size` 在 logits 中的处理 | `vllm/model_executor/layers/logits_processor.py` | L123- |

---

## 7. 常见问题

### 7.1 为什么 embedding 不用 ColumnParallel 而是用 all-reduce？

ColumnParallel 要求输入不切（每卡都有完整 input），输出切。但 embedding 的输入是 **token id**（不是张量），lookup 是行选择。

如果用 ColumnParallel 切 hidden_dim（每 rank 只算 hidden/p 的 embedding）：

- 每卡都需要 lookup 完整 vocab → 权重不切，没意义
- 反而需要 all-gather 合并 hidden_dim

所以只能切 vocab_size 维（行选择 → all-reduce）。

### 7.2 lm_head 为什么不切 hidden_dim？

如果 lm_head 切 hidden_dim：

- 输入 hidden 要切（每 rank 算自己那段 hidden）
- 但 hidden 来自上一层的 RMSNorm + all-gather（已经合并）
- 再切就要 reduce-scatter，引入额外通信

切 vocab_size 更合理：

- 每 rank 算自己那段 vocab 的 logits
- 后续 sample 时通信量小

---

## 8. 小结

- VocabParallelEmbedding：按 vocab 切，行选择，forward 后 all-reduce
- ParallelLMHead（lm_head）：按 vocab 切（Column），logits 分片，可 all-gather
- 输入 embedding 和 lm_head 常共享权重（tie_word_embeddings）
- 完整 TP forward 的通信点：embedding 后、每个 block 的 attn/mlp 后、lm_head 后

---

**下一步**：[../L4-进阶场景/08_TP组合并行.md](../L4-进阶场景/08_TP组合并行.md) — TP 跟 PP、EP 怎么组合。
