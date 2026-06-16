# 知识点 02：Linear 层 TP 切分

> 核心问题：Y = X×A + b 怎么切？Column 和 Row 在数学上有什么区别？

---

## 1. 记号约定

| 符号 | 含义 |
|------|------|
| X | 输入，shape `[M, K]`（M 是 token 数，K 是 hidden_dim） |
| A | 权重，shape `[K, N]`（N 是 output_dim） |
| Y | 输出，shape `[M, N]` |
| p | TP world size（卡数） |
| 下标 `i` | 第 i 个 TP rank 的分片 |

---

## 2. ColumnParallelLinear：沿 A 的列切

### 2.1 切法

```
A = [A_1 | A_2 | ... | A_p]   沿 N 维（列）切，每张卡拿 [K, N/p]
```

### 2.2 forward

```
每张卡计算：
    Y_i = X × A_i           shape: [M, N/p]
```

- 每张卡都有完整 X（**输入不切**）
- 每张卡只算输出的 1/p 列

### 2.3 怎么得到完整 Y？

两种选择（vLLM 用 `gather_output` 参数控制）：

**选 A：all-gather（`gather_output=True`）**

```
Y = all_gather([Y_1, Y_2, ..., Y_p], dim=-1)
```

每张卡都得到完整 Y。代价：通信 N×M×dtype_size。

**选 B：保持分片（`gather_output=False`，更常用）**

```
每张卡保留自己的 Y_i（不通信）
```

**用途**：如果下一个算子可以接受分片输入（如 RowParallelLinear 的输入），就不通信。

### 2.4 偏置 b

b 的 shape 是 `[N]`。Column Parallel 下，b 沿 N 维切：

```
b_i = b[i*N/p : (i+1)*N/p]      shape: [N/p]
```

每张卡用自己那段 b_i。

---

## 3. RowParallelLinear：沿 A 的行切

### 3.1 切法

```
       ┌─────┐
       │ A_1 │
       │ A_2 │
A  =   │ ... │     沿 K 维（行）切，每张卡拿 [K/p, N]
       │ A_p │
       └─────┘

同时 X 也按列切：X = [X_1 | X_2 | ... | X_p]    shape: [M, K/p]
```

### 3.2 forward

```
每张卡计算：
    Y_i = X_i × A_i          shape: [M, N]

最终结果（每张卡得到部分和）：
    Y = sum_i Y_i
```

### 3.3 怎么得到完整 Y？

**all-reduce**：

```
Y = all_reduce([Y_1, Y_2, ..., Y_p])  # 求和
```

每张卡都得到完整的 Y。

### 3.4 偏置 b

Row Parallel 下，b 是 `[N]`，**不切**（每张卡都持有完整 b 的副本）。

**关键**：b **只在 rank 0 加**，然后走 all-reduce。原因：

```
正确结果: Y = sum_i (X_i × A_i) + b

如果每 rank 都加 b 再 all-reduce:
   all_reduce(output_i + b) = sum_i (output_i) + p × b   ❌ 加了 p 次

只在 rank 0 加 b 再 all-reduce:
   all_reduce(output_0 + b, output_1, ..., output_{p-1})
   = (output_0 + b) + sum_{i>0}(output_i)
   = sum_i(output_i) + b                                  ✓
```

vLLM 源码（`vllm/model_executor/layers/linear.py:1457`）：

```python
# Only fuse bias add into GEMM for rank 0
bias_ = None if (self.tp_rank > 0 or self.skip_bias_add) else self.bias
output_parallel = self.quant_method.apply(self, input_parallel, bias_)

if self.reduce_results and self.tp_size > 1:
    output = tensor_model_parallel_all_reduce(output_parallel)
```

---

## 4. Column + Row 的经典配对

MLP 块：

```
X: [M, K]
   ↓ ColumnParallelLinear (fc1: K → 4K)
   ↓ GeLU
h: [M, 4K/p]   ← 分片！不通信！
   ↓ RowParallelLinear (fc2: 4K → K)
   ↓ all-reduce
Y: [M, K]      ← 完整
```

**关键洞察**：

- fc1 的 `gather_output=False`，输出保持分片
- fc2 的 `input_is_parallel=True`，输入假设已经是分片
- 中间激活 `h` 是分片的，**省了一次通信**
- 整个 MLP 只有 fc2 后一次 all-reduce

这就是 Megatron 论文的精髓：**配对切分，把通信收敛到 1 次**。

---

## 5. QKV 的情况：MergedColumnParallelLinear

Attention 的 qkv_proj 比普通 MLP 复杂，因为 Q、K、V 是三个不同矩阵。

### 5.1 朴素做法

```
q_proj, k_proj, v_proj: 三个 ColumnParallelLinear
每个独立切，每个 gather_output=False
```

### 5.2 vLLM/Megatron 的合并做法

vLLM 用 `QKVParallelLinear` 把 Q/K/V 合并成一个大权重：

```
[q_proj | k_proj | v_proj]  拼接成一个矩阵

按 Column 切：
  tp_rank i 拿到：[q_proj 的第 i 段 | k_proj 的第 i 段 | v_proj 的第 i 段]
```

**好处**：

- 一个 GEMM 同时算 Q/K/V，kernel 启动开销 1/3
- 显存连续，访存更友好

**复杂点**：GQA / MQA 下，K/V 的 head 数比 Q 少（如 Q=32 heads, K=V=8 heads）。

详见 [03_attention_TP切分.md](03_attention_TP切分.md)。

---

## 6. ReplicatedLinear：不切的层

并非所有 Linear 都需要切。vLLM 有 `ReplicatedLinear`（`linear.py:289`）：

- 权重每张卡都完整副本
- 不需要通信
- 适用场景：head_size 太小切不动、或 layer norm 之前的 final norm 等

---

## 7. 切分的数学不变量

无论怎么切，TP 都满足：

```
完整计算 Y = X × A + b

TP 后：Y = sum_i (X_i × A_i) + b   (Row Parallel)
        或
        Y = concat_i (X × A_i)     (Column Parallel, gather 后)
```

数值结果**完全等价**（除了浮点累加误差）。

---

## 8. TP 切分的实际效果

以 Llama 70B 为例，TP=8（单节点 8×A100 80GB）：

| 层 | 完整权重 | TP=8 单卡权重 |
|----|---------|---------------|
| qkv_proj | [8192, 8192] ≈ 134MB | [8192, 1024] ≈ 17MB |
| o_proj | [8192, 8192] ≈ 134MB | [1024, 8192] ≈ 17MB |
| gate_up_proj | [8192, 22528] ≈ 369MB | [8192, 2816] ≈ 46MB |
| down_proj | [22528, 8192] ≈ 369MB | [2816, 8192] ≈ 46MB |

每层 MLP+Attention 权重 ≈ 1GB → TP=8 单卡 ≈ 126MB，单卡显存压力降到 1/8。

---

## 9. 小结

- ColumnParallelLinear：切输出维度，输出分片，可选 gather
- RowParallelLinear：切输入维度，输出求和，必须 all-reduce
- Megatron 精髓：Column + Row 配对，中间激活分片省通信
- QKVParallelLinear：Q/K/V 合并权重，一次 GEMM 出三个

---

**下一步**：[03_attention_TP切分.md](03_attention_TP切分.md) — Attention 的 head 怎么切，GQA 怎么处理。
