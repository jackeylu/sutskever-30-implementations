# Paper 18: Relational Recurrent Neural Networks

**论文标题**: Relational Recurrent Neural Networks
**作者**: Adam Santoro, Matthew Botvinick, Daan Wierstra, Yee Whye, David Pfau (DeepMind)
**发表年份**: 2018 (NeurIPS 2018)
**引用次数**: 1,500+ (关系推理在 RNN 中的应用)

---

## 📚 论文背景与核心问题

### 研究动机

**问题**：传统 RNN 难以学习复杂的关系推理

```
传统 LSTM:
  ✓ 擅长序列建模
  ✓ 可以学习长期依赖
  ✗ 不擅长显式的关系推理
  ✗ 难以处理多个实体间的交互

关系推理任务:
  - bAbI 任务 (需要多步推理)
  - 程序执行
  - 图推理
  - 逻辑谜题
```

**核心洞察**：
```
RNN + 注意力 = 关系推理能力

Relational RNN:
  - 保留 LSTM 的序列建模能力
  - 加入关系推理模块（通过注意力）
  - 显式建模记忆槽之间的相互作用
```

### 与其他论文的关系

```
Paper 13 (Transformer): 自注意力机制
  → RNN 可以借鉴！

Paper 16 (Relation Network): 显式关系推理
  → 应用到 RNN 中

Paper 18 (Relational RNN):
  → LSTM + Relation Network = Relational RNN
  → 结合两者的优势
```

---

## 🎯 Relational RNN 核心架构

### 整体架构

```
输入序列: x₁, x₂, ..., x_T

对于每个时间步 t:
  1. LSTM 处理输入 → h_proposal (提议隐藏状态)
  2. 关系记忆更新 → h_relational (关系推理)
  3. 组合两者 → h_new (最终输出)

关键: 关系记忆使用多头注意力推理槽位间的关系
```

### 三层结构

```
┌─────────────────────────────────────────┐
│  Layer 1: LSTM (Proposal)                 │
│  x_t → LSTM → h_proposal                  │
│  (处理当前输入，考虑历史)                 │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│  Layer 2: Relational Memory Core         │
│  记忆槽: M = [m₁, m₂, ..., m_N]            │
│                                          │
│  对所有槽位 (包括 h_proposal):            │
│    attention(m_i, m_j) ∀ i,j             │
│  → 学习槽位间的关系                       │
│  → 更新记忆                               │
│  → 输出 h_relational                       │
└─────────────────────────────────────────┘
                ↓
┌─────────────────────────────────────────┐
│  Layer 3: Combination                    │
│  h_final = tanh(W @ [h_proposal,        │
│                      h_relational])   │
└─────────────────────────────────────────┘
```

---

## 🧮 组件 1: 多头注意力 (Multi-Head Attention)

### 数学公式

```
对于每个头 h:

  Q_h = M @ W_q^h      (Query)
  K_h = M @ W_k^h      (Key)
  V_h = M @ W_v^h      (Value)

  Attention_h(Q, K, V) = softmax(QK^T / √d_k) @ V

拼接所有头:
  Output = [Head₁, Head₂, ..., Head_H] @ W_o

其中:
  M: 记忆槽 + 当前输入 (N+1 个向量)
  H: 头的数量
  d_k: 每个头的维度
```

### 实现要点

**为什么多头？**
```
单头注意力:
  - 只能学习一种关系模式
  - 可能不够表达

多头注意力:
  - 不同头学习不同关系
  - 头 1: 空间关系
  - 头 2: 语义相似性
  - 头 3: 时序依赖
  ...
```

**缩放点积**:
```
scores = Q @ K^T / √d_k

为什么缩放?
  - 防止点积过大
  - 防止 softmax 进入饱和区
  - 稳定梯度
```

### 完整实现

```python
def multi_head_attention(M, W_q, W_k, W_v, W_o, num_heads):
    """
    M: (N, d_model) - 记忆槽 + 输入
    W_q, W_k, W_v: 每个头的投影权重
    W_o: 输出投影
    num_heads: 头数量
    """
    N, d_model = M.shape
    d_k = d_model // num_heads

    heads = []
    for h in range(num_heads):
        # 投影
        Q = M @ W_q[h]  # (N, d_k)
        K = M @ W_k[h]  # (N, d_k)
        V = M @ W_v[h]  # (N, d_k)

        # 缩放点积注意力
        scores = Q @ K.T / np.sqrt(d_k)  # (N, N)
        attn_weights = softmax(scores, axis=-1)
        head_out = attn_weights @ V  # (N, d_k)

        heads.append(head_out)

    # 拼接所有头
    concatenated = np.concatenate(heads, axis=-1)  # (N, d_model)

    # 输出投影
    output = concatenated @ W_o  # (N, d_model)

    return output
```

---

## 🧠 组件 2: 关系记忆核心 (Relational Memory Core)

### 核心思想

```
传统 RNN 隐藏状态:
  h_t - 单个向量，难以存储多个实体

Relational Memory:
  M = [m₁, m₂, ..., m_N] - 多个"槽位"
  每个槽位可以存储不同信息

关键: 槽位通过注意力交互！
```

### 记忆更新过程

```python
class RelationalMemory:
    def __init__(self, mem_slots, d_model, num_heads):
        self.mem_slots = mem_slots
        self.d_model = d_model
        self.num_heads = num_heads

        # 注意力权重（每头独立）
        self.W_q = [np.random.randn(d_model, d_model // num_heads) * 0.1
                    for _ in range(num_heads)]
        self.W_k = [np.random.randn(d_model, d_model // num_heads) * 0.1
                    for _ in range(num_heads)]
        self.W_v = [np.random.randn(d_model, d_model // num_heads) * 0.1
                    for _ in range(num_heads)]
        self.W_o = np.random.randn(d_model, d_model) * 0.1

        # MLP 处理注意力输出
        self.W_mlp1 = np.random.randn(d_model, d_model * 2) * 0.1
        self.W_mlp2 = np.random.randn(d_model * 2, d_model) * 0.1

        # LSTM 风格的门（每个槽位独立）
        self.W_gate_i = np.random.randn(d_model, d_model) * 0.1
        self.W_gate_f = np.random.randn(d_model, d_model) * 0.1
        self.W_gate_o = np.random.randn(d_model, d_model) * 0.1

        # 初始化记忆
        self.memory = np.random.randn(mem_slots, d_model) * 0.01

    def step(self, input_vec):
        """更新记忆"""
        # 1. 将输入添加到记忆
        M_tilde = np.concatenate([self.memory, input_vec[None]], axis=0)

        # 2. 多头自注意力
        attended = multi_head_attention(
            M_tilde, self.W_q, self.W_k, self.W_v, self.W_o, self.num_heads
        )

        # 3. 残差连接
        gated = attended + M_tilde

        # 4. 逐行 MLP
        hidden = relu(gated @ self.W_mlp1)
        mlp_out = hidden @ self.W_mlp2

        # 5. 记忆门控（LSTM 风格）
        new_memory = []
        for i in range(self.mem_slots):
            m = mlp_out[i]

            # 计算门
            i_gate = sigmoid(m @ self.W_gate_i)  # 输入门
            f_gate = sigmoid(m @ self.W_gate_f)  # 遗忘门
            o_gate = sigmoid(m @ self.W_gate_o)  # 输出门

            # 候选记忆
            candidate = tanh(m)

            # 更新槽位
            new_slot = f_gate * self.memory[i] + i_gate * candidate
            new_memory.append(o_gate * tanh(new_slot))

        self.memory = np.array(new_memory)

        # 输出是最后一行（对应输入）
        return mlp_out[-1]
```

### 记忆门控的作用

```
LSTM 风格门控:

输入门 i_gate:
  控制多少新信息进入记忆
  → 选择性更新

遗忘门 f_gate:
  控制保留多少旧记忆
  → 长期记忆能力

输出门 o_gate:
  控制输出多少记忆内容
  → 选择性读取

优势:
  - 防止记忆被过度覆盖
  - 可以长期保存信息
  - 类似 LSTM 的门控机制，但用于记忆槽
```

---

## 🔄 组件 3: 完整 Relational RNN Cell

### 架构组合

```python
class RelationalRNNCell:
    """
    完整的关系 RNN 单元

    结合:
    1. LSTM (处理输入)
    2. Relational Memory (关系推理)
    3. 组合层 (融合信息)
    """

    def __init__(self, input_size, hidden_size, mem_slots, num_heads):
        # LSTM 组件
        self.lstm = LSTMCell(input_size, hidden_size)

        # 关系记忆
        self.rm = RelationalMemory(mem_slots, hidden_size, num_heads)

        # 组合层
        self.W_combine = np.random.randn(2 * hidden_size, hidden_size) * 0.1
        self.b_combine = np.zeros(hidden_size)

    def forward(self, x):
        # 1. LSTM 提议
        concat = np.concatenate([x, self.h])
        gates = concat @ self.lstm.W + self.lstm.b

        i, f, o, g = np.split(gates, 4)
        i = sigmoid(i)
        f = sigmoid(f)
        o = sigmoid(o)
        g = tanh(g)

        self.c = f * self.c + i * g
        h_proposal = o * tanh(self.c)

        # 2. 关系记忆推理
        h_relational = self.rm.step(h_proposal)

        # 3. 组合
        combined = np.concatenate([h_proposal, h_relational])
        self.h = tanh(combined @ self.W_combine + self.b_combine)

        return self.h
```

### 信息流动

```
输入 x_t
  ↓
LSTM: h_{t-1}, x_t → h_proposal
  ↓                    ↓
Relational Memory:    M
  - [M, h_proposal]     [所有槽位]
  - Attention(M, M)
  - Update M → h_relational
  ↓
Combine: [h_proposal, h_relational] → h_t

输出: h_t
```

---

## 🎮 应用：序列排序任务

### 任务描述

```
输入: [5, 2, 8, 1, 9, 3]
输出: [1, 2, 3, 5, 8, 9]

挑战:
  1. 需要记住所有数字
  2. 需要推理相对大小关系
  3. 需要按正确顺序输出
```

### 为什么这个任务难？

```
传统 LSTM:
  - 单个隐藏向量
  - 难以同时维护多个数字
  - 容易混淆/遗忘

Relational RNN:
  - 多个记忆槽位
  - 每个槽位可以存储一个数字
  - 注意力可以比较槽位间的大小关系
  - 更容易推理排序
```

### 任务生成

```python
def generate_sorting_task(seq_len=10, max_digit=20, batch_size=64):
    """
    生成排序任务

    输入: 随机整数序列
    输出: 排序后的序列
    """
    # 生成随机序列
    x = np.random.randint(0, max_digit, size=(batch_size, seq_len))

    # 排序
    y = np.sort(x, axis=1)

    # One-hot 编码
    X = np.eye(max_digit)[x]
    Y = np.eye(max_digit)[y]

    return X, Y
```

---

## 💡 核心洞察

### 1. 记忆作为计算图

```
传统 RNN:
  h_t = RNN(x_t, h_{t-1})

Relational RNN:
  M_t = Attention(M_{t-1}, m_t)

关键: 记忆槽形成计算图
  - 每个槽位是一个节点
  - 注意力是边
  - 多步推理 → 多跳计算
```

### 2. 归纳偏置 (Inductive Bias)

```
不同架构的归纳偏置:

CNN:
  - 局部连接
  - 平移不变性

RNN:
  - 时序连接
  - 时序不变性

Relational RNN:
  - 关系偏置
  - 排列不变性
  - 多实体推理
```

### 3. 注意力 vs 记忆

```
注意力 (Transformer):
  - 静态: 注意位置固定
  - 无状态: 每次独立计算

关系记忆 (Relational RNN):
  - 动态: 记忆随时间演化
  - 有状态: 记忆跨时间步累积
  - RNN + 注意力的优势
```

### 4. 与其他模型的对比

| 模型 | 序列建模 | 关系推理 | 记忆机制 |
|------|---------|---------|---------|
| **LSTM** | ✓ | ✗ | 隐藏状态 |
| **Transformer** | ✗ | ✓ | 无状态 |
| **Neural Turing Machine** | ✓ | ✓ | 外部记忆 |
| **Relational RNN** | ✓ | ✓ | 内部记忆槽 |

---

## 📊 性能分析

### bAbI 任务结果

```
任务类型          | LSTM | Relational RNN
------------------|------|-----------------
单步推理          | 95%  | 98%
两步推理          | 82%  | 94%
三步推理          | 61%  | 89%
复杂推理          | 33%  | 76%

关键: 多步推理任务上 Relational RNN 显著优于 LSTM
```

### 排序任务结果

```
序列长度 | LSTM 错误率 | Relational RNN 错误率
---------|-----------|----------------------
5       | 12.3%     | 3.2%
10      | 28.7%     | 8.5%
15      | 45.1%     | 14.3%
20      | 62.4%     | 21.7%

观察: 序列越长，Relational RNN 优势越大
```

### 为什么性能更好？

```
1. 显式的关系建模:
   - 注意力直接计算槽位对的关系
   - 不需要从梯度中"学习"关系

2. 多个记忆槽位:
   - 可以分别存储不同的实体
   - 避免信息混淆

3. 注意力的并行计算:
   - 所有槽位对同时交互
   - 高效的多步推理

4. 门控机制:
   - 保护重要记忆不被覆盖
   - 选择性更新和读取
```

---

## 🔗 与其他论文的关联

### 前置论文

1. **Paper 13: Transformer**
   - 多头注意力机制
   - Relational RNN 借鉴此架构

2. **Paper 16: Relation Network**
   - 显式关系推理
   - g_θ 和 f_φ 函数
   - Relational RNN 将此应用到序列

3. **Paper 03: LSTM**
   - 门控机制
   - Relational Memory 使用 LSTM 风格门控

### 后续影响

1. **Transformer-XL (2019)**
   - 结合 Transformer 和循环记忆
   - 类似思路

2. **Compressive Transformer (2020)**
   - 压缩记忆
   - 关系记忆的现代版本

3. **Set Transformer (2020)**
   - 集合上的注意力
   - 类似的排列不变性

---

## 🛠️ 实践指南

### PyTorch 实现

```python
import torch
import torch.nn as nn

class RelationalMemoryCore(nn.Module):
    def __init__(self, mem_slots, d_model, num_heads=4):
        super().__init__()
        self.mem_slots = mem_slots
        self.d_model = d_model
        self.num_heads = num_heads

        # Multi-head attention
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first=True)

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model)
        )

        # Gating
        self.input_gate = nn.Linear(d_model, d_model)
        self.forget_gate = nn.Linear(d_model, d_model)
        self.output_gate = nn.Linear(d_model, d_model)

    def forward(self, input_vec, memory):
        """
        input_vec: (B, d_model)
        memory: (B, N, d_model)
        """
        B = input_vec.size(0)

        # Augment memory with input
        memory_augmented = torch.cat([memory, input_vec.unsqueeze(1)], dim=1)

        # Self-attention
        attended, _ = self.mha(memory_augmented, memory_augmented, memory_augmented)

        # Residual + MLP
        mlp_out = self.mlp(attended + memory_augmented)

        # Extract memory portion
        mem_updates = mlp_out[:, :-1, :]  # (B, N, d_model)

        # Gating
        i_gate = torch.sigmoid(self.input_gate(mem_updates))
        f_gate = torch.sigmoid(self.forget_gate(mem_updates))
        o_gate = torch.sigmoid(self.output_gate(mem_updates))

        # Update memory
        candidate = torch.tanh(mem_updates)
        new_memory = f_gate * memory + i_gate * candidate
        new_memory = o_gate * torch.tanh(new_memory)

        return new_memory, mlp_out[:, -1, :]
```

### 使用示例

```python
# 初始化
mem_slots = 4
d_model = 128
seq_len = 10
batch_size = 32

rm = RelationalMemoryCore(mem_slots, d_model)

# 初始化记忆
memory = torch.randn(batch_size, mem_slots, d_model)

# 处理序列
for t in range(seq_len):
    x_t = data[:, t, :]  # (B, d_model)
    memory, output = rm.forward(x_t, memory)

# 输出最终结果
final_output = output
```

---

## 🧪 实践挑战

### 基础练习

1. **实现多头注意力**:
   ```python
   def multi_head_attention(M, num_heads):
       # TODO: 实现 Q, K, V 投影
       # TODO: 实现缩放点积
       # TODO: 拼接多头
       pass
   ```

2. **实现关系记忆**:
   ```python
   class RelationalMemory:
       def step(self, input_vec):
           # TODO: 将输入添加到记忆
           # TODO: 应用自注意力
           # TODO: MLP 处理
           # TODO: 门控更新
           pass
   ```

3. **排序任务测试**:
   - 训练 Relational RNN
   - 与 LSTM 对比
   - 分析不同序列长度的表现

### 进阶练习

1. **可视化注意力**:
   ```python
   # 提取注意力权重
   attn_weights = self.mha.attention_weights

   # 绘制热图
   plt.imshow(attn_weights[0].detach().numpy())
   plt.colorbar()
   ```

2. **消融实验**:
   - 移除门控
   - 减少头数
   - 减少记忆槽位

3. **调整记忆槽位数量**:
   ```python
   # 研究最优槽位数量
   for slots in [2, 4, 8, 16]:
       rm = RelationalMemoryCore(slots, d_model)
       # 训练并评估
   ```

### 研究方向

1. **稀疏注意力**:
   - 只注意最重要的槽位
   - 减少计算量

2. **层次化记忆**:
   - 多层关系记忆
   - 不同抽象层级

3. **持续学习**:
   - 动态添加新槽位
   - 终身学习

4. **跨模态应用**:
   - 视觉-语言
   - 语音-文本

---

## ❓ 常见问题

### Q1: Relational RNN vs Transformer?

```
Transformer:
  - 纯注意力，无循环
  - 并行计算整个序列
  - 适合静态数据

Relational RNN:
  - 注意力 + RNN
  - 顺序处理
  - 适合动态/流数据

选择:
  - 序列任务 → Relational RNN
  - 非序列任务 → Transformer
```

### Q2: 记忆槽位数量如何选择？

```
经验法则:

简单任务 (2-4 实体):
  - 4-8 个槽位

中等任务 (5-10 实体):
  - 8-16 个槽位

复杂任务 (>10 实体):
  - 16-32 个槽位

考虑:
  - 计算资源: O(N²) 其中 N = 槽位数
  - 任务复杂度
  - 实体数量
```

### Q3: 为什么需要门控？

```
没有门控的问题:
  - 记忆快速被覆盖
  - 无法长期保存信息
  - 训练不稳定

门控的好处:
  - 选择性写入 (输入门)
  - 选择性保留 (遗忘门)
  - 选择性读取 (输出门)
  - 类似 LSTM 的成功机制
```

### Q4: 与 Neural Turing Machine 的区别？

```
Neural Turing Machine:
  - 外部记忆矩阵
  - 读写头（学习）
  - 难以训练

Relational RNN:
  - 内部记忆槽位
  - 注意力机制（固定结构）
  - 更容易训练

共同点:
  - 都有外部记忆
  - 都需要推理能力

差异:
  - NTM: 通用但难训练
  - Relational RNN: 结构化但易训练
```

---

## 📝 学习检查清单

完成以下任务以确保掌握 Relational RNN：

- [ ] 理解多头注意力的数学推导
- [ ] 实现关系记忆核心
- [ ] 理解 LSTM 门控的作用
- [ ] 在排序任务上测试
- [ ] 与 LSTM baseline 对比
- [ ] 可视化注意力权重
- [ ] 调整超参数（槽位、头数）
- [ ] 阅读 Transformer 论文
- [ ] 探索更多关系推理任务

---

## 🔗 延伸阅读

### 必读论文

1. **Relational Recurrent Neural Networks (NeurIPS 2018)**
   - Santoro et al. (DeepMind)
   - [arXiv:1806.01822](https://arxiv.org/abs/1806.01822)

2. **Attention Is All You Need (2017)**
   - Vaswani et al.
   - 见 Paper 13

3. **A Simple Neural Network Module for Relational Reasoning (2017)**
   - Santoro et al.
   - 见 Paper 16

### 相关资源

- **DeepMind bAbI 项目**:
  - [https://research.fb.com/downloads/babi/](https://research.fb.com/downloads/babi/)

- **神经图灵机**:
  - [https://arxiv.org/abs/1410.5401](https://arxiv.org/abs/1410.5401)

---

## 🎯 核心要点回顾

1. **核心创新**:
   ```
   LSTM + 注意力 = Relational RNN

   LSTM: 序列建模
   注意力: 关系推理
   结合: 两者优势
   ```

2. **关键组件**:
   ```
   - 多头注意力: 并行计算多种关系
   - 关系记忆: 多槽位存储
   - LSTM 门控: 选择性更新
   - 组合层: 融合信息
   ```

3. **性能优势**:
   ```
   多步推理任务显著优于 LSTM
   - bAbI: +15-40%
   - 排序: +10-20%
   ```

4. **设计原则**:
   ```
   - 记忆槽位数量随任务复杂度
   - 多头注意力增加表达能力
   - 门控保护重要记忆
   - 残差连接稳定训练
   ```

5. **应用场景**:
   ```
   - 算法推理 (排序、搜索)
   - 问题回答 (bAbI)
   - 程序执行
   - 图推理
   - 多实体交互
   ```

6. **实践要点**:
   ```
   - 从简单任务开始
   - 逐步增加复杂度
   - 监控记忆使用情况
   - 可视化注意力权重
   - 与 LSTM baseline 对比
   ```

**Relational RNN 展示了如何将注意力机制成功整合到循环网络中，为序列建模提供更强的关系推理能力。**

---

*"We introduce the Relational Core, a memory-augmented neural network that...uses multi-head dot product attention to allow for greater parallelism and improve generalization."*
*— Santoro et al., 2018*
