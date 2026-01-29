# Paper 14: Neural Machine Translation by Jointly Learning to Align and Translate

**论文标题**: Neural Machine Translation by Jointly Learning to Align and Translate
**作者**: Dzmitry Bahdanau, KyungHyun Cho, Yoshua Bengio (Université de Montréal)
**发表年份**: 2014 (arXiv 2014, ICLR 2015)
**引用次数**: 15,000+ (注意力机制的开山之作)

---

## 📚 论文背景与核心问题

### 研究动机

在 Bahdanau Attention 出现之前，神经机器翻译 (NMT) 使用**固定长度的上下文向量**：

```
传统 Seq2Seq 框架:
  Encoder: 读取整个输入序列
           → 压缩成固定长度向量 c
           → 信息瓶颈！

  Decoder: 从 c 生成输出序列
           → c 必须包含所有信息
```

### 核心问题

#### 1. 信息瓶颈 (Information Bottleneck)

```
短句: "I love cats"
  → 编码成向量 c (维度 512)
  → 可能足够

长句: "The quick brown fox jumps over the lazy dog near the river..."
  → 编码成向量 c (维度 512)
  → 信息损失！
  → 翻译质量下降
```

**实验观察**：
```
句子长度 → BLEU 分数
  10     → 28.5
  20     → 22.3
  30     → 15.8  ← 显著下降！
  50     → 9.2   ← 几乎不可用
```

#### 2. 缺乏对齐 (No Alignment)

传统 seq2seq 无法知道输出词对应输入词的哪个位置：

```
输入: "I love cats"
输出: "我 爱 猫"

问题:
  生成 "我" 时，应该关注 "I"     还是 "love" 还是 "cats"？
  生成 "爱" 时，应该关注哪个词？
  生成 "猫" 时，应该关注哪个词？

传统 seq2seq 不知道！
```

#### 3. 长距离依赖

```
输入: "The woman who I met at the party yesterday and who is very smart..."

生成 "is" 时，需要回溯到 "woman"
但固定向量 c 无法保留这种精确的对应关系
```

### Bahdanau 的创新

**核心洞察**：**每个解码步骤应该关注输入序列的不同部分！**

```
传统:  c (固定)  →  y₁, y₂, y₃, ...
Attention: c₁ → y₁,  c₂ → y₂,  c₃ → y₃,  ...
          ↑         ↑         ↑
       关注不同   关注不同   关注不同
       的部分     的部分     的部分
```

**革命性意义**：
- 不再将整个序列压缩成一个固定向量
- 每个时间步都有**动态的上下文向量**
- 自动学习源语言和目标语言的对齐

---

## 🎯 Bahdanau 注意力机制

### 整体架构

```
编码器 (双向 RNN):
  输入: x₁, x₂, ..., x_T
  隐藏状态: h₁, h₂, ..., h_T (每个位置都有表示)

解码器 (RNN + Attention):
  t=1: s₀ → Attention(h₁..h_T) → c₁ → y₁
  t=2: s₁ → Attention(h₁..h_T) → c₂ → y₂
  t=3: s₂ → Attention(h₁..h_T) → c₃ → y₃
  ...
```

### 核心组件

#### 1. 双向编码器

```python
class EncoderRNN:
    def __init__(self, input_size, hidden_size):
        # 前向 RNN
        self.W_fwd = np.random.randn(hidden_size, input_size + hidden_size) * 0.01

        # 后向 RNN
        self.W_bwd = np.random.randn(hidden_size, input_size + hidden_size) * 0.01

    def forward(self, inputs):
        """前向和后向传播，拼接隐藏状态"""
        # 前向 pass: 从左到右
        h_fwd = []
        h = np.zeros((hidden_size, 1))
        for x in inputs:
            concat = np.vstack([x, h])
            h = np.tanh(np.dot(self.W_fwd, concat))
            h_fwd.append(h)

        # 后向 pass: 从右到左
        h_bwd = []
        h = np.zeros((hidden_size, 1))
        for x in reversed(inputs):
            concat = np.vstack([x, h])
            h = np.tanh(np.dot(self.W_bwd, concat))
            h_bwd.append(h)
        h_bwd = list(reversed(h_bwd))

        # 拼接前向和后向
        annotations = [np.vstack([h_f, h_b]) for h_f, h_b in zip(h_fwd, h_bwd)]

        return annotations
```

**为什么用双向？**
```
句子: "The bank of the river"

单向 RNN:
  h₁ ("The")    → 只有上下文 "The"
  h₂ ("bank")   → 有上下文 "The bank"
  h₃ ("of")     → 有上下文 "The bank of"
  ...

双向 RNN:
  h₁ ("The")    → 前向: "The"    + 后向: "整个句子"
  h₂ ("bank")   → 前向: "The bank" + 后向: "of the river"
  h₃ ("of")     → 前向: "The bank of" + 后向: "the river"
  ...

每个位置都有完整的上下文！
```

#### 2. 注意力评分 (Alignment Model)

**核心公式**：

```
e_ij = v_a^T · tanh(W_a · s_{i-1} + U_a · h_j)
```

其中：
- `s_{i-1}`: 解码器上一时刻的隐藏状态
- `h_j`: 编码器第 j 个位置的隐藏状态
- `W_a, U_a, v_a`: 可学习参数
- `e_ij`: 位置 i 的输出与位置 j 的输入的对齐分数

**代码实现**：

```python
class BahdanauAttention:
    def __init__(self, hidden_size, annotation_size):
        # 注意力参数
        self.W_a = np.random.randn(hidden_size, hidden_size) * 0.01
        self.U_a = np.random.randn(hidden_size, annotation_size) * 0.01
        self.v_a = np.random.randn(1, hidden_size) * 0.01

    def compute_alignment(self, decoder_hidden, encoder_annotation):
        """计算对齐分数"""
        # e_ij = v_a^T * tanh(W_a * s_{i-1} + U_a * h_j)
        alignment = np.dot(self.v_a, np.tanh(
            np.dot(self.W_a, decoder_hidden) +
            np.dot(self.U_a, encoder_annotation)
        ))
        return alignment[0, 0]
```

**为什么用 tanh？**
```
tanh 是非线性激活：
  - 将线性组合映射到 [-1, 1]
  - 增加模型表达能力
  - 允许学习复杂的对齐模式
```

#### 3. 注意力权重 (Attention Weights)

对每个解码步骤 i，计算对所有编码器位置 j 的注意力分布：

```
α_ij = exp(e_ij) / Σ_k exp(e_ik)
```

**性质**：
1. 归一化: `Σ_j α_ij = 1`
2. 非负: `α_ij ≥ 0`
3. 可以解释为概率分布

**代码实现**：

```python
def softmax(x, axis=-1):
    """数值稳定的 softmax"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

# 计算所有位置的分数
scores = [self.compute_alignment(decoder_hidden, h_j)
          for h_j in encoder_annotations]

# Softmax 得到权重
attention_weights = softmax(np.array(scores))
```

#### 4. 上下文向量 (Context Vector)

**核心公式**：

```
c_i = Σ_j α_ij · h_j
```

**直观理解**：
- 加权求和所有编码器隐藏状态
- 权重是注意力权重 `α_ij`
- 高权重 → 该位置的贡献大

**代码实现**：

```python
# 计算上下文向量
context = sum(alpha * h for alpha, h in zip(attention_weights, encoder_annotations))
```

**示例**：

```
编码器状态: h₁, h₂, h₃, h₄, h₅

注意力权重: α = [0.1, 0.6, 0.2, 0.05, 0.05]

上下文向量:
  c = 0.1·h₁ + 0.6·h₂ + 0.2·h₃ + 0.05·h₄ + 0.05·h₅

解释: 主要关注 h₂ (权重 0.6)，其次是 h₃ (权重 0.2)
```

---

## 🔄 带注意力的解码器

### 解码器结构

每个时间步的更新：

```
输入: y_{i-1} (上一输出), s_{i-1} (上一隐藏状态), h₁..h_T (编码器状态)

1. 计算注意力:
   c_i = Attention(s_{i-1}, h₁..h_T)

2. 更新隐藏状态:
   s_i = RNN(s_{i-1}, y_{i-1}, c_i)

3. 生成输出:
   y_i = softmax(W_o · [s_i; c_i; y_{i-1}] + b_o)
```

### 完整实现

```python
class AttentionDecoder:
    def __init__(self, output_size, hidden_size, annotation_size):
        # 注意力机制
        self.attention = BahdanauAttention(hidden_size, annotation_size)

        # RNN 输入: [上一输出; 上下文]
        rnn_input_size = output_size + annotation_size
        self.W_dec = np.random.randn(hidden_size, rnn_input_size + hidden_size) * 0.01

        # 输出层
        output_input_size = hidden_size + annotation_size + output_size
        self.W_out = np.random.randn(output_size, output_input_size) * 0.01

    def step(self, prev_output, decoder_hidden, encoder_annotations):
        """单步解码"""
        # 1. 计算注意力和上下文
        context, attention_weights = self.attention.forward(
            decoder_hidden, encoder_annotations
        )

        # 2. RNN 更新隐藏状态
        rnn_input = np.vstack([prev_output, context])
        concat = np.vstack([rnn_input, decoder_hidden])
        new_hidden = np.tanh(np.dot(self.W_dec, concat))

        # 3. 生成输出
        output_input = np.vstack([new_hidden, context, prev_output])
        output = np.dot(self.W_out, output_input)

        return output, new_hidden, attention_weights
```

### 解码过程示例

```
任务: 英文 → 法文

输入: "I love cats"
编码器状态: h₁(I), h₂(love), h₃(cats)

解码过程:
  t=1:
    s₀ → Attention → 关注 h₁ (α₁=0.8)
    c₁ = 0.8·h₁ + 0.1·h₂ + 0.1·h₃
    s₁ = RNN(s₀, <START>, c₁)
    y₁ = "Je"

  t=2:
    s₁ → Attention → 关注 h₂ (α₂=0.7)
    c₂ = 0.1·h₁ + 0.7·h₂ + 0.2·h₃
    s₂ = RNN(s₁, "Je", c₂)
    y₂ = "aime"

  t=3:
    s₂ → Attention → 关注 h₃ (α₃=0.9)
    c₃ = 0.05·h₁ + 0.05·h₂ + 0.9·h₃
    s₃ = RNN(s₂, "aime", c₃)
    y₃ = "les"
  ...
```

---

## 📊 注意力可视化

### 对齐矩阵 (Alignment Matrix)

```
      I    love   cats    ...
Je   0.8   0.1    0.1    ← 生成 "Je" 时主要关注 "I"
aime 0.1   0.7    0.2    ← 生成 "aime" 时主要关注 "love"
les  0.05  0.05   0.9    ← 生成 "les" 时主要关注 "cats"
...
```

**可视化代码**：

```python
import matplotlib.pyplot as plt

# attention_matrix: (output_len, input_len)
plt.figure(figsize=(10, 8))
plt.imshow(attention_matrix, cmap='Blues', aspect='auto')
plt.colorbar(label='Attention Weight')
plt.xlabel('Input Position (Source)')
plt.ylabel('Output Position (Target)')
plt.title('Bahdanau Attention Alignment')
plt.show()
```

### 注意力模式分析

#### 1. 对角模式 (Monotonic Alignment)

```
输入:  A  B  C  D  E
输出:  α  β  γ  δ  ε

对齐:
  α  █  ▒  ▒  ▒  ▒
  β  ▒  █  ▒  ▒  ▒
  γ  ▒  ▒  █  ▒  ▒
  δ  ▒  ▒  ▒  █  ▒
  ε  ▒  ▒  ▒  ▒  █

解释: 词序相同的语言对（如英语-法语）
```

#### 2. 交叉模式 (Reordering)

```
输入:  A  B  C  D  E
输出:  ε  δ  γ  β  α  (倒序)

对齐:
  ε  ▒  ▒  ▒  ▒  █  ← 关注最后
  δ  ▒  ▒  ▒  █  ▒
  γ  ▒  ▒  █  ▒  ▒
  β  ▒  █  ▒  ▒  ▒
  α  █  ▒  ▒  ▒  ▒  ← 关注第一个

解释: 词序不同的语言对（如英语-日语）
```

#### 3. 多对一模式

```
输入:  I  do  not  know
输出:  我   不    知    道

对齐:
  我   █  ▒  ▒  ▒    ← "I"
  不   ▒  █  █  ▒    ← "do not" (合并)
  知   ▒  ▒  ▒  █    ← "know"
  道   ▒  ▒  ▒  █    ← "know" (一个词对应多个词)
```

---

## 🆚 对比：有注意力 vs 无注意力

### 无注意力（固定上下文）

```
编码器:
  h₁, h₂, ..., h_T → 最后只保留 h_T

上下文:
  c = h_T (固定)

解码器:
  y₁ = Decoder(s₀, c)
  y₂ = Decoder(s₁, c)  ← 使用相同的 c
  y₃ = Decoder(s₂, c)  ← 使用相同的 c
  ...

问题:
  - 信息瓶颈: 所有信息压缩到 h_T
  - 长序列遗忘: h₁..h_{T-1} 的信息丢失
```

### 有注意力（动态上下文）

```
编码器:
  h₁, h₂, ..., h_T → 保留所有状态

上下文:
  c₁ = Σ α₁j · h_j  (每个输出步不同)
  c₂ = Σ α₂j · h_j
  c₃ = Σ α₃j · h_j
  ...

优势:
  - 无信息瓶颈: 每步都能访问所有编码器状态
  - 自动对齐: 学习哪些位置重要
  - 长序列友好: 不随序列长度衰减
```

### 性能对比

原论文实验结果（英→法翻译）：

```
模型                  BLEU 分数
────────────────────────────────
传统 Seq2Seq (无注意力):  24.6
+ Bahdanau Attention:     28.5  ← 提升 3.9 点！

长句子 (> 30 词):
  无注意力:              15.8
  有注意力:              22.3  ← 提升 6.5 点！
```

---

## 🔍 注意力评分函数对比

### Bahdanau Attention (Additive)

```
score(s, h) = v^T · tanh(W₁·s + W₂·h)

优点:
  - 表达能力强（有非线性）
  - 适用于不同维度的 s 和 h

缺点:
  - 参数多 (W₁, W₂, v)
  - 计算较慢
```

### Luong Attention (Multiplicative, 2015)

```
score(s, h) = s^T · h

优点:
  - 参数少（只有矩阵乘法）
  - 计算快

缺点:
  - 表达能力较弱（纯线性）
  - 要求 s 和 h 维度相同
```

### Scaled Dot-Product (Transformer, 2017)

```
score(s, h) = s^T · h / √d_k

优点:
  - 缩放防止梯度消失
  - 最快（可以向量化）
  - Transformer 标准

缺点:
  - O(d²) 内存
```

### 代码对比

```python
def bahdanau_score(s, h, W_a, U_a, v_a):
    """加性注意力 (Bahdanau)"""
    return np.dot(v_a.T, np.tanh(np.dot(W_a, s) + np.dot(U_a, h)))[0, 0]

def luong_score(s, h):
    """乘性注意力 (Luong)"""
    return np.dot(s.T, h)[0, 0]

def scaled_dot_product_score(s, h):
    """缩放点积 (Transformer)"""
    d_k = s.shape[0]
    return np.dot(s.T, h)[0, 0] / np.sqrt(d_k)

# 使用示例
score_bahdanau = bahdanau_score(s, h, W_a, U_a, v_a)
score_luong = luong_score(s, h)
score_scaled = scaled_dot_product_score(s, h)
```

---

## 🔗 与其他论文的关联

### 前置论文

1. **Paper 02: Character RNN**
   - 基础的 RNN 结构
   - Bahdanau 在此基础上添加注意力

2. **Paper 08: Seq2Seq for Sets**
   - Encoder-Decoder 框架
   - Bahdanau 为其添加注意力

### 后续影响论文

1. **Luong Attention (2015)**
   - 简化注意力评分（dot product）
   - 提出 global 和 local attention

2. **Show, Attend and Tell (2015)**
   - 注意力用于图像描述生成
   - 首次将注意力应用到视觉

3. **Attention Is All You Need (2017, Paper 13)**
   - 完全基于注意力的 Transformer
   - 自注意力（self-attention）
   - 缩放点积成为标准

4. **BERT, GPT (2018)**
   - 基于自注意力的预训练模型
   - 注意力成为 NLP 核心组件

---

## 💡 核心洞察

### 1. 软对齐 (Soft Alignment)

**传统方法**（统计机器翻译）：
```
硬对齐: IBM Model 2, 3
  - 词对词的确定映射
  - 不可微分
  - 需要单独训练对齐模型
```

**Bahdanau 方法**：
```
软对齐: 注意力权重
  - 连续的概率分布
  - 可微分
  - 与翻译模型联合训练
  - 自动学习对齐！
```

### 2. 动态上下文

```
固定上下文 (无注意力):
  c(constant) → y₁, y₂, y₃, ...

动态上下文 (有注意力):
  c₁ → y₁
  c₂ → y₂
  c₃ → y₃
  ...

每步根据需要"检索"相关信息！
```

### 3. 可解释性

注意力提供了模型决策的可视化：

```
输入: "The cat sat on the mat"
输出: "Le chat s'est assis sur le tapis"

可视化:
  Le    █  ▒  ▒  ▒  ▒  ▒  ▒  (关注 "The")
  chat  ▒  █  ▒  ▒  ▒  ▒  ▒  (关注 "cat")
  s'est ▒  ▒  █  ▒  ▒  ▒  ▒  (关注 "sat")
  assis ▒  ▒  █  ▒  ▒  ▒  ▒  (关注 "sat")
  sur   ▒  ▒  ▒  █  ▒  ▒  ▒  (关注 "on")
  le    ▒  ▒  ▒  ▒  ▒  ▒  █  (关注 "the")
  tapis ▒  ▒  ▒  ▒  ▒  █  ▒  (关注 "mat")
```

**优势**：
- 调试更容易
- 可以发现错误
- 用户信任度更高

### 4. 计算图视角

从计算图的角度：

```
无注意力:
  h₁, ..., h_T → 最后一个 h_T → c → y₁, ..., y_T

  依赖链长: T 步
  梯度消失风险: 高

有注意力:
  h₁, ..., h_T ──┐
                  ├→ c_i → y_i (每个 i 都有直接连接)
  s_{i-1} ────────┘

  依赖链长: 1 步（直接）
  梯度消失风险: 低
```

---

## 🛠️ 实践指南

### PyTorch 实现

```python
import torch
import torch.nn as nn

class BahdanauAttentionPT(nn.Module):
    def __init__(self, hidden_size, encoder_size):
        super().__init__()
        self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
        self.U_a = nn.Linear(encoder_size, hidden_size, bias=False)
        self.v_a = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_annotations):
        """
        decoder_hidden: (batch, hidden_size)
        encoder_annotations: (batch, seq_len, encoder_size)
        """
        # 计算对齐分数
        # (batch, seq_len, hidden_size)
        aligned = self.W_a(decoder_hidden).unsqueeze(1) + \
                  self.U_a(encoder_annotations)
        aligned = torch.tanh(aligned)

        # (batch, seq_len, 1)
        scores = self.v_a(aligned).squeeze(-1)

        # Softmax
        attention_weights = torch.softmax(scores, dim=-1)

        # 计算上下文
        # (batch, 1, seq_len) × (batch, seq_len, encoder_size)
        context = torch.bmm(attention_weights.unsqueeze(1),
                           encoder_annotations).squeeze(1)

        return context, attention_weights
```

### 使用 HuggingFace

```python
from transformers import EncoderDecoderModel

# 使用预训练的 seq2seq 模型
model = EncoderDecoderModel.from_pretrained("facebook/bart-base")

# 生成（内部使用注意力）
outputs = model.generate(input_ids,
                        attention_mask=attention_mask,
                        max_length=50)
```

### 训练技巧

1. **Teacher Forcing**:
   ```python
   # 训练时使用真实标签作为输入
   # 而不是模型自己的预测
   decoder_input = target_ids[:, :-1]
   decoder_target = target_ids[:, 1:]
   ```

2. **梯度裁剪**:
   ```python
   # 注意力可能导致梯度爆炸
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

3. **标签平滑**:
   ```python
   # 防止过度自信
   criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
   ```

---

## 🧪 实践挑战

### 基础练习

1. **实现 Bahdanau 注意力**:
   ```python
   def bahdanau_attention(decoder_hidden, encoder_annotations):
       # TODO: 实现加性注意力
       pass
   ```

2. **可视化注意力**:
   - 训练一个简单的 seq2seq 模型
   - 绘制对齐矩阵
   - 分析注意力模式

3. **对比不同评分函数**:
   ```python
   def compare_attention_functions():
       # Bahdanau vs Luong vs Scaled Dot-Product
       pass
   ```

### 进阶练习

1. **实现双向注意力解码器**:
   ```python
   class BidirectionalDecoder:
       # 同时考虑前向和后向注意力
       pass
   ```

2. **覆盖机制 (Coverage Mechanism)**:
   ```python
   # 防止重复关注同一位置
   class CoverageAttention:
       def __init__(self):
           self.coverage_vector = None

       def update_coverage(self, attention_weights):
           if self.coverage_vector is None:
               self.coverage_vector = attention_weights
           else:
               self.coverage_vector += attention_weights
       pass
   ```

3. **多头注意力 (Multi-Head)**:
   - 实现 Transformer 风格的多头注意力
   - 对比单头 vs 多头

### 研究方向

1. **高效注意力**:
   - 稀疏注意力
   - 局部注意力
   - 线性复杂度注意力

2. **可解释注意力**:
   - 注意力是否真的等于因果关系？
   - 如何量化可解释性？

3. **跨模态注意力**:
   - 视觉-语言
   - 语音-文本

---

## ❓ 常见问题

### Q1: Bahdanau vs Luong 注意力的区别？

| 特性 | Bahdanau (2014) | Luong (2015) |
|------|----------------|--------------|
| **评分函数** | `v·tanh(W·s + U·h)` | `s·h` |
| **计算复杂度** | O(d²) | O(d²) 但更快 |
| **使用的状态** | `s_{i-1}` (上一时刻) | `s_i` (当前时刻) |
| **类型** | Global（所有位置） | Global + Local |
| **参数量** | 多 | 少 |

**选择建议**:
- 长序列: Bahdanau (表达能力更强)
- 短序列/追求速度: Luong

### Q2: 为什么需要双向编码器？

```
单向: "The bank"
  h₁("The")    → 只知道 "The"
  h₂("bank")   → 知道 "The bank"

  问题: h₁ 不知道后面是什么

双向:
  h₁("The")    → 前向 "The" + 后向 "整个句子"
  h₂("bank")   → 前向 "The bank" + 后向 "整个句子"

  优势: 每个位置都有完整上下文
```

### Q3: 注意力权重可以解释吗？

**可以，但要谨慎**：

```
支持证据:
  - 注意力与人类对齐高度相关
  - 可以发现明显错误
  - 帮助理解模型决策

反对证据:
  - 注意力 ≠ 因果关系
  - 模型可能通过其他路径传递信息
  - 高权重不一定表示重要性

结论:
  - 作为启发式工具很有用
  - 不应作为唯一的解释
```

### Q4: Bahdanau 注意力 vs 自注意力？

```
Bahdanau Attention (跨注意力):
  Query: 解码器状态
  Key/Value: 编码器状态
  应用: Seq2Seq（翻译、对话）

Self-Attention:
  Query, Key, Value: 同一序列
  应用: Transformer, BERT, GPT

关系:
  Self-Attention 是 Bahdanau 的特例
  （Query 和 Key/Value 来自同一序列）
```

---

## 📝 学习检查清单

完成以下任务以确保掌握 Bahdanau 注意力：

- [ ] 理解固定上下文的问题
- [ ] 推导 Bahdanau 注意力的数学公式
- [ ] 手动实现加性注意力
- [ ] 理解双向编码器的作用
- [ ] 可视化注意力对齐矩阵
- [ ] 对比 Bahdanau vs Luong 注意力
- [ ] 实现 attention decoder
- [ ] 在简单翻译任务上测试
- [ ] 分析不同序列长度下的表现
- [ ] 阅读 Transformer 论文（自注意力）

---

## 🔗 延伸阅读

### 必读论文

1. **Neural Machine Translation by Jointly Learning to Align and Translate (ICLR 2015)**
   - Bahdanau et al.
   - [arXiv:1409.0473](https://arxiv.org/abs/1409.0473)

2. **Effective Approaches to Attention-based Neural Machine Translation (EMNLP 2015)**
   - Luong et al.
   - 提出 Luong attention
   - [arXiv:1508.04025](https://arxiv.org/abs/1508.04025)

3. **Show, Attend and Tell (ICML 2015)**
   - Xu et al.
   - 注意力用于图像描述
   - [arXiv:1502.03044](https://arxiv.org/abs/1502.03044)

4. **Attention Is All You Need (2017)**
   - Vaswani et al.
   - 自注意力和 Transformer
   - 见 Paper 13

### 相关资源

- **Distill.pub: Visualizing Attention**:
  - [https://distill.pub/2016/augmented-rnns/](https://distill.pub/2016/augmented-rnns/)

- **The Annotated Transformer**:
  - [http://nlp.seas.harvard.edu/annotated-transformer](http://nlp.seas.harvard.edu/annotated-transformer)

- **Seq2Seq Attention Visualization**:
  - [https://github.com/jadore801120/attention-is-all-you-need-pytorch](https://github.com/jadore801120/attention-is-all-you-need-pytorch)

---

## 🎯 核心要点回顾

1. **革命性创新**:
   ```
   2014 年之前: 固定上下文向量（信息瓶颈）
   2014 年之后: 动态上下文向量（每步不同）
   ```

2. **核心机制**:
   ```
   对齐分数: e_ij = v·tanh(W·s + U·h)
   注意力权重: α_ij = softmax(e_ij)
   上下文向量: c_i = Σ α_ij · h_j
   ```

3. **关键优势**:
   - 解决信息瓶颈
   - 自动学习对齐
   - 处理长序列
   - 提供可解释性

4. **架构组件**:
   - 双向编码器（完整上下文）
   - 加性注意力（alignment model）
   - 动态上下文（每步计算）
   - 解码器（RNN + attention）

5. **深远影响**:
   - 启发了 Luong attention
   - 促进了图像注意力
   - 铺垫了 Transformer
   - 注意力成为深度学习标准组件

6. **实践要点**:
   - 适合需要可解释性的任务
   - 长序列表现优秀
   - 可以可视化对齐
   - 现代多用变体（Luong, Self-Attention）

**Bahdanau Attention 是注意力机制的起点，开启了现代 NLP 的新时代。**

---

*"The proposed approach of jointly learning to align and translate achieves significantly improved translation performance over the basic encoder–decoder approach."*
*— Bahdanau et al., 2014*
