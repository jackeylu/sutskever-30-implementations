# Paper 13: Attention Is All You Need

**论文标题**: Attention Is All You Need
**作者**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin (Google Brain)
**发表年份**: 2017 (NeurIPS 2017)
**引用次数**: 100,000+ (深度学习史上最具影响力的论文之一)

---

## 📚 论文背景与核心问题

### 研究动机

在 Transformer 出现之前，序列建模主要依赖 RNN/CNN：

```
RNN (LSTM/GRU):
  ✅ 可以处理变长序列
  ✅ 自然地捕获时序依赖
  ❌ 顺序计算（无法并行）
  ❌ 长距离依赖梯度消失

CNN (Temporal Convolutional Networks):
  ✅ 可以并行计算
  ✅ 梯度流动好
  ❌ 需要很深才能捕获长距离依赖
  ❌ 固定的感受野大小
```

### Transformer 的革命性洞察

**"Attention Is All You Need"**：
- 不需要 RNN 的循环连接
- 不需要 CNN 的滑动窗口
- **只需要注意力机制**！

### 核心优势

| 特性 | RNN/LSTM | CNN | **Transformer** ⭐ |
|------|----------|-----|-------------------|
| 并行计算 | ❌ 顺序 | ✅ | ✅ 完全并行 |
| 长距离依赖 | ❌ 梯度消失 | ⚠️ 需要深层 | ✅ 直接连接 |
| 感受野 | 线性增长 | 线性增长 | **全局（一步到位）** |
| 计算复杂度 | O(n) | O(k·n) | O(n²) (注意) |
| 可解释性 | 低 | 中 | **高（注意力可视化）** |

---

## 🎯 缩放点积注意力 (Scaled Dot-Product Attention)

### 基础定义

Transformer 的核心是**自注意力机制 (Self-Attention)**：

```
输入序列 X = [x₁, x₂, ..., xₙ]

每个位置生成三个向量:
  Query (查询):   Qᵢ = W_Q · xᵢ
  Key (键):      Kᵢ = W_K · xᵢ
  Value (值):    Vᵢ = W_V · xᵢ

注意力计算:
  Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

### 直观理解

```
类比信息检索系统:
  Query (查询): "我想要什么信息？"
  Key (键):    "每个位置有什么信息？"
  Value (值):  "实际的信息内容"

注意力权重 = Query 与 Key 的匹配度
输出 = 加权平均 Value（权重 = 注意力权重）
```

### 数学推导

#### 1. 计算注意力分数

```python
# Q, K, V: (seq_len, d_k)
scores = Q · K^T  # (seq_len, seq_len)

# 每个 score[i,j] 表示位置 i 与位置 j 的相关性
```

**问题**: 当 `d_k` 很大时，点积会很大！
```
假设 Q, K 的每个分量是均值为 0、方差为 1 的随机变量
则 Q·K^T 的方差 = d_k

d_k 很大 → softmax 进入饱和区 → 梯度消失
```

**解决方案**: 缩放！
```
scaled_scores = (Q · K^T) / √d_k
```

#### 2. Softmax 归一化

```python
attention_weights = softmax(scaled_scores, axis=-1)
```

归一化使得每一行的和为 1：
```
attention_weights[i, :] = [αᵢ₁, αᵢ₂, ..., αᵢₙ]
其中 αᵢ₁ + αᵢ₂ + ... + αᵢₙ = 1
```

#### 3. 加权求和 Value

```python
output = attention_weights · V
```

对于每个位置 i：
```
output[i] = αᵢ₁·V₁ + αᵢ₂·V₂ + ... + αᵢₙ·Vₙ
```

### 完整实现

```python
def softmax(x, axis=-1):
    """数值稳定的 softmax"""
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力

    Args:
        Q: (seq_len_q, d_k)
        K: (seq_len_k, d_k)
        V: (seq_len_v, d_v)
        mask: (seq_len_q, seq_len_k) 可选

    Returns:
        output: (seq_len_q, d_v)
        attention_weights: (seq_len_q, seq_len_k)
    """
    d_k = Q.shape[-1]

    # 1. 计算注意力分数
    scores = np.dot(Q, K.T) / np.sqrt(d_k)

    # 2. 应用 mask（如果提供）
    if mask is not None:
        # 将被 mask 的位置设为 -∞
        scores = scores + (mask * -1e9)

    # 3. Softmax 得到注意力权重
    attention_weights = softmax(scores, axis=-1)

    # 4. 加权求和
    output = np.dot(attention_weights, V)

    return output, attention_weights
```

### 注意力矩阵可视化

```
Attention Weights Matrix:

         Key Position
        0   1   2   3   4
Query
  0    [0.6 0.2 0.1 0.05 0.05]  ← 主要关注位置 0
  1    [0.1 0.7 0.1 0.05 0.05]  ← 主要关注位置 1
  2    [0.05 0.1 0.6 0.15 0.1]  ← 主要关注位置 2,3
  3    [0.05 0.05 0.15 0.6 0.15] ← 主要关注位置 3
  4    [0.05 0.05 0.1 0.2 0.6]   ← 主要关注位置 4

颜色越亮 = 注意力权重越大
```

**观察**：
- 对角线最亮：每个位置最关注自己（自注意力）
- 可以看到远距离的依赖（不像 RNN 需要逐步传递）

---

## 🧩 多头注意力 (Multi-Head Attention)

### 动机

**问题**: 单个注意力头只能捕获一种模式

**解决方案**: 使用多个头，每个头学习不同的注意力模式！

```
类比人类视觉:
  - 一个头关注语法结构
  - 一个头关注语义关系
  - 一个头关注指代关系
  ...
```

### 数学表达

```
MultiHead(Q, K, V) = Concat(head₁, ..., headₕ) · W^O

其中:
  headᵢ = Attention(Q · Wᵢ_Q, K · Wᵢ_K, V · Wᵢ_V)
```

### 架构详解

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0  # 确保 d_model 能被 num_heads 整除

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 每个头的维度

        # Q, K, V 的投影矩阵（所有头共享）
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1

        # 输出投影
        self.W_o = np.random.randn(d_model, d_model) * 0.1
```

### 分步实现

#### Step 1: 线性投影

```python
def forward(self, Q, K, V, mask=None):
    # Q, K, V: (seq_len, d_model)

    # 投影到 d_model 维度
    Q_proj = np.dot(Q, self.W_q.T)  # (seq_len, d_model)
    K_proj = np.dot(K, self.W_k.T)
    V_proj = np.dot(V, self.W_v.T)
```

#### Step 2: 分割成多个头

```python
def split_heads(self, x):
    """
    将 (seq_len, d_model) 分割成 (num_heads, seq_len, d_k)

    例如: d_model=64, num_heads=8
    输入: (10, 64)
    输出: (8, 10, 8)
    """
    seq_len = x.shape[0]
    # 先 reshape: (seq_len, num_heads, d_k)
    x = x.reshape(seq_len, self.num_heads, self.d_k)
    # 再 transpose: (num_heads, seq_len, d_k)
    return x.transpose(1, 0, 2)

# 使用
Q_heads = self.split_heads(Q_proj)  # (num_heads, seq_len, d_k)
K_heads = self.split_heads(K_proj)
V_heads = self.split_heads(V_proj)
```

#### Step 3: 对每个头计算注意力

```python
head_outputs = []
self.attention_weights = []

for i in range(self.num_heads):
    head_out, head_attn = scaled_dot_product_attention(
        Q_heads[i], K_heads[i], V_heads[i], mask
    )
    head_outputs.append(head_out)
    self.attention_weights.append(head_attn)

# head_outputs[i]: (seq_len, d_k)
```

#### Step 4: 合并头

```python
def combine_heads(self, x):
    """
    将 (num_heads, seq_len, d_k) 合并成 (seq_len, d_model)
    """
    # x: (num_heads, seq_len, d_k)
    seq_len = x.shape[1]
    # transpose: (seq_len, num_heads, d_k)
    x = x.transpose(1, 0, 2)
    # reshape: (seq_len, d_model)
    return x.reshape(seq_len, self.d_model)

# 使用
heads = np.stack(head_outputs, axis=0)  # (num_heads, seq_len, d_k)
combined = self.combine_heads(heads)     # (seq_len, d_model)
```

#### Step 5: 最终投影

```python
output = np.dot(combined, self.W_o.T)  # (seq_len, d_model)
```

### 完整代码

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1

    def split_heads(self, x):
        seq_len = x.shape[0]
        x = x.reshape(seq_len, self.num_heads, self.d_k)
        return x.transpose(1, 0, 2)

    def combine_heads(self, x):
        seq_len = x.shape[1]
        x = x.transpose(1, 0, 2)
        return x.reshape(seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        # 投影
        Q = np.dot(Q, self.W_q.T)
        K = np.dot(K, self.W_k.T)
        V = np.dot(V, self.W_v.T)

        # 分头
        Q = self.split_heads(Q)  # (num_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 每个头计算注意力
        head_outputs = []
        self.attention_weights = []

        for i in range(self.num_heads):
            head_out, head_attn = scaled_dot_product_attention(
                Q[i], K[i], V[i], mask
            )
            head_outputs.append(head_out)
            self.attention_weights.append(head_attn)

        # 合并头
        heads = np.stack(head_outputs, axis=0)
        combined = self.combine_heads(heads)

        # 最终投影
        output = np.dot(combined, self.W_o.T)

        return output
```

### 多头注意力可视化

```
Head 1:  Head 2:  Head 3:  Head 4:
[■ □ □]  [□ ■ □]  [■ □ □]  [□ □ ■]
[□ ■ □]  [□ □ ■]  [□ ■ □]  [□ □ ■]
[□ □ ■]  [□ ■ □]  [□ □ ■]  [□ ■ □]

每个头关注不同的模式！
```

**实际例子**（BERT）：
- Head 0-5: 关注语法（next sentence prediction）
- Head 6-11: 关注语义（语义关系、指代消解）

### 参数量分析

```
d_model = 512, num_heads = 8

参数量:
  W_q: 512 × 512 = 262K
  W_k: 512 × 512 = 262K
  W_v: 512 × 512 = 262K
  W_o: 512 × 512 = 262K
  总计: ~1M 参数

vs 单头注意力 (d_k = 512):
  W_q: 512 × 512 = 262K
  W_k: 512 × 512 = 262K
  W_v: 512 × 512 = 262K
  W_o: 512 × 512 = 262K
  总计: ~1M 参数

参数量相同！
但多头可以捕获更丰富的表示。
```

---

## 📍 位置编码 (Positional Encoding)

### 为什么需要？

**问题**: 自注意力机制是**排列不变**的

```
Input:  "I love cats"
Input:  "Cats I love"

如果不添加位置信息，Transformer 认为这两句话完全相同！
```

**解决方案**: 显式添加位置信息

### 正弦/余弦位置编码

#### 数学定义

```
PE_(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE_(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

其中:
  pos: 位置索引 (0, 1, 2, ..., seq_len-1)
  i:   维度索引 (0, 1, 2, ..., d_model/2)
```

#### 实现

```python
def positional_encoding(seq_len, d_model):
    """
    正弦位置编码

    Args:
        seq_len: 序列长度
        d_model: 模型维度

    Returns:
        pe: (seq_len, d_model)
    """
    pe = np.zeros((seq_len, d_model))

    # 位置索引: (seq_len, 1)
    position = np.arange(0, seq_len)[:, np.newaxis]

    # 除数项: (d_model/2,)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    # 偶数维度用 sin
    pe[:, 0::2] = np.sin(position * div_term)

    # 奇数维度用 cos
    pe[:, 1::2] = np.cos(position * div_term)

    return pe
```

### 为什么这样设计？

#### 1. 不同频率编码

```
维度 0,1:   波长 = 2π           (快速变化，捕获细粒度位置)
维度 2,3:   波长 = 2π × 100     (较慢变化)
维度 4,5:   波长 = 2π × 10000   (很慢变化，捕获粗粒度位置)
...
```

**可视化**：
```
维度 0:  ╱╲╱╲╱╲╱╲╱╲╱╲   (高频)
维度 10: ╱──╲──╱──╲──      (中频)
维度 50: ╱──────╲──────     (低频)
```

#### 2. 位置关系

**性质**: PE(pos + k) 可以表示为 PE(pos) 的线性函数

```
sin(a + b) = sin(a)cos(b) + cos(a)sin(b)
cos(a + b) = cos(a)cos(b) - sin(a)sin(b)

→ 位置 k 相对于位置 pos 的关系可以学习！
```

#### 3. 外推能力

```
训练时最大长度: 128
测试时输入长度: 200

正弦编码可以平滑外推到未见过的位置！
（vs 学习的位置编码，无法外推）
```

### 使用方式

```python
# 输入嵌入
X_embedded = embedding_layer(X)  # (seq_len, d_model)

# 位置编码
pe = positional_encoding(seq_len, d_model)

# 相加（不是拼接！）
X_with_pos = X_embedded + pe
```

**为什么相加而不是拼接？**
- 相加: 不增加维度，计算高效
- 位置信息可以与内容信息"交互"

---

## 🔄 前馈网络 (Feed-Forward Network)

### 结构

每个位置独立应用相同的 FFN：

```
FFN(x) = ReLU(x · W₁ + b₁) · W₂ + b₂

维度变换:
  x: (seq_len, d_model)
  → hidden: (seq_len, d_ff)
  → output: (seq_len, d_model)

其中 d_ff = 4 × d_model (通常)
```

### 实现

```python
class FeedForward:
    def __init__(self, d_model, d_ff):
        self.W1 = np.random.randn(d_model, d_ff) * 0.1
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * 0.1
        self.b2 = np.zeros(d_model)

    def forward(self, x):
        # 第一层: 扩展
        hidden = np.maximum(0, np.dot(x, self.W1) + self.b1)

        # 第二层: 压缩回原维度
        output = np.dot(hidden, self.W2) + self.b2

        return output
```

### 作用

1. **增加非线性**:
   - 注意力机制是线性组合
   - FFN 提供必要的非线性变换

2. **位置独立变换**:
   - 每个位置的 FFN 参数共享
   - 类似于 1×1 卷积（Point-wise）

3. **特征空间扩展**:
   ```
   d_model = 512
   d_ff = 2048

   先映射到更高维空间（2048）
   再投影回原维度（512）

   类似于"信息压缩-扩展"
   ```

---

## 📏 层归一化 (Layer Normalization)

### vs Batch Normalization

```
BatchNorm:
  - 对 batch 维度归一化
  - 依赖于 batch size
  - CNN 中常用

LayerNorm:
  - 对特征维度归一化
  - 不依赖 batch size
  - Transformer / RNN 中常用
```

### 数学

```python
class LayerNorm:
    def __init__(self, d_model, eps=1e-6):
        self.gamma = np.ones(d_model)  # 缩放参数
        self.beta = np.zeros(d_model)  # 平移参数
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: (seq_len, d_model)

        Returns:
            output: (seq_len, d_model)
        """
        # 对每个位置，在特征维度上计算均值和方差
        mean = x.mean(axis=-1, keepdims=True)  # (seq_len, 1)
        std = x.std(axis=-1, keepdims=True)    # (seq_len, 1)

        # 归一化
        normalized = (x - mean) / (std + self.eps)

        # 缩放和平移（可学习参数）
        output = self.gamma * normalized + self.beta

        return output
```

### 为什么用 LayerNorm？

1. **序列长度可变**:
   ```
   BatchNorm 需要足够的样本统计
   但 NLP 中序列长度差异很大
   ```

2. **训练稳定**:
   ```
   Transformer 很深（6-12 层）
   LayerNorm 帮助梯度流动
   ```

3. **位置独立**:
   ```
   LayerNorm 作用于每个位置
   不影响序列长度
   ```

---

## 🧱 完整 Transformer Block

### 架构

```
Input (seq_len, d_model)
  ↓
┌─────────────────────────────────┐
│  1. Multi-Head Self-Attention   │
│     (捕获序列内关系)             │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│  2. Residual Connection + Norm  │
│     x = LayerNorm(x + Attn(x))  │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│  3. Feed-Forward Network        │
│     (位置独立变换)               │
└─────────────────────────────────┘
  ↓
┌─────────────────────────────────┐
│  4. Residual Connection + Norm  │
│     x = LayerNorm(x + FFN(x))   │
└─────────────────────────────────┘
  ↓
Output (seq_len, d_model)
```

### 实现

```python
class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        Args:
            x: (seq_len, d_model)
            mask: (seq_len, seq_len) 可选

        Returns:
            output: (seq_len, d_model)
        """
        # 1. 多头自注意力 + 残差连接 + 层归一化
        attn_output = self.attention.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_output)

        # 2. 前馈网络 + 残差连接 + 层归一化
        ff_output = self.ff.forward(x)
        x = self.norm2.forward(x + ff_output)

        return x
```

### 残差连接的作用

**ResNet 的遗产**（见 Paper 10）：
```
x = LayerNorm(x + Sublayer(x))

作用:
  1. 梯度高速公路
  2. 允许网络学习"恒等映射"
  3. 稳定深层网络训练
```

---

## 🎭 因果自注意力 (Causal Self-Attention)

### 何时需要？

**自回归生成**（如 GPT）：
```
生成第 t 个 token 时，只能看到前 t-1 个 token
不能看到未来信息！
```

### 因果掩码

```python
def create_causal_mask(seq_len):
    """
    创建因果掩码（下三角矩阵）

    Args:
        seq_len: 序列长度

    Returns:
        mask: (seq_len, seq_len)
        0 = 可以 attend
        1 = 被 mask（不能 attend）
    """
    mask = np.triu(np.ones((seq_len, seq_len)), k=1)
    return mask
```

**示例** (seq_len=5):
```
mask =
    [[0, 1, 1, 1, 1],   ← 位置 0 只能看到自己
     [0, 0, 1, 1, 1],   ← 位置 1 可以看到 0,1
     [0, 0, 0, 1, 1],   ← 位置 2 可以看到 0,1,2
     [0, 0, 0, 0, 1],   ← 位置 3 可以看到 0,1,2,3
     [0, 0, 0, 0, 0]]   ← 位置 4 可以看到 0,1,2,3,4
```

### 应用掩码

```python
# 在 softmax 之前
scores = np.dot(Q, K.T) / np.sqrt(d_k)

# 将被 mask 的位置设为 -∞
scores = scores + (mask * -1e9)

# Softmax 后，被 mask 的位置权重 ≈ 0
attention_weights = softmax(scores, axis=-1)
```

### Bidirectional vs Causal

```
Bidirectional Attention (BERT):
  - 可以看到整个序列
  - 适合理解任务（分类）

Causal Attention (GPT):
  - 只能看到过去
  - 适合生成任务（语言建模）
```

---

## 🏗️ 完整 Transformer 架构

### Encoder-Decoder (原论文)

```
Encoder:
  Input Embedding + Positional Encoding
    ↓
  [Transformer Block] × N (N=6)
    ↓
  Memory (编码后的表示)

Decoder:
  Target Embedding + Positional Encoding
    ↓
  [Transformer Block] × N (N=6)
    ├─ Masked Multi-Head Attention (self)
    ├─ Multi-Head Attention (cross-attention with Encoder)
    └─ Feed-Forward
    ↓
  Linear + Softmax
    ↓
  Output Probabilities
```

### 变体

1. **Encoder-only** (BERT):
   ```
   Input → [Encoder Block] × N → Output

   应用: 分类、理解任务
   ```

2. **Decoder-only** (GPT):
   ```
   Input → [Decoder Block] × N → Output
   (只用 Masked Self-Attention)

   应用: 生成任务
   ```

3. **Encoder-Decoder** (原始 Transformer, T5):
   ```
   Encoder → Memory → Decoder → Output

   应用: 翻译、序列到序列
   ```

---

## 🔗 与其他论文的关联

### 前置论文

1. **Paper 06: Pointer Networks**
   - 注意力机制的早期应用
   - "软指针"类似于注意力权重

2. **Paper 10: ResNet**
   - 残差连接
   - LayerNorm + Residual 稳定训练

3. **Paper 12: Graph Neural Networks**
   - GNN 的消息传递类似于自注意力
   - Transformer = 完全连接图上的 GNN

### 后续影响论文

1. **BERT (2018)**
   - Encoder-only Transformer
   - 双向上下文理解

2. **GPT 系列 (2018-2023)**
   - Decoder-only Transformer
   - 自回归语言模型

3. **Vision Transformer (ViT, 2020)**
   - Transformer 用于图像
   - Patch embedding

4. **BART, T5 (2019-2020)**
   - Encoder-Decoder Transformer
   - 序列到序列任务

---

## 💡 核心洞察

### 1. 注意力即全连接图

```
自注意力 = 完全连接图上的消息传递

每个位置:
  - 向所有位置发送"查询"
  - 从所有位置接收"键-值"
  - 加权聚合得到更新

类似 GNN，但:
  - GNN: 稀疏邻接矩阵
  - Transformer: 密集邻接矩阵（权重 = 注意力）
```

### 2. 并行化是关键

```
RNN (LSTM):
  t=0: 串行计算
  t=1: 需要等 t=0 完成
  t=2: 需要等 t=1 完成
  → 训练慢

Transformer:
  所有位置同时计算注意力
  → 完全并行
  → 训练快 100x+
```

### 3. O(n²) 复杂度的权衡

```
注意力计算: O(n² · d)

n: 序列长度
d: 模型维度

短序列 (n < 1024): Transformer 优势明显
长序列 (n > 4096): O(n²) 成为瓶颈

解决方案:
  - Sparse Attention (Longformer, BigBird)
  - Linear Attention (Performer, Linformer)
  - Hierarchical Attention
```

### 4. 归纳偏置的差异

```
CNN:
  强归纳偏置（局部性、平移不变性）
  → 适合数据充足的任务

Transformer:
  弱归纳偏置（几乎没有假设）
  → 适合大规模预训练
  → 可以从数据中学习所有模式
```

---

## 🛠️ 实践指南

### PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionPT(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性投影
        Q = self.W_q(query)  # (batch, seq_len, d_model)
        K = self.W_k(key)
        V = self.W_v(value)

        # 分头
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # 注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 加权求和
        context = torch.matmul(attn_weights, V)

        # 合并头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 最终投影
        output = self.W_o(context)

        return output, attn_weights
```

### 使用 HuggingFace Transformers

```python
from transformers import BertModel, GPT2Model

# BERT (Encoder-only)
bert = BertModel.from_pretrained('bert-base-uncased')
outputs = bert(input_ids)
last_hidden_state = outputs.last_hidden_state  # (batch, seq_len, 768)

# GPT-2 (Decoder-only)
gpt2 = GPT2Model.from_pretrained('gpt2')
outputs = gpt2(input_ids)
hidden_states = outputs.last_hidden_state  # (batch, seq_len, 768)
```

### 训练技巧

1. **学习率调度**:
   ```python
   # 原论文使用带 warmup 的 Adam
   lr = d_model^(-0.5) · min(step^(-0.5), step · warmup^(-1.5))
   ```

2. **标签平滑**:
   ```python
   # 防止过度自信
   label_smoothing = 0.1
   ```

3. **Dropout**:
   ```python
   # Transformer 使用大量 Dropout
   dropout = 0.1  # 几乎所有地方
   ```

---

## 🧪 实践挑战

### 基础练习

1. **实现注意力机制**:
   ```python
   def scaled_dot_product_attention(Q, K, V):
       # TODO: 实现注意力计算
       pass
   ```

2. **可视化注意力**:
   - 训练一个小 Transformer
   - 可视化不同头的注意力模式

3. **位置编码实验**:
   - 对比正弦 vs 学习的位置编码
   - 测试外推能力

### 进阶练习

1. **实现完整 Transformer**:
   ```python
   class Transformer:
       def __init__(self, vocab_size, d_model, num_heads, num_layers):
           # TODO: 初始化所有组件
           pass

       def forward(self, input_ids, mask=None):
           # TODO: 实现前向传播
           pass
   ```

2. **实现不同变体**:
   - Encoder-only (BERT-style)
   - Decoder-only (GPT-style)
   - Encoder-Decoder (T5-style)

3. **优化注意力计算**:
   ```python
   # 实现 Flash Attention
   # 减少 HBM 访问次数
   ```

### 研究方向

1. **高效注意力**:
   - Linear Attention
   - Sparse Attention
   - Local Attention

2. **长序列建模**:
   - 分层注意力
   - 压缩记忆
   - Recurrent Memory Transformer

3. **多模态扩展**:
   - Vision-Language Transformers
   - Audio-Visual Transformers

4. **因果推理**:
   - Interpretable Attention
   - Attention as Explanation

---

## ❓ 常见问题

### Q1: 为什么除以 √d_k？

**问题**: 当 d_k 很大时，点积会很大

```
假设 Q, K ~ N(0, 1)
则 Q·K^T 的方差 = d_k

d_k = 64 → 点积方差 = 64
d_k = 512 → 点积方差 = 512

Softmax 输入很大 → 进入饱和区 → 梯度消失
```

**解决**: 缩放到单位方差
```
(Q·K^T) / √d_k → 方差 = 1
```

### Q2: 多头 vs 单头？

**参数量相同**:
```
单头 (d_k = 512):  512 × 512 × 4 = 1M
多头 (8×64):       512 × 512 × 4 = 1M
```

**表达能力不同**:
```
单头: 只能学习一种注意力模式
多头: 可以学习 8 种不同的模式
```

**类比**:
```
单头 = 一个专家
多头 = 8 个专家，各有所长
```

### Q3: 为什么不用 RNN？

**训练效率**:
```
RNN: 串行计算
  t=0 → t=1 → t=2 → ... → t=n
  → 无法并行，训练慢

Transformer: 并行计算
  所有位置同时计算
  → GPU 利用率高，训练快 100x+
```

**长距离依赖**:
```
RNN: 梯度需要逐步传递
  → 梯度消失

Transformer: 直接连接
  → 梯度流动畅通
```

### Q4: Transformer 的局限性？

1. **计算复杂度**: O(n²)
   - 长序列很慢
   - 需要近似算法

2. **弱归纳偏置**:
   - 需要大量数据
   - 小数据下不如 CNN

3. **可解释性争议**:
   - 注意力权重 ≠ 因果关系
   - 需要谨慎解释

---

## 📝 学习检查清单

完成以下任务以确保掌握 Transformer：

- [ ] 手动实现缩放点积注意力
- [ ] 理解为什么需要缩放 (1/√d_k)
- [ ] 实现多头注意力（分头、合并）
- [ ] 理解位置编码的数学原理
- [ ] 实现完整的 Transformer Block
- [ ] 理解因果掩码的作用
- [ ] 使用 PyTorch/TensorFlow 训练 Transformer
- [ ] 可视化注意力模式
- [ ] 对比 Encoder/Decoder 架构
- [ ] 阅读 BERT/GPT 论文

---

## 🔗 延伸阅读

### 必读论文

1. **Attention Is All You Need (NeurIPS 2017)**
   - Vaswani et al.
   - [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)

2. **BERT: Pre-training of Deep Bidirectional Transformers (2019)**
   - Devlin et al.
   - [arXiv:1810.04805](https://arxiv.org/abs/1810.04805)

3. **Language Models are Unsupervised Multitask Learners (GPT-2, 2019)**
   - Radford et al.
   - [OpenAI Blog](https://openai.com/research/better-language-models)

4. **An Image is Worth 16x16 Words: ViT (2020)**
   - Dosovitskiy et al.
   - [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

### 相关资源

- **The Annotated Transformer**:
  - Harvard's annotated implementation
  - [http://nlp.seas.harvard.edu/annotated-transformer](http://nlp.seas.harvard.edu/annotated-transformer)

- **Illustrated Transformer**:
  - Jay Alammar's blog
  - [http://jalammar.github.io/illustrated-transformer](http://jalammar.github.io/illustrated-transformer)

- **HuggingFace Transformers**:
  - 最流行的 Transformer 库
  - [https://huggingface.co](https://huggingface.co)

---

## 🎯 核心要点回顾

1. **革命性突破**:
   - 抛弃 RNN/CNN，纯注意力架构
   - 完全并行化，训练效率大幅提升

2. **核心组件**:
   ```
   Scaled Dot-Product Attention:
     Attention(Q, K, V) = softmax(QK^T / √d_k) V

   Multi-Head Attention:
     多个头并行，捕获不同模式

   Positional Encoding:
     正弦/余弦函数注入位置信息

   Feed-Forward + LayerNorm + Residual:
     稳定深层网络训练
   ```

3. **关键优势**:
   - 并行计算（vs RNN 串行）
   - 全局感受野（vs CNN 局部）
   - 可解释性（注意力可视化）

4. **架构变体**:
   - Encoder-only (BERT): 理解任务
   - Decoder-only (GPT): 生成任务
   - Encoder-Decoder (T5): 翻译任务

5. **深远影响**:
   - 现代所有 LLM 的基础
   - 从 NLP 扩展到 CV、多模态
   - 开启了大模型时代

6. **实践要点**:
   - 使用成熟库（HuggingFace）
   - 注意 O(n²) 复杂度
   - 根据任务选择架构

**Transformer 彻底改变了深度学习，开启了预训练大模型的新时代。**

---

*"The Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output."*
*— Vaswani et al., 2017*
