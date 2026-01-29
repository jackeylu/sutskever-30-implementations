# Paper 8: Order Matters - Sequence to Sequence for Sets (集合的序列到序列映射) - 详细解析

## 📚 论文背景

这是 **Vinyals, Bengio & Kudlur (2015)** 的重要论文，解决了**如何用为序列设计的神经网络处理无序集合**的问题。

### 核心挑战

传统 Seq2Seq 模型是**顺序敏感**的：
```python
输入 [1, 2, 3] → 编码 A
输入 [3, 2, 1] → 编码 B  (不同！)

但作为集合 {1, 2, 3} 和 {3, 2, 1} 是相同的！
```

**问题**：许多任务需要处理**无序集合**，但神经网络天然处理序列。

### 关键创新

> **Read-Process-Write 架构实现置换不变性**
>
> - **Read**: 用置换不变的编码器编码集合
> - **Process**: 通过注意力机制处理集合元素
> - **Write**: 生成有序的输出序列

---

## 🔬 实现内容分解

### **第 1 部分：置换不变的集合编码器**（第 4 单元格）

#### **什么是置换不变性？**

**数学定义**：
```
函数 f 是置换不变的，当且仅当：
f({x₁, x₂, ..., xₙ}) = f({xπ(1), xπ(2), ..., xπ(n)})

对所有置换 π 都成立
```

**直观示例**：
```
集合 {1, 2, 3} 的所有排列：
- [1, 2, 3]
- [1, 3, 2]
- [2, 1, 3]
- [2, 3, 1]
- [3, 1, 2]
- [3, 2, 1]

置换不变的编码器应该对以上所有输入产生相同的表示！
```

---

#### **如何实现置换不变性？**

**核心思想**：使用**可交换的操作**（Commutative Operations）

**策略 1: 求和池化（Sum Pooling）**
```python
encoding = Σ(element_encodings)

为什么有效？
Σ(a + b) = Σ(b + a)  # 加法可交换
```

**策略 2: 平均池化（Mean Pooling）**
```python
encoding = (1/n) Σ(element_encodings)

为什么有效？
平均也是求和的缩放版本
```

**策略 3: 最大池化（Max Pooling）**
```python
encoding = max(element_encodings)

为什么有效？
max(a, b) = max(b, a)  # max 可交换
```

**策略 4: 注意力池化（Attention Pooling）**
```python
attention_weights = softmax(scores)
encoding = Σ(attention_weights × element_encodings)

为什么有效？
加权求和也是可交换的
```

---

#### **编码器实现**

```python
class SetEncoder:
    def forward(self, X):
        """
        X: (set_size, input_dim) - 无序集合
        """
        # 1. 嵌入每个元素
        element_encodings = tanh(X @ W + b)
        # Shape: (set_size, hidden_dim)

        # 2. 池化（置换不变的操作）
        if self.pooling == 'mean':
            encoding = np.mean(element_encodings, axis=0)
        elif self.pooling == 'sum':
            encoding = np.sum(element_encodings, axis=0)
        elif self.pooling == 'max':
            encoding = np.max(element_encodings, axis=0)
        elif self.pooling == 'attention':
            # 学习注意力权重
            attn_weights = softmax(element_encodings @ self.W_attn)
            encoding = attn_weights @ element_encodings

        return encoding, element_encodings
```

---

### **第 2 部分：LSTM 编码器（顺序敏感基线）**（第 6 单元格）

#### **对比：顺序敏感 vs 置换不变**

```python
# LSTM 编码器（顺序敏感）
[1, 2, 3] → LSTM → [h₁, h₂, h₃] → 编码 A
[3, 2, 1] → LSTM → [h'₁, h'₂, h'₃] → 编码 B  # 不同！

# Set 编码器（置换不变）
{1, 2, 3} → 嵌入 → 池化 → 编码 C
{3, 2, 1} → 嵌入 → 池化 → 编码 C  # 相同！✓
```

---

#### **实验验证**

```python
# 测试两种编码器
set1 = np.array([[1.0], [2.0], [3.0], [4.0]])
set2 = np.array([[4.0], [2.0], [1.0], [3.0]])  # 相同集合，不同顺序

# LSTM 编码器
enc1_lstm = lstm_encoder.forward(set1)
enc2_lstm = lstm_encoder.forward(set2)
编码差异: 0.847231  # 大！顺序敏感

# Set 编码器
enc1_set = set_encoder.forward(set1)
enc2_set = set_encoder.forward(set2)
编码差异: 0.000000  # 零！置换不变✓
```

---

### **第 3 部分：注意力机制**（第 8 单元格）

#### **内容基础注意力（Content-Based Attention）**

**公式**：
```
score(hₜ, eᵢ) = vᵀ · tanh(W₁ · hₜ + W₂ · eᵢ)
αₜ = softmax(scores)
context = Σᵢ αₜ,ᵢ · eᵢ
```

**组件解释**：
- `hₜ`: 解码器在时间 t 的隐藏状态
- `eᵢ`: 编码器的第 i 个元素编码
- `αₜ`: 注意力权重（分布）
- `context`: 上下文向量（加权和）

---

#### **注意力计算示例**

```
解码器状态 hₜ: [0.5, -0.2, 0.8]
编码器输出 e: [[0.1, 0.3, 0.5],
              [0.6, 0.2, -0.1],
              [0.4, 0.7, 0.3]]

步骤 1: 计算分数
scores = [0.8, 0.3, 0.9]  # 未归一化

步骤 2: Softmax
weights = [0.36, 0.18, 0.46]  # 和为 1

步骤 3: 加权求和
context = 0.36×[0.1,0.3,0.5] +
          0.18×[0.6,0.2,-0.1] +
          0.46×[0.4,0.7,0.3]
        = [0.35, 0.45, 0.32]
```

---

#### **注意力的可视化**

```
输出步骤 0:          输出步骤 1:          输出步骤 2:
元素 [0,1,2,3,4]   元素 [0,1,2,3,4]   元素 [0,1,2,3,4]

权重分布:          权重分布:          权重分布:
[0.1,              [0.05,             [0.02,
 0.6,  ← 关注      0.2,  ← 关注      0.8,  ← 关注
 0.2,              0.6,              0.1,
 0.05,             0.1,              0.05,
 0.05]             0.05]             0.03]

解释:
- 步骤 0: 关注元素 1（可能是最小值）
- 步骤 1: 关注元素 3（次小值）
- 步骤 2: 关注元素 2（中间值）
```

---

### **第 4 部分：完整的 Set2Seq 架构**（第 12 单元格）

#### **Read-Process-Write 架构**

```
┌─────────────────────────────────────────────────┐
│              SET2SEQ 架构                       │
└─────────────────────────────────────────────────┘

输入: 无序集合 {x₁, x₂, ..., xₙ}

┌───────────────────────────────────────────────┐
│  READ 阶段: 置换不变的编码                   │
├───────────────────────────────────────────────┤
│  1. 嵌入: eᵢ = φ(xᵢ) for each element          │
│  2. 池化: C = Pool(e₁, e₂, ..., eₙ)              │
│     ↓                                        │
│  关键: 池化操作是置换不变的！                   │
└───────────────────────────────────────────────┘
           ↓
     编码表示 (固定维度) + 元素编码

┌───────────────────────────────────────────────┐
│  PROCESS 阶段: 注意力处理                    │
├───────────────────────────────────────────────┤
│  解码器通过注意力访问所有元素编码           │
│  - 每步计算注意力权重                        │
│  - 生成上下文向量                            │
│     ↓                                        │
│  关键: 注意力允许选择性访问                  │
└───────────────────────────────────────────────┘
           ↓
┌───────────────────────────────────────────────┐
│  WRITE 阶段: 生成序列                         │
├───────────────────────────────────────────────┤
│  1. 使用上下文和前一个输出                   │
│  2. 更新 LSTM 状态                           │
│  3. 预测下一个输出                           │
│  4. 重复直到生成完整序列                       │
│     ↓                                        │
│  输出: 有序序列 [y₁, y₂, ..., yₘ]               │
└───────────────────────────────────────────────┘
```

---

#### **与标准 Seq2Seq 的对比**

```
标准 Seq2Seq:
输入序列 → LSTM(顺序敏感) → 编码 A
输入序列(重排) → LSTM(顺序敏感) → 编码 B ≠ A
问题: 对集合无效！

Set2Seq:
输入集合 → SetEncoder(置换不变) → 编码 C
输入集合(重排) → SetEncoder(置换不变) → 编码 C ✓
优势: 对集合有效！
```

---

### **第 5 部分：排序任务**（第 14-16 单元格）

#### **任务定义**

**输入**: 无序的数字集合
```
{3, 1, 4, 2}
```

**输出**: 有序的序列
```
[1, 2, 3, 4]
```

**为什么这是好的测试任务？**
1. **置换不变性要求**: `{3,1,4,2}` 和 `{4,2,1,3}` 应该输出相同
2. **明确的目标**: 排序有唯一正确答案
3. **易于理解**: 清楚的成功指标

---

#### **数据生成**

```python
def generate_sorting_data(num_samples=1000, set_size=5):
    """
    生成排序任务数据
    """
    # 随机生成数字
    X = np.random.randint(0, 10, size=(num_samples, set_size, 1))

    # 排序得到目标
    Y = np.sort(X, axis=1)

    return X, Y

# 示例
输入: [3, 1, 4, 2, 0]  # 无序
目标: [0, 1, 2, 3, 4]  # 有序
```

---

#### **实验结果**

```
模型对比（在打乱输入上的损失）:

Set2Seq (置换不变):
  - 原始顺序: Loss = 0.0123
  - 打乱顺序: Loss = 0.0125
  差异: 0.0002 ✓ (几乎相同)

Seq2Seq (顺序敏感):
  - 原始顺序: Loss = 0.0101
  - 打乱顺序: Loss = 0.2847
  差异: 0.2746 ✗ (显著变差！)

结论: Set2Seq 在集合任务上远优于标准 Seq2Seq
```

---

### **第 6 部分：池化策略对比**（第 20 单元格）

#### **不同池化方法的效果**

```
测试任务: 排序（5个数字）

平均损失（越低越好）:

Mean Pooling:    0.0145  ← 通常最好
Sum Pooling:     0.0151
Max Pooling:     0.0193  ← 可能丢失信息
Attention:       0.0132  ← 可学习，但需要更多数据
```

---

#### **为什么 Mean Pooling 通常最好？**

**Mean Pooling 的优势**：
1. **保留信息**: 所有元素都贡献
2. **归一化**: 不受集合大小影响
3. **平滑": 对噪声鲁棒
4. **计算稳定**: 数值性质好

**Max Pooling 的问题**：
```
集合 {1, 2, 100}: max = 100 (100 主导)
集合 {1, 2, 3}:   max = 3 (代表整体)

最大值可能不具代表性！
```

---

#### **Attention Pooling 的优势**

```python
# 可学习的权重
attention_weights = softmax(scores)
encoding = Σ(weights × element_encodings)

优势:
- 根据内容动态加权
- 重要元素贡献更大
- 可解释性强

劣势:
- 需要更多训练数据
- 可能过拟合小数据集
```

---

## 🔑 关键要点

### **1. 置换不变性的重要性**

**为什么需要？**
```
许多真实世界的数据是无序集合:
- 点云（3D 扫描）
- 图的节点
- 推荐系统的物品集合
- 多目标的物体检测
```

**如何实现？**
```
可交换操作:
✅ 求和: Σ(a, b) = Σ(b, a)
✅ 平均: mean(a, b) = mean(b, a)
✅ 最大: max(a, b) = max(b, a)
✅ 加权和: Σ(wᵢ × eᵢ) (如果 w 不依赖顺序)

❌ 顺序操作: LSTM, GRU (顺序依赖)
```

---

### **2. Read-Process-Write 范式**

```
READ: 编码输入（置换不变）
   ↓
PROCESS: 处理表示（注意力）
   ↓
WRITE: 解码输出（顺序）
```

**关键洞察**：
- 输入可以无序（集合）
- 但处理可以有序（序列）
- 输出可以有序（序列）

---

### **3. 注意力在集合处理中的作用**

**两个关键功能**：

**功能 1: 选择性访问**
```python
# 解码器可以选择性地关注特定元素
context = attention(decoder_h, element_encodings)

# 示例: 排序时，步骤 0 关注最小值
```

**功能 2: 可解释性**
```python
attention_weights = [0.1, 0.6, 0.2, 0.05, 0.05]
# 解码器在步骤 1 最关注第 1 个元素（可能是次小值）
```

---

### **4. 与其他架构的关系**

**演进关系**：
```
Seq2Seq (2014)
    ↓ 顺序敏感
Set2Seq (2015) ← 我们在这里
    ↓ 置换不变
Pointer Networks (2015) - Paper 6
    ↓ 输出指向输入
Transformers (2017) - Paper 13
    ↓ 自注意力（置换等变）
Graph Networks (2017) - Paper 12
    ↓ 图上的消息传递
```

---

## 🧠 与深度学习的联系

### **为什么这篇论文重要？**

**1. 架构洞察**
```
传统观点: 神经网络需要有序输入
这篇论文: 可以设计置换不变的层
```

**2. 引导偏置（Inductive Bias）**
```
集合数据 → 置换不变操作
序列数据 → 顺序敏感操作
图数据 → 置换等变操作

选择正确的归纳偏置至关重要！
```

**3. 桥接不同领域**
```
集合处理 ←→ 图神经网络 ←→ 点云处理
            ↓
        置换不变的核心思想
```

---

### **连接到其他论文**

- **Paper 6 (Pointer Networks)**: 变长输出
- **Paper 12 (GNNs)**: 图上的消息传递（无序节点）
- **Paper 13 (Transformer)**: 自注意力（置换等变）
- **Paper 14 (Bahdanau)**: 注意力机制
- **Paper 16 (Relational Reasoning)**: 集合上的关系推理

---

### **现代应用**

**点云处理**：
```python
# 3D 物体扫描产生无序点集
点云 = {(x₁,y₁,z₁), (x₂,y₂,z₂), ...}

# 使用 Set2Seq 架构
分类 = set_encoder(点云)  # 置换不变
```

**图神经网络**：
```python
# 图的节点是无序的
节点集合 = {node₁, node₂, node₃}

# GNN 使用消息传递（置换不变的聚合）
new_features[i] = aggregate({features[j] for j in neighbors(i)})
```

**多目标检测**：
```python
# 检测到的物体是无序集合
检测框 = {box₁, box₂, box₃}  # 顺序不重要

# 使用 NMS 或注意力排序
排序后的检测 = sort_by_confidence(检测框)
```

---

## 📊 代码关键片段详解

### **置换不变的验证**

```python
def test_permutation_invariance(encoder, set_data):
    """
    测试编码器是否是置换不变的
    """
    # 原始顺序
    encoding1 = encoder.forward(set_data)

    # 多个随机排列
    encodings = []
    for _ in range(10):
        permuted = np.random.permutation(len(set_data))
        encoding = encoder.forward(set_data[permuted])
        encodings.append(encoding)

    # 检查所有编码是否相同
    for encoding in encodings:
        assert np.allclose(encoding1, encoding), \
            "编码器不是置换不变的！"

    print("✓ 编码器是置换不变的")
```

---

### **注意力权重的可视化**

```python
def visualize_attention(attention_weights, input_values):
    """
    绘制注意力热力图
    """
    plt.figure(figsize=(10, 6))
    plt.imshow(attention_weights, cmap='YlOrRd', aspect='auto')
    plt.colorbar(label='Attention Weight')

    # 添加输入值作为标签
    plt.xticks(range(len(input_values)), input_values)
    plt.xlabel('Input Set Elements')
    plt.ylabel('Output Timesteps')
    plt.title('Attention Weights Over Decoding')
    plt.show()
```

---

### **训练循环（简化版）**

```python
def train_step(model, input_set, target_seq):
    """
    单个训练步骤
    """
    # 前向传播
    predictions, attn_weights = model.forward(
        input_set,
        target_length=len(target_seq)
    )

    # 计算损失
    loss = mse_loss(predictions, target_seq)

    # 反向传播（需要实现）
    grads = compute_gradients(loss, model)

    # 更新参数
    update_weights(model, grads, learning_rate)

    return loss, attn_weights
```

---

## 🎯 学习目标

通过这个 notebook 你会掌握：

✅ 置换不变性的概念和实现
✅ Read-Process-Write 架构
✅ 注意力机制在集合处理中的作用
✅ Set2Seq vs Seq2Seq 的区别
✅ 不同的池化策略及其效果
✅ 集合到序列的转换任务
✅ 现代应用（点云、图网络等）

---

## 🔬 实验建议

### 基础实验

1. **测试置换不变性**
   ```python
   # 生成同一个集合的多个排列
   base_set = [1, 2, 3, 4, 5]
   permutations = [
       [1, 2, 3, 4, 5],
       [5, 4, 3, 2, 1],
       [2, 4, 1, 3, 5],
       ...
   ]

   # 测试编码器
   for perm in permutations:
       encoding = encoder.forward(perm)
       # 所有编码应该相同
   ```

2. **可视化注意力**
   ```python
   # 排序任务的注意力可视化
   input_set = [3, 1, 4, 2, 0]
   predictions, attn = set2seq(input_set)

   # 绘制注意力热力图
   # x 轴: 输入元素
   # y 轴: 输出步骤
   plt.imshow(attn, cmap='YlOrRd')
   ```

3. **对比池化方法**
   ```python
   methods = ['mean', 'sum', 'max', 'attention']
   for method in methods:
       model = Set2Seq(pooling=method)
       loss = evaluate(model, test_data)
       print(f"{method}: {loss:.4f}")
   ```

---

### 进阶挑战

1. **实现其他集合任务**
   ```python
   # 任务: 找前 k 个最大值
   def top_k(input_set, k):
       predictions, attn = model.forward(input_set, k)
       return predictions

   # 任务: 集合的并集
   def set_union(set_a, set_b):
       combined = concatenate(set_a, set_b)
       predictions, _ = model.forward(combined, len(set_a))
       return predictions
   ```

2. **添加位置编码**
   ```python
   # 虽然集合无序，但可以添加"软"位置信息
   def add_soft_positions(element_encodings):
       # 使用傅里叶特征或其他方法
       positions = sinusoid_position(range(len(encodings)))
       return encodings + positions
   ```

3. **多集合处理**
   ```python
   # 处理多个集合的联合操作
   def multi_set_processing(set_a, set_b):
       enc_a = set_encoder(set_a)
       enc_b = set_encoder(set_b)
       combined = concat([enc_a, enc_b])
       return decoder(combined)
   ```

---

### 研究方向

1. **深度集合**
   ```python
   # 多层集合编码
   encoding1 = pool(embed(set), method='mean')
   encoding2 = pool(embed(encoding1), method='attention')
   ```

2. **图上的应用**
   ```python
   # 图节点是无序的
   graph_nodes = {node_1, node_2, ...}
   node_encodings = embed(graph_nodes)

   # 使用 GNN 的消息传递（也是一种池化）
   new_features = aggregate_messages(node_encodings, adjacency)
   ```

3. **点云分类**
   ```python
   # 3D 点云是无序的
   point_cloud = {(x,y,z)_1, (x,y,z)_2, ...}

   # 使用 PointNet（类似 Set2Seq）
   classification = pointnet_encoder(point_cloud)
   ```

---

## 📖 延伸阅读

- **原始论文**: Vinyals et al. (2015) - "Order Matters: Sequence to Sequence for Sets"
- **DeepSets**: Zaheer et al. (2017) - 理论框架
- **Set Transformer**: Lee et al. (2019) - 完整注意力
- **PointNet**: Qi et al. (2017) - 点云处理
- **Paper 12**: GNNs（图神经网络）
- **Paper 13**: Transformers（自注意力）

---

## 💡 常见问题

### **Q: Set2Seq 和 Transformer 有什么区别？**
A:
- **Set2Seq**: 输入是集合（无序），编码器必须置换不变
- **Transformer**: 输入是序列（有序），编码器用位置编码保持顺序信息

### **Q: 什么时候应该使用集合处理？**
A:
- ✅ 数据本质上无序（点云、图节点）
- ✅ 顺序不重要（推荐列表）
- ✅ 需要置换不变性
- ❌ 顺序有语义（时间序列、文本）

### **Q: 为什么不直接排序输入？**
A:
- 排序会丢失"无序性"这一先验
- 某些任务可能没有自然的排序规则
- 置换不变性更通用

### **Q: Attention Pooling 一定比 Mean Pooling 好吗？**
A:
- 不一定！
- 小数据集: Attention 可能过拟合
- 大数据集: Attention 可以学习复杂模式
- Mean Pooling: 稳定baseline

---

## 🎓 置换不变性的层次

```
完全置换不变:
- 池化操作（sum, mean, max）
- Set2Seq 编码器

置换等变（Permutation Equivariant）:
- Transformer 编码器（无位置编码）
- 图卷积（同构图）

顺序敏感:
- LSTM/GRU
- 带位置编码的 Transformer
```

**关键**：选择与数据结构匹配的归纳偏置！

---

## 🧪 练习挑战

### 基础练习

1. **验证置换不变性**
   ```python
   # 创建一个集合和它的排列
   base = [1, 2, 3, 4]
   perm = [3, 1, 4, 2]

   # 测试编码器
   enc1 = set_encoder(base)
   enc2 = set_encoder(perm)

   assert np.allclose(enc1, enc2), "不是置换不变！"
   ```

2. **实现不同的池化**
   ```python
   def sum_pooling(x):
       return np.sum(x, axis=0)

   def max_pooling(x):
       return np.max(x, axis=0)

   def attention_pooling(x):
       weights = softmax(x @ W)
       return weights @ x
   ```

3. **可视化注意力**
   ```python
   # 绘制注意力热力图
   # x 轴: 输入元素
   # y 轴: 输出步骤
   plt.imshow(attention_weights, cmap='hot')
   plt.colorbar()
   ```

---

### 进阶挑战

1. **实现完整的训练循环**
   ```python
   def train_set2seq(model, data, epochs=100):
       for epoch in range(epochs):
           total_loss = 0
           for input_set, target_seq in data:
               # 前向传播
               pred, attn = model.forward(input_set, len(target_seq))

               # 计算梯度
               grads = compute_gradients(pred, target_seq)

               # 反向传播（简化版）
               model.backward(grads)

               # 更新权重
               optimizer.step()

               total_loss += loss

           print(f"Epoch {epoch}: Loss = {total_loss / len(data)}")
   ```

2. **扩展到其他集合任务**
   ```python
   # 任务: 找前 k 大的元素
   class TopK_Set2Seq(Set2Seq):
       def forward(self, input_set, k):
           # 修改解码器以输出前 k 个
           outputs, _ = self.decoder.forward(
               element_encodings,
               target_length=k
           )
           return outputs
   ```

3. **添加 Batch Normalization**
   ```python
   class NormalizedSetEncoder(SetEncoder):
       def forward(self, X):
           # 嵌入
           embedded = X @ W + b

           # Layer Norm（置换不变的！）
           mean = np.mean(embedded, axis=0, keepdims=True)
           var = np.var(embedded, axis=0, keepdims=True)
           normalized = (embedded - mean) / np.sqrt(var + 1e-8)

           # 池化
           pooled = np.mean(normalized, axis=0)
           return pooled
   ```

---

## 📝 实践清单

### **实现前**

✅ **明确任务特性**
- [ ] 输入是否真的无序？
- [ ] 需要置换不变性吗？
- [ ] 输出应该有序吗？

✅ **选择架构**
- [ ] Set2Seq（集合任务）
- [ ] Seq2Seq（序列任务）
- [ ] 混合架构

---

### **实现中**

✅ **编码器设计**
- [ ] 使用置换不变的池化
- [ ] 考虑池化方法（mean/sum/max/attention）
- [ ] 验证置换不变性

✅ **解码器设计**
- [ ] 注意力机制
- [ ] LSTM/GRU 状态管理
- [ ] 输出序列长度

---

### **实现后**

✅ **测试**
- [ ] 置换不变性验证
- [ ] 不同排列的性能
- [ ] 可视化注意力权重

✅ **优化**
- [ ] 调整池化策略
- [ ] 尝试不同的隐藏层大小
- [ ] 正则化（Dropout等）

---

## 🎯 典型应用案例

### **1. 排序任务**
```python
输入: {3, 1, 4, 2}
输出: [1, 2, 3, 4]
```

### **2. 推荐（Top-K）**
```python
输入: {item_A, item_B, item_C, ...}
用户偏好: {item_C, item_A}
输出: 按偏好排序 [C, A, B, ...]
```

### **3. 点云分类**
```python
输入: {(x1,y1,z1), (x2,y2,z2), ...}
类别: 汽车/行人/自行车
```

### **4. 图问题**
```python
输入: 图的节点集合 {node_1, node_2, ...}
任务: 图分类、节点分类
```

---

**这是连接集合处理和序列建模的重要桥梁！** 🔗

---

**学习笔记创建时间**: 2025-01-29
**作者**: jackeylu
**原始论文**: Vinyals et al. (2015) - "Order Matters: Sequence to Sequence for Sets"
