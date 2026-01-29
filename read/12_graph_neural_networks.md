# Paper 12: Neural Message Passing for Quantum Chemistry

**论文标题**: Neural Message Passing for Quantum Chemistry
**作者**: Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, George E. Dahl (Google Brain)
**发表年份**: 2017 (ICLR 2017)
**引用次数**: 4,500+

---

## 📚 论文背景与核心问题

### 研究动机

传统深度学习主要处理**规则数据**：

```
图像: 规则网格结构 (2D grid)
文本: 序列结构 (1D sequence)
音频: 时间序列 (time series)
```

但现实世界中大量数据是**图结构**的：

```
分子: 原子为节点，化学键为边
社交网络: 用户为节点，关系为边
引文网络: 论文为节点，引用为边
知识图谱: 实体为节点，关系为边
推荐系统: 用户/商品为节点，交互为边
```

**核心挑战**：
1. 图的大小可变（不像图像固定尺寸）
2. 节点没有固定顺序（排列不变性）
3. 需要聚合邻居信息（不规则结构）

### MPNN 的贡献

**Message Passing Neural Networks (MPNN)** 提供了一个**统一的框架**，涵盖了当时主要的 GNN 架构：

```
MPNN 框架
├── Graph Convolutional Networks (GCN)
├── GraphSAGE
├── Graph Attention Networks (GAT)
├── Molecular GNNs
└── ... 其他 GNN 变体
```

**核心思想**：通过**消息传递 (message passing)** 机制，节点逐步聚合邻居信息。

---

## 🎯 图神经网络基础

### 1. 图的表示

#### 数学定义

图 `G = (V, E)` 由以下组成：

```
V: 节点集合 (vertices/nodes)
E: 边集合 (edges)

节点特征: h_v ∈ R^d_v  (每个节点的特征向量)
边特征:   e_vw ∈ R^e   (每条边的特征向量)
```

#### 代码实现

```python
class Graph:
    """简单的图表示"""
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.edges = []           # 边列表: [(src, tgt), ...]
        self.node_features = []   # 节点特征
        self.edge_features = {}   # 边特征: {(src, tgt): features}

    def add_edge(self, src, tgt, features=None):
        """添加有向边"""
        self.edges.append((src, tgt))
        if features is not None:
            self.edge_features[(src, tgt)] = features

    def get_neighbors(self, node):
        """获取节点的所有邻居"""
        neighbors = []
        for src, tgt in self.edges:
            if src == node:
                neighbors.append(tgt)
        return neighbors
```

#### 实例：水分子 (H₂O)

```python
# 创建水分子图
water = Graph(num_nodes=3)

# 添加边（O-H 键）
water.add_edge(0, 1)  # O → H
water.add_edge(0, 2)  # O → H
water.add_edge(1, 0)  # H → O (无向图需要双向)
water.add_edge(2, 0)  # H → O

# 节点特征: [原子序数, 化合价]
water.set_node_features([
    np.array([8, 2]),  # 氧原子
    np.array([1, 1]),  # 氢原子
    np.array([1, 1]),  # 氢原子
])

print(f"节点数: {water.num_nodes}")        # 3
print(f"边数: {len(water.edges)}")         # 4
print(f"氧原子的邻居: {water.get_neighbors(0)}")  # [1, 2]
```

---

## 🔄 消息传递框架

### 核心算法

MPNN 由两个阶段组成：

#### 阶段 1: 消息传递 (Message Passing)

重复 `T` 步：

```
对于每个节点 v:
  1. 从邻居收集消息:
     m_v^{t+1} = Σ_{u∈N(v)} M_t(h_v^t, h_u^t, e_vu)

  2. 更新节点状态:
     h_v^{t+1} = U_t(h_v^t, m_v^{t+1})
```

其中：
- `M_t`: 消息函数（Message function）
- `U_t`: 更新函数（Update function）
- `N(v)`: 节点 `v` 的邻居集合
- `h_v^t`: 节点 `v` 在时间步 `t` 的隐藏状态
- `e_vu`: 从 `v` 到 `u` 的边特征

#### 阶段 2: 读出 (Readout)

聚合所有节点信息得到图级表示：

```
ŷ = R({h_v^T | v ∈ G})
```

其中 `R` 是读出函数（Readout function）

### 直观理解

```
时间步 0: 每个节点只知道自己的信息
时间步 1: 节点聚合直接邻居的信息
时间步 2: 节点聚合 2-hop 邻居的信息
...
时间步 T: 节点的感受野覆盖 T-hop 邻居
```

---

## 🧮 消息传递详解

### 1. 消息函数 (Message Function)

**作用**: 计算从邻居 `u` 传给节点 `v` 的消息

#### 基础实现

```python
def message(self, h_source, h_target, e_features):
    """
    计算消息

    Args:
        h_source: 源节点 u 的隐藏状态
        h_target: 目标节点 v 的隐藏状态
        e_features: 边 (u→v) 的特征

    Returns:
        message: 从 u 传给 v 的消息向量
    """
    # 拼接源节点、目标节点、边特征
    if e_features is None:
        e_features = np.zeros(self.edge_dim)

    concat = np.concatenate([h_source, h_target, e_features])

    # 通过消息网络
    message = np.tanh(np.dot(self.W_msg, concat) + self.b_msg)
    return message
```

#### 数学表达

```
m_uv = M(h_u, h_v, e_uv)
     = tanh(W_msg · [h_u || h_v || e_uv] + b_msg)
```

其中 `||` 表示向量拼接。

**直观理解**：
- 消息包含：发送者的信息、接收者的信息、边的特征
- 这使得消息是"上下文感知"的

### 2. 聚合函数 (Aggregation Function)

**作用**: 将所有邻居的消息合并为一个向量

#### 常用聚合方式

```python
def sum_aggregation(messages):
    """求和聚合"""
    if len(messages) == 0:
        return np.zeros(self.hidden_dim)
    return np.sum(messages, axis=0)

def mean_aggregation(messages):
    """平均聚合"""
    if len(messages) == 0:
        return np.zeros(self.hidden_dim)
    return np.mean(messages, axis=0)

def max_aggregation(messages):
    """最大池化"""
    if len(messages) == 0:
        return np.zeros(self.hidden_dim)
    return np.max(messages, axis=0)
```

#### 聚合方式对比

| 聚合方式 | 优点 | 缺点 | 适用场景 |
|---------|------|------|----------|
| **Sum** | 保留完整信息，区分度高 | 对度敏感 | 分子图（固定度） |
| **Mean** | 归一化，稳定 | 丢失强度信息 | 社交网络（度变化大） |
| **Max** | 捕获最显著特征 | 丢失其他信息 | 特征匹配 |
| **Attention** | 加权聚合 | 计算开销大 | 异构图 |

### 3. 更新函数 (Update Function)

**作用**: 用聚合的消息更新节点状态

```python
def update(self, h_node, aggregated_message):
    """
    更新节点状态

    Args:
        h_node: 节点当前状态
        aggregated_message: 聚合后的消息

    Returns:
        h_new: 更新后的节点状态
    """
    # 拼接当前状态和消息
    concat = np.concatenate([h_node, aggregated_message])

    # 通过更新网络
    h_new = np.tanh(np.dot(self.W_update, concat) + self.b_update)
    return h_new
```

#### 数学表达

```
h_v^{t+1} = U(h_v^t, m_v^{t+1})
          = tanh(W_update · [h_v^t || m_v^{t+1}] + b_update)
```

**直观理解**：
- 当前状态 `h_v^t`: 节点自己的历史信息
- 聚合消息 `m_v^{t+1}`: 从邻居收到的信息
- 更新: 结合两者得到新状态

### 4. 完整的消息传递层

```python
class MessagePassingLayer:
    """单层消息传递"""
    def __init__(self, node_dim, edge_dim, hidden_dim):
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.hidden_dim = hidden_dim

        # 消息函数的参数
        self.W_msg = np.random.randn(hidden_dim, 2*node_dim + edge_dim) * 0.01
        self.b_msg = np.zeros(hidden_dim)

        # 更新函数的参数
        self.W_update = np.random.randn(node_dim, node_dim + hidden_dim) * 0.01
        self.b_update = np.zeros(node_dim)

    def forward(self, graph, node_states):
        """
        一步消息传递

        Args:
            graph: 图对象
            node_states: 当前所有节点的状态

        Returns:
            new_states: 更新后的节点状态
        """
        new_states = []

        for v in range(graph.num_nodes):
            # 1. 收集来自邻居的消息
            messages = []
            for u in graph.get_neighbors(v):
                edge_feat = graph.edge_features.get((u, v), None)
                msg = self.message(node_states[u], node_states[v], edge_feat)
                messages.append(msg)

            # 2. 聚合消息
            aggregated = self.aggregate(messages)

            # 3. 更新节点状态
            h_new = self.update(node_states[v], aggregated)
            new_states.append(h_new)

        return new_states
```

---

## 🏗️ 完整 MPNN 架构

### 网络结构

```python
class MPNN:
    """消息传递神经网络"""
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_layers, output_dim):
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1. 嵌入层: 将原始特征映射到隐藏空间
        self.embed_W = np.random.randn(hidden_dim, node_feat_dim) * 0.01

        # 2. 消息传递层
        self.mp_layers = [
            MessagePassingLayer(hidden_dim, edge_feat_dim, hidden_dim*2)
            for _ in range(num_layers)
        ]

        # 3. 读出层: 图级预测
        self.readout_W = np.random.randn(output_dim, hidden_dim) * 0.01
        self.readout_b = np.zeros(output_dim)

    def forward(self, graph):
        """
        前向传播

        Returns:
            prediction: 图级别的预测
            history: 每一步的节点状态（用于可视化）
        """
        # 1. 嵌入节点特征
        node_states = []
        for feat in graph.node_features:
            embedded = np.tanh(np.dot(self.embed_W, feat))
            node_states.append(embedded)

        # 2. 消息传递（T 步）
        states_history = [node_states]
        for layer in self.mp_layers:
            node_states = layer.forward(graph, node_states)
            states_history.append(node_states)

        # 3. 读出: 聚合所有节点状态
        graph_repr = np.sum(node_states, axis=0)  # 求和池化

        # 4. 最终预测
        output = np.dot(self.readout_W, graph_repr) + self.readout_b

        return output, states_history
```

### 训练流程

```python
# 伪代码
def train_mpnn(graphs, labels):
    """
    训练 MPNN

    Args:
        graphs: 图数据集
        labels: 图标签（如分子能量）
    """
    # 初始化网络
    mpnn = MPNN(node_feat_dim=2, edge_feat_dim=2,
                hidden_dim=64, num_layers=3, output_dim=1)

    # 训练循环
    for epoch in range(num_epochs):
        for graph, label in zip(graphs, labels):
            # 前向传播
            prediction, _ = mpnn.forward(graph)

            # 计算损失
            loss = mse_loss(prediction, label)

            # 反向传播
            gradients = compute_gradients(loss, mpnn)

            # 更新参数
            update_parameters(mpnn, gradients)

    return mpnn
```

---

## 📊 消息传递可视化

### 节点状态的演化

```python
# 运行 MPNN 并记录历史
mpnn = MPNN(node_feat_dim=2, edge_feat_dim=2,
            hidden_dim=8, num_layers=3, output_dim=1)

prediction, history = mpnn.forward(water)

# history 是一个列表，每个元素是一步的节点状态
# history[0]: 初始状态
# history[1]: 第1步消息传递后
# history[2]: 第2步消息传递后
# ...
```

### 可视化代码

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(history), figsize=(16, 4))

for step, states in enumerate(history):
    # 将节点状态堆叠成矩阵
    states_matrix = np.array(states).T  # (hidden_dim, num_nodes)

    ax = axes[step]
    im = ax.imshow(states_matrix, cmap='RdBu', aspect='auto')
    ax.set_title(f'Step {step}')
    ax.set_xlabel('Node')
    ax.set_ylabel('Hidden Dimension')

plt.colorbar(im, label='Activation')
plt.suptitle('Node Representations Through Message Passing')
plt.show()
```

### 观察结果

```
Step 0 (初始状态):
  每个节点的状态只来自自身特征

Step 1:
  节点聚合了 1-hop 邻居的信息
  - 氧原子: 聚合了 2 个氢原子的信息
  - 氢原子: 聚合了氧原子的信息

Step 2:
  节点聚合了 2-hop 邻居的信息
  - 氧原子: 间接看到另一个氢原子
  - 感受野扩大到整个分子

Step 3:
  节点的表示已经融合了全局信息
```

---

## 🌐 GNN 变体

MPNN 框架统一了多种 GNN 架构：

### 1. Graph Convolutional Networks (GCN)

**特点**: 简化的消息传递

```python
# GCN 的消息传递
h_v^{t+1} = σ(Σ_{u∈N(v)∪{v}} (1/√(d_u d_v)) · W · h_u^t)
```

**关键差异**:
- 消息函数: `M(h_u, h_v) = W · h_u`
- 聚合: 加权求和（度归一化）
- 更新: 简单的激活函数

**优点**:
- 简单高效
- 理论基础扎实（谱图理论）

**缺点**:
- 表达能力有限（只能在同构节点间传递）
- 归一化对稀疏图不稳定

### 2. GraphSAGE

**特点**: 采样邻居，归纳学习

```python
# GraphSAGE 的聚合
h_v^{t+1} = σ(W · CONCAT(h_v^t, AGG({h_u^t, u∈N(v)})))
```

**关键差异**:
- **采样**: 不是聚合所有邻居，而是采样固定数量
- **归纳**: 可以泛化到未见过的图/节点

**聚合方式**:
- Mean aggregator: `AGG = mean`
- Max pooling: `AGG = max`
- LSTM aggregator: `AGG = LSTM`

**适用场景**:
- 大规模图（无法聚合所有邻居）
- 动态图（不断添加新节点）

### 3. Graph Attention Networks (GAT)

**特点**: 注意力机制

```python
# GAT 的消息传递
α_vu = softmax_v(LeakyReLU(a^T [Wh_v || Wh_u]))
h_v^{t+1} = σ(Σ_{u∈N(v)∪{v}} α_vu · W · h_u^t)
```

**关键差异**:
- 消息权重 `α_vu` 是学习得到的（注意力系数）
- 不同邻居有不同的重要性

**优点**:
- 自适应地关注重要邻居
- 可解释性强（可以看注意力权重）

**应用**:
- 引文网络（某些论文更重要）
- 推荐系统（某些物品更相关）

### 4. Graph Isomorphism Network (GIN)

**特点**: 最强的表达能力

```python
# GIN 的更新
h_v^{t+1} = MLP((1 + ε) · h_v^t + Σ_{u∈N(v)} h_u^t)
```

**关键差异**:
- 使用求和聚合（+ MLP）
- 理论证明：和 WL test 一样强大

**表达能力**:
- 可以区分所有非同构图
- 比 GCN、GraphSAGE 更强

---

## 🔗 与其他论文的关联

### 前置论文

1. **Paper 10: ResNet**
   - GNN 中的消息传递类似于 ResNet 的跳跃连接
   - 都关注梯度流动和信息传播

2. **Paper 08: Seq2Seq for Sets**
   - 都处理无序数据
   - GNN 处理图，Seq2Seq 处理集合
   - 都需要排列不变性

3. **Paper 06: Pointer Networks**
   - 注意力机制
   - GAT 使用注意力聚合邻居

### 后续影响论文

1. **Graph Transformer (2020)**
   - 将 Transformer 应用于图
   - 全局注意力而非局部消息传递

2. **Equivariant GNNs (2021)**
   - 尊重物理对称性（旋转、平移）
   - 用于分子动力学模拟

3. **Temporal GNNs**
   - 处理动态图
   - 图结构随时间变化

---

## 💡 核心洞察

### 1. 感受野的扩展

```
时间步 t:
t=0: 节点只知道自己
t=1: 节点知道 1-hop 邻居
t=2: 节点知道 2-hop 邻居
...
t=T: 节点知道 T-hop 邻居

类似 CNN的感受野扩展！
```

### 2. 排列不变性

```
GNN 的输出不应该依赖节点的顺序:

Permutation Invariance:
  f(G) = f(π(G))

其中 π 是节点排列

实现方式:
  使用聚合函数 (sum, mean, max)
  这些函数对顺序不敏感
```

### 3. 归纳 vs 直推学习

```
直推学习 (Transductive):
  - 训练和测试在同一个图上
  - GCN, GNN 早期工作
  - 适合固定图（如引文网络）

归纳学习 (Inductive):
  - 训练在一组图上，测试在另一组图上
  - GraphSAGE, GIN
  - 适合分子预测（新分子）
```

### 4. 过度平滑 (Over-smoothing)

```
问题: 深层 GNN 中，所有节点的表示趋于相同

原因: 反复聚合邻居信息
  h_v^{t+1} = UPDATE(h_v^t, AGGREGATE(neighbors))

解决:
  - 使用跳连连接 (skip connections)
  - 限制深度（通常 < 5 层）
  - 使用不同的聚合方式
```

---

## 🛠️ 实践指南

### PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PyTorchMPNN(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, num_layers, output_dim):
        super().__init__()

        self.embedding = nn.Linear(node_feat_dim, hidden_dim)
        self.convs = nn.ModuleList([
            nn.Linear(hidden_dim * 2, hidden_dim)
            for _ in range(num_layers)
        ])
        self.readout = nn.Linear(hidden_dim, output_dim)

    def forward(self, node_features, edge_index):
        """
        Args:
            node_features: (num_nodes, feat_dim)
            edge_index: (2, num_edges) - [source_nodes, target_nodes]
        """
        # 嵌入
        x = F.relu(self.embedding(node_features))

        # 消息传递
        for conv in self.convs:
            # 聚合邻居（简化实现）
            row, col = edge_index
            messages = x[col]  # 获取邻居特征

            # 聚合（这里用 mean）
            agg = torch.zeros_like(x)
            agg.scatter_add_(0, row.unsqueeze(1).expand_as(messages), messages)
            deg = torch.zeros_like(x)
            deg.scatter_add_(0, row.unsqueeze(1).expand_as(row), row)
            agg = agg / (deg + 1)

            # 更新
            x = F.relu(conv(torch.cat([x, agg], dim=1)))

        # 读出
        graph_repr = torch.mean(x, dim=0)  # 全局平均池化
        output = self.readout(graph_repr)

        return output
```

### PyTorch Geometric

```python
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

class PyG_GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # 消息传递
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))

        # 读出
        x = torch.mean(x, dim=0)
        x = self.fc(x)

        return x

# 使用
data = Data(x=node_features, edge_index=edge_index, y=label)
model = PyG_GNN(input_dim=7, hidden_dim=64, output_dim=1)
```

### 设计 GNN 的建议

1. **选择合适的聚合方式**:
   ```python
   # 同质图（所有节点类型相同）
   aggregation = 'mean'  # 或 'sum'

   # 异构图（多种节点类型）
   aggregation = 'attention'  # GAT
   ```

2. **确定网络深度**:
   ```python
   # 小分子图
   num_layers = 3  # 足够覆盖整个分子

   # 大社交网络
   num_layers = 2  # 避免过度平滑
   ```

3. **处理边特征**:
   ```python
   # 分子图：键类型（单键、双键等）
   # 知识图谱：关系类型
   # 推荐系统：交互强度

   # 使用边特征的 GNN
   message = f(node_u, node_v, edge_uv)
   ```

4. **读出策略**:
   ```python
   # 图分类
   readout = 'global_mean_pool'  # 或 'global_max_pool'

   # 节点分类
   readout = None  # 每个节点单独预测

   # 边预测
   readout = 'concat'  # 拼接两个节点的表示
   ```

---

## 🧪 实践挑战

### 基础练习

1. **实现简单 GNN**:
   ```python
   class SimpleGNN:
       def __init__(self, hidden_dim, num_layers):
           # TODO: 初始化参数
           pass

       def forward(self, graph):
           # TODO: 实现消息传递
           pass
   ```

2. **分子属性预测**:
   - 在 ZINC 数据集上训练
   - 预测分子的溶解度

3. **可视化消息传递**:
   - 绘制每一步的节点表示
   - 观察信息如何传播

### 进阶练习

1. **实现不同的聚合函数**:
   ```python
   class Aggregator:
       def sum(self, messages):
           pass

       def mean(self, messages):
           pass

       def max(self, messages):
           pass

       def attention(self, messages, query):
           pass
   ```

2. **GraphSAGE 风格采样**:
   ```python
   def sample_neighbors(node, num_samples):
       """随机采样固定数量的邻居"""
       neighbors = get_all_neighbors(node)
       if len(neighbors) > num_samples:
           return random_sample(neighbors, num_samples)
       return neighbors
   ```

3. **GAT 注意力可视化**:
   - 训练 GAT 模型
   - 可视化注意力权重
   - 分析哪些邻居更重要

### 研究方向

1. **表达能力**:
   - 研究 GNN 的表达能力极限
   - 设计更强的聚合函数

2. **可扩展性**:
   - 大图（数十亿节点）的训练
   - 分布式 GNN 训练

3. **动态图**:
   - 时变图结构
   - 事件驱动的更新

4. **异构图**:
   - 多种节点类型
   - 多种边类型

---

## ❓ 常见问题

### Q1: GNN 和 CNN 的区别？

**CNN (图像)**:
```
规则网格结构
固定邻居数量（上下左右）
平移不变性
感受野通过卷积核扩大
```

**GNN (图)**:
```
不规则图结构
可变邻居数量
排列不变性
感受野通过消息传递扩大
```

**联系**:
```
CNN 可以看作是 GNN 的特例
（规则图上的 GNN）
```

### Q2: 何时使用 GNN？

**适合 GNN 的任务**:
```
1. 数据本身是图结构
   - 分子、化合物
   - 社交网络
   - 知识图谱

2. 数据可以建模为图
   - 场景理解（物体关系图）
   - 程序分析（AST/CFG）
   - 推荐系统（用户-物品二部图）

3. 需要捕获关系信息
   - 引文关系
   - 交互关系
   - 依赖关系
```

**不适合 GNN 的任务**:
```
1. 规则数据（图像、文本）
   → 使用 CNN、Transformer

2. 没有明确关系的数据
   → 使用 MLP、标准深度学习
```

### Q3: 如何处理大图？

**挑战**:
```
- 内存：无法存储整个图
- 计算：聚合所有邻居太慢
```

**解决方案**:
```python
# 1. 采样 (GraphSAGE)
neighbors = random_sample(all_neighbors, k=10)

# 2. 分批训练
subgraph = extract_subgraph(graph, seed_nodes)

# 3. 邻居缓存
cache_neighbor_features()

# 4. 分布式训练
partition_graph(graph, num_machines)
```

### Q4: GNN 的表达能力有多强？

**理论结果**:
```
1. GNN ≤ WL Test (Weisfeiler-Lehman)
   - GNN 无法区分所有非同构图
   - GIN 达到 WL Test 的能力

2. 一阶 GNN 无法捕获某些结构
   - 环状结构
   - 复杂的子图模式

3. 高阶 GNN（子图 GNN）能力更强
```

**实际中**:
```
- 大多数任务不需要区分所有非同构图
- GIN 的表达能力已经足够
- 过度追求表达能力可能导致过拟合
```

---

## 📝 学习检查清单

完成以下任务以确保掌握 GNN：

- [ ] 理解图的数学表示
- [ ] 手动实现消息传递机制
- [ ] 推导 MPNN 的前向传播
- [ ] 实现不同的聚合函数
- [ ] 使用 PyTorch Geometric 训练 GNN
- [ ] 在分子数据集上测试
- [ ] 可视化消息传递过程
- [ ] 理解过度平滑问题
- [ ] 对比 GCN、GraphSAGE、GAT
- [ ] 探索 GNN 在你的领域中的应用

---

## 🔗 延伸阅读

### 必读论文

1. **Neural Message Passing for Quantum Chemistry (ICLR 2017)**
   - Justin Gilmer et al.
   - MPNN 统一框架
   - [arXiv:1704.01212](https://arxiv.org/abs/1704.01212)

2. **Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017)**
   - Thomas N. Kipf, Max Welling
   - GCN 经典论文
   - [arXiv:1609.02907](https://arxiv.org/abs/1609.02907)

3. **GraphSAGE: Inductive Representation Learning on Large Graphs (NeurIPS 2017)**
   - William L. Hamilton et al.
   - 归纳学习
   - [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)

4. **Graph Attention Networks (ICLR 2018)**
   - Petar Veličković et al.
   - 注意力机制
   - [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)

5. **How Powerful are Graph Neural Networks? (ICLR 2019)**
   - Keyulu Xu et al.
   - GIN 表达能力分析
   - [arXiv:1810.00826](https://arxiv.org/abs/1810.00826)

### 相关资源

- **PyTorch Geometric**:
  - 最流行的 GNN 库
  - [https://pyg.org](https://pyg.org)

- **Deep Graph Library (DGL)**:
  - 多框架支持（PyTorch, MXNet, TensorFlow）
  - [https://www.dgl.ai](https://www.dgl.ai)

- **Graph Neural Network Course**:
  - Stanford CS224W
  - [http://web.stanford.edu/class/cs224w](http://web.stanford.edu/class/cs224w)

---

## 🎯 核心要点回顾

1. **核心思想**：
   - 节点通过消息传递聚合邻居信息
   - 多步传播扩大感受野

2. **MPNN 框架**：
   ```
   消息: m_v = Σ M(h_u, h_v, e_uv)
   更新: h_v = U(h_v, m_v)
   读出: ŷ = R({h_v})
   ```

3. **关键组件**：
   - 消息函数 M：计算从邻居传来的信息
   - 聚合函数：sum/mean/max/attention
   - 更新函数 U：结合自身和消息
   - 读出函数 R：图级表示

4. **主要变体**：
   - GCN：简化、谱方法
   - GraphSAGE：采样、归纳学习
   - GAT：注意力机制
   - GIN：最强表达能力

5. **应用领域**：
   - 分子性质预测
   - 社交网络分析
   - 推荐系统
   - 知识图谱

6. **实践要点**：
   - 选择合适的聚合方式
   - 控制网络深度（避免过度平滑）
   - 归纳 vs 直推学习
   - 使用成熟的库（PyG、DGL）

**GNN 将深度学习的成功从规则数据（图像、文本）扩展到了不规则数据（图），开启了图深度学习的新时代。**

---

*"Message passing neural networks provide a unified framework for building neural networks that operate on graph-structured data."*
*— Justin Gilmer et al., 2017*
