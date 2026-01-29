# Paper 15: Identity Mappings in Deep Residual Networks

**论文标题**: Identity Mappings in Deep Residual Networks
**作者**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (微软亚洲研究院 MSRA)
**发表年份**: 2016 (ECCV 2016)
**引用次数**: 10,000+ (ResNet v2)

---

## 📚 论文背景与核心问题

### 研究动机

在 Paper 10 中，ResNet 通过残差连接实现了训练 152 层网络。但在尝试更深的网络（> 200 层）时，遇到了新的问题：

```
原始 ResNet (Post-activation):
  x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
                                        ↑
                                   问题在这里！
```

**核心问题**：原始 ResNet 的恒等路径**并不干净**！

### 原始 ResNet 的问题

#### 1. 恒等路径被阻断

```
原始 ResNet:
  y = ReLU(F(x) + x)

问题分析:
  - 残差路径: F(x)
  - 恒等路径: x
  - 相加后: F(x) + x  ✓ 干净
  - ReLU 后: ReLU(F(x) + x)  ✗ 阻断了恒等路径！
```

**为什么这是问题？**

```
理想情况:
  如果 F(x) = 0 (残差路径学习零)
  则 y = x (完美的恒等映射)

实际情况 (原始 ResNet):
  如果 F(x) = 0
  则 y = ReLU(0 + x) = ReLU(x)
  → 不是恒等映射！
  → x 如果有负值，会被截断
  → 梯度流动受阻
```

#### 2. 梯度流动受阻

```
反向传播时:
  ∂L/∂x = ∂L/∂y · (∂F/∂x + 1) · ∂ReLU/∂(F(x)+x)

ReLU 的导数:
  ∂ReLU/∂z = 1 if z > 0, 0 if z ≤ 0

问题:
  - 如果 F(x) + x ≤ 0, ReLU 导数为 0
  - 梯度完全消失！
  - 即使 F(x) + x > 0, 梯度也受 ReLU 影响
```

#### 3. 深度网络训练困难

实验观察：
```
网络深度     原始 ResNet 训练误差
───────────────────────────────
  18 层         6.2%
  34 层         5.8%
  50 层         6.5%  ← 开始上升
 101 层        12.3%  ← 明显恶化
 152 层        18.7%  ← 几乎无法训练
```

**Kaiming He 的洞察**：
```
问题不是深度，而是架构细节！
如果恒等路径真正干净，1000 层也应该能训练。
```

---

## 🎯 Pre-activation ResNet

### 核心创新

**将激活函数移到卷积之前！**

```
原始 ResNet (Post-activation):
  x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
                                              ↑
                                          ReLU 阻断

Pre-activation ResNet:
  x → BN → ReLU → Conv → BN → ReLU → Conv → (+x)
                                              ↑
                                         干净的恒等路径！
```

### 架构对比

#### 原始 ResNet Block

```python
class OriginalResidualBlock:
    def forward(self, x):
        # 第一层
        out = conv1(x)
        out = batch_norm(out)
        out = relu(out)

        # 第二层
        out = conv2(out)
        out = batch_norm(out)

        # 残差连接
        out = out + x

        # 激活（在相加之后！）
        out = relu(out)

        return out
```

**前向传播**：
```
y = ReLU(F(x) + x)

其中 F(x) = W₂ · ReLU(BN(W₁ · x))
```

#### Pre-activation ResNet Block

```python
class PreActivationResidualBlock:
    def forward(self, x):
        # 第一层: BN → ReLU → Conv
        out = batch_norm(x)
        out = relu(out)
        out = conv1(out)

        # 第二层: BN → ReLU → Conv
        out = batch_norm(out)
        out = relu(out)
        out = conv2(out)

        # 残差连接（相加后没有激活！）
        out = out + x

        return out
```

**前向传播**：
```
y = F'(x) + x

其中 F'(x) = W₂ · ReLU(BN(W₁ · ReLU(BN(x))))
```

### 关键差异

| 方面 | 原始 ResNet | Pre-activation ResNet |
|------|-------------|----------------------|
| **激活位置** | Conv 之后 | Conv 之前 |
| **恒等路径** | x → ... → (+x) → ReLU | x → ... → (+x) |
| **恒等是否干净** | ❌ 被 ReLU 阻断 | ✅ 完全干净 |
| **梯度流动** | 较好 | **更好** |
| **训练难度** | 中等 | **更容易** |

---

## 📐 梯度流动分析

### 原始 ResNet 的梯度流动

```
y = ReLU(F(x) + x)

反向传播:
  ∂L/∂x = ∂L/∂y · (∂F/∂x + 1) · ∂ReLU/∂(F(x)+x)

ReLU 导数:
  ∂ReLU/∂z = 1 if z > 0
           = 0 if z ≤ 0

问题:
  1. 如果 F(x) + x ≤ 0, 梯度完全消失
  2. 即使 > 0, ReLU 导数也会削弱梯度
  3. 恒等路径不是真正的恒等
```

**可视化**：
```
梯度流动 (原始 ResNet):

输出梯度
  ↓
[ReLU 导数可能为 0] ← 梯度杀手
  ↓
[残差路径梯度 + 恒等路径梯度]
  ↓
输入梯度

如果 ReLU 导数 = 0:
  输出梯度 → 0 → 输入梯度 = 0  ← 梯度消失！
```

### Pre-activation 的梯度流动

```
y = F'(x) + x

反向传播:
  ∂L/∂x = ∂L/∂y · (∂F'/∂x + 1)

关键:
  - 没有 ReLU 阻断！
  - 恒等路径导数恒为 1
  - 梯度可以自由流动
```

**可视化**：
```
梯度流动 (Pre-activation):

输出梯度
  ↓
[残差路径梯度 + 恒等路径梯度]
  ↓
  ↓  ← 没有任何阻断！
  ↓
输入梯度

恒等路径: 梯度 = 输出梯度 × 1
残差路径: 梯度 = 输出梯度 × ∂F'/∂x

两者相加 → 梯度更强！
```

### 数学证明

**定理**：假设 `F'(x)` 是 Lipschitz 连续的，且 Lipschitz 常数 L < 1，则：

```
||∂L/∂x|| ≤ (L + 1) ||∂L/∂y||
```

**证明**：
```
∂L/∂x = ∂L/∂y · (∂F'/∂x + 1)

取范数:
  ||∂L/∂x|| = ||∂L/∂y|| · ||∂F'/∂x + 1||
           ≤ ||∂L/∂y|| · (||∂F'/∂x|| + 1)
           ≤ ||∂L/∂y|| · (L + 1)

如果 L < 1, 则 (L+1) < 2
→ 梯度不会消失！
→ 梯度不会爆炸！
```

**原始 ResNet 的问题**：
```
∂L/∂x = ∂L/∂y · (∂F/∂x + 1) · ∂ReLU/∂z

如果 ∂ReLU/∂z = 0:
  ||∂L/∂x|| = 0  ← 梯度消失
```

---

## 🔍 激活函数放置位置分析

### 论文测试的四种配置

#### Configuration A: Original ResNet (Post-activation)

```
x → Conv → BN → ReLU → Conv → BN → (+x) → ReLU
```

**特点**：
- 激活在卷积之后
- 最后有 ReLU
- 恒等路径被阻断

**评分**：★★★☆☆

#### Configuration B: BN after addition

```
x → Conv → BN → ReLU → Conv → BN → (+x) → BN → ReLU
```

**特点**：
- 相加后有 BN
- 最后有 ReLU
- 恒等路径被 BN 和 ReLU 双重阻断

**评分**：★★☆☆☆

**为什么更差？**
```
BN 也会改变信号:
  y = BN(F(x) + x)
    = γ · (F(x) + x - μ) / σ + β

不是恒等映射！
```

#### Configuration C: ReLU before addition

```
x → BN → ReLU → Conv → BN → ReLU → Conv → ReLU → (+x)
```

**特点**：
- 第二个卷积后有 ReLU
- 相加前 ReLU
- 恒等路径仍然被阻断

**评分**：★★☆☆☆

#### Configuration D: Full Pre-activation (Winner!) ✨

```
x → BN → ReLU → Conv → BN → ReLU → Conv → (+x)
```

**特点**：
- 激活在卷积之前
- 相加后没有任何操作
- **恒等路径完全干净**

**评分**：★★★★★

### 为什么 BN → ReLU → Conv 顺序？

**传统顺序**：Conv → BN → ReLU
```
原因:
  1. BatchNorm 最初设计在 Conv 之后
  2. VGG, GoogLeNet 都用这个顺序
```

**Pre-activation 顺序**：BN → ReLU → Conv
```
优势:
  1. 恒等路径干净
  2. BN 作为正则化（在 Conv 之前）
  3. 激活作为预处理
  4. 信号在进入 Conv 前已经被规范化
```

**实验结果**：
```
架构               CIFAR-10 错误率
────────────────────────────────
Original (Conv-BN-ReLU):    6.7%
Pre-activation (BN-ReLU-Conv): 5.8%  ← 更好！
```

---

## 🧪 实验结果

### 1. 网络深度极限测试

**问题**：原始 ResNet 能训练多深？

```
原始 ResNet:
  最大深度: ~152 层
  超过后训练非常困难

Pre-activation ResNet:
  最大深度: 1001 层！✨
  200 层训练稳定
  1000+ 层仍然可训练
```

### 2. CIFAR-10 结果

```
架构                   深度    错误率 (%)
────────────────────────────────────────
ResNet-original         110     6.43
ResNet-preactivation    110     5.71  ← 提升 0.72%
ResNet-original       1001     N/A    (无法训练)
ResNet-preactivation  1001     7.54  ← 成功训练！
```

**关键发现**：
```
110 层:
  Pre-activation 比 original 好 0.72%

1001 层:
  Original 完全无法训练
  Pre-activation 成功训练！
```

### 3. ImageNet 结果

```
架构                   深度    Top-1 错误 (%)
──────────────────────────────────────────
ResNet-original          50       24.7
ResNet-preactivation     50       24.2  ← 更好
ResNet-original         152       23.1
ResNet-preactivation    152       22.8  ← 更好
ResNet-preactivation    200       22.6  ← 更深更好
```

**观察**：
```
1. Pre-activation 一致地优于 original
2. 200 层比 152 层更好（没有退化）
3. 证明了恒等映射的重要性
```

### 4. 激活函数分析

论文还测试了不同激活函数：

```
激活函数          CIFAR-10 错误率 (%)
────────────────────────────────
ReLU                 5.71
Leaky ReLU           5.85
PReLU                6.02
ELU                  6.15

结论: ReLU 仍然是最优选择
```

---

## 💡 核心洞察

### 1. 恒等映射的重要性

```
问题: 为什么恒等映射如此重要？

回答:
  如果最优映射是 identity
  残差学习 F(x) = H(x) - x 应该学习 F(x) = 0

  但如果架构阻断了恒等路径
  即使 F(x) = 0, 输出也不是 identity
  → 网络无法"退化为"恒等映射
  → 优化困难
```

### 2. 架构细节的影响

```
Kaiming He 的洞察:

"小改动可以有大的影响"

将 ReLU 从相加后移到相加前:
  → 代码只改几行
  → 但能训练 1000+ 层网络
  → 性能提升 0.5-1%

教训: 深度学习中，细节很重要！
```

### 3. 梯度流动的干净路径

```
原始 ResNet:
  梯度路径: [输出] → ReLU → [+操作] → [输入]
           ↑ 阻断点

Pre-activation:
  梯度路径: [输出] → [+操作] → [输入]
           ↑ 完全畅通

关键: 设计网络时，要为梯度提供"高速公路"
```

### 4. BN 作为正则化

```
传统观点:
  BN 用于加速收敛、稳定训练

新见解 (Pre-activation):
  BN 在 Conv 之前
  → BN 改变输入分布
  → 对后续 Conv 起正则化作用
  → 防止过拟合
```

---

## 🛠️ 实践指南

### 何时使用 Pre-activation？

```
使用 Pre-activation (ResNet v2):
  ✓ 网络很深 (>50 层)
  ✓ 训练不稳定
  ✓ 追求极致性能

使用 Original (ResNet v1):
  ✓ 网络较浅 (<50 层)
  ✓ 向后兼容性重要
  ✓ 实现简单优先
```

### PyTorch 实现

```python
import torch.nn as nn

class PreActBlock(nn.Module):
    """Pre-activation ResNet block"""
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 第一个 pre-activation 层
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, stride=stride,
                              padding=1, bias=False)

        # 第二个 pre-activation 层
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                              kernel_size=3, stride=1,
                              padding=1, bias=False)

        # Shortcut 连接（维度不匹配时需要投影）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels,
                                     kernel_size=1, stride=stride,
                                     bias=False)

    def forward(self, x):
        # Pre-activation: BN → ReLU → Conv
        out = self.relu1(self.bn1(x))
        out = self.conv1(out)

        # Pre-activation: BN → ReLU → Conv
        out = self.relu2(self.bn2(out))
        out = self.conv2(out)

        # 残差连接（相加后没有激活！）
        out += self.shortcut(x)

        return out

class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_channels = 64

        # 第一个卷积（使用普通顺序，没有残差）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # 残差层
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        # 输出层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # 第一层（普通卷积）
        out = self.relu(self.bn1(self.conv1(x)))

        # 残差层
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # 输出
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

# 使用
def PreActResNet18():
    return PreActResNet(PreActBlock, [2, 2, 2, 2])

def PreActResNet34():
    return PreActResNet(PreActBlock, [3, 4, 6, 3])

def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3])
```

### TensorFlow/Keras 实现

```python
import tensorflow as tf
from tensorflow.keras import layers

class PreActBlock(layers.Layer):
    def __init__(self, filters, stride=1):
        super().__init__()
        self.bn1 = layers.BatchNormalization()
        self.relu1 = layers.ReLU()
        self.conv1 = layers.Conv2D(filters, 3, strides=stride,
                                   padding='same', use_bias=False)

        self.bn2 = layers.BatchNormalization()
        self.relu2 = layers.ReLU()
        self.conv2 = layers.Conv2D(filters, 3, strides=1,
                                   padding='same', use_bias=False)

        self.shortcut = layers.Conv2D(filters, 1, strides=stride,
                                     use_bias=False)

    def call(self, x, training=False):
        # Pre-activation
        out = self.relu1(self.bn1(x, training=training))
        out = self.conv1(out)

        out = self.relu2(self.bn2(out, training=training))
        out = self.conv2(out)

        # Residual connection (no activation after!)
        out = out + self.shortcut(x)

        return out
```

### 训练技巧

1. **学习率调度**：
   ```python
   # Step decay
   lr = 0.1 * (0.1 ** (epoch // 30))
   ```

2. **权重初始化**：
   ```python
   # Kaiming 初始化
   nn.init.kaiming_normal_(conv.weight, mode='fan_out',
                          nonlinearity='relu')
   ```

3. **数据增强**：
   ```python
   # 对 ImageNet 很重要
   transforms.Compose([
       transforms.RandomResizedCrop(224),
       transforms.RandomHorizontalFlip(),
       transforms.ColorJitter(brightness=0.4, contrast=0.4,
                              saturation=0.4, hue=0.1),
       transforms.ToTensor(),
       transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
   ])
   ```

---

## 🔗 与其他论文的关联

### 前置论文

1. **Paper 10: ResNet (Original)**
   - 引入残差连接
   - Paper 15 是其改进版本

2. **Paper 07: AlexNet**
   - 都使用 ReLU
   - 但 ReLU 位置不同

### 后续影响论文

1. **ResNeXt (2017)**
   - 使用 pre-activation
   - 加入分组卷积

2. **Wide ResNet (2016)**
   - 探索宽度 vs 深度
   - 使用 pre-activation

3. **DenseNet (2017)**
   - 受 pre-activation 启发
   - 所有层都用 BN-ReLU-Conv

4. **ResNet v3 (2017)**
   - 进一步改进
   - 引入 squeeze-and-excitation

---

## 🧪 实践挑战

### 基础练习

1. **实现 Pre-activation Block**:
   ```python
   class PreActBlock:
       def __init__(self, in_channels, out_channels):
           # TODO: 初始化 BN, ReLU, Conv
           pass

       def forward(self, x):
           # TODO: 实现 BN → ReLU → Conv
           pass
   ```

2. **对比实验**:
   ```python
   def compare_blocks():
       # Original vs Pre-activation
       # 测试梯度流动
       # 测试训练稳定性
       pass
   ```

3. **可视化激活**:
   - 绘制不同层的激活分布
   - 对比两种架构

### 进阶练习

1. **实现完整 Pre-activation ResNet**:
   ```python
   class PreActResNet:
       def __init__(self, depth, num_classes):
           # TODO: 支持不同深度
           pass
   ```

2. **测试极端深度**:
   - 训练 200 层网络
   - 训练 500 层网络
   - 分析训练动态

3. **研究其他归一化**:
   - Group Norm
   - Layer Norm
   - 替换 Batch Norm

### 研究方向

1. **Skip Connection 的其他设计**:
   - 稠密连接 (DenseNet)
   - 分组连接
   - 注意力连接

2. **自适应架构**:
   - 动态深度
   - DropBlock 训练

3. **理论分析**:
   - 为什么 pre-activation 更好？
   - 恒等映射的理论保证

---

## ❓ 常见问题

### Q1: ResNet v1 vs ResNet v2 的区别？

| 特性 | ResNet v1 (2015) | ResNet v2 (2016) |
|------|------------------|------------------|
| **论文** | Paper 10 | Paper 15 |
| **激活位置** | Conv 之后 | Conv 之前 |
| **恒等路径** | 被 ReLU 阻断 | 完全干净 |
| **可训练深度** | ~152 层 | 1000+ 层 |
| **性能** | 优秀 | 更好 |
| **流行度** | 更常用 | 深度网络首选 |

### Q2: 为什么 BN → ReLU → Conv 更好？

```
原因 1: 恒等路径干净
  x → (+x)  (没有修改)

原因 2: BN 作为正则化
  在 Conv 之前规范化输入

原因 3: 激活作为预处理
  ReLU 后信号非负
  更适合卷积操作

原因 4: 梯度流动更好
  反向传播时没有阻断
```

### Q3: 实际应该用哪个版本？

```
推荐策略:

1. 如果深度 < 50 层:
   → 使用 ResNet v1 (original)
   → 实现简单，性能足够

2. 如果深度 > 50 层:
   → 使用 ResNet v2 (pre-activation)
   → 训练更稳定

3. 如果追求极致性能:
   → 使用 ResNet v2
   → 或 ResNeXt, SE-ResNet

4. 如果迁移学习:
   → 使用预训练的 ResNet v1
   → 模型 zoo 更多 v1 模型
```

### Q4: Pre-activation 有什么缺点？

```
缺点 1: 不太直观
  传统的 Conv-BN-ReLU 更符合直觉

缺点 2: 第一层处理特殊
  第一个卷积不能用 pre-activation
  需要单独处理

缺点 3: 调试稍复杂
  激活在卷积前，可能不习惯

缺点 4: 生态支持较少
  大部分预训练模型是 v1
```

---

## 📝 学习检查清单

完成以下任务以确保掌握 Pre-activation ResNet：

- [ ] 理解原始 ResNet 的问题
- [ ] 推导 Pre-activation 的梯度流动
- [ ] 实现 BN → ReLU → Conv 顺序
- [ ] 对比 Post-activation vs Pre-activation
- [ ] 理解恒等映射的重要性
- [ ] 训练 > 100 层的网络
- [ ] 分析不同深度的性能
- [ ] 可视化梯度流动
- [ ] 阅读 DenseNet 论文（相关概念）
- [ ] 在实际任务中使用 Pre-activation

---

## 🔗 延伸阅读

### 必读论文

1. **Identity Mappings in Deep Residual Networks (ECCV 2016)**
   - Kaiming He et al.
   - [arXiv:1603.05027](https://arxiv.org/abs/1603.05027)

2. **Deep Residual Learning for Image Recognition (CVPR 2016)**
   - 原始 ResNet 论文
   - 见 Paper 10

3. **Aggregated Residual Transformations for Deep Neural Networks (CVPR 2017)**
   - ResNeXt
   - [arXiv:1611.05431](https://arxiv.org/abs/1611.05431)

4. **Densely Connected Convolutional Networks (CVPR 2017)**
   - DenseNet，受 Pre-activation 启发
   - [arXiv:1608.06993](https://arxiv.org/abs/1608.06993)

### 相关资源

- **PyTorch 实现**:
  ```python
  import torchvision.models as models
  # ResNet v1
  resnet50 = models.resnet50(pretrained=True)
  ```

- **TensorFlow 实现**:
  ```python
  from tensorflow.keras.applications import ResNet50
  model = ResNet50(weights='imagenet')
  ```

- **官方代码**:
  - Facebook Research: [https://github.com/facebookresearch/fb.resnet.torch](https://github.com/facebookresearch/fb.resnet.torch)

---

## 🎯 核心要点回顾

1. **问题识别**:
   ```
   原始 ResNet: ReLU 在相加之后
   → 阻断恒等路径
   → 梯度流动受阻
   → 深度网络训练困难
   ```

2. **解决方案**:
   ```
   Pre-activation: BN → ReLU → Conv
   → 激活在卷积之前
   → 相加后没有激活
   → 恒等路径完全干净
   ```

3. **数学保证**:
   ```
   y = F(x) + x

   如果 F(x) = 0:
     输出 = x (真正的恒等映射)

   梯度: ∂L/∂x = ∂L/∂y · (∂F/∂x + 1)
   → 恒等项恒为 1
   → 梯度永不消失
   ```

4. **实验验证**:
   - 1001 层网络成功训练
   - 一致的性能提升
   - 更稳定的训练

5. **设计原则**:
   - 为梯度提供干净路径
   - 恒等映射应该真正恒等
   - 小改动可以有大影响

6. **实践建议**:
   - 深度网络用 Pre-activation
   - 浅层网络可以 Original
   - 迁移学习常用 v1

**Paper 15 展示了如何通过理解架构细节来改进深度网络，是"理论指导实践"的典范。**

---

*"We present a clean and trainable architecture that enables us to train extremely deep networks... The improvement is by no means trivial, considering that it is achieved by a simple architectural redesign."*
*— Kaiming He et al., 2016*
