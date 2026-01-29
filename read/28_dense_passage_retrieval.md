# Paper 28: Dense Passage Retrieval (DPR) for Open-Domain QA

**论文标题**: Dense Passage Retrieval for Open-Domain Question Answering
**作者**: Vladimir Karpukhin, Barlas Oğuz, Sewon Min, et al. (Meta AI, 2020)
**类型**: 检索架构 / 信息检索

---

## 📚 问题背景与核心挑战

### 开放域问答 (Open-Domain QA)

```

传统问题:

用户问题: "法国的首都是哪里？"

传统方案 (Pipeline):
  1. 实体识别: 提取 "法国"
  2. 知识库查询: "法国" → "巴黎"
  3. 关系推理: "首都" → 位置
  4. 答案合成: "巴黎"

问题:
  - 需要多个模块
  - 错误累积
  - 难以端到端训练
  - 无法扩展到新领域
```

### DPR 的解决方案

```

DPR (Dense Passage Retrieval):

核心思想:
  将 QA 简化为检索问题

两阶段:
  1. 检索 (Retrieval):
     找到相关文档段落
     → Dense Passage Retrieval

  2. 阅读 (Reading):
     从段落提取答案
     → Extractive QA (如 BERT)

本论文聚焦第 1 阶段！
```

### 检索范式演变

```

稀疏检索 (Sparse Retrieval):

  BM25, TF-IDF
  关键词匹配
  词频统计

缺点:
  ✗ 依赖词汇精确匹配
  ✗ 无法处理语义相似性
  ✗ "car" ≠ "automobile"
  ✗ 无法处理同义词

密集检索 (Dense Retrieval):

  DPR, ColBERT
  语义嵌入
  神经网络编码

优点:
  ✓ 语义匹配
  ✓ 处理同义词/改写
  ✓ 端到端训练
```

---

## 🏗️ DPR 架构

### 双编码器 (Dual Encoder)

```

架构:

Query 编码器: Question → BERT → q_emb
  - 编码问题
  - 产生密集向量

Passage 编码器: Passage → BERT → p_emb
  - 编码文档段落
  - 产生密集向量

相似度: sim(q, p) = q_emb · p_emb
  - 点积 (余弦相似度，因为向量已归一化)
```

### 详细结构

```python
# 伪代码

class DPR:
    def __init__(self):
        # 两个独立的 BERT
        self.question_encoder = BERT()
        self.passage_encoder = BERT()

    def encode_question(self, question):
        # [CLS] question [SEP]
        # 提取 [CLS] token 表示作为向量
        return self.question_encoder(question)

    def encode_passage(self, passage):
        # [CLS] passage [SEP]
        return self.passage_encoder(passage)

    def retrieve(self, question, passages, k=10):
        q_emb = self.encode_question(question)  # (768,)
        p_embs = self.encode_passages(passages)    # (N, 768)

        # 计算相似度
        similarities = p_embs @ q_emb  # (N,)

        # 返回 top-k
        return top_k(similarities)
```

### 关键设计决策

```

1. 为什么用 BERT?
   - 预训练语言模型
   - 理解上下文
   - 丰富表示

2. 为什么用 [CLS] token?
   - BERT 设计用于分类
   - 捕捉序列整体表示
   - 对 QA 任务有效

3. 为什么点积而不是余弦?
   - 向量已 L2 归一化
   - 点积 = 余弦相似度
   - 更快的 MIPS (最大内积搜索)

4. 为什么不用交叉编码?
   - 简单解耦
   - 易于扩展
   - 支持离线索引
```

---

## 🔍 训练目标

### 对比损失 (Contrastive Loss)

```

InfoNCE (Noise Contrastive Estimation):

给定:
  - 问题 q
  - 正样本 p⁺ (正确段落)
  - 负样本 p⁻ (不相关段落)

损失:

L = -log [ exp(sim(q, p⁺)) / (exp(sim(q, p⁺)) + Σⱼ exp(sim(q, pⱼ))) ]

直观:
  - 正样本相似度应该高
  - 负样本相似度应该低
  - 使用 softmax 归一化
```

### 负样本策略

```

In-Batch Negatives:

批次: [(q₁, p₁⁺), (q₂, p₂⁺), ..., (q_B, p_B⁺)]

对于 q₁:
  正样本: p₁⁺
  负样本: p₂⁺, p₃⁺, ..., p_B⁺

优势:
  ✓ 无需额外数据
  ✓ 每个批次都有负样本
  ✓ 梯度流经所有样本
  ✗ 批次大小影响负样本质量

Hard Negatives:
  - BM25 top 结果 (但不是正确答案)
  - 随机采样
  - 跨批次负样本
```

### 训练流程

```

输入:
  - QA 对: (question, passage, label)
  - label: 是否相关

训练步骤:

1. 随机采样 QA 对
2. 对于每个问题:
   - 正样本: 标记为相关的段落
   - 负样本: 批次中的其他段落 (in-batch)
   - BM25 top 结果 (hard negatives)
   - 随机段落
3. 计算对比损失
4. 反向传播更新编码器
5. 重复多个 epoch
```

---

## 🎯 推理: MIPS (Maximum Inner Product Search)

### 检索效率

```

暴力搜索 (Brute Force):

查询时间:
  - 编码查询: O(1)
  - 比较 N 个段落: O(N)
  - 总计: O(N)

问题:
  - N 可能很大 (百万级)
  - 每个查询都慢

解决方案: MIPS
```

### MIPS 原理

```

Maximum Inner Product Search:

目标: 找到 max p q · p_emb

优化: 预处理段落向量
  - 离线构建索引
  - 近似最近邻搜索
  - FAISS, HNSW, Annoy

流程:
  离线:
    1. 编码所有段落: p_embs (N, 768)
    2. 构建索引 (FAISS)
    3. 保存到磁盘

  在线:
    1. 编码问题: q_emb (768,)
    2. 搜索索引: top-k 最相似
    3. 返回段落 ID

复杂度:
  - 离线: O(N log N) 构建索引
  - 在线: O(log N) 每个查询
```

### FAISS 简介

```

FAISS (Facebook AI Similarity Search):

功能:
  - 快速最近邻搜索
  - 支持 L2 距离、点积
  - CPU/GPU 加速

算法:
  - 索引量化 (Quantization)
  - 聚类 (Clustering)
  - HNSW (层次可导航小世界图)

性能:
  - 百万级向量: 毫秒级搜索
  - 十亿级向量: 亚秒级搜索
  - 理论无限扩展
```

---

## 📊 实验结果

### 数据集

```

来源: 自然语言生成数据

数据集:
  - Natural Questions
  - WebQuestions
  - TriviaQA
  - CuratedTREC

训练集:
  - 问题: 59k QA 对
  - 段落: 21M 文本段落

验证集:
  - 问题: 6k
  - 段落: 7k

测试集:
  - 问题: 7k
  - 段落: 7k

特点:
  - 真实用户问题
  - Wikipedia 段落
  - 多样化主题
```

### 评估指标

```

Recall@k:
  准确答案在前 k 个结果中的比例

  Recall@1: 精确答案排名第 1
  Recall@5: 精确答案在前 5 个

MRR (Mean Reciprocal Rank):
  平均倒数排名

  MRR = mean(1/rank_of_correct_answer)

例子:
  正确答案排名第 2: MRR = 1/2 = 0.5
  正确答案排名第 5: MRR = 1/5 = 0.2
```

### DPR vs BM25

```

数据集         BM25   DPR    改善
---------------------------------------
Natural Q     59.1   78.4    +19.3%
WebQuestions  55.0   75.0    +20.0%
TriviaQA      66.7   68.0    +1.3%
TREC-CAR      70.9   79.4    +8.5%

平均改善: +12.5%

关键洞察:
  ✓ DPR 在所有数据集上优于 BM25
  ✓ 改善幅度因数据集而异
  ✓ 语义匹配 > 词汇匹配
```

### 消融实验

```

架构选择:

编码器:
  - BoW (词袋): 基线
  - TF-IDF: 强基线
  - BiLSTM: 中等
  - BERT: 最佳

嵌入位置:
  - [CLS] token: 最佳
  - Mean pooling: 中等
  - Max pooling: 较差

相似度:
  - 点积: 最佳 (配合归一化)
  - 余弦: 相同 (配合归一化)
  - 欧氏距离: 较差

训练目标:
  - 对比损失: 最佳
  - Triplet loss: 中等
  - 分类损失: 较差
```

---

## 🔄 架构演进

### DPR 基础

```

原始 DPR (2020):

编码器:
  - BERT-base (110M 参数)
  - 单向
  - [CLS] token 表示

训练:
  - 对比损失
  - In-batch 负样本
  - BM25 hard negatives
```

### ColBERT (2020)

```

改进: Late Interaction

核心思想:
  - DPR 比较早期融合 (点积)
  - ColBERT 比较所有 token 对

架构:
  - 逐层计算 token 相似度
  - 深层特征交互
  - 最后聚合

效果:
  ✓ 更好的排序
  ✗ 更慢推理
  ✗ 更复杂
```

### ANCE (2020)

```

改进: 近似最近邻负样本

问题:
  - In-batch 负样本可能不够"难"
  - 对简单任务足够, 对难任务不足

解决:
  - 使用跨批次负样本
  - 难机抽样 + BM25
  - 构建全局难负样本池

效果:
  ✓ 收敛更快
  ✓ 最终性能更好
```

---

## 🔧 实现细节

### BERT 编码器

```python
import torch
from transformers import BertModel, BertTokenizer

class DenseRetriever:
    def __init__(self):
        self.model_name = 'bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(self.model_name)

    def encode_passages(self, passages):
        """编码多个段落"""
        # Tokenize
        encoded = self.tokenizer(
            passages,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        # Encode
        with torch.no_grad():
            outputs = self.model(**encoded)
            # 使用 [CLS] token (第一个)
            embeddings = outputs.last_hidden_state[:, 0, :]  # (B, 768)

        # L2 归一化
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings

    def encode_questions(self, questions):
        """编码问题"""
        return self.encode_passages(questions)
```

### 训练代码

```python
def train_dpr(model, train_dataloader, optimizer, epochs):
    model.train()

    for epoch in range(epochs):
        for batch in train_dataloader:
            questions = batch['question']
            passages = batch['passage']
            labels = batch['label']  # 1 if relevant else 0

            # 编码
            q_emb = model.encode_questions(questions)    # (B, 768)
            p_emb = model.encode_passages(passages)      # (B, 768)

            # 正样本相似度
            pos_scores = torch.sum(q_emb * p_emb, dim=1)  # (B,)

            # 负样本 (in-batch)
            # 对每个问题, 使用批次中其他段落作为负样本
            batch_size = q_emb.size(0)
            neg_scores = []

            for i in range(batch_size):
                # 找到其他所有段落的索引
                neg_indices = [j for j in range(batch_size) if j != i]

                # 计算负样本相似度
                neg_p_emb = p_emb[neg_indices]  # (B-1, 768)
                neg_score = torch.sum(q_emb[i:i+1] * neg_p_emb, dim=1)  # (B-1,)

                neg_scores.append(neg_score)

            neg_scores = torch.stack(neg_scores, dim=1)  # (B, B-1)

            # 拼接正负分数
            all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # (B, B)

            # 计算对比损失
            # 正确答案应该在索引 i 的位置
            target_indices = torch.arange(batch_size)

            # 提取正确分数
            correct_scores = all_scores[torch.arange(batch_size), target_indices]

            # Softmax
            exp_scores = torch.exp(all_scores)
            sum_exp_scores = torch.sum(exp_scores, dim=1, keepdim=True)
            probs = exp_scores / sum_exp_scores

            # 负对数似然
            log_prob = torch.log(probs[torch.arange(batch_size), target_indices] + 1e-8)

            loss = -log_prob.mean()

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

### 推理代码

```python
def retrieve_faiss(query, passage_embeddings, index, k=10):
    """
    使用 FAISS 进行 MIPS 检索

    Args:
        query: (768,) 问题嵌入
        passage_embeddings: (N, 768) 段落嵌入
        index: FAISS 索引
        k: 返回结果数量

    Returns:
        top_k_indices: (k,) 段落 ID
    """
    # FAISS 需要 float32, L2 归一化
    query = query.astype('float32').reshape(1, -1)
    passages = passage_embeddings.astype('float32')

    # 搜索
    scores, indices = index.search(query, k)  # (1, k)
    scores = scores[0]  # (k,)
    indices = indices[0]  # (k,)

    return indices, scores
```

---

## 💡 核心洞察

### 1. 稠疏 vs 密集检索

```

BM25 (稀疏):
  关键词匹配
  "法国" ≠ "巴黎"
  "首都" ≠ "位置"

DPR (密集):
  语义相似度
  "法国" ≈ "巴黎" (上下文相关)
  "首都" ≈ "位置" (语义相关)

关键:
  ✓ DPR 捕捉语义
  ✗ BM25 精确匹配更好
```

### 2. 表示学习

```

端到端训练好处:

1. 学习任务特定表示:
   - QA 需要"相关性"理解
   - 不同于一般语言模型

2. 利用上下文:
   - 问题提供查询意图
   - 段落提供候选答案
   - 训练信号明确

3. 优化排序:
   - 不只是相关性
   - 考虑精确度
   - 优化检索排序
```

### 3. 负样本挖掘

```

In-Batch Negatives:

优点:
  ✓ 高效利用数据
  ✓ 自然提供困难样本
  ✓ 训练稳定

Hard Negatives:
  ✓ BM25 top results
  ✓ 语义不相关
  ✓ 避免伪负样本

最佳策略:
  - In-batch (基础)
  - BM25 top (困难负样本)
  - 随机 (多样性)
```

### 4. 索引构建

```

离线索引构建:

1. 编码所有段落:
   passages → embeddings (N, 768)

2. 构建索引:
   使用 FAISS 构建 MIPS 索引
   - 支持添加新段落
   - 支持删除旧段落

3. 更新策略:
   - 增量更新
   - 定期重建索引
```

---

## 🌐 应用领域

### 1. 开放域问答

```

传统搜索引擎:

查询: "Python 列表推导式"

流程:
  1. 关键词提取: "Python", "list", "derivation"
  2. 检索: 匹配关键词
  3. 排序: BM25 分数
  4. 提取答案

DPR:

流程:
  1. 编码: "Python list comprehension derivation"
  2. 编码: Wikipedia 段落
  3. 检索: 语义匹配
  4. 提取答案

优势:
  ✓ 理解查询意图
  ✓ 匹配语义相似
  ✓ 处理同义词/改写
```

### 2. RAG (Retrieval-Augmented Generation)

```

RAG 系统架构:

用户问题
  ↓
1. 检索相关文档 (DPR)
  ↓
2. 拼接到 prompt (如 "基于以下文档回答:")
  ↓
3. LLM 生成答案
  ↓
4. 返回答案 + 引用

关键:
  - DPR 提供相关上下文
  - 提高答案准确性
  - 减少幻觉

应用:
  - ChatGPT + 知识库
  - 企业知识助手
  - 文档问答
```

### 3. 语义搜索

```

传统搜索:
  "关键词 AND 匹配"
  "布尔查询"

DPR 语义搜索:
  "机器学习算法"

检索结果:
  - 支持语义相似的文档
  - 不依赖精确关键词
  - 发现隐含关联

应用:
  - 学术论文搜索
  - 专利检索
  - 法律文档搜索
```

### 4. 推荐系统

```

内容推荐:

用户 → DPR 编码历史 → 查询

物品:
  - 文章、视频
  - 商品描述
  - 用户生成内容

匹配:
  用户兴趣向量 vs 物品向量
  - 语义相似度

推荐:
  Top-k 最相关物品
```

---

## 🔗 与其他技术的联系

### 前置技术

```

1. TF-IDF / BM25:
  稀疏检索
  - 基线方法
  - DPR 的参照

2. Word2Vec / GloVe:
  词嵌入
  - 密集向量
  - DPR 的前身

3. Siamese Networks:
  度量学习
  - 成对损失
  - 相似架构

4. BERT:
  语言模型预训练
  - 上下文编码
  - DPR 使用其编码器
```

### 后续发展

```

1. ColBERT (2020):
  - Late Interaction
  - 更细粒度匹配
  - 更好的排序

2. ANCE (2020):
  - 跨批次负样本
  - 更难负样本
  - 收敛更快

3. RocketQA (2020):
  - 多轮推理
  - 对话式 QA

4. Contriever (2021):
  - 无监督训练
  - 大规模预训练
```

---

## 📝 实践指南

### 训练建议

```

数据准备:
  ✓ 高质量 QA 对
  ✓ 负样本多样性
  ✗ 避免假负样本

模型配置:
  ✓ BERT-base (开始)
  ✓ 尝试 BERT-large (优化)
  ✗ 不要从头训练 (太贵)

超参数:
  学习率: 1e-5 到 5e-5
  批次大小: 64-128
  训练轮次: 30-40
  温度: 超参数: 0.1
```

### 推理优化

```

加速检索:

1. FAISS 索引:
   - IVF + PQ (倒排文件 + 乘积量化)
   - 压缩: 100× 加速
   - 精度: <1% 下降

2. 层次索引:
   - 粗到细
   - 先快速过滤
   - 后精确排序

3. GPU 加速:
   - 大批次编码
   - GPU 搜索
```

### 评估技巧

```

指标:

Recall@k:
  - 准确答案在前 k 个结果中
  - 主要指标

MRR:
  - 平均倒数排名
  - 考虑排名位置

NDCG:
  - 归一化折损增益
  - 考虑排序质量

策略:
  - 主指标: Recall@k (通常 k=10 或 100)
  - 辅助: MRR
  - 调试: NDCG
```

---

## ⚠️ 常见陷阱

### 1. 过度拟合

```

现象:
  训练集表现好
  测试集表现差

原因:
  - 训练数据有限
  - 特定领域模式
  - 词汇偏差

解决:
  ✓ 数据增强
  ✓ 正则化
  ✓ Dropout
  ✓ 早停
```

### 2. 负样本问题

```

伪负样本:

现象:
  某些段落看起来不相关
  - 实际相关
  - 导致训练冲突

解决:
  ✓ 人工筛选
  ✓ 困难负样本挖掘
  ✓ 降低负样本权重
```

### 3. 计算资源

```

内存需求:

编码 100 万段落:
  - 段落: 768 维 × 4 bytes × 1M
  - 需要 ~3GB 内存

索引构建:
  - FAISS 索引: ~10GB
  - RAM: 需要足够内存

解决:
  - 批处理编码
  - 压缩索引
  - 分布式部署
```

### 4. 基线陷阱

```

BM25 很强！

现象:
  DPR 在某些任务上提升有限

原因:
  - BM25 已经优化很好
  - 任务依赖关键词

策略:
  ✓ 混合 BM25 + DPR
  ✓ BM25 候选, DPR 重排序
  ✓ 两阶段检索
```

---

## 🎓 核心要点回顾

1. **核心思想**:
   ```
   端到端训练密集检索
   ```

2. **架构**:
   ```
   双编码器 + 点积相似度
   ```

3. **训练**:
   ```
   对比损失 + In-batch 负样本
   ```

4. **推理**:
   ```
   MIPS (FAISS) + 离线索引
   ```

5. **评估**:
   ```
   Recall@k, MRR
   ```

6. **优势**:
   ```
   语义匹配 > 词汇匹配
   ```

7. **限制**:
   ```
   需要训练数据
   内存密集
   索引更新成本
   ```

8. **应用**:
   ```
   开放域 QA、RAG、语义搜索
   ```

9. **演进**:
   ```
   DPR → ColBERT → ANCE → Contriever
   ```

10. **实践**:
    ```
    FAISS 索引
    混合检索
    超参数调优
    ```

---

**Dense Passage Retrieval 是开放域问答系统的重大突破。通过学习密集向量表示，DPR 实现了语义级别的文档检索，显著超越了传统的 BM25 稀疏检索方法。这为现代 RAG 系统和语义搜索引擎奠定了基础。**

*"Retrieval is the bottleneck of open-domain QA. Dense retrieval enables semantic matching beyond simple keyword overlap."*
*— Karpukhin et al., 2020*

*"DPR shows that dense embeddings trained with contrastive objectives can outperform decades of sparse retrieval heuristics."*
*— Meta AI Research*
