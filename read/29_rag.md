# Paper 29: Retrieval-Augmented Generation (RAG)
## 知识密集型任务的检索增强生成

**Paper**: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
**Authors**: Patrick Lewis, Ethan Perez, Aleksandra Piktus, et al. (Meta AI, 2020)
**Notebook**: `29_rag.ipynb`

---

## 核心思想 / Core Idea

**问题** / The Problem:
- 参数模型的知识受限于训练数据
- Parametric models have limited knowledge from training data
- 更新知识需要重新训练
- Updating knowledge requires retraining

**解决方案** / The Solution:
- **RAG**: 检索增强生成
- **RAG**: Retrieval-Augmented Generation
- 结合密集检索(DPR) + 序列生成(BART)
- Combines dense retrieval (DPR) + seq2seq generation (BART)
- 两全其美:外部知识 + 强大生成能力
- Best of both worlds: external knowledge + powerful generation

**关键创新** / Key Innovation:
- 端到端可微分
- End-to-end differentiable
- 检索器和生成器联合训练
- Joint training of retriever and generator
- 非参数知识(检索) + 参数知识(生成)
- Non-parametric knowledge (retrieval) + parametric knowledge (generation)

---

## 架构概览 / Architecture Overview

```
输入查询 (Input Query x)
    ↓
检索器 (Retriever - DPR)
    ↓
Top-k 文档 (Top-k Documents z)
    ↓
生成器 (Generator - BART)
    ↓
输出答案 (Output Answer y)
```

**两个变种** / Two Variants:
1. **RAG-Sequence**: 对整个序列在文档上边缘化
2. **RAG-Sequence**: Marginalize over documents for entire sequence
3. **RAG-Token**: 对每个token在文档上边缘化
4. **RAG-Token**: Marginalize over documents for each token

---

## 第一部分: 检索器 / Part 1: The Retriever

### 1.1 密集检索 / Dense Retrieval

RAG使用类似DPR的密集检索器:

**查询编码** / Query Encoding:
```python
q_emb = E_Q(x)  # BERT encoder
```

**文档编码** / Document Encoding:
```python
d_emb = E_D(z)  # BERT encoder
```

**检索概率** / Retrieval Probability:
```
P(z|x) ∝ exp(q_emb · d_emb)
```

**特点** / Characteristics:
- 使用双编码器架构
- Dual encoder architecture
- 预先索引所有文档
- Pre-index all documents
- 查询时快速检索
- Fast retrieval at query time

### 1.2 简化实现 / Simplified Implementation

```python
import numpy as np

def softmax(x):
    """Softmax激活函数"""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

class SimpleRetriever:
    """简化的密集检索器(类似DPR)"""

    def __init__(self, embedding_dim):
        self.embedding_dim = embedding_dim
        # 查询编码器权重
        self.query_encoder_W = np.random.randn(embedding_dim, embedding_dim) * 0.01

    def encode_query(self, query_tokens):
        """将查询编码为密集向量"""
        # 简化: 使用随机投影
        query_vec = np.mean(query_tokens, axis=0)
        encoded = np.dot(self.query_encoder_W, query_vec)
        # L2归一化
        return encoded / (np.linalg.norm(encoded) + 1e-8)

    def retrieve(self, query_embedding, document_embeddings, k=5):
        """
        检索top-k文档

        参数:
            query_embedding: 查询向量
            document_embeddings: 所有文档的嵌入 [N, embedding_dim]
            k: 返回文档数量

        返回:
            indices: 文档索引
            probs: 检索概率
        """
        # 计算相似度(点积)
        similarities = np.dot(document_embeddings, query_embedding)

        # 获取top-k
        top_k_indices = np.argsort(similarities)[::-1][:k]
        top_k_scores = similarities[top_k_indices]

        # 转换为概率
        probs = softmax(top_k_scores)

        return top_k_indices, probs

# 使用示例
embedding_dim = 64
retriever = SimpleRetriever(embedding_dim)

# 虚拟数据
query_tokens = np.random.randn(10, embedding_dim)
document_embeddings = np.random.randn(20, embedding_dim)
# 归一化文档
document_embeddings = document_embeddings / (
    np.linalg.norm(document_embeddings, axis=1, keepdims=True) + 1e-8
)

# 检索
query_emb = retriever.encode_query(query_tokens)
top_indices, top_probs = retriever.retrieve(query_emb, document_embeddings, k=5)

print(f"检索到的文档: {top_indices}")
print(f"检索概率: {top_probs}")
print(f"概率和: {np.sum(top_probs):.4f}")
```

**输出示例** / Example Output:
```
检索到的文档: [12, 5, 18, 3, 9]
检索概率: [0.35, 0.25, 0.20, 0.12, 0.08]
概率和: 1.0000
```

---

## 第二部分: 生成器 / Part 2: The Generator

### 2.1 序列到序列模型 / Seq2Seq Model

RAG使用BART作为生成器:
- 输入: 查询 + 检索到的文档
- Input: Query + Retrieved Document
- 输出: 答案概率分布
- Output: Answer probability distribution

### 2.2 简化实现 / Simplified Implementation

```python
class SimpleGenerator:
    """简化的序列生成器(类似BART)"""

    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # 编码器
        self.encoder_W = np.random.randn(hidden_dim, embedding_dim) * 0.01

        # 解码器
        self.decoder_W = np.random.randn(hidden_dim, embedding_dim) * 0.01
        self.output_W = np.random.randn(vocab_size, hidden_dim) * 0.01

    def generate_prob(self, query_tokens, doc_tokens, target_tokens):
        """
        计算 P(y | x, z)

        参数:
            query_tokens: 查询token
            doc_tokens: 文档token
            target_tokens: 目标token

        返回:
            log_prob: 对数概率
        """
        # 编码查询 + 文档
        combined = np.concatenate([query_tokens, doc_tokens], axis=0)
        encoder_hidden = np.tanh(
            np.dot(self.encoder_W, np.mean(combined, axis=0))
        )

        # 解码目标
        log_prob = 0
        for target_token in target_tokens:
            decoder_hidden = np.tanh(
                np.dot(self.decoder_W, target_token)
            )

            # 结合编码器和解码器
            combined_hidden = encoder_hidden + decoder_hidden

            # 输出分布
            logits = np.dot(self.output_W, combined_hidden)
            probs = softmax(logits)

            # 计算目标token概率
            target_idx = np.argmax(target_token)  # 简化
            log_prob += np.log(probs[target_idx] + 1e-8)

        return log_prob

# 使用示例
vocab_size = 1000
generator = SimpleGenerator(vocab_size, embedding_dim, hidden_dim=128)

# 虚拟token(嵌入)
query = np.random.randn(5, embedding_dim)
doc = np.random.randn(20, embedding_dim)
target = np.random.randn(8, embedding_dim)

log_prob = generator.generate_prob(query, doc, target)
print(f"\nLog P(y | x, z): {log_prob:.4f}")
```

---

## 第三部分: RAG-Sequence / Part 3: RAG-Sequence

### 3.1 数学公式 / Mathematical Formulation

**核心思想** / Core Idea:
- 每个文档生成完整序列
- Each document generates full sequence
- 然后加权组合
- Then weighted combination

**公式** / Formula:
```
P_RAG-Seq(y|x) = Σ_z∈top-k P(z|x) · P(y|x,z)
```

其中:
- `P(z|x)`: 检索概率(来自retriever)
- `P(y|x,z)`: 生成概率(来自generator)

### 3.2 实现细节 / Implementation Details

```python
class RAGSequence:
    """RAG-Sequence模型"""

    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def forward(self, query_tokens, target_tokens,
                document_embeddings, documents_tokens, k=5):
        """
        RAG-Sequence前向传播

        P(y|x) = Σ_z P(z|x) * P(y|x,z)

        参数:
            query_tokens: 查询
            target_tokens: 目标答案
            document_embeddings: 所有文档嵌入
            documents_tokens: 所有文档token
            k: 检索文档数

        返回:
            log_prob: 对数概率
            doc_indices: 使用的文档索引
            doc_probs: 文档权重
        """
        # 检索文档
        query_emb = self.retriever.encode_query(query_tokens)
        doc_indices, doc_probs = self.retriever.retrieve(
            query_emb, document_embeddings, k=k
        )

        # 在文档上边缘化
        total_prob = 0

        for doc_idx, p_z_given_x in zip(doc_indices, doc_probs):
            # 获取文档token
            doc_tokens = documents_tokens[doc_idx]

            # P(y | x, z)
            log_p_y_given_xz = self.generator.generate_prob(
                query_tokens, doc_tokens, target_tokens
            )
            p_y_given_xz = np.exp(log_p_y_given_xz)

            # P(z|x) * P(y|x,z)
            total_prob += p_z_given_x * p_y_given_xz

        return np.log(total_prob + 1e-8), doc_indices, doc_probs

# 创建RAG-Sequence模型
rag_seq = RAGSequence(retriever, generator)

# 创建虚拟文档
num_docs = 20
documents_tokens = [
    np.random.randn(15, embedding_dim) for _ in range(num_docs)
]

# 测试
log_prob, used_docs, used_probs = rag_seq.forward(
    query_tokens=query,
    target_tokens=target,
    document_embeddings=document_embeddings,
    documents_tokens=documents_tokens,
    k=5
)

print("\nRAG-Sequence:")
print(f"Log P(y|x): {log_prob:.4f}")
print(f"使用的文档: {used_docs}")
print(f"文档权重: {used_probs}")
```

**特点** / Characteristics:
- ✅ 一致性: 所有token使用相同的文档
- ✅ Consistency: Same document for all tokens
- ✅ 适合事实性问答
- ✅ Good for factoid QA
- ✅ 更易于解释
- ✅ More interpretable

---

## 第四部分: RAG-Token / Part 4: RAG-Token

### 4.1 数学公式 / Mathematical Formulation

**核心思想** / Core Idea:
- 每个token可以来自不同文档
- Each token can come from different document
- 更灵活的知识组合
- More flexible knowledge combination

**公式** / Formula:
```
P_RAG-Token(y|x) = ∏_{i=1}^{|y|} Σ_z P(z|x) · P(y_i|x,z,y_{<i})
```

**区别** / Key Difference:
- RAG-Sequence: 先生成完整序列,再组合
- RAG-Sequence: Generate full sequences, then combine
- RAG-Token: 每个token独立组合
- RAG-Token: Combine for each token independently

### 4.2 实现细节 / Implementation Details

```python
class RAGToken:
    """RAG-Token模型(简化)"""

    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def forward_token(self, query_tokens, target_token,
                     document_embeddings, documents_tokens, k=5):
        """
        计算单个token的 P(y_i | x)

        P(y_i | x) = Σ_z P(z|x) * P(y_i|x,z)
        """
        # 检索文档
        query_emb = self.retriever.encode_query(query_tokens)
        doc_indices, doc_probs = self.retriever.retrieve(
            query_emb, document_embeddings, k=k
        )

        # 对这个token边缘化
        token_prob = 0

        for doc_idx, p_z_given_x in zip(doc_indices, doc_probs):
            doc_tokens = documents_tokens[doc_idx]

            # P(y_i | x, z) - 简化
            log_p = self.generator.generate_prob(
                query_tokens, doc_tokens, [target_token]
            )
            p_yi_given_xz = np.exp(log_p)

            token_prob += p_z_given_x * p_yi_given_xz

        return token_prob, doc_indices, doc_probs

    def forward(self, query_tokens, target_tokens,
                document_embeddings, documents_tokens, k=5):
        """
        完整序列概率

        P(y|x) = ∏_i P(y_i|x)
        """
        log_prob_total = 0

        for target_token in target_tokens:
            token_prob, _, _ = self.forward_token(
                query_tokens, target_token,
                document_embeddings, documents_tokens,
                k
            )
            log_prob_total += np.log(token_prob + 1e-8)

        return log_prob_total

# 创建RAG-Token模型
rag_token = RAGToken(retriever, generator)

# 测试
log_prob_token = rag_token.forward(
    query_tokens=query,
    target_tokens=target,
    document_embeddings=document_embeddings,
    documents_tokens=documents_tokens,
    k=5
)

print("\nRAG-Token:")
print(f"Log P(y|x): {log_prob_token:.4f}")
print("\n区别: RAG-Token可以为不同token使用不同文档!")
```

**特点** / Characteristics:
- ✅ 灵活性: 每个token选择最相关文档
- ✅ Flexibility: Choose most relevant doc per token
- ✅ 多源知识融合
- ✅ Multi-source knowledge fusion
- ✅ 适合长文本生成
- ✅ Good for long-form generation

---

## 第五部分: 两种模式对比 / Part 5: Comparing Two Modes

### 5.1 可视化对比 / Visual Comparison

```
RAG-Sequence:
查询 → [Doc1, Doc2, Doc3]
  ↓
Doc1 → 生成完整答案 y1
Doc2 → 生成完整答案 y2
Doc3 → 生成完整答案 y3
  ↓
最终答案 = 0.5*y1 + 0.3*y2 + 0.2*y3

RAG-Token:
查询 → [Doc1, Doc2, Doc3]
  ↓
Token1: Doc1(0.5) + Doc2(0.3) + Doc3(0.2)
Token2: Doc1(0.2) + Doc2(0.6) + Doc3(0.2)
Token3: Doc1(0.3) + Doc2(0.3) + Doc3(0.4)
  ↓
最终答案 = [Token1, Token2, Token3]
```

### 5.2 文档权重矩阵 / Document Weight Matrix

**RAG-Sequence权重** / RAG-Sequence Weights:
```
Token\Doc  Doc1  Doc2  Doc3  Doc4  Doc5
Token1     0.5   0.3   0.1   0.05  0.05
Token2     0.5   0.3   0.1   0.05  0.05  ← 相同
Token3     0.5   0.3   0.1   0.05  0.05  ← 相同
```

**RAG-Token权重** / RAG-Token Weights:
```
Token\Doc  Doc1  Doc2  Doc3  Doc4  Doc5
Token1     0.5   0.3   0.1   0.05  0.05
Token2     0.2   0.6   0.1   0.05  0.05  ← 不同!
Token3     0.1   0.2   0.6   0.05  0.05  ← 不同!
```

### 5.3 何时使用哪个? / When to Use Which?

**使用RAG-Sequence** / Use RAG-Sequence:
- ✅ 事实性问答
- ✅ Factoid QA
- ✅ 短答案
- ✅ Short answers
- ✅ 单一信息源足够
- ✅ Single source sufficient

**使用RAG-Token** / Use RAG-Token:
- ✅ 长文本生成
- ✅ Long-form generation
- ✅ 多跳推理
- ✅ Multi-hop reasoning
- ✅ 需要组合多个信息源
- ✅ Need to combine multiple sources

---

## 第六部分: 训练 / Part 6: Training

### 6.1 端到端训练 / End-to-End Training

**损失函数** / Loss Function:
```python
Loss = -log P(y* | x)
```

其中:
- `y*`: 真实答案
- `P(y|x)`: RAG-Sequence或RAG-Token的概率

### 6.2 梯度流动 / Gradient Flow

```
损失 L = -log P(y|x)
    ↓
梯度反向传播到:
    ↓
1. 生成器 (BART参数)
   Generator (BART parameters)
    ↓
2. 查询编码器 (检索器参数)
   Query encoder (retriever parameters)
    ↓
3. 文档编码器 (通常冻结)
   Document encoder (usually frozen)
```

**为什么冻结文档编码器?** / Why Freeze Document Encoder?
- 文档预先索引
- Documents pre-indexed
- 重新编码所有文档太昂贵
- Re-encoding all documents too expensive
- 查询端优化足够
- Query-side optimization sufficient

### 6.3 训练细节 / Training Details

**论文实现** / Paper Implementation:
- 检索器: DPR with BERT-base
- 生成器: BART-large (400M参数)
- 知识库: Wikipedia (21M passages)
- Top-k: k=5 or k=10
- 索引: FAISS for fast retrieval

**优化器** / Optimizer:
- AdamW
- 学习率: 1e-4 (generator), 5e-5 (retriever)
- 批大小: 64
- Warmup: 500 steps

---

## 第七部分: 实验结果 / Part 7: Experimental Results

### 7.1 主要结果 / Main Results

**Natural Questions (Open)**:
```
BART (无检索):        27.0% EM
RAG-Sequence:         44.5% EM  (+17.5%)
RAG-Token:            44.1% EM  (+17.1%)
```

**TriviaQA**:
```
BART:                 50.1%
RAG:                  56.8%    (+6.7%)
```

**WebQuestions**:
```
BART:                 27.6%
RAG:                  45.2%    (+17.6%)
```

### 7.2 消融实验 / Ablation Studies

**不同k值** / Different k Values:
```
k=1:   38.2% EM
k=5:   44.5% EM  ← 最优
k=10:  44.1% EM
k=20:  43.8% EM
```

**RAG-Sequence vs RAG-Token**:
- 事实性QA: Sequence稍好
- Factoid QA: Sequence slightly better
- 长文本生成: Token更好
- Long-form: Token better

### 7.3 与基线对比 / Comparison with Baselines

| 模型 | 知识来源 | 参数化 | 性能 |
|------|----------|--------|------|
| T5-11B | 记忆 | ✓ | 好 |
| REALM | 检索 | 混合 | 更好 |
| **RAG** | **检索** | **✓** | **最好** |

**关键洞察** / Key Insight:
- RAG结合了非参数和参数知识
- RAG combines non-parametric and parametric knowledge
- 比纯参数模型更小但更强
- Smaller but stronger than pure parametric models

---

## 第八部分: 优势与局限 / Part 8: Advantages and Limitations

### 8.1 优势 / Advantages

**1. 事实准确性** / Factual Accuracy:
```
✅ 访问外部知识库
✅ Access to external knowledge base
✅ 减少幻觉
✅ Reduces hallucination
✅ 答案可验证
✅ Verifiable answers
```

**2. 可扩展性** / Scalability:
```
✅ 无需重新训练即可添加知识
✅ Add knowledge without retraining
✅ 更新索引而非权重
✅ Update index instead of weights
✅ 无限知识容量
✅ Unlimited knowledge capacity
```

**3. 可解释性** / Interpretability:
```
✅ 可检查检索到的文档
✅ Can inspect retrieved documents
✅ 追踪答案来源
✅ Trace answer sources
✅ 调试更容易
✅ Easier to debug
```

**4. 效率** / Efficiency:
```
✅ 比纯参数模型更小
✅ Smaller than pure parametric models
✅ RAG: 400M + 检索开销
✅ T5-11B: 11B参数
✅ 更低推理成本
✅ Lower inference cost
```

**5. 时效性** / Up-to-date:
```
✅ 更新知识库即可
✅ Just update knowledge base
✅ 无需重新训练模型
✅ No need to retrain model
✅ 实时信息
✅ Real-time information
```

### 8.2 局限性 / Limitations

**1. 检索错误** / Retrieval Errors:
```
❌ 错误文档 → 错误答案
❌ Wrong docs → wrong answers
❌ 依赖检索质量
❌ Depends on retrieval quality
❌ 无法检索到答案
❌ Sometimes answer not retrieved
```

**2. 延迟** / Latency:
```
❌ 检索增加开销
❌ Retrieval adds overhead
❌ 不如纯生成器快
❌ Slower than pure generator
❌ 需要优化索引
❌ Need to optimize index
```

**3. 索引维护** / Index Maintenance:
```
❌ 更新需要重新编码
❌ Updates require re-encoding
❌ 存储成本
❌ Storage cost
❌ 版本管理复杂
❌ Complex version management
```

**4. 内存需求** / Memory Requirements:
```
❌ 需要完整文档索引
❌ Need full document index
❌ FAISS索引占用内存
❌ FAISS index uses memory
❌ 大规模部署困难
❌ Hard to deploy at scale
```

---

## 第九部分: 实践技巧 / Part 9: Production Tips

### 9.1 混合检索 / Hybrid Retrieval

```python
def hybrid_retrieval(query, dense_index, sparse_index, alpha=0.5):
    """
    结合密集和稀疏检索

    参数:
        query: 用户查询
        dense_index: DPR/FAISS索引
        sparse_index: BM25索引
        alpha: 密集检索权重
    """
    # 密集检索
    dense_scores = dense_index.search(query)
    dense_scores = normalize(dense_scores)

    # 稀疏检索
    sparse_scores = sparse_index.search(query)
    sparse_scores = normalize(sparse_scores)

    # 组合
    combined = alpha * dense_scores + (1 - alpha) * sparse_scores
    return combined
```

### 9.2 缓存策略 / Caching Strategy

```python
from functools import lru_cache

class CachedRAG:
    """带缓存的RAG"""

    def __init__(self, rag, cache_size=1000):
        self.rag = rag
        self.cache = lru_cache(maxsize=cache_size)

    def retrieve_and_generate(self, query):
        """检查缓存后再检索"""
        # 检查缓存
        cached = self.cache(query)
        if cached is not None:
            return cached

        # 检索 + 生成
        result = self.rag.retrieve_and_generate(query)

        # 更新缓存
        self.cache[query] = result
        return result
```

### 9.3 异步检索 / Async Retrieval

```python
import asyncio

async def async_rag_pipeline(query, rag_model):
    """异步RAG流程"""

    # 任务1: 检索文档(可以并行)
    task1 = asyncio.create_task(
        rag_model.retrieve_async(query)
    )

    # 任务2: 同时开始编码查询
    task2 = asyncio.create_task(
        rag_model.encode_query_async(query)
    )

    # 等待两个任务
    docs, query_emb = await asyncio.gather(task1, task2)

    # 生成答案
    answer = await rag_model.generate_async(
        query_emb, docs
    )

    return answer
```

### 9.4 降级策略 / Fallback Strategy

```python
class RobustRAG:
    """带降级策略的RAG"""

    def __init__(self, rag, parametric_model):
        self.rag = rag
        self.parametric = parametric_model

    def answer(self, query, max_retries=2):
        """尝试RAG,失败则使用参数模型"""
        try:
            # 尝试RAG
            for attempt in range(max_retries):
                try:
                    return self.rag.retrieve_and_generate(query)
                except RetrievalError:
                    if attempt == max_retries - 1:
                        raise

            # 检索失败,使用参数模型
            return self.parametric.generate(query)

        except Exception as e:
            # 完全失败,返回默认响应
            return self.default_response(query)
```

### 9.5 监控指标 / Monitoring Metrics

```python
class RAGMonitor:
    """RAG系统监控"""

    def __init__(self):
        self.metrics = {
            'retrieval_latency': [],
            'generation_latency': [],
            'total_latency': [],
            'retrieval_recall': [],
            'answer_accuracy': [],
        }

    def track_retrieval(self, query, retrieved_docs, relevant_docs):
        """跟踪检索质量"""
        recall = len(set(retrieved_docs) & set(relevant_docs))
        recall /= len(relevant_docs)
        self.metrics['retrieval_recall'].append(recall)

    def track_generation(self, predicted, ground_truth):
        """跟踪生成质量"""
        accuracy = (predicted == ground_truth)
        self.metrics['answer_accuracy'].append(accuracy)

    def report(self):
        """生成报告"""
        return {
            'avg_retrieval_recall': np.mean(self.metrics['retrieval_recall']),
            'avg_answer_accuracy': np.mean(self.metrics['answer_accuracy']),
            'avg_total_latency': np.mean(self.metrics['total_latency']),
        }
```

---

## 第十部分: 现代扩展 / Part 10: Modern Extensions

### 10.1 RETRO (DeepMind, 2022)

**关键改进** / Key Improvements:
- 在每一层检索
- Retrieve at every layer
- 不仅在输入层
- Not just at input layer

```
传统RAG:
输入 → [检索] → 层1 → 层2 → ... → 输出

RETRO:
输入 → [检索] → 层1 → [检索] → 层2 → [检索] → ...
```

### 10.2 Atlas (Meta AI, 2022)

**关键改进** / Key Improvements:
- 改进的训练策略
- Improved training strategy
- 对比学习 + RAG
- Contrastive learning + RAG
- 更好的检索器-生成器对齐
- Better retriever-generator alignment

### 10.3 Toolformer (Meta AI, 2023)

**关键改进** / Key Improvements:
- 通过API调用检索
- Retrieve via API calls
- 学习何时检索
- Learn when to retrieve
- 自主工具使用
- Autonomous tool usage

### 10.4 Self-RAG (2023)

**关键改进** / Key Improvements:
- 自我反思检索
- Self-reflective retrieval
- 评估检索质量
- Evaluate retrieval quality
- 按需重新检索
- Re-retrieve on demand

```
Self-RAG流程:
1. 生成答案
2. 评估: 是否需要检索?
3. 如果是: 检索相关文档
4. 重新生成答案
5. 重复直到满意
```

### 10.5 WebGPT (OpenAI, 2021)

**关键改进** / Key Improvements:
- 交互式检索
- Interactive retrieval
- 浏览器接口
- Browser interface
- 人工反馈
- Human feedback
- 强化学习微调
- RLHF tuning

---

## 第十一部分: 应用场景 / Part 11: Applications

### 11.1 开放域问答 / Open-Domain QA

**场景** / Scenario:
```
用户: "谁发明了电话?"
系统:
  1. 检索Wikipedia相关文章
  2. 生成答案: "亚历山大·格雷厄姆·贝尔于1876年发明了电话"
```

**实现** / Implementation:
```python
def open_domain_qa(query, rag_model, kb):
    """开放域问答系统"""
    # 检索
    docs = rag_model.retrieve(query, kb, k=5)

    # 生成
    answer = rag_model.generate(query, docs)

    # 返回答案+引用
    return {
        'answer': answer,
        'sources': docs
    }
```

### 11.2 聊天机器人 / Chatbots with Knowledge

**场景** / Scenario:
```python
class KnowledgeableChatbot:
    """有知识库的聊天机器人"""

    def __init__(self, rag_model, knowledge_base):
        self.rag = rag_model
        self.kb = knowledge_base
        self.history = []

    def chat(self, user_message):
        # 结合历史
        context = self.history + [user_message]
        query = " ".join(context[-3:])  # 最近3轮

        # 检索相关知识
        docs = self.rag.retrieve(query, self.kb, k=3)

        # 生成回复
        response = self.rag.generate(query, docs)

        # 更新历史
        self.history.append(user_message)
        self.history.append(response)

        return response
```

### 11.3 文档问答 / Document QA

**场景** / Scenario:
```python
class DocumentQA:
    """文档问答系统"""

    def __init__(self, documents):
        # 预处理文档
        self.passages = self.split_into_passages(documents)
        self.index = self.build_index(self.passages)
        self.rag = RAG()

    def split_into_passages(self, documents):
        """将文档分割成段落"""
        passages = []
        for doc in documents:
            # 按段落/章节分割
            chunks = doc.split('\n\n')
            passages.extend(chunks)
        return passages

    def answer(self, query):
        """回答关于文档的问题"""
        # 从文档中检索
        docs = self.rag.retrieve(query, self.index, k=5)

        # 生成答案
        answer = self.rag.generate(query, docs)

        return answer
```

### 11.4 事实核查 / Fact-Checking

**场景** / Scenario:
```python
def fact_check(claim, rag_model, trusted_sources):
    """
    事实核查系统

    返回: 真/假/不确定 + 证据
    """
    # 检索相关文章
    docs = rag_model.retrieve(claim, trusted_sources, k=10)

    # 生成核查结果
    verification = rag_model.generate(
        f"验证以下说法: {claim}",
        docs
    )

    # 提取证据
    evidence = extract_evidence(docs, verification)

    return {
        'verdict': classify_verdict(verification),
        'evidence': evidence,
        'confidence': calculate_confidence(verification)
    }
```

### 11.5 研究助手 / Research Assistant

**场景** / Scenario:
```python
class ResearchAssistant:
    """研究助手系统"""

    def __init__(self, paper_database):
        self.db = paper_database
        self.rag = RAG()

    def literature_review(self, topic):
        """生成文献综述"""
        # 检索相关论文
        papers = self.db.search(topic, top_k=20)

        # 生成综述
        review = self.rag.generate(
            f"生成关于'{topic}'的文献综述",
            papers
        )

        return review

    def find_related_work(self, paper):
        """查找相关工作"""
        # 提取关键词
        keywords = extract_keywords(paper)

        # 检索相关工作
        related = self.db.search(keywords, top_k=10)

        return related

    def suggest_citations(self, draft):
        """建议引用"""
        # 检索相关论文
        docs = self.rag.retrieve(draft, self.db, k=5)

        return docs
```

---

## 第十二部分: 深入理解 / Part 12: Deep Dive

### 12.1 为什么RAG有效? / Why RAG Works?

**1. 非参数记忆** / Non-Parametric Memory:
```
参数模型:
- 知识压缩在权重中
- Knowledge compressed in weights
- 容量有限
- Limited capacity
- 更新需要重新训练
- Updates require retraining

非参数检索:
- 知识存储在文档中
- Knowledge stored in documents
- 无限容量
- Unlimited capacity
- 更新只需添加文档
- Updates just add documents
```

**2. 两阶段推理** / Two-Stage Reasoning:
```
第一阶段: 检索(Retriever)
- 快速访问相关知识
- Fast access to relevant knowledge
- 粗粒度匹配
- Coarse-grained matching

第二阶段: 生成(Generator)
- 精细推理和综合
- Fine-grained reasoning and synthesis
- 自然语言表达
- Natural language expression
```

**3. 注意力机制** / Attention Mechanism:
```
RAG中的注意力 = 检索权重

P(y|x) = Σ_z P(z|x) · P(y|x,z)
         ↑      ↑
      注意力  检索到的知识

类似Transformer的多头注意力:
- 每个文档=一个"头"
- Each document = one "head"
- 加权组合信息
- Weighted combination of information
```

### 12.2 与其他方法的联系 / Connection to Other Methods

**1. 记忆网络** / Memory Networks:
```
RAG ≈ 连续记忆网络
RAG ≈ Continuous Memory Network

- 读取操作: 检索
- Read operation: Retrieval
- 写入操作: 更新索引
- Write operation: Update index
- 两者都是可微分的
- Both are differentiable
```

**2. 查询-键-值** / Query-Key-Value:
```
DPR检索 = 注意力机制

Query:  用户问题
Key:    文档嵌入
Value:  文档内容

Attention(Q,K,V) = softmax(Q·K^T) · V
Retrieval(Q,D)   = softmax(Q·D_emb) · D
```

**3. 管道并行** / Pipeline Parallelism:
```
RAG = 信息检索 + 生成管道

阶段1: 检索(并行)
Stage 1: Retrieval (parallel)

阶段2: 生成(顺序)
Stage 2: Generation (sequential)

类似GPipe的管道并行!
Similar to GPipe pipeline parallelism!
```

### 12.3 理论分析 / Theoretical Analysis

**容量分析** / Capacity Analysis:
```
纯参数模型:
C_param = O(N)  # N=参数量

RAG模型:
C_RAG = O(N) + O(M)  # N=参数, M=文档数

优势:
- M可以无限增长
- M can grow infinitely
- N保持较小
- N stays small
```

**泛化分析** / Generalization:
```
参数知识:
- 学习模式
- Learn patterns
- 跨任务泛化
- Generalize across tasks

非参数知识:
- 存储事实
- Store facts
- 零样本泛化到新知识
- Zero-shot generalize to new knowledge

RAG = 两者的最佳组合
RAG = Best of both worlds
```

---

## 第十三部分: 实现细节 / Part 13: Implementation Details

### 13.1 FAISS索引 / FAISS Indexing

```python
import faiss
import numpy as np

class FAISSIndex:
    """FAISS索引管理"""

    def __init__(self, embedding_dim, index_type='flat'):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.index = None

    def build_index(self, embeddings):
        """
        构建FAISS索引

        参数:
            embeddings: [N, embedding_dim] 文档嵌入
        """
        n_docs = embeddings.shape[0]

        if self.index_type == 'flat':
            # 精确搜索(慢但准确)
            self.index = faiss.IndexFlatIP(self.embedding_dim)

        elif self.index_type == 'ivf':
            # 倒排文件(快但近似)
            nlist = int(np.sqrt(n_docs))  # 聚类中心数
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(
                quantizer, self.embedding_dim, nlist
            )
            self.index.train(embeddings)  # 训练聚类

        elif self.index_type == 'hnsw':
            # 分层导航小世界图(最快)
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, M=32)

        # 添加向量
        self.index.add(embeddings.astype('float32'))

        return self.index

    def search(self, query_embedding, k=10):
        """搜索top-k文档"""
        scores, indices = self.index.search(
            query_embedding.astype('float32'),
            k
        )
        return scores[0], indices[0]

# 使用示例
embedding_dim = 768
n_docs = 100000

# 创建虚拟嵌入
doc_embeddings = np.random.randn(n_docs, embedding_dim).astype('float32')
# L2归一化(用于内积搜索)
faiss.normalize_L2(doc_embeddings)

# 构建索引
index = FAISSIndex(embedding_dim, index_type='ivf')
index.build_index(doc_embeddings)

# 搜索
query = np.random.randn(1, embedding_dim).astype('float32')
faiss.normalize_L2(query)
scores, indices = index.search(query, k=10)

print(f"Top-10文档索引: {indices}")
print(f"相似度分数: {scores}")
```

### 13.2 批处理推理 / Batch Inference

```python
class BatchRAG:
    """批处理RAG推理"""

    def __init__(self, retriever, generator, batch_size=32):
        self.retriever = retriever
        self.generator = generator
        self.batch_size = batch_size

    def retrieve_batch(self, queries):
        """批量检索"""
        # 编码所有查询
        query_embeddings = [
            self.retriever.encode_query(q) for q in queries
        ]
        query_embeddings = np.array(query_embeddings)

        # 批量检索
        all_docs = []
        for q_emb in query_embeddings:
            docs, probs = self.retriever.retrieve(q_emb, k=5)
            all_docs.append((docs, probs))

        return all_docs

    def generate_batch(self, queries, retrieved_docs):
        """批量生成"""
        answers = []

        for query, (docs, probs) in zip(queries, retrieved_docs):
            # 为每个查询生成答案
            answer = self.generator.generate(query, docs)
            answers.append(answer)

        return answers

    def forward_batch(self, queries):
        """批量前向传播"""
        # 批量检索
        retrieved_docs = self.retrieve_batch(queries)

        # 批量生成
        answers = self.generate_batch(queries, retrieved_docs)

        return answers
```

### 13.3 分布式部署 / Distributed Deployment

```python
from concurrent.futures import ThreadPoolExecutor

class DistributedRAG:
    """分布式RAG系统"""

    def __init__(self, retriever_nodes, generator_nodes):
        self.retriever_nodes = retriever_nodes
        self.generator_nodes = generator_nodes
        self.executor = ThreadPoolExecutor(max_workers=10)

    def parallel_retrieve(self, queries):
        """并行检索多个节点"""
        futures = []
        for query in queries:
            future = self.executor.submit(
                self.retriever_nodes[0].retrieve,
                query
            )
            futures.append(future)

        # 收集结果
        results = [f.result() for f in futures]
        return results

    def parallel_generate(self, queries_docs):
        """并行生成"""
        futures = []
        for query, docs in queries_docs:
            future = self.executor.submit(
                self.generator_nodes[0].generate,
                query, docs
            )
            futures.append(future)

        results = [f.result() for f in futures]
        return results
```

---

## 第十四部分: 评估方法 / Part 14: Evaluation Methods

### 14.1 检索评估 / Retrieval Evaluation

```python
def evaluate_retrieval(queries, ground_truth_docs, retriever, k_values=[1,5,10]):
    """
    评估检索质量

    指标:
    - Recall@k: 检索到的相关文档比例
    - MRR: 平均倒数排名
    """
    results = {k: [] for k in k_values}
    mrr_scores = []

    for query, relevant_docs in zip(queries, ground_truth_docs):
        # 检索
        retrieved = retriever.retrieve(query, k=max(k_values))

        # 计算Recall@k
        for k in k_values:
            recall = len(set(retrieved[:k]) & set(relevant_docs))
            recall /= len(relevant_docs)
            results[k].append(recall)

        # 计算MRR
        for rank, doc in enumerate(retrieved, 1):
            if doc in relevant_docs:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)

    # 汇总结果
    summary = {
        f'Recall@{k}': np.mean(results[k]) for k in k_values
    }
    summary['MRR'] = np.mean(mrr_scores)

    return summary

# 示例
queries = [...]
ground_truth = [[0, 5, 12], [3, 8, 15], ...]  # 相关文档索引

metrics = evaluate_retrieval(queries, ground_truth, retriever)
print(metrics)
# 输出: {'Recall@1': 0.85, 'Recall@5': 0.95, 'Recall@10': 0.98, 'MRR': 0.92}
```

### 14.2 端到端评估 / End-to-End Evaluation

```python
def evaluate_rag(queries, answers, rag_model):
    """
    评估RAG端到端性能

    指标:
    - Exact Match (EM): 完全匹配
    - F1 Score: token级别F1
    - BLEU: n-gram重叠
    """
    from collections import Counter
    from nltk.translate.bleu_score import sentence_bleu

    em_scores = []
    f1_scores = []
    bleu_scores = []

    for query, reference in zip(queries, answers):
        # 生成答案
        prediction = rag_model.retrieve_and_generate(query)

        # Exact Match
        em = (prediction.strip() == reference.strip())
        em_scores.append(em)

        # F1 Score (token级别)
        pred_tokens = prediction.split()
        ref_tokens = reference.split()

        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            f1 = 0.0
        else:
            precision = num_common / len(pred_tokens)
            recall = num_common / len(ref_tokens)
            f1 = 2 * (precision * recall) / (precision + recall)

        f1_scores.append(f1)

        # BLEU Score
        bleu = sentence_bleu([ref_tokens], pred_tokens)
        bleu_scores.append(bleu)

    # 汇总
    return {
        'EM': np.mean(em_scores),
        'F1': np.mean(f1_scores),
        'BLEU': np.mean(bleu_scores)
    }
```

### 14.3 事实一致性评估 / Factual Consistency

```python
def check_factual_consistency(answer, retrieved_docs, nli_model):
    """
    检查答案与检索文档的事实一致性

    使用NLI(自然语言推断)模型
    """
    consistency_scores = []

    for doc in retrieved_docs:
        # 检查: 答案是否能从文档中推断
        result = nli_model.predict(
            premise=doc,
            hypothesis=answer
        )

        # Entailment = 一致
        # Contradiction = 矛盾
        # Neutral = 中立
        if result['label'] == 'entailment':
            consistency_scores.append(1.0)
        elif result['label'] == 'contradiction':
            consistency_scores.append(0.0)
        else:
            consistency_scores.append(0.5)

    return {
        'max_consistency': max(consistency_scores),
        'avg_consistency': np.mean(consistency_scores)
    }
```

---

## 关键要点总结 / Key Takeaways Summary

### 核心概念 / Core Concepts

**1. RAG架构** / RAG Architecture:
```
查询 → 检索器(DPR) → 文档 → 生成器(BART) → 答案
```

**2. 两种模式** / Two Modes:
```
RAG-Sequence: P(y|x) = Σ_z P(z|x) · P(y|x,z)
  - 一致性更好
  - 适合事实性QA

RAG-Token: P(y|x) = ∏_i Σ_z P(z|x) · P(y_i|x,z)
  - 灵活性更好
  - 适合长文本生成
```

**3. 训练** / Training:
```
端到端可微分
Loss = -log P(y*|x)
梯度流到检索器和生成器
```

**4. 优势** / Advantages:
- ✅ 事实准确性
- ✅ 可扩展性
- ✅ 可解释性
- ✅ 效率(比大模型小)
- ✅ 时效性

**5. 应用** / Applications:
- 开放域问答
- 聊天机器人
- 文档QA
- 事实核查
- 研究助手

**6. 现代发展** / Modern Extensions:
- RETRO: 每层检索
- Atlas: 改进训练
- Self-RAG: 自我反思
- Toolformer: API调用

### 数学公式总结 / Mathematical Summary

```
检索:
P(z|x) ∝ exp(E_Q(x) · E_D(z))

RAG-Sequence:
P_RAG-Seq(y|x) = Σ_z P(z|x) · P_seq2seq(y|x,z)

RAG-Token:
P_RAG-Token(y|x) = ∏_i Σ_z P(z|x) · P(y_i|x,z,y_{<i})

损失:
L = -log P(y*|x)
```

### 实现要点 / Implementation Checklist

- [ ] 选择合适的RAG模式(Sequence vs Token)
- [ ] 使用FAISS构建快速索引
- [ ] 实现混合检索(dense + sparse)
- [ ] 添加缓存和异步优化
- [ ] 准备降级策略
- [ ] 监控检索质量和生成质量
- [ ] 评估端到端性能

---

## 练习 / Exercises

### 练习1: 实现简单RAG / Exercise 1: Implement Simple RAG

```python
# 实现一个简单的RAG系统
class SimpleRAG:
    def __init__(self, documents, embedding_dim=128):
        # TODO: 初始化
        pass

    def build_index(self):
        # TODO: 构建文档索引
        pass

    def retrieve(self, query, k=5):
        # TODO: 实现检索
        pass

    def generate(self, query, docs):
        # TODO: 实现生成
        pass

    def answer(self, query):
        # TODO: 端到端问答
        pass
```

### 练习2: 比较RAG-Sequence和RAG-Token / Exercise 2: Compare RAG Variants

```python
# 比较两种模式的性能
def compare_rag_modes(queries, answers, rag_sequence, rag_token):
    # TODO: 实现对比
    # 提示: 测量延迟、准确性、一致性
    pass
```

### 练习3: 优化检索质量 / Exercise 3: Optimize Retrieval

```python
# 实现混合检索
def hybrid_search(query, dense_index, sparse_index, alpha=0.5):
    # TODO: 结合密集和稀疏检索
    pass
```

---

## 参考资源 / References

**论文** / Papers:
1. Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
2. Karpukhin et al. (2020). "Dense Passage Retrieval for Open-Domain Question Answering"
3. Borgeaud et al. (2022). "Improving language models by retrieving from trillions of tokens" (RETRO)

**代码** / Code:
- HuggingFace Transformers: RAG实现
- FAISS: 高效相似度搜索
- LangChain: RAG应用框架

**博客** / Blogs:
- "Introduction to RAG" - Pinecone
- "Building RAG Systems" - Towards Data Science
- "RAG vs Fine-tuning" - Cohere

---

## 思考问题 / Reflection Questions

1. **RAG vs Fine-tuning**: 何时应该使用RAG而不是微调模型?
   **RAG vs Fine-tuning**: When should you use RAG instead of fine-tuning?

2. **检索质量**: 如何处理检索失败的情况?
   **Retrieval Quality**: How to handle retrieval failures?

3. **知识更新**: 如何高效地更新知识库而不重新索引?
   **Knowledge Updates**: How to efficiently update the knowledge base without re-indexing?

4. **多模态RAG**: 如何将RAG扩展到图像、视频等模态?
   **Multimodal RAG**: How to extend RAG to images, videos, and other modalities?

5. **评估指标**: 除了EM和F1,还有什么指标适合评估RAG?
   **Evaluation Metrics**: Beyond EM and F1, what other metrics are suitable for RAG?

---

**下一步**: Paper 30 - 最后一篇! 🎉
**Next**: Paper 30 - The final one! 🎉
