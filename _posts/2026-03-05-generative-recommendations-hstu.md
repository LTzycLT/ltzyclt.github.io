---
title: "论文精读：Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations"
date: 2026-03-05
tags:
  - 论文精读
  - 生成式推荐
  - 推荐系统
  - Transformer
categories:
  - 技术
---

## 1. 论文基本信息
* **论文标题**：Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations
* **核心作者**：Jiaqi Zhai, Lucy Liao, Xing Liu 等（Meta AI 团队）
* **核心贡献**：提出了**生成式推荐模型（Generative Recommenders, GRs）**和新型序列架构 **HSTU（Hierarchical Sequential Transduction Units）**，成功将推荐系统转化为纯序列转换任务，并在工业界实现了 1.5 万亿参数规模的模型落地，验证了推荐系统领域的 Scaling Law。

---

## 2. 研究背景与动机

过去十年中，深度学习推荐模型（DLRMs）一直是推荐系统领域的基石。然而，在工业级规模下，DLRMs 面临着严峻的扩展性瓶颈：

* **算力扩展性差**：尽管 DLRMs 使用了海量数据和数千个特征进行训练，但工业界的大多数 DLRM 无法随着算力的增加而持续提升模型质量。
* **特征工程繁琐**：DLRMs 高度依赖异构特征（如类别特征、数值特征），由于新内容不断加入，特征空间基数极高（通常在十亿级别）。
* **Transformer 的挑战**：尽管 Transformer 在语言和视觉领域取得了成功，但推荐系统面临着无明确结构的特征、持续变化的十亿级词表，以及长达 $10^{5}$ 的用户序列带来的巨大计算成本挑战。

---

## 3. 核心创新 I：推荐任务的生成式重构 (Generative Recommenders)

作者将用户行为视为生成建模中的新模态，将检索和排序任务重构为**生成式序列转换任务**。

### 3.1 统一异构特征空间

在 GR 中，传统的异构特征被统一编码为时间序列：

* **类别特征（Categorical Features）**：以用户交互的最长序列为主时间序列，将其他变化缓慢的特征（如画像、关注列表）压缩并合并到主序列中。
* **数值特征（Numerical Features）**：数值特征（如 CTR）变化极快，完全序列化不可行。但因为类别特征已经序列化，只要序列模型足够强大且采用目标感知（Target-aware）机制，就可以完全移除数值特征。

### 3.2 排序与检索的序列化转化

* **检索阶段（Retrieval）**：转化为预测下一个内容的概率分布 $p(\Phi_{i+1}|u_{i})$。
* **排序阶段（Ranking）**：将内容（Items，$\Phi$）与用户动作（Actions，$a$）交替排列，使排序任务可以公式化为 $p(a_{i+1}|\Phi_{0},a_{0},\Phi_{1},a_{1},...,\Phi_{i+1})$，从而在因果自回归设置中实现目标感知（Target-aware）交叉注意力。

### 3.3 生成式训练

通过采用生成式训练，对用户 $i$ 按照 $1/n_{i}$ 的比率进行采样，将序列模型的整体训练时间复杂度从 $O(N^{3}d+N^{2}d^{2})$ 大幅降低至 $O(N^{2}d+Nd^{2})$。

---

## 4. 核心创新 II：HSTU 高性能自注意力编码器

HSTU 是一种专为高基数、非平稳流式推荐数据设计的新架构。

### 4.1 HSTU 的数学表达

HSTU 用统一的模块替换了 DLRM 的特征提取、特征交叉和表征转换。其单层包含逐点投影、空间聚合和逐点转换：

$$U(X),V(X),Q(X),K(X)=Split(\phi_{1}(f_{1}(X))$$

$$A(X)V(X)=\phi_{2}(Q(X)K(X)^{T}+rab^{p,t})V(X)$$

$$Y(X)=f_{2}(Norm(A(X)V(X))\odot U(X))$$

*(注：$\phi_{1}, \phi_{2}$ 使用 SiLU 激活函数；$rab^{p,t}$ 为相对注意力偏置。)*

### 4.2 核心改进点

* **逐点聚合注意力（Pointwise Aggregated Attention）**：放弃了 Softmax，改用逐点注意力。这不仅能捕捉用户偏好的相对顺序，还能保留交互的绝对强度信息，对非平稳词表更鲁棒。
* **极低的显存占用**：HSTU 采用全融合设计并去除了传统的 Feedforward 层，单层激活状态显存仅为 **14d**（相比标准 Transformer 的 **33d** 大幅降低），允许模型变得更深。

---

## 5. 核心创新 III：极高的训练与推理效率

### 5.1 训练优化：随机长度算法 (Stochastic Length, SL)

利用推荐序列的高度时间重复性，SL 算法以概率 $N_{c}^{\alpha}/n_{c,j}^{2}$ 保留或截断序列。在模型质量几乎无损的前提下，大幅增加了稀疏度，降低了训练成本。

### 5.2 推理优化：M-FALCON 算法

针对排序阶段成千上万的候选集，提出了 M-FALCON 算法。

* 通过修改注意力掩码并引入微批处理（Microbatching，$b_{m}$），将交叉注意力的复杂度从 $O(b_{m}n^{2}d)$ 降至 $O(n^{2}d)$。
* 支持了复杂度高出 **285 倍** 的交叉注意力模型，在相同的推理预算下实现了 1.50x 到 2.99x 的加速。

---

## 6. 实验结果与商业价值

### 6.1 效率与离线提升

* 在 8192 序列长度下，HSTU 相比 FlashAttention-2 版本的 Transformer，训练速度提升达 **15.2 倍**，推理速度提升 **5.3 倍到 5.6 倍**。
* 在 NDCG 等指标上，最高超越基线模型达 **65.8%**。

### 6.2 在线 A/B 测试

1.5 万亿参数的 GR 模型已在十亿级用户的互联网平台上部署，在线 A/B 测试中核心指标提升了 **12.4%**。

### 6.3 验证推荐领域的 Scaling Law

GR 的模型质量在跨越三个数量级的训练算力下（最高接近 GPT-3/LLaMa-2 级别算力），严格呈现出与算力相关的幂律分布（Power-law），打破了传统 DLRM 质量早早饱和的魔咒，铺平了通往推荐系统基础模型（Foundation Models）的道路。