---
title: "OneRec: 首个工业级端到端生成式推荐系统"
date: 2026-03-06
categories:
  - Paper Reading
  - Generative Recommendation
tags:
  - OneRec
  - Generative Recommendation
  - DPO
  - Kuaishou
  - End-to-End
math: true
comments: true
---

> **论文：** OneRec: Unifying Retrieve and Rank with Generative Recommender and Preference Alignment
> **来源：** Kuaishou Inc.
> **链接：** arXiv:2502.18965v1

## 1. 背景与动机

现代推荐系统普遍采用级联排序架构：召回 → 粗排 → 精排。每一阶段独立优化，前一阶段的效果成为后一阶段的上界，限制了整体系统的性能上限。

生成式推荐（Generative Recommendation, GR）近年来受到关注，它通过自回归方式直接生成物品标识符。然而，现有的生成式模型仅作为召回阶段的选择器，其推荐精度仍无法与复杂的多阶段级联排序系统相比。

OneRec 的核心目标是：**用一个统一的生成模型替代级联学习框架，实现真正的端到端单阶段推荐**。这是首个在真实工业场景中显著超越传统多级排序管道的端到端生成模型。

![](/images/2502_18965v1/page_1_Figure_2.jpeg)

*图1: (a) OneRec 统一端到端生成架构 (b) 传统级联排序系统*

## 2. 核心方法

OneRec 包含三个关键组件：

### 2.1 Balanced RQ-KMeans 语义ID

**问题背景**：现有工作（如TIGER）使用 RQ-VAE 将多模态嵌入编码为语义token，但存在"沙漏现象"（hourglass phenomenon）——码本分布不均，部分码本过度使用而部分闲置。

**解决方案**：采用多层平衡量化机制，使用**残差K-Means量化算法**替代RQ-VAE。

**算法细节**（Algorithm 1）：

```
输入: 物品集合 V，聚类数 K
1. 计算每个簇目标大小 w = |V|/K
2. 随机初始化质心
3. repeat:
4.   初始化未分配集合 U = V
5.   for 每个簇 k:
6.     按到质心距离升序排序 U
7.     分配前 w 个物品给簇 k
8.     更新质心为该簇物品均值
9.     从 U 移除已分配物品
10. until 收敛
```

**关键特性**：
- 强制平衡：每个簇恰好包含 |V|/K 个物品
- 层次索引：L=3层，每层K=8192个码本
- 残差计算：$$r_i^{(l+1)} = r_i^l - c_{(s_i^l)}^l$$

### 2.2 会话级列表生成（Session-wise Generation）

**与传统方法的区别**：

| 方法 | 生成粒度 | 上下文建模 |
|------|----------|------------|
| 传统Point-wise | 逐个预测下一个物品 | 缺乏列表级依赖 |
| OneRec Session-wise | 一次生成5-10个视频组成会话 | 建模会话内物品间的相对内容和顺序关系 |

**高质量会话定义标准**：
- 用户实际观看视频数 ≥ 5
- 总会观看时长超过阈值
- 用户有交互行为（点赞、收藏、分享）

**模型架构**：

基于 T5 的 Encoder-Decoder 结构：

```
Encoder: H = Encoder(H_u)  # 处理用户历史交互序列
Decoder: 自回归生成目标会话
```

**Sparse MoE 扩展**：

为在合理成本下扩展模型容量，Decoder 的 FFN 层采用 MoE 架构：

$$
\mathbf{H}_{t}^{l+1} = \sum_{i=1}^{N_{\text{MoE}}} \left( g_{i,t} \, \text{FFN}_{i} \left( \mathbf{H}_{t}^{l} \right) \right) + \mathbf{H}_{t}^{l}
$$

其中：
- $$N_{MoE} = 24$$（总专家数）
- $$K_{MoE} = 2$$（每token激活专家数）
- 稀疏门控：只有Top-2专家被激活

**训练目标**：

在会话语义ID序列上进行Next Token Prediction：

$$
\mathcal{L}_{\text{NTP}} = -\sum_{i=1}^{m} \sum_{j=1}^{L} \log P(s_i^{j+1} \mid \text{context})
$$

### 2.3 迭代偏好对齐（Iterative Preference Alignment）

**核心挑战**：NLP中的DPO依赖人工标注的偏好对，但推荐系统中每个请求只有一个展示机会，无法同时获得正负样本。

**解决方案架构**：

![](/images/2502_18965v1/page_2_Figure_2.jpeg)

*图2: OneRec 整体框架：(i) 会话训练阶段 (ii) IPA阶段*

#### 2.3.1 奖励模型训练

奖励模型 $$R(u, S)$$ 评估用户 u 对会话 S 的偏好程度：

1. **Target-aware表示**：$$e_i = v_i \odot u$$（物品与用户行为的target attention）
2. **会话内交互**：通过Self-Attention层建模会话内物品间关系
3. **多目标预测**：同时预测 swt（会话观看时长）、vtr（观看概率）、wtr（关注概率）、ltr（点赞概率）

损失函数：

$$
\mathcal{L}_{RM} = -\sum_{xtr \in \{swt,vtr,wtr,ltr\}} \left( y^{xtr} \log(\hat{r}^{xtr}) + (1-y^{xtr})\log(1-\hat{r}^{xtr}) \right)
$$

#### 2.3.2 迭代DPO训练

**自难负采样策略**：

1. 用当前模型 $$M_t$$ 通过 Beam Search 为每个用户生成 $$N=128$$ 个不同响应
2. 用奖励模型计算每个响应的奖励值 $$r_u^n = R(u, S_u^n)$$
3. 选择奖励最高的作为正例 $$S_u^w$$，最低的作为负例 $$S_u^l$$

**DPO损失**：

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma \left( \beta \log \frac{M_{t+1}(S_u^w \mid \mathcal{H}_u)}{M_t(S_u^w \mid \mathcal{H}_u)} - \beta \log \frac{M_{t+1}(S_u^l \mid \mathcal{H}_u)}{M_t(S_u^l \mid \mathcal{H}_u)} \right)
$$

**迭代训练流程**（Algorithm 2）：

- 仅采样 $$r_{DPO} = 1\%$$ 的数据进行DPO训练
- 每轮迭代用更新后的模型重新生成偏好对
- 交替使用 NTP 和 DPO 损失

## 3. 实验结果

### 3.1 离线性能

| 模型 | swt-max | vtr-max | ltr-max |
|------|---------|---------|---------|
| TIGER-1B | 0.1368 | 0.6776 | 0.0579 |
| OneRec-1B | 0.1529 (+11.8%) | 0.7013 | 0.0660 (+14.0%) |
| OneRec-1B+IPA | **0.1933** (+41.3%) | **0.7646** | **0.1203** (+107.8%) |

关键发现：
1. Session-wise生成显著优于Point-wise生成（TIGER）
2. 仅1%的DPO训练带来显著收益：swt-max +4.04%，ltr-max +5.43%
3. IPA策略优于所有DPO变体（DPO、IPO、cDPO、rDPO、CPO、simPO、S-DPO）

### 3.2 消融实验

**DPO采样比例**：

![](/images/2502_18965v1/page_6_Figure_10.jpeg)

*图4: DPO采样比例消融。1%比例已能带来显著提升，继续增加收益有限。*

- 1%采样达到95%的最大性能
- 5%采样需要5倍GPU资源，但性能提升不显著

**模型规模扩展**：

![](/images/2502_18965v1/page_7_Figure_4.jpeg)

*图6: OneRec 模型规模扩展性。参数量从0.05B扩展到1B，性能持续提升。*

- OneRec-0.1B 比 OneRec-0.05B 提升 14.45%
- 0.2B、0.5B、1B 分别再提升 5.09%、5.70%、5.69%

### 3.3 预测动态分析

![](/images/2502_18965v1/page_7_Figure_2.jpeg)

*图5: 语义ID各层softmax输出概率分布。红星表示奖励最高的物品。*

- IPA使预测分布产生显著置信度偏移
- 第一层熵最高（6.00），逐层递减（第二层3.71，第三层0.048）
- 自回归机制导致层次化不确定性降低

### 3.4 在线A/B测试

在Kuaishou主页面视频推荐场景进行严格A/B测试（1%主流量）：

| 模型 | 总观看时长 | 平均观看时长 |
|------|------------|--------------|
| OneRec-0.1B | +0.57% | +4.26% |
| OneRec-1B | +1.21% | +5.01% |
| OneRec-1B+IPA | **+1.68%** | **+6.56%** |

OneRec 在真实工业环境中显著超越复杂的多阶段排序系统。

## 4. 系统部署

![](/images/2502_18965v1/page_4_Figure_22.jpeg)

*图3: OneRec 在线部署架构*

**部署组件**：
1. **训练系统**：XLA + bfloat16混合精度训练
2. **在线服务系统**：KV Cache解码 + float16量化 + Beam Search (beam=128)
3. **DPO样本服务器**：实时生成偏好数据

**推理优化**：
- MoE架构仅激活13%参数
- KV Cache减少GPU内存开销
- float16量化

## 5. 关键结论与启示

1. **端到端生成的可行性**：OneRec证明工业级端到端生成式推荐是可行的，且能超越传统级联系统

2. **会话级建模的必要性**：相比逐点生成，会话级生成能更好地保持上下文连贯性和多样性

3. **DPO在推荐中的独特应用**：
   - 奖励模型解决推荐中正负样本无法同时获取的问题
   - 自难负采样比随机采样更有效
   - 1%采样比例即可达到大部分收益

4. **Scaling Law再次验证**：从0.05B到1B，模型性能持续提升

5. **局限性**：论文指出在交互指标（如点赞）上的表现仍有提升空间，未来工作将关注多目标建模

## 6. 与HSTU的对比

| 维度 | HSTU (Meta) | OneRec (Kuaishou) |
|------|-------------|-------------------|
| 核心贡献 | 验证推荐Scaling Law | 实现端到端单阶段生成 |
| 架构 | 生成概率→TopK检索 | 直接生成物品序列 |
| ID表示 | 随机整数ID | Balanced RQ-KMeans语义ID |
| 优化重点 | 训练和推理效率 | DPO偏好对齐 |
| 在线效果 | A/B +12.4% | A/B +1.68%观看时长 |

两者互补：HSTU验证了生成式推荐的Scaling潜力，OneRec实现了真正的端到端统一架构。

---

*本文基于 OneRec 论文整理，仅用于学术研究记录。*
