---
title: TrieRec 论文精读
date: 2026-03-04
tags:
  - 论文精读
  - 生成式推荐
  - Transformer
  - 位置编码
categories:
  - 推荐系统
  - 学术论文
---

# TrieRec: Trie-aware Generative Recommendation

> 论文链接: [arXiv:2602.21677](https://arxiv.org/abs/2602.21677)

## 核心贡献

提出 **TrieRec**，解决生成式推荐中**忽略树结构信息**的关键局限，通过两种位置编码：
- **Trie-aware Absolute Positional Encoding (TAE)**
- **Topology-aware Relative Positional Encoding (TRE)**

在三个GR backbone上实现**平均8.83%相对提升**。

---

## 问题背景

### 生成式推荐流程

1. **Item Tokenization**: 通过层级量化将items映射为token序列，形成Trie树
2. **Autoregressive Generation**: 用Transformer预测下一个item的token序列

### 现有问题

将item codes视为**扁平的1D序列**，丢失了层级item树中的关键**拓扑信息**。

---

## 什么是拓扑信息？

### Trie树结构

```
                    root
                   /    \
                 a1      a2          ← Depth=1
                /  \    /  \
              b1   b2  b1   b2       ← Depth=2
             / \   |   |   ...
           c1  c2  c3  c4  ...       ← Depth=3
```

### 拓扑信息的核心要素

| 要素 | 含义 |
|------|------|
| **Depth** | 节点在树中的层级 |
| **共同前缀** | 两个token共享的路径 |
| **LCA距离** | 最近公共祖先的距离 |

### 为什么需要LCA？

| Token对 | LCA距离 | 关系 |
|---------|---------|------|
| c2(篮球) ↔ c3(足球) | 1 | 高度相似 |
| c2(篮球) ↔ c1(鼠标) | 3 | 完全不同 |

- **Depth** = 知道在"第几层"
- **LCA** = 知道和"谁"是同一家的

---

## 方法创新

### TAE (Trie-aware Absolute Positional Encoding)

编码每个token的**深度信息**：

$$\tilde{e}_{i,l} = e_{i,l} + P_{i,l}$$

### TRE (Topology-aware Relative Positional Encoding)

编码token之间的**LCA距离**：

$$\tilde{A}_{u,v} = \text{Softmax}\left(\frac{(e_u W^Q)(e_v W^K)^T}{\sqrt{d}} + B(d_{LCA}, \delta_u, \delta_v)\right)$$

---

## 实验结果

### 数据集
- Amazon-Beauty
- Amazon-Toys
- Amazon-Food

### Baselines
- TIGER (RQ-VAE)
- LETTER (CIKM'2024)
- CoST (RecSys'2024)
- CofiRec (ArXiv'2025)
- DiscRec (ArXiv'2025)

### 结果
- 三个backbone上平均**8.83%相对提升**
- 训练和推理开销在相同数量级

---

## 核心结论

1. 生成推荐中**结构信息**很重要
2. 显式重建trie拓扑结构比隐式方法更有效
3. TAE和TRE**协同工作**提供全面的结构感知

---

## 与GPR的关联

本文与腾讯GPR生成式推荐高度相关，涉及：
- 层级量化策略
- 位置编码创新
- 对比学习方法

---

## 参考

- [TIGER: Tree-structured Identifiers for Generative Recommendation](https://arxiv.org/abs/2308.10873)
- [LETTER: Hierarchical Quantization with Collaborative Signals](https://arxiv.org/)
- [CoST: Contrastive Learning for Generative Recommendation](https://arxiv.org/)