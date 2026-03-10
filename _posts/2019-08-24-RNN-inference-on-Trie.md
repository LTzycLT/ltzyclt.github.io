---
title: "RNN Inference on Trie"
date: 2019-08-24
categories:
  - tech
tags:
  - RNN
  - Trie
  - TensorFlow
  - Beam Search
math: true
comments: true
---

# 背景

最近在研究自动问答时看到 Google 发的一篇文章 [《Smart Reply: Automated Response Suggestion for Email》](https://research.google.com/pubs/pub45189)。

主要讲在 Gmail 中如何根据对方发送的邮件，利用 seq2seq 的模型自动生成一小段文本让用户直接点选回复。

![图片描述](/images/rnn-inference-on-trie-1.png)

邮件和聊天对话不同，场合更加正式，因此推荐的文本也需要更加正式，回复多样性，有趣性在这里并不是主要考虑点。文中的解决方案是预先用半监督的方法构建了一批高质量回答候选集合 R，每次通过模型给这些回答打分，从中选出概率最高的那个。

Google 的工程对性能的要求是比较严格的，因此如果每次都需要便利一遍集合 R，那肯定是无法接受的。

考虑利用 trie 树优化。假设 seq2seq inference 阶段当前已经生成 word0 word1 ... wordn-1 的前缀，在从词表中选择第 n 个位置的词的时候，只能选择 R 中前缀也同样是 word0 word1 ... wordn-1 的回答中出现过的第 n 个词。相当于在用 R 构建的 trie 上到节点 u 的时候，只能选择所有子节点 v 对应的词，而不是整个词表。

RNN inference 一般同时会使用 beam search 提高生成语句质量。即每次不仅仅保留当前最优解，而是前 k 个最优解都保留，相当于 trie 每次同时从 u0 u1 u2...uk-1 节点向后搜索, 再取概率最大的 v0 v1...vk-1 个节点。

感觉比较有意思，文章没有给出具体代码，所以自己在 tensorflow 中实现了下在 trie 上 beam search 的 inference。可能看下去需要以下一些基础知识。

- [RNN inference](https://en.wikipedia.org/wiki/Recurrent_neural_network)
- [trie](https://en.wikipedia.org/wiki/Trie)
- [beam search](https://en.wikipedia.org/wiki/Beam_search)
- [tensorflow](https://www.tensorflow.org/)

# 实现

基于 tensorflow contrib 模块中 beam_search_decoder.py 修改

详细 PATH: `tensorflow/contrib/seq2seq/python/ops/beam_search_decoder.py`

## 准备工作

找到一批句子，分词，生成大小为 V 的词表，对每个词重新从 0 开始编号。把句子中的词用 0 ~ V-1 替换，构建 trie 树，假设构建的 trie 树节点个数为 N。

## 初始化

在 tensorflow 中 trie 树用稀疏矩阵存储，矩阵 M 的大小为 n * V，如果 trie 树上节点 v 是节点 u 的儿子节点，那么 M[u][v_word_id] = v，v_word_id 是 v 节点上单词编号。

## 流程

改动集中在这个函数

```python
def _beam_search_step(time, logits, next_cell_state, beam_state, batch_size,
                      beam_width, end_token, length_penalty_weight,
                      coverage_penalty_weight):
    """Performs a single step of Beam Search Decoding.
```

主要两步：

(a) 把不在当前节点后续节点中的单词概率给最小，这样 beam 取 top k 就不会考虑不在树上的句子。

(b) 把当前的 u0 u1 ... uk-1 个 trie 树节点转移到 beam 取出的 v0 v1 ... vk-1 个新节点。

### 详细代码

(a) 更新当前步骤的 log_prob

```python
child_nodes = M[u]

child_nodes = self._decoder_trie_fn(beam_state.trie_nodes)  # 找到节点 u 的 M[u] 数组，可以利用查找 embedding 的方式

trie_mask_neg = array_ops.ones_like(step_log_probs, dtype=step_log_probs.dtype)
trie_mask_neg = trie_mask_neg * step_log_probs.dtype.min  # 构造一个值都为 -inf 的对应维度数组
trie_mask_pos = array_ops.zeros_like(step_log_probs, dtype=step_log_probs.dtype)  # 构造 1 个值为 0 的对应维度数组
trie_mask = array_ops.where(gen_math_ops.equal(child_nodes, 0), trie_mask_neg, trie_mask_pos)
step_log_probs = step_log_probs + trie_mask * trie_finished_mask  # 如果单词不在 child 中，则加 -inf，否则加 0
```

(b) 这一步考虑了很久，找到 `_tensor_gather_helper` 函数可以直接帮助实现

word_indices 是选中的 top k 的单词下标

`_tensor_gather_helper` 相当于直接从 child_nodes 中根据 word_indices 取出里面的值，实现了 u -> v 的转移过程

```python
next_trie_nodes = _tensor_gather_helper(
    gather_indices=word_indices,
    gather_from=child_nodes,
    batch_size=batch_size,
    range_size=beam_width * vocab_size,
    gather_shape=[-1])
```