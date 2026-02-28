---
title: 'A Survey on Dialogue Systems: Recent Advances and New Frontiers'
date: 2018-01-15 16:04:17
tags:
  - dialogue systems
  - NLP
  - survey
---

This is a reading note on dialogue systems, covering recent advances and new frontiers in the field.

## Generative Approaches

Generative approaches are more proper for open-domain conversations, as they can generate responses that never appeared in training data.

Key papers:
- [Neural Responding Machine for Short-Text Conversation](https://arxiv.org/abs/1503.02364)
- [A Neural Conversational Model](https://arxiv.org/abs/1506.05869)

### Dialogue Context

How to make context more useful? An empirical study on context-aware neural conversational models:
- [How to make context more useful?](http://www.aclweb.org/anthology/P17-2036)

### Response Diversity

PaperWeekly 第十八期 --- 提高seq2seq方法所生成对话的流畅度和多样性

**Objective Function:**
- [A Diversity-Promoting Objective Function for Neural Conversation Models](https://arxiv.org/pdf/1510.03055.pdf)
- [An attentional neural conversation model with improved specificity](https://arxiv.org/abs/1606.01292)

**Beam Search:**
- Batra. Diverse beam search: Decoding diverse solutions from neural sequence models
- Generating long and diverse responses with neural conversation models
- A simple, fast diverse decoding algorithm for neural generation

**Rerank:**
- A diversity-promoting objective function for neural conversation models
- A neural network approach to context-sensitive generation of conversational response

**Latent Variable:**
Latent variable is designed to make high-level decisions like topic or sentiment.

### Other Topics (TODO)
- Topics and personalities
- Outside knowledge base
- Interactive learning
- Evaluation (another deep learning model)

## Retrieval Approaches

Retrieval-based approaches are informative and fluent.

- [An information retrieval approach to short text conversation](https://arxiv.org/pdf/1408.6988.pdf)

### Single-turn Response Matching

### Multi-turn Response Matching

## Hybrid Approaches

- A sequence to sequence and rerank based chatbot engine
- An ensemble of retrieval-and generation-based dialog systems

---

*This survey was compiled in 2018 as a reading note on dialogue systems.*