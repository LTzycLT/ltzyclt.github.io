---
title: "autoresearch：让 AI Agent 自主做深度学习研究"
date: 2026-03-17
categories:
  - AI Research
  - Engineering
tags:
  - AI Agent
  - LLM Training
  - Karpathy
  - Muon
  - Transformer
math: true
comments: true
---

> **项目来源**: [@karpathy](https://github.com/karpathy/autoresearch), March 2026
> **核心理念**: 给 AI Agent 一个真实的 LLM 训练环境，让它在无人监督的情况下自主实验，人类睡一觉醒来看结果

## 为什么要看这个项目

Karpathy 三月初发了 autoresearch 之后，朋友圈和 Twitter 都刷屏了。花了个周末把代码从头到尾读了一遍，记录一下。

autoresearch 的核心想法很直接：**把 AI agent 放进一个可运行的 LLM 训练环境，让它自主修改代码、运行实验、评估结果，不断迭代，人类只需要定义研究目标**。项目刻意保持极简——整个仓库只有三个核心文件，没有分布式训练，没有复杂配置，一张 GPU，一个文件，一个指标。这种极简背后有一个我觉得很对的工程直觉：**约束越明确，自主研究越可行**。

类似的事情之前也有人做过——Sakana 的 AI Scientist、各种 NAS 方法、DeepMind 的 FunSearch——但 autoresearch 的路子不太一样。它给 agent 的自由度高很多，不是在预定义的搜索空间里挑选，而是直接让 agent 改代码。架构、优化器、超参数都可以动，当然也更难控制。

## 背景与动机

传统深度学习研究的瓶颈不在于缺乏想法，而在于**实验吞吐量**。研究员需要手动修改代码、等待训练、分析结果、再改、再跑。这个循环慢且累。

autoresearch 的出发点是：如果训练时间可以被固定在 5 分钟以内，且评估指标足够客观，那么 AI agent 就可以接管整个「提出想法 → 实现 → 验证 → 决策」的循环。人类只需要在高层次上定义研究方向（通过 `program.md`），然后让 agent 跑一晚上。

这件事在 2026 年初才变得可行，原因有两个：一是现代 LLM 的代码能力已经够强，能可靠地修改 PyTorch 训练脚本而不引入 bug；二是 flash attention 等技术让 5 分钟的训练预算下也能跑到有意义的规模。

## 整体架构：三文件极简设计

```
prepare.py   — 固定的常量、数据准备、tokenizer、dataloader、评估（不可修改）
train.py     — Agent 唯一可以修改的文件（模型、优化器、超参数）
program.md   — 人类编写的 Agent 指令（研究目标、行为规范）
```

这个设计有一个清晰的**权限边界**：

- `prepare.py` 是**不变量**。评估函数 `evaluate_bpb` 是 ground truth，不能被 agent 触碰，否则 agent 可以通过修改评估函数来「作弊」。
- `train.py` 是**搜索空间**。架构、优化器、超参数全部都可以改，agent 在这里发挥创造力。
- `program.md` 是**研究策略层**。这是人类唯一需要编程的东西——不是 Python，而是一份 Markdown 指令文档。

## 评估指标：Bits Per Byte (BPB)

项目使用 **val_bpb**（validation bits per byte）作为唯一指标，越低越好。

```python
@torch.no_grad()
def evaluate_bpb(model, tokenizer, batch_size):
    """
    Sums per-token cross-entropy (in nats), sums target byte lengths,
    then converts nats/byte to bits/byte.
    Special tokens (byte length 0) are excluded.
    Uses fixed MAX_SEQ_LEN so results are comparable across configs.
    """
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(val_loader)
        loss_flat = model(x, y, reduction='none').view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += nbytes.sum().item()
    return total_nats / (math.log(2) * total_bytes)
```

选 BPB 不选 perplexity，关键原因是 **BPB 与 vocab size 无关**。不同 vocab size 下，per-token cross-entropy 不可直接比较——更大的 vocab 意味着 softmax 在更多类上计算，但每个 token 覆盖更多文本，两者效果混在一起。BPB 通过归一化到字节级别消除了这种不可比性，让不同 tokenizer 方案的模型可以公平对决。

固定时间预算（300s）+ 固定评估协议（BPB, EVAL_TOKENS=40×524288），这样不管 agent 怎么折腾 tokenizer，结果都能直接比。

说完评估，再看看数据这块。

## 数据：BOS 对齐的 Best-Fit Packing Dataloader

数据来源是 `climbmix-400b-shuffle`，Karpathy 自己整理的混合预训练数据集。

Dataloader 这里有个巧妙的设计：**Best-Fit Packing**。

```python
def make_dataloader(tokenizer, B, T, split, buffer_size=1000):
    """
    BOS-aligned dataloader with best-fit packing.
    Every row starts with BOS. Documents packed using best-fit to minimize cropping.
    When no document fits remaining space, crops shortest doc to fill exactly.
    100% utilization (no padding).
    """
```

传统 dataloader 通常会 padding 或者截断文档，存在 token 浪费。这里用一个 buffer 维护候选文档，对每一行用「最大适配文档」（largest fitting document）来填充，当没有文档能完整放入剩余空间时，才裁剪最短文档的前面部分填满。结果是 **token 利用率 100%，无 padding**。

配合 `pin_memory` + `non_blocking copy` 实现了 CPU-GPU 异步数据传输：

```python
cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=True)
gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device="cuda")
# ...
gpu_buffer.copy_(cpu_buffer, non_blocking=True)
```

## 模型架构：2026 年标准 GPT 配置

默认配置是一个 **8 层、~50M 参数**的 GPT，agent 可以自由修改。看完这些默认配置，感觉是把 2026 年初各种小 trick 集大成了——以下是几个有意思的架构选择：

### Value Residual（来自 ResFormer）

```python
# Value residual: mix in value embedding with input-dependent gate per head
if ve is not None:
    ve = ve.view(B, T, self.n_kv_head, self.head_dim)
    gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
    v = v + gate.unsqueeze(-1) * ve
```

每隔一层，注入一个额外的 value embedding（来自 token embedding 空间）。这个 embedding 通过一个输入相关的 sigmoid gate 混入。这是 ResFormer 的思路：让浅层的 token 表示直接参与深层的 attention 计算，缓解深层网络的信息遗忘问题。

gate 初始化使得 sigmoid(0)×2 = 1.0，即训练初期 value embedding 以等量权重叠加到 attention value 上（v_new = v + ve），随训练逐步学习最优混合比例。

### ResFormer 全局残差（resid_lambdas + x0_lambdas）

```python
x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
```

每层的输入不只来自上一层的输出，还有一个来自 embedding 层 `x0` 的加权贡献。两个标量（`resid_lambdas`, `x0_lambdas`）都是可学习的，初始化为 (1.0, 0.1)。这是一种轻量级的全局残差连接，帮助梯度更顺畅地流回浅层。

### SSSL 窗口注意力模式

```python
WINDOW_PATTERN = "SSSL"  # S=short window (seq/2), L=long (full seq)
```

四层一个周期：前三层用半序列长度的窗口注意力，第四层用全局注意力。最后一层始终强制为全局注意力。窗口注意力使用 Flash Attention 3（H100 用 Hopper 专版，其他 GPU 用 community 版本）。

这是计算效率与长程依赖的折中。需要注意的是，因为这里窗口大小 w = seq_len/2 是随序列长度线性增长的，复杂度仍然是 O(n²) 量级（具体是 O(n × n/2) = O(n²/2)），主要收益来自常数倍加速。只有当窗口大小固定为与 n 无关的常数时才能实现 O(n) 复杂度——但在 Flash Attention 实现下，即使是这种 n/2 窗口也能带来明显的 wall-clock 提速。

### QK Norm + RoPE

```python
q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
q, k = norm(q), norm(k)
```

先施加 RoPE 旋转位置编码，再对 Q、K 做 RMS Norm。QK Norm 是稳定注意力训练的常用技巧，防止注意力 logits 过大。

### Squared ReLU

```python
def forward(self, x):
    x = self.c_fc(x)
    x = F.relu(x).square()  # relu²
    x = self.c_proj(x)
    return x
```

MLP 使用 `relu(x)²` 而非 GeLU。相比 GeLU，squared ReLU 的稀疏性更强，且计算更简单，在某些配置下表现更好。

### Softcap Logits

```python
softcap = 15
logits = softcap * torch.tanh(logits / softcap)
```

输出 logits 通过 15×tanh(x/15) 软截断，防止极端 logit 值导致的数值不稳定。这是 Gemma 系列模型引入的技巧。

## 优化器：MuonAdamW

老实说这部分我看了好几遍才理清楚。整个优化器把好几个独立的想法缝合在了一起，但缝合得还挺漂亮的。

### Muon：矩阵参数的正交化优化器

```python
# Polar express orthogonalization
polar_express_coeffs = [
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    ...  # 5 sets of (a, b, c)
]

for a, b, c in polar_express_coeffs[:ns_steps]:
    A = X.mT @ X
    B = b * A + c * (A @ A)
    X = a * X + X @ B
```

Muon（Momentum + Orthogonalization Updates）的核心思想是：对矩阵参数的**梯度**做正交化处理后再更新。正交化通过 Newton-Schulz 迭代实现，代码中用了 5 步的"Polar Express"多项式近似（预先计算好的系数），在 bfloat16 下高效运行。

正交化的直觉是：**梯度矩阵**的不同奇异值方向应当被均等对待——通过提取梯度的极分解（polar decomposition）中的正交因子作为更新方向，避免某些方向的更新因奇异值差异而过大或过小。

### NorMuon：方差归一化

```python
# NorMuon variance reduction
v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
second_momentum_buffer.lerp_(v_mean, 1 - beta2)
step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()
g = g * final_scale.to(g.dtype)
```

在正交化之后，对每行（或列，取决于矩阵方向）的更新幅度做归一化，类似 Adam 的二阶矩估计，但作用在行/列维度而非元素维度。用 beta2 动量平滑方差估计。

### Cautious Weight Decay

```python
# Only apply weight decay when parameter and gradient are aligned
mask = (g * stacked_params) >= 0
stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)
```

这个设计挺有意思：只在梯度没有把参数推向零的时候才施加 weight decay。具体来说，当梯度已经在把参数推向零（参数为正但梯度为负，或反过来），此时 weight decay 的衰减效果与梯度方向一致，双重惩罚可能过度。所以用一个 mask 把这种情况过滤掉，只在梯度和参数同号时才 decay。

### 分层学习率策略

| 参数组 | 优化器 | 默认 LR |
|--------|--------|---------|
| lm_head (unembedding) | AdamW | 0.004 |
| wte (embedding) | AdamW | 0.6 |
| value_embeds | AdamW | 0.6 |
| 矩阵参数 | Muon | 0.04 |
| resid_lambdas | AdamW | 0.005 (scalar×0.01) |
| x0_lambdas | AdamW | 0.5 |

所有 AdamW 的 LR 都乘以 $$1/\sqrt{model\_dim/768}$$ 做维度缩放，这是 μP（maximal update parametrization）思想的简化版本：更宽的模型应该用更小的学习率。

### torch.compile 兼容性技巧

```python
# 0-D CPU tensors to avoid torch.compile recompilation when values change
self._adamw_step_t = torch.tensor(0.0, dtype=torch.float32, device="cpu")
```

优化器内部用 0-D CPU tensor 存储每步变化的超参数值，避免 `torch.compile` 因为标量变化而重新编译（每次 `.fill_()` 更新值而非传入新标量）。这种细节不起眼，但在实际训练中能省不少编译开销。

## LR Schedule 设计

```python
WARMUP_RATIO = 0.0      # 无 warmup
WARMDOWN_RATIO = 0.5    # 后 50% 时间线性 decay 到 0
FINAL_LR_FRAC = 0.0     # 最终 LR 为 0
```

后半段（50%的时间）做线性 LR decay 到 0。这是一个激进的 cooldown 策略。在固定 5 分钟的预算下，给足够的 decay 时间有助于充分利用训练窗口。

Muon 的 momentum 还有一个 warmup：前 300 步从 0.85 线性升到 0.95，减少早期大动量带来的不稳定。

## Agent 行为规范（program.md）

前面都是相对传统的 ML 工程，但 `program.md` 才是这个项目真正让我觉得新鲜的东西——**人类通过编写 Markdown 来编程 AI 的研究行为**。

几个关键设计：

**LOOP FOREVER 原则**：
```
NEVER STOP: Once the experiment loop has begun, do NOT pause to ask the human
if you should continue. You are autonomous. The loop runs until the human
interrupts you, period.
```
Agent 不应该等待人类确认，不应该询问「要继续吗」。这是纯自主模式。

**Git 分支管理实验**：每次实验前 commit，结果好则保留（branch advances），结果差则 `git reset`。这样 git history 就是实验日志，每个 commit 代表一个「有效进展」。

**复杂度判断原则**：
```
Simplicity criterion: A 0.001 val_bpb improvement that adds 20 lines of hacky code?
Not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep.
```
引入了对「代码复杂度」的明确评估——提升效果要与引入复杂度相称。这个原则让我想到 Occam's razor 在工程上的实践化。

**结果记录格式**（TSV，不用 CSV 因为 description 里可能有逗号）：
```
commit  val_bpb  memory_gb  status  description
```

## 工程细节：GC 管理

```python
if step == 0:
    gc.collect()
    gc.freeze()
    gc.disable()
elif (step + 1) % 5000 == 0:
    gc.collect()
```

Python 的 GC 在训练过程中可能导致约 500ms 的停顿。通过在第一步 `freeze()`（冻结已有对象，不再追踪）+ `disable()`（关闭自动 GC），消除这个干扰。每 5000 步做一次手动 GC。我以前在训练代码里也这么干过，但确实第一次在开源项目里看到有人把这个写进来。

## 实验规模估算

按 README 的说法，H100 上每个实验约 5 分钟 → 每小时 12 个实验 → 睡 8 小时可以跑约 **100 个实验**。

这个数字挺让人兴奋的：人类研究员一天能做 2-3 个精心设计的实验，而 autoresearch 可以在同等时间内探索 100 个想法。即使其中大多数想法是失败的，这种探索密度也可能找到人类不会轻易尝试的改进。

## 我的一些思考

下面是我自己瞎琢磨的一些东西，不一定对。

**5 分钟的固定预算，到底在优化什么？**

固定 wall-clock 时间是一个很实用的选择——毕竟真正的成本是 GPU-hours 而不是 FLOPS。但这里有个微妙的问题：agent 会天然倾向于「在 5 分钟内训更多 token」的方向优化，比如减小模型、提高吞吐、减少 VRAM 换取更大 batch。这跟「在固定 token 数下达到最低 loss」其实是两回事，可能导致不同的架构偏好。

另外，如果平台变了（比如从 H100 换到 4090），之前找到的最优配置还是最优的吗？固定时间预算让结果不可跨平台迁移，这是有意识的 trade-off。我倾向于认为在工程实践中这不是大问题——反正你最终也要在目标硬件上重新调参。

**program.md：用自然语言编程研究策略**

`program.md` 本质上是一种「元编程」——它不告诉 agent 具体改什么，而是定义了如何判断好坏、如何记录、如何决策。这个接口有很多可以玩的地方：比如写入已知的「负面发现」（哪些方向试过没用）、设定优先探索方向、甚至给出 budget allocation 策略（前 50% 时间做大胆探索，后 50% 做精细调参）。

一个我比较好奇的问题是：随着实验积累，`results.tsv` 里有越来越多的历史数据，agent 能不能从中归纳规律来指导下一步探索？目前 agent 主要靠 LLM 的内部知识提想法，没有显式利用历史。一个可能的方向是把 results.tsv 结构化后注入 context，或者用类似 bandit 的算法来平衡探索与利用。

**Muon 在 bfloat16 下跑 5 步正交化，够不够？**

Polar Express 正交化在 bfloat16 下只跑 5 步，数值精度是否足够是一个可以实验的点。同时，NorMuon 的行/列维度归一化与 AdamW 的元素级归一化有本质差异——在很宽（width >> depth）或很窄的矩阵上，两种策略的效果可能不同。

代码里 Muon 组是按「相同 shape 的参数」分组的，相同 shape 的参数会被 stack 在一起做批量正交化。如果 agent 修改了 n_kv_head（GQA），会产生更多不同 shape 的 Q/K/V 矩阵，对 Muon 分组有影响——不知道这个会不会成为 agent 探索 GQA 时的一个隐形障碍。

**自主研究的边界在哪？**

权限边界的核心是防止 agent「hack 评估指标」而不是「提升真实能力」。但 BPB 本身也有局限——一个过拟合 val shard 分布的模型可能有好的 BPB 但泛化性差。val shard 被固定（`VAL_SHARD = MAX_SHARD = 6542`），如果 agent 发现了某种结构性的过拟合方式，这个评估框架可能检测不到。

还有一个更根本的问题：autoresearch 在 ~50M 参数规模上搜索最优配置，但这些发现能迁移到更大规模吗？深度学习里有太多「小模型最优 ≠ 大模型最优」的案例。代码中用了 $$1/\sqrt{model\_dim/768}$$ 维度缩放（μP 思想的简化版），某种程度上考虑了可迁移性，但远远不够。我觉得这可能是 autoresearch 范式最大的未解难题——**agent 发现的"好东西"在大模型上还灵不灵？**

最后一个大胆点的想法：目前 agent 的探索基本都在 Transformer 框架内（改层数、改 head、改激活函数……），它有没有可能跳出这个范式？比如试试 state space model 或者其他非 attention 架构？这可能取决于 LLM 自身的知识边界——agent 的创造力上限就是它训练数据里见过的东西。

## 总结

读完这个项目，印象最深的不是某个具体的架构或优化器技巧，而是它展示的一种可能性：用 Markdown 来「编程」AI 的研究行为。三个文件、一张卡、一个指标，把约束定义清楚了，自主研究就变得可行了。技术栈本身（Value Residual + QK Norm + RoPE + FA3 + Muon + BPB）也可以当作 2026 年初 LLM 训练的工程最佳实践速查表来用——就算不关心 AI 自主研究这个方向，光看 train.py 里的这些实现细节也不亏。
