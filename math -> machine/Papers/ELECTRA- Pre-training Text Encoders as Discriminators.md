---
title: "ELECTRA: Pre-training Text Encoders as Discriminators"
authors: ["Kevin Clark", "Minh-Thang Luong", "Quoc V. Le", "Christopher D. Manning"]
year: 2020
arxiv: "2003.10555"
url: https://arxiv.org/abs/2003.10555
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, transformers, llm]
---
## The Core Idea

[[BERT- Pre-training of Deep Bidirectional Transformers|BERT]] wastes most of its training signal. It masks 15% of the tokens, then asks the model to guess what they were. So for every sentence of 512 tokens, the loss only touches ~77 positions. The other 423 positions produce no gradient at all. You pay full price for the forward pass and get 15% of the learning.

ELECTRA changes the question. Instead of "what word was hidden here?", it asks **"is this word real, or did something fake it?"** — for *every single token*. That is a binary classification at all $n$ positions, not a 30k-way softmax at $0.15n$ positions.

To make the question hard, you need fake tokens that are *plausible*. Random words would be too easy — the model would just learn a unigram frequency detector. So they run a small BERT (a "generator") over the masked input, sample its predictions, and paste those samples in. The main model (the "discriminator") then has to spot them.

> [!NOTE] Replaced token detection ^replaced-token-detection
> Corrupt the input by swapping some tokens for samples from a small masked language model, then train an encoder to label each position real/fake. The loss is defined over all positions, which is why it is more sample-efficient than MLM.

The setup looks like a [[Generative Adversarial Networks|GAN]] but is **not adversarial**. The generator is trained with plain maximum likelihood on the MLM objective, not to fool the discriminator. You cannot backprop through the sampling step, and the RL workaround was worse (see ablations). The generator is a helpful teacher, not an opponent.

Two extra things this unlocks:

1. **No pretrain/finetune mismatch.** BERT sees `[MASK]` during pre-training and never again at fine-tuning time. ELECTRA's discriminator never sees `[MASK]` at all — it only sees real text with some words swapped.
2. **Cheap models get good.** ELECTRA-Small: 14M params, 4 days on *one* V100, scores 79.9 GLUE — beating GPT-1 (117M params, 30× the compute, 78.8).

The conceptual lineage is [[Distributed Representations of Words and Phrases (negative sampling)|negative sampling]]. The authors say it plainly: ELECTRA is CBOW-with-negative-sampling scaled up, with a Transformer instead of a bag-of-vectors encoder and a *learned* proposal distribution instead of unigram frequencies.

## The Methodology

Two networks, both Transformer encoders.

**Generator $G$.** A small masked language model. Pick $k$ random positions $\bm{m}$, replace them with `[MASK]`, predict the originals with a softmax over the vocabulary:

$$p_G(x_t \mid \bm{x}) = \frac{\exp\!\left(e(x_t)^\top h_G(\bm{x})_t\right)}{\sum_{x'}\exp\!\left(e(x')^\top h_G(\bm{x})_t\right)}$$

**Discriminator $D$** (this is ELECTRA). A sigmoid head on top of the encoder, one scalar per position:

$$D(\bm{x}, t) = \sigma\!\left(w^\top h_D(\bm{x})_t\right)$$

**Building the input.**

$$m_i \sim \text{unif}\{1,n\}, \quad \bm{x}^{\text{masked}} = \textsc{replace}(\bm{x}, \bm{m}, \texttt{[MASK]})$$
$$\hat{x}_i \sim p_G(x_i \mid \bm{x}^{\text{masked}}) \text{ for } i \in \bm{m}, \quad \bm{x}^{\text{corrupt}} = \textsc{replace}(\bm{x}, \bm{m}, \hat{\bm{x}})$$

**The two losses.** Generator gets standard MLM [[Cross Entropy|cross entropy]]:

$$\mathcal{L}_{\text{MLM}} = \mathbb{E}\left(\sum_{i \in \bm{m}} -\log p_G(x_i \mid \bm{x}^{\text{masked}})\right)$$

Discriminator gets binary cross entropy over **all $n$ positions**:

$$\mathcal{L}_{\text{Disc}} = \mathbb{E}\left(\sum_{t=1}^{n} -\mathbb{1}(x_t^{\text{corrupt}} = x_t)\log D(\bm{x}^{\text{corrupt}},t) - \mathbb{1}(x_t^{\text{corrupt}} \neq x_t)\log(1 - D(\bm{x}^{\text{corrupt}},t))\right)$$

Joint objective over the corpus:

$$\min_{\theta_G, \theta_D} \sum_{\bm{x} \in \mathcal{X}} \mathcal{L}_{\text{MLM}}(\bm{x}, \theta_G) + \lambda\,\mathcal{L}_{\text{Disc}}(\bm{x}, \theta_D)$$

with $\lambda = 50$. The discriminator loss is **not** backpropagated into the generator — you cannot, because of the sampling step.

**Three details that matter:**

- If the generator happens to sample the *correct* token, that position is labelled **real**, not fake. This "moderately improved" downstream results.
- No noise vector fed to the generator (unlike a GAN).
- After pre-training, **throw the generator away**. Only $D$ is fine-tuned.

**Weight sharing.** Only the token and positional **embeddings** are tied between $G$ and $D$; the generator's embedding size equals the discriminator's hidden size. The intuition: MLM's softmax over the full vocabulary densely updates *every* token embedding each step, whereas the discriminator only touches embeddings for tokens that actually appear or get sampled. So let MLM do the embedding learning.

**Generator size.** Making $G$ the same size as $D$ doubles compute per step. Best results come from a generator **1/4 to 1/2** the size of the discriminator (they shrink hidden/FFN/head count, keep depth).

**Hyperparameters (Appendix A, Table 6):**

| | Small | Base | Large |
|---|---|---|---|
| Layers | 12 | 12 | 24 |
| Hidden | 256 | 768 | 1024 |
| Embedding size | 128 | 768 | 1024 |
| Gen. size multiplier | 1/4 | 1/3 | 1/4 |
| Mask % | 15 | 15 | **25** |
| LR | 5e-4 | 2e-4 | 2e-4 |
| Batch | 128 | 256 | 2048 |
| Train steps | 1M | 766K | 400K / 1.75M |

[[Adam- A Method for Stochastic Optimization|Adam]] with $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=$ 1e-6, weight decay 0.01, linear decay, 10k warmup, dropout 0.1. No Next Sentence Prediction — later work showed it does not help. Dynamic masking (decided on the fly), as in [[RoBERTa- A Robustly Optimized BERT Pretraining Approach|RoBERTa]].

ELECTRA-Large uses **25% masking** because at 15% the generator was too accurate and there were too few genuinely-fake tokens to learn from.

Data: Wikipedia + BooksCorpus (3.3B tokens) for most models; XLNet's 33B-token corpus for Large. Fine-tuning: linear classifiers for GLUE, XLNet's QA head for SQuAD, layer-wise LR decay 0.8 (Base/Small) or 0.9 (Large), median of 10 runs reported.

## Ablation Studies and Experiments

**Small models, GLUE dev (Table 1).** Everything trained on comparable FLOPs.

| Model | Train FLOPs | Params | Hardware | GLUE |
|---|---|---|---|---|
| ELMo | 3.3e18 | 96M | 14d, 3× 1080 | 71.2 |
| GPT | 4.0e19 | 117M | 25d, 8× P6000 | 78.8 |
| BERT-Small | 1.4e18 | 14M | 4d, 1 V100 | 75.1 |
| **ELECTRA-Small** | 1.4e18 | 14M | 4d, 1 V100 | **79.9** |
| ELECTRA-Small @ 6h | 8.9e16 | 14M | 6h, 1 V100 | 74.1 |
| BERT-Base | 6.4e19 | 110M | 4d, 16 TPUv3 | 82.2 |
| **ELECTRA-Base** | 6.4e19 | 110M | 4d, 16 TPUv3 | **85.1** |

ELECTRA-Small beats BERT-Small by **+4.8 GLUE** at identical FLOPs. ELECTRA-Base (85.1) beats **BERT-Large** (84.0) at a fraction of the compute. Even 6 hours on one GPU gets 74.1.

**Large models, GLUE dev (Table 2).**

| Model | Train FLOPs | GLUE avg |
|---|---|---|
| BERT (ours) | 7.1e20 (1×) | 87.2 |
| RoBERTa-100K | 6.4e20 (0.90×) | 87.9 |
| RoBERTa-500K | 3.2e21 (4.5×) | 88.9 |
| XLNet | 3.9e21 (5.4×) | 89.1 |
| **ELECTRA-400K** | 7.1e20 (1×) | **89.0** |
| **ELECTRA-1.75M** | 3.1e21 (4.4×) | **89.5** |

ELECTRA-400K matches XLNet/RoBERTa-500K at **less than a quarter of the compute**.

**SQuAD (Table 4).** ELECTRA-1.75M: SQuAD 2.0 test 88.7 EM / 91.4 F1, a new state of the art at the time, beating ALBERT which used 10× the FLOPs. ELECTRA does noticeably better on SQuAD 2.0 than 1.1 — the authors guess that "is this answerable or is the question a plausible fake?" is close to the pre-training task.

**The efficiency ablation — the one that matters (Table 5).** Four "stepping stones" between BERT and ELECTRA, all base-sized:

| | ELECTRA | All-Tokens MLM | Replace MLM | ELECTRA 15% | BERT |
|---|---|---|---|---|---|
| GLUE | **85.0** | 84.3 | 82.4 | 82.4 | 82.2 |

Read this carefully:

- **ELECTRA 15%** (discriminator loss only on the masked positions) drops to 82.4 — basically BERT. **Learning from all tokens is doing almost all the work.**
- **Replace MLM** (MLM but corrupt with generator samples instead of `[MASK]`) gets 82.4 vs BERT's 82.2. So fixing the `[MASK]` mismatch is worth about **+0.2**, not much. BERT's existing 10%-random / 10%-keep trick already partly handles it, but not fully.
- **All-Tokens MLM** (generative, predicts every position, with an explicit copy-probability head) reaches 84.3 — closing most but not all of the gap. The remaining **0.7** is ELECTRA's own contribution beyond just "more positions".

**Model size sweep (Figure 4).** The gap between ELECTRA and BERT *grows as models shrink*. And small ELECTRA trained fully to convergence still lands higher than converged BERT — so this is not merely faster training, it is a better final model. Their speculation: ELECTRA does not have to model the full 30k-way distribution at every position, so parameters go further.

**Weight tying (500k steps, gen = disc size).** No tying 83.6 → tie token embeddings 84.3 → tie everything 84.4. Tying all encoder weights buys +0.1 and forces the generator to be the same size, which is far more expensive. Embeddings only.

**Generator size.** Best at 1/4–1/2 of discriminator. A too-strong generator makes the task too hard; the discriminator "may have to use many of its parameters modeling the generator rather than the actual data distribution."

### What did not work

- **Adversarial generator (REINFORCE, Appendix F).** Trained $G$ to maximise $\mathcal{L}_{\text{Disc}}$ with a policy gradient and a learned baseline. Underperformed maximum likelihood. Two reasons: (a) the adversarial generator is a *worse* MLM — 58% accuracy vs 65% for the MLE one, because [[Simple Statistical Gradient-Following Algorithms (REINFORCE)|REINFORCE]] is sample-inefficient in a 30k-word action space; (b) its output distribution collapsed to low entropy, dumping most mass on one token, so the fakes lacked diversity. Classic text-GAN failure.
- **Two-stage training** (train $G$ alone for $n$ steps, init $D$ from $G$, then train $D$ with $G$ frozen). Scores jump at the switch but never beat joint training. Worse: *without* the weight init, the discriminator sometimes "fail[ed] to learn at all beyond the majority class" — the generator was too far ahead. Joint training gives a free curriculum: weak generator early, strong generator late.
- **Smart masking** (mask rare tokens more, or train a model to pick tokens BERT would find hard). "Fairly minor speedups."
- **Weakening the generator by other means** — raising softmax temperature, or forbidding it from sampling the true token. Neither helped, even though smaller generators did.
- **A sentence-level contrastive head** (leave 20% of inputs uncorrupted, add a "was this whole input corrupted?" head). **Slightly decreased** downstream scores. Surprising, given how much NSP-style objectives were in vogue.
- **ELECTRA as an MLM (Appendix G).** You can invert the optimal-discriminator equation to recover $p_{\text{data}}$ and score words. BERT-Base still wins at masked prediction: 77.9% vs 75.5%. ELECTRA is a better *encoder*, not a better *generator*.

## Worth Remembering

- The **headline lesson is about the loss surface area, not the discriminative framing**. ELECTRA-15% collapsing to BERT-level is the single most informative number in the paper. If you take one design principle away: *make your self-supervised loss touch every position*.
- $\lambda = 50$ is not a typo. The BCE-per-token loss is numerically tiny compared to the vocab-softmax cross entropy, so the discriminator term needs a big multiplier. They searched over [1, 10, 20, 50, 100].
- The mask rate is coupled to generator strength. At Large scale the generator got too good at 15%, so almost nothing was actually replaced and the discriminator had nothing to learn. They bumped to 25%. If you scale this, watch the fraction of positions that are genuinely fake.
- **Compute cost is not free.** You run two networks per step. The generator at 1/4 size is cheap but not zero. All the FLOP comparisons in the paper already include the generator, which is the honest thing to do.
- The authors explicitly argue that papers should report FLOPs and parameter counts alongside accuracy. Appendix E documents their FLOP-counting assumptions (matmul = $2mn$, backward = forward, dense embedding lookups) — worth copying if you ever need to make a compute claim.
- **ELECTRA is unusually good at CoLA** (grammatical acceptability): 71.7 test vs RoBERTa's 67.8. Makes sense — "does this sentence contain something that does not belong?" *is* the pre-training task.
- Their own BERT-Large baseline (87.2) scored below RoBERTa-100K (87.9) at similar compute, which they flag as suspicious — possibly under-tuned, possibly the data. The [[On the Difficulty of Evaluating Baselines|baseline problem]] again.
- Connection worth chasing: ELECTRA is essentially [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)|noise-contrastive estimation]] with a learned, context-conditional noise distribution. The generator *is* the negative sampler, and the quality of your negatives is the whole game — same story as [[A Simple Framework for Contrastive Learning (SimCLR)|SimCLR]]'s hard negatives.
- Practical caveat: ELECTRA gives you an encoder with no language-modelling head. If you want generation, you kept the wrong network. Also, [[Distillation|distilling]] ELECTRA was never tried and the authors suspect it would produce even better tiny models than TinyBERT/MobileBERT.
- Open direction they name: make the generator autoregressive and add a "replaced *span* detection" task, combining this with SpanBERT.

## Links

Related: [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[RoBERTa- A Robustly Optimized BERT Pretraining Approach]] · [[Attention Is All You Need]] · [[Generative Adversarial Networks]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Efficient Estimation of Word Representations (word2vec)]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[Cross Entropy]] · [[Adam- A Method for Stochastic Optimization]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Distillation]] · [[Mode Collapse]] · [[On the Difficulty of Evaluating Baselines]] · [[A Simple Framework for Contrastive Learning (SimCLR)]]

New topics worth writing: Noise-Contrastive Estimation, GLUE benchmark, SQuAD, XLNet and permutation language modelling, ALBERT parameter sharing, SpanBERT, TinyBERT / MobileBERT distillation, text GANs and why they underperform MLE, layer-wise learning rate decay
```
