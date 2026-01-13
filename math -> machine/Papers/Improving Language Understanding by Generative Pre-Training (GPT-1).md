---
title: Improving Language Understanding by Generative Pre-Training (GPT-1)
authors:
  - Radford et al.
year: 2018
url: https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf
priority: Must-Read
read_on: 2026-08-21
tags:
  - paper
  - transformers
  - llm
  - GPT1
---
## The Core Idea

Train one big Transformer decoder to do a single dumb thing — predict the next word — on 7,000 unpublished books. Then, for each downstream task, bolt on **one** linear layer and fine-tune the whole thing. That is it. No task-specific encoders, no task-specific attention modules, no hand-built architecture per dataset.

Why this was not obvious in 2018: labelled NLP data is tiny, unlabelled text is infinite, and everyone agreed pre-training helped — but only at the *word* level. [[Auto-regressive models|Word embeddings]] like word2vec/GloVe gave you a fixed vector per word and nothing else. The two open questions were (1) what unsupervised objective learns transferable *sentence and document* level structure, and (2) how do you get that knowledge into a task model without inventing a new architecture each time. The standard answers were bad: ELMo froze the pre-trained model and fed its hidden states as extra features into a fresh task-specific network, which means every task re-learns a pile of new parameters from scratch.

GPT-1's two answers:

1. **Plain [[Auto-regressive models|left-to-right]] language modelling is enough** if the model is a Transformer and the corpus has long contiguous text. The paper is explicit that BooksCorpus was chosen over the 1B Word Benchmark (same size) because the latter is shuffled at sentence level, destroying long-range structure.
2. **Reshape the task, not the model.** Structured inputs (premise + hypothesis, document + question + answer) get flattened into one token sequence with delimiter tokens. The pre-trained network never changes shape. The only new weights in the entire transfer step are the output matrix $W_y$ and the embeddings for the delimiter tokens.

What it unlocks: a *task-agnostic* model beating hand-crafted, task-specific, sometimes 9-model-ensemble baselines on 9 of 12 datasets. This is the template that BERT, GPT-2, GPT-3 and everything after runs on. See [[BERT- Pre-training of Deep Bidirectional Transformers]] for the bidirectional counter-move a few months later.

## The Methodology

### Stage 1 — unsupervised pre-training

Maximise the standard log-likelihood of the next token given the previous $k$:

$$L_1(\mathcal{U}) = \sum_i \log P(u_i \mid u_{i-k},\dots,u_{i-1}; \Theta)$$

This is just [[Cross Entropy|cross-entropy]] on next-token prediction. The forward pass:

$$h_0 = U W_e + W_p$$
$$h_l = \text{transformer\_block}(h_{l-1}), \quad l \in [1, n]$$
$$P(u) = \text{softmax}(h_n W_e^\top)$$

$W_e$ is the token embedding matrix, $W_p$ the position embeddings, and note $W_e$ is **tied** — the same matrix is used to embed input tokens and to project the final hidden state back to vocabulary logits.

> [!NOTE] Decoder-only Transformer
> Take [[Attention Is All You Need|the original Transformer]] and throw away the encoder and the encoder-decoder cross-attention. What is left is a stack of blocks, each = [[Causal Attention|masked self-attention]] + position-wise feedforward, with residual connections and layer norm. The mask stops position $i$ from seeing positions $> i$, which is what makes next-token prediction a valid training signal. ^decoder-only

**Exact spec:** 12 layers, 768-dim hidden states, 12 attention heads, 3072-dim feedforward inner layer, [[Gated Activation|GELU]] activation, learned (not sinusoidal) position embeddings, BPE vocabulary with 40,000 merges. Sequence length 512, batch size 64, 100 epochs over BooksCorpus. Adam with max LR 2.5e-4, linear warmup over 2000 updates then cosine anneal to 0. Dropout 0.1 on residual, embedding and attention paths. Decoupled [[Regularization|L2 weight decay]] $w = 0.01$ on all non-bias, non-gain weights (this is AdamW, pre-publication). Weight init $\mathcal{N}(0, 0.02)$ — the paper says this crude init was fine *because* layer norm is used everywhere. Final token-level perplexity on the corpus: **18.4**.

### Stage 2 — supervised fine-tuning

Feed the labelled input $x_1,\dots,x_m$ through the pre-trained stack, take the final block's activation at the **last** token $h_l^m$, and push it through one linear layer:

$$P(y \mid x_1,\dots,x_m) = \text{softmax}(h_l^m W_y)$$

Maximise $L_2(\mathcal{C}) = \sum_{(x,y)} \log P(y\mid x)$. But they also keep the language modelling loss around as an auxiliary term:

$$L_3(\mathcal{C}) = L_2(\mathcal{C}) + \lambda \cdot L_1(\mathcal{C}), \quad \lambda = 0.5$$

Claimed to improve generalisation and speed up convergence. (The ablation complicates this — see below.)

**Fine-tuning hyperparameters:** LR 6.25e-5, batch size 32, classifier dropout 0.1, linear decay with warmup over 0.2% of training, **3 epochs is enough for most tasks**.

### The input transformations (the actually clever bit)

Every input gets a random-init start token `<s>` and end token `<e>`. Then:

- **Entailment:** `<s> premise $ hypothesis <e>` — `$` is a delimiter token.
- **Similarity:** no natural order between the two sentences, so build *both* orderings, run each through the model independently, and **add the two $h_l^m$ vectors element-wise** before the linear layer. This bakes in symmetry rather than hoping the model learns it.
- **Multiple-choice QA / commonsense:** for context $z$, question $q$ and each candidate answer $a_k$, build `<s> z q $ a_k <e>`. Run each candidate separately, get one scalar logit per candidate, softmax across candidates.

So a $k$-way multiple choice question costs $k$ forward passes.

## Ablation Studies and Experiments

**Headline results.** State of the art on 9 of 12 datasets, often beating ensembles.

| Task | GPT | Previous best |
|---|---|---|
| MNLI-m / -mm | 82.1 / 81.4 | 80.6 / 80.1 (SAN, 3× ensemble) |
| SNLI | 89.9 | 89.3 (ESIM+ELMo, 5×) |
| SciTail | 88.3 | 83.3 |
| QNLI | 88.1 | 82.3 |
| **RTE** | **56.0** | **61.7** ← loses |
| Story Cloze | 86.5 | 77.6 |
| RACE (overall) | 59.0 | 53.3 (BiAttention MRU, 9×) |
| CoLA (Matthews corr.) | 45.4 | 35.0 |
| SST-2 | 91.3 | 93.2 ← loses |
| MRPC (F1) | 82.3 | 86.0 ← loses |
| STS-B (Pearson) | 82.0 | 81.0 |
| QQP (F1) | 70.3 | 66.1 |
| **GLUE overall** | **72.8** | 68.9 |

The CoLA jump (35.0 → 45.4) is the loudest signal: CoLA asks "is this sentence grammatical?", and a model that only saw next-word prediction picked up a lot of syntax for free.

### The three ablations (Table 5, average score over 8 tasks)

| Variant | Avg |
|---|---|
| Transformer + aux LM (full) | 74.7 |
| Transformer, **no pre-training** | **59.9** |
| Transformer, **no aux LM** | **75.0** |
| **LSTM** + aux LM | **69.1** |

Read these carefully:

1. **Pre-training is the whole ballgame.** Removing it costs 14.8 points. STS-B collapses from 82.0 to 30.9, QNLI from 88.1 to 71.2. The architecture alone is worth almost nothing.
2. **The auxiliary LM loss did not work on average.** 75.0 without it vs 74.7 with it — it is *net negative*. It helps only on the larger datasets (MNLI 81.1 → 81.8, QNLI 86.9 → 88.1, QQP 69.8 → 70.3) and hurts on the small ones (CoLA 47.9 → 45.4, MRPC 84.9 → 82.3, STS-B 83.2 → 82.0, RTE 54.4 → 56.0 is the exception). So $\lambda$ should really be tuned per dataset size, and the paper's own headline configuration is not the best one by average score. Nice honest table.
3. **The Transformer is worth 5.6 points over a single-layer 2048-unit LSTM** in the identical framework. The LSTM wins on exactly one dataset (MRPC). The stated reason is longer-range dependency handling — "structured attentional memory" — and it shows most on the long-context tasks.

### Layer transfer curve

Transferring more of the pre-trained stack helps monotonically. Going from embeddings-only to full 12-layer transfer is worth up to **9%** on MultiNLI. Every layer carries useful function; there is no plateau where the top layers become "too task-specific to reuse".

### Zero-shot probes

To show the pre-trained model already *contains* task ability before any fine-tuning, they built heuristic zero-shot solvers and tracked them across pre-training updates:

- **CoLA:** score = average token log-probability, then threshold.
- **SST-2:** append the word `very`, restrict the output distribution to just `positive` and `negative`, take the higher.
- **RACE:** pick the answer with highest average token log-prob given document + question.
- **Winograd (DPRD):** substitute each candidate referent for the pronoun, keep whichever substitution makes the *rest of the sequence* more likely.

All four improve steadily and stably as pre-training proceeds. The LSTM's zero-shot curves are much noisier — evidence the Transformer's inductive bias helps transfer, not just capacity. This section is the direct ancestor of GPT-2's "language models are unsupervised multitask learners".

## Worth Remembering

- **RTE is the loss they admit to.** 56.0 vs 61.7 for a multi-task BiLSTM, on the smallest NLI set (2,490 examples). Their guess: multi-task training would fix it. They did not try. This is exactly the gap MT-DNN and later multi-task fine-tuning schemes filled.
- **The dataset choice was a deliberate design decision, not convenience.** BooksCorpus over 1B Word Benchmark specifically because 1B Word is sentence-shuffled. If you are pre-training and your corpus is chopped into independent short chunks, you are throwing away the thing that makes pre-training work.
- **`$` as delimiter, `<s>`/`<e>` as boundaries, all randomly initialised.** These get gradient signal only during fine-tuning. A handful of new embedding vectors is the entire architectural cost of a new task. Compare BERT's `[CLS]`/`[SEP]`, which are learned during pre-training instead.
- **Last-token pooling.** The classification head reads $h_l^m$ — the final position. In a [[Causal Attention|causal]] model that is the only position that has seen the whole sequence, so it is forced. BERT can use a dedicated `[CLS]` at position 0 because it is bidirectional.
- **AdamW and GELU both appear here before they were standard.** Loshchilov & Hutter's decoupled weight decay was still an arXiv preprint; GELU was a 2016 preprint nobody used.
- Practical caveat: 12 layers, 768 hidden, ~117M parameters, 100 epochs on ~800M words. Small by any modern standard. The strength of the result is that this was enough to beat ensembles of purpose-built models.
- Open question the paper does not touch: they never scale the model. Layer-transfer and zero-shot curves both point at "more is better", but the scaling story waits for GPT-2/GPT-3.
- Practical caveat 2: the multiple-choice input transformation runs one forward pass per candidate answer. Inference cost scales with the number of options.

## Links

Related: [[Attention Is All You Need]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Causal Attention]] · [[Auto-regressive models]] · [[Cross Entropy]] · [[Foundation Models]] · [[In Context Learning]] · [[Seq2Seq models]] · [[Query, Key, and Value (QKV)]] · [[Regularization]] · [[Gated Activation]] · [[Linear Projection]] · [[Distillation]]

New topics worth writing: Transfer learning and fine-tuning, Byte-Pair Encoding, Layer Normalization, AdamW and decoupled weight decay, GELU activation, Perplexity, ELMo and contextual word vectors, GLUE benchmark, Zero-shot evaluation, Learning rate warmup and cosine schedules, Auxiliary losses and multi-task learning, Weight tying in language models
