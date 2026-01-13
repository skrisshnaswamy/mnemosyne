---
title: "BERT: Pre-training of Deep Bidirectional Transformers"
authors:
  - Jacob Devlin
  - Ming-Wei Chang
  - Kenton Lee
  - Kristina Toutanova
year: 2018
arxiv: "1810.04805"
url: https://arxiv.org/abs/1810.04805
priority: Must-Read
read_on: 2026-08-21
tags:
  - paper
  - transformers
  - llm
  - BERT
---
## The Core Idea

Before BERT, the standard way to pre-train a language model was to read text left to right and predict the next word. That is what OpenAI GPT did. The problem: to understand a word, you often need the words *after* it too. "He went to the **bank** to deposit money" — the disambiguating evidence is on the right.

ELMo tried to fix this by training a left-to-right model and a right-to-left model separately, then gluing their outputs together at the end. That is shallow. Each half never sees the other half's context while it is thinking.

Why did nobody just train a deep bidirectional model? Because with normal next-word prediction it is cheating. If every layer can attend to every position, then in a multi-layer network the word you are predicting can "see itself" through a path of intermediate representations. The loss goes to zero and the model learns nothing.

The trick: **stop predicting the next word. Corrupt the input instead, and predict what was corrupted.** Randomly hide 15% of the tokens and ask the model to fill them in, using everything on both sides. This is the Cloze task from 1953 reading-comprehension research. Now bidirectional attention is safe, because the answer is genuinely not in the input.

That single change unlocks a Transformer encoder where **every layer** conditions on both directions. Add a second, cheap objective — predict whether two text spans were actually adjacent in the corpus — and you get a model that also understands relationships between sentences.

What it buys you: one pre-trained model, fine-tuned with **one extra output layer**, beats bespoke hand-engineered architectures on 11 different NLP tasks. GLUE score 72.8 → 80.5. This is the paper that made "download a checkpoint and fine-tune" the default workflow in NLP.

> [!NOTE] Masked Language Model (MLM)
> A pre-training objective where a random subset of input tokens is replaced with a `[MASK]` placeholder, and the model must recover the original tokens from the surrounding context on both sides. Unlike next-word prediction, it permits deep bidirectional conditioning. ^masked-language-model

## The Methodology

### Architecture

A plain Transformer **encoder** stack — [[Attention Is All You Need]] with the decoder deleted and no causal mask. No [[Causal Attention]] anywhere; every token attends to every token. Standard [[Query, Key, and Value (QKV)]] self-attention.

| | Layers $L$ | Hidden $H$ | Heads $A$ | Params |
|---|---|---|---|---|
| BERT$_{\text{BASE}}$ | 12 | 768 | 12 | 110M |
| BERT$_{\text{LARGE}}$ | 24 | 1024 | 16 | 340M |

BASE was deliberately sized to match OpenAI GPT so the comparison would be honest — the *only* meaningful architectural difference is the attention mask.

Activation is `gelu`, not `relu`. Dropout 0.1 everywhere.

### Input representation

WordPiece tokenisation, 30,000-token vocabulary. Every sequence starts with `[CLS]`. Two text spans are packed into one sequence and separated by `[SEP]`.

Each token's input vector is a **sum of three [[Linear Projection|embeddings]]**:

$$E_{\text{input}} = E_{\text{token}} + E_{\text{segment}} + E_{\text{position}}$$

The segment embedding is a learned vector saying "you are in span A" or "you are in span B". Position embeddings are **learned**, not the sinusoids of the original Transformer.

Notation used throughout: $C \in \mathbb{R}^H$ is the final hidden vector at `[CLS]`; $T_i \in \mathbb{R}^H$ is the final hidden vector at token $i$.

### Objective 1: Masked LM

Pick 15% of WordPiece positions at random. For each chosen position $i$:

- 80% of the time → replace with `[MASK]`
- 10% of the time → replace with a **random** token
- 10% of the time → leave it **unchanged**

Then feed $T_i$ through a softmax over the 30k vocabulary and take [[Cross Entropy]] loss against the true token.

Why the 80/10/10 mess? Because `[MASK]` never appears at fine-tuning time, so a model trained on 100% `[MASK]` would face a distribution shift. The random-replacement and keep-unchanged cases mean the model **does not know which positions it will be graded on**, so it is forced to build a good contextual representation of *every* token, not just the obviously-corrupted ones. Random replacement only touches 1.5% of tokens overall (10% of 15%), so it does not poison the data much.

Note this only predicts 15% of tokens per batch, versus 100% for a left-to-right LM — so MLM needs more steps to converge.

### Objective 2: Next Sentence Prediction (NSP)

Sample two spans A and B. 50% of the time B genuinely follows A (`IsNext`), 50% of the time B is a random span from elsewhere in the corpus (`NotNext`). Feed $C$ (the `[CLS]` vector) through a binary classifier.

```
Input  = [CLS] the man went to [MASK] store [SEP]
         he bought a gallon [MASK] milk [SEP]     → IsNext

Input  = [CLS] the man [MASK] to the store [SEP]
         penguin [MASK] are flight ##less birds [SEP] → NotNext
```

Trivially generated from any raw text. Aimed at question answering and entailment, which are fundamentally about *pairs* of texts.

**Total loss** = mean MLM likelihood + mean NSP likelihood. One sum, trained jointly, via ordinary [[Backpropagation]].

### Pre-training data and hyperparameters

- BooksCorpus (800M words) + English Wikipedia (2,500M words, text passages only, no lists/tables/headers). 3.3B words total.
- **Document-level corpus is critical** — a shuffled sentence-level corpus like Billion Word Benchmark cannot give you long contiguous spans.
- Batch: 256 sequences × 512 tokens = 128,000 tokens/batch. 1,000,000 steps ≈ 40 epochs.
- Adam, lr $10^{-4}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, L2 weight decay 0.01, 10k warmup steps, linear decay.
- Speed hack: 90% of steps at sequence length 128, last 10% at 512 (attention is quadratic in length, so long sequences are disproportionately expensive). The 512-length phase mainly exists to learn the higher position embeddings.
- BASE: 4 Cloud TPUs (16 chips), 4 days. LARGE: 16 Cloud TPUs (64 chips), 4 days.

### Fine-tuning

Swap in task inputs, add one output layer, train **all** parameters end to end. Cheap: ≤1 hour on one Cloud TPU for any result in the paper.

- **Classification (GLUE):** take $C$, add $W \in \mathbb{R}^{K \times H}$, loss $= \log(\text{softmax}(CW^\top))$. That $W$ is the *only* new parameter matrix.
- **SQuAD span extraction:** learn a start vector $S$ and end vector $E$, both in $\mathbb{R}^H$. Question goes in segment A, passage in segment B.
$$P_i = \frac{e^{S \cdot T_i}}{\sum_j e^{S \cdot T_j}}$$
Span score is $S \cdot T_i + E \cdot T_j$; take the max over $j \ge i$. Objective is the sum of log-likelihoods of the correct start and end.
- **SQuAD 2.0 (unanswerable questions):** treat "no answer" as a span starting and ending at `[CLS]`. Predict a real answer only when $\hat{s}_{i,j} > s_{\text{null}} + \tau$, where $s_{\text{null}} = S \cdot C + E \cdot C$ and $\tau$ is tuned on dev to maximise F1.
- **SWAG multiple choice:** build 4 sequences (context + each candidate ending), one learned vector dotted with each $C$, softmax over the 4 scores.

Fine-tuning grid: batch ∈ {16, 32}, lr ∈ {5e-5, 3e-5, 2e-5}, epochs ∈ {2, 3, 4}. Large datasets (100k+ examples) are far less sensitive to these than small ones.

## Ablation Studies and Experiments

### Headline numbers

**GLUE test set:**

| System | MNLI-m/mm | QNLI | MRPC | CoLA | RTE | Avg |
|---|---|---|---|---|---|---|
| BiLSTM+ELMo+Attn | 76.4/76.1 | 79.8 | 84.9 | 36.0 | 56.8 | 71.0 |
| OpenAI GPT | 82.1/81.4 | 87.4 | 82.3 | 45.4 | 56.0 | 75.1 |
| BERT$_{\text{BASE}}$ | 84.6/83.4 | 90.5 | 88.9 | 52.1 | 66.4 | **79.6** |
| BERT$_{\text{LARGE}}$ | 86.7/85.9 | 92.7 | 89.3 | 60.5 | 70.1 | **82.1** |

Official leaderboard: GPT 72.8 → BERT$_{\text{LARGE}}$ 80.5.

**SQuAD 1.1:** BERT$_{\text{LARGE}}$ ensemble + TriviaQA pre-fine-tuning hits 93.2 test F1, beating the #1 leaderboard ensemble (91.7) and human performance (91.2). A **single** BERT beats the top ensemble. Dropping TriviaQA costs only 0.1–0.4 F1 — the data augmentation was almost irrelevant.

**SQuAD 2.0:** 83.1 test F1 vs 78.0 previous best. +5.1.

**SWAG:** 86.3 test vs ESIM+ELMo 59.2 and GPT 78.0. Beats one expert human (85.0), below 5-annotator human (88.0).

### Ablation 1 — which pre-training task matters (Table 5, BASE architecture, dev set)

| | MNLI-m | QNLI | MRPC | SST-2 | SQuAD F1 |
|---|---|---|---|---|---|
| BERT$_{\text{BASE}}$ | 84.4 | 88.4 | 86.7 | 92.7 | 88.5 |
| No NSP | 83.9 | 84.9 | 86.5 | 92.6 | 87.9 |
| LTR & No NSP | 82.1 | 84.3 | 77.5 | 92.1 | 77.8 |
| + BiLSTM on top | 82.1 | 84.1 | 75.7 | 91.6 | 84.9 |

Two clean readings:

1. **NSP helps, moderately.** Removing it costs 3.5 points on QNLI and 0.5 on MNLI. Real, but not the main event. (Later work — RoBERTa — would show NSP is largely dispensable.)
2. **Bidirectionality is the big win.** Comparing "No NSP" to "LTR & No NSP" with identical data and hyperparameters: MRPC drops 86.5 → 77.5, SQuAD drops 87.9 → 77.8. Token-level tasks collapse without right-side context, which is intuitive — you cannot pick an answer span if each token only knows what came before it.

**What did not work:** bolting a randomly-initialised BiLSTM on top of the LTR model to recover right-context. It rescued SQuAD partially (77.8 → 84.9) but stayed 3.6 F1 behind bidirectional pre-training, and it **hurt** every GLUE task (MRPC 77.5 → 75.7). The lesson: you cannot patch in bidirectionality at fine-tuning time with fresh parameters. It has to be baked into the pre-trained representation.

The authors also argue against the ELMo-style solution of concatenating separate LTR and RTL models: (a) twice the cost, (b) incoherent for QA since the RTL model cannot condition the answer on the question, (c) strictly weaker, because a deep bidirectional model fuses both directions at *every* layer, not just the last one.

### Ablation 2 — model size (Table 6, dev accuracy, 5 restarts averaged)

| L | H | A | MLM ppl | MNLI-m | MRPC | SST-2 |
|---|---|---|---|---|---|---|
| 3 | 768 | 12 | 5.84 | 77.9 | 79.8 | 88.4 |
| 6 | 768 | 12 | 4.68 | 81.9 | 84.8 | 91.3 |
| 12 | 768 | 12 | 3.99 | 84.4 | 86.7 | 92.9 |
| 12 | 1024 | 16 | 3.54 | 85.7 | 86.9 | 93.3 |
| 24 | 1024 | 16 | 3.23 | 86.6 | 87.8 | 93.7 |

Strictly monotonic improvement, **including on MRPC which has only 3,600 labelled examples**. This was the surprise. Prior feature-based work (Peters 2018b, Melamud 2016) had found scaling stopped helping. The authors' hypothesis: when you fine-tune the whole model and add only a handful of fresh randomly-initialised parameters, tiny downstream datasets *can* exploit a huge pre-trained representation. Feature-based approaches, which freeze the backbone, cannot.

This is the empirical seed of the whole scaling-laws era.

### Ablation 3 — training steps (Figure 5)

- 500k → 1M steps buys almost 1.0% MNLI accuracy. So yes, the enormous pre-training budget is genuinely needed.
- MLM does converge slower than LTR (it only gets gradient signal on 15% of tokens), but MLM's **absolute** accuracy overtakes LTR almost immediately. The slower convergence is a non-issue.

### Ablation 4 — masking strategy (Table 8, dev)

| Mask | Same | Rnd | MNLI (FT) | NER (FT) | NER (feature-based) |
|---|---|---|---|---|---|
| 80% | 10% | 10% | 84.2 | 95.4 | 94.9 |
| 100% | 0% | 0% | 84.3 | 94.9 | 94.0 |
| 80% | 0% | 20% | 84.1 | 95.2 | 94.6 |
| 80% | 20% | 0% | 84.4 | 95.2 | 94.7 |
| 0% | 20% | 80% | 83.7 | 94.8 | 94.6 |
| 0% | 0% | 100% | 83.6 | 94.9 | 94.6 |

**Fine-tuning is remarkably robust** to the masking recipe — MNLI moves inside a 0.8-point band. The famous 80/10/10 split is close to irrelevant if you fine-tune.

Where it *does* matter is the feature-based setting: 100% `[MASK]` drops NER feature-based F1 to 94.0, because the frozen model has never seen an uncorrupted sentence and cannot adjust. Also, 100% random replacement is clearly bad (83.6 MNLI) — you need real `[MASK]` tokens for the model to know something is missing.

### Ablation 5 — feature-based use (Table 7, CoNLL-2003 NER)

Freeze BERT, extract activations, feed a fresh 2-layer 768-d BiLSTM classifier:

| Features used | Dev F1 |
|---|---|
| Embeddings only | 91.0 |
| Last hidden layer | 94.9 |
| Second-to-last hidden | 95.6 |
| Weighted sum of all 12 | 95.5 |
| Weighted sum of last 4 | 95.9 |
| **Concat last 4 hidden** | **96.1** |
| *Full fine-tuning (BASE)* | *96.4* |

Concatenating the top four layers lands 0.3 F1 behind full fine-tuning. Two takeaways: BERT works fine as a frozen feature extractor if you need to precompute representations once and run many cheap experiments on top; and the **last** layer is not the best layer — it is over-specialised to the MLM objective. Second-to-last beats last.

## Worth Remembering

**Limitations the authors admit.**

- MLM only supervises 15% of tokens per batch, so pre-training is sample-inefficient compared to a left-to-right LM that gets signal from every position. (ELECTRA later attacked exactly this.)
- The `[MASK]` token creates a pre-train/fine-tune mismatch. The 80/10/10 recipe is a patch, and the ablation shows it is a patch that mostly does not matter when fine-tuning.
- BERT$_{\text{LARGE}}$ fine-tuning is **unstable on small datasets**. The authors ran several random restarts (same checkpoint, different data shuffling and classifier init) and picked the best on dev. If you fine-tune a large BERT on a 3k-example task and get a bad result, try again with a different seed before concluding anything.

**Practical caveats.**

- BERT is an encoder. It cannot generate text autoregressively — there is no [[Auto-regressive models|autoregressive]] factorisation to sample from. For generation you want a decoder, i.e. the GPT line.
- Attention is $O(n^2)$ in sequence length, hard-capped at 512 tokens by the learned position embeddings. There is no extrapolation beyond 512 because those embedding rows simply do not exist. Compare [[Flash Attention]] for the memory side of the same problem.
- Every downstream task gets its **own** fine-tuned copy of all 110M/340M parameters. No parameter sharing across tasks. This is what makes [[Distillation]] and adapter methods attractive later.
- Fine-tuning is cheap enough (≤1 hour, one TPU) that exhaustive grid search over the 2×3×3 hyperparameter grid is the recommended procedure.

**Surprising results worth carrying.**

- The TriviaQA data augmentation on SQuAD was worth only 0.1–0.4 F1. Almost all of the SQuAD gain came from pre-training, not extra supervised QA data.
- Model size helped a 3,600-example task. That was not the prevailing belief in 2018.
- MRPC is the ablation canary: it is small and unlike the pre-training objective, so it swings hardest (86.5 → 77.5 when you remove bidirectionality). Watch small-data tasks when you want to see whether a representation change actually matters.

**Follow-up questions.** Is NSP doing anything, or is it just training on longer contiguous spans? (RoBERTa says the latter.) What happens if you scale the masking rate? Can you get bidirectionality *and* dense supervision at the same time? Why does the last layer underperform the second-to-last as a frozen feature — how much of the top layer is spent on vocabulary bookkeeping rather than semantics?

## Links

Related: [[Attention Is All You Need]] · [[Causal Attention]] · [[Query, Key, and Value (QKV)]] · [[Cross Entropy]] · [[Foundation Models]] · [[Seq2Seq models]] · [[Auto-regressive models]] · [[Backpropagation]] · [[Linear Projection]] · [[Distillation]] · [[Flash Attention]] · [[Regularization]] · [[In Context Learning]]

New topics worth writing: Masked Language Modelling, Transfer Learning and Fine-tuning, WordPiece and Subword Tokenization, GLUE benchmark, SQuAD and extractive question answering, ELMo and contextual word embeddings, GELU activation, Learned vs sinusoidal positional embeddings, Adam with warmup and linear decay, Scaling laws for pre-trained models, Named Entity Recognition, Frozen feature extraction vs full fine-tuning, RoBERTa, ELECTRA
