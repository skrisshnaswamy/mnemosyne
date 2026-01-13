---
title: Language Models are Few-Shot Learners (GPT-3)
authors:
  - Tom B. Brown
  - Benjamin Mann
  - Nick Ryder
  - Melanie Subbiah
  - Jared Kaplan
  - Prafulla Dhariwal
  - Arvind Neelakantan
  - Pranav Shyam
  - Girish Sastry
  - Amanda Askell
  - Sandhini Agarwal
  - Ariel Herbert-Voss
  - Gretchen Krueger
  - Tom Henighan
  - Rewon Child
  - Aditya Ramesh
  - Daniel M. Ziegler
  - Jeffrey Wu
  - Clemens Winter
  - Christopher Hesse
year: 2020
arxiv: "2005.14165"
url: https://arxiv.org/abs/2005.14165
priority: Must-Read
read_on: 2026-08-21
tags:
  - paper
  - transformers
  - llm
  - optimization
  - GPT3
---
## The Core Idea

Before this paper, the recipe was: pre-train a big language model, then **fine-tune** it — run gradient descent on a few thousand labelled examples for each new task. See [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] and [[BERT- Pre-training of Deep Bidirectional Transformers]]. That works, but you need a labelled dataset per task, and the fine-tuned model tends to become brittle outside its narrow training distribution.

The claim here is: if you make the model big enough, you do not need fine-tuning at all. You just **write the task into the prompt**. Give the model a few examples of input→output as plain text, then a fresh input, and let it predict the continuation. No weight updates. Not one gradient step at test time.

> [!NOTE] In-context learning ^in-context-learning
> The model "learns" a task inside a single forward pass, purely by conditioning on examples in its context window. The weights never change. Pre-training is the outer loop; reading the prompt is the inner loop.

Why this did not exist before: it is a **scale** phenomenon. GPT-2 (1.5B) could do this a little; the results were bad. GPT-3 is 175B parameters — over 10× larger than any previous dense model. The key evidence is not that GPT-3 is good, it is that the *gap between zero-shot and few-shot grows with model size*. Small models barely benefit from having examples in their context. Big models benefit a lot. So "the ability to learn from a prompt" is itself something that emerges from scale.

Three settings they define, all inference-only:

- **Zero-shot**: only a natural-language instruction. `Translate English to French: cheese =>`
- **One-shot**: instruction + one demonstration.
- **Few-shot**: instruction + $K$ demonstrations, typically $K \in [10, 100]$, as many as fit in 2048 tokens.

What this unlocks: one model, many tasks, specified by text. This is the paper that made "prompting" a thing, and it is the direct ancestor of every instruction-following chat model. See [[In Context Learning]], [[Foundation Models]].

## The Methodology

**Architecture.** Essentially GPT-2, scaled. A decoder-only [[Attention Is All You Need|Transformer]] with [[Causal Attention|causal masking]], pre-normalisation (LayerNorm before the sublayer, not after), modified initialisation, and byte-level BPE tokenisation. One change: layers **alternate between dense attention and locally banded sparse attention** (from the Sparse Transformer), to cut the cost of long contexts.

Feed-forward width is always $d_{ff} = 4 \cdot d_{model}$. Context window is $n_{ctx} = 2048$ tokens for every model size.

Eight models were trained, spanning three orders of magnitude:

| Name | Params | Layers | $d_{model}$ | Heads | $d_{head}$ | Batch (tokens) | LR |
|---|---|---|---|---|---|---|---|
| Small | 125M | 12 | 768 | 12 | 64 | 0.5M | $6.0\times10^{-4}$ |
| Medium | 350M | 24 | 1024 | 16 | 64 | 0.5M | $3.0\times10^{-4}$ |
| Large | 760M | 24 | 1536 | 16 | 96 | 0.5M | $2.5\times10^{-4}$ |
| XL | 1.3B | 24 | 2048 | 24 | 128 | 1M | $2.0\times10^{-4}$ |
| 2.7B | 2.7B | 32 | 2560 | 32 | 80 | 1M | $1.6\times10^{-4}$ |
| 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 2M | $1.2\times10^{-4}$ |
| 13B | 13B | 40 | 5140 | 40 | 128 | 2M | $1.0\times10^{-4}$ |
| **GPT-3** | **175B** | **96** | **12288** | **96** | **128** | **3.2M** | $0.6\times10^{-4}$ |

Note the pattern: bigger model → bigger batch, smaller learning rate. They chose batch size by measuring the **gradient noise scale** during training.

**Objective.** Nothing new. Plain autoregressive next-token prediction — maximise

$$\mathcal{L} = \sum_i \log P(x_i \mid x_{i-k}, \dots, x_{i-1}; \Theta)$$

which is just [[Cross Entropy|cross-entropy]] on the next token. See [[Auto-regressive models]].

**Data.** 300 billion tokens total, from five sources, sampled *not* in proportion to their size:

| Source | Tokens | Weight in mix | Epochs over 300B |
|---|---|---|---|
| Common Crawl (filtered) | 410B | 60% | 0.44 |
| WebText2 | 19B | 22% | 2.9 |
| Books1 | 12B | 8% | 1.9 |
| Books2 | 55B | 8% | 0.43 |
| Wikipedia | 3B | 3% | 3.4 |

So Wikipedia is seen 3.4 times and most of Common Crawl is never seen even once. They deliberately accepted a bit of overfitting on clean data for higher average quality.

The Common Crawl filter is worth knowing. They trained a **logistic regression classifier** (Spark tokenizer + HashingTF features) to distinguish WebText/Wikipedia/books (positive) from raw Common Crawl (negative). Then they kept a document if

$$\texttt{np.random.pareto}(\alpha) > 1 - \texttt{document\_score}, \qquad \alpha = 9$$

This mostly keeps high-scoring documents but lets some low-scoring ones through, preserving diversity. Then fuzzy dedup via MinHashLSH with 10 hashes, which shrank the corpus ~10%. 45TB compressed raw → 570GB after filtering.

**Training details.** Adam with $\beta_1 = 0.9$, $\beta_2 = 0.95$, $\epsilon = 10^{-8}$. Global gradient-norm clip at 1.0. Cosine LR decay to 10% of peak over 260B tokens, then flat. Linear warmup over the first 375M tokens. Batch size ramped linearly from 32k tokens up to full over the first 4–12B tokens. Weight decay 0.1 as light [[Regularization|regularisation]]. Sequences are always the full 2048 tokens, packing multiple documents together separated by an end-of-text token — no special masking between documents. Trained on V100s with model parallelism split along both depth and width, plus fully half-precision memory optimisations ([[Mixed Precision training]]).

**Evaluation mechanics.** For multiple choice, they score each candidate completion by the model's likelihood, usually per-token normalised for length. For three datasets (ARC, OpenBookQA, RACE) they instead normalise by the unconditional probability:

$$\frac{P(\text{completion} \mid \text{context})}{P(\text{completion} \mid \texttt{"Answer: "})}$$

Binary classification gets semantic labels ("True"/"False", not 0/1). Free-form generation uses beam search, width 4, length penalty $\alpha = 0.6$.

## Ablation Studies and Experiments

**Scaling holds.** Validation loss follows a power law in compute for two more orders of magnitude past the original scaling-laws paper, with barely any bend in the curve.

**Where GPT-3 wins outright (few-shot beats fine-tuned SOTA):**

| Task | Prior SOTA | GPT-3 0-shot | 1-shot | few-shot |
|---|---|---|---|---|
| LAMBADA (acc) | 68.0 | 76.2 | 72.5 | **86.4** |
| LAMBADA (ppl) | 8.63 | 3.00 | 3.35 | **1.92** |
| PTB (ppl, 0-shot only) | 35.8 | **20.5** | — | — |
| TriviaQA | 68.0 (RAG, retrieval) | 64.3 | 68.0 | **71.2** |
| PIQA | 79.4 | 80.5 | 80.5 | **82.8** |

The LAMBADA trick is instructive. A vanilla LM does not know the answer must be exactly one word, so it spreads probability over valid multi-word continuations. Framing it as fill-in-the-blank in the few-shot prompt (`Alice went to visit her friend ____. → Bob`) fixes this. This *hurts* the smallest model by ~20% but *helps* GPT-3 by 10%. And critically, the fill-in-blank format is worse than zero-shot in the **one-shot** setting — one example is not enough to establish the pattern.

**Where it is respectable but behind:**

- CoQA: 85.0 F1 few-shot vs 90.7 fine-tuned SOTA, only a few points below human.
- SuperGLUE: 71.8 few-shot ($K=32$) vs 89.0 SOTA, but it beats fine-tuned BERT-Large (69.0) on 4 of 8 tasks. Notably, **fewer than 8 examples per task** are enough to beat BERT-Large's overall SuperGLUE score — a model fine-tuned on 125K examples.
- Winogrande: 77.7 few-shot vs 84.6 SOTA vs 94.0 human.

**Translation** (BLEU): a clean demonstration that in-context examples do real work. Zero-shot En→Fr is 25.2; one-shot jumps to 28.3 (+7 BLEU average across pairs); few-shot 32.6. Into-English is much stronger than out-of-English (Fr→En 39.2 vs En→Fr 32.6) — it is an English model. En→Ro is a disaster at 21.0, more than 10 BLEU below unsupervised NMT baselines; they blame the GPT-2 byte-level BPE tokenizer, which was built for English.

**What did not work:**

- **WiC** (is this word used the same way in two sentences?): **49.4% — exactly chance.** They tried many phrasings and formulations; none helped.
- **ANLI R1/R2**: every model below 175B sits at ~33%, i.e. random for a 3-way task. Even GPT-3 only reaches 36.8 / 34.0. R3 gets to 40.2, "signs of life".
- **RACE-h**: 46.8 vs 90.0 SOTA. Reading comprehension with multiple choice is weak.
- **DROP**: 36.5 F1 vs 89.1 SOTA. Discrete reasoning over passages.
- **Reversing words**: 0.44% accuracy at 175B few-shot. Complete failure. No model can do it.

The authors spot the pattern themselves: GPT-3 is bad at tasks that require **comparing two pieces of text** (WiC, ANLI, RTE, CB). Their hypothesis is that this is a cost of being purely autoregressive — no bidirectionality, no denoising objective. A [[BERT- Pre-training of Deep Bidirectional Transformers|masked-LM]] style model would likely be stronger here.

**Arithmetic** is the cleanest emergence result. Few-shot, 175B:

| | 2D+ | 2D− | 3D+ | 3D− | 4D+ | 5D+ | 2D× | 1DC |
|---|---|---|---|---|---|---|---|---|
| 0-shot | 76.9 | 58.0 | 34.2 | 48.3 | 4.0 | 0.7 | 19.8 | 9.8 |
| 1-shot | 99.6 | 86.4 | 65.5 | 78.7 | 14.0 | 3.5 | 27.4 | 14.3 |
| few-shot | **100.0** | **98.9** | **80.4** | **94.2** | 25.5 | 9.3 | 29.2 | 21.3 |

The 13B model — the second largest — gets 2-digit addition right only ~55% of the time few-shot and everything else under 10%. This is a sharp jump, not a smooth curve. To check it was not memorisation they searched training data for the test problems in the forms `<NUM1> + <NUM2> =` and `<NUM1> plus <NUM2>`: **17 of 2000 addition problems matched (0.8%)**, 2 of 2000 subtraction (0.1%). And the errors look like real arithmetic errors — forgetting to carry a 1.

**Word scrambling** shows in-context learning most starkly. Random-insertion (`s.u!c/c!e.s s i/o/n` → `succession`): 8.26% zero-shot → 45.4% one-shot → **67.2% few-shot**. The model essentially cannot do it without examples. Since BPE tokens average ~0.7 words, succeeding means pulling apart token substructure, which is genuinely hard.

**SAT analogies**: 65.2% few-shot. The average human college applicant scores 57%. Random is 20%.

**News article detection**: humans judged ~200-word articles. Accuracy at spotting machine text falls monotonically with model size — 76% (125M), 61% (350M), 55% (13B), **52% (175B)**, where 50% is chance. A deliberately-bad control model was caught 86% of the time. Longer ~500-word articles gave the same result: 52%. And people spent *more* time per article as models got bigger, so it is not inattention.

**Contamination analysis.** This is the honest and slightly painful part. They tried to strip 13-gram overlaps between training data and every benchmark test set, but **a bug meant the filtering only partly worked**, and retraining was too expensive. So instead they built a "clean" subset of each benchmark (no 13-gram collision with any training doc) and compared scores.

Findings:
- Mostly negligible. No correlation between contamination level and score change.
- **PIQA**: 29% flagged, 3-point absolute drop on clean subset → marked with asterisk. But a 25× smaller model showed the same drop, suggesting statistical bias in which examples got flagged rather than memorisation.
- **Winograd**: 45% flagged, 2.6% drop, 132 schemas genuinely in training data → asterisk.
- **Wikipedia LM benchmarks + Children's Book Test**: almost entirely contained in training data. **Results simply not reported.** PTB survived only because it predates the modern internet.
- **DROP/QuAC/SQuAD2**: >90% flagged, but inspection showed only the *source passages* were in training, never the question-answer pairs. False positives.
- The clean/all variance grows with contamination but has no bias in either direction, which is the strongest argument that GPT-3 is not just memorising.

Also worth noting from Figure 4.1: the train/validation gap barely grows with model size or training time. GPT-3 175B is not overfitting its corpus.

## Worth Remembering

**Compute.** 3.64e+03 petaflop/s-days, 3.14e+23 FLOPs. Their rule of thumb for dense transformers: 6 FLOPs per parameter per token (2 for forward — one multiply, one add — times 3 to account for the backward pass, since computing $\partial \text{loss}/\partial \text{params}$ and $\partial \text{loss}/\partial \text{acts}$ each cost about a forward pass). See [[Backpropagation]].

**Inference is cheap relative to training.** Generating 100 pages of text from trained GPT-3 costs ~0.4 kW-hr, a few cents. The authors explicitly point at [[Distillation]] as the fix for deployment cost, and note it had never been tried at hundreds of billions of parameters.

**Trained on 300B tokens with 175B parameters — undertrained by later standards.** They cite scaling laws to justify "train much larger models on many fewer tokens than is typical". Chinchilla later showed this ratio was wrong. Worth holding in mind when reading the loss curves.

**Limitations the authors admit:**
- Long samples lose coherence, repeat themselves semantically, contradict themselves.
- Bad at "common sense physics" — *"If I put cheese into the fridge, will it melt?"* — despite doing well on PIQA.
- No bidirectionality, no denoising objective. They conjecture a bidirectional model at this scale would fine-tune better.
- The pre-training objective **weights every token equally**. It has no notion of what matters. They suggest learning the objective from humans or fine-tuning with RL — which is exactly what became RLHF two years later.
- Terrible **pre-training** sample efficiency. Test-time sample efficiency is near-human; training-time is nowhere close. GPT-3 sees far more text than a human sees in a lifetime.
- Not grounded — no video, no physical interaction.

**The deepest open question, stated plainly by the authors:** is few-shot learning actually *learning* a new task at inference time, or just *recognising* a task already absorbed during pre-training? They think it varies by task — word-unscrambling looks learned de novo, translation must have been learned in pre-training. They admit it is not even clear what humans learn from scratch. This is still not resolved.

**Bias findings** (preliminary, their word):
- 83% of 388 occupations were more likely to be followed by a male identifier. Average bias score $\frac{1}{n}\sum_{\text{jobs}}\log\frac{P(\text{female}|C)}{P(\text{male}|C)}$ was $-1.11$ neutral, $-2.14$ for "The **competent** {occupation} was a", $-1.15$ for "incompetent". So competence skews male harder.
- Women described with appearance words ("beautiful" co-occurred 158 times, "gorgeous" 28); men with a wider adjective range.
- Sentiment by race: 'Asian' ranked 1st in 3 of 7 models; 'Black' ranked lowest in 5 of 7. The gaps narrowed slightly at larger sizes.
- 'terrorism', 'violent', 'terrorist' were in the top 40 words for Islam.
- One counterpoint: on Winogender pronoun resolution, 175B was the *only* model where female-occupation accuracy exceeded male (81.7% vs 76.7%) — weak evidence that bigger models may be more robust here.

**Practical caveats if you wanted to use this:**
- $K$ (number of demonstrations) is a hyperparameter. Bigger is usually but not always better; they tuned it on dev sets.
- Prompt phrasing matters enormously and is not principled. WiC failed across many formulations they tried.
- Length normalisation of candidate completions matters for multiple choice, and the right normalisation is dataset-dependent.
- Never fine-tuned — the paper explicitly leaves fine-tuned GPT-3 to future work.

## Links

Related: [[In Context Learning]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Attention Is All You Need]] · [[Auto-regressive models]] · [[Causal Attention]] · [[Cross Entropy]] · [[Foundation Models]] · [[Distillation]] · [[Mixed Precision training]] · [[Regularization]] · [[Seq2Seq models]] · [[Deep Learning]]

New topics worth writing: Scaling laws for neural language models, Gradient noise scale and critical batch size, Byte-pair encoding, Sparse Transformer / locally banded attention, Data contamination and n-gram decontamination, Chinchilla compute-optimal scaling, RLHF and learning objectives from human preferences, Nucleus sampling, MinHash LSH deduplication, Prompt engineering, SuperGLUE benchmark
