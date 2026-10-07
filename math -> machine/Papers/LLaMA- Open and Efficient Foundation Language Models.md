---
title: "LLaMA: Open and Efficient Foundation Language Models"
authors: ["Hugo Touvron", "Thibaut Lavril", "Gautier Izacard", "Xavier Martinet", "Marie-Anne Lachaux", "Timothée Lacroix", "Baptiste Rozière", "Naman Goyal", "Eric Hambro", "Faisal Azhar", "Aurelien Rodriguez", "Armand Joulin", "Edouard Grave", "Guillaume Lample"]
year: 2023
arxiv: "2302.13971"
url: https://arxiv.org/abs/2302.13971
priority: Low-Priority
read_on: 2026-09-30
tags: [paper, transformers, llm, scaling]
---
## The Core Idea

Everyone had been racing to build bigger models. Then [[Training Compute-Optimal Large Language Models (Chinchilla)|Chinchilla]] showed that for a fixed *training* budget, you do better with a smaller model fed more data. LLaMA takes that one step further and points out the thing Chinchilla ignored: **you train a model once, but you serve it millions of times.**

So the right question is not "what is the best model I can train for $X of compute?" It is "what is the best model I can *run cheaply* at a given quality?"

The answer: take a small model and train it far past the point where the scaling laws say to stop. Chinchilla's rule would put a 10B model on 200B tokens. LLaMA trains a 7B model on **1.0T tokens** — five times more than "optimal" — and the loss is *still falling* at the end. The 33B and 65B models get 1.4T tokens.

The payoff is stark: **LLaMA-13B beats GPT-3 (175B) on most benchmarks while being 10× smaller**, and runs on one GPU. LLaMA-65B is competitive with Chinchilla-70B and PaLM-540B.

> [!NOTE] Inference-optimal vs compute-optimal
> Compute-optimal minimises training loss for a fixed training FLOP budget. Inference-optimal minimises the *serving* cost of hitting a target quality. Because serving cost scales with parameter count and training cost is paid once, inference-optimal always pushes you towards smaller models trained on more tokens than Chinchilla recommends. ^inference-optimal

The second contribution is political rather than technical: **every token comes from publicly available data.** Chinchilla, PaLM and GPT-3 all lean on corpora described as "Books – 2TB" or "social media conversations" that nobody outside can touch. LLaMA's 1.4T tokens are all reproducible, and the weights were released to researchers. That release is what made the open-weight ecosystem of 2023 happen.

Why did this not exist before? Partly belief — the field assumed parameter count was the lever. Partly cost: training 7B on 1T tokens looks *wasteful* if your metric is training loss per FLOP. It only looks smart when you put inference on the ledger.

## The Methodology

### Data — 1.4T tokens, all public

| Source | Share | Epochs | Disk |
|---|---|---|---|
| English CommonCrawl (CCNet) | 67% | 1.10 | 3.3 TB |
| C4 | 15% | 1.06 | 783 GB |
| GitHub (Apache/BSD/MIT only) | 4.5% | 0.64 | 328 GB |
| Wikipedia (20 languages) | 4.5% | 2.45 | 83 GB |
| Books (Gutenberg + Books3) | 4.5% | 2.23 | 85 GB |
| ArXiv (LaTeX) | 2.5% | 1.06 | 92 GB |
| StackExchange | 2.0% | 1.03 | 78 GB |

Details that mattered:

- **CommonCrawl** went through CCNet: line-level dedup, fastText language ID to drop non-English, an n-gram model to drop low-quality text. Then an extra classifier trained to tell "pages cited as references on Wikipedia" from "random pages" — anything not classified as reference-like was thrown away.
- **C4 was included on purpose even though it is also CommonCrawl.** Exploratory runs showed that *two differently-filtered* crawls beat one. C4's filter is heuristic (punctuation, word counts); CCNet's is model-based. The diversity of filtering noise helps.
- **ArXiv** was stripped of everything before the first section and the bibliography, comments removed, user macros inline-expanded.
- Most data is seen **once**. Wikipedia and Books get ~2 epochs.
- [[Neural Machine Translation of Rare Words with Subword Units|BPE]] via SentencePiece, with two quirks: **numbers split into individual digits**, and byte-fallback for unknown UTF-8.

### Architecture — a [[Attention Is All You Need|Transformer]] with three swaps

Everything is a decoder-only [[Causal Attention|causal]] Transformer. Three changes from the 2017 original, each borrowed:

1. **Pre-normalisation with RMSNorm** (from GPT-3 / [[On Layer Normalization in the Transformer Architecture|Pre-LN]]). Normalise the *input* of each sub-layer, not the output. RMSNorm drops the mean-subtraction of [[Layer Normalization|LayerNorm]] and just divides by the root-mean-square:
   $$\bar{x}_i = \frac{x_i}{\sqrt{\tfrac{1}{d}\sum_j x_j^2}}\, g_i$$
   Cheaper, and it trains stably.
2. **SwiGLU instead of ReLU** (from PaLM). A [[Gated Activation|gated]] feedforward: $\text{SwiGLU}(x) = (\text{Swish}(xW_1) \odot xW_3)W_2$. Because gating needs a third matrix, the hidden width is set to $\tfrac{2}{3}\cdot 4d$ rather than $4d$, so the parameter count stays roughly the same.
3. **[[RoPE|Rotary position embeddings]]** instead of learned absolute positions (from GPT-Neo). Position becomes a rotation applied to queries and keys at *every* layer.

| params | $d$ | heads | layers | LR | batch (tokens) | tokens |
|---|---|---|---|---|---|---|
| 6.7B | 4096 | 32 | 32 | $3.0\times10^{-4}$ | 4M | 1.0T |
| 13.0B | 5120 | 40 | 40 | $3.0\times10^{-4}$ | 4M | 1.0T |
| 32.5B | 6656 | 52 | 60 | $1.5\times10^{-4}$ | 4M | 1.4T |
| 65.2B | 8192 | 64 | 80 | $1.5\times10^{-4}$ | 4M | 1.4T |

### Optimisation

[[Decoupled Weight Decay Regularization (AdamW)|AdamW]] with $\beta_1=0.9$, $\beta_2=0.95$ (note: $\beta_2$ lowered from the usual 0.999 — standard practice for large LM runs). Cosine schedule decaying to **10% of peak LR**. Weight decay 0.1. [[On the difficulty of training Recurrent Neural Networks|Gradient clipping]] at 1.0. **2,000 warmup steps.** Batch size is a constant 4M tokens for every model size.

### Systems

- Memory-efficient causal [[Multi-Head Attention|multi-head attention]] from `xformers`: never materialise the attention matrix, and skip computing the masked-out upper triangle entirely. Forward from Rabe & Staats, backward from [[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention]].
- Hand-written backward for the transformer block instead of [[Pytorch Autograd|autograd]], so they can choose which activations to keep (the expensive ones — linear-layer outputs) and which to recompute.
- [[Megatron-LM- Training Multi-Billion Parameter Models Using Model Parallelism|Model]] and sequence parallelism, with all-reduce [[Collective Communication|communication]] overlapped against activation compute.
- Throughput: **380 tokens/sec/GPU on 2048 A100-80GB** for the 65B. That is ~21 days for 1.4T tokens.

## Ablation Studies and Experiments

This is not an ablation paper. There is no table isolating RMSNorm from SwiGLU from RoPE — the architecture choices are inherited on faith from PaLM, GPT-3 and GPT-Neo. What the paper *does* contain is a very wide benchmark sweep plus training-curve evidence for the central claim.

### Common sense reasoning, zero-shot

| Model | BoolQ | PIQA | SIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA |
|---|---|---|---|---|---|---|---|---|
| GPT-3 175B | 60.5 | 81.0 | – | 78.9 | 70.2 | 68.8 | 51.4 | 57.6 |
| Chinchilla 70B | 83.7 | 81.8 | 51.3 | 80.8 | 74.9 | – | – | – |
| PaLM 540B | 88.0 | 82.3 | – | 83.4 | 81.1 | 76.6 | 53.0 | 53.4 |
| LLaMA 7B | 76.5 | 79.8 | 48.9 | 76.1 | 70.1 | 72.8 | 47.6 | 57.2 |
| LLaMA 13B | 78.1 | 80.1 | 50.4 | 79.2 | 73.0 | 74.8 | 52.7 | 56.4 |
| LLaMA 65B | 85.3 | 82.8 | 52.3 | 84.2 | 77.0 | 78.9 | 56.0 | 60.2 |

65B beats Chinchilla-70B everywhere except BoolQ, and beats PaLM-540B everywhere except BoolQ and WinoGrande. 13B beats GPT-3 on most of these at 10× fewer parameters.

### Closed-book QA

TriviaQA, exact match: 65B goes 68.2 (0-shot) → 73.0 (64-shot), against Chinchilla-70B's 55.4 → 64.6. **13B scores 56.6 zero-shot, beating Chinchilla-70B's 55.4** — and 13B fits on a single V100 at inference.

NaturalQuestions 64-shot: LLaMA-65B 39.9 vs PaLM-540B 39.6 vs Chinchilla 35.5 vs GPT-3 29.9.

### Maths — the surprise

| | MATH | +maj1@k | GSM8k | +maj1@k |
|---|---|---|---|---|
| PaLM 540B | 8.8 | – | 56.5 | – |
| Minerva 62B | 27.6 | 43.4 | 52.4 | 68.5 |
| LLaMA 65B | 10.6 | 20.5 | 50.9 | 69.7 |

Minerva is PaLM fine-tuned on 38.5B tokens of ArXiv and maths web pages. **LLaMA-65B beats Minerva-62B on GSM8k (50.9 vs 52.4 at pass@1 — close; 69.7 vs 68.5 with majority voting) with no maths fine-tuning at all.** On MATH it loses badly (10.6 vs 27.6), so the effect is limited to easier word problems. Majority voting ([[Chain of Thought|self-consistency]], $k=256$ for MATH, $k=100$ for GSM8k) roughly doubles MATH and adds ~19 points to GSM8k.

### Code

HumanEval pass@1, temperature 0.1: 65B gets 23.7 vs PaLM-62B's 15.9 and LaMDA-137B's 14.0. MBPP pass@1: 37.7 vs PaLM-540B's 36.8. **LLaMA-13B (15.8 HumanEval) beats LaMDA-137B (14.0).** No code fine-tuning; PaLM-Coder shows that would add ~10 points.

### MMLU — the clear loss

5-shot average: LLaMA-65B **63.4**, Chinchilla-70B **67.5**, PaLM-540B **69.3**. This is the one place LLaMA is plainly behind. The authors' explanation is data: their books-and-papers slice (ArXiv + Gutenberg + Books3) is only **177 GB**, while Gopher/Chinchilla/PaLM used up to **2 TB of books**. Supporting evidence: Gopher beats GPT-3 on MMLU specifically while being comparable elsewhere, and Gopher had the books. The gap is widest in Humanities (61.8 vs 63.6) and Social Science (72.9 vs 79.3).

> [!NOTE] Book data buys exam performance
> MMLU is essentially a multiple-choice exam over academic knowledge. It appears to be the benchmark most sensitive to long-form, edited, book-length text in the pretraining mix — not to parameter count. ^books-buy-mmlu

### Training-curve evidence

Figure 2 tracks benchmark scores over the run. Most metrics improve steadily and track training [[Perplexity|perplexity]]. Two exceptions worth noting:

- **SIQA is noisy.** Performance jumps around so much the authors say it "may indicate that this benchmark is not reliable."
- **WinoGrande decouples from perplexity.** 33B and 65B sit at similar scores through training despite different losses.

Both are useful warnings for anyone using these as progress signals.

### Instruction tuning — one experiment, not a study

A single run following [[Exploring the Limits of Transfer Learning (T5)|Flan]]-style protocol (Chung et al. 2022). MMLU 63.4 → **68.9**, beating Flan-PaLM-62B (59.6) and OPT-IML-Max-30B (43.2). Still far from `code-davinci-002` at 77.4. The paper explicitly declines to investigate further — see [[Instruction Tuning]] and [[Training language models to follow instructions with human feedback|InstructGPT]] for what this line becomes.

### What did not work / was not tried

- **No architecture ablation.** You cannot tell from this paper how much RMSNorm, SwiGLU or RoPE each contributed.
- **MMLU was not fixed.** They diagnosed the book-data shortfall and shipped anyway.
- **No code fine-tuning**, explicitly out of scope.
- **No [[RLHF]]**, no safety tuning. The released models are raw base models.

## Worth Remembering

**Toxicity gets worse with scale, and gets worse when you ask for politeness.** RealToxicityPrompts scores (higher = more toxic): 7B 0.106, 13B 0.104, 33B 0.107, **65B 0.128**. And with a "respectful" prefix prepended, 65B goes to **0.141** — *higher* than without it. The prompt asking for politeness made the largest model more toxic. The authors note that the trend with size may only hold within a model family (Gopher is bigger than Chinchilla but not more toxic).

**Gender bias is measurable and mechanical.** On WinoGender coreference, LLaMA-65B scores 81.7 on "their/them/someone" but 78.8 on "her/her/she" and only 72.1 on "his/him/he". On the "gotcha" subset — where the pronoun's gender does *not* match the occupation's stereotypical gender — 65B drops to 75.0 and 63.3. The model is using occupational gender priors instead of reading the sentence.

**CrowS-Pairs:** LLaMA averages 66.6 vs GPT-3's 67.2 and OPT-175B's 69.5 — marginally better overall, but **worse on religion (79.0 vs OPT's 68.6)**, age and gender.

**TruthfulQA:** 65B scores 0.57 truthful / 0.53 truthful-and-informative, against GPT-3-175B's 0.28 / 0.25. Better, but still [[Hallucination|hallucinating]] on nearly half of questions.

**Carbon, stated plainly.** Training all four models: ~2,638 MWh, ~1,015 tCO₂eq (at the US average 0.385 kg CO₂e/kWh, PUE 1.1). The 65B alone: 1,022,362 GPU-hours, 449 MWh, 173 tCO₂eq — comparable to OPT-175B's 137 t and BLOOM's 183 t. The argument for release is partly environmental: the training is already paid for.

**Practical caveats if you wanted to use this:**

- These are base models. They complete text; they do not answer questions. You need [[Instruction Tuning|instruction tuning]] or few-shot prompting.
- Context length is short (2048 tokens). No [[Long Context|long-context]] tricks here, though RoPE made later extension possible.
- Evaluation protocol matters and differs between papers. LLaMA normalises multiple-choice likelihood by completion character count (following Gao et al.), except OpenBookQA and BoolQ where it uses $P(\text{completion}\mid\text{context})/P(\text{completion}\mid\text{``Answer:"})$. Cross-paper numbers are not strictly comparable.
- Their NaturalQuestions and TriviaQA splits differ from GPT-3's and PaLM's (the unfiltered TriviaQA eval server was gone).
- 20 languages of Wikipedia are in the mix, but 67% of the data is English CommonCrawl. This is an English model.

**The strategic lesson, restated.** The choice of how many tokens to train on is not a training-efficiency question. It is a question about where you expect your costs to be over the model's life. If you serve a lot, overtrain a small model. This reasoning is now standard — it is why every subsequent open model family trains far past Chinchilla-optimal.

**Open questions this raises:**
- How far past Chinchilla can you push before the loss curve genuinely flattens? The paper shows 7B still improving at 1T but does not find the wall.
- Can you close the MMLU gap purely with more high-quality book-like data, or is there a parameter-count floor for that benchmark?
- The "two differently-filtered CommonCrawl dumps beat one" finding is reported in a single sentence with no numbers. That is an interesting claim about filter diversity that deserved an experiment.

## Links

Related: [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Scaling Laws for Neural Language Models]] · [[Attention Is All You Need]] · [[RoFormer- Enhanced Transformer with Rotary Position Embedding]] · [[RoPE]] · [[On Layer Normalization in the Transformer Architecture]] · [[Layer Normalization]] · [[Gated Activation]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Megatron-LM- Training Multi-Billion Parameter Models Using Model Parallelism]] · [[Neural Machine Translation of Rare Words with Subword Units]] · [[Tokenization]] · [[Chain of Thought]] · [[Instruction Tuning]] · [[Exploring the Limits of Transfer Learning (T5)]] · [[Training language models to follow instructions with human feedback]] · [[Hallucination]] · [[Perplexity]] · [[Collective Communication]] · [[Distributed Training]] · [[Foundation Models]] · [[The Bitter Lesson (essay)]]

New topics worth writing: RMSNorm, CCNet data pipeline, inference-optimal scaling, MMLU, RealToxicityPrompts, CrowS-Pairs, WinoGender, TruthfulQA, HumanEval, GSM8k, Minerva, Books3 / The Pile, SentencePiece, model carbon accounting, open-weight model licensing
