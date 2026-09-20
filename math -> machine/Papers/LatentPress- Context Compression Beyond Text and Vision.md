---
title: "LatentPress: Context Compression Beyond Text and Vision"
authors: ["Zhou et al."]
year: 2026
arxiv: "2609.01507"
url: https://arxiv.org/abs/2609.01507
priority: Good-To-Read
read_on: 2026-09-06
tags: [paper, transformers, llm, vision]
---
## The Core Idea

When a language model needs to remember a long chat history or a long document, we normally squash it into **text** — a summary, a retrieved snippet, a pruned prompt. DeepSeek-OCR squashes it into a **picture** of text, which then has to be read back out with OCR. Both routes assume the compressed thing must be human-readable before a model can use it.

LatentPress asks: why? The consumer is a machine. Give it machine food.

The compressed context is a short sequence of **continuous vectors** that live in the reader model's input-embedding space. They are pasted straight in front of the question, through the same `inputs_embeds` port the model already uses for word embeddings. No text is ever recovered at inference time. The decoder is never touched — it stays frozen — and the only trained thing is a small adapter, 4.2M–26.2M parameters, about **0.1% of the decoder**.

> [!NOTE] Soft token
> A vector fed into a transformer at the embedding layer that does not correspond to any word in the vocabulary. It is "soft" because it is a free-floating point in embedding space, not one of the discrete points that the tokenizer can produce. ^soft-token

Soft-token compression is not new (Gist, AutoCompressor, ICAE, xRAG). What is new is the *combination*: reader fully frozen, tiny adapter, **no reconstruction step at inference**, and a compression rate that can **vary per segment** based on the structure of the input. ICAE, for instance, decodes memory vectors back to text before answering, and trains an LLM-scale encoder with [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]]. xRAG freezes the reader but compresses a single retrieved passage into a single token — it cannot handle a whole multi-turn history.

The second, sneakier idea: **not all turns deserve the same compression**. In a chat log, the answer-bearing facts ("I've been collecting vintage cameras for three months") come from the *user*. The assistant's replies are long and padded. So keep user turns at full resolution and squash assistant turns 8–32×. The ablation shows this single choice is doing enormous work.

What it unlocks: writing takes **43 ms** per conversation (one forward pass, no autoregression), and reading is **5–9× faster** than reading the raw context, because the prefix is 4–8× shorter and [[Attention Is All You Need|attention]] cost is quadratic in it.

## The Methodology

**The setup.** A context $x = (x_1, \ldots, x_T)$ is a sequence of segments (dialogue turns, or document chunks). A frozen decoder $f_\theta$ must answer a question $q$:

$$m = \textsc{Write}_\phi(x; \pi), \qquad y = f_\theta\big([m; \mathrm{emb}(q)]\big)$$

$\phi$ is the writer (trained), $\pi$ is the per-segment compression schedule (hand-specified), $\theta$ is the decoder (frozen forever).

**The writer.** This is the whole architecture and it is embarrassingly small:

1. Take the **bottom 2 transformer layers of the reader itself**, deep-copy them, and freeze them. These are the "encoder". Because they come from the reader, their output already lives in a space the reader understands — that is what "reader-matched" means.
2. On top, a single trainable **linear adapter $A \in \mathbb{R}^{d \times d}$, initialised to the identity matrix**. Identity init means the writer starts out producing something very close to the raw token embeddings, and only drifts as training pushes it.
3. Mean-pool groups of $k$ neighbouring positions into one soft token.

That is it. 12.85M params for Qwen2.5-7B, 16.78M for Qwen3-8B, 4.20M for Qwen3-1.7B, 26.22M for Qwen2.5-14B. Since the soft tokens are tied to a specific reader's embedding space, you train **one writer per reader**.

Conceptually the writer fuses the literal embedding $E_i$ with a contextual abstraction $c_i$: $h_i = H(E_i, c_i)$. The paper flags that $H$ *could* be a learned gate (à la [[Gated Activation|highway/GRU-style gating]]) but they use a lightweight fixed version and leave learned fusion to future work.

**The compression schedule $\pi = (k_1, \ldots, k_T)$.**

- *Uniform*: $k_i = k$ for all segments, $k \in \{4, 8, 16\}$. Used for documents.
- *Role-based*: $k_{\text{user}} = 1$ (user turns bypass the writer entirely, keeping raw token embeddings), $k_{\text{assistant}} \in \{8, 16, 32\}$.

The overall ratio is an *emergent* number, not a target: $\rho = |\mathcal{C}| / \sum_i \lceil n_i / k_{r_i} \rceil$. Role-based with $k_a = 32$ happens to give $7.70\times$ on LongMemEval because of the user/assistant token mix.

**The loss.** Two terms, both computed teacher-forced on a target sequence $y$ of length $N$, with padding masked out:

$$\mathcal{L}(\phi) = \mathcal{L}_{\mathrm{rec}} + \lambda \mathcal{L}_{\mathrm{fkl}}, \qquad \lambda = 1.0$$

$$\mathcal{L}_{\mathrm{rec}} = -\frac{1}{N}\sum_{t=1}^N \log p_{\mathrm{comp},t}(y_t), \qquad \mathcal{L}_{\mathrm{fkl}} = \frac{1}{N}\sum_{t=1}^N \mathrm{KL}\big(p_{\mathrm{full},t} \,\|\, p_{\mathrm{comp},t}\big)$$

The first is plain [[Cross Entropy|cross-entropy]] — the compressed prefix must let the decoder reproduce the target tokens. The second is a [[Distilling the Knowledge in a Neural Network|distillation]] term: run the *same frozen decoder* twice, once on the full raw context and once on the soft prefix, and pull the second distribution towards the first with [[KL Divergence|forward KL]]. The teacher and student are literally the same weights; only the context differs. This is self-distillation of *reading behaviour*.

Note the naming trap: "reconstruction-free" describes the **inference interface**, not the training signal. Training does use a reconstruction loss; inference never decodes back to text.

**Training.** AdamW ([[Decoupled Weight Decay Regularization (AdamW)]]), lr $1\times10^{-4}$, 1000 steps, 400 chunks, batch size 1, chunk length 2048 tokens, bf16 decoder / fp32 writer head. Corpus for the memory experiments: **2,000 UltraChat conversations, text only, no QA labels**. Evaluation is therefore zero-shot on LongMemEval.

**Evaluation.** LongMemEval, 500 questions, *oracle* setting — each question is paired with only its ground-truth evidence session, so retrieval is idealised away and only compression + reading are measured. Judge is Llama-3.1-70B-Instruct with the official per-question-type prompts. Greedy decoding (temperature 0), max 64 new tokens for the soft-token reader.

## Ablation Studies and Experiments

**LongMemEval, Qwen2.5-7B reader, zero-shot from UltraChat:**

| Method | Compression | Overall | user-fact |
|---|---|---|---|
| uncompressed evidence | $1\times$ | 0.490 | 0.946 |
| LatentPress $k_a{=}8$ | $4.62\times$ | 0.476 ± 0.014 | 0.938 |
| LatentPress $k_a{=}16$ | $6.27\times$ | 0.478 ± 0.020 | 0.891 |
| **LatentPress $k_a{=}32$** | $7.70\times$ | **0.504 ± 0.024** | 0.938 |
| ICAE | $4.12\times$ | 0.452 | 0.548 |
| ICAE | $17.28\times$ | 0.174 | 0.209 |
| DeepSeek-OCR | $2.33\times$ | 0.426 | 0.797 |
| DeepSeek-OCR | $9.34\times$ | 0.312 | 0.594 |
| text summary | $12.06\times$ | 0.184 | 0.297 |

Read that top row again: **more tokens are not automatically better**. The raw evidence reader only gets 0.490, and the $7.70\times$ compressed reader beats it. The task still needs multi-session aggregation, temporal reasoning and abstention, so raw text is not an upper bound.

Text summarisation is catastrophic (0.184). The per-category breakdown explains why: abstention accuracy is 0.967 (the reader correctly says "I don't know" when the summary dropped the fact), but *temporal* collapses to 0.016 and *multi-session* to 0.017. Abstractive summarising throws away exactly the precise personal facts the benchmark asks about.

**Across backbones** (all zero-shot, 500 questions):

| Reader | LP $k_a{=}8$ | $k_a{=}16$ | $k_a{=}32$ | OCR $2.33\times$ | OCR $9.34\times$ | summary |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | 0.476 | 0.478 | **0.504** | 0.426 | 0.312 | 0.184 |
| Qwen3-8B | 0.506 | 0.514 | 0.494 | **0.542** | 0.408 | 0.348 |
| Qwen3-1.7B | **0.434** | 0.424 | 0.416 | 0.264 | 0.156 | 0.106 |

Honest reporting: on **Qwen3-8B, OCR at low compression wins** (0.542 vs 0.514). LatentPress only overtakes it as compression rises. The flat-vs-decaying shape is the real story — LatentPress's accuracy barely moves from $4.6\times$ to $7.7\times$, while OCR falls monotonically.

**The ablation that carries the paper** (Qwen2.5-7B, $k_a = 8$):

| Setting | User turns | Assistant turns | Compression | Overall |
|---|---|---|---|---|
| Full LatentPress | raw embeddings | learned soft tokens | $4.62\times$ | **0.476** |
| No-learning | raw embeddings | *pooled* embeddings | $4.62\times$ | 0.325 |
| User-only | raw embeddings | *deleted* | $9.6\times$ | 0.217 |
| Role-swapped | learned soft tokens | raw embeddings | $1.10\times$ | 0.087 |

Three clean lessons. (1) The learned writer contributes ~0.15 absolute over naive mean-pooling at identical compression — it is not just the role schedule. (2) Deleting assistant turns costs another 0.11, so the squashed assistant context still carries complementary information. (3) **Reversing the roles is a disaster: 0.087, at only $1.10\times$ compression.** You can burn almost no compression budget and still destroy the task if you compress the wrong thing. That sensitivity is exactly why the authors flag learned allocation as the obvious next step.

**Encoder-training ablation** — the one thing that did *not* work:

| Borrowed encoder | $k_a{=}8$ | $k_a{=}16$ | $k_a{=}32$ |
|---|---|---|---|
| fine-tuned | 0.454 | 0.460 | 0.438 |
| **frozen** | **0.476** | **0.478** | **0.504** |

Fine-tuning the two borrowed layers loses at *every* rate and, worse, **degrades as compression grows** (0.460 → 0.438) instead of improving. The diagnosis: fine-tuning on UltraChat overfits the training distribution; frozen layers keep the reader's general representation. So even the "encoder" is frozen — literally only a $d \times d$ matrix is learned.

**Judge-free sanity check.** Token-level F1 against gold spans gives the same ordering (LatentPress $k_a{=}32$: 0.251; OCR $9.34\times$: 0.160; uniform soft-token $k{=}8$: 0.052). Absolute F1 is low for everyone because readers answer in sentences and gold answers are short spans. Two independent metrics agreeing is a decent guard against judge artefacts — worth stealing as a habit.

**LongBench-QA (long documents, uniform pooling).** Raw baselines: Qwen2.5-14B 47.93, Qwen2.5-7B 43.80, Qwen3-8B 30.80 (non-thinking mode).

*Cross-domain* (writer trained on LongMemEval-derived QA, evaluated on documents) is only half a success: at $4\times$ it slightly beats raw everywhere (45.13 / 49.88 / 32.79) but collapses after — Qwen2.5-14B falls to 37.08 at $8\times$ and 30.34 at $16\times$.

*In-domain* (writer trained on LongBench training splits) is much stronger:

| Reader | raw | $f4$ | $f8$ | $f16$ |
|---|---|---|---|---|
| Qwen2.5-7B | 43.80 | **49.06** | 43.77 | 37.78 |
| Qwen3-8B | 30.80 | **39.62** | 36.93 | 26.12 |
| Qwen2.5-14B | 47.93 | **57.99** | 52.18 | 40.30 |

$4\times$ beats raw on all three, sometimes by 10 points. **$16\times$ loses to raw on all three.** The pattern is consistent: mild compression acts like a denoiser/focuser; aggressive compression destroys verbatim detail that extractive QA needs.

**Failure modes at high compression** (Appendix D.3) — six named patterns, and only one is about information: (i) unanswerable collapse — abstaining on a question it answered from full context; (ii) JSON envelope artefacts (`unanswerable {"answer": "unanswerable"}`); (iii) blank output; (iv) repetition loops (`Answer: Answer: Answer: …`), most common at $16\times$; (v) Qwen3 `</think>` tags leaking into the answer; (vi) well-formed but factually wrong. The first five are **decoding/format pathologies**, not semantic loss. So part of Qwen3-8B's poor high-compression numbers is the soft prefix knocking the model out of its instruction-following mode, not the memory being wrong.

**Efficiency.** Write cost per conversation, Qwen3-8B, bf16, one H100, batch of 8:

- LatentPress: **43 ms** (one forward pass)
- ICAE: 350–700 ms ($8$–$15\times$ slower — it encodes with the *full* LLM, not two layers)
- Text summarisation: 407–645 ms ($9$–$15\times$)
- DeepSeek-OCR: 844–1056 ms ($\approx 22\times$ — render pages, then autoregressive optical decoding)

Read cost, warm-loaded, 30 LongBench examples, seconds/example at $f8$:

| Reader | Raw | LatentPress | Cached OCR |
|---|---|---|---|
| Qwen2.5-7B | 2.44 | 0.49 | 2.71 |
| Qwen2.5-14B | 4.14 | 0.49 | 4.34 |
| Qwen3-8B | 3.97 | 0.43 | 4.03 |

$5.0$–$9.2\times$ faster than raw. Note cached OCR is barely faster than raw — because $2.6\times$–$9.9\times$ compression at the vision-token level still leaves the reader a long text prefix after reconstruction.

## Worth Remembering

**The oracle setting is a big asterisk.** LongMemEval is evaluated with only the ground-truth evidence session supplied. Retrieval is assumed perfect. The authors are explicit that the full LongMemEval-S/M haystacks exceed the history lengths the writer was trained on, and that pairing LatentPress with a retriever is future work. So "0.504 beats 0.490 raw" is a statement about *reading*, not about a memory system.

**One writer per reader.** Soft tokens live in a specific model's embedding space. Swap the decoder, retrain the adapter. That is cheap (minutes of H100 time) but it does mean the compressed store is not portable — the exact interoperability property that made text the default. This is the real cost of the paper's thesis, and it is not much discussed.

**Identity initialisation is the quiet good idea.** Starting $A = I$ means the writer begins as "pass the embeddings through unchanged" and only learns to deviate. Combined with frozen borrowed layers, the whole thing is a perturbation of the reader's own behaviour rather than a new learned encoder. Compare the [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]] zero-init trick — same philosophy, start at the identity function.

**The distillation target is the model itself.** $\mathrm{KL}(p_{\mathrm{full}} \| p_{\mathrm{comp}})$ uses the frozen reader as its own teacher, with the *only* difference being what it can see. This is a clean template for any "make the model behave the same with less input" problem, and it sidesteps needing a bigger teacher.

**Compression schedule is entirely hand-specified.** The role heuristic ($k_{\text{user}}{=}1$) is a guess that happens to fit LongMemEval, where the benchmark's answers are user facts by construction. On a different distribution — agent traces where tool observations carry the facts — the same heuristic would be the role-swapped ablation, i.e. 0.087. Do not port the heuristic; port the idea that allocation matters and measure it.

**Practical caveat if you build this.** Budget for output-format breakage. A soft prefix is out-of-distribution for an instruction-tuned model's input, and Qwen3 in particular started leaking `</think>` and emitting JSON envelopes. You will likely need constrained decoding or a format-repair pass before the accuracy numbers are usable.

**Open question worth chasing.** The authors want to learn $\pi$ with [[Proximal Policy Optimization Algorithms|RL]] against downstream answer reward under a latency budget. That is a discrete allocation problem over segments with a non-differentiable reward — the obvious formulation, and the ablation table already shows the reward surface is steep.

## Links

Related: [[Distilling the Knowledge in a Neural Network]] · [[KL Divergence]] · [[Cross Entropy]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Attention Is All You Need]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]] · [[Train Short, Test Long (ALiBi)]] · [[In Context Learning]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[On-policy Distillation with Verifiable Reward]] · [[Matryoshka Representation Learning]] · [[Linear Projection]] · [[Prefix Sliding for efficient test-time scaling]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Distillation]]

New topics worth writing: prompt compression (LLMLingua family), Gist tokens, ICAE / in-context autoencoders, xRAG, DeepSeek-OCR and optical context compression, LongMemEval, LongBench, latent reasoning (Coconut), LLM-as-judge evaluation protocols, soft prompts vs prefix tuning
