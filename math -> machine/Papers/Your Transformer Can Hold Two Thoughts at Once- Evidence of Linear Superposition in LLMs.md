---
title: "Your Transformer Can Hold Two Thoughts at Once: Evidence of Linear Superposition in LLMs"
authors: ["Tikhonov et al."]
year: 2026
arxiv: "2609.29845"
url: https://arxiv.org/abs/2609.29845
priority: Good-To-Read
read_on: 2026-09-28
tags: [paper, transformers, llm]
---
## The Core Idea

Take two unrelated documents, $A$ and $B$. Tokenise both. At every position, average the two token embeddings:

$$e_t(z) = \tfrac{1}{2}\big(E(x_t) + E(y_t)\big)$$

Feed that single mixed sequence through an ordinary, unmodified pretrained LLM. Nothing else changes — same weights, same [[Causal Attention|causal mask]], one forward pass.

You would expect garbage. Averaging two high-dimensional vectors that mean different things should land you somewhere that means neither. Instead, the next-token prediction for *both* documents survives. Across Pythia, Llama and Qwen, the token the model would have predicted for $A$ alone appears in the top-10 of the mixed output about **30–40%** of the time, top-100 about **60–65%** of the time — out of a vocabulary of 50,000+.

The authors call this the **Superposition Linearity Hypothesis**: the model behaves approximately linearly end-to-end, so a linear mix of inputs gives roughly a mix of outputs.

> [!NOTE] Superposition Linearity Hypothesis
> If two input streams are linearly combined at the embedding layer, the model's output distribution approximates the average of the two independent output distributions: $P_{\text{mix}} \approx \tfrac12(P(\cdot|A) + P(\cdot|B))$. ^superposition-linearity

Two findings make this more than a curiosity.

**First, it is architectural, not learned.** They track it across Pythia training checkpoints. Superposition is *strongest at random initialisation* and gets steadily worse as pretraining proceeds. The Transformer is born able to do this, and the language-modelling objective slowly grinds it away. This is the opposite of an emergent capability.

**Second, it is reversible cheaply.** A short self-[[Distillation|distillation]] run — teaching the model to match the average of its own two single-stream outputs — pushes Pythia-2.8B's [[KL Divergence|KL]] to the ideal mixture from $1.86$ down to $0.27$.

What it *unlocks*, in theory: two continuations from one forward pass, and half the [[KV Cache|KV-cache]] per stream. What it actually delivers today is much weaker, and the honest reading of the numbers is in the last section.

This builds directly on the "your Transformer is secretly linear" line of work (Razzhigaev et al., 2024) which found layer-to-layer transitions are near-affine. The new claim is that the *global* input→output map inherits that linearity. See also [[Disentangling Representation Evolution in Transformers through Directional Decomposition]] for the residual-stream-geometry flavour of the same question.

## The Methodology

### Measuring the effect

Three instruments, all on FineWeb / FineWeb-Edu and TinyStories pairs, context lengths 32–512.

**1. Rank survival.** Run $A$ alone, take $\hat t = \arg\max \ell_A$. Then run the mixed input and ask: where does $\hat t$ rank in $\ell_{\text{mix}}$? Report the CDF $P(\text{rank} \le k)$.

**2. Distributional shape.** Compare $P_{\text{mix}}$ against the analytical target $P_{\text{target}} = \tfrac12(P_A + P_B)$ using KL, Jensen–Shannon, and a Wasserstein distance over the top-256 tokens (ground metric = cosine distance between token embeddings). To make numbers comparable across models they define a normalised ratio:

$$\mathcal{R}_{\mathcal{D}} = \frac{\mathbb{E}\big[\mathcal{D}(P_{\text{target}} \,\|\, P_{\text{mix}})\big]}{\mathbb{E}\big[\mathcal{D}(P(\cdot|A) \,\|\, P(\cdot|B))\big]}$$

$\mathcal{R} < 1$ means the mixed output is closer to the ideal mixture than two unrelated contexts are to each other. Base models sit at $\mathcal{R}_{\text{KL}} \approx 0.31$–$0.42$.

**3. Hidden-state additivity.** Three forward passes ($A$, $B$, mixed). Mean-centre each hidden state by its per-layer mean, then measure the $\ell_2$ distance between the *normalised* mixed state and the *normalised* sum of the two single-stream states:

$$\epsilon_{l,t} = \left\|\frac{\tilde h^{(\text{mix})}_{l,t}}{\|\tilde h^{(\text{mix})}_{l,t}\|_2} - \frac{\tilde h^{(A)}_{l,t} + \tilde h^{(B)}_{l,t}}{\|\tilde h^{(A)}_{l,t} + \tilde h^{(B)}_{l,t}\|_2}\right\|_2$$

Averaged over layers this gives $\bar{\mathcal{E}}$, the number that grows monotonically over pretraining.

### The restoration fine-tune

Plain self-distillation. Student = pretrained weights, teacher = frozen copy of the same model. Target is the teacher's averaged predictions on the two streams separately; student sees the mixed embedding.

$$\mathcal{L} = D_{KL}\big(P_{\text{target}} \,\|\, M_{\text{student}}(z)\big)$$

FineWeb, sequence length 128, ~200k steps, AdamW at $10^{-4}$ for the backbone and $10^{-3}$ for heads, distillation temperature $2.0$. Cost: ~114 GPU-hours on 2×A100-80GB for Pythia-2.8B. The paper claims this touches under $0.025\%$ of the original pretraining data volume.

### Why plain sampling from the mixed output fails

This is the sharpest piece of reasoning in the paper. If *logits* are being averaged, then probabilities scale with the **geometric** mean, not the arithmetic one:

$$P'_{\text{target}}(t) \propto \exp\!\Big(\tfrac12(\ell_A(t) + \ell_B(t))\Big) \propto \sqrt{P_A(t)\,P_B(t)}$$

A token that is certain under $A$ and near-zero under $B$ gets crushed. So even when both true tokens sit comfortably in the top-5, neither can reliably be rank 1, and greedy decoding alternates incoherently between the two documents.

> [!NOTE] Geometric-mean obstruction
> Averaging logits multiplies probabilities. Any token strong in one stream and weak in the other is suppressed, so you can fit the mixed distribution well and still be unable to decode either stream. Fitting $\neq$ decoding. ^geometric-mean-obstruction

### Joint Contrastive decoding

The proof-of-concept fix. A small model from the same family ($0.5$B guiding $3$B) provides per-stream direction, and you do arithmetic on logits:

$$\tilde\ell^{(A)} = \ell_{\text{large}}(z) + \alpha\,\ell_{\text{small}}(A) - \beta\,\ell_{\text{small}}(B)$$

and symmetrically for $B$. $\alpha, \beta$ start at $1$ and are trained jointly with the backbone under symmetric per-stream cross-entropy. Structurally this is the same "push towards one thing, away from another" move as [[Classifier-Free Guidance]].

### The attention-patching experiment

To ask *why* attention doesn't destroy the mixture, they run three single-stream perturbations on text $A$:

| Setup | What is changed |
|---|---|
| Vanilla | nothing |
| **Donor patch** | post-softmax attention weights from an unrelated text $C$ substituted at every layer and head; Q/K/V, [[RoPE]], value paths from $A$ untouched |
| **Permutation patch** | $A$'s own attention rows randomly permuted within the causal prefix — row sums and weight multisets preserved, positional structure destroyed |

Donor patching keeps natural attention *shape* but decouples it from content. Permutation keeps the LM-head frequency prior but destroys shape. Together they separate the two ingredients.

## Ablation Studies and Experiments

### The frequency-prior control (Appendix B) — the most important sanity check

Could this all be "the model just predicts common tokens"? They take single-stream logits for $A$ and ask where $B$'s ground-truth token lands. Top-3: **1.12%**. Top-10: **2.63%**. Top-100: **10.41%**. Against 30–40% and 60–65% under actual superposition. The effect is an order of magnitude above the frequency floor.

### Predictable vs content tokens — where the aggregate number is lying

Qwen2.5-3B, median rank of $A$'s vanilla top-1 token, split by token type (~65% of positions are "predictable" — punctuation, function words; ~35% are content words):

| Setup | Predictable (med. rank / top-1 %) | Content (med. rank / top-1 %) |
|---|---|---|
| Embedding mixing, base | 6 / 24.8% | **284** / 4.2% |
| Donor attention patch | 3 / 33.0% | 111 / 5.4% |
| Permutation patch | 3,079 / 1.3% | 19,246 / 0.0% |
| Embedding mixing, fine-tuned | 8 / 21.6% | **5** / 22.8% |

Three readings fall out.

**The headline 30–40% is carried by the boring tokens.** Base-model superposition on actual content words is close to collapse (median rank 284 out of 150k).

**Permutation is the decisive negative control.** Frequency prior intact, attention shape destroyed → everything collapses. $\mathcal{R}_{\text{KL}}$ goes $0.27 \to 0.68$, top-10 agreement $53\% \to 10.1\%$, median rank $8 \to 8{,}148$. So what survives donor patching is the *joint* contribution of the frequency prior plus the structural shape of natural attention (locality bands, [[Sparse Attention#The bit that surprised me 🪤|attention sinks]], head specialisation) — properties any natural text shares with any other. Neither ingredient alone is enough.

**Fine-tuning fixes the hard half, not the easy half.** Content-position median rank $284 \to 5$; predictable positions barely move ($6 \to 8$). The fine-tune is not gaming the frequency prior — it genuinely restores parallel semantic processing.

### LAMBADA — where self-agreement metrics disagree with task metrics

LAMBADA targets are content words at the end of long passages, i.e. exactly the regime the predictable majority cannot rescue. 200 prompts, truncated to 128 tokens.

| | argmax accuracy | target median rank |
|---|---|---|
| Vanilla | 73% | — |
| Donor attention patch | **0.5%** | 2,350 |
| Embedding mixing | **2.25%** | 339 |

Donor patching looks benign by self-agreement (median rank 8) and is catastrophic by task accuracy. And the order flips: mixing has *worse* self-agreement (median 19 vs 8) but $4.5\times$ the LAMBADA accuracy and $7\times$ better target rank. So embedding mixing carries case-specific signal beyond "frequency prior + attention shape" — and does it for two streams at once.

### Distributional metrics, base models (FineWeb)

$\mathcal{R}_{\text{KL}}$ / $\mathcal{R}_{\text{JS}}$ / $\mathcal{R}_{\text{WS}}$ at $L=32$:

- Pythia-160M: 0.35 / 0.40 / 0.63
- Pythia-2.8B: 0.42 / 0.54 / 0.71
- Llama-3.1-8B: 0.37 / 0.57 / 0.69

Note the direction: **bigger models are worse at superposition.** Consistent with the pretraining-degrades-it story. After fine-tuning, Pythia-2.8B drops to $\mathcal{R}_{\text{KL}} = 0.06$.

$L=512$ is very slightly better than $L=32$, and total-variation distance to the target is a bit high for the first ~20 positions then flat — so this is not a short-context artefact.

### Layer-wise geometry (Appendix F)

Linearity score $1 - \min_A \|\tilde X A - \tilde Y\|_F^2$ between consecutive layers has a **U shape**: layers 0–5 high, middle layers (6–20) down to ~$0.65$, final third back above $0.95$. That terminal near-linear stage is the proposed mechanism — the superposed signal isn't destroyed on its way out to the unembedding.

### Three streams (Appendix I)

$N=3$ still works, with graceful degradation: $\mathcal{R}_{\text{KL}}$ rises by $+0.04$ to $+0.09$. Worse at $L=512$ than $L=32$. No qualitative break.

### Confidence matters (Appendix D)

Tokens the single-stream model predicted with $P > 0.5$ survive mixing at median rank $\approx 3$. Low-confidence predictions get eaten by interference.

### Decoding results — and what failed

LAMBADA mean accuracy across both streams from one superposed pass; Jaccard = token overlap between the two generated streams (lower = better separation):

| Backbone / guide | Method | single (big/small) | mixed | Jaccard |
|---|---|---|---|---|
| Qwen2.5-3B / 0.5B | Pretrained | 0.592 / 0.437 | 0.168 | 0.126 |
| Qwen2.5-3B / 0.5B | Joint Contrastive | 0.592 / 0.437 | **0.345** | **0.061** |
| Llama-3.2-3B / 1B | Joint Contrastive | 0.643 / 0.540 | **0.430** | **0.067** |
| Pythia-1.4B / 160M | Joint Contrastive | 0.499 / 0.225 | 0.110 | 0.080 |

**What did not work:**

- **Two-Head separation** (learnable $W_A, W_B$ to break the averaging symmetry, two output heads, dynamic loss balancing). Llama-3.2-3B: mixed accuracy $0.105$ — *worse* than raw pretrained mixing ($0.182$) — and Jaccard $0.235$, i.e. the two streams became more alike, not less. Without the balancing trick it collapses onto one stream entirely.
- **Mixed Distillation alone** (the Sec. 4 objective, no guide model). Barely moves decoding: Qwen $0.168 \to 0.207$, Llama $0.182 \to 0.174$. Excellent at fitting the mixed distribution, near-useless for reading the streams back out. This is the geometric-mean obstruction made visible.
- **Post-hoc logit arithmetic** — freeze both models, fit only $\alpha, \beta$ on a calibration set. Loses to jointly trained Joint Contrastive.
- **The restoration fine-tune costs real single-stream quality.** Pythia-2.8B LAMBADA $0.544 \to 0.357$; Qwen2.5-3B $0.602 \to 0.460$. FineWeb [[Perplexity]] for tuned distillation: Qwen $12.44 \to 65.52$, Llama $11.20 \to 71.44$. That is not a tax, that is a different model.
- **TinyStories LLM-as-judge** (GPT-5.2 grader, 1–10). Independent baseline grammar $4.52$ → raw superposed $2.59$ → fine-tuned $2.13$... sorry, fine-tuned $3.69$. But creativity stays dead: $5.72 \to 2.08 \to 2.13$. Fine-tuning recovers grammar and consistency; it does not recover interesting text.

## Worth Remembering

**The throughput claim does not survive contact with the numbers.** Appendix G.5, tokens/s, prompt 128 / generate 128:

| Pair | Separate (fused) | Guided | Big, batch=2 | 2× Big sequential |
|---|---|---|---|---|
| Pythia-2.8B+160M | 109.6 | 79.4 | **113.1** | 56.0 |
| Llama-3B+1B | 81.5 | 52.4 | 81.4 | 41.3 |

The "$2\times$" in the conclusion is measured against running two streams *sequentially*. Against ordinary batch-size-2 inference — which is what any serving stack already does — fused decoding is at best a tie, and the Guided variant (the only one that decodes acceptably) is **30% slower**. Memory tells the same story: Guided peaks at $12.93$ GB for Llama-3B+1B vs $6.72$ GB for a single vanilla pass, because both models must be resident. The KV-cache halving is real in principle but is not demonstrated as a net win here. Contrast with [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)|PagedAttention]] and [[Continuous Batching]], which attack the same bill and actually deliver.

**Worse: the small model alone is competitive.** Llama Joint Contrastive gets $0.430$ mixed accuracy; just running the $1$B guide on each stream independently gets $0.540$. Same for perplexity — Joint Contrastive lands at $19.31$ against the $0.5$B model's $18.96$. The current best decoder does not beat "run the small model twice." The authors are upfront that this is a proof of concept, not a system.

**So what is the actual value?** The scientific result, not the engineering one. Two things are genuinely surprising and reusable:

1. Superposition is **maximal at initialisation and degrades with training**, and degrades more in bigger models. That is a statement about what the [[Cross Entropy|cross-entropy]] objective does to representation geometry, and it runs against the usual "capabilities emerge with scale" intuition.
2. The permutation-vs-donor dissociation is a clean method for asking *what part of attention is doing the work* on any given behaviour. Permutation patching (keep row sums, destroy structure) is a control worth stealing.

**The predictable/content split is a general warning.** ~65% of token positions are punctuation and function words. Any aggregate next-token metric on natural text is dominated by them. This paper's own headline number was two-thirds artefact until they stratified — and their Appendix H shows the split is systematic, not a hand-tuned heuristic (rank 2–10 buckets are 22/25 function words; rank 2000–20000 buckets are almost all content). Check this before trusting any "signal survives" claim.

**Limitations the authors state.** Contexts $\le 512$, mostly English, text-only. They explicitly have not tested mixing across modalities, where embedding geometries differ far more.

**Connections.** Prior multiplexing work (DataMUX, MIMONets, RevMUX, Superposed Decoding) *engineers* superposition with dedicated mux/demux layers or binding keys. The distinct claim here is that off-the-shelf models already have it, and fine-tuning *restores* rather than creates. Separately, "superposition" in the Elhage et al. sense — many features sharing few dimensions — is a different phenomenon that happens to share the word; don't conflate them.

**Open questions.** Can the geometric-mean obstruction be dodged by mixing with unequal weights, or by mixing in a rotated basis rather than the raw embedding basis? Does a model pretrained *with* a superposition term in the loss keep both single-stream quality and linearity, instead of trading one for the other? And is the deep-layer terminal linearity causal, or just correlated — would forcing mid-layers to be more linear help?

## Links

Related: [[Attention]] · [[Attention Is All You Need]] · [[Multi-Head Attention]] · [[Causal Attention]] · [[Embeddings]] · [[KV Cache]] · [[KL Divergence]] · [[Cross Entropy]] · [[Distillation]] · [[Classifier-Free Guidance]] · [[Speculative Decoding]] · [[Prefill and Decode]] · [[Continuous Batching]] · [[Perplexity]] · [[Tokenization]] · [[Sampling Parameters]] · [[Disentangling Representation Evolution in Transformers through Directional Decomposition]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]] · [[Cost and Latency]] · [[Linear Projection]]

New topics worth writing: Residual stream linearity, Feature superposition (Elhage toy models), Input multiplexing (DataMUX / RevMUX), Attention patching as an interpretability method, Function vs content token stratification in LM metrics, LAMBADA, Jensen–Shannon divergence, Wasserstein distance on token distributions, Self-distillation objectives
