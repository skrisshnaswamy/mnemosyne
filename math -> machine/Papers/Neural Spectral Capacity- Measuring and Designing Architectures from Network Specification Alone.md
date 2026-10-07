---
title: "Neural Spectral Capacity: Measuring and Designing Architectures from Network Specification Alone"
authors: ["Zhu et al."]
year: 2026
arxiv: "2609.23087"
url: https://arxiv.org/abs/2609.23087
priority: Must-Read
read_on: 2026-09-25
tags: [paper, transformers, scaling]
---
## The Core Idea

Two Transformers can have the exact same parameter count and the exact same FLOPs, and still train to very different quality. #Params and #FLOPs are blind to *how* the budget was spent — 18 layers of width 512 versus 12 layers of width 640, 8 heads versus 16, an FFN ratio of 2 versus 4. That blindness is a problem, because "how should I spend the budget?" is the only question architecture design and structured pruning ever ask.

**Neural Spectral Capacity (NSC)** is a third scalar to sit beside #Params and #FLOPs. It takes the architecture *specification* — a list of matrix shapes — and returns a number. No weights, no data, no forward pass, no gradients. Same input as a parameter count, but sensitive to structure.

The trick is two steps stacked.

**Step 1: score a matrix by its singular values, not its size.** Treat a weight matrix $W \in \mathbb{R}^{m\times n}$ as a noisy communication channel $y = Wx + z$, with Gaussian probe input and Gaussian noise. The mutual information of that channel has a closed form that depends only on $W$'s [singular values](Derivative):

$$\psi(W) \;\coloneqq\; \ln\det(I + W^\top W) \;=\; \sum_{i=1}^{\min(m,n)}\ln(1+\sigma_i^2)$$

> [!NOTE] Spectral capacity
> The sum of $\ln(1+\sigma_i^2)$ over a matrix's singular values. Reads the matrix as $\min(m,n)$ parallel sub-channels, the $i$-th carrying signal-to-noise ratio $\sigma_i^2$. Zero singular values contribute zero, so it is well defined for rank-deficient matrices. ^spectral-capacity

This already breaks the tie with parameter count. Take the $4096\times 11008$ `gate_proj` matrix from LLaMA-7B layer 0: full rank gives $\psi \approx 3174$, its rank-1 truncation gives $\psi \approx 2.5$ — a $1270\times$ drop with $mn$ unchanged at 45M.

**Step 2: kill the need for the actual weights.** $\psi$ needs an SVD, which needs a matrix. But at initialization the weights are i.i.d. random with a known variance $s^2$, and the [Marchenko–Pastur law](Markov%20Chain%20Monte%20Carlo) says the *distribution* of the squared singular values of such a matrix is not random at all in the large-size limit — it converges to a fixed density determined only by the aspect ratio $\gamma = \min(m,n)/\max(m,n)$. So integrate $\ln(1+Ms^2\lambda)$ against that density and you get a deterministic function of $(m, n, s)$.

That is the whole unlock. The score becomes a formula in the architecture spec. And because the formula is a **sum over layers**, maximising it under a parameter or FLOP budget is a **knapsack problem**, solvable exactly by [[Dynamic Programming]] — not searched, *solved*.

This is what did not exist before. Prior training-free architecture proxies (SNIP, SynFlow, NASWOT, W-PCA, ZeroLM) all instantiate a random network and *measure* something on it with sampled inputs, and all are black-box scalars of the whole network. So you have to wrap them in evolutionary or random search, which returns the best architecture you happened to *visit*. NSC returns the best architecture that *exists* under the proxy, in under a second on one CPU core, for spaces up to $10^{32}$ candidates.

## The Methodology

**Which matrices count.** Linear projection weights only: per-head Q/K/V projections, the attention output projection, FFN matrices. Convolutions are reshaped to their im2col form $\mathbb{R}^{c_{\text{out}}\times c_{\text{in}}k^2}$ first. Embeddings, biases, [normalization](Layer%20Normalization) layers and nonlinearities are excluded.

Critically, [[Multi-Head Attention]] is decomposed **per head** — each head's Q/K/V projection in its own $d_h = d/H$ subspace is scored separately, rather than treating the fused $W_{QKV} \in \mathbb{R}^{3d\times d}$ as one matrix. Without this, two architectures differing only in head count would score identically. This is also what lets NSC score MHA vs [[Grouped Query Attention|GQA]] vs [[Fast Transformer Decoding- One Write-Head is All You Need (MQA)|MQA]] head sharing by construction.

**The aggregation.** Within a layer, sum. Across layers, sum.

$$\texttt{NSC}(f) = \sum_{l=1}^{L}\sum_{j=1}^{J_l} \psi_{\mathrm{MP}}(m_{l,j}, n_{l,j}, s_{l,j})$$

The within-layer sum has a real justification: block-diagonal composition adds log-determinants, so summing a layer's matrices *is* the spectral capacity of those matrices composed in parallel. For attention heads this parallelism is literal — they run side by side in disjoint subspaces.

The across-layer sum has **no such justification**, and the authors say so. Layers run in sequence, not parallel. Summing them is an empirical modelling choice. They tested the alternatives (Table 1, Spearman $\rho$ vs trained performance):

| Rule | FlexiBERT | GPT-2 | AutoFormer-T |
|---|---|---|---|
| $\sum_l \Psi_l$ | 0.884 | **0.968** | 0.810 |
| $\min_l \Psi_l$ | 0.178 | 0.813 | **−0.173** |
| $\prod_l \Psi_l$ | **0.917** | 0.712 | 0.816 |

The bottleneck rule ($\min$) is the information-theoretically "correct" serial composition, and it collapses. Structurally so: its score is always *one* layer's capacity, so on AutoFormer-Tiny with only 8 distinct layer configurations, 895 of 1001 architectures get the same score and the correlation flips negative — adding layers can only lower a minimum, but deeper is better in that space. The product conflates depth with quality: fine on FlexiBERT where depth takes two values, drops to 0.712 on GPT-2 where depth varies freely. Sum is the only rule that is both competitive and **additive**, and additivity is what NSC-DP needs.

**Where $s$ comes from.** A fixed initialization scheme, held constant across all candidates. Under [[Understanding the difficulty of training deep feedforward networks (Xavier init)|Xavier]], $s^2 = 2/(m+n)$; under [[Delving Deep into Rectifiers (He init, PReLU)|Kaiming]] fan-in, $s^2 = 2/n$; under HuggingFace's truncated normal, $s = 0.02$. So $s$ is itself a function of the shapes, and NSC reduces to a function of dimensions alone.

**NSC-DP (Algorithm 1).** Split the architecture as $a = (g, x_1, \dots, x_{L(g)})$:
- $g$ = network-level decisions (depth, embedding dim, stage layout). These fix $L$ and enter every layer's capacity, so they cannot be decomposed — they are enumerated exhaustively.
- $x_l$ = layer-level decisions (heads, FFN width, MLP ratio, LoRA rank). Conditioned on $g$, capacity separates: $\texttt{NSC}(a) = \sum_l \Psi_l(x_l; g)$.

Budgets have the same shape: $R(a) = \sum_l c_l(x_l; g) \le B$. So for each fixed $g$, the inner problem is a bounded or multiple-choice knapsack, solved exactly by DP in $O(L \cdot B \cdot |\mathcal{X}_l|)$. Each $\psi_{\mathrm{MP}}$ evaluation is a 1D quadrature costing ~10μs and is cached by $(m,n,s)$ — the whole LLaMA-7B search space contains only **13 distinct shape tuples**.

**Two concrete instantiations.**

*Transformer-XL:* $g = (d_{\text{model}}, L)$, $x_l = d_{\text{ff},l}$. Value per layer $\Psi_l = 2\psi_{\mathrm{MP}}(d_{\text{ff},l}, d_{\text{model}}, s)$ (the FFN pair; attention is fixed by $g$), cost $c_l = d_{\text{ff},l}(2d_{\text{model}}+1)$, budget 38.4M non-embedding params ±5%.

*LoNAS-LLaMA-7B pruning:* depth and width are fixed by LLaMA, so there is no outer loop at all — just one multiple-choice knapsack. Each of 32 blocks picks $(r_\ell, h_\ell)$ from [[LoRA|LoRA]] rank $\in \{32, 28\}$ and FFN width $\in \{11008, 9632, 8256, 6880, 5504\}$. That is $10^{32}$ candidates, solved in 0.46s.

**The free Pareto front.** The DP table already holds the optimum at *every* reachable budget $b \le B$. So one 0.46s backward pass gives you the whole accuracy-vs-FLOPs curve as $O(B)$ table lookups. Every proxy baseline needs a fresh search per budget.

## Ablation Studies and Experiments

**Ranking quality (FlexiBERT, 500 BERT architectures vs GLUE).**

| Method | $\tau$ | $\tau$ on pairs within 10% #Params | Time (500 archs) |
|---|---|---|---|
| **NSC** | **0.695** | **0.505** | **2 ms** (CPU) |
| #Params | 0.485 | 0.082 | — |
| #FLOPs | 0.552 | 0.329 | 2 ms |
| W-PCA | 0.635 | 0.417 | 88 s (A100) |
| ZeroLM | 0.527 | 0.355 | 63 s |
| SNIP | 0.289 | 0.237 | 73 s |
| GradNorm | 0.171 | 0.164 | 74 s |

The middle column is the one that matters. Restrict to architecture pairs whose parameter counts differ by less than 10% — the realistic capacity-allocation setting, where everything is the same size by construction — and #Params falls to 0.082, essentially random. NSC keeps 0.505. Same pattern on every family: 0.503 vs 0.322 (AutoFormer-T), 0.372 vs 0.116 (NATS-Bench-SSS), 0.535 vs 0.414 (MobileNetV3).

They also ran Kendall **partial** correlation controlling for #Params and #FLOPs jointly, which is threshold-free. On FlexiBERT, NSC keeps $\tilde\tau = 0.477$ where both #Params and #FLOPs collapse to $\le 0.06$.

**Architecture search.**

| Task | Method | Result | Search cost |
|---|---|---|---|
| Transformer-XL / WikiText-103 | TXL Base (human) | 23.279 PPL | — |
| | Synaptic Div. | 23.135 | 903 s GPU |
| | W-PCA | 23.669 | 965 s GPU |
| | **NSC-DP** | **23.087** | **2.0 s CPU** |
| AutoFormer-Tiny / ImageNet | AutoFormer oracle | 75.308 top-1 | 24 GPU-days |
| | TF-TAS | 75.234 | 0.5 GPU-day |
| | W-PCA | 74.752 | 206 s |
| | **NSC-DP** | 75.276 | **0.03 s** |

**LLaMA-7B → 5.7B structured pruning**, average over 8 commonsense tasks (BoolQ, PIQA, SIQA, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA):

| Method | Params | Avg₈ | Cost |
|---|---|---|---|
| Unpruned supernet | 6.7 B | 69.72 | — |
| **NSC-DP** | 5.7 B | **65.47** | **0.5 s CPU, no data** |
| W-PCA | 5.7 B | 63.83 | 45 min GPU |
| GradNorm | 5.5 B | 62.45 | 22 min |
| SNIP | 5.4 B | 60.73 | 22 min |
| SynFlow | 5.2 B | 60.68 | 23 min |

NSC-DP dominates every baseline on every individual task, and the baselines all need the pretrained weights plus a WikiText-103 calibration mini-batch (2 sequences × 2048 tokens). NSC-DP needs neither — ~5900× faster than the strongest baseline.

**The big consistency ablation, which is the one I would want to see.** The MP law assumes i.i.d. entries. Trained weights are emphatically not i.i.d. — heavy-tailed spectra, low-rank structure, LoRA factorisation. So: run NSC-DP twice on the LoNAS-LLaMA supernet, once with the closed-form $\psi_{\mathrm{MP}}$ and once with $\psi_W$ computed from the *actual trained* singular values. Across all 129 Pareto-optimal subnets:

- Kendall $\tau$ and Spearman $\rho$ between the two scores: **1.0000**
- Identical LoRA rank choices at every block in every bin
- Mean SVD-oracle regret: **0.039%**, max 0.104%
- Closed form: 0.5s CPU, no checkpoint. SVD route: ~6 min on an RTX 5090 plus a 13 GB checkpoint. **720×**.

So in this search space, the dependence of $\psi$ on *shape* swamps its dependence on the learned spectrum by an order of magnitude. That is either reassuring or damning depending on your priors.

**Things that did not work, or worked worse.**

- **The bottleneck aggregation is a structural failure**, not a near-miss. It is the serially-correct composition rule and it goes anti-correlated on AutoFormer-Tiny (−0.173).
- **NSC loses on GPT-2 under joint partial control**: $\tilde\tau = 0.329$ versus W-PCA 0.382 and ZeroLM 0.377. In that search space #Params and #FLOPs are rank-equivalent (fixed FFN ratio, only $d_{\text{model}}$ and $L$ vary), and after residualising both away NSC is third.
- **AutoFormer-Base**: NSC-DP hits 82.086 top-1, third behind TF-TAS (82.094) and the oracle (82.384). It ties or loses to the oracle at every tier — the win is always cost, never accuracy.
- **Initialization convention does matter a bit.** Xavier and Kaiming agree at pairwise $\rho \ge 0.995$. Constant-$s$ truncated normal is looser: on spec-faithful FlexiBERT, $\rho$ drops 0.884 → 0.788. Interestingly GPT-2 *improves* under TruncNorm ($\tau$ 0.814 → 0.849) because that is HuggingFace's actual default — matching $s$ to the real init helps.
- **They found a bug in the benchmark everyone else reports on.** The released FlexiBERT proxy-evaluation code hard-codes `hidden_size=256` and `intermediate_size=1024` for every candidate, collapsing two of the four search dimensions. 251 of 500 architectures declare $h = 128$ but are scored at $h = 256$. Fixing it moves NSC from $\tau = 0.544$ → 0.695 and W-PCA from 0.522 → 0.638, but moves **ZeroLM down**, 0.543 → 0.527 — its prior lead was partly an artefact of the collapsed spec.
- **Perfect correlations in Figure 1 are from 10 architectures** with a deliberate size spread. $\rho(\psi_0, \psi_T) = 1.00$ and $|\rho(\psi_T, \text{PPL})| = 1.00$ look impressive and mean little; the honest range at scale is $\rho \in [0.78, 0.97]$.

**What the ablations reveal about what is actually doing the work.** Three things, in order:

1. **Per-head decomposition of attention.** Without it, head count is invisible.
2. **Additive across-layer aggregation.** Not principled, but the only rule that both ranks well and enables exact DP.
3. **The MP closed form.** This is the *cost* win, not the *accuracy* win — the SVD-based $\psi_W$ ranks identically. It buys you 720× and removes the need for weights.

What is emphatically *not* doing the work: any information-theoretic claim about the forward pass. The channel is a device for defining $\psi$; the authors explicitly disclaim that the network realises it.

## Worth Remembering

**The concavity result, and why it is a double-edged finding.** Appendix D.4 proves that $\psi_{\mathrm{MP}}(d_{\text{ff}}, d_{\text{model}}, s)$ is strictly concave in $d_{\text{ff}}$. Two proofs, in fact: a deterministic one via the MP fixed-point equations and an envelope argument, and a probabilistic one via the matrix determinant lemma (adding a row changes $\psi$ by $\ln(1 + w^\top(I+W^\top W)^{-1}w)$, and rank-1 PSD updates can only grow eigenvalues, so marginal gains shrink).

Consequence, by Jensen: under a fixed total FFN budget, the NSC-optimal per-layer allocation is **uniform**. And indeed NSC-DP's discovered Transformer-XL is 18 layers, $d_{\text{model}} = 512$, $d_{\text{ff},l} \equiv 1152$ at every layer. On LLaMA it picks $r_\ell = 32$ everywhere (the larger LoRA rank; $\psi$ is monotone in projection dimension) and near-uniform FFN widths.

Uniform allocation is what humans already do. TXL Base is uniform. So NSC-DP's entire improvement on Transformer-XL comes from its *outer* enumeration re-picking $(d_{\text{model}}, L)$ — the part that is brute force, not DP. The exact-DP machinery provably returns the boring answer on the per-layer axis. That is a real limitation for anyone hoping this discovers non-obvious per-layer allocations.

**Global optimality is with respect to the proxy, not to quality.** The paper is careful about this and you should be too. NSC-DP returns $\arg\max \texttt{NSC}$ under the budget. Whether that architecture is good depends entirely on whether NSC correlates with quality — $\tau \approx 0.5$–0.8, which is useful but far from a ranking oracle.

**What NSC cannot see.** No nonlinearity. No normalization. No residual connections. No positional encoding ([[RoPE]] versus learned versus [[Train Short, Test Long (ALiBi)|ALiBi]] score identically). No tokenizer, no data, no task, no optimizer, no learning rate. It is a pure shape functional. That is the whole point and also the whole limitation: it can rank two FFN width allocations but it has nothing to say about whether you should use SwiGLU, whether to use pre-norm or post-norm, or how to set warmup.

**Why a score from random weights predicts trained performance at all.** The partial mechanism they offer: $\partial\psi/\partial\sigma_i = 2\sigma_i/(1+\sigma_i^2)$ peaks uniquely at $\sigma_i = 1$ — exactly where variance-preserving [[Understanding the difficulty of training deep feedforward networks (Xavier init)|Xavier]]/[[Delving Deep into Rectifiers (He init, PReLU)|Kaiming]] init puts the bulk of the spectrum. So $\psi$ is most sensitive in the regime initialization actually occupies. Note this is $\nabla_W \psi$, not $\nabla_W \mathcal{L}_{\text{task}}$; the two have no derived relationship, and the connection to training dynamics is purely empirical.

**Placement in the random-matrix literature.** Everyone else uses the MP law as a *null model*: Martin & Mahoney read heavy-tailed departures from MP in trained weights as a training-quality diagnostic; Berlyand et al. prune singular values *inside* the MP bulk as residual noise. At initialization the whole spectrum is in the bulk, so Berlyand's criterion would classify as noise exactly the mass NSC counts. The reframe here is that MP is the **deterministic equivalent** — the capacity of the bulk *is* the budget the specification hands to training.

**The aspect-ratio result is a nice concrete fact.** At fixed $mn = 4\times10^6$ with Xavier init, a $2000\times2000$ square matrix has $\psi_{\mathrm{MP}} \approx 1161$; a $4000\times1000$ matrix has $\approx 909$. A 28% gap at identical parameter count. Square maximises $\psi$ — the spectral analogue of uniform power allocation across parallel channels. Consistent with the folk wisdom that balanced layer dimensions train better.

**Practical caveats if you wanted to use this.**
- Fix one init scheme across all candidates, and prefer the one your model will actually use.
- Accuracy of the MP limit: relative error ~0.4% at $\min(m,n) = 128$, ~$10^{-4}$ by 1024. Fine for anything Transformer-sized; be careful below 128.
- It transfers to CNNs surprisingly well despite being derived for attention/FFN — best on CIFAR-10 NATS-Bench-SSS (93.44% vs oracle 93.65%), scoring all 32,768 candidates *exhaustively* in 0.54s. Meanwhile W-PCA, the strongest Transformer proxy, collapses to 43.29 ± 3.49% on ImageNet16-120 (NSC: 46.87%). Operator-agnostic beats operator-specific here.
- The MoE and GQA extensions are claimed "by construction" and never actually run. Per-layer expert allocation is additive and per-head decomposition already exists, so the claim is plausible, but it is untested.

**Open questions I would want answered.** Does NSC survive when you *do* change the nonlinearity or normalization — i.e. is its correlation an artefact of benchmarks that hold those fixed? Does it say anything useful about depth-vs-width at frontier scale, where [[Training Compute-Optimal Large Language Models (Chinchilla)|Chinchilla]]-style token budgets interact with shape? And given that the concavity proof forces uniform allocation, what does NSC-DP actually buy over "make everything uniform and grid-search $(d_{\text{model}}, L)$"?

## Links

Related: [[Attention]] · [[Multi-Head Attention]] · [[Grouped Query Attention]] · [[GQA- Training Generalized Multi-Query Transformer Models]] · [[Fast Transformer Decoding- One Write-Head is All You Need (MQA)]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[LoRA]] · [[Mixture of Experts]] · [[Dynamic Programming]] · [[Understanding the difficulty of training deep feedforward networks (Xavier init)]] · [[Delving Deep into Rectifiers (He init, PReLU)]] · [[Old Optimizer, New Norm- An Anthology (Muon)]] · [[Understanding Deep Learning Requires Rethinking Generalization]] · [[Deep Double Descent- Where Bigger Models and More Data Hurt]] · [[The Lottery Ticket Hypothesis]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Let's Scale Step by Step- Compute-Efficient Hyperparameter Transfer for Large-Scale Mixture-of-Experts]] · [[Mixed Dimension Embeddings]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Perplexity]] · [[Linear Projection]] · [[Derivative]] · [[Practical Bayesian Optimization of Machine Learning Algorithms]] · [[Matryoshka Representation Learning]]

New topics worth writing: Marchenko–Pastur law and random matrix theory for deep learning, Shannon transform and MIMO channel capacity, neural architecture search, training-free (zero-cost) NAS proxies, knapsack problems and multiple-choice knapsack, structured pruning of LLMs, once-for-all supernets and weight-sharing NAS, Kendall partial correlation and windowed rank controls, singular value spectra of trained weights, Jensen's inequality and concave resource allocation, μP and hyperparameter transfer
