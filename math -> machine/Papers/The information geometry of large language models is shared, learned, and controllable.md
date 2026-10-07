---
title: "The information geometry of large language models is shared, learned, and controllable"
authors: ["Dario Picozzi"]
year: 2026
arxiv: "2609.11063"
url: https://arxiv.org/abs/2609.11063
priority: Good-To-Read
read_on: 2026-09-28
tags: [paper, transformers, llm, optimization, theory]
---
## The Core Idea

A language model turns an internal activation vector into a probability distribution over the next token. Two things about that fact are usually confused.

The first is **the inside**. The hidden state has no privileged coordinate system. Take any invertible matrix $A$, replace the hidden layer $h$ with $Ah$, and fold $A^{-1}$ into everything downstream. The model computes the *exact same function*. But Euclidean distances between activations change. So do cosine similarities, [[Embeddings|embedding]] neighbourhoods, feature-importance scores, [[LoRA]]-style regularisers, and every "are these two models similar?" measure built on activation geometry. All of them are reading a coordinate choice.

The second is **the outside**. The space of probability distributions *does* have a privileged geometry. Chentsov's theorem says so: require a metric to be invariant under transformations that preserve statistical information, and the Fisher–Rao metric is the unique answer up to an overall scale.

The paper's move: **stop measuring in activation space, measure in output space.** Push the Fisher–Rao metric back through the network to the layer you want to intervene on. You get a matrix $G$ that answers "how much does the model's behaviour change if I nudge this activation?" — and that answer is coordinate-free.

> [!NOTE] Pullback Fisher metric
> With $p_h = \mathrm{softmax}(W_U h_L(h) + b_U)$, output Fisher $H_h = \mathrm{Cov}_{p_h}(w)$ (covariance of unembedding rows under $p$), and $J_h = \partial h_L / \partial h$, the metric at the intervention layer is $G_h = J_h^\top H_h J_h$. Its meaning: $\mathrm{KL}(p_h \| p_{h+\delta h}) = \tfrac{1}{2}\delta h^\top G_h \delta h + O(\|\delta h\|^3)$. Length in this metric *is* behaviour change. ^pullback-fisher

Four things fall out that did not exist before, because nobody had a single object doing all of them:

1. **Cross-model comparison with no shared anything.** Run the same prompts through two models. Each gives a matrix of Fisher–Rao distances between its own next-token distributions. Compare the two matrices by rank correlation. No shared vocabulary, no shared architecture, no activation alignment. This is [[Attention|representational]] similarity analysis moved to the output.
2. **A prediction, not an observation, of convergence.** The [[The Bitter Lesson (essay)|Platonic Representation Hypothesis]] observes that independently trained models converge. Here it *follows* from predictive fit: two accurate models of the same language must converge in output geometry. Activation convergence has no such theorem.
3. **Minimum-disturbance control.** Want to change one behaviour without breaking others? The damped [[Momentum|natural gradient]] $\delta h \propto (G + \alpha R)^{-1} q$ is the provably minimal-disturbance step, and the paper predicts *in advance* how much worse a plain Euclidean step will be.
4. **One correction across five operations.** Replace the Euclidean inner product with $G$ in steering, knowledge editing, feature attribution, sparse-dictionary learning and fine-tuning. Every one improves.

The reason this did not exist before is not that Fisher information is new — [[Adam- A Method for Stochastic Optimization|Adam]], [[Tensor Programs V- Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (muP)|μP]] and K-FAC all touch curvature. It is that computing $G$ at an internal layer of a billion-parameter model over a 250,000-token vocabulary looks impossible. The paper's matrix-free solve makes it cheap, and proves the iteration count is independent of model width.

## The Methodology

### The metric and the step

Write the next-token law as $p = \mathrm{softmax}(W_U h_L + b_U)$, with unembedding rows $w_a$. The output Fisher is the probability-weighted covariance of those rows:

$$H = W_U^\top\left(\mathrm{diag}(p) - pp^\top\right)W_U = \sum_a p_a (w_a - \bar{w})(w_a - \bar{w})^\top, \quad \bar{w} = \sum_b p_b w_b$$

Note the centring is automatic. $H$ never sees the mean read-out direction — the same gauge freedom [[Cross Entropy|softmax]] has.

For an intervention at layer $\ell$, let $J = \partial h_L / \partial h$. Then $G = J^\top H J$. It can be singular: its null space is exactly the activation directions the read-out maps to no change in $p$. Those directions are invisible in behaviour by construction.

Given an objective gradient $q = \nabla_h \phi$, the step is

$$\delta h \propto (G + \alpha R)^{-1} q$$

with damping $\alpha = c\,\lambda_{\max}(G, R)$, typically $c = 10^{-2}$, and reference metric $R$. Native experiments use $R = I$.

**The covariance test that matters.** Under $h \mapsto Ah$, $G \mapsto A^{-\top}GA^{-1}$. If you also transport $R \mapsto A^{-\top}RA^{-1}$, the step maps as $\delta h \mapsto A\delta h$ exactly — verified numerically to relative error $7.4 \times 10^{-13}$ across condition numbers up to 100. If you instead *reset* $R$ to a fresh identity in the new chart, the direction changes by median relative error 1.23. So the damped solve is covariant only if you carry the reference metric with you.

### Computing it without building it

Two matrix-free routes, both avoiding a $|V| \times d$ Jacobian:

**Low-rank Woodbury.** Restrict to the top-$|S|$ tokens by mass, renormalise, factor $H_S = BB^\top$ with $B = W_U[S,:]^\top(\mathrm{diag}(p_S) - p_S p_S^\top)^{1/2}$. Form $M = J^\top B$ with $|S|$ vector–Jacobian products. Then $(G_S + \alpha I)^{-1}q = \alpha^{-1}(q - M(\alpha I + M^\top M)^{-1}M^\top q)$ — an $|S| \times |S|$ solve.

**Exact conjugate gradients.** Compute $Gv$ in two passes: a JVP gives $Jv$, the closed form gives $H(Jv)$, a VJP gives $J^\top(\cdot)$.

> [!NOTE] Width-independent iteration bound
> With $\alpha = c\,\lambda_{\max}(G)$, the regularised condition number is $\kappa \le (1+c)/c$ — it depends only on $c$, not on $d$. Standard Chebyshev CG convergence then gives a worst-case iteration count independent of model width. At $c = 10^{-2}$ and Euclidean relative residual $10^{-6}$: **85 iterations**, at any scale. ^cg-bound

All solves run in float32. $G$ is severely ill-conditioned ($\kappa \sim 10^5$–$10^7$); in bfloat16 the recovered direction has absolute cosine only $0.056$–$0.542$ against the float32 answer. Do not use bf16 here.

### Why cross-entropy produces this geometry

Fix a checkpoint. Let $P$ be the population law of $(h, Y)$ and define the model-induced joint $Q(dh, a) = P_H(dh)\,q(a\mid h)$. Then

$$\mathcal{L}(q) - H_P(Y \mid H) = \mathrm{KL}(P \| Q)$$

Excess cross-entropy *is* the KL divergence to a Gibbs law whose token-conditioned slices form one exponential family with the read-out rows $w_a$ as natural parameters. Writing $B(h) = \log\sum_a \exp(w_a^\top h + b_a)$, its Hessian $\nabla^2 B(h) = \mathrm{Cov}_Q(w_Y \mid h)$ — which is exactly the output Fisher. Next-token training induces this geometry; it is not bolted on afterwards.

### Identification: what behaviour pins down

Let $T$ be the language-defined read-out subspace, $S$ a candidate, $d(S,T)$ their chordal distance, and $\mathcal{K}(S)$ the predictive risk after all nuisance parameters (intercept, per-context coordinates) are optimised away. The global margin is

$$\mathcal{K}(S) \ge c\, d^2(S, T), \qquad c = \min\left\{\frac{p_*^2 m_z}{64},\ \frac{w_* p_*}{4r}\right\} > 0$$

with $p_*$ the minimum outcome probability, $w_*$ the minimum context weight, $m_z = \lambda_{\min}(\Sigma_z)$. Zero risk means exact recovery at matched rank. Finite risk gives a square-root rate: $d \le (\sqrt{\zeta_r} + \sqrt{\varepsilon_m})/\sqrt{c}$, where $\zeta_r$ is the rank-$r$ approximation error and $\varepsilon_m$ the excess KL.

The contrast: for activation geometry there is an explicit counterexample. Four hidden rows, two diagonal charts, an identical non-constant read-out — Pearson agreement $-0.2053$, Spearman exactly $-4/17$, at *identical outputs*. No positive floor on hidden-geometry agreement can be a function of output risk alone.

### The spectral prediction

Use the standardised centred read-out $U = \mathrm{orth}(P_0 W_U)$, with $P_0 = I - |V|^{-1}\mathbf{1}\mathbf{1}^\top$ removing the softmax gauge. Write $u_a$ for its rows, $\bar{u} = \sum_a p_a u_a$, and define the **weighted profile**

$$q_a = p_a \lVert u_a - \bar{u} \rVert^2$$

Set $\widehat{\lambda}_k = q_{(k)}$ (sorted descending). The effective-dimension prediction with **no fitted parameters** is

$$\widehat{N}_{\mathrm{eff}}(\alpha) = \sum_{k=1}^{r}\frac{q_{(k)}}{q_{(k)} + \alpha}$$

against the measured identity $N_{\mathrm{eff}}(\alpha) = \sum_k \lambda_k/(\lambda_k + \alpha)$.

The inheritance theorem bounds it: with $M_K$ the Gram matrix of the top-$K$ centred unit read-out directions, frame bounds $a_K = \lambda_{\min}(M_K)$, $A_K = \lambda_{\max}(M_K)$, and tail spread $\varepsilon_K$,

$$a_K q_k \le \lambda_k(H) \le A_K q_k + \varepsilon_K \quad (k \le K)$$

A more robust fractional-frame version needs only that enough leading Gram modes stay above a threshold $b$: $\lambda_k(H) \ge b\, q_{k + r_K(b)}$ where $r_K(b)$ counts Gram eigenvalues below $b$.

### The cost-ratio prediction

Let $A = G + \alpha R \succ 0$. For target change $q^\top\delta = b$, the unique minimum-cost step is $\delta^\star = bA^{-1}q/(q^\top A^{-1}q)$, and every other feasible step pays an **exact** excess:

$$\tfrac{1}{2}\delta^\top A\delta - \frac{b^2}{2q^\top A^{-1}q} = \tfrac{1}{2}\lVert \delta - \delta^\star \rVert_A^2$$

Define the $G$-cost per unit squared objective change as $\Pi_G(u) = (u^\top G u)/(q^\top u)^2$. The prediction of how much worse Euclidean control will be, computed *before* intervening, with no fitted scale:

$$R_{\mathrm{pred}}(q) = \frac{\Pi_G(q)}{\Pi_G(A^{-1}q)}$$

In the undamped case this reduces to the Kantorovich form $\Gamma_A(q) = (q^\top A q)(q^\top A^{-1}q)/(q^\top q)^2 \ge 1$, bounded above by $(M+m)^2/4Mm$ for spectrum in $[m, M]$. Equality at 1 exactly when $q$ sits in a single eigenspace — the isotropic limit where natural and Euclidean coincide. **Damping monotonically destroys the anisotropy and the advantage together**; $\Gamma_{A_\alpha}(q)$ is non-increasing in $\alpha$ and tends to 1.

### Experimental setup

Models: Pythia 70M–6.9B, GPT-2/GPT-2-XL, GPT-Neo, Qwen2, Qwen1.5-Chat, BLOOM, Mamba, RWKV, Mistral-7B, StarCoder2, OLMo-2-1B. Interventions default to the residual stream at $L = \mathrm{round}(0.45 \times \mathrm{depth})$.

Objectives are summed continuation log-odds, $\phi(h) = \sum_t [\log p(y_w^{(t)}) - \log p(y_l^{(t)})]$, with $q = \nabla_h \phi$. All comparisons are **matched-response**: binary-search each method's step scale until the *realised* $\Delta\phi$ hits a target, then read off every other quantity there.

Off-target change has three conventions: KL over the complement of the objective's token support; a concept-decomposed KL; and for scalar objectives the Fisher-orthogonal residual $\mathrm{KL}(p_0\|p_s) - \Delta\phi^2/(2\mathrm{Var}_{p_0}\phi)$, clamped at zero.

## Ablation Studies and Experiments

### The metric actually predicts output change

99 model–depth–prompt cells, 11 models, six families, 125M–1.5B, injection at quarter/half/three-quarter depth. The prediction $\tfrac12 \delta h^\top G\delta h$ against realised KL: **median ratio 1.000**, range $0.948$–$1.224$. No fitted scale.

### Identification holds quantitatively

Held-out profiled loss against squared subspace distance, Pythia 70M/160M/410M/1.4B, rank-32 reference from Pythia-6.9B, two disjoint calibration folds and 100 held-out contexts: $R^2 \ge 0.9999$ on all eight paths. Real read-out deviations have steeper slopes than angle-matched random controls (slope ratios $1.14$–$1.27$).

**The controlled language experiment is the causal half.** Transformer, gated-recurrent and diagonal-recurrent models, two capacities each, trained on eight pairs of synthetic languages with known conditional laws, *identical* token frequencies and conditional entropy. At matched accuracy (excess cross-entropy below $\Delta^2/32$):

- Changing the assigned law recovers **99.78%** of the imposed squared geometric separation.
- Changing architecture within a law changes geometry by **0.22%** — the normalised across-law effect exceeds the cross-architecture effect by $0.9957$.
- The assigned law is the closer geometric match in all 8 pairs (exact sign test $P = 1/256$).

### Output geometry is shared; activation geometry is not

Ten models, seven training pipelines, four tokenizers, 70M–7B, transformer/state-space/recurrent. Natural text battery of 200 WikiText prefixes.

| geometry | mean rank agreement |
|---|---|
| output Fisher–Rao | **0.88** |
| mid-layer activations | 0.62 |
| last-layer activations | 0.61 |

The gap's bootstrap interval excludes zero. Under random anisotropic reparameterisations with condition number swept to $10^4$: output agreement constant to machine precision; CKA drops 10%, mutual $k$-NN 15%, cosine RSA 28%. Partialling out character $n$-gram surface geometry or the strongest corpus $n$-gram predictor leaves $\ge 98\%$ of the agreement.

Tokenizer-independent check: push each next-token law to the distribution of the first byte of remaining text. Cross-tokenizer agreement **0.910**, up from 0.899 at token level; Bhattacharyya affinity does not decrease on any of 28 pairs.

**Why it holds is more interesting than that it holds.** The worst-case rank-stability bound is vacuous here: median per-context root-probability disagreement is $0.30$–$0.71$, while the median entry-pair margin is only $0.08$–$0.14$. The rescue is **coherence** — the measured $|\langle w_x, u(y)\rangle|/\rho(x)$ has median $0.024$–$0.034$. For calibration, fully isotropic disagreement in a $|V| \approx 5\times10^4$ space would give $\approx 0.004$. So disagreement is structured, but an order of magnitude below the level that would flip the relational order. Models disagree a lot per context; they disagree in directions nearly orthogonal to the relational structure.

The weakest pair (70M vs 6.9B) has coherence $0.085$ and correspondingly the lowest agreement (0.777) — the mechanism tracks the data.

### The attenuation identity, and where it breaks

Write each model's relational map as $C_m = S + e_m$ against a fixed reference. Then, with no assumptions:

$$\mathrm{Corr}(C_m, C_n) = \frac{v_S + c_m + c_n + k_{mn}}{\sqrt{(v_S + v_m + 2c_m)(v_S + v_n + 2c_n)}}$$

This reconstructs all 112 measured model-pair configurations to $10^{-10}$. Fitted on half the contexts, the exchangeable restriction predicts held-out agreement with pooled median absolute error $0.009$.

**But the estimator has a training-time onset, and the paper reports it honestly.** Of 120 full-sample attempts, **24 are undefined** (non-positive variance denominator, all at steps $\le 64$). The defined early predictions are bad: median error $0.25$–$0.31$ at steps 1–16, $0.056$ at step 64. From step 256 onward: $0.0036$–$0.014$. What this identifies is *when* an exchangeable predictive regime begins, not a universally valid formula.

There is also a $\pm 1/2$ signature for idiosyncratic deviations: mutually orthogonal equal-norm residuals give $\cos(w_{ij}, w_{ik}) = +1/2$ and $\cos(w_{ij}, w_{jk}) = -1/2$. Measured: $+0.57$ / $-0.51$ on the cross-architecture trio, $+0.487$ / $-0.487$ across PolyPythias seeds. A common disagreement axis would instead approach 1.

### Shared vs residual: the shared half carries the meaning

Consensus = mean of rank-transformed distance matrices; residual = each model minus consensus. Alignment against a fixed Sentence-BERT geometry:

| battery | consensus | residual |
|---|---|---|
| natural | 0.044 | $-0.001$ |
| templated | 0.102 | $+0.003$ |
| factual-relation | **0.440** | $+0.051$ |

On the factual battery, $+0.231$ (interval $+0.159$ to $+0.316$) survives partialling out surface form.

Transfer probe (each prompt = its Fisher–Rao distances to 40 fixed anchors, $z$-scored, multinomial logistic, eight-way):

| representation | accuracy |
|---|---|
| within-model | 0.724 |
| consensus alone | **0.700** |
| across models | 0.659 |
| residual, within its own model | 0.407 |
| residual, across models | 0.091 (below 0.125 chance) |

**What is *not* shared is the important negative.** Raw cross-family eigenvector overlap in the top six modes is 0.698. Against a shuffled-word null (0.025) that looks like total sharing. Against a **displacement-magnitude-matched null** that preserves each word's displacement size and randomises only direction, the null is 0.636 — so the genuine residual is $+0.063$, about **9%** of the raw number. The other 91% follows from the shared output distributions alone. The residual concentrates in leading modes ($+0.080$ ranks 0–5, falling to $+0.013$ past rank 90) and is graded by lineage: $+0.088$ within family, $+0.064$ across, $+0.061$ with a shared tokenizer alone.

A related scale confound, from the Aristotelian-view critique: CKA carries a non-vanishing permutation null that grows with dimension ($0.316$ here), and its raw score more than doubles across the ladder ($0.35 \to 0.73$). Rank agreement of distance matrices has null $0.001$, dimension-independent, and stays flat ($0.85 \to 0.89$).

### Human completions

Peelle et al. norms, 512 contexts, 32 native samples at temperature 1, no top-$k$/nucleus, eight-token limit (reached by 27–36% of samples, all retained), embedded with MiniLM and mapped through a three-bandwidth Gaussian kernel with 768 random Fourier features:

| Pythia size | 70M | 160M | 410M | 1B | 1.4B | 2.8B |
|---|---|---|---|---|---|---|
| squared distance | 0.339 | 0.326 | 0.308 | 0.303 | 0.300 | 0.296 |

Slope against $\log$ params $-0.0118$ ($-0.0130$, $-0.0106$). Reliability-corrected CKA $0.82$–$0.85$, not monotone between adjacent sizes.

A risk-to-alignment relation fitted only on Pythia sizes and checkpoints, applied to OLMo checkpoints and five external models **without refitting**: correlation $0.758$, RMSE $0.0295$.

Independent replication on DERCo (640 positions from five unused narratives, 64,000 responses, positions chosen by a fixed hash before seeing any response value): slope $-0.00872$ ($-0.01202$, $-0.00556$). Final-minus-initial checkpoint change $-0.121$ for Pythia-160M, $-0.110$ for OLMo.

**Model-only calibration.** A 65-outcome affine ridge map, trained purely on model→model probability laws on the *earlier* corpus with no human response involved, fixed before DERCo was examined. Expected human log-score gains: BLOOM $+0.0507$, Pythia $+0.0452$, Mamba $+0.0408$, RWKV $+0.0387$, GPT-2 $+0.0220$, GPT-Neo $+0.0196$, **Qwen2 $-0.0030$** nats.

**What failed:** maps trained *on human responses* from the earlier corpus made things worse in all seven families ($-0.182$ to $-0.106$ nats). The gain is specific to model-law calibration, not generic cross-corpus fitting.

On the de Varda probability-direction corpus (1,726 contexts), simultaneous 95% lower bounds on directional recovery are positive in 6 of 7 families ($0.319$–$0.411$); **BLOOM is $-0.0012$**. And full-law reconstruction bounds stay non-positive under every calibrated map tried — directions transfer, the complete law does not.

### The spectrum, and the ablation that splits the mechanism

Output-Fisher eigenvalues fall as $\lambda_i \sim i^{-\beta}$ with $\beta \approx 1$ over ranks 2–80, from 70M to 6.9B, across families and vocabularies. The fitted exponent tracks the model's own weighted token profile: median deviation $0.04$, $r = 0.91$.

Held-out prediction on a text-disjoint 64-context battery in nine external-family models (BLOOM, GPT-Neo, Mamba, Qwen2, RWKV), predictor fixed before exact spectra were revealed:

| family | spectral error (decades) | $\vert\Delta\beta\vert$ | $N_{\mathrm{eff}}$ RMSE | vs flat $\lambda$ | vs shuffled $q$ | vs flat-$N_{\mathrm{eff}}$ |
|---|---|---|---|---|---|---|
| BLOOM | 0.0617 | 0.0856 | 0.0270 | 6.21× | 19.4× | 7.83× |
| GPT-Neo | 0.0771 | 0.0623 | 0.0312 | 4.89× | 13.5× | 7.88× |
| Mamba | 0.0852 | 0.0734 | 0.0460 | 5.42× | 15.0× | 6.44× |
| Qwen2 | 0.0884 | 0.0782 | 0.0360 | 6.32× | 17.1× | 7.85× |
| RWKV | 0.0833 | 0.0758 | 0.0521 | 9.22× | 25.6× | 6.53× |

Exponent correlation 0.968 Pearson, 0.955 Spearman. Exact inheritance certificates hold in 98.4–100% of cells; zero violations anywhere.

**The clean mechanism split.** 14 models, 7 families, 192 fresh contexts, common 27,450-outcome representation. Compare the weighted profile $q_a = p_a\|u_a - \bar u\|^2$ against the probability-only profile $(r/V)p_{(k)}$:

- **Effective dimension**: probability-only wins in **all 7 families** (mean log-ratio $+0.511$, exact one-sided $P = 1/128$).
- **Spectral exponent**: weighted wins in **all 7 families** (mean log-ratio $-0.379$, same $P$).

So probability concentration sets *how many* modes are resolved; the learned read-out directions set *how fast their strengths decay*. Two separable contributions, each cleanly attributed.

**What the spectral story does not support.** A single global power law is preferred by BIC in under 7% of model–context pairs; a crossover is detected in *every* pair. All power-law claims are scoped to the rank-2–80 core window with median tail exponent 1.73 beyond it. Frame constants are far from ideal: median $A_K \approx 14$ (range 5–33) at $K \approx 64$, and every curve has $\ge 9$ Gram eigenvalues below $1/2$, so $a_K < 1/2$ throughout. The worst-case inheritance band is therefore loose on real read-outs — the spectrum tracks the profile tightly anyway, because the tail's collective modes lie along the same cluster directions as the top block's, where Weyl's additive worst case is nowhere near attained.

A factorial attribution nails the tail steepening: it survives a random read-out with the real profile ($+0.47$) and **vanishes** for a pure power-law profile with random read-out ($-0.07$). Not a finite-width random-matrix edge. Real read-out contributes a secondary $+0.27$.

**Estimator honesty.** Against exactly computed held-out spectra, nominal 95% interval coverage is $0.90$ for continuous effective dimensions and only $0.85$ for discrete threshold counts. Primary claims therefore use the continuous form.

### Cluster anatomy, and the fresh-context failure

Weighted $K$-means on the frame rows achieves mean capture 0.718 and worst-direction capture 0.252. Of 128 cells, 26 contain fewer than 8 tokens, including 6 singletons; those cells hold probability mass 0.3315 and every one contains a token in the top 158 by reference probability. So the high-probability head needs near-singleton resolution while the remaining mass forms interchangeable clusters, with the exact bound $|\langle d, y\rangle| \le \|d\|\sqrt{1 - q_{\min}(L)}\|y\|$ for within-cell redistributions $d$.

**This does not generalise to a fixed partition.** On fresh contexts, the proposed within-cell attenuation failed in **all six model-halves**, with within-to-cross median-response ratios of $1.22$–$3.11$. The anatomy is within-battery; a universal partition is not established.

### Acquisition timing

Acquisition time on a log scale, $\tau = \log_2 t$, where $t$ is the first step after which a fact's margin stays above threshold.

Corpus $n$-gram margins at unigram/bigram/trigram level, measured **before training**, applied without refitting to fresh facts with zero prefix overlap:

| size | trajectory $R^2$ | status agreement | timing error ($\log_2$ steps) | constant baseline | permutation baseline |
|---|---|---|---|---|---|
| 70M | 0.790 | 0.792 | 0.77 | 1.04 | 2.02 |
| 160M | 0.792 | 0.854 | 0.95 | 2.20 | 2.56 |
| 410M | 0.775 | 0.807 | 0.96 | 0.98 | 2.34 |

The selected model hands weight from coarse to fine statistics during training; at final checkpoints the trigram-to-unigram weight ratio is $2.7$–$4.9$.

**The randomised depth experiment.** Otherwise identical facts assigned by a cyclic $5\times5$ Latin square to five evidence-depth rungs, in paired arms with **byte-identical initial weights and batch streams** (12 seeds, 45 quintets, 225 facts, 40,000 steps). No statistic below the assigned depth distinguishes the answers; the deciding margin appears at and above it.

Over the 10th–20th percentile band, deepest vs shallowest shift $\Delta = 2.103$ in $\log_2$ steps (interval $1.79$–$2.40$) — **4.3× more training steps**. The shift varies by only $0.167$ across the analysed quantiles. Deepest vs intermediate: $1.365$ ($\approx 2.6\times$). Stable under leave-one-seed-out and an alternative RNG stream.

**Crucially, what depth does *not* delay:** the onset of *unsigned* movement along the target–alternative margin direction is unchanged. Depth delays persistent signed commitment, not the start of motion.

Fully crossed generalisation: GPT-2-, GPT-NeoX- and Llama-style decoders × copy-back-reference and four-cue-parity languages, crossover design with initial parameters, minibatch streams, exposure counts and shallow:deep mixture all held fixed (8 seeds, 384 matched blocks per cell, 96 runs). Delays $0.810$–$1.514$ $\log_2$ steps ($1.75$–$2.86$×); all 6 intervals and all 48 seed means positive; **all 12 control intervals (fixed-treatment and label-randomisation) contain zero**. Construction modulates magnitude: parity exceeds back-reference by $0.414$, though the within-architecture contrast for GPT-2 is $0.189$ with interval $[-0.015, 0.384]$ — not resolved.

Cumulative curves across 70M/160M/410M align better by held-out loss than by training step: median absolute error **2.1 facts out of 300**, about 3× lower than step-alignment. This is descriptive, not causal.

**The impossibility result.** In the positive-rate model $T_j = W_j/\kappa_j$, there exist two rate profiles with identical evidence depths and opposite acquisition orders. Nominal depth alone cannot order acquisition time without an assumption on learner-relative effective rates. Randomised assignment is what does the identification work.

### Smooth motion, abrupt benchmarks

For argmax decisions, if $y$ is the unique maximiser of $p$ with margin $m = p_y - \max_{j\ne y}p_j$, a flip requires $\rho(p,q) \ge m/2$. Zero violations across 45,315 observed cross-seed flips. On held-out LAMBADA, flip frequency falls from $0.519$ to exactly zero as $z = m/(2\rho)$ crosses 1 — where zero is *forced* by the theorem. A smooth geometric trajectory produces an apparently abrupt benchmark transition.

**The gap in the controllability story.** Objective sensitivity alone cannot determine learning rate. With $A = I$, $q = (1,0)$ and unit training covectors $(1,0)$ vs $(0,1)$, sensitivity and squared update norm are identical but the objective rate is 1 or 0. The exact factorisation is

$$\text{objective rate}^2 = \text{sensitivity} \times \text{update norm}^2 \times \text{alignment}^2$$

On 80 held-out Pythia-70M examples, mean alignment$^2 = 0.681$ and the log-sensitivity/log-rate correlation is only $0.541$. **The missing quantity is objective–update alignment.**

Relatedly: a read-out update confined to the current column span — including scalar decoupled weight decay, as in [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] — produces *exactly zero* subspace rotation. Only the normal component $B = (I - P)\dot{Q}$ rotates.

### Control: predicted in advance

**11 models, six families, three objectives, three relative depths, 12 contexts per objective = 1,188 cells, 3 magnitudes each = 3,564 measurements, 3,515 usable (98.6%).** All from predictor state fixed before any response.

- Median measured-to-predicted ratio **0.967** (interval $0.957$–$0.975$)
- Log–log slope **0.982** (interval $0.966$–$0.997$)

**Relinearised finite paths.** Recompute both Fisher and Euclidean directions after every accepted step, trust region 0.02 nats, common cumulative objective checkpoints at 0.05/0.1/0.2/0.4 nats, 216 prompts, 16 models, 18 cells. Cumulative local KL ratios (Euclidean/Fisher) span **11.5× to 127.1×**, all 72 paired intervals above 1.

**Reusable control.** Learn one shared activation update from 4 donor prompts per objective, using the Fisher metric averaged over 4 *reference* prompts, then apply it unchanged to 8 unseen targets with no target gradients. Across BLOOM-560M, Pythia-410M, Pythia-1.4B, Qwen2-1.5B:

- All 24 model–objective–target settings have positive mean target effect (smallest $+0.0128$ nats).
- Reference-sequence KL is **2.90–6.29×** lower than Euclidean at matched donor change.
- At least **2.13×** lower than a *donor-only* Fisher metric — so averaging over the reference set, not just using any Fisher metric, is what buys preservation.

Instruction-tuned frontiers (Qwen2-1.5B-Instruct, replicated on Qwen1.5-1.8B-Chat), at equal aggregate mean behaviour change: off-target KL reduced **8–74×** for anti-sycophancy, **30–190×** for capital-city truth, **4–550×** for style.

**What did not resolve:** the 20-item multiple-choice capability check moves in increments of $1/20 = 0.05$ and showed **no Fisher–Euclidean difference at all**, while off-target KL on the same models differed by 36–50×. A coarse discrete benchmark cannot see a distributional effect this size. Contrastive activation addition often does not reach the target change for these objectives.

### The one correction across five operations

| operation | construction | headline | base |
|---|---|---|---|
| steering | $(G+\alpha I)^{-1}q$ per prompt | prediction tracks reality, median 0.967, slope 0.982; up to 72.7× lower off-target KL, 92–100% prompt wins | 3,515 obs / 1,188 cells |
| editing | fixed Fisher edit, shared gate | Euclidean/Fisher other-token KL ratio **10.57×** ($8.52$–$13.37$); $+5$ log-odds on 30/30 | GPT-2, 30 held-out CounterFact |
| feature importance | $\tfrac12\Delta h^\top G\Delta h$ | $\rho = 0.997$ with exact ablation KL, vs **0.896** for activation magnitude | Pythia-410M, 303 SAE ablations |
| attribution | objective-effect/Fisher-cost | selected-token change fraction $\rho = 0.935$ vs **0.506** for attribution patching; sign agreement $0.746$ vs $0.354$ | same 303 |
| dictionary learning | code-Gram natural gradient on decoder | matched cosine $0.167 \to 0.480$, 3/3 seeds | planted features, $\beta = 0.5$ |
| fine-tuning | exact natural-gradient LoRA | AdamW+KL pays **1.57×** the preservation-KL frontier AUC ($1.14$–$1.98$); 60–250× lower off-target at 410M | rank-4 70M, 8 seeds, 16 prompts |

Attribution patching is *not* uniformly worse — it remains the better predictor of raw absolute effect magnitude ($0.987$ vs $0.525$). The Fisher ratio wins on **selectivity**, which is the different question.

Dictionary learning also needed practical care: residual activations are heavy-tailed (max $\approx 15\times$ median), and per-token norms are winsorised at $\approx$ the 95th percentile. Alongside recovery, reconstruction variance explained rose $0.535 \to 0.622$ and the fraction of features ever activated rose $21.8\% \to 62.3\%$ — recovery alone would not have caught dead features.

### Where the advantage comes from, and where it goes

**Layer.** Off-target advantage geometric means fall $11.79 \to 11.08 \to 6.62$ at quarter, half and three-quarter depth. Every model mean is lower at three-quarter than at quarter depth; 9 of 11 paths decrease monotonically (Mamba-130M and Qwen2-1.5B rise at mid-depth). The metric pulled back through more of the network is more anisotropic; near the near-linear read-out there is less to exploit.

**Magnitude.** Measured-to-predicted ratio at intervention fractions $\Delta C/C = 0.01, 0.03, 0.10$: **$0.918$, $0.860$, $0.755$**. All 11 model paths decrease strictly. This is a local prediction attenuating as you push harder, quantified rather than waved away. The cubic correction is

$$\mathrm{KL}(p_0\|p_t) = \tfrac12 t^2 a^\top C a + \tfrac16 t^3\{T(a,a,a) + 3a^\top C b\} + O(t^4)$$

with $T$ the Amari–Chentsov contraction and the second term the network Hessian above the intervention layer — zero at an affine read-out, non-zero internally. **The Amari–Chentsov term alone is not enough:** including the network-Hessian term cuts mean absolute error against finite differences from $0.345$ to $0.0176$ (ratio 0.051), and affine anchors correctly return the map-Hessian term as exactly zero.

**Erosion prediction is regime-limited and the paper says so.** In the moderate-intervention regime (matched targets on monotone scan segments), the first cubic component rank-correlates with gentle-to-aggressive erosion at $+0.49$ to $+0.75$ across 410M–2.8B. **The association reverses sign beyond that regime**, and on the pooled reach-extended population.

### Geometry of concepts, including a clean dissociation

Cyclic registries (weekdays, months) trace near-rotations in the Fourier-1 phase representation: fitted weekday angle $0.902$ against ideal $2\pi/7 = 0.898$. Analogy relations (capital-of, gender, tense, comparative) do something structurally different — they move mass between *disjoint* token sets.

The double dissociation across 12 concept sets:

| | probability-displacement alignment | permuted control | isometry defect |
|---|---|---|---|
| relations | 0.49 | 0.33 | 1.02 |
| cyclic | $\approx 0.00$ | — | 0.29 |

Ordinal concepts fall between. The pullback metric characterises *reweighting*; movement between disjoint token sets additionally needs a ground metric on tokens, which this construction does not supply.

## Worth Remembering

**Limitations the author is explicit about.**

- Full GPU-resident pullback solves fit only **through 2.8B parameters**. At 6.9B the evidence narrows to graph-free spectra, a CPU-only mid-layer pullback calibration, and a descriptive CPU-offload iteration analysis.
- The **fine-tuning result is restricted to local preservation**. Task transfer and capability retention were measured separately and "did not support a broader conclusion". No safety-specific retention claim.
- The **GPT-2-XL editing benchmark is descriptive, not a method ranking** — the Fisher system computes a query-time gated edit, the ROME/MEMIT/GRACE/SERAC comparators are persisted editors under different access conditions. The controlled comparison is the 30-edit matched fixed-delta one on GPT-2.
- **Direct item-level cost↔meaning association fails.** Under Freedman–Lane residual permutation with Holm correction, Fisher-cost↔sharing and sensitivity↔sharing survive (adjusted $p = 0.001$ each), but cost↔meaning at matched training effect gives observed $-0.016$, adjusted $p = 1.0$, and sharing↔meaning is marginal ($p = 0.074$). The meaning link is supported at the *band* level, not item by item.
- **Native cross-tokenizer agreement remains empirical.** The quantitative convergence bound needs a shared outcome space; the first-byte coarse-graining supplies one, but the token-level cross-tokenizer number is not covered by the theorem.
- Equal-story human results are conditional on the **five observed narratives**, not a population-of-stories claim, and the coarse 65-bin outcomes do not establish individual-word recovery.

**Surprising results.**

The clean separation of probability concentration (sets $N_{\mathrm{eff}}$) from read-out structure (sets spectral decay) is the sort of mechanism split most papers assert and do not test. Seven families, unanimous in both directions, exact sign test.

The **92% number** in the eigenvector-sharing analysis should recalibrate how you read every "models converge" result. Raw top-mode overlap of 0.698 collapses to a genuine $+0.063$ once you match displacement magnitudes. The displacement-magnitude-matched null is the methodological contribution here, independent of the metric.

Model-only calibration improving predictions of *human* word choices in 6 of 7 families, while human-trained calibration made things worse in all 7, is genuinely odd and worth sitting with.

A width-independent CG iteration bound at fixed relative damping — 85 iterations at $c = 10^{-2}$, tolerance $10^{-6}$, any model size — is the fact that makes the whole framework practical.

**Connections.** The DPO relation is precise and worth keeping straight: [[DPO]] parameterises an implicit reward through $\beta\log[\pi/\pi_{\mathrm{ref}}]$ and updates *parameters*; this updates *activations* preconditioned by the pullback metric. Different objects. The KL leash in [[RLHF]] and [[PPO]] answers "stay near the old model in parameter or output space"; the reference-metric construction here instead limits change on a *declared set of protected prompts* — a behaviour-level trust region. That parameter-space version is proposed, not built.

The identification result is the honest version of what [[Understanding Contrastive Learning through Alignment and Uniformity|representation-similarity]] work has been groping at: convergence in output geometry *follows* from predictive fit, while activation convergence is an empirical property of training with an explicit counterexample showing no floor exists.

For [[Off-Policy Evaluation]] and [[Contextual Bandit]] work: the reusable-control result is the transferable piece. A minimal-disturbance update, learned on a donor set and validated on a protected reference set, is structurally the same move as constraining a policy update to stay close to logged behaviour on a held-out slate — the machinery here just supplies the right local metric for "close".

**Practical caveats for anyone using this.**

1. **Float32 minimum.** $\kappa \sim 10^5$–$10^7$; bf16 directions align at cosine as low as 0.056. This is not a precision nicety, it is a correctness requirement.
2. **Transport your reference metric.** Resetting $R$ to a fresh identity after a coordinate change breaks covariance (median relative error 1.23). If you reparameterise, carry $R$ with you.
3. **Damping kills the advantage.** $\Gamma_{A_\alpha}(q)$ is monotonically non-increasing in $\alpha$ and tends to 1. Large $\alpha$ buys stability and spends the whole benefit.
4. **Inject early-to-mid.** The advantage is $\approx 12\times$ at quarter depth and $\approx 6.6\times$ at three-quarter depth. Near the read-out there is little anisotropy to exploit.
5. **$R_{\mathrm{pred}}$ is local.** Expect $\approx 0.92$, $0.86$, $0.75$ of the predicted advantage at intervention fractions $0.01$, $0.03$, $0.10$.
6. **Do not validate with coarse discrete benchmarks.** A 20-item multiple-choice check saw zero difference where off-target KL differed 36–50×. Your measurement resolution can be worse than your effect.
7. **The reference-set choice is load-bearing.** Donor-only Fisher was $\ge 2.13\times$ worse than reference-averaged Fisher at preservation. Pick the prompts you actually care about protecting.

**Follow-up questions.** Does the parameter-space reference-metric trust region actually work for preference fine-tuning, or does the aggregation over prompts destroy the advantage the way damping does? Can the objective–update alignment term — the piece the factorisation identifies as missing — be measured cheaply enough to serve as a training diagnostic? Given that the residual eigenvector sharing is graded by lineage ($+0.088$ within family vs $+0.064$ across) and concentrated in leading modes, is the residual a usable fingerprint for model provenance? And does the displacement-magnitude-matched null, applied retroactively, survive the existing convergence literature?

## Links

Related: [[Attention]] · [[Cross Entropy]] · [[KL Divergence]] · [[Maximum Likelihood]] · [[Momentum]] · [[Derivative]] · [[Vector Jacobian Product]] · [[Backpropagation]] · [[Embeddings]] · [[Fine-Tuning]] · [[LoRA]] · [[QLoRA- Efficient Finetuning of Quantized LLMs]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[DPO]] · [[Direct Preference Optimization (DPO)]] · [[RLHF]] · [[PPO]] · [[Training language models to follow instructions with human feedback]] · [[Preference Learning]] · [[Reward Hacking]] · [[Distillation]] · [[Distilling the Knowledge in a Neural Network]] · [[Perplexity]] · [[Tokenization]] · [[Neural Machine Translation of Rare Words with Subword Units]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Tensor Programs V- Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (muP)]] · [[Adam- A Method for Stochastic Optimization]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Old Optimizer, New Norm- An Anthology (Muon)]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Representation Degeneration Problem in Training NLMs]] · [[How Contextual are Contextualized Word Representations]] · [[Whitening Sentence Representations]] · [[Sentence-BERT]] · [[The Linear Representation Hypothesis Needs a Group Action]] · [[Efficient Estimation of Word Representations (word2vec)]] · [[Sparse Readout Prism- Explaining Logit-Lens Scores in Features Instead of Tokens]] · [[A Unified Approach to Interpreting Model Predictions (SHAP)]] · [[Saliency]] · [[Mixed Precision training]] · [[Quantization]] · [[Off-Policy Evaluation]] · [[Contextual Bandit]] · [[Counterfactual Reasoning and Learning Systems]] · [[The Bitter Lesson (essay)]] · [[Troubling Trends in Machine Learning Scholarship]] · [[Shortcut Learning in Deep Neural Networks]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[Understanding Deep Learning Requires Rethinking Generalization]]

New topics worth writing: Fisher–Rao metric and information geometry, Chentsov's theorem, natural gradient descent, Amari–Chentsov tensor, pullback metric, effective dimension and damped resolution, activation steering, sparse autoencoders for interpretability, attribution patching, knowledge editing (ROME/MEMIT/CounterFact), Platonic Representation Hypothesis, representational similarity analysis, centred kernel alignment and its dimension-dependent null, relative representations, Woodbury identity, conjugate gradients and Chebyshev convergence bounds, K-FAC, Grassmann manifolds and chordal subspace distance, Schur–Horn majorization, Hellinger distance and Bhattacharyya affinity, Zipf's law in language statistics, cloze completion norms, displacement-magnitude-matched nulls, Freedman–Lane permutation testing, Latin-square randomisation designs
