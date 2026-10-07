---
title: "Towards Looped Models Done Right, Part II: Rethinking at Fixed Points"
authors: ["Benhao Huang", "Chufan Shi", "Junlin Chen", "Shicheng Wen", "Zhengzhong Liu", "Eric Xing", "Xuezhe Ma"]
year: 2026
arxiv: "2610.06833"
url: https://arxiv.org/abs/2610.06833
priority: Good-To-Read
read_on: 2026-10-06
tags: [paper, transformers, llm, rl, optimization, vision]
---
## The Core Idea

A **looped language model** reuses the same block of transformer layers many times instead of stacking many different layers. You pay for each extra loop four separate times: more activations to store while training, more key/value vectors to cache while decoding, more compute to read the prompt, and a second replay of the whole loop every time you score a rollout in [[Reinforcement Learning|RL]].

The claim here: **if the loop converges to a fixed point, you can throw away the path and keep only the endpoint — and that one fact collapses all four costs at once.**

> [!NOTE] Fixed point
> A state $z^\star$ that the loop no longer changes: $z^\star = F_\theta(z^\star; x)$. The equation mentions only the endpoint. Nothing in it refers to how you got there. ^fixed-point

That is the whole lever. Once states sit near $z^\star$, extra applications of $F_\theta$ barely move anything, so:

1. **Training** — backpropagate through only the last $b$ loops. The missing part of the gradient shrinks geometrically in $b$.
2. **Decoding** — each earlier token keeps only its *final* KV, and every loop of every later token attends to that one copy. KV cache shrinks $3\times$ at five loops.
3. **Prefill** — train a small non-recurrent student to jump straight to $z^\star$. Up to $1.79\times$ faster prefill.
4. **RL** — the rollout already produced $z^R$. Save it, and build the gradient from that one saved state instead of re-running the loop. $2\times$ faster update.

Two of these (1 and 2) are already in use — Huginn and Ouro do them. The contribution is explaining *why* they work, and that the explanation then hands you 3 and 4 for free.

The diagnostic that makes the whole argument credible is a known split in the literature. Reusing terminal KV across loops barely dents Huginn's zero-shot GSM8K, but it **collapses** Ouro's at four loops. Huginn's states converge; Ouro's do not. Same trick, opposite outcome, and convergence is the thing that differs.

So the rest of the paper is about **making the fixed points good**, because none of the four shortcuts is safe otherwise. Two training knobs shape them, and both existing choices are flawed:

- The **depth prior** — the distribution you sample the loop count $R$ from during training. Train at a single fixed depth and you get the best loss *at that depth* but KV sharing breaks catastrophically (GSM8K $50.6 \to 21.2$ at 1.6B). Huginn's fixed broad prior keeps sharing alive but spreads supervision thinner than sharing actually requires. Fix: **learn the prior** from how well each sampled depth predicts, with an entropy term to stop it narrowing.
- **Input injection** — re-adding the input embedding at every loop so the fixed point still depends on the prompt. In Huginn and Parcae, the carried-over state can have a component pointing along the injected input, which silently amplifies or cancels it depending on sign. Fix: **project that component out** (OrthoInj), so every loop receives the input at identical strength.

## The Methodology

### The model

Three parts, following Huginn:

- **Prelude** $e = P_\theta(x)$ — one non-shared block, embeds the input.
- **Recurrent core** — two shared blocks, iterated. Start from a random $z^0$, run $z^r = F_\theta(z^{r-1}; x)$ for $r = 1 \dots R$, injecting $e$ every time.
- **Coda** — one non-shared block, maps the final state to logits.

Written as $1{+}2{\times}R{+}1$. At $R=5$ that is a *logical* depth of 12 blocks built from 4 distinct ones.

The state is $z^r = (\mathbf{H}^r, C^r)$: hidden states plus the KV banks written on that loop (one per physical attention layer).

### Convergence is not uniform across tokens

Measured by relative change per token per loop:

$$\rho_{r,t} = \frac{\lVert h_t^r - h_t^{r-1}\rVert_2}{\lVert h_t^{r-1}\rVert_2 + \epsilon}, \quad \epsilon = 10^{-8}$$

A token is **converged from depth $r$** if $\rho_{r',t} < \tau$ (they use $\tau = 2\%$) for every $r' \geq r$ up to the observation horizon.

The finding that drives the design: convergence depth is **heterogeneous**. Tokens in one sequence settle at wildly different depths, not in causal order, and depth correlates only weakly with position (Spearman $0.14$). So the depth you happen to sample during training decides whether the supervised loops get to read a *settled* prefix — which is exactly what KV sharing hands them at inference.

### Shortcut 1 — truncated backprop (training)

Three gradients to keep straight. At a fixed point, let $J_\star = \partial_z F_\theta(z^\star;x)$ and $B_\star = \partial_\theta F_\theta(z^\star;x)$:

$$D_\star = (I - J_\star)^{-1} B_\star = \sum_{k=0}^{\infty} J_\star^k B_\star, \qquad D_b = \sum_{k=0}^{b-1} J_\star^k B_\star, \qquad \tilde{D}_b = \sum_{k=0}^{b-1}\Big(\prod_{j=1}^{k} J_{R-j}\Big) B_{R-1-k}$$

- $D_\star$ is the **implicit / equilibrium gradient** (what [[A Tutorial on Energy-Based Learning|DEQ]]-style models solve for). It needs a stable fixed point to exist.
- $D_b$ truncates its Neumann series. Bias is exactly $D_\star - D_b = J_\star^b D_\star$ — geometric decay in $b$.
- $\tilde{D}_b$ is **truncated BPTT (TBPTT)**: same shape, but the Jacobians are evaluated at the *visited* states $z^r$, not at $z^\star$.

That last distinction is the practical point. $\tilde{D}_b$ is the **exact** gradient of the loss through the last $b$ loops with the earlier state held fixed — so it is a well-defined descent direction from *any* state, converged or not. Implicit gradients applied from initialization destabilize training, which is why DEQ methods need a warm-up phase of unrolled backprop. TBPTT needs no warm-up and no switch: it is as stable as full BPTT early, and slides into $D_b$ once fixed points form.

> [!NOTE] Truncation alone shapes fixed points
> Not a side effect — a cause. At fixed $R=5$ with full BPTT, extrapolating to $R=64$ raises perplexity $>25\%$ and terminal KV sharing raises it $21\%$. With TBPTT at $b=2$: at most $2\%$ and $3\%$. Backpropagating through the whole trajectory lets the model exploit the path; truncating forces it to be path-indifferent. ^truncation-shapes-fp

### Shortcut 2 — terminal KV sharing (decoding)

Normally the KV cache grows linearly in $R$: 12 banks at $R=5$. Terminal sharing keeps 4 — one per *physical* attention layer — and every loop of every later token attends to that same terminal copy.

There is a genuine train/test gap here. Teacher-forced training runs all tokens in parallel, so loop $r$ of the current token reads prefix banks *from depth $r$*. Sharing instead hands it terminal banks from loop 1. The gap is small when prefix banks barely change across depth and the current token's update **contracts**, so an early-context error decays over later loops.

Lemma 1 is the clean limiting statement. With $f_\theta(h; C)$ the hidden-state part of the update, $C^r_{<t}$ the prefix context, and the prefix converging ($C^r_{<t} \to C^\star_{<t}$): if $f_\theta(\cdot; C)$ is $\kappa$-contractive with $\kappa < 1$ and $f_\theta(h; \cdot)$ is $\beta$-Lipschitz, then both

$$h^{r+1} = f_\theta(h^r; C^r_{<t}) \quad\text{and}\quad \tilde{h}^{r+1} = f_\theta(\tilde{h}^r; C^\star_{<t})$$

converge to the **same unique** $h^\star$, from any start. Joint iteration and frozen-prefix decoding agree in the limit. Any continuous readout then gives the same next-token distribution.

At finite depth the error is bounded (their eq. 12) by

$$\lVert \bar{h}^R - h^\star\rVert \leq \kappa^R \lVert \bar{h}^0 - h^\star\rVert + \frac{\beta(1-\kappa^R)}{1-\kappa}\eta$$

where $\eta$ is how far the frozen context sits from the true terminal context. Two terms: unfinished token refinement, and imperfect context.

Honest caveat they make themselves: prefixes that actually *pass* their strict convergence test need **~100+ loops** (median 152 for fixed-depth, 114 for PLN). At $R=5$ you are nowhere near exact convergence. So sharing in practice rests on the finite-depth bound plus the measured perplexity, not on the lemma's limit. Their empirical control makes this vivid — freeze a *converged* prefix and the endpoint gap is $\sim 4\times 10^{-7}$; freeze a depth-5 prefix and it is $0.27$ (fixed-depth) or $0.033$ (PLN). Five to six orders of magnitude.

### Shortcut 3 — distilled prefill

Because decoding only ever reads terminal KV, prefill only has to produce the *endpoint*. A student that had to reproduce every loop's KV (as prompt-KV predictors do) would be a much harder ask.

A lightweight $S_\psi$ replaces the iterated core during prefill: from $e = P_\theta(x)$ it predicts $\hat{z} = S_\psi(e) \approx z_\theta^R(x)$. Prefill then applies **one** teacher loop plus the coda to $\hat{z}$ and keeps each physical layer's KV bank. Decoding is untouched — the teacher runs at its normal depth with terminal sharing.

Loss: normalized hidden-state regression plus a next-token KL term.

$$\mathcal{L}_{\rm distill}(\psi) = \mathbb{E}_x\left[\frac{\lVert \hat{z} - z_\theta^R(x)\rVert_F^2}{N d\, \sigma_{\rm T}^2} + \frac{1}{N}\sum_{t=1}^{N}\mathrm{KL}(p_t \,\|\, \hat{p}_t)\right]$$

$N$ = prompt length, $d$ = hidden width, $\sigma_{\rm T}^2$ = a fixed teacher-state energy calibrated on training data, so term one is per-coordinate squared error on the teacher's own scale. Term two compares the teacher's depth-$R$ next-token distribution with the one you get by pushing $\hat{z}$ through the frozen coda and head.

The student is **initialized from the teacher's recurrent core**, so before any training it roughly outputs the teacher's state after one loop; the loss then drags that output to the depth-$R$ endpoint. Trained on $1/4$ of the teacher's pretraining tokens.

### Shortcut 4 — rollout-state reuse in RL

RL normally runs the recurrent forward pass twice: once without gradients to sample, once with gradients to score. The second pass is pure waste if you are near a fixed point.

Full BPTT computes $\tilde{D}_R$. When every visited state is close to $z^\star$, its Jacobians are close to $J_\star, B_\star$, so

$$\underbrace{\tilde{D}_R = \sum_{k=0}^{R-1}\Big(\prod_{j=1}^{k} J_{R-j}\Big)B_{R-1-k}}_{\text{full BPTT: all } R \text{ loops}} \;\approx\; \underbrace{\sum_{k=0}^{b-1} J_R^k B_R}_{\text{reuse: endpoint only}}$$

The update is on-policy, so the saved $z^R$ **equals** what a fresh forward pass would give. Mechanically (Algorithm 1):

1. $\bar{z} \leftarrow \mathrm{stopgrad}(z^R)$, marked requires-grad.
2. $f \leftarrow F_\theta(\bar{z}; x)$ — the **only** pass through the recurrent block. Record its autograd graph $\mathcal{G}$.
3. $\tilde{z} \leftarrow \bar{z} + (f - \mathrm{stopgrad}(f))$ — forward value is still $z^R$, but gradients now flow.
4. Policy-gradient loss on the coda readout of $\tilde{z}$; take $g = \partial\mathcal{L}/\partial\tilde{z}$.
5. Loop $b-1$ times: $v \leftarrow g + J_R^\top v$, each step a backward through the *same recorded* $\mathcal{G}$.
6. One final backward through $\mathcal{G}$ to $\theta$.

So the cost and memory of the update become **independent of $R$**. In code, steps 5–6 are a backward hook on $\tilde{z}$.

### The learned depth prior

Huginn samples depth from a shifted Poisson–log-normal:

$$\xi \sim \mathcal{N}(\mu, \sigma_\xi^2), \quad K \mid \xi \sim \mathrm{Poisson}(e^\xi), \quad R = K+1$$

with $\sigma_\xi = \tfrac12$ and $\mu = \log(\bar{R}-1) - \sigma_\xi^2/2$ so $\mathbb{E}[R] = \bar{R}$. Fixed, never adapts.

Replace it with a categorical $p_\phi = \mathrm{softmax}(\phi)$ over $\mathcal{R} = \{1,\dots,64\}$, initialized from the capped PLN. Each microbatch draws $r \sim p_\phi$ and is supervised at that depth. The prior minimizes three terms:

$$\mathcal{J}_{\rm prior}(\phi) = \underbrace{-\widehat{\mathbb{E}}_n[\mathrm{stopgrad}(A_n)\log p_\phi(r)]}_{\text{prediction quality}} - \underbrace{\lambda_H H(p_\phi)}_{\text{depth diversity}} + \underbrace{\lambda_m(\mathbb{E}_{p_\phi}[R] - \bar{R})^2}_{\text{budget control}}$$

- **Reward**: inverse perplexity, $u = \exp(-\ell)$ where $\ell$ is the detached mean next-token [[Cross Entropy|CE]].
- **Advantage**: $A_n = s_{n-1}(u - \hat{u}_{n-1})$. An EMA [[Policy Gradient#^baseline|baseline]] $\hat{u}$ centres the reward; $s_{n-1} = \sigma_{\ell}/\sigma_u$ rescales it to CE scale, following PopArt. The rescaling is what lets $\lambda_H$ and $\lambda_m$ stay fixed across different reward shapes.
- **Entropy**: without it, feedback-under-budget piles supervision onto depths that already predict well, and fixed-point shaping dies. Since depths start at 1, raising $\lambda_H$ pushes mass mostly *upward*, toward $R_{\max}$ — exactly where settled prefix context lives. So $\lambda_H$ is a direct performance-vs-memory knob.
- **Budget**: $\lambda_m = 1$, $\bar{R} = 5$.

This is a [[Simple Statistical Gradient-Following Algorithms (REINFORCE)|REINFORCE]]-style controller on a 64-way categorical, with its own Adam (LR $10^{-3}$) and a 32-update warm-up where it only collects its baseline. It adds **no extra forward pass** — the reward is the model CE you already computed. Backprop is capped at $b_{\max} = 10$ loops.

### Orthogonal input injection

Additive injection has the form

$$\zeta_t^r = M h_t^r + q_t, \qquad \mathbf{H}^{r+1} = \Gamma_\theta(\zeta_1^r, \dots, \zeta_N^r)$$

Parcae uses a diagonal carryover $M = \Lambda = \mathrm{diag}(\exp(-\Delta \odot \alpha))$ with learned positive $\Delta, \alpha$, so entries land in $(0,1)$ and $\lVert\Lambda\rVert_2 < 1$, and injects $q_t = \Delta \odot W e_t$.

The problem: $\Lambda h_t^r$ can have a component *along* $q_t$. Depending on sign it amplifies or cancels the injection, so how much input the fixed point retains depends on the state. **OrthoInj** removes that component:

$$\zeta_t^r = Q_{q_t}\Lambda h_t^r + q_t, \qquad Q_{q_t} = I - \frac{q_t q_t^\top}{\lVert q_t\rVert_2^2 + \epsilon}$$

With $\epsilon = 0$ and $q_t \neq 0$, the component of $\zeta_t^r$ along $q_t$ is **exactly** $q_t$ at every loop including the fixed point; the orthogonal part is free. Contractiveness survives: $\lVert Q_{q_t}\Lambda\rVert_2 \leq \lVert\Lambda\rVert_2 < 1$, since a projection has operator norm $\leq 1$. They use $\epsilon = 10^{-6}$ with FP32 inner products, and drop Parcae's prelude RMSNorm.

### Training setup

| Scale | Params | Tokens | Width | Updates |
|---|---|---|---|---|
| S | 100M | 21.5B | 1,536 | 5,120 |
| M | 400M | 85.9B | 3,072 | 20,480 |
| L | 1.6B | 343.6B | 6,144 | 81,920 |

Llama blocks, [[Grouped Query Attention|GQA]] with 4:1 query:KV head ratio, head dim 64, sequence length 8,192, global batch 512 sequences (4.19M tokens/update). [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] with $\beta = (0.90, 0.98)$, weight decay 0.10, [[On the difficulty of training Recurrent Neural Networks|gradient clipping]] 1.0, warmup–stable–decay schedule: 5% linear warmup, final 10% cosine to $0.1\times$ peak. Peak LR swept per scale and tuned per run. bfloat16. Jais64k tokenizer (64,256 vocab), untied input/output embeddings. Data is a weighted multilingual mix including TxT360.

Two baselines matter: **Untied 4** stacks 4 distinct blocks — matches the looped model's parameters *and* cache size. **Untied 12** stacks 12 — matches its logical depth, with $3\times$ the non-embedding parameters and KV cache.

## Ablation Studies and Experiments

Unless noted, everything is evaluated at $R=5$ **with terminal KV sharing** — 4 banks instead of 12.

### Depth priors (Table 2)

Validation PPL / downstream average (7 tasks) / GSM8K:

| Prior | S | M | L |
|---|---|---|---|
| Fixed $R{=}5$ | 5.95 / 39.45 / 1.36 | 4.21 / 48.32 / 3.34 | 3.24 / 56.90 / 21.23 |
| Fixed PLN-5 | 5.73 / 41.32 / 1.52 | 4.00 / 49.06 / 13.72 | 3.10 / 59.30 / 47.61 |
| Learned, $\lambda_H{=}0$ | 5.67 / 41.54 / 1.44 | 3.92 / 49.72 / 14.33 | 3.12 / 58.44 / 36.47 |
| Learned, $\lambda_H{=}0.01$ | 5.67 / 41.65 / 1.59 | 3.93 / 49.07 / 15.62 | **3.07 / 60.00 / 47.92** |
| Untied 4 | 6.57 / 39.45 / 1.97 | 4.68 / 44.08 / 2.35 | 3.66 / 51.08 / 7.20 |
| Untied 12 | 4.99 / 43.85 / 1.90 | 3.60 / 51.73 / 16.15 | 2.89 / 61.47 / 48.82 |

**Fixed-depth training breaks sharing, badly.** Compare Table 2 against Table 8 (same checkpoints, full cache). At L, fixed-depth GSM8K goes $50.64 \to 21.23$ when you force sharing. At M, $15.54 \to 12.89$ on a lower base — more than half its accuracy at both scales in relative terms. Every sampled-depth prior loses little or nothing. This is the paper's cleanest result: **concentrating supervision at one depth buys you accuracy there and nothing transferable.**

**The entropy term is load-bearing at scale.** At L, dropping $\lambda_H$ from $0.01$ to $0$ costs GSM8K $47.92 \to 36.47$ — 11.5 points. And the direction flips depending on whether you share: *without* sharing (Table 8), $\lambda_H = 0$ has marginally *lower* validation PPL at M and L. So the entropy term is not improving the model; it is buying tolerance to sharing, and you only see the purchase when you share.

**The learned prior at L matches fixed-depth training with $3\times$ less cache.** $60.00$ shared vs $59.90$ full-cache fixed-depth. That is the headline comparison.

**Against the untied baselines, the trend is the interesting part.** Lead over Untied 4 (same parameters, same cache) grows with scale: $+2.2 \to +5.0 \to +8.9$ downstream points. Gap to Untied 12 (3× parameters) narrows: PPL excess $13.6\% \to 8.9\% \to 6.3\%$, and at L only $1.5$ downstream points and $0.9$ GSM8K points behind. On GSM8K it beats Untied 4 by $40.7$ points.

**A nice sanity check**: at S, freezing the *learned* final distribution from update one and retraining lowers validation PPL a further $0.8\%$. So the controller finds a genuinely better depth distribution — it is not just the exploration noise helping.

### Input injection (Table 3)

| Injection | S PPL / AVG | M PPL / AVG | L PPL / AVG |
|---|---|---|---|
| DEQ-QKV | 6.30 / 39.33 | — | — |
| Huginn-Linear | 5.90 / 40.46 | 3.99 / 48.67 | — |
| Parcae-Decay | 5.75 / 41.28 | 3.96 / 49.03 | 3.09 / 59.22 |
| **OrthoInj** | **5.67 / 41.65** | **3.94 / 49.16** | **3.08 / 59.26** |

Lowest validation PPL and highest downstream average at every scale, and lowest *training* loss at every scale. But be honest about size: the L-scale margin is $0.01$ PPL and $0.04$ downstream points, with one seed. The PPL improvement over Parcae is quoted as $0.3$–$1.4\%$, and at L it is at the bottom of that range. The S-scale gap ($5.75 \to 5.67$) is where the effect is actually visible.

Also: the gains **decompose**. Dropping Parcae's prelude normalization helps, the projection helps, and they add. So OrthoInj as shipped is two changes, not one.

A separate stability note hiding in the appendix: **without** a core-exit RMSNorm, S-scale PLN-25 training hit a non-finite loss at all five learning rates tried. Traced to the injection interface. Huginn-Linear needs that exit norm to train at all at depth.

### RL with rollout-state reuse (Table 4)

L learned-prior model, Dr. GRPO at $R=6$, 16 samples per prompt. 400 GSM8K training questions / 272 MBPP+ tasks.

| Update method | GSM8K p@1 | MATH500 p@1 | Update time | MBPP+ p@1 | Update time |
|---|---|---|---|---|---|
| Before RL | 40.63 | 13.40 | — | 30.88 | — |
| Full BPTT | **63.20** | **18.27** | 2.77s | **40.88** | 1.03s |
| Recompute + Neumann-4 | 60.05 | 17.32 | 2.09s | 37.12 | 0.65s |
| Reuse + Neumann-4 | 61.65 | 17.82 | **1.39s** | 39.38 | **0.51s** |
| Reuse + Neumann-3 | 61.30 | 18.77 | 1.30s | 38.00 | 0.48s |

$1.99\times$ faster on GSM8K, $2.02\times$ on MBPP+ (scoring + backward only; rollout generation still dominates wall-clock).

**The control is what makes this readable.** "Recompute + Neumann-4" uses the *same* Neumann gradient but rebuilds $z^R$ by replaying all loops without gradients. Its gradient matches reuse's at whole-model cosine $0.997$–$0.999$. So the gap between those two rows is pure **run-to-run variation**: $1.6$ points on GSM8K, $0.5$ on MATH500, $2.3$ on MBPP+. Reuse trails full BPTT by no more than that variation anywhere. Properly measured noise floor — more papers should do this.

GSM8K gains transfer to MATH500, which RL never trained on: $+3.9$ to $+5.4$ points across all methods.

### Distilled prefill (Table 5)

Downstream average / 8K prefill latency:

| Prefill path | S | M | L |
|---|---|---|---|
| Teacher ($R{=}5$) | 41.65 / 195ms | 49.07 / 496ms | 60.00 / 1482ms |
| Three loops ($R{=}3$) | 41.37 / 153ms | 47.94 / 370ms | 58.39 / 1056ms |
| Two loops ($R{=}2$) | 40.14 / 131ms | 45.91 / 304ms | 54.96 / 849ms |
| One loop ($R{=}1$) | 38.05 / 111ms | 41.19 / 246ms | 47.98 / 642ms |
| **Distilled** | **40.63 / 126ms** | **46.77 / 299ms** | **55.53 / 830ms** |

Read it as a latency-matched comparison, which is the only fair way: distilled prefill is $1$–$4\%$ faster than teacher-at-$R{=}2$ and $0.5$–$0.9$ points better. Real but small.

**The gap to the full teacher grows with scale**: $1.0$ (S) → $2.3$ (M) → $4.5$ (L) points. That is the wrong direction, and the authors say so.

**Early stopping is a strong, boring baseline.** $R{=}3$ stays within $1.6$ points of the teacher but saves only $21$–$29\%$ of latency, vs $35$–$44\%$ for distillation. $R{=}1$ is $12$–$23\%$ faster than the student but loses $2.7$–$3.5\times$ as much accuracy. The useful operating range is narrow.

**What makes the student work:** teacher initialization, not the loss. It adds $0.7$ (S) and $1.1$ (M) downstream points over random init under the same objective, and lifts M MBPP+ from $2.50$ to $6.12$. The KL term's distinct contribution shows on generation specifically — it lifts a *randomly* initialized student's M MBPP+ from $0.25$ to $2.50$.

### What did not work, collected

- **Implicit/equilibrium gradients from initialization** destabilize training. This is why TBPTT, not DEQ-style implicit differentiation, is the right tool — it needs no warm-up phase and no estimator switch.
- **Fixed-depth training**, full stop, if you want any of the four shortcuts.
- **Learned prior without entropy** at L: GSM8K drops 11.5 points under sharing, while looking *slightly better* on unshared PPL. A trap for anyone evaluating without the cache setting they intend to ship.
- **Uniform depth priors** as controls: $U_{1:64}$ never gets pulled down to the target mean within budget — final prior mean $6.15$ against target $5$.
- **No core-exit RMSNorm** with Huginn-Linear injection: non-finite loss at every learning rate tried.
- **Prefill gains saturate early** — the whole prefill story is bounded by how little the later loops contribute.

### What the ablations say about mechanism

The depth prior is doing the real work, not OrthoInj. OrthoInj's margins are $0.01$–$0.08$ PPL and fractions of a downstream point. The prior's effect is tens of GSM8K points under sharing. If you only took one thing from this paper, take the prior.

And the prior's mechanism is specifically **exposure to converged context**. Their appendix formalizes it: with backprop window $b(R)$ and a prefix converged from depth $r_c$, the expected number of supervised loops reading converged prefix KV is

$$w_{r_c} = \sum_R p(R)\min\{b(R), (R - r_c)_+\}$$

Fixed $R=5$ gives $w_{r_c} = 0$ for any $r_c \geq 5$ — **zero** exposure. PLN-5 gives $w_5 = 1.10$ ($p(R>5) = 35.4\%$) and $w_{10} = 0.15$. Exposure concentrates on prefixes converging within ~10 loops, which is exactly what $b_{\max} = 10$ allows.

They then argue, correctly, that exposure is **necessary but not sufficient**. Their counterexample: $z^r = (z_a, (-1)^r z_b)$ has constant loss if the head reads only $z_a$, but any KV reading the second component alternates forever. Random-depth loss does not imply KV stationarity. Which is why Table 2 measures sharing directly instead of trusting the proxy.

## Worth Remembering

**Limitations the authors state.** Up to 1.6B parameters, **one seed per configuration**, ablations do not exhaust the design space. Given that several headline margins (OrthoInj at L; several downstream averages) are smaller than the $1.6$–$2.3$ point run-to-run variation they themselves measured in the RL section, treat single-scale wins as suggestive and the *scale trends* as the real evidence.

**The fixed points are not actually fixed points at $R=5$.** Their own convergence test needs ~100+ loops to pass. So the four shortcuts rest on the finite-depth bound $\kappa^R\lVert \bar h^0 - h^\star\rVert + \beta(1-\kappa^R)\eta/(1-\kappa)$ and measured perplexity, not on convergence. "Near a fixed point" is doing a lot of load-bearing work with a loose definition. The real mechanism is probably better described as *contractiveness* — $\kappa < 1$, so errors decay — than as convergence.

**The transferable methodological idea.** Measure your efficiency trick's noise floor by building a control that computes nearly the same gradient a different way (cosine $0.997$–$0.999$), then report the gap between control and method as variation. That is how you know whether $61.65$ vs $63.20$ means anything. Cheap, and almost nobody does it.

**The evaluation-protocol trap.** Tables 2 and 8 are the same checkpoints and disagree on which prior is best, because one shares KV and one does not. Your choice of prior is only decidable once you know your serving configuration. Same lesson as [[On Sampled Metrics for Item Recommendation (KDD)|sampled metrics]] — the measurement protocol changes the ranking of methods, not just the numbers.

**A connection worth drawing.** The rollout-state reuse trick — save the forward state, build the gradient from one recorded block application via repeated [[Vector Jacobian Product|VJP]]s — is structurally [[FlashAttention- Fast and Memory-Efficient Exact Attention|recomputation]] run in reverse. Instead of discarding activations and recomputing, you keep *one* state and re-use *one* graph $b-1$ times. If you build RL infrastructure, the `stopgrad` identity trick in step 3 (forward value unchanged, gradients enabled) is worth stealing independently of looped models.

**Practical caveats for use.**

- $\lambda_H$ is a direct performance-vs-memory dial. $0.01$ was right at L; not swept at L, and at M its effect on sharing is non-monotonic across tasks. You will need to sweep it for your own cache configuration.
- OrthoInj as published bundles two changes (drop prelude norm, add projection). If you adopt it, test both.
- Distilled prefill's gap to the teacher *grows* with model size. At 1.6B it is already $4.5$ points. Do not assume this extrapolates favourably.
- Terminal KV sharing is only safe on models trained with a broad or learned depth prior. Bolting it onto a fixed-depth looped model is the Ouro failure mode.

**Open questions.** Does the gap to Untied 12 keep narrowing past 1.6B, or does it plateau? Can the distilled student serve standalone, or as a [[Speculative Decoding|speculative decoding]] drafter? The authors raise a possible link between recurrent state refinement and [[Flow Matching|flow-based]] language modelling and leave it entirely open. And nothing here addresses whether the learned prior survives contact with a real serving stack where depth must be chosen per request.

## Links

Related: [[Attention Is All You Need]] · [[Grouped Query Attention]] · [[KV Cache]] · [[Prefill and Decode]] · [[Distillation]] · [[A Tutorial on Energy-Based Learning]] · [[Vector Jacobian Product]] · [[Backpropagation]] · [[On the difficulty of training Recurrent Neural Networks]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[Policy Gradient]] · [[GRPO]] · [[Test-Time Compute]] · [[Speculative Decoding]] · [[Long Short-Term Memory (Neural Computation)]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Layer Normalization]] · [[Mixture of Experts]] · [[Flow Matching]] · [[Cross Entropy]] · [[Perplexity]] · [[Gated Recurrent Transformers- Expressive Depth through Recurrent Modulation]] · [[Looping Beyond Twice- A Scalable Recipe for Looped Mixture-of-Experts]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]]

New topics worth writing: Deep Equilibrium Models, Implicit differentiation and the implicit function theorem, Neumann series for matrix inversion, Truncated backpropagation through time, Contraction mapping theorem, Universal Transformer, Terminal KV sharing, Depth prior learning, PopArt reward normalization, Dr. GRPO, Path independence in equilibrium models, Jacobian regularization for stability, Adaptive computation time and PonderNet, Warmup–stable–decay schedules
