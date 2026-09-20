---
title: "JEPA-Anything: Learning Predictive Models across Different Worlds"
authors: ["Cui et al."]
year: 2026
arxiv: "2609.20800"
url: https://arxiv.org/abs/2609.20800
priority: Must-Read
read_on: 2026-09-19
tags: [paper, self-supervised, vision]
---
## The Core Idea

A JEPA (joint-embedding predictive architecture) learns by predicting *representations*, not pixels. One encoder sees part of the world (the context), a second, slowly-updated encoder sees the target, and a predictor maps context → target inside the embedding space. This paper's complaint: that target is **one vector, predicted by one head**. When a world state mixes several kinds of structure — where a thing is, what it is, how fast it moves, which patient event fires next — they all get crammed into a single embedding trained by a single squared-error term. The loud, easy-to-predict, high-variance directions win the gradient tug-of-war. Weaker predictive structure gets squeezed out or duplicated across redundant latent directions.

The fix is **orthogonal predictive factorization (OPF)**. Split the $d$-dimensional target into $K$ learned subspaces, each of width $r$ with $Kr = d$. Give each subspace its own predictor head. Force the subspaces to be orthogonal to each other — no overlap — and force each one to stay *active* (its coordinates must keep varying across a batch). Then glue the $K$ predicted pieces back into one complete $d$-dimensional latent state, which you can decode, plan with, or feed back in for the next rollout step.

> [!NOTE] Orthogonal Predictive Factorization
> Instead of one predictor hitting one target embedding, learn $K$ projection matrices $P_k \in \mathbb{R}^{d\times r}$ that carve the target space into non-overlapping blocks, predict each block with its own head, then re-assemble. Predictive capacity is *allocated* rather than left to compete. ^opf

The important thing to notice: this is **not disentanglement**. Nobody says "factor 1 = position, factor 2 = colour". The paper explicitly cites the impossibility result that unsupervised semantic disentanglement is not identifiable from data alone. Factor identities here emerge only from *what is predictable*, and one factor may hold several related variables. The orthogonality buys a **geometric** property, not a semantic one: the assembly map is well conditioned, so prediction errors do not get amplified when you re-synthesise the state. That matters enormously for rollouts, where errors compound.

Why it did not exist before: JEPA lineage work ([[Self-Supervised Learning from Images with I-JEPA]], V-JEPA) was aimed at representation quality for a downstream probe, where a monolithic embedding is fine. The moment you start *reusing the prediction as the next input*, the geometry of how you reassemble the state becomes a first-class concern. The redundancy-reduction family ([[Barlow Twins]], [[VICReg]]) regularises the statistics of a *single* embedding; OPF instead puts orthogonality on the *basis* that reads the target, and keeps an explicit synthesis map around.

What it unlocks, in the authors' framing, is one core that they bolt onto seven very different domains — vision, single-cell biology, clinical records, control, molecular dynamics, PDEs, weather — by swapping only the tokenizer and the context/target sampler.

## The Methodology

**The interface.** A raw observation $x$ from domain $\delta$ goes through an adapter $\mathcal{A}_\delta$ that makes content tokens $H = \{h_i\}$ and structural descriptors $S = \{s_i\}$ (a patch coordinate, a timestamp, a graph position, an entity id, or nothing). A view sampler $\mathcal{V}_\delta$ picks context indices $C$ and target indices $T$:

$$x \xrightarrow{\mathcal{A}_\delta} (H, S) \xrightarrow{\mathcal{V}_\delta} (H_C, S_C, T, S_T)$$

Everything after this line is identical across all seven domains. That is the whole claim of the word "Anything".

**Encoders.** Online encoder gives $z_c = f_\theta(H_C, S_C)$. Target encoder gives $z_t = f_{\bar\theta}(H,S)_t$ for each $t \in T$, with parameters updated by exponential moving average and no gradient:

$$\bar\theta \leftarrow m\bar\theta + (1-m)\theta$$

Standard BYOL-style asymmetry — see [[Bootstrap Your Own Latent (BYOL)]] and [[Momentum Contrast (MoCo)|MoCo]]'s [[Momentum Contrast (MoCo)#^momentum-encoder|momentum encoder]].

**The factorization.** Stop-gradient the target, then analyse it with $K$ trainable projectors:

$$\widetilde z_t = \operatorname{sg}(z_t), \qquad z_t^{(k)} = P_k^\top \widetilde z_t$$

The projectors are trainable and receive gradients from the predictive, orthogonality and activity terms — so the *subspaces themselves* are learned, shaped by what turns out to be predictable.

Each factor has its own head $q_k$ reading the shared context plus the target descriptor:

$$\widehat z_t^{(k)} = q_k(z_c, s_t)$$

Concatenate into $\widehat u_t \in \mathbb{R}^d$, then re-assemble with the Moore–Penrose pseudoinverse of the analysis map:

$$\widehat z_t = (P^\top)^\dagger \widehat u_t, \qquad P = [P_1, \ldots, P_K]$$

When $P$ is exactly orthogonal, $(P^\top)^\dagger = P$ and this collapses to the simple sum $\widehat z_t = \sum_k P_k \widehat z_t^{(k)}$.

**The transition.** For anything with time, actions or interventions, let $\xi_t$ be the exogenous input (action, intervention label, known forcing). The adapter shoves it into the context tokens or the descriptor, and:

$$\widehat u_{t+1} = [q_1(z_t, \xi_t, s_{t+1}); \ldots; q_K(z_t, \xi_t, s_{t+1})], \qquad \widehat z_{t+1} = (P^\top)^\dagger \widehat u_{t+1}$$

Apply repeatedly for a latent rollout. A planner scores the trajectory with a domain reward. Compare [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)|Dreamer]]'s latent imagination and [[Mastering Diverse Domains through World Models (DreamerV3)|DreamerV3]].

**The four loss terms.** Prediction is plain per-factor L2 — deliberately *not* cosine, because state synthesis needs the magnitudes, not just directions:

$$\mathcal{L}_{\mathrm{pred}} = \frac{1}{K|T|r}\sum_{t\in T}\sum_{k=1}^{K}\left\|\widehat z_t^{(k)} - z_t^{(k)}\right\|_2^2$$

Orthogonality has two halves — orthonormal columns *within* a projector, and near-zero overlap *between* projectors:

$$\mathcal{L}_{\mathrm{orth}} = \sum_{k=1}^{K}\left\|P_k^\top P_k - I_r\right\|_F^2 + \sum_{1\le i<j\le K}\left\|P_i^\top P_j\right\|_F^2$$

Two hinge-style activity floors, one on the projected targets (shapes the projectors, since the target is stopped) and one on the online encoder (a direct anti-collapse gradient, lifted straight from [[VICReg]]):

$$\mathcal{L}_{\mathrm{fac}} = \frac{1}{Kr}\sum_{k,j}\max(0, \gamma_{\rm fac} - \sigma^{\rm fac}_{k,j}), \qquad \mathcal{L}_{\mathrm{enc}} = \frac{1}{d}\sum_{j}\max(0, \gamma_{\rm enc} - \sigma^{\rm enc}_{j})$$

with $\sigma = \sqrt{\operatorname{Var} + \epsilon}$. Total:

$$\mathcal{L}_{\mathrm{OPF}} = \mathcal{L}_{\mathrm{pred}} + \lambda_{\mathrm{orth}}\mathcal{L}_{\mathrm{orth}} + \lambda_{\mathrm{fac}}\mathcal{L}_{\mathrm{fac}} + \lambda_{\mathrm{enc}}\mathcal{L}_{\mathrm{enc}}$$

and it is **added** to whatever loss the original domain implementation already used: $\mathcal{L}^{(\delta)}_{\mathrm{train}} = \mathcal{L}^{(\delta)}_{\mathrm{base}} + \mathcal{L}_{\mathrm{OPF}}$.

**Hyperparameters that mattered.** Almost every dynamics experiment used $d=128$, $K=4$, $r=32$, with $\lambda_{\mathrm{orth}}=0.10$, $\lambda_{\mathrm{fac}}=0.05$, $\lambda_{\mathrm{enc}}=0.02$. Locomotion used a tiny $d=32$, $K=4$, $r=8$ and *flipped* the weights: $\lambda_{\mathrm{orth}}=0.02$, $\lambda_{\mathrm{fac}}=0.10$. CITRIS Pong used $d=160$, $K=5$, $r=32$. Vision used $K=4$ on DINOv3 ViT-S ($d=384$, $r=96$) and SigLIP2 Base ($d=768$, $r=192$). Clinical used GPT-2 small, $d=768$, $K=4$, $r=192$.

**Three usage modes.** *Representation mode*: keep the online encoder, throw away EMA copy, projectors and heads, attach a probe. *Operational mode*: keep everything, use Eq. 5 to make next states and roll. *Diagnostic mode*: keep the projectors only, compute $u_k(x) = P_k^\top z_\delta(x)$ on encoder states and stare at the coordinates.

**One equivariance detail worth stealing.** For molecular dynamics on an $O(3)$-equivariant backbone, OPF is applied to the 64-channel multiplicity axis of the $l=1$ sector. An $l=1$ irrep has 3 components, so the flat block is $64\times3=192$ scalars — the projection acts as $P_k \otimes I_3$, identically on all three components, which preserves equivariance.

## Ablation Studies and Experiments

The controls are honest: in every standard-JEPA comparison the adapter, encoder, sampler, optimisation budget, split, readout and base loss are held fixed. Standard JEPA adds a monolithic predictive term; JEPA-Anything adds $\mathcal{L}_{\mathrm{OPF}}$.

**The mechanism audit (CITRIS Pong, 5 paired seeds, 2500 steps) is the best table in the paper.** It compares OPF against a *capacity-matched unconstrained multi-head* — same number of heads, no orthogonality:

| Metric | Unconstrained multi-head | Orthogonal factorization |
|---|---|---|
| Cross-factor subspace overlap ↓ | $0.4550 \pm 0.0420$ | $(5.18 \pm 0.34)\times10^{-16}$ |
| $\sigma_{\min}(P)$ ↑ | $0.00513 \pm 0.00233$ | $0.999989$ |
| Condition number $\kappa_2(P)$ ↓ | $438.52 \pm 170.00$ | $1.00005$ |
| Transpose synthesis NMSE ↓ | $0.7886 \pm 0.0299$ | $(2.98 \pm 0.11)\times10^{-14}$ |

Read that first row. Without the constraint, the heads learn **45% overlapping** subspaces — they really do duplicate each other. The condition number of 438 means a small factor-prediction error can be amplified ~438× during re-synthesis. This is the component doing the work: it is the *orthogonality*, not the multi-head-ness.

**Intervention prediction (CITRIS Pong, 4-channel MSE, 5 paired seeds).** Single intervention $0.009541 \to 0.006218$ (**−34.8%**). Combined, held-out intervention $0.009441 \to 0.008223$ (−12.9%). Six-step free rollout $0.009478 \to 0.008665$ (−8.6%). The gain is *biggest* on the easy in-distribution case and shrinks as you demand compositional generalisation — worth noting, since the headline number is the easiest setting.

**Ten-task matched dynamics benchmark** (CausalWorld, DMC, PDEBench, WeatherBench2; seeds 11/23/37/53/71). All ten improve. Rollout table:

| Task | JEPA step 1 → 6 | Anything step 1 → 6 |
|---|---|---|
| CausalWorld state | 0.01306 → 0.01307 | 0.01037 → 0.01079 |
| DMC pixel | 0.008539 → 0.008594 | 0.008498 → 0.008526 |
| PDEBench Burgers | 0.001830 → 0.006369 | 0.001101 → 0.004014 |
| PDEBench shallow water | 0.007090 → 0.010510 | 0.003999 → 0.006522 |

DMC pixel is a rounding error (0.8% at step 1). The real wins are the PDEs, where the one-step error roughly halves.

**APEBench** (reported separately because normalisation differs): Burgers held-out late MSE $0.16621 \to 0.08389$ (−49.5%), six-step rollout $0.29186 \to 0.16153$ (−44.7%), Kuramoto–Sivashinsky $1.1012 \to 0.9558$ (−13.2%). Every seed improves in all three.

**The capacity-matched 50-step Burgers test is where the story gets less flattering.** 3 seeds, 2000 training steps:

| Split | Std H20/H50 | Anything H20/H50 | Change |
|---|---|---|---|
| ID | 1.1185 / 0.8181 | 1.0532 / 0.7923 | −5.84% / −3.15% |
| High-freq OOD | 1.0649 / 0.8452 | 1.0116 / 0.8214 | −5.01% / −2.82% |

Under a parameter-matched budget the 45% APEBench gain becomes 3–6%, and it **shrinks with horizon** — the opposite of what you'd want from a claim about rollout stability. The advantage narrows from H20 to H50 on both splits.

**What did not work.** Continuous-control planning with a CEM planner: Walker2d and HalfCheetah improve, **Hopper is worse** with OPF than with standard JEPA. The authors state this plainly — "planning performance is environment-dependent". Capacity is genuinely matched: parameter differences from dense JEPA are 0.000/0.006/0.126/0.246% for $K \in \{1,2,4,8\}$, and head FLOP differences 0.000/0.742/0.992/1.449%. So the Hopper regression is not a budget artefact.

Also unflattering: on **controlled visual binding**, standard JEPA barely beats a frozen checkpoint at all, and JEPA-Anything's margin over standard JEPA is in the third decimal place. DINOv3 INJ: frozen .569 → JEPA .572 → Anything .581. SigLIP2 INJ: .484 → .483 (JEPA *hurts*) → .490. Collapse rate .433 → .426 → .417. These are ten-seed means with no error bars reported. On this task the method is doing next to nothing.

And on molecules, **TrajCast-JEPA is worse than training from scratch** on water one-step MAE (0.00452 vs 0.00387) — plain JEPA pretraining actively hurt. OPF recovers it (0.00376). Full table, means over 5 seeds, Å:

| System | Metric | Scratch | TrajCast-JEPA | Anything |
|---|---|---|---|---|
| Water | MAE / RMSD | 0.00387 / 3.331 | 0.00452 / 2.536 | 0.00376 / 2.459 |
| Quartz | MAE / RMSD | 0.01080 / 2.089 | 0.01043 / 1.912 | 0.01011 / 1.877 |
| Paracetamol | MAE / RMSD | 0.00901 / 3.155 | 0.00777 / 1.868 | 0.00705 / 1.776 |
| Benzene | MAE / RMSD | 2.59e−5 / 0.0958 | 2.10e−5 / 0.0701 | 2.05e−5 / 0.0645 |

RMSD is the median final-position error after 100 free autoregressive steps — the real stability measure. Anything wins all eight cells.

**Single cell** (scGPT backbone, ~800k kidney cells pretraining, 5 seeds). PBMC-10K fine-tuned AvgBIO: scGPT .7531, Cell-JEPA .7830, Anything .8301. Zero-shot AvgBIO: .5288 → .7194 → .7752 — note the enormous jump is from JEPA itself, not from OPF. Perturbation Pearson: Norman .631 → .787 → .814; Adamson .905 → .937 → .942.

**Factor-usage ablation (locomotion).** Replace one factor with its training-set mean throughout a 20-step rollout and a paired CEM eval. Masking *any* factor raises rollout MSE in all three environments. In HalfCheetah, masking F1 or F3 cuts return in 5/5 seeds; F2 or F4 in 4/5. Factor-wise standard deviations stay in a narrow band (0.692–0.715 Hopper, 0.587–0.601 Walker2d, 0.673–0.678 HalfCheetah) — nothing is dead. Across nine predictive tasks, an average of **3.782 of 4** factors are effective, dominant-factor share 0.324 (uniform would be 0.25). Single-factor ridge probes on Hopper read out foot angle ($R^2=0.294$), torso height (0.347), torso angle (0.424), leg angle (0.310) — differentiated preferences, not clean one-to-one labels.

## Worth Remembering

**The proposition is trivial but the corollary is the point.** If $P_i^\top P_j = 0$ and $P_k^\top P_k = I_r$, then $P$ is orthogonal, $\|P^\top z\|^2 = \|z\|^2$, and $z = \sum_k P_k P_k^\top z$ exactly. The corollary: with error $e$ in the concatenated factor predictions, $\|\widehat z - z\|_2 = \|e\|_2$ and $\kappa_2(P)=1$. Without cross-factor orthogonality, duplicate directions make $P$ rank-deficient and near-duplicates send $\kappa_2$ to infinity. Appendix A gives the soft version: if $\|P^\top P - I_d\|_2 \le \epsilon < 1$, then $\|\widehat z - z\|_2 \le \|e\|_2/\sqrt{1-\epsilon}$, with no reconstruction bias for exact coordinates. So the measured orthogonality residual *is* your error-amplification budget. That's a clean diagnostic you could log during training.

**The biology claim needs a cold eye.** Factor coordinates "nominated" IL-18 + NT5E/CD73 blockade, and wet-lab work in co-culture, 3 organoids, 3 tumour fragments and mice supported it. There is no negative control on the nomination procedure, no count of how many candidates were nominated and discarded, and no baseline method nominating alternatives. "Domain-specific intervention analysis applied to the orthogonal factor coordinates" is not described anywhere. Treat as an existence proof, not evidence the factorization caused it.

**The Kepler result is a sanity check, not a finding.** Fitted slope $-1.4991$ against the true $-3/2$ with $R^2 = 0.9999999$, from spectral analysis of latent modes on *simulated* position–velocity trajectories. One run. Any method that preserves orbital frequency information would do this; it shows the latent didn't destroy the signal.

**Limitations the authors own.** Orthogonal factors do not establish causal mechanisms. A shared architecture does not guarantee aligned representations across domains. Uncertainty calibration under shift is unsolved and they flag it as the blocker for using these models to *choose* experiments — relevant if you care about [[Uncertainty#^epistemic|epistemic uncertainty]] and exploration.

**Practical caveats if you wanted to use this.** (1) It is an additive loss on top of whatever you already train, with three coefficients — cheapest possible thing to bolt on, and the locomotion setting shows they need retuning by domain, not just copied. (2) $K$ and $r$ are constrained by $Kr = d$, so you are partitioning fixed capacity, not adding any. (3) Parameter overhead is genuinely negligible (<0.3% params, <2% head FLOPs at $K=8$). (4) Where it helps most is **iterated prediction** — PDEs, molecular rollouts. Where it barely registers is terminal readout on a strong pretrained vision encoder. (5) Every reported comparison against standard JEPA is a *mean*; error bars appear only in Tables 5, 7 and 8 and Figures 6–7. Several headline improvements are well inside plausible seed noise.

**Open questions.** Why does Hopper regress? Hopper is the environment most prone to catastrophic termination — a factorised state might be *smoothing* exactly the sharp transition the planner needs. Does the gain come from orthogonality or just from the extra anti-collapse pressure of $\mathcal{L}_{\mathrm{fac}}$ + $\mathcal{L}_{\mathrm{enc}}$? The multi-head audit isolates geometry but not the activity floors — an ablation with $\lambda_{\mathrm{orth}}=0$ and the floors on is missing. And nothing tests $K$ beyond the calibration note: no accuracy-vs-$K$ curve anywhere, despite "configurable predictive capacity" being a headline contribution.

**Connections.** The orthogonality term is structurally Barlow-Twins-shaped but applied to the projector Gram matrix rather than an embedding cross-correlation. The activity floors are VICReg's variance term, verbatim. The stop-gradient-plus-EMA is BYOL. The novelty is narrow and real: putting the constraint on the *analysis basis* and keeping an explicit synthesis map so that prediction and re-assembly are numerically safe.

## Links

Related: [[Self-Supervised Learning from Images with I-JEPA]] · [[VICReg]] · [[Barlow Twins]] · [[Bootstrap Your Own Latent (BYOL)]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]] · [[Mode Collapse]] · [[Exploring Simple Siamese Representation Learning (SimSiam)]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[A Tutorial on Energy-Based Learning]] · [[Counterfactual Reasoning and Learning Systems]] · [[Momentum Contrast (MoCo)]] · [[Fundamentals]] · [[Uncertainty]] · [[An Image is Worth 16x16 Words (ViT)]]

New topics worth writing: Moore–Penrose pseudoinverse and matrix conditioning, Disentangled representation learning and the Locatello impossibility result, CITRIS and causal factor identification from temporal interventions, Cross-entropy method (CEM) planning, Equivariant neural networks and irreps, PDE surrogate benchmarks (PDEBench / APEBench), Single-cell foundation models (scGPT), Autoregressive rollout error compounding
