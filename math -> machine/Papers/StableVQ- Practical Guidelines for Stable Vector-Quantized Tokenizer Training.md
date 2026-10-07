---
title: "StableVQ: Practical Guidelines for Stable Vector-Quantized Tokenizer Training"
authors: ["Tang et al."]
year: 2026
arxiv: "2609.26774"
url: https://arxiv.org/abs/2609.26774
priority: Good-To-Read
read_on: 2026-09-27
tags: [paper, vision]
---
## The Core Idea

A VQ tokenizer turns an image into a list of integers. An encoder makes a grid of continuous vectors, each vector is swapped for its nearest entry in a learned codebook, and a decoder rebuilds the image from those entries. The chronic bug is **codebook collapse**: most entries never get picked, so the vocabulary is much smaller than advertised.

> [!NOTE] Shared-projection codebook ^shared-projection
> Instead of each code being its own free parameter, every code is written as $\tilde{\mathbf e}_k = f_\theta(\mathbf e_k)$ where $f_\theta$ is one function shared by all codes. A gradient from any selected code now flows into $f_\theta$ and therefore nudges *every* code. SimVQ makes $f_\theta$ a single linear layer; FVQ makes it two ViT blocks.

Shared projections mostly fixed utilisation — SimVQ and FVQ both report 100%. But training is still brittle: it breaks if you change the codebook initialisation scale, it can sit at low utilisation for thousands of steps, or utilisation can suddenly crash mid-run.

The claim of this paper: that brittleness is not the fault of quantisation. It is because the encoder–decoder and the codebook are **entangled** — neither module can do its own job alone, so the system only works when the two happen to help each other out.

Three concrete symptoms, each with a named cause:

1. **Codes packed in a tiny region, tokens spread wide.** Few codes get used, so the codebook gets almost no targets. Meanwhile the encoder's outputs thrash around because the straight-through gradient and the commitment loss pull in different directions. Result: long low-utilisation phases, sometimes NaN.
2. **Codes spread wide, tokens packed in a small region.** Codes inside the token region get claimed fast, the losses go quiet, and every code outside that region is permanently unreachable by nearest-neighbour lookup. Dead codes forever.
3. **Both distributions already aligned, but the token cloud shifts.** If the codebook fails to track the shift for one moment, the now-wrong straight-through gradients *widen* the gap rather than close it — a positive feedback loop that collapses utilisation in a few steps.

What this unlocks: with the three fixes below, a **single linear projection** beats FVQ's two ViT blocks (rFID 1.22 vs 1.70 at 16k codes). You no longer need to hand-engineer the projector or hunt for a lucky initialisation.

## The Methodology

Start from standard VQ. Encoder gives $\mathbf z = E(\mathbf x)$, each spatial vector $\mathbf z_{ij}$ is replaced by its nearest code, and the loss is

$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta\underbrace{\|\mathbf z - \text{sg}[\hat{\mathbf z}]\|^2}_{\text{commitment}} + \underbrace{\|\text{sg}[\mathbf z] - \hat{\mathbf z}\|^2}_{\text{VQ}}$$

`sg` is stop-gradient. Commitment pulls encoder outputs toward their code; VQ pulls codes toward the encoder outputs. The reconstruction gradient reaches the encoder through the **straight-through estimator** (STE), which just pretends $\partial\mathcal L/\partial\mathbf z_{ij} \approx \partial\mathcal L/\partial\hat{\mathbf z}_{ij}$.

StableVQ adds three parameter-free pieces.

### 1 · Dynamic STE — fix the encoder's gradient

The STE lie is only harmless when the token is *close* to its code. When a token sits far from the code it got assigned, the gradient copied back is a bad estimate of the true reconstruction direction. It fights the commitment loss, pushes the two distributions further apart, and in the worst case blows up to NaN.

The fix weights each token by how good its match is, *relative to the best token that chose the same code*:

$$w_{ij} = \mathrm{sg}\!\left[\frac{d^{*}_{k^{*}_{ij}}}{\|\mathbf z_{ij} - f_\theta(\mathbf e_{k^{*}_{ij}})\|_2^2}\right] \in (0,1], \qquad d^{*}_k = \min_{(i',j')}\|\mathbf z_{i'j'} - f_\theta(\mathbf e_k)\|_2^2$$

So $d^*_k$ is the distance from code $k$ to the *closest* token in the batch. The best-matched token gets $w=1$ and full gradient; a token twice as far (squared) gets a quarter of the gradient. Then the STE is rewritten as

$$\hat{\mathbf z}_{ij} = w_{ij}\cdot\mathbf z_{ij} + \text{sg}[\hat{\mathbf z}_{ij} - w_{ij}\cdot\mathbf z_{ij}]$$

Forward value is unchanged; only the backward path is scaled. Nice property: when everything is well matched, all $w_{ij}\approx 1$ and this **reduces exactly to plain STE**. No threshold, no hyperparameter, self-deactivating.

Pilot check: freeze the codebook, train only encoder–decoder. Plain STE spikes the commitment loss and diverges to NaN. Dynamic STE stays flat.

### 2 · Region VQ Loss — fix the codebook's target

> [!NOTE] Asymmetry of the VQ loss ^vq-asymmetry
> Every *token* gets an explicit target (the commitment loss). Only the *selected* codes get one. With a shared projection, unselected codes do receive some gradient through $f_\theta$, but it is diffuse and undirected, and it dies away once a small subset of codes already covers the tokens well enough. So the codebook cannot guarantee its own utilisation — it relies on the encoder wobbling around to accidentally activate new codes.

Region VQ gives every dead code a real target by **propagating targets outward from active codes**.

Keep a FIFO queue of the last $W$ steps' selected indices. That gives:
- $\mathcal S_t$ — codes selected right now
- $\mathcal A_t$ — codes selected anywhere in the window
- $\mathcal N_t = [K]\setminus\mathcal A_t$ — persistently dead codes

Each active code $k$ gets the mean of the tokens assigned to it as its target. It also gets a **quota** $q_k \propto n_k$ (tokens assigned to it) of dead codes, and hands its target down to its $q_k$ nearest neighbours in $\mathcal N_t$. A dead code receiving from several sources averages them:

$$\mathbf t_j = \begin{cases} \frac{1}{|\mathcal U_j|}\sum_{u\in\mathcal U_j}\mathbf z_u, & j\in\mathcal S_t\\[4pt] \frac{1}{|\mathcal K_j|}\sum_{k\in\mathcal K_j}\mathbf t_k, & j\in\mathcal N_t,\ |\mathcal K_j|>0 \end{cases}$$

$$\mathcal L_{\text{code}} = \frac{1}{|\mathcal M_t|}\sum_{k\in\mathcal M_t}\|f_\theta(\mathbf e_k) - \text{sg}[\mathbf t_k]\|_2^2$$

Codes with no target keep themselves as target, giving zero loss. Only codes with $n_k > 1$ are allowed to propagate — a code claimed by a single token is not a reliable source.

Pilot check: freeze the encoder, fit the codebook to a fixed token cloud. Standard VQ loss stalls at **12.5% utilisation after 5000 steps**. Region VQ hits **full utilisation by step 500**.

### 3 · Decoupled Schedule — two jobs, two learning rates

The encoder–decoder has a messy multi-objective task and wants warmup-plus-annealing. The codebook has one clean job — track a moving distribution — and wants a **constant, high learning rate from step zero**, because the token distribution moves fastest at the very start, exactly when warmup is throttling it.

This explains an existing puzzle. SimVQ uses a constant LR everywhere, keeps utilisation, and pays with bad rFID (2.89). FVQ uses warmup-plus-annealing, gets good rFID, and had to buy back utilisation with a fancier projector. StableVQ: encoder–decoder on 10% warmup → 27% plateau → 63% anneal down to $0.01\times$, peak $1\mathrm{e}{-4}$; codebook on flat $1\mathrm{e}{-3}$.

### Setup

ImageNet $256\times256$, VQGAN-style encoder–decoder, downsample $f=16$ → $16\times16 = 256$ tokens per image. Adam, $\beta_1=0.9,\beta_2=0.95$, batch 128, 40 or 120 epochs. Codebook $16{,}384$ or $262{,}144$ codes at dim 256. Uniform codebook init. Discriminator for adversarial supervision. Metrics computed on-the-fly, not save-then-reload — which matters, because reload quantises to 8-bit PNG and changes the numbers.

## Ablation Studies and Experiments

**Reconstruction, ImageNet 256×256, 256 tokens.**

| Method | Projector | Ep | Codebook | rFID↓ | LPIPS↓ | Usage↑ | UR-AUC↑ |
|---|---|---|---|---|---|---|---|
| LlamaGen | — | 40 | 16k×8 | 2.19 | 0.2281 | 97% | — |
| LlamaGen | — | 40 | 16k×256 | 9.21 | — | **0.29%** | — |
| IBQ (bigger AE, 330 ep) | — | 330 | 16k×256 | 1.37 | 0.2235 | 96% | — |
| SimVQ | Linear-1 | 40 | 16k×256 | 2.89 | 0.2492 | 100% | 2.17 ± 0.32 |
| FVQ | ViTBlock-2 | 40 | 16k×256 | 1.70 | 0.2176 | 100% | 8.08 ± 0.24 |
| **StableVQ** | **Linear-1** | 40 | 16k×256 | **1.22** | 0.2235 | 100% | **60.59 ± 1.52** |
| FVQ | ViTBlock-2 | 40 | 262k×256 | 1.29 | 0.2003 | 100% | — |
| **StableVQ** | Linear-1 | 40 | 262k×256 | **1.05** | **0.1947** | 100% | — |
| **StableVQ** | Linear-1 | 120 | 262k×256 | **0.92** | **0.1893** | 100% | — |

The LlamaGen row at 16k×256 is the whole motivation in one line: keep everything, raise embedding dim from 8 to 256, and utilisation falls off a cliff to 0.29%.

**UR-AUC is the honest stability metric.** Vary only the Gaussian init scale of the codebook base, keep everything else fixed, log usage over early training, and take the normalised area under the usage curve averaged over mismatch settings:

$$\mathrm{UR\text{-}AUC} = \frac{1}{|\mathcal M|}\sum_{m\in\mathcal M}\frac{1}{T-t_0}\int_{t_0}^{T}U_m(t)\,dt$$

SimVQ 2.17, FVQ 8.08, StableVQ **60.59**. All three report "100% usage" in the main table; only StableVQ recovers usage quickly when the initial geometry is unfavourable. This is the number to take away — final-usage columns hide everything.

**Ablation 1 — codebook expansion** (tiny init range, case (a); linear projector, warmup-anneal peak $1\mathrm{e}{-4}$):

| Region VQ | Dyn. STE | De. Sch. | Peak commit | Util.↑ | rFID↓ |
|---|---|---|---|---|---|
| | | | 53.11 | 49.13% | 2.06 |
| ✓ | | | >200 (**NaN**) | — | — |
| | ✓ | | <0.1 | **1.27%** | 6.68 |
| | | ✓ | <0.1 | 83.77% | 1.71 |
| ✓ | ✓ | | <0.1 | 21.66% | 2.07 |
| ✓ | | ✓ | <0.1 | 100% | 1.72 |
| | ✓ | ✓ | <0.1 | 100% | 1.75 |
| ✓ | ✓ | ✓ | <0.1 | 100% | **1.70** |

Two rows here are more informative than the winning row.

**Region VQ alone diverges to NaN.** Giving every code a target is useless if the codebook's learning rate is still crawling through warmup — it cannot act on the targets fast enough, the gap widens, commitment explodes.

**Dynamic STE alone drops utilisation to 1.27%.** This is the paper's sharpest finding. Stabilise the encoder and utilisation *collapses* — proving that conventional VQ training was activating codes by accident, riding encoder oscillation. The instability was doing load-bearing work. Remove it without replacing the mechanism and you get a very stable, very dead codebook.

Region VQ + either of the other two recovers 100%. All three is best but only marginally over the pairs.

**Ablation 2 — codebook shrinkage** (case (b), $\ell_2$-normalised space, dim 1024, FVQ's ViTBlock-2 projector):

| Init | Region VQ | Util.↑ | rFID↓ |
|---|---|---|---|
| uniform | | 18.75% | 2.44 |
| gaussian | | 62.5% | 1.95 |
| uniform | ✓ | **100%** | **1.82** |
| gaussian | ✓ | **100%** | 1.90 |

FVQ's utilisation swings from 18.75% to 62.5% purely on initialisation choice. Region VQ removes the dependence entirely.

**Projector capacity is not the answer.** With the encoder frozen and a $10^{-4}$ init, neither a 2-layer linear nor a 2-layer MLP projector reaches high usage by 5k steps even at LR $5\times10^{-3}$. ViTBlock-2 does well at moderate LR but **crashes at high LR**. With Region VQ, all projector types reach high usage and the workable LR range widens a lot. So FVQ's expressive projector was compensating for a missing objective, not adding capability.

**Versus other STE fixes** (15 epochs, Region VQ on, everything else fixed):

| Method | Status | Usage | rFID↓ | PSNR↑ | LPIPS↓ |
|---|---|---|---|---|---|
| VQ-STE++ | **collapsed** | — | — | — | — |
| Rotation Trick | done | 100% | 3.64 | 20.40 | 0.2568 |
| Dynamic STE | done | 100% | **2.79** | **21.19** | **0.2359** |

VQ-STE++ collapses without its $k$-means init and norm bottleneck to pre-align things. The Rotation Trick rescales gradients by $\|q\|/\|z\|$, which *amplifies* the gradient whenever $\|q\|>\|z\|$ — including for badly matched pairs — and becomes exactly 1 (no effect) under $\ell_2$ normalisation. Dynamic STE only ever attenuates, and stays active under normalisation.

**Versus explicit distribution matching** (synthetic non-Gaussian mixture, $\zeta$ = how non-Gaussian, usage at 10k steps):

| Method | $\zeta{=}0$ | $\zeta{=}2$ | $\zeta{=}4$ | ms/step |
|---|---|---|---|---|
| Wasserstein VQ | 99.9% | 62.7% | 34.8% | 6.98 |
| MMD VQ | 99.9% | 92.5% | 75.6% | 295.6 |
| Region VQ | 99.8% | 93.2% | **99.4%** | 8.64 |

Wasserstein VQ matches Gaussian moments, so it falls apart when the target is a separated mixture. MMD VQ survives better but is $34\times$ slower from pairwise kernels. Region VQ assumes nothing parametric — it just moves targets locally.

**Overhead.** Decoupled Schedule: free. Dynamic STE: within noise. Region VQ: +0.85% to +2.16% step time, +0.04% to +1.24% memory. The cost *drops* as utilisation rises, since fewer dead codes need targets. All training-only; inference unchanged.

**Generation** (IBQ-style class-conditional AR transformer on the tokens):

| Tokenizer | Generator | Param | FID↓ | IS↑ | Rec.↑ |
|---|---|---|---|---|---|
| IBQ | IBQ-B | 342M | 2.88 | 254.7 | 0.51 |
| StableVQ | IBQ-B | 342M | **2.35** | 256.0 | **0.58** |
| IBQ | IBQ-L | 649M | 2.45 | 267.5 | 0.52 |
| StableVQ | IBQ-L | 649M | **2.18** | 250.4 | **0.59** |

Note the pattern: FID and recall improve, IS goes *down* slightly at IBQ-L. Better reconstruction buys coverage/diversity more than it buys per-sample classifier confidence.

## Worth Remembering

**The load-bearing instability.** Dynamic STE alone dropping utilisation from 49% to 1.27% is the result to carry away. Conventional VQ training relied on encoder noise to activate codes. Any intervention that stabilises one module in an entangled system can make the *system* worse, because you removed something that was quietly doing a job. Check the pairwise ablations before shipping a single-component fix.

**Report recovery, not final usage.** Every shared-projection method shows 100% usage in the headline table. The UR-AUC spread is 2.17 / 8.08 / 60.59. If you evaluate a tokenizer on final utilisation you cannot distinguish "robust" from "got a lucky init". Sweep the init scale and log the usage curve.

**Architecture was compensating for a missing objective.** FVQ's two ViT blocks existed to hold utilisation under an annealing schedule. Fix the schedule (decouple it) and the objective (Region VQ) and a single linear layer wins outright. Generally: before adding capacity, check whether the thing you are compensating for is an optimisation artefact.

**Region VQ is a nearest-neighbour heuristic.** Targets propagate to the $q_k$ *nearest* dead codes in projected space. This assumes the dead codes near an active code belong near it — plausible, and evidently good enough, but there is no guarantee and no theory. On the synthetic benchmark they had to auto-enlarge quotas when recipient sets overlapped too much, which is a real implementation wrinkle.

**Both biggest moves are backward-only or target-only.** Dynamic STE changes no forward value. Region VQ changes only what the codebook regresses toward. Neither touches inference. Cheap to try on an existing SimVQ or FVQ codebase.

**Where the dial actually sits.** Once stability stops being the bottleneck, the authors say reconstruction quality is decided by the optimisation recipe — they use a StyleGAN-T-style discriminator. Their rFID gains are therefore partly a training-recipe story, not only a quantisation story. Treat 1.22 vs 1.70 as "StableVQ's whole recipe" rather than "these three components alone".

**Unaddressed.** Only images, only $f=16$ / 256 tokens. No test on video or audio codecs despite the framing suggesting it should transfer. The FIFO window length $W$ is never ablated. And the semantic quality of the tokens — whether they are good for understanding tasks, not just reconstruction — is explicitly left as future work.

## Links

Related: [[VQ-VAE]] · [[Autoencoder]] · [[Mode Collapse]] · [[Latent Diffusion]] · [[Variational Autoencoder]] · [[Evaluating Generative Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Cyclical Learning Rates for Training Neural Networks]] · [[SGDR- Stochastic Gradient Descent with Warm Restarts]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Denoising Diffusion Probabilistic Models]] · [[Backpropagation]] · [[Vector Jacobian Product]] · [[Discrete Diffusion]] · [[Tokenization]] · [[Adam- A Method for Stochastic Optimization]]

New topics worth writing: straight-through estimator, codebook collapse, rFID, LPIPS, shared-projection reparameterisation, Wasserstein distribution matching for quantisers, MMD, FSQ / finite scalar quantization, separation of concerns in joint optimisation, learning-rate schedule decoupling per module
