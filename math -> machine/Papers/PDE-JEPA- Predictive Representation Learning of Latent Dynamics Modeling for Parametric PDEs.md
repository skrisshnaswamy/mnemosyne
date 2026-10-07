---
title: "PDE-JEPA: Predictive Representation Learning of Latent Dynamics Modeling for Parametric PDEs"
authors: ["Zhentao Tan", "Jianrong Zhang", "Ruijie Quan", "Yi Yang"]
year: 2026
arxiv: "2609.34715"
url: https://arxiv.org/abs/2609.34715
priority: Good-To-Read
read_on: 2026-10-05
tags: [paper, self-supervised, vision]
---
## The Core Idea

For physics simulation, people usually learn a latent space by **reconstruction**: squash the physical field into a small code, decode it back, and ask the decoder to recover the original. That is an [[Autoencoder]]. The latent code is then evolved forward in time by a small network.

This paper asks a different question: *what makes a latent space good to evolve?* and splits it into two things that turn out not to be the same thing.

**First finding — predictive pretraining gives you a better-informed latent space.** Instead of reconstructing pixels, pretrain with [[JEPA]]: mask part of the trajectory, and predict the *latent features* of the masked part, not the raw field. Probing frozen features shows JEPA features let you read out both the instantaneous physical field and the governing parameter (viscosity, wave speed, forcing coefficients) better than reconstruction features do.

**Second finding — being informative does not make it evolvable.** When you actually roll the model forward step by step, feeding its own predictions back in, vanilla JEPA is *worse* than the reconstruction baseline on Wave-2D and Vorticity in-distribution. The information is there; the geometry is wrong.

> [!NOTE] Evolvability vs informativeness
> A probe reads a representation of an *observed* state. A rollout repeatedly evolves *predicted* states with no observations to lean on. A latent space can score well on the first and badly on the second. ^evolvable-not-informative

Why the geometry is wrong, concretely: physical trajectories turn gently, latent trajectories zig-zag. On Vorticity, the mean turning angle between consecutive steps is $31.3^\circ$ in field space but $60.5^\circ$ in JEPA latent space. On Burgers, $18.2^\circ$ versus $59.4^\circ$. A zig-zag path is harder for a small step-predictor to extrapolate, and errors compound — the same compounding problem as in [[Imitation Learning#^errors-compound|behaviour cloning]].

So **PDE-JEPA** adds two things on top of frozen JEPA features:

1. **PAG** — a tiny projector that bends the latent trajectory so its *turning geometry* matches the physical field's turning geometry, while an anchor loss stops it from destroying the information JEPA found.
2. **PSP** — a latent time-derivative that is split into a parameter-free part plus parameter-scaled parts, mirroring how PDE coefficients actually enter the equation. This is the piece that buys out-of-distribution extrapolation to unseen governing parameters.

Headline: across 9 parametric PDE benchmarks, 33.4% average error reduction in-distribution and 51.4% when extrapolating to parameters never seen in training.

## The Methodology

### The problem setup

A *parametric* PDE is a whole family of physics problems at once:

$$\frac{\partial \mathbf{u}}{\partial t} = \mathcal{F}(t, \mathbf{x}, \mathbf{u}, \nabla\mathbf{u}, \nabla^2\mathbf{u}, \ldots; \bm{\xi})$$

$\mathbf{u}(t,\mathbf{x})$ is the physical field (vorticity, wave height, concentration). $\bm{\xi}$ is the **governing parameter** — viscosity $\nu$, wave speed $c$, damping $k$, forcing frequency. Fix $\bm{\xi}$ and you have one system. Vary it and you have a family. The model must forecast long rollouts *and* work for $\bm{\xi}$ values outside the training range.

### Stage 1 — JEPA pretraining (then freeze)

A [[An Image is Worth 16x16 Words (ViT)|ViT]] with tubelet size 1 (so patch embedding never mixes adjacent frames) encodes the trajectory into $N$ spatial tokens per timestep, each of dimension $D$.

The loss is standard [[Self-Supervised Learning from Images with I-JEPA|I-JEPA]]:

$$\mathcal{L}_{\mathrm{JEPA}}=\frac{1}{|\mathcal{M}|}\sum_{i \in \mathcal{M}}\left\| P_\phi(E_\theta(\mathbf{x}_\mathcal{C}), \mathbf{m})_i - \mathrm{sg}(\bar{E}_\theta(\mathbf{x})_i) \right\|_1$$

Context encoder sees visible tokens $\mathcal{C}$; the predictor guesses the features at masked positions $\mathcal{M}$; the targets come from an [[Momentum#^ema-weights|EMA copy]] of the encoder with the gradient stopped (`sg`). Stop-gradient plus EMA target is the anti-collapse machinery from [[Bootstrap Your Own Latent (BYOL)]].

Two design details that matter:

- $\bm{\xi}$ is **never** shown to the encoder or the JEPA predictor during pretraining. The encoder is forced to be parameter-agnostic.
- Pretraining sees whole trajectories, but downstream states are encoded **frame by frame**, so $\mathbf{z}_t$ cannot leak future observations.

The encoder is frozen for everything that follows.

### Stage 2 — PAG: Physics-Aligned Latent Geometry

A token-wise residual MLP:

$$\mathbf{q}_t = \mathbf{z}_t + G_\phi(\mathbf{z}_t), \qquad G_\phi(\mathbf{z}) = W_2\,\sigma(W_1\,\mathrm{LN}(\mathbf{z}))$$

$\sigma$ is GELU, LN is [[Layer Normalization|layernorm]], and $W_2$ is **zero-initialised** so the module starts as the identity — a correction to the coordinates, not a relearning of the representation. It does not mix across tokens or timesteps; it only rotates/stretches channels.

**The geometry loss.** Take normalised step directions in both spaces:

$$\mathbf{d}^u_t = \frac{\mathbf{u}_{t+1}-\mathbf{u}_t}{\|\mathbf{u}_{t+1}-\mathbf{u}_t\|_2+\epsilon}, \qquad \mathbf{d}^q_t = \frac{\mathbf{q}_{t+1}-\mathbf{q}_t}{\|\mathbf{q}_{t+1}-\mathbf{q}_t\|_2+\epsilon}$$

Measure how much the path turns over a lag $\ell$ with a dot product, $s^u_{t,\ell}=\langle \mathbf{d}^u_t, \mathbf{d}^u_{t+\ell}\rangle$ and likewise $s^q_{t,\ell}$. Then match them:

$$\mathcal{L}_{\mathrm{geo}}=\sum_{\ell \in \{1,2,4\}} w_\ell \,\mathrm{SmoothL1}\!\left(s^q_{t,\ell},\, \mathrm{sg}(s^u_{t,\ell})\right)$$

The clever bit: physical fields live in $\mathbb{R}^{CHW}$ and latents in $\mathbb{R}^{ND}$, different dimensions. You cannot compare the vectors. But a cosine between two consecutive step-directions is a **scalar**, so the two spaces become comparable. Lags 1, 2 and 4 capture both local wiggle and longer-range curvature.

**Two guards against breaking what JEPA learned.** An identity anchor,

$$\mathcal{L}_{\mathrm{anchor}}=\frac{\|\mathbf{q}-\mathbf{z}\|_2^2}{\|\mathbf{z}\|_2^2+\epsilon}$$

and an auxiliary causal predictor that must still predict $\Delta\mathbf{q}_t = \mathbf{q}_{t+1}-\mathbf{q}_t$ in the new coordinates:

$$\mathcal{L}_{dyn}=\frac{\|\widehat{\Delta\mathbf{q}}_t - \Delta\mathbf{q}_t\|_2^2}{\|\Delta\mathbf{q}_t\|_2^2+\epsilon}$$

Total: $\mathcal{L}_{\mathrm{align}}=\mathcal{L}_{dyn}+\lambda_{\mathrm{geo}}\mathcal{L}_{\mathrm{geo}}+\lambda_{\mathrm{anchor}}\mathcal{L}_{\mathrm{anchor}}$. Afterwards, throw the auxiliary predictor away and freeze the projector too. Note the projector itself never sees $\bm{\xi}$ — parameters enter only through the auxiliary predictor.

### Stage 3 — PSP: Physics-Structured Latent Predictor

A normal conditional predictor would concatenate $\bm{\xi}$ to the input and let the network do whatever it likes with it. That fits inside the training range and extrapolates unpredictably outside it. Instead, the latent dynamics are a [[Neural ODE|continuous-time vector field]] with the parameter's entry point *structured by hand*:

$$\frac{d\mathbf{q}}{dt}=C_\theta(\mathbf{q})+\sum_{j=1}^{M} r_j(\bm{\xi})\,D_\theta^{(j)}(\mathbf{q})$$

$C_\theta$ is the shared, parameter-free evolution. Each $D_\theta^{(j)}$ is a state-dependent *response*, multiplied by a normalised scalar coefficient $r_j(\bm{\xi})$. The motivation is simply the form of the equations. Navier–Stokes in vorticity form:

$$\frac{\partial\omega}{\partial t} = -(\mathbf{u}\cdot\nabla)\omega + \nu\nabla^2\omega$$

Viscosity $\nu$ multiplies one term. So the latent version is $\dot{\mathbf{q}} = C_\theta(\mathbf{q}) + r(\nu)D_\theta(\mathbf{q})$. The branches are not required to *be* the real operators — the decomposition is only an inductive bias on *where the parameter can act*. Because $r_j$ is linear in the (normalised) coefficient, pushing $\nu$ past the training range extrapolates linearly rather than arbitrarily.

In practice there is a shared backbone $\mathcal{H}_\theta(\mathbf{q})$ and one head per branch. Each dataset gets a hand-written decomposition and hand-chosen normalisation (Appendix C.1.1) — e.g. Wave-2D uses $\dot{\mathbf{q}}=C_\theta+r_{c^2}D^c_\theta - r_k D^k_\theta$ with $r_{c^2}=c^2/\sqrt{\langle c^4\rangle_{\mathrm{tr}}}$ and $r_k = k/\sqrt{\langle k^2\rangle_{\mathrm{tr}}}$. Note the signs are copied from the PDE.

Integration is fixed-step **RK4 with 4 substeps** per observation interval. Supervision:

$$\mathcal{L}_{\mathrm{pre}}=\frac{\|\widehat{\mathbf{q}}_{t+1}-\mathbf{q}_{t+1}\|_2^2}{\|\Delta\mathbf{q}_t\|_2^2+\epsilon}$$

Dividing by the size of the true step is a relative loss — it stops slow-moving systems from being ignored.

### Stage 4 — decoder, trained on rollouts

This stage is easy to miss and is quietly important. With encoder, projector and predictor all frozen, the decoder is trained on latents produced by **autoregressive rollout from $\mathbf{x}_0$ alone**. Future ground-truth frames are only targets, never encoded as inputs. So the decoder sees the drifted latent distribution it will actually face at inference — a direct fix for the train/test mismatch that bites step-by-step models.

The decoder is convolutional: project $D \to D_0$, reshape the token grid to an $N_h \times N_w$ feature map, then alternate $2\times$ upsample and residual blocks until you reach $H\times W$. The time axis is folded into the batch, so the decoder models no temporal structure at all — all dynamics live in the latent predictor.

### Data and compute

9 benchmarks: Advection, Burgers, Heat, Wave-B, Combined (1D); Wave-2D, Vorticity, HeterNS, Gray–Scott (2D). Mostly 12,000 training trajectories each, from the Zebra / UniSolver / ENMA generators. Metric is relative $L^2$ over the full rollout. Optimiser is AdamW; predictors use $\beta_2=0.95$ instead of $0.999$. 4× RTX PRO 6000 (96 GB), ~9,000 total GPU hours.

## Ablation Studies and Experiments

### In-distribution (relative $L^2$, lower better)

Baselines span four families: parametric solvers (FNO, CAPE, CoDA, GEPS), in-context solvers (ViT variants, Zebra), foundation models (UniSolver, MPP, DPOT-S, Poseidon-T) and latent solvers (LE-PDE, LNS, MAE-PDE, ENMA).

| Benchmark | Best baseline | PDE-JEPA | Change |
|---|---|---|---|
| Heat | 0.0933 (Poseidon-T) | **0.0274** | −70.6% |
| Wave-B | 0.1093 (Poseidon-T) | **0.0350** | −68.0% |
| Burgers | 0.0869 (LE-PDE) | **0.0428** | −50.7% |
| Wave-2D | 0.2070 (Zebra) | **0.1140** | −44.9% |
| Vorticity | 0.0592 (LNS) | **0.0348** | −41.2% |
| Combined | 0.0085 (CAPE) | **0.0074** | −12.9% |
| Gray–Scott | 0.0323 (UniSolver) | **0.0284** | −12.1% |
| HeterNS | 0.0098 (UniSolver) | **0.0089** | −9.2% |
| Advection | **0.0068** (CoDA) | 0.0074 | +8.8% |

**The one loss is informative.** Advection is pure translation — the shape never changes, it just moves. Several methods are already below $10^{-2}$; the benchmark is saturated and the task reduces to faithful reconstruction, which is exactly what reconstruction-based latents are built for. Gains grow with the nonlinearity of the dynamics.

### Out-of-distribution governing parameters

| Benchmark | Best baseline | PDE-JEPA | Change |
|---|---|---|---|
| Combined ($\alpha$ beyond $[0,1]$) | 0.038 (UniSolver) | **0.008** | −77.9% |
| Wave-2D | 0.610 (LNS) | **0.157** | −74.2% |
| Gray–Scott | 0.083 (Poseidon-T) | **0.033** | −59.7% |
| HeterNS, viscosity | 0.037 (UniSolver) | **0.011** | −69.5% |
| HeterNS, forcing | 0.105 (UniSolver) | **0.103** | −1.5% |
| Vorticity | 0.320 (Zebra) | **0.288** | −9.7% |

The degradation from ID to OOD is small on three systems: Combined $0.0074 \to 0.0084$, Wave-2D $0.1140 \to 0.157$, GS $0.0284 \to 0.0337$. Vorticity OOD is the hard case — the OOD split drops viscosity to $[10^{-5},10^{-4}]$, two orders of magnitude below the training support $[10^{-3},10^{-2}]$, which changes the flow regime, not just a coefficient. The 9.7% gain there is modest and the absolute error (0.288) is large.

### Which component does the work

Rollout error, vanilla JEPA → +PAG → +PSP:

| | Vorticity ID | Vorticity OOD | Wave-2D ID | Wave-2D OOD |
|---|---|---|---|---|
| Vanilla | .086 | .491 | .363 | .502 |
| + PAG | .040 | .397 | .143 | .321 |
| + PSP | .034 | .288 | .114 | .157 |

Clean division of labour. **PAG is the in-distribution fix** (−53% on Vorticity ID, −61% on Wave-2D ID). **PSP is the extrapolation fix** (−27% and −51% on OOD, versus only −15% and −20% on ID).

### Does PAG actually do what it claims?

Measured directly on $\mathbf{z}$ vs $\mathbf{q}$, with **no dynamics predictor involved** — so this is a property of the projector, not of the forecasting model. Error against the physical trajectory's geometry:

| Dataset | Turning-angle MAE | 2nd-order variation MAE | Lag-1 cosine MAE |
|---|---|---|---|
| Vorticity | 29.1° → 6.3° (−78%) | .46 → .10 (−77%) | .36 → .05 (−84%) |
| Wave-2D | 35.5° → 10.1° (−71%) | .46 → .14 (−68%) | .50 → .14 (−74%) |
| Burgers | 41.4° → 20.8° (−49%) | .73 → .37 (−49%) | .43 → .17 (−59%) |
| Gray–Scott | 28.8° → 20.4° (−29%) | .39 → .27 (−30%) | .39 → .26 (−32%) |

The size of the geometry fix tracks the size of the rollout gain. Vorticity and Wave-2D get the biggest geometric correction and the biggest error drop; Burgers and GS get smaller ones in both.

### Does PSP actually respond correctly to parameters?

Change only $\bm{\xi}$, hold the initial condition fixed, and compare the predicted change against the numerical solver's change. `Cos.` is directional agreement (higher better), `Amp.` is magnitude ratio (1.0 is right).

| | Field Cos. | Field Amp. | Latent Cos. | Latent Amp. |
|---|---|---|---|---|
| Vorticity, PAG only | .626 | .846 | .913 | 1.142 |
| Vorticity, +PSP | **.708** | **.987** | **.924** | **1.043** |
| Wave-2D, PAG only | .958 | .979 | .939 | .990 |
| Wave-2D, +PSP | **.986** | **.996** | **.947** | **1.002** |

The amplitude column is the real evidence. Without the structured decomposition the model under- or over-reacts to a parameter change (.846, 1.142); with it the response magnitude lands near 1. That is exactly what you'd expect from forcing the parameter to enter as a linear scalar multiplier.

### What did not work, or barely moved

- **Vanilla JEPA alone is a regression.** Better probes, worse rollouts — .086 ID on Vorticity against .0592 for LNS. Predictive pretraining is a starting point, not a method.
- **Advection:** beaten by CoDA, a much older and simpler approach.
- **HeterNS forcing-frequency OOD:** 1.5% improvement. PSP gives forcing frequency $m$ its own branch with coefficient $m$ itself (no normalisation), and that linear treatment evidently does not capture how an unseen spatial forcing pattern changes the flow. Changing a *coefficient* extrapolates; changing a *function's shape* does not.
- **Vorticity OOD** remains at 0.288 absolute — a two-orders-of-magnitude viscosity shift is not recovered by a linear response term.

## Worth Remembering

**The honest cost of PSP.** This is not a learned mechanism. For every PDE family the authors hand-write the decomposition, the number of branches, the signs, and the normalisation constants, reading them off the analytical equation (Appendix C.1.1). You need to know the governing equation's structure, and you need the numerical value of $\bm{\xi}$ at inference. That makes PDE-JEPA a strong method for *known* parametric families and not an off-the-shelf solver for an unlabelled trajectory dataset. The flip side: this is precisely what buys the extrapolation, and it is honest about it.

**Four separate frozen stages.** Pretrain → freeze → align → freeze → predictor → freeze → decoder. Nothing is trained end to end. That makes each ablation interpretable but also means there is no mechanism for the encoder to be improved by the forecasting objective. Someone should check whether a light end-to-end finetune helps or collapses it.

**The reusable trick, independent of PDEs.** Comparing trajectory *geometry* across two spaces of different dimension by matching scalar cosine-of-consecutive-directions is general. Anything where a latent path must mirror an observable path — latent planning, [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)|world models]], [[Mastering Diverse Domains through World Models (DreamerV3)|Dreamer]]-style imagination — could use it. The authors cite "temporal straightening for latent planning" as the nearest relative.

**The rollout-trained decoder deserves more attention than it gets.** It is mentioned only in the appendix and never ablated, yet training the decoder on self-generated latents is a textbook fix for compounding error. How much of the headline 33.4% comes from the decoder rather than PAG/PSP is unanswered. The Table 2 ablation holds that stage constant, so it is not isolated anywhere in the paper.

**The zero-initialised residual.** $W_2 = 0$ so the projector begins as exactly the identity. Same trick as [[LoRA#^lora-init|LoRA's zero-init B matrix]] and adaLN-zero in [[Diffusion Transformer|DiT]]: start from the pretrained behaviour and let the gradient decide how far to move. Cheap and always worth doing when bolting a module onto a frozen model.

**Open question for your own work.** Observation 2 — informative features that roll out badly — is the same disease as embedding evaluation in retrieval: a probe or a linear readout says a representation is good, and downstream performance says otherwise. The diagnostic here (measure trajectory geometry, not just readability) is a template for asking the same question of embeddings used in sequential decision systems.

**Caveat on the reported averages.** The 33.4% and 51.4% figures are means of per-benchmark *relative* improvements over the *best* baseline on each benchmark, and the best baseline differs every time. That is a generous framing: no single competitor is being beaten by 33% across the board. Vorticity OOD (9.7%) and HeterNS forcing (1.5%) sit in the same average as Combined OOD (77.9%).

## Links

Related: [[JEPA]] · [[Self-Supervised Learning from Images with I-JEPA]] · [[JEPA-Anything- Learning Predictive Models across Different Worlds]] · [[LeJEPA- Provable and Scalable Self-Supervised Learning]] · [[Bootstrap Your Own Latent (BYOL)]] · [[Autoencoder]] · [[Neural ODE]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Imitation Learning]] · [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Embedding Physics Priors in Robot Learning- A Survey]] · [[Layer Normalization]] · [[LoRA]] · [[Momentum]]

New topics worth writing: Fourier Neural Operator, neural operators for parametric PDEs, Runge–Kutta integration, reduced-order modelling / latent dynamics for simulation, Zebra in-context PDE pretraining, Poseidon PDE foundation models, relative $L^2$ rollout error as a metric, temporal straightening for latent planning
