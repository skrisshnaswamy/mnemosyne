---
title: "Percolation Dynamics in Optimization : Variance Cascades and Discrete Scale Invariance"
authors: ["Sai Niranjan Ramachandran", "Suvrit Sra"]
year: 2026
arxiv: "2609.02373"
url: https://arxiv.org/abs/2609.02373
priority: Good-To-Read
read_on: 2026-09-16
tags: [paper, transformers, optimization]
---
## The Core Idea

When you train a network with SGD, symmetry in the architecture creates flat regions of parameter space that act like traps. Two neurons that the architecture treats as interchangeable can slide into a configuration where they are *identical*, and once they are identical, gradient noise cannot pull them apart again. The network quietly becomes a smaller network. This was already known — Chen et al. (2023) called it "stochastic collapse."

What was *not* known: the **timing**. Does the collapse happen smoothly, a little each step? This paper says no. It happens in **discrete bursts**, and the bursts arrive on a **geometric schedule** — each gap to the final collapse is a fixed fraction of the previous gap.

The trick that makes this claim possible is a change of viewpoint. Treat each independent subnetwork as a **node in a graph**. Draw an edge between two nodes when their parameter trajectories have provably fused into the same invariant set. Training then *is* a percolation process: edges appear, clusters grow, and at some critical density the whole thing connects.

> [!NOTE] Percolation
> A model from physics: you keep adding random links between points, and at a critical link density a "giant" connected cluster suddenly appears. The size of that cluster is the order parameter. ^percolation

The twist that separates this from ordinary percolation: because $n$ neurons are permutation-symmetric, they cannot merge one at a time. The symmetry group $S_n$ binds all $n$ of them **simultaneously**. So the largest cluster does not grow $C_1 \to C_1 + 1$; it jumps $C_1 \to nC_1$. Multiplication, not addition.

That multiplicative growth rule breaks the continuous scaling symmetry that normal phase transitions have, leaving only a discrete subgroup. The result is **Discrete Scale Invariance (DSI)**: the critical densities $p_i$ at which merges happen satisfy

$$\lim_{i\to\infty}\frac{p_c - p_{ni}}{p_c - p_i} = \frac{1}{\lambda^{(n)}}, \qquad \lambda^{(n)} = n^{\sigma}$$

where $p_c$ is the final global collapse and $\sigma$ is the usual percolation critical exponent. For pairwise merges ($n=2$) under mean-field percolation ($\sigma=1$), $\lambda = 2$ exactly. The merge times double.

What this unlocks, if it holds: the cascade is a **forecast**. You watch three variance spikes, fit a line in log space, and you get $p_c$ — the moment of global structural collapse — before it happens. In the grokking experiment the cascade sits immediately before the generalisation jump.

The paper also extends the trapping argument to [[Adam- A Method for Stochastic Optimization|Adam]] and [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], which matters because nobody trains transformers with plain SGD.

## The Methodology

**Step 1 — continuous approximation.** Replace discrete SGD with an Itô SDE, Stochastic Gradient Flow:

$$d\boldsymbol{\theta}_t = -\nabla\mathcal{L}(\boldsymbol{\theta}_t)\,dt + \sqrt{\eta\Sigma(\boldsymbol{\theta}_t)}\,dW_t$$

The noise covariance factors as $D(\boldsymbol{\theta}) = D_m \cdot D_s(\boldsymbol{\theta})$, where the magnitude $D_m = \eta/2\beta$ is set purely by learning rate and batch size, and the *shape* $D_s$ is set by architecture and data. That split matters later: you control collapse with $\eta$.

**Step 2 — the transverse distance.** For subnetwork $i$ and invariant set $A$, track how far it sits off the set:

$$Y_t^{(i)} = \|\boldsymbol{\theta}_t^{(i)} - \pi_A(\boldsymbol{\theta}_t^{(i)})\|_2^2$$

Apply Itô's lemma. The infinitesimal generator is

$$\mathscr{A}Y_t^{(i)} = -2(\boldsymbol{\theta}_t^{(i)} - \pi_A(\boldsymbol{\theta}_t^{(i)}))^\top \nabla_{\boldsymbol{\theta}^{(i)}}\mathcal{L}(\boldsymbol{\theta}_t) + \eta\,\mathrm{Tr}(\Sigma_{ii}(\boldsymbol{\theta}_t))$$

Two competing terms: an inward pull from the [[Derivative#Gradient|gradient]], an outward push from noise. When the generator is $\le 0$, $Y$ is a **supermartingale** — in expectation it only shrinks. Doob's maximal inequality then bounds the escape probability by $\epsilon^{-1}\mathbb{E}[Y_{t_0}]$. The subnetwork is trapped.

The geometry is what makes this clean. Permutation symmetry means the invariant set is the eigenspace of a permutation matrix $Q$ for eigenvalue $1$ — a flat **affine** subspace. All principal curvatures are zero, so there is no geometric drift term, and the transverse motion reduces to a well-behaved 1D diffusion. (Continuous symmetries would give curved manifolds and break this; the authors flag it as future work.)

**Step 3 — from fluctuating graph to monotone graph.** Noise breaks and reforms edges constantly, so the raw graph is not monotone and percolation theory does not apply. The fix is a two-timescale argument. On the fast scale $\tau_{fast}$, the transverse process reaches local stationarity and the Fokker–Planck probability current at the boundary vanishes ($J_{ss}(\epsilon)=0$) — every transient break is exactly cancelled by a transient reform. On the slow scale $\tau_{slow}$, the expected distance $\mathbb{E}[Y_t]$ shrinks monotonically, so probability mass concentrates inside $\epsilon$ and $\frac{d}{dt}\mathcal{P}_e(t) > 0$. Coarse-graining over $\tau_{slow}$ leaves a pure-condensation filtration $G_{\tau_1} \subseteq G_{\tau_2} \subseteq \cdots$, indexed by edge density $p \in [0,1]$.

**Step 4 — order parameter and the detector.** $\mathcal{O}(p) = |C_{max}(p)|/N$. Each block-merge of $n$ components of size $i$ jumps it by $\Delta\mathcal{O} = (n-1)i/N$.

You cannot see these jumps in one training run, because noise shifts *where* each run jumps. So you run an ensemble of seeds and take the **relative variance**:

$$R_v(p) = \frac{\mathbb{E}[(\mathcal{O}(p) - \mathbb{E}[\mathcal{O}(p)])^2]}{\mathbb{E}[\mathcal{O}(p)]^2}$$

Near a merge, half the seeds have jumped and half have not — a bimodal mixture with weight $w$ — giving $\mathrm{Var} = w(1-w)(\Delta\mathcal{O})^2$. At $w=1/2$ this spikes sharply. **Variance spikes are the observable; the order parameter itself looks smooth.**

**Step 5 — the toy collapse condition.** For a two-layer product $\mathcal{L}(w) = \frac14(w^2-\mu)^2$ with multiplicative noise $D(w) \approx \frac12\zeta^2 w^2$, the stationary distribution follows potential $\Psi(w) = \frac{w^2}{2} - (\frac{2\mu}{\zeta^2}-2)\ln w$. It becomes non-normalisable (mass collapses to a delta at the origin) exactly when

$$\mu \le \frac{\zeta^2}{2}$$

Signal curvature below half the noise variance. Since $\zeta^2 \propto \eta$, **a high learning rate causes collapse and a low one reverses it**. That is the lever used in the task-shift experiment.

**Step 6 — the Adam extension.** Three problems, three fixes.

1. *State*. Adam is not Markov in $\boldsymbol{\theta}$ alone, so the invariant set is lifted to $\tilde{A} = \{(\boldsymbol{\theta},m,v) : P_\pi\boldsymbol{\theta}=\boldsymbol{\theta}, P_\pi m = m, P_\pi v = v\}$.
2. *Symmetry group shrinks*. Elementwise squaring in $v_t$ commutes with coordinate permutations but **not** with general rotations: $(Qg)_i^2$ mixes coordinates before squaring. So the admissible group narrows from orthogonal $Q$ to permutations $P_\pi$. Luckily $S_n$ neuron permutation is already in that class, so nothing upstream breaks.
3. *Heavy tails*. Transformer gradients have $\mathbb{E}|g_{t,i}|^p \le \sigma^p$ with $p \in (1,2]$ — possibly **infinite variance**. Fix: truncate at $\tau_t = (\sigma^p t/\log(1/\delta))^{1/p}$, paying a bias $b_t = O(\tau_t^{1-p})$ by Hölder + Markov.

The two EMA recursions are then linear filters. Via the one-sided $z$-transform, $\hat{M}(z) = \frac{1-\beta_1}{1-\beta_1 z^{-1}}\hat{G}(z)$, memory $\tau_1 = 1/(1-\beta_1)$. The second moment cannot be built from $\hat{G}(z)$ because squaring is convolution, not multiplication; instead they differentiate the characteristic-function transform twice, $-\partial_u^2\Phi(z,u)|_{u=0} = \mathcal{Z}\{\mathbb{E}[\hat g_t^{\circ 2}]\}$, which is linear in $u$ and therefore commutes with the sum.

The key claim is that Adam's preconditioner $P(\boldsymbol{\theta}_t) = \mathrm{diag}(1/(\sqrt{\hat v}+\epsilon))$, restricted to a symmetric block, is *nearly a scalar*: $P = c(\boldsymbol{\theta}_t)I + E_t$. A positive scalar rescaling preserves the affine geometry, so the whole SGD argument survives. Two error sources are bounded:

- **Cross-sectional** (deterministic): all coordinates in a block share one projection point $\bar\theta^{blk}$, so $|\theta_i - \theta_j| \le 2\sqrt\epsilon$, and $L$-Lipschitz gradients give $|\hat v_i - \hat v_j| \le 4L\tau_t\sqrt\epsilon$.
- **Temporal** (probabilistic): drift $O(L_m\tau_2\Delta_{\max})$ with $L_m = 6L^2$, plus filtered noise variance $O((1-\beta_2)\tau_t^4)$ by Wiener–Khinchin, valid only on the event that gradient autocorrelation is shorter than $\tau_2$.

Total residual $R_t = O(b_t + L_m\tau_2\Delta_{\max} + L\tau_t\sqrt\epsilon)$. If the drift margin beats $R_t$, trapping holds — **up to a stopping time** $\tau_\epsilon^{blk} \wedge \tau_E$. The scalar $c(\boldsymbol{\theta}_t)$ cancels in the DSI ratio, so $\lambda^{(n)} = n^\sigma$ is unchanged.

## Ablation Studies and Experiments

All empirical work on one Tesla T4, roughly one day of compute. 20 seeds for tabular, 15 for vision and grokking, 500 bootstrap resamples.

**Measurement change for real networks.** In deep nets weights rarely hit exactly zero, so the discrete cluster count is replaced by **spectral effective rank**: SVD the layer, normalise singular values into $p_k$, and take $\mathcal{O}(t) = \exp(-\sum p_k \log p_k)$ — the exponentiated Shannon entropy of the spectrum. (Compare [[Understanding Dimensional Collapse in Contrastive Learning|dimensional collapse]], which is the same diagnostic used for a different pathology.)

This softening has a theoretical cost, and they derive it. If $n$ components carrying fractional spectral mass $m$ merge, the effective rank ratio is $n^m$, not $n$, so the observed factor relaxes to

$$\lambda_{eff} = n^{m\sigma}$$

That is why the empirical $\lambda$ values are fractional. It also lets them run the inference backwards.

**Signal processing.** Raw variance decays as SGD settles into a basin, which would swamp the spikes. They work in $\log(\mathrm{Var}[\mathcal{O}(t)])$ and subtract a global linear secant before smoothing.

**The toy models.** A kinematic $K=3$ simulation with forced pairwise merges at $t=700$ and $t=1400$ gives $\lambda = 2.00$ exactly. An unconstrained $K=6$ GELU MLP under SGD with a high learning rate gives $\lambda \approx 2.00$ from two peaks. Then at $t=3000$ they flip both levers — raise target frequency (bigger $\mu$), drop learning rate (smaller $\zeta^2$) — violating $\mu \le \zeta^2/2$. A **reactive fragmentation peak** appears: the merged parameters split apart again. Collapse is reversible, exactly as the threshold predicts.

**Main results table (with the null model, which is the honest part):**

| Setting | Peaks | $\lambda$ | $R^2$ | FPR |
|---|---|---|---|---|
| Modular arithmetic, Transformer + AdamW | 3 | 2.11 | 1.00 | **0.1%** |
| FashionMNIST | 3 | 1.57 | 1.00 | 10.7% |
| UCI Heart Disease | 4 | 1.71 | 0.98 | **4.9%** |
| UCI Abalone | 6 | 1.28 | 0.97 | 15.6% |
| UCI German Credit | 4 | 1.99 | 0.86 | **80.2%** |
| MNIST | 2 | — | — | — |
| UCI Digits | 1 | — | — | — |

FPR comes from 1000 **phase-randomised spectral surrogates** — shuffle the phases of the variance trajectory, keeping the power spectrum, destroying local temporal structure, and count how often you "find" a cascade in pure noise.

**What did not work, and it is in the table.** German Credit produces a beautiful-looking 4-peak cascade with $\lambda = 1.99$, suspiciously close to the theoretical $2$ — and it is **noise**. 80.2% of phase-randomised surrogates produce something as good. The authors keep it as an explicit negative control. Without the null model, ordinary SGD variance mimics geometric scaling. MNIST and Digits also fail to produce full cascades — two peaks and one peak respectively, not enough to fit anything.

**The grokking run.** Transformer on mod-17 arithmetic, AdamW, weight decay 0.5, lr 0.003, 30000 epochs. Final train loss $0.005 \pm 0.004$, val loss $0.030 \pm 0.033$, effective rank collapses to $10.93 \pm 0.99$. The 3-peak cascade sits immediately before the generalisation jump with $R^2 = 1.00$ and FPR $0.1\%$. The authors explicitly decline to claim the cascade *causes* grokking.

**Inverting $\lambda$ to read off structure.** Since $\lambda_{eff} = n^{m\sigma}$, with $\sigma=1$ you can back out the variance mass $m$ if you assume $n$. Grokking's $\lambda = 2.11$ **exceeds** the hard ceiling for any pairwise merge ($\max_{m\le1} 2^m = 2$), which forces $n \ge 3$. Assuming $n=3$: $3^m = 2.11 \Rightarrow m \approx 0.68$, i.e. the dismantled memorisation circuits held ~68% of the spectral variance. FashionMNIST at $n=2$ gives $m \approx 0.65$; Heart Disease $m \approx 0.77$; Abalone $m \approx 0.36$. These are hypotheses, not measurements — the split between $n$ and $m$ is not identified from $\lambda$ alone.

**Adam diagnostic (Remark D.28), reported honestly.** On the grokking run, $\tau_{corr} = 1$ step against $\tau_2 \approx 1000$, so the low-correlation event held for all 60000 steps. But: $v_{\min}^{local} \approx 4.7\times10^{-13}$ for the tracked pair, and the network-wide minimum was $3.2\times10^{-23}$ — both *below* $\epsilon$. Assumption D.17 is therefore only meaningful block-by-block, never globally. The tracked pair had not merged by the end, and $\mathrm{corr}(|\theta_i-\theta_j|, |v_i-v_j|) = -0.11$. The tail index $p$ was never estimated directly; they report $\max|g|/\mathrm{std}(g) \approx 77$ as circumstantial evidence.

## Worth Remembering

**The discontinuity may not survive the limit, and they say so.** Riordan & Warnke (2011) proved that Achlioptas-type explosive percolation is actually *continuous* in the thermodynamic limit — earlier numerical claims of discontinuity were finite-size illusions. Corollary C.18 concedes the same fork here. The jump $\Delta\mathcal{O} = (n-1)i(N)/N$ survives $N\to\infty$ only if the merging components are already macroscopic, $i(N) = \Theta(N)$. If $i(N) = O(1)$, the jump vanishes as $O(1/N)$ and the transition is asymptotically smooth. **Which regime real architectures live in is unknown.** Assumption D.26 simply *assumes* $n(N)/N \to c > 0$ for the Adam result. The defence — that finite-$N$ topology is discontinuous regardless — is fair for practitioners but weakens the theory.

**The Adam result is a stopped result, not a global one.** Everything holds on $t \le \tau_\epsilon^{blk} \wedge \tau_E$. If gradient autocorrelation ever exceeds $\tau_2 = 1/(1-\beta_2)$, the guarantee lapses. With $\beta_2 = 0.999$ that is a 1000-step memory — a real constraint if your gradients are correlated across a curriculum or a data-ordering artefact.

**The nicest small result is the reversibility.** $\mu \le \zeta^2/2$ says collapse is a *tug of war between signal curvature and noise*, and both sides are things you control. It gives a mechanical account of why a high learning rate acts as [[Regularization|regularisation]] — not by penalising weights, but by kicking parameters into symmetric traps they cannot leave. And a learning-rate drop or a task shift un-collapses the network, which connects to catastrophic forgetting (Kirkpatrick et al.) from an unusual direction.

**Connection to scaling laws.** Remark C.26 suggests that the smooth power laws of [[Scaling Laws for Neural Language Models]] and [[Training Compute-Optimal Large Language Models (Chinchilla)|Chinchilla]] might be **ensemble averages over an underlying DSI cascade of symmetry-breaking microtransitions**. That is a real prediction: sharpen the measurement and the smooth curve should break into steps. This also mirrors the ensemble-vs-realisation distinction the paper needs internally — Definition C.24 is explicit that individual runs jump discretely while the ensemble mean is smooth, and the two are not in conflict because they describe different objects.

**Practical caveats if you wanted to use this.**
- You need an **ensemble**. The whole detector is variance across seeds. One run tells you nothing. That is expensive.
- You need the **null model**. German Credit is the proof. Fitting a log-linear cascade to unfiltered variance will find one whether or not it is there.
- The toy $\lambda = 2.00$ headline is a **two-point fit**, which the authors flag in the figure caption. A line through two points always has $R^2 = 1$.
- Inverting $\lambda$ for $n$ and $m$ is underdetermined. Only the ceiling argument for grokking ($\lambda > 2 \Rightarrow n \ge 3$) is genuinely forced.

**Open follow-ups.** Does this extend to curved invariant manifolds (continuous rotation symmetries), where $H_M \ne 0$ adds a geometric drift that couples transverse and tangential motion? Can $R_v$ spikes actually drive learning-rate scheduling, as the conclusion proposes? And does the cascade appear at billion-parameter scale, where effective rank is expensive and seeds are unaffordable?

## Links

Related: [[Adam- A Method for Stochastic Optimization]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[The Lottery Ticket Hypothesis]] · [[Understanding Deep Learning Requires Rethinking Generalization]] · [[Deep Double Descent- Where Bigger Models and More Data Hurt]] · [[Regularization]] · [[Momentum]] · [[Cross Entropy]] · [[Attention Is All You Need]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Mode Collapse]]

New topics worth writing: Grokking and delayed generalisation, Percolation theory and explosive percolation, Discrete Scale Invariance, Stochastic Gradient Flow as an SDE, Supermartingales and Doob's maximal inequality, Fokker–Planck equations and stationary distributions, Reeb graphs and topological data analysis, Permutation symmetry in neural loss landscapes, Effective rank as a capacity measure, Heavy-tailed gradient noise, Phase-randomised surrogate null models
