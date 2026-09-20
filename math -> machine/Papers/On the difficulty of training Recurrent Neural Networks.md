---
title: "On the difficulty of training Recurrent Neural Networks"
authors: ["Razvan Pascanu", "Tomas Mikolov", "Yoshua Bengio"]
year: 2012
arxiv: "1211.5063"
url: https://arxiv.org/abs/1211.5063
priority: Must-Read
read_on: 2026-09-17
tags: [paper, theory]
---
## The Core Idea

Recurrent nets multiply the same matrix over and over when you [[Backpropagation|backprop]] through time. Multiplying one number by itself many times either blows up or dies. The same is true for matrices. That is the whole problem, and it was already known from Bengio et al. (1994). What this paper adds is *why the fix is so simple*, and the fix itself: **clip the gradient by its norm**.

The new claim is geometric. When gradients explode in an RNN, they explode along a direction $\mathbf{v}$ — the eigenvector of $\mathbf{W}_{rec}$ with the largest eigenvalue. The authors show, with a toy linear model, that if the first derivative explodes along $\mathbf{v}$, the *second* derivative does too. So the error surface does not have a smoothly steep region; it has a **wall**. A thin, near-vertical cliff sitting next to a wide flat valley.

That picture explains everything about why training fails and how to fix it. [[Backpropagation#^exploding-gradients|Standard SGD]] walks along the flat floor, touches the wall, and gets catapulted somewhere random — all the progress so far is destroyed. The direction of the gradient at the wall is fine; it points back into the valley. Only the *magnitude* is insane. So: keep the direction, throw away the length.

> [!NOTE] Gradient norm clipping
> If $\|\mathbf{g}\| \ge \tau$, replace $\mathbf{g}$ with $\frac{\tau}{\|\mathbf{g}\|}\mathbf{g}$. Otherwise leave it alone. One hyperparameter, one line of code, no second-order information. ^gradient-clipping

Why this did not exist before as a *justified* method: Mikolov was already clipping element-wise in his PhD thesis, and it worked, but nobody had a story for why such a crude hack should be safe. The wall hypothesis gives one. It also explains why a second-order method is not needed — and would not even help. A Newton step divides gradient by curvature, but if gradient and curvature explode at *different rates*, the ratio still explodes. Clipping does not care about the rate.

The paper's second contribution — a regulariser for vanishing gradients — is the part history forgot, because LSTM won that fight. Still worth knowing, and covered below.

## The Methodology

**The model.** A plain RNN, but written in an unusual order:

$$\mathbf{x}_t = \mathbf{W}_{rec}\,\sigma(\mathbf{x}_{t-1}) + \mathbf{W}_{in}\mathbf{u}_t + \mathbf{b}$$

Note $\sigma$ is applied to the *previous* state, then multiplied. This makes the Jacobian clean.

**Where the explosion lives.** Total cost $\mathcal{E} = \sum_t \mathcal{E}_t$. Write each gradient as a sum over *temporal contributions*:

$$\frac{\partial \mathcal{E}_t}{\partial \theta} = \sum_{1\le k\le t} \frac{\partial \mathcal{E}_t}{\partial \mathbf{x}_t}\,\frac{\partial \mathbf{x}_t}{\partial \mathbf{x}_k}\,\frac{\partial^+ \mathbf{x}_k}{\partial \theta}$$

Here $\frac{\partial^+ \mathbf{x}_k}{\partial \theta}$ is the *immediate* derivative — treat $\mathbf{x}_{k-1}$ as a constant. The dangerous factor is the transport term:

$$\frac{\partial \mathbf{x}_t}{\partial \mathbf{x}_k} = \prod_{t \ge i > k} \mathbf{W}_{rec}^T \, \mathrm{diag}(\sigma'(\mathbf{x}_{i-1}))$$

A product of $t-k$ matrices. This is the same object as the chain of multiplications in any deep net, except the matrix is *the same one every time*, so the eigenvalues compound cleanly.

**The conditions, stated precisely.** Let $\gamma$ bound $|\sigma'|$ ($\gamma = 1$ for $\tanh$, $\gamma = 1/4$ for sigmoid) and $\lambda_1$ be the spectral radius of $\mathbf{W}_{rec}$.

- $\lambda_1 < \frac{1}{\gamma}$ is **sufficient** for vanishing. Proof: each Jacobian has 2-norm $\le \|\mathbf{W}_{rec}^T\|\,\|\mathrm{diag}(\sigma')\| < \frac{1}{\gamma}\gamma = 1$. Call that bound $\eta < 1$. Then by induction the long-term term shrinks like $\eta^{t-k}$ — exponentially in the gap.
- $\lambda_1 > \frac{1}{\gamma}$ is **necessary** for exploding. Not sufficient. Invert the proof above.

In the appendix they sharpen this for the linear case with a power-iteration argument: decompose $\frac{\partial \mathcal{E}_t}{\partial \mathbf{x}_t}$ in the eigenbasis of $\mathbf{W}_{rec}$; after $l = t-k$ steps, the term with the largest eigenvalue $\lambda_j$ dominates because $|\lambda_i/\lambda_j|^l \to 0$, so the whole thing $\approx c_j \lambda_j^l \mathbf{q}_j^T$. Growth is exponential *and confined to one direction*.

**The dynamical-systems view.** An RNN with fixed $\theta$ is a map applied repeatedly, so the state falls into an attractor. Change $\theta$ slowly and behaviour changes smoothly — except at **bifurcation boundaries**, where attractors appear, vanish or change shape. Doya (1993) blamed exploding gradients on crossing bifurcations. This paper corrects that: crossing a bifurcation is **neither necessary nor sufficient** (a bifurcation is global; your state may sit in a basin that is untouched). What *is* sufficient is a **local** event — crossing the boundary between two basins of attraction. A tiny change in $\theta$ then sends $\mathbf{x}_\infty$ somewhere entirely different, so $\Delta\mathbf{x}_t$ is huge, so the gradient is huge. In Figure 3 there are only two values of $b$ with a bifurcation, but a whole *range* with a basin crossing.

They also extend this to input-driven nets, which prior work fudged by calling the input "bounded noise". Instead: split the step into a fixed map $\tilde{F}(\mathbf{x}) = \mathbf{W}_{rec}\sigma(\mathbf{x}) + \mathbf{b}$ and a time-varying kick $U_t(\mathbf{x}) = \mathbf{x} + \mathbf{W}_{in}\mathbf{u}_t$. $U_t$ resists analysis, but $\tilde{F}$ is a proper autonomous system and tells you *where* the dangerous boundaries are.

Neat dual reading of vanishing: if $\frac{\partial \mathbf{x}_t}{\partial \mathbf{x}_k} \to 0$, then $\mathbf{x}_t$ genuinely does not depend on $\mathbf{x}_k$ — the model has converged into an attractor and forgotten. Vanishing gradient is not a numerical artefact; it is the model having no memory.

**Clipping the exploding side.** Algorithm 1, the whole thing, applied to the full gradient vector after BPTT. It differs from Mikolov's element-wise clipping only in that norm-clipping preserves the direction, so you are guaranteed to still be moving in a descent direction for the current minibatch. They report the two behave similarly in practice. Setting $\tau$: look at the running average gradient norm over many updates. Training is insensitive to it, and small thresholds are fine.

Read it as an *instantaneous* learning-rate adaptation — unlike [[Adam- A Method for Stochastic Optimization|Adam]] or Adagrad, which average gradient statistics over time and therefore cannot react to a single abrupt spike.

**The vanishing-gradient regulariser.** Penalise the backward signal for changing length as it travels back one step:

$$\Omega = \sum_k \left( \frac{\left\| \frac{\partial \mathcal{E}}{\partial \mathbf{x}_{k+1}} \frac{\partial \mathbf{x}_{k+1}}{\partial \mathbf{x}_k} \right\|}{\left\| \frac{\partial \mathcal{E}}{\partial \mathbf{x}_{k+1}} \right\|} - 1 \right)^2$$

Two subtleties. First, it only asks for norm preservation **in the direction the error actually points**, not for all eigenvalues to be 1 — a much softer ask. Second, for efficiency they take only the immediate derivative of $\Omega$ w.r.t. $\mathbf{W}_{rec}$, treating $\mathbf{x}_k$ and $\frac{\partial \mathcal{E}}{\partial \mathbf{x}_{k+1}}$ as constants. The $\frac{\partial \mathcal{E}}{\partial \mathbf{x}_k}$ values fall out of BPTT for free.

Why a *regulariser* and not just better optimisation? Their argument: forcing $\|\frac{\partial \mathbf{x}_t}{\partial \mathbf{x}_k}\|$ up makes the model sensitive to *all* inputs from $k$ to $t$, most of which are junk. So error goes **up** first, and only later does the model learn to ignore the junk. A descent method will never take that step. You have to force it. The soft constraint also pushes you away from the attractor and towards the basin boundaries — which is exactly where gradients explode. So the two fixes need each other: **the regulariser makes clipping mandatory.**

## Ablation Studies and Experiments

Three variants throughout: **SGD**, **SGD-C** (+clipping), **SGD-CR** (+clipping +regulariser). Hyperparameters by grid search on validation.

**Pathological synthetic tasks** (from Hochreiter & Schmidhuber 1997, success = under 1% error on 10,000 test sequences).

*Temporal order*: a long stream of distractor symbols with two symbols from $\{A,B\}$ hidden at random early and mid positions; classify the order ($AA$, $AB$, $BA$, $BB$). 50 hidden units, lr 0.001, $\alpha = 2$, clip threshold 6.

- Plain SGD and SGD-C both **fail beyond length 20** — vanishing kills them.
- Below length 20, SGD-C clearly beats SGD. This is the key diagnostic: the exploding problem appears exactly when the task needs more memory. The model starts with $\lambda_1 < 1$ (one attractor, memory decays fast); more memory demands larger $\lambda_1$; past a threshold you enter the rich regime where explosions happen. Exploding gradients are *evidence you are learning long dependencies*, not just a nuisance.
- SGD-CR: **100% success up to length 200**, matching the longest sequences Martens & Sutskever used with Hessian-Free.

*Generalisation*: one model trained on mixed lengths 50–200 generalises to sequences **up to 400 steps** — twice the training length — still under 1% error.

*Other tasks*, all with SGD-CR: addition, multiplication, 3-bit temporal order, and noiseless memorisation all at 100%. Addition beat Hessian-Free, which degrades as sequences approach 200. Noiseless memorisation needed a **separate model per length** (50/100/150/200) — the one-model-for-all trick did not transfer there.

*What barely worked*: the **random permutation** problem — 100 symbols, only the last is predictable — succeeded in **1 run out of 8**. The honest failure in the paper.

**Polyphonic music prediction** (negative log-likelihood per step, lower better; sigmoid RNN, 200-step sequences, clip threshold 8):

| Dataset | fold | SGD | SGD+C | SGD+CR |
|---|---|---|---|---|
| Piano-midi.de | train | 6.87 | 6.81 | 7.01 |
| | test | 7.56 | 7.53 | **7.46** |
| Nottingham | train | 3.67 | 3.21 | 3.24 |
| | test | 3.80 | 3.48 | **3.46** |
| MuseData | train | 8.25 | 6.54 | 6.51 |
| | test | 7.11 | 7.00 | **6.99** |

Note the size of the jumps. On Nottingham and MuseData, **clipping alone does nearly all the work** (3.80 → 3.48; 7.11 → 7.00); the regulariser adds a rounding error on top (→3.46, →6.99). Piano-midi is the exception where the regulariser is the only thing that helps, and it does so by making *train* error worse (6.81 → 7.01) while test improves.

**Character-level language modelling**, Penn Treebank, bits/character, 500 sigmoid units, no biases, clip threshold 45:

| Task | fold | SGD | SGD+C | SGD+CR |
|---|---|---|---|---|
| next char | train | 1.46 | 1.34 | 1.36 |
| | test | 1.50 | 1.42 | **1.41** |
| 5th char ahead | train | N/A | 3.76 | 3.70 |
| | test | N/A | 3.89 | **3.74** |

The 5-steps-ahead variant is a deliberate stress test: predicting further out should make long-term structure matter more. It does — the regulariser gap widens to 0.15 bits (3.89 → 3.74) versus 0.01 on the standard task. Plain SGD could not be run at all ("N/A"). This is the strongest evidence the regulariser is doing what it claims.

**The interpretation of the ablations.** Train error *and* test error both improve with clipping. So **clipping is solving an optimisation problem, not regularising** — it is not trading fit for generalisation, it is just letting SGD reach places it could not. The regulariser is the opposite: it usually makes train error slightly worse and test error slightly better.

**What did not work, in prior approaches they dismiss:**

- **L1/L2 on $\mathbf{W}_{rec}$** keeps the spectral radius under 1, so gradients provably cannot explode — but this traps the model in the single-point-attractor regime where all information decays exponentially. No generator networks, no long memory. You have prevented explosion by preventing learning.
- **Echo State Networks** dodge the problem by not learning $\mathbf{W}_{rec}$ or $\mathbf{W}_{in}$ at all. Same trap: spectral radius under 1 by construction. Leaky-integration units help, but act as a low-pass filter, so they suit low-frequency signals only. And since the weights are random, nobody knows how big such a model must be for a real task.
- **Pre-programming the regime** (Doya) requires knowing the target asymptotic behaviour in advance, which you usually do not, and is hard to set up even when you do. Worse, it does not stop basin-boundary crossings, which need no bifurcation.
- **Teacher forcing** works, empirically, and is poorly understood — but needs a target at **every** timestep.
- **[[Long Short-Term Memory (Neural Computation)|LSTM]]** fixes vanishing structurally and says nothing about exploding.
- **Hessian-Free with structural damping** works, and the paper offers a guess as to why: in high dimensions long- and short-term components are probably near-orthogonal, so the Hessian can rescale them separately — but you cannot guarantee that. The fact that structural damping (which shrinks $\|\frac{\partial \mathbf{x}_t}{\partial\theta}\|$) *also* helps suggests curvature and gradient explode at **different rates**, meaning curvature alone is not enough to tame the gradient.

## Worth Remembering

- The paper's own explanation for why Hessian-Free beat other second-order methods: it uses the full Hessian (so it catches exploding directions that are not axis-aligned) and **re-estimates it every step** (so it sees abrupt curvature changes), where most second-order methods average curvature over many steps under a smoothness assumption that walls violate.

- The wall hypothesis needs the valley to be **wide** — you need a large flat region beside the wall for the clipped step to land in. If the valley were narrow, clipping would just bounce you off the other side and you would want a proper second-order method.

- Practical schedule detail from the appendix, easy to miss: on natural tasks the regularisation coefficient $\alpha$ follows a $1/t$ **decay** over epochs (e.g. $\alpha_t = \frac{1}{2t}$). Their reasoning: $\Omega$ forces attention onto long-term correlations *at the expense of short-term ones*, so once the long-term structure is in, you want to let go and let the model use nearby context. On the synthetic tasks $\alpha$ is constant.

- Clipping thresholds are not comparable across setups without knowing the reduction. Threshold 6 for the synthetic tasks, 8 for music (where they take the **mean** over sequence length), 45 for language modelling (where they take the **sum**). Learning rates also differ by 10–50× between the clipped and unclipped runs — SGD without clipping needed lr 0.001 on PTB versus 0.01 with it. Clipping lets you use a bigger learning rate, which is a large part of the practical win.

- Everything here transfers to any architecture with repeated multiplication — the analysis is about $\prod \mathbf{W}$, not about recurrence specifically. Clipping is standard in [[Proximal Policy Optimization Algorithms|PPO]] and in transformer pretraining for exactly this reason, long after RNNs stopped being used.

- Limitation the authors flag: the chaotic regime is explicitly excluded ("some of the following observations may not hold"). Also, the analysis of $\tilde{F}$ tells you where boundaries *are likely* to be, but the input maps $U_t$ can oppose the movement, so it is a heuristic, not a guarantee.

- The clipping half was adopted universally within about two years. The regulariser was not, because [[Long Short-Term Memory (Neural Computation)|LSTM]] plus clipping was simpler and worked better. Worth noticing that the surviving idea is the crude one-line hack, and the principled soft constraint died.

- Open question this raises: the regulariser's penalty targets norm preservation only along the error direction. That is close in spirit to what a residual connection does structurally in [[Deep Residual Learning for Image Recognition (ResNet)|ResNet]] — enforce an identity path so the Jacobian is near 1 by construction rather than by penalty. Architecture beat optimisation here, as it usually does.

## Links

Related: [[Backpropagation]] · [[Long Short-Term Memory (Neural Computation)]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Vector Jacobian Product]] · [[Understanding the difficulty of training deep feedforward networks (Xavier init)]] · [[Delving Deep into Rectifiers (He init, PReLU)]] · [[Adam- A Method for Stochastic Optimization]] · [[Auto-regressive models]] · [[Seq2Seq models]] · [[Regularization]] · [[Derivative]] · [[Pytorch Autograd]] · [[Momentum]] · [[GPU processing]]

New topics worth writing: Backpropagation through time, Spectral radius and the power iteration method, Bifurcation and basins of attraction, Hessian-Free optimisation and structural damping, Echo state networks and reservoir computing, Teacher forcing, Gradient clipping in modern training recipes
