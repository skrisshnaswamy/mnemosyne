---
title: "Decoupled Weight Decay Regularization (AdamW)"
authors: ["Loshchilov & Hutter"]
year: 2017
arxiv: "1711.05101"
url: https://arxiv.org/abs/1711.05101
priority: Must-Read
read_on: 2026-08-24
tags: [paper, transformers, llm, optimization, vision]
---
## The Core Idea

Everyone calls it "weight decay" in their optimiser config. For plain SGD that name is fine. For Adam it is a lie, and the lie costs you real accuracy.

Two different things get confused:

- **L2 regularisation**: add $\frac{\lambda'}{2}\lVert\theta\rVert_2^2$ to the loss. The extra term $\lambda'\theta$ flows into the gradient, and then through whatever the optimiser does to gradients.
- **Weight decay**: multiply every weight by $(1-\lambda)$ each step, separately from the gradient. Nothing to do with the loss.

For SGD these are the same thing if you set $\lambda' = \lambda/\alpha$. That equivalence is why the field started using the words interchangeably.

For Adam they are **not** the same, and no choice of $\lambda'$ makes them the same. The reason: Adam divides each parameter's update by $\sqrt{\hat v_t}$, a running estimate of that parameter's gradient magnitude. If you fold the L2 penalty into the gradient, the penalty gets divided too. So a weight that historically sees big gradients gets its regularisation shrunk by the same big divisor. The weights that most need pulling toward zero are exactly the ones that get pulled least.

> [!NOTE] Decoupled weight decay
> Apply the shrink $-\eta_t \lambda \theta_{t-1}$ *outside* the adaptive rescaling, added directly to the parameter update. Every weight then decays at the same rate $\lambda$, regardless of its gradient history. ^decoupled-weight-decay

The fix is one line of code. It turns Adam from "generalises worse than SGD+momentum on image tasks" into "competitive with SGD+momentum" — a 15% relative cut in test error on CIFAR-10 and ImageNet32x32 at the default $\alpha=0.001$. It also makes the learning rate and the decay factor roughly independent hyperparameters, so you can tune them one at a time instead of on a diagonal.

This is why every modern transformer recipe says `AdamW`, not `Adam`. [[Improving Language Understanding by Generative Pre-Training (GPT-1)|GPT-1]] used it; the paper cites that as an early adopter.

## The Methodology

**The one-line change.** Standard Adam-with-L2 does, at line 6:

$$g_t \leftarrow \nabla f_t(\theta_{t-1}) + \lambda \theta_{t-1}$$

then feeds $g_t$ into the moment estimates $m_t, v_t$ and updates

$$\theta_t \leftarrow \theta_{t-1} - \eta_t\,\alpha\,\hat m_t / (\sqrt{\hat v_t} + \epsilon)$$

AdamW drops the $+\lambda\theta_{t-1}$ from line 6 and puts it in line 12 instead:

$$\theta_t \leftarrow \theta_{t-1} - \eta_t\left(\alpha\,\hat m_t / (\sqrt{\hat v_t}+\epsilon) \;+\; \lambda\,\theta_{t-1}\right)$$

Everything else — $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, [[Momentum|the bias-corrected moments]] $\hat m_t = m_t/(1-\beta_1^t)$, $\hat v_t = v_t/(1-\beta_2^t)$ — is untouched. Same for SGDW: move $\lambda\theta$ out of the gradient and out of the [[Momentum|momentum]] buffer, apply it straight to $\theta$.

**Why the divide-by-$\sqrt{v}$ matters, formally.** Proposition 2: if the optimiser step is $\theta_{t+1} \leftarrow \theta_t - \alpha \mathbf{M}_t \nabla f_t(\theta_t)$ with preconditioner $\mathbf{M}_t \neq k\mathbf{I}$, then L2 gives $-\alpha\lambda'\mathbf{M}_t\theta_t$ while weight decay gives $-\lambda\theta_t$. These match for all $\theta_t$ only if $\mathbf{M}_t$ is a scalar times identity. Adam's $\mathbf{M}_t = \mathrm{diag}(1/\sqrt{\hat v_t})$ is not.

Proposition 3 gives the interpretation. Freeze the preconditioner at $\mathbf{M} = \mathrm{diag}(s)^{-1}$. Then decoupled weight decay $\lambda$ is exactly L2 on a *rescaled* norm:

$$f_t^{\text{sreg}}(\theta) = f_t(\theta) + \frac{\lambda'}{2\alpha}\lVert \theta \odot \sqrt{s} \rVert_2^2$$

So decoupled decay regularises parameter $i$ in proportion to $\sqrt{s_i}$ — more regularisation for the historically-large-gradient dimensions, which is the opposite of what naive L2 does under Adam.

**Bayesian filtering justification (Aitchison 2018, added after the preprint).** View optimisation as tracking a posterior over the best value of each weight. The mean update is $\mu_{post} = \mu_{prior} + \Sigma_{post} g$, so the preconditioner *is* the posterior uncertainty. The state-transition prior is $P(\theta_{t+1}\mid\theta_t) = \mathcal{N}((I - A)\theta_t, Q)$. Setting $A = \lambda I$ gives you multiplication by $(1-\lambda)$ each step — i.e. decoupled weight decay falls out of the prior for free, applied *before* the uncertainty-scaled likelihood update. L2 would require the decay to depend on per-parameter uncertainty, which the framework does not produce. (See [[Kalman Filter]] for the filtering machinery.)

**Two extras bundled in the appendix.**

*Normalized weight decay.* The best $\lambda$ shifts as you train longer. They reparameterise:

$$\lambda = \lambda_{\text{norm}}\sqrt{\frac{b}{BT}}$$

with $b$ = batch size, $B$ = number of training points, $T$ = total epochs. So $\lambda_{\text{norm}}$ is "the decay if you got one batch pass".

*AdamWR.* AdamW plus [[Momentum|cosine annealing]] with warm restarts (SGDR). The schedule multiplier is $\eta_t = 0.5 + 0.5\cos(\pi T_{cur}/T_i)$, reset to 1 every $T_i$ epochs with $T_{i+1} = T_{mult}\cdot T_i$, e.g. $T_0 = 100$, $T_{mult}=2$.

**Experimental setup.** Shake-Shake-style 3-branch [[Deep Residual Learning for Image Recognition (ResNet)|ResNets]] (26 2x64d, 11.6M params; 26 2x96d, 25.6M params), batch size 128, standard CIFAR augmentation. Datasets: CIFAR-10 and ImageNet32x32 (1.2M images downsampled to $32\times32$). Budgets from 100 to 1800 epochs.

## Ablation Studies and Experiments

**Learning rate schedule × decay type (Figure 1, 26 2x64d, CIFAR-10, 100 epochs).** Three schedules — fixed LR, step-drop (at epochs 30/60/80), cosine annealing — each with Adam-L2 vs AdamW, swept over a 2D grid of $\alpha \times \lambda$. AdamW wins in all three. The gap *grows* with the better schedule. Cosine annealing beats both other schedules outright. Note the mild surprise: Adam adapts per-parameter learning rates already, and people therefore skip global LR schedules with it — that instinct is wrong, the global multiplier still buys a lot.

**Hyperparameter separability (Figure 2).** This is the plot that carries the argument. For SGD-with-L2, the band of good settings runs along the *diagonal* of the $(\alpha, \lambda)$ grid — change one without the other and you get worse. Example: at the best corner $\alpha = 1/2$, $\lambda = \frac{1}{8}\times 0.001$, moving either alone hurts. For SGDW the good band aligns with an axis: even at a badly-tuned $\alpha = 1/1024$, tuning $\lambda$ alone finds a good value ($\frac{1}{4}\times 0.001$). Same story for Adam → AdamW.

**Adam-with-L2 gets nothing from L2.** The striking negative result: Adam's best scores with non-zero $\lambda'$ were "comparable to the best ones obtained without the L2 regularization, i.e. when $\lambda = 0$." The regularizer was, in effect, doing no work at all. That is the whole failure in one sentence.

**Long runs, 1800 epochs, 26 2x96d (Figure 3).** 12 L2 settings for Adam, 7 normalized-decay settings for AdamW, fixed $\alpha = 0.001$. Learning curves overlap for the first half of training, then AdamW pulls ahead on *both* training loss and test error. Critically, they compare test error at matched training loss — AdamW is still better, so this is genuine generalisation, not just faster convergence. Same qualitative picture on ImageNet32x32.

**Does very long training make schedules unnecessary?** No. Standard Adam (L2 + fixed LR) for 1800 epochs on the bigger 26 2x96d net was at best comparable to AdamW with cosine annealing on the *smaller* 2x64d net trained for 18× fewer epochs.

**Warm restarts (Figure 4).** AdamWR reaches AdamW's final quality up to 10× faster (measured at the first restart). It closed most of the Adam↔SGDWR gap on CIFAR-10 and matched it on ImageNet32x32. Bonus finding: the restart variants generalise *better* than their non-restart counterparts, not just faster.

**Normalized weight decay (SuppFigure 3).** Without it, the optimal raw $\lambda$ shrinks noticeably as the epoch budget grows. With the $\sqrt{b/(BT)}$ scaling, the optimum sits at roughly the same $\lambda_{\text{norm}}$ across budgets, across CIFAR-10 and ImageNet32x32 (where an epoch is 24× longer), and even across AdamW and SGDW — around $\lambda_{\text{norm}} \in \{0.025, 0.05\}$. Had they reused the raw CIFAR-10 $\lambda$ on ImageNet32x32, it would have been ~5× too large.

**What didn't work / honest caveats.** Their earlier attempt at Adam + warm restarts (before this fix) had better anytime performance but was still not competitive with SGDR — precisely because the L2 regularisation was inert. The square-root normalization rule is admitted to be under-tested: "our choice of normalization is merely one possibility informed by few experiments"; the durable claim is that *some* normalization helps.

## Worth Remembering

- **The practical rule:** if your optimiser divides the update by anything per-parameter, decouple the decay. PyTorch's `Adam(weight_decay=...)` is L2, `AdamW(weight_decay=...)` is decoupled. They are different algorithms with the same-named argument.
- **The $\lambda$ values do not transfer** between `Adam` and `AdamW`. In Adam-L2, the effective decay is scaled by $\alpha$; in AdamW it is not. Copying a decay value across is a common silent bug.
- **Scale $\lambda$ when you change the training budget**, batch size, or dataset size. This is the part everyone forgets. $\lambda \propto \sqrt{b/(BT)}$.
- Authors admit the evidence is image-classification-only: "must be verified on a wider range of tasks." History was kind to them here — AdamW became the default for [[Attention Is All You Need|transformers]] and [[Foundation Models|large-scale pretraining]].
- They conjecture the same fix applies to AdaGrad and AMSGrad. Zhang et al. 2018 later confirmed it for K-FAC too.
- **A subtlety about what decay actually does.** With decoupled decay, the shrink is applied at the *same rate* to all weights, including ones with tiny gradients that L2-under-Adam would have hammered relatively harder. So it is not simply "more regularisation" — it is *differently distributed* regularisation, proportional to $\sqrt{s_i}$ in the fixed-preconditioner picture.
- **Open question worth chasing:** in practice people exclude biases and LayerNorm/BatchNorm gains from weight decay. This paper says nothing about that; it decays everything. The exclusion convention came later and is folklore-ish.
- Connects to [[Regularization]] generally: this is a case where two formulations that look identical in the textbook diverge the moment you change the optimiser. Worth carrying as a general warning about "equivalences" that hold only under vanilla [[Backpropagation|gradient descent]].

## Links

Related: [[Regularization]] · [[Momentum]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[Attention Is All You Need]] · [[Backpropagation]] · [[Derivative#Gradient|gradient]] · [[Kalman Filter]] · [[Loss, Objectives, and Business Alignment]] · [[MLE_L4_Question_Bank]] · [[Deep Learning]] · [[Uncertainty]]

New topics worth writing: Adam optimiser (full derivation of the moment estimates and bias correction), SGDR / cosine annealing with warm restarts, AdaGrad and RMSProp, preconditioning and diagonal approximations to the Hessian, the sharp-minima generalisation debate (Keskar et al. vs Dinh et al.), AMSGrad and Adam's convergence failure, Shake-Shake regularization, K-FAC
