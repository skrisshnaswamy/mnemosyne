---
title: "On-Policy or Off-Policy Learning? A Systematic Study of Distillation Dynamics"
authors: ["Julianna Piskorz", "Antonin Berthon", "Mihaela van der Schaar"]
year: 2026
arxiv: "2609.35259"
url: https://arxiv.org/abs/2609.35259
priority: Good-To-Read
read_on: 2026-10-03
tags: [paper, llm, rl, optimization, vision]
---
## The Core Idea

A popular belief in post-training is that **where the training data comes from matters more than almost anything else.** If the model trains on its own samples ("on-policy"), the story goes, it forgets less, changes fewer weights, and generalises better than if it trains on a fixed dataset ("off-policy"). This belief comes mostly from comparing [[Training language models to follow instructions with human feedback|SFT]] against RL with verifiable rewards.

The problem: those two things differ in *many* ways at once — the loss, the reward signal, how dense the supervision is, the optimiser, the learning rate. So you cannot tell which difference caused which effect.

This paper builds a setting where you can change **only** the rollout policy. Strong-to-weak distillation: a big frozen teacher, a small student, a token-level [[KL Divergence|KL]] loss. Swap who generates the text, keep everything else identical.

The result overturns the folklore. On-policy rollouts give **no consistent advantage** in final accuracy, in [[Fine-Tuning#The failure modes 🪤|catastrophic forgetting]], or in how sparse the weight updates are. What actually controls those things:

- **KL direction** (forward vs reverse) drives task accuracy, training stability, and how much of the output space the student keeps.
- **Learning rate** drives forgetting and update sparsity. Almost all of it.

And there is a clean asymmetry explaining *when* rollout policy does matter: **forward KL barely cares where the data comes from; reverse KL cares a lot** and wants the student's own samples. This falls out of the gradient algebra, which the authors prove, and then confirm by sliding the rollout policy along a continuous student→teacher dial.

One place on-policy data genuinely wins: generalising to a **harder version** of the training task (+10–15 points). But that edge does not survive a later RLVR stage.

Why this did not exist before: nobody had decoupled the two choices. Forward KL was *always* paired with teacher rollouts, reverse KL *always* with student rollouts, because those pairings are the two chain-rule halves of sequence-level KL. Breaking the pairing is the whole experiment.

> [!NOTE] Rollout policy
> The model that *generates the text you train on*. Off-policy distillation (OffPD) trains on teacher-generated text. On-policy distillation (OnPD) trains on text the student just produced. It is separate from the loss you compute on that text. ^rollout-policy

> [!NOTE] Semi-gradient
> In OnPD the loss depends on $\theta$ twice — through the student distribution *and* through the sampled trajectory. The paper stops gradients through sampling, so the update is only the first half. This means the OnPD + reverse-KL update is **not** the true gradient of sequence-level reverse KL, despite the usual claim. ^semi-gradient

## The Methodology

### The objective

One loss, with two knobs. For prompt $x$ and completion $y$ of length $L_y$ sampled from rollout policy $\rho$:

$$\mathcal{L}(\theta)=\mathbb{E}_{x\sim p_{\text{data}}}\,\mathbb{E}_{y\sim\rho(\cdot|x)}\left[\frac{1}{L_y}\sum_{n=1}^{L_y}\mathcal{D}\big(\pi_S^\theta(\cdot|x,y_{<n})\,\|\,\pi_T(\cdot|x,y_{<n})\big)\right]$$

- $\rho=\pi_T$ → OffPD. $\rho=\pi_S^\theta$ → OnPD.
- $\mathcal{D}$ is forward or reverse KL, computed over the **whole vocabulary** at every token, not from a sampled token. That is what lets them break the pairing: with full-vocabulary KL, the KL direction no longer forces a rollout source.

Forward KL (mode-covering — punishes the student for missing anything the teacher likes):
$$D_{\text{F-KL}}=\sum_{v\in\mathcal{V}}\pi_T(v)\log\frac{\pi_T(v)}{\pi_S(v)}$$

Reverse KL (mode-seeking — punishes the student for putting mass where the teacher has none):
$$D_{\text{R-KL}}=\sum_{v\in\mathcal{V}}\pi_S(v)\log\frac{\pi_S(v)}{\pi_T(v)}$$

### The experimental grid

Four factors, varied independently:

| Factor | Values |
|---|---|
| Rollout policy | OnPD, OffPD (later: a continuous $\lambda$ spectrum) |
| KL direction | forward, reverse |
| Learning rate | $1\times10^{-5}$, $5\times10^{-5}$ (sweep to $6\times10^{-5}$) |
| Seeds | 42, 43, 44 |

### Models and tasks

- Main: **Llama-3.1-8B teacher → Llama-3.2-1B student.** Replicated with **Qwen2.5-7B → Qwen2.5-1.5B.**
- Same family on purpose: shared [[Tokenization|tokenizer]] and vocabulary, so token distributions are directly comparable.
- Three tasks: **MedReason** (medical MCQ), **Science** (SciKnowEval MCQ), **Countdown-3** (arithmetic — hit a target using three numbers exactly once each).
- Teachers are built per task, so there is a real capability gap: 500 steps of LoRA SFT (rank 256) on worked GPT-5 demonstrations, then 500 steps of [[GRPO]] for Science and MedReason (reward 2.0 for correct + up to 0.5 for correct `<think>`/`<answer>` tags). Final teacher accuracy: Llama MedReason 85.5%, Science 67.5%, Countdown-3 99.0%. Frozen thereafter.

### Training details that mattered

- **Full-parameter** fine-tuning of the student (no [[LoRA]]), 150 steps, effective batch 32 prompts, one completion each.
- Fused [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], no weight decay, constant LR after 10 warmup steps, [[On the difficulty of training Recurrent Neural Networks|gradient clipping]] at norm 1.0.
- **Temperatures are fixed and asymmetric**: student at $T=1.0$, teacher at $T=0.5$ — for both sampling and the KL logits. The teacher produced degenerate tails at $T=1.0$. Ablated in Appendix A.12; the best achievable accuracy of each method does not move.
- 150 steps is enough: learning curves plateau, and nearly all forgetting happens in the first 50 steps.

### How forgetting and sparsity are measured

- **Forgetting** = drop in the unweighted mean over seven OOD benchmarks: MMLU-Pro, TruthfulQA-MC1, IFEval, HumanEval-Instruct, EQ-Bench, BBQ, ToxiGen (3,935 examples total, via lm-eval-harness).
- **Update sparsity** = fraction of parameters that barely moved:
$$S_\tau=\frac{1}{P}\sum_{i=1}^{P}\mathbb{I}\big[|\theta_i^{\text{final}}-\theta_i^{\text{init}}|<\tau\big],\quad \tau=10^{-6}$$

### The rollout-policy spectrum

The clever instrument. Instead of two options, build a dial $\lambda$ that mixes the two policies in log space:

$$(z_\lambda)_v=\tfrac{1}{2}\big(\log\pi_S(v)+\log\pi_T(v)\big)+\tfrac{\lambda}{2}\big(\log\pi_S(v)-\log\pi_T(v)\big)$$

then $\pi_\lambda=\operatorname{softmax}(z_\lambda)$. So $\lambda=-1$ is exactly the teacher, $\lambda=+1$ exactly the student, $\lambda=0$ their geometric midpoint. Past $\pm1$ you **extrapolate** — sampling tokens one model likes and the other does not.

Extrapolation can blow up on tokens both models hate, so (following contrastive decoding) they clip the log-ratio at $c=2\log 3\approx2.2$ and mask any token below $\alpha=0.2$ of the favoured policy's max probability. This caps odds amplification at 3× over $\lambda\in[-2,2]$.

> [!NOTE] Likelihood-ratio spectrum
> A one-parameter family that interpolates *and extrapolates* between two policies by linearly mixing their log-probabilities. Turns a binary design choice (on- vs off-policy) into a continuous axis you can plot performance against. ^rollout-spectrum

### The gradient analysis

Differentiating each KL with respect to the student logits $z_S^\theta$:

$$\nabla_\theta D_{\text{F-KL}}=\sum_{v\in\mathcal{V}}\big(\pi_S^\theta(v)-\pi_T(v)\big)\nabla_\theta (z_S^\theta)_v$$

$$\nabla_\theta D_{\text{R-KL}}=\sum_{v\in\mathcal{V}}\pi_S^\theta(v)\left[\log\frac{\pi_S^\theta(v)}{\pi_T(v)}-D_{\text{R-KL}}\right]\nabla_\theta (z_S^\theta)_v$$

Read the two weights side by side. That is the whole paper's mechanism.

- Forward KL's weight is a **difference of two probabilities**. It lives in $[-1,1]$. Bounded, always.
- Reverse KL's weight is a probability times an **unbounded log-ratio**. It vanishes where the student has no mass (so the student can never learn a mode it has already dropped), and explodes where the student is confident but the teacher is not.

They turn this into theorems. With a bound $B$ on the largest singular value of the student-logit [[Derivative#Jacobian|Jacobian]], forward KL is Lipschitz in the rollout distribution:

$$\big\|\bar\nabla_\theta\mathcal{L}_{\text{F-KL}}(\theta;\rho)-\bar\nabla_\theta\mathcal{L}_{\text{F-KL}}(\theta;\rho')\big\|_2\le 2\sqrt{2}\,B\,\mathbb{E}_x\big[\mathrm{TV}(\rho,\rho')\big]$$

and each trajectory's gradient has second moment $\le 2B^2$. **No such bound exists for reverse KL** — they build a two-token counterexample where two rollout policies sit $\varepsilon$ apart in total variation, yet their expected reverse-KL updates differ by $\tfrac{\varepsilon\sqrt{2}}{8}\log\frac{1-\delta}{\delta}$, which diverges as $\delta\to0$. Note the student is *uniform* in that construction; the blow-up comes from the teacher assigning near-zero probability to a token the student supports. A bound returns only if you separately cap the log-ratio range at $R$, giving $2BR\cdot\mathbb{E}[\mathrm{TV}]$.

## Ablation Studies and Experiments

### Headline: rollout policy does almost nothing

Averaged over the three tasks, best mean in-distribution accuracy:

| | OnPD | OffPD |
|---|---|---|
| Best accuracy | 72% | 73% |

Now split by KL direction instead:

| | Accuracy range across rollout policies and LRs |
|---|---|
| Forward KL | 71–73% |
| Reverse KL | 35–72% |

Forward KL is flat. Reverse KL swings 37 points, mostly with learning rate.

### Forgetting is a learning-rate story

| Learning rate | Change in mean OOD score |
|---|---|
| $1\times10^{-5}$ | at most −1.3 points |
| $5\times10^{-5}$ | −11.2 to −14.0 points |

Rollout-policy differences are small next to that, and where they exist, **OffPD forgets *less*** in nearly every matched comparison — the opposite of the hypothesis under test.

The sweep from $1\times10^{-5}$ to $6\times10^{-5}$ on Countdown-3 shows OOD score falling monotonically and sparsity falling roughly linearly, with the same shape under both rollout policies.

The most useful practical finding: **forgetting is not the price of learning the task.** Forward KL at $1\times10^{-5}$ gets top in-distribution accuracy with essentially zero OOD loss. Two models with equal task accuracy can differ by 13 points in retained capability. (The authors connect this to the "cliff" phenomenon in SFT.)

### Sparsity is also a learning-rate story

| Learning rate | Update sparsity ($\tau=10^{-6}$) |
|---|---|
| $1\times10^{-5}$ | 85.3–89.6% |
| $5\times10^{-5}$ | 51.9–60.0% |

**OffPD produced updates at least as sparse as OnPD in every matched comparison.** The "RL finetunes small subnetworks" claim does not reproduce when rollout policy is the only thing you change. Robust to tightening $\tau$ to $10^{-8}$.

The paper also tabulates (Appendix E) the learning rates behind the public checkpoints used in the original sparsity claim: SFT checkpoints median $5\times10^{-6}$, RL/preference checkpoints median $5\times10^{-7}$ — a 10× gap. The confound is right there in the hyperparameters.

### The spectrum experiment — this is the real result

Countdown-3, 150 steps, sweeping $\lambda\in[-2,2]$:

- **Forward KL**: accuracy stays above 80% across the *entire* spectrum, varying by only **5.2 points** at $1\times10^{-5}$.
- **Reverse KL**: large swings in mean and huge seed variance. At the low LR it clearly prefers student-favoured rollouts ($\lambda>0$) and degrades on teacher-favoured ones. At the high LR it is unstable and sometimes collapses outright.
- Forgetting and sparsity: barely move along $\lambda$. Learning rate dominates again.

Qwen2.5 reproduces the asymmetry, and adds a surprise: at the high learning rate, **student-favoured rollouts ($\lambda>0$) produce *more* forgetting** under forward KL, monotonically. On-policy data actively hurt retention there.

### Output coverage: forward KL wins

[[LoRA|pass@$k$]] on Countdown-3 at $1\times10^{-5}$: for $\lambda>0$ (on-policy end), the two KL directions have *similar* pass@1 — but **forward KL reaches significantly higher pass@10**. Reverse KL's mode-seeking behaviour narrows the output distribution even when single-shot accuracy looks the same. Replicated on Qwen2.5.

### Generalisation: the one place on-policy wins

Evaluating the same Countdown-3 checkpoints, with no further training, on **Countdown-4E** (four operands drawn from 1–10, $T=0.5$): performance rises steadily along the spectrum, and on-policy rollouts ($\lambda>0$) beat off-policy by **10–15 points pass@$k$** under *both* KL directions. This is the paper's clearest pro-on-policy result.

### …but it does not survive RLVR

Take those Countdown-3 checkpoints, run 300 steps of GRPO/DAPO on Countdown-4 (correctness-only reward: 2 or 0, LR $2\times10^{-6}$, LoRA rank 256, 8 completions per prompt):

- Reverse-KL checkpoints from LR $1\times10^{-5}$ shoot up fast — including OffPD ones starting near 0% — then **collapse**.
- Forward-KL OffPD checkpoints from $1\times10^{-5}$ (and reverse-KL OffPD from $5\times10^{-5}$) climb steadily and **finish highest**, despite starting near 0% accuracy on Countdown-4.

So the best RLVR starting points were the off-policy ones. The initial generalisation edge of on-policy data is not a durable edge. Consistent with "Good SFT optimizes for SFT, better SFT prepares for RL".

### What did not work, and what the negative controls rule out

**Gradient clipping is not the explanation.** The unclipped global gradient norm exceeds 1.0 at *every* logged step, so clipping was always active — a real worry that it was hiding differences. Turning it off: reverse KL deteriorates badly (as the unbounded log-ratio predicts), forward KL stays effective even at high LR, and there is still **no consistent OnPD–OffPD gap**. Forgetting and sparsity remain LR-driven.

Their explanation for why clipping changes accuracy but not forgetting is worth keeping: [[Momentum#Momentum inside Adam|Adam]] normalises by $\hat{v}_t$, so a uniform scale change mostly cancels. One unclipped spike biases $m_t$ toward a bad batch *and* inflates the slow-decaying $v_t$, which then suppresses corrective updates — especially damaging for OnPD, because a bad early update poisons every subsequent rollout. But the net parameter displacement stays small, so forgetting and sparsity are untouched.

**Sampled KL is not the explanation.** Replacing full-vocabulary KL with single-sample estimators — forward becomes $-\log\pi_S^\theta(y_n)$ with $y_n\sim\pi_T$, reverse becomes the score-function surrogate $\operatorname{sg}[\log\pi_S^\theta-\log\pi_T]\log\pi_S^\theta$ with $y_n\sim\pi_S^\theta$ — reproduces every qualitative conclusion.

**Short rollouts are not the explanation (mostly).** Teacher responses on the three main tasks average only 94–135 tokens. They repeat on Numina-MATH (teacher responses ~622 tokens, 20K examples, Qwen2.5-Math-1.5B teacher → Qwen2.5-1.5B student, 1,000 steps). Here **OnPD + reverse KL does take the top MATH-500 score** (59.0% vs 53.1% untrained baseline, all configs landing 53.3–59.0%), which is consistent with the "on-policy helps on harder generalisation" finding. But forward KL stays more robust, and learning rate still rules: at $1\times10^{-5}$ all methods keep baseline OOD and 92–93% sparsity; at $5\times10^{-5}$ OOD drops from 41.4% to 33.7–35.9% and sparsity to 63–70%. **One run per condition** — the authors label it suggestive, and the reported "best checkpoint along the trajectory" metric is optimistic by construction.

**The forward-KL low-KL trap.** A genuinely surprising diagnostic. Since $D_{\text{KL}}(T\|S)=\mathrm{CE}(T,S)-H(T)$, a *small* forward KL can hide two *enormous* terms. In OnPD forward-KL runs on Science, steps 10–20 showed low KL while teacher entropy and cross-entropy both approached ~12 nats (≈163k effective choices) — teacher and student both nearly uniform and nearly matched, so the gradient is near zero and useless. Useful supervision was concentrated at early token positions, and the low-entropy region *advanced along the trajectory* as training went on: fix the early tokens, and the teacher can then say something informative about later ones. Self-reinforcing. Yet step-50 accuracy was fine (37% Science, 56% MedReason, OOD down only 0.8–1.3 points). Re-sampling at $T\in\{0.3,0.5,0.7,1.0\}$ showed accuracy holding to 0.7 and crashing at 1.0 — **the capability was there, just not reachable at the training temperature.** Equal final numbers under OnPD and OffPD can conceal wildly different optimisation paths.

**Teacher-style transfer — the odd one out.** Instruct a Qwen2.5-3B teacher to reason in Spanish, distil into Qwen2.5-1.5B on Science, then have GPT-5 label 1,000 student responses per condition. Every configuration transfers to Spanish almost completely — **except OnPD + reverse KL**, which keeps the student's English. Mechanism: on student-generated *English* prefixes, the teacher still supports English continuations despite its instruction; mode-seeking reverse KL is happy to match that English mode without being forced to cover the Spanish alternatives. Forward KL, being mode-covering, is dragged onto the Spanish tokens. OffPD transfers even under reverse KL, because the prefixes themselves are Spanish. One seed, one judge.

**Likelihood dynamics (Figure 21).** During OffPD the teacher is ~90% confident on its own trajectories and the student climbs slowly toward it. During OnPD the student starts ~80% confident on its own text while the teacher gives it only ~20% — and under forward KL the student's confidence *collapses* early and only slowly recovers. So OnPD and OffPD really are operating in different regimes, even where final accuracy matches.

## Worth Remembering

**The practical rule.** If you are doing strong-to-weak distillation: use **forward KL at a small learning rate** ($1\times10^{-5}$ here) with **teacher-generated data**. You get top accuracy, near-zero forgetting, the sparsest updates, the best pass@$k$, and the best downstream RLVR behaviour — and you can generate the data once and reuse it forever. OnPD requires continual generation throughout training. The authors' explicit ask: **treat OffPD as a mandatory baseline** in on-policy distillation papers.

**If you must use reverse KL**, it wants student rollouts, and it wants gradient clipping. Without clipping it falls apart. Its gradient weight $\pi_S(v)\big[\log\frac{\pi_S(v)}{\pi_T(v)}-D_{\text{R-KL}}\big]$ is the whole reason — unbounded above, zero where the student has already dropped a mode.

**The deeper caution.** Much of the "on-policy is special" literature compares [[Training language models to follow instructions with human feedback|SFT]] to [[GRPO]]/[[PPO]] at learning rates that differ by an order of magnitude. Learning rate is a confound in that whole genre. Before believing any claim about forgetting or update sparsity, check whether the learning rates were matched.

**What the theory does and does not say.** Theorem C.1 bounds a *one-step expected semi-gradient* in total-variation distance between rollout distributions. It does not prove that different rollout policies yield similar models, similar final accuracy, or similar forgetting — and the authors are careful about this. The empirical results are *consistent with* the bound, not implied by it. The bound is also loose when $\mathrm{TV}\approx1$ or when $B$ is large.

**Limitations the authors own.** Students ≤1.5B parameters; reasoning traces ≤2,000 tokens; one teacher per task, held fixed throughout. Whether OnPD/OffPD differences emerge at larger scale, longer horizons, or specific student–teacher mismatches is open. The Numina-MATH long-rollout experiment is single-seed. The style-transfer experiment is single-seed, single-judge. The MATH-500 numbers select the best checkpoint using the same evaluations they report.

**Open questions this leaves me with.**
- Does the forward-KL robustness hold when the teacher is from a *different* family (no shared tokenizer, so you cannot compare token distributions directly and must fall back on sampled estimators)?
- The bounded-log-ratio corollary predicts reverse-KL instability should localise at prefixes with extreme student–teacher log-ratios. That is a directly testable diagnostic nobody has run — you could log the ratio range per prefix and check it predicts the gradient spikes.
- The "low forward KL hides high entropy" effect ($D_{\text{KL}}=\mathrm{CE}-H$) is a trap for anyone monitoring distillation loss as a health metric. **Log cross-entropy and teacher entropy separately.** The loss curve lies, in the same way [[Denoising Objective#^loss-curve-lies|diffusion loss curves]] lie.
- Why did off-policy checkpoints prepare better for RLVR despite worse starting accuracy? Possibly the mode-covering, broad-output-distribution checkpoints leave more for exploration to work with — which would connect to [[Exploration vs Exploitation]] and the pass@$k$ result. Untested here.

**One tempting confound they checked and I would have missed.** Teacher temperature was 0.5 and student 1.0 throughout, which means OnPD and OffPD used different sampling temperatures. Appendix A.12 swaps them on MedReason: forward KL moves ≤0.3 points at the low LR, and best achievable performance is unchanged for both methods. Also, the $\lambda$ spectrum itself mixes both policies at $T=1.0$, so $\lambda=\pm1$ gives a temperature-matched OnPD/OffPD comparison for free.

## Links

Related: [[KL Divergence]] · [[Fine-Tuning]] · [[On-Policy vs Off-Policy]] · [[Distilling the Knowledge in a Neural Network]] · [[Distillation]] · [[GRPO]] · [[Training language models to follow instructions with human feedback]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[Do We Really Need KL Divergence for On-Policy Distillation of Large Language Models]] · [[On-policy Distillation with Verifiable Reward]] · [[Learning from Teacher Continuations at Student States]] · [[What Does Privileged Information Add to On-Policy Self-Distillation]] · [[Mode Collapse]] · [[Cross Entropy]] · [[Maximum Likelihood]] · [[On the difficulty of training Recurrent Neural Networks]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Adam- A Method for Stochastic Optimization]] · [[Cyclical Learning Rates for Training Neural Networks]] · [[The Lottery Ticket Hypothesis]] · [[Derivative]] · [[Sampling Parameters]] · [[Exploration vs Exploitation]] · [[LoRA]] · [[Test-Time Compute]]

New topics worth writing: catastrophic forgetting as a measurable quantity, parameter-update sparsity as a diagnostic, total variation distance, Lipschitz continuity of gradient operators, contrastive decoding, score-function gradient estimators, mode-covering vs mode-seeking as a design axis, learning-rate confounds in post-training comparisons, Countdown as an RL benchmark, pass@k as a coverage metric, multi-stage post-training pipelines (SFT → distillation → RLVR), the learning-forgetting cliff
