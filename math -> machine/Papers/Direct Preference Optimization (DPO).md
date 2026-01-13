---
title: "Direct Preference Optimization (DPO)"
authors: ["Rafael Rafailov", "Archit Sharma", "Eric Mitchell", "Stefano Ermon", "Christopher D. Manning", "Chelsea Finn"]
year: 2023
arxiv: "2305.18290"
url: https://arxiv.org/abs/2305.18290
priority: Must-Read
read_on: 2026-08-25
tags: [paper, llm, rl, theory]
---
## The Core Idea

RLHF as of 2022 was a three-stage machine: fine-tune on demonstrations, fit a separate reward model on human preference pairs, then run [[Proximal Policy Optimization Algorithms|PPO]] against that reward model with a KL penalty. Three networks in memory, sampling from the policy inside the training loop, and all the usual RL instability. See [[Training language models to follow instructions with human feedback]] for the full pipeline.

The insight here: **you never needed the reward model or the RL**. The KL-constrained reward maximisation problem has a known closed-form solution,

$$\pi_r(y\mid x)=\frac{1}{Z(x)}\pi_{\text{ref}}(y\mid x)\exp\!\left(\tfrac{1}{\beta}r(x,y)\right),$$

and you can just *invert* it. Solve for $r$ in terms of $\pi$:

$$r(x,y)=\beta\log\frac{\pi_r(y\mid x)}{\pi_{\text{ref}}(y\mid x)}+\beta\log Z(x).$$

The intractable partition function $Z(x)$ depends only on the prompt $x$, not on the response $y$. And the Bradley-Terry preference model only ever looks at *differences* of rewards between two responses to the same prompt. So $Z(x)$ cancels. What is left is a preference probability written purely in terms of the policy you actually want.

That turns preference learning into a plain binary classification problem — a logistic loss on log-probability ratios. No reward model, no sampling during training, no value function, no PPO clipping, no reward hacking loop.

> [!NOTE] Implicit reward
> The language model *is* the reward model. $\hat r_\theta(x,y)=\beta\log\frac{\pi_\theta(y\mid x)}{\pi_{\text{ref}}(y\mid x)}$ — how much more likely the tuned model finds a response than the frozen starting model. Theorem 1 shows this reparameterisation loses no generality: every Bradley-Terry-consistent reward class contains exactly one function of this form. ^implicit-reward

## The Methodology

**The preference model.** Bradley-Terry says a human prefers $y_1$ over $y_2$ with probability
$$p^*(y_1\succ y_2\mid x)=\sigma\big(r^*(x,y_1)-r^*(x,y_2)\big).$$
Standard RLHF fits $r_\phi$ to this by [[Cross Entropy|negative log-likelihood]]. DPO substitutes the implicit-reward expression and fits the policy instead.

**The loss.** Given a static dataset $\mathcal{D}=\{(x, y_w, y_l)\}$ of prompt, preferred response, dispreferred response:

$$\mathcal{L}_{\text{DPO}}=-\mathbb{E}_{(x,y_w,y_l)\sim\mathcal{D}}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w\mid x)}{\pi_{\text{ref}}(y_w\mid x)}-\beta\log\frac{\pi_\theta(y_l\mid x)}{\pi_{\text{ref}}(y_l\mid x)}\right)\right]$$

That is the whole algorithm. Four forward passes per pair (policy and frozen reference, on $y_w$ and $y_l$), one `logsigmoid`. The reference log-probs can even be precomputed.

The actual PyTorch from the appendix:

```python
pi_logratios  = pi_yw_logps  - pi_yl_logps
ref_logratios = ref_yw_logps - ref_yl_logps
losses = -F.logsigmoid(beta * (pi_logratios - ref_logratios))
```

**What the gradient does.**

$$\nabla_\theta\mathcal{L}_{\text{DPO}}=-\beta\,\mathbb{E}\Big[\underbrace{\sigma\big(\hat r_\theta(x,y_l)-\hat r_\theta(x,y_w)\big)}_{\text{weight: how wrong the implicit reward is}}\big[\nabla_\theta\log\pi(y_w\mid x)-\nabla_\theta\log\pi(y_l\mid x)\big]\Big]$$

Push up the winner, push down the loser — but scaled by a per-example weight that is near 1 when the model currently ranks the pair *backwards* and near 0 when it already gets the pair right. This weight is the load-bearing part. Strip it out and you get the Unlikelihood baseline, which degenerates (see below).

**$\beta$** plays the same role as in the RL objective: strength of the KL leash to $\pi_{\text{ref}}$. Small $\beta$ = free to drift; large $\beta$ = stay close. See [[KL Divergence]].

**Pipeline.** Set $\pi_{\text{ref}}=\pi^{\text{SFT}}$ and initialise $\pi_\theta$ from it too. If no SFT model exists (Anthropic-HH case), build one by maximum-likelihood on the *preferred* completions only — this matters, because the derivation assumes the preference data was sampled from $\pi_{\text{ref}}$, and skipping this step leaves a distribution mismatch.

**Hyperparameters that mattered.** $\beta=0.1$ default ($\beta=0.5$ for TL;DR), batch size 64, RMSprop at lr `1e-6`, linear warmup over 150 steps. The authors stress they barely tuned any of it.

**Tasks and models.**
- *Controlled sentiment*: IMDb prefixes, GPT-2-large SFT, preferences generated synthetically by a RoBERTa sentiment classifier (so ground-truth reward is known).
- *Summarisation*: Reddit TL;DR, GPT-J-6B, human preferences from Stiennon et al.
- *Single-turn dialogue*: Anthropic Helpful & Harmless, 170k dialogues, Pythia-2.8B.

Evaluation is GPT-4 win rate against a reference (test-set human summary, or the chosen response).

## Ablation Studies and Experiments

**Reward–KL frontier (sentiment, 22 runs sweeping conservativeness).** Because the true reward function is a known classifier, they can plot achieved reward against $\text{KL}(\pi\Vert\pi_{\text{ref}})$. DPO's frontier *strictly dominates* PPO's — more reward at every KL budget. It even beats **PPO-GT**, an oracle PPO given the true reward function instead of a learned one. DPO and PPO optimise the same objective; DPO just optimises it better.

**Summarisation (TL;DR, GPT-4 win rate vs human reference summaries).**

| Method | Win rate |
|---|---|
| DPO (temp 0.0) | ~61% |
| PPO (temp 0.0, its best) | ~57% |
| Best-of-$N$ (learned RM) | below DPO's max |
| Preferred-FT | ≈ SFT, no real gain |

DPO is also far more robust to sampling temperature: PPO's quality collapses toward the base GPT-J at high temperature, DPO's stays flat.

**Dialogue (Anthropic-HH).** DPO is the *only* cheap method that beats the dataset's own chosen responses. It matches Best-of-128 (which plateaus around 64–128 samples and is hopeless at serving time — $N$ generations per query). A publicly available PPO-trained HH model could not be made to beat base Pythia-2.8B at any prompt or temperature.

**Out-of-distribution transfer.** Take the TL;DR-trained policies, run them on CNN/DailyMail news articles:

| | temp 0 | temp 0.25 |
|---|---|---|
| DPO | **0.36** | **0.31** |
| PPO | 0.26 | 0.23 |

Notable because PPO gets to use extra *unlabelled* prompts during its RL phase; DPO only ever sees the labelled pairs. The offline-only method still generalises better.

**Human validation of the GPT-4 judge** (25 volunteers, 25 judgments each). Human agreement with GPT-4 was 65–86%; human–human agreement was 65% for DPO-vs-PPO and 87% for PPO-vs-PPO. GPT-4 agrees with humans about as often as humans agree with each other. Head-to-head, humans preferred DPO (temp 0.25) over PPO (temp 0.0) **58%** of the time.

**What did not work:**

- **Unlikelihood** — naively maximise $\log p(y_w\mid x)$ and minimise $\log p(y_l\mid x)$ with a coefficient $\alpha$. It produces garbage: Table 3 shows completions that are literally `"when when when when when..."` repeated to the length limit. Unconstrained likelihood *minimisation* has no floor, so the model destroys itself. This is exactly the ablation of DPO's $\sigma(\hat r_l - \hat r_w)$ weighting term. Dropped entirely from the summarisation and dialogue experiments.
- **Preferred-FT** (supervised fine-tuning on the chosen completion only) barely moves over SFT on summarisation. Just imitating winners is not enough — the contrast with the loser carries the signal.
- **GPT-4 prompt sensitivity.** The plain "which summarises better" prompt (S) made GPT-4 prefer longer, more repetitive summaries than humans do. They added "without including unimportant or irrelevant details… precise and concise" (prompt C), which tracked human judgments better, and used C for headline numbers. Worth remembering as a caveat for anyone doing LLM-as-judge.
- **Failure modes of DPO itself** (Tables 9, 10): DPO's outputs are verbose and sometimes confidently wrong — it invented a WWII "all-inclusive association", and answered "7 plus 2" with a rambling non-answer, losing to a one-token ground truth.

## Worth Remembering

- **Why PPO was unstable, explained by this framework** (§5.2). The RL objective, written properly, contains a term $\beta\log\sum_y \pi_{\text{ref}}(y\mid x)\exp(\tfrac{1}{\beta}r_\phi(x,y))$ — the soft value function of the reference policy. It does not change the optimum, but omitting it makes the [[Simple Statistical Gradient-Following Algorithms (REINFORCE)|policy gradient]] high-variance. Prior work patched this with learned value functions or human-completion baselines (single-sample Monte Carlo estimates). DPO's reparameterisation makes the term cancel exactly, so no baseline is needed at all.

- **Reward function equivalence classes.** Two rewards differing by any $f(x)$ induce identical preferences (Lemma 1) *and* identical optimal policies (Lemma 2). The Bradley-Terry family is under-specified this way. DPO's contribution is picking the unique member of each class for which $\sum_y \pi_{\text{ref}}(y\mid x)\exp(\tfrac{1}{\beta}r(x,y))=1$ — i.e. the one whose induced optimal policy is already a valid normalised distribution.

- **It generalises past pairs.** Appendix A.3 derives the Plackett-Luce version for full rankings of $K$ responses; Bradley-Terry is the $K=2$ case. The same $Z(x)$ cancellation works.

- **This is offline learning.** The dataset is fixed; you never sample from $\pi_\theta$. That is the source of the speed, and also the source of the risk — see [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] on action distribution shift. If your preference pairs came from a model very different from your $\pi_{\text{ref}}$, the derivation's assumption breaks.

- **Admitted limitations.** Largest model tested was 6B — nothing about frontier scale. They flag that they do not know how reward over-optimisation manifests in DPO, and wonder whether the slight late-training dip in Figure 3 (right) is an instance of it. They also do not know whether self-labelling with the DPO policy could exploit unlabelled prompts the way PPO does.

- **Practical caveat.** You hold two copies of the model in memory (policy + frozen reference). Cheaper than PPO's four networks, but not free. Precomputing reference log-probs over the dataset removes the reference forward pass entirely.

- **The $\beta$ dial is your only real knob**, and it is not free — $\beta=0.1$ on dialogue but $\beta=0.5$ on summarisation, and the authors admit they never meaningfully swept it, so the 61% number is probably a floor.

## Links

Related: [[Training language models to follow instructions with human feedback]] · [[Proximal Policy Optimization Algorithms]] · [[KL Divergence]] · [[Cross Entropy]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] · [[BPR- Bayesian Personalized Ranking from Implicit Feedback]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Mode Collapse]] · [[Distilling the Knowledge in a Neural Network]] · [[Auto-regressive models]] · [[Markov Decision Process]] · [[Regularization]]

New topics worth writing: Bradley-Terry model, Plackett-Luce ranking model, Control-as-inference / KL-regularised RL, Partition function and log-sum-exp, Reward hacking and over-optimisation, LLM-as-judge evaluation, Best-of-N sampling / rejection sampling, IPO and KTO (post-DPO preference objectives), Unlikelihood training
