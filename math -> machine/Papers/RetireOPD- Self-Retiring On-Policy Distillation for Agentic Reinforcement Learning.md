---
title: "RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning"
authors: ["Yu et al."]
year: 2026
arxiv: "2609.20784"
url: https://arxiv.org/abs/2609.20784
priority: Good-To-Read
read_on: 2026-09-27
tags: [paper, llm, rl, vision]
---
## The Core Idea

When you train an LLM agent with reinforcement learning, it gets **one number** at the end of a long episode: did you finish the task or not. Hundreds of tokens, one scalar. That is a terrible teaching signal — the classic [[Credit Assignment]] problem.

A popular patch is **on-policy distillation** (OPD): run a second copy of the model that gets to see extra help — a retrieved "skill" hint, a cheat sheet — and make the student match that teacher's token-by-token probabilities on the student's *own* trajectories. The student never sees the hint, so it has to absorb the behaviour into its weights. Dense supervision, no extra context at inference.

This paper breaks two assumptions that everybody was making about that recipe.

**One: giving a model privileged information does not make it a good teacher.** In the standard setup (OPSD), teacher and student are the *same* weights with different prompts. But the gradient updates only ever come from the skill-free student objective. So nobody ever trains the model to actually *use* the skill text. Result: prompting Qwen2.5-7B with the skills gets **23.4%** success on ALFWorld — worse than vanilla GRPO at 81.2%. The "teacher" is a worse agent than its student. Scale does not fix it.

**Two: teacher supervision has an expiry date.** Track the teacher–student gap during joint GRPO+OPD training. It shrinks, then it *widens again*. What is happening: once the student has internalised whatever the skills taught, reward optimisation starts pushing it toward actions the teacher does not like. The two gradients now fight. Keep matching the teacher and you get pinned near the teacher's ceiling.

> [!NOTE] Adaptive Retirement
> Instead of a fixed schedule for when to stop distilling, read it off the training signal. Drop the teacher when (a) the teacher–student log-prob gap stops shrinking and (b) the student has reached a set fraction of the teacher's success rate. Then continue with RL alone. ^adaptive-retirement

The reason a fixed schedule cannot work: the crossover point lands anywhere from step 50 to step 90 depending on model size and task. One annealing curve is too early somewhere and too late somewhere else.

The payoff: the student ends up **beating its own teacher** in every single setting. On Qwen2.5-3B / ALFWorld, teacher 79.7%, student 93.8%.

## The Methodology

Three stages. Same base model, same architecture for teacher and student.

**Stage 1 — build a teacher that can actually use the hint.**

Take the base model, feed it the skill context $c^+$, and train it with [[GRPO]] on environment reward:

$$\phi^* = \arg\max_\phi \; \mathbb{E}_{x \sim \mathcal{D},\, \tau \sim \pi_\phi(\cdot \mid x, c^+)}\big[R(\tau)\big]$$

Then **freeze** it. This is the whole fix for failure mode one — the teacher is now a policy that has been rewarded for exploiting its cheat sheet, not just handed one. Skill bank comes from SkillRL; retrieval is plain keyword matching.

The numbers on this stage alone: unoptimised skill-prompted 3B = 28.9%, after reward optimisation = 79.7%. The 7B goes 23.4% → 90.6%.

**Stage 2 — joint training of a skill-free student.**

$$\mathcal{L}_{\text{student}}(\theta) = \mathcal{L}_{\text{GRPO}}(\theta) + \lambda \mathcal{L}_{\text{OPD}}(\theta), \qquad \lambda = 0.01$$

GRPO part is standard: sample $G=8$ trajectories per task, normalise rewards within the group to get the advantage

$$\hat{A}^{(i)} = \frac{R(\tau^{(i)}) - \text{Mean}(\{R\})}{\text{std}(\{R\})}$$

then the usual clipped ratio objective (see [[PPO]] for the clip).

OPD part is reverse [[KL Divergence]] between student and frozen teacher, evaluated on the *student's* trajectory prefixes:

$$\mathcal{L}_{\text{OPD}}(\theta) = \mathbb{E}\left[\frac{1}{|y|}\sum_{t} D_{\text{KL}}\big(\pi_\theta(\cdot \mid x, y_{<t}) \,\|\, \pi_T(\cdot \mid x, c^+, y_{<t})\big)\right]$$

Computing that exactly means summing over the whole vocabulary at every position. Too expensive. They use a one-sample Monte Carlo estimate — just score the token the student actually emitted under both models:

$$\Delta_t = \log \pi_T(y_t \mid x, c^+, y_{<t}) - \log \pi_\theta(y_t \mid x, y_{<t})$$

Because $y_t$ was drawn from $\pi_\theta$, $-\Delta_t$ is an unbiased single-sample estimate of the reverse KL at that position. The teacher never generates its own trajectory — it only *scores* the student's. One forward pass per batch.

**Stage 3 — the retirement trigger.**

Chop training into windows of $W=5$ steps. Average $\Delta_t$ over tokens and trajectories to get a per-step **alignment progress** $K_n$, then average over the window:

$$K_n = \frac{1}{G}\sum_{i=1}^{G} \frac{1}{|y_n^{(i)}|}\sum_{t} \Delta_t^{(i)}, \qquad \bar{K}_m = \frac{1}{W}\sum_{n=mW}^{mW+W-1} K_n$$

Two monitored quantities:

$$\rho_m^{(K)} = \frac{\bar{K}_m - \bar{K}_{m-1}}{\bar{K}_m} \qquad\qquad \eta_m = \frac{SR_m + SR_{m-1}}{2\,SR_T}$$

$\rho$ negative means the gap is still closing. $\rho \ge 0$ means it has stalled or reversed. $\eta$ is the student's success rate as a fraction of the teacher's, smoothed over two windows.

Retire at the first window where both fire:

$$m^* = \inf\{m \ge 2 : \rho_m^{(K)} \ge \delta \;\wedge\; \eta_m \ge \gamma\}$$

Defaults: $\delta = 0$, $\gamma = 0.9$. After that, $\lambda$ effectively goes to zero — pure GRPO, and no more teacher forward passes, so training gets *cheaper* after retirement.

**Why the two conditions, not one.** $\rho$ alone can fire on noise early on when the student is bad. $\eta$ alone does not tell you whether the gradients are fighting. You need both.

**The first-order argument (Appendix A).** Let $g = \nabla_\theta J$ (reward direction) and $h = \nabla_\theta D$ (discrepancy direction). Joint update is $\theta^+ = \theta + \eta(g - \lambda h)$. Then

$$J(\theta^+) - J(\theta) = \eta\big(\|g\|^2 - \lambda\langle g, h\rangle\big) + O(\eta^2)$$

So $\langle g,h\rangle < 0$ → distillation *helps* reward. $\langle g,h\rangle > 0$ → it hurts. And the discrepancy itself moves as

$$\frac{d}{dt}D(\theta) = \langle g,h\rangle - \lambda\|h\|^2$$

A non-decreasing $D$ therefore implies $\langle g,h\rangle \ge \lambda\|h\|^2 > 0$: conflict. That is the link from "the measured gap stopped shrinking" to "the gradients are opposed". Retiring buys you $\eta\lambda\langle g,h\rangle$ of extra reward per step.

**Setup:** Qwen2.5-Instruct at 1.5B / 3B / 7B. ALFWorld and WebShop. LR $1\times 10^{-6}$, batch 16, 150 steps, KL penalty 0.01, validation set 128.

## Ablation Studies and Experiments

**Main numbers** (ALFWorld success rate / WebShop accuracy):

| Model | GRPO | GiGPO | Skill-GRPO (teacher) | GRPO+OPD | RetireOPD |
|---|---|---|---|---|---|
| 1.5B | 72.8 / 56.8 | 86.7 / 65.0 | 81.2 / 67.9 | 87.5 / 69.9 | **89.8 / 75.8** |
| 3B | 75.0 / 63.3 | 92.2 / 69.5 | 79.7 / 64.8 | 82.8 / 74.2 | **93.8 / 77.3** |
| 7B | 81.2 / 72.6 | 90.8 / 72.8 | 90.6 / 78.9 | 92.2 / 81.2 | **95.3 / 84.4** |

Gains over GRPO: +17.0 / +18.8 / +14.1 on ALFWorld, +19.0 / +14.0 / +11.8 on WebShop. Beats GiGPO everywhere. Beats its own teacher everywhere.

**The three-way branch experiment — this is the important one.** From the retirement checkpoint (76.6% success), continue three ways on 3B/ALFWorld:

- Keep GRPO+OPD → plateaus, discrepancy keeps widening.
- Switch to pure OPD → gap closes nicely, performance **flat or degrades**.
- Drop OPD, pure GRPO → 76.6% → **93.8%**.

So the teacher was actively the bottleneck. Closing the gap and improving the task are, after the crossover, different objectives.

**Retirement criteria ablation** (3B, ALFWorld):

| Setting | Retires at | Success |
|---|---|---|
| Full RetireOPD | 60 | **92.2** |
| w/o alignment stagnation ($\rho$) | 50 | 89.0 |
| w/o relative competence ($\eta$) | 100 | 89.0 |
| w/o retirement | never | 82.8 |
| w/o OPD (GRPO only) | 0 | 75.0 |

Both single-signal variants land at 89.0 — one retires too early, one too late, same cost. Complementary, as claimed.

**Teacher quality matters less once you retire.** With OPD kept throughout, a 7B teacher gives 89.8% vs a 3B teacher's 82.8% — a 7-point dependence on teacher quality. With adaptive retirement, 92.2% vs 91.4% — the gap nearly vanishes. Retirement decouples you from teacher quality, which is a genuinely useful robustness property.

**Adaptive vs linear annealing.** They reimplemented ATOD's schedule ($\lambda_{KL} = 1 - 0.9p(s)$, $\lambda_{RL} = 0.1 + p(s)$, $p(s)=\min(s/80,1)$). Annealing rises faster early then plateaus; adaptive keeps climbing. Teacher utility does not decay on a smooth curve — it has a knee.

**Threshold sensitivity.** Vary $\gamma \in [0.8, 1.0]$ and $\delta \in \{0, 0.03, 0.05\}$: retirement step moves from 55 to 95, success stays in 89.1–92.2. Retirement step stays within $\pm 5$ of default for $\gamma \in [0.80, 0.96]$, $\delta \in [-0.10, 0.04]$. Not a knife-edge hyperparameter, which is the main thing you want to know before using it.

**What did not work:**

- **Skill-Prompt at inference** is catastrophic. 1.5B: 20.8 score / **1.6%** accuracy on WebShop vs vanilla's 5.5%. Small models handed a cheat sheet get *worse*. The hint is context the model cannot use.
- **OPSD (shared-weight self-distillation)** is the weakest distillation baseline: 14.1 / 28.1 / 32.8 ALFWorld across the three scales. And GRPO+OPSD (72.7 / 78.1 / 79.7) is worse than plain GRPO+OPD everywhere.
- **Distilling from a bigger unoptimised model** fails badly. OPD from a 7B teacher into a 3B student: 38.3% ALFWorld, 2.3% WebShop accuracy. OPD from 14B into 7B: 64.8 / 4.7. Bigger teacher, no reward training, worse than nothing on WebShop.
- **Pure OPD** from even a *good* teacher caps out near the teacher (74.2% for 3B vs teacher's 79.7%). No reward signal, no chance to exceed.

**Case study (Appendix C).** On a trajectory where the agent holds lettuce at the sink and should `clean` it: the trained teacher gives that token log-prob $-0.826$; the OPSD teacher gives $-16.625$. Same privileged text, wildly different ability to convert it into a signal. And Figure 9 shows the conflict directly: a student action that leads to task success (`go to microwave`) gets strong *negative* teacher score.

## Worth Remembering

**The transferable idea is not RetireOPD, it is "read the schedule off the gradients".** Any time you blend an auxiliary objective with your main one — a KL leash, a distillation term, an imitation prior — the same argument applies: $\frac{d}{dt}D = \langle g,h\rangle - \lambda\|h\|^2$. If your regulariser's value stops falling, it is fighting you. That is a measurable quantity you almost certainly already log. Compare to [[RLHF]]'s fixed KL coefficient and to [[Imitation Learning]]'s DAgger-style schedules.

**Teacher supervision is scaffolding, not a target.** Framing it that way is a small rhetorical move with real consequences — it makes "surpass the teacher" the expected outcome rather than a surprise.

**Practical caveats:**

- The reverse-KL estimate is single-sample. High variance per token, and reverse KL is [[KL Divergence#Reverse KL — $D_{KL}(Q \parallel P)$, "**mode-seeking**"|mode-seeking]] — the student is encouraged to concentrate on one of the teacher's modes rather than cover them all. That is probably fine here (you want decisive actions) but worth knowing.
- Cost: stage 1 is a full extra RL run. RetireOPD is roughly 2× the compute of GRPO, mitigated slightly by the teacher being dropped partway.
- $\rho_m^{(K)} = (\bar K_m - \bar K_{m-1})/\bar K_m$ has $\bar K_m$ in the denominator. As the gap closes $\bar K \to 0$ and this ratio gets numerically unstable. The paper does not discuss it.
- Only two benchmarks, both text-only, both fairly short-horizon. No tool use, no code, no coding agent. Whether the knee is this clean on 50-step web tasks is untested.
- Teacher and student are the *same size*. This is not model compression. It is context internalisation — closer in spirit to [[Distillation#Self-distillation — how a model teaches itself|self-distillation]] and to the privileged-critic pattern.

**Surprising results worth carrying:**

1. A model given helpful hints can perform *worse* than the same model with no hints (1.6% vs 5.5%). Privileged context is only privileged if the policy was trained to consume it. Compare [[Multi-Agent Reinforcement Learning#CTDE — the pattern that makes it work|CTDE]], where the privileged critic is always trained.
2. The two retirement signals, used alone, give *identical* final numbers (89.0) by failing in opposite directions.
3. Adaptive retirement makes the method robust to teacher quality. If you cannot afford a great teacher, retiring early is a substitute.

**Open questions:** Could you retire *per-skill* rather than globally — some behaviours internalise faster than others? Does $\langle g, h\rangle$ measured directly (it is computable) give a cleaner trigger than the discrepancy proxy? And what happens if you re-hire the teacher later with a fresh skill bank — a curriculum of scaffolds rather than one?

## Links

Related: [[GRPO]] · [[Credit Assignment]] · [[KL Divergence]] · [[Distillation]] · [[On-policy Distillation with Verifiable Reward]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[PACT- From Credit Assignment to Critic Alignment]] · [[DRACO- Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training]] · [[PPO]] · [[RLHF]] · [[Imitation Learning]] · [[Reward Function]] · [[Agentic Workflows]] · [[Mind2Dialogue- Training Human-Aware Language Models by Simulating User Mental States]] · [[Best Practice Critic Optimization]] · [[TTPO- Test-Time Policy Optimization]]

New topics worth writing: ALFWorld, WebShop, GiGPO (group-in-group policy optimization), skill banks and skill retrieval for agents, gradient conflict and objective interference, privileged-information learning, adaptive curriculum scheduling from training signals
