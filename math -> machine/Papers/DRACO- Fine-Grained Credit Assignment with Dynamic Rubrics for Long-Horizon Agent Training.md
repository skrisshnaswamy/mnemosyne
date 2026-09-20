---
title: "DRACO: Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training"
authors: ["Shubham Gandhi", "Saurabh Goyal", "Kiran Kate", "Yara Rizk"]
year: 2026
arxiv: "2609.04094"
url: https://arxiv.org/abs/2609.04094
priority: Good-To-Read
read_on: 2026-09-16
tags: [paper, llm, rl]
---
## The Core Idea

Reinforcement learning on agents usually works because someone wrote a checker. Unit tests for code, exact-match for maths, a task-pass script for a tool-using agent. That checker gives a clean terminal reward that is hard to fake. Most real agent jobs — customer support, research assistants, back-office workflows — have no such checker, and writing one is often as hard as doing the task.

This paper trains in what it calls the **outcome-blind** setting: during training the reward never looks at task success or a gold answer, not even once. The reward comes entirely from written criteria ("did the agent paginate to the end?", "did it deduplicate artists before following them?") graded by a frozen LLM judge.

Two things make that work where it normally would not.

**First, the rubric is regenerated during training, per task and per rollout.** A rubric written once for a whole task distribution goes stale: after 25 steps of training the policy passes nearly every criterion, the pass rate pins at ~95%, and the reward stops telling any two rollouts apart. Criteria generated fresh each step, and explicitly kept only if *some* rollout in the group failed them, keep producing a spread.

**Second, the single rubric score is smeared back over the individual steps that earned it.** A trajectory on AppWorld is tens of interdependent tool calls. Giving every token of a 20-turn rollout the same advantage is the classic credit-assignment problem — a successful rollout still contains lucky and wasted steps, a failed one is mostly correct steps. The trick here is that the judge already says *which* steps each criterion's verdict was about. That citation is free attribution. Turn it into per-step weights in closed form. No learned attribution head, no per-step judge call, no value network.

> [!NOTE] Outcome-blind RL
> Training where the reward signal consults no ground-truth success check and no gold answer at any point. Evaluation may still use a verifier; training may not. ^outcome-blind

The headline: on AppWorld `test_normal`, Task Goal Completion goes 69.4 → 85.3 over the untrained Qwen3.6-27B, and — the surprising part — **beats GRPO trained on AppWorld's actual unit tests** (80.0) by 5.3 points, while never touching them.

## The Methodology

### The backbone: GRPO

Group Relative Policy Optimization. For task $x$, sample $G$ trajectories, give each a scalar reward $R_i$, and standardise within the group:

$$A_i=\frac{R_i-\mathrm{mean}(\{R_j\})}{\mathrm{std}(\{R_j\})}$$

The group mean *is* the baseline, so no critic is trained (unlike [[Proximal Policy Optimization Algorithms|PPO]] with a value head, or [[High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)|GAE]]). Then every token of $\tau_i$ gets the same multiplier on its log-prob gradient:

$$\nabla_\theta\mathcal{J}=\mathbb{E}\Big[\sum_{t=1}^{N_i}A_i\,\nabla_\theta\log\pi_\theta(y_{i,t}\mid y_{i,<t},x)\Big]$$

DRACO changes two things: where $R_i$ comes from, and the fact that one $A_i$ multiplies all $N_i$ tokens.

### Part 1 — dynamic rubrics

Three judge calls, then a merge, per training step:

1. **From the task alone.** Judge reads the agent system prompt plus the first user message, emits ≤15 criteria. Cached across the group. The prompt explicitly bans "table stakes" criteria ("uses only documented APIs", "wraps code in tags") because every rollout passes those and they carry zero signal.
2. **From each rollout.** Judge reads one trajectory plus the existing criteria, adds ≤10 more. Instructed to *anchor each new criterion on something this rollout actually got wrong*, then generalise the wording without softening the anchor.
3. **Merge.** All $G$ trajectories are shown in full, candidates are deduplicated down to ≤24. Criteria must be MECE (mutually exclusive, collectively exhaustive), so one mistake is not punished twice — the reward in the next step is a *rate*, so overlapping criteria double-count.
4. **Discriminative dropout.** Keep a criterion only if at least one group member failed it. Anything the whole group passes contributes nothing after group standardisation, so it is deleted.

Scoring: the frozen judge returns `pass` / `fail` / `not applicable` per criterion — deliberately coarse — plus a justification and **the step numbers responsible**. With $p_i$ passes and $f_i$ fails:

$$R_i=\frac{p_i-f_i}{p_i+f_i}$$

Normalising by verdicts actually cast, not by $K$, keeps trajectories comparable when different numbers of criteria apply. A criterion whose trigger never fired must score 0, not +1 — the prompt is explicit that a never-tested criterion cannot earn a free pass.

### Part 2 — the credit rule

One step = one agent turn = one emitted `<code>` block. Tokens outside any step (turn glue, tool-result echoes) are "gap" tokens: advantage 0, excluded from the accounting.

Let $p_j, f_j$ count the passed and failed criteria that **cite step $j$**.

$$Q_j=\frac{p_j}{p_j+f_j}\in[0,1]$$

A step nobody cited inherits $\bar{Q}$, the mean over cited steps.

The sign of $A_i$ decides whether the whole trajectory is reinforced or suppressed; credit only decides *where inside* that push lands:

$$w_j=\begin{cases}Q_j,& A_i\ge 0\ \text{(reinforce good steps)}\\ 1-Q_j,& A_i<0\ \text{(suppress bad steps)}\end{cases}$$

With $n_j$ tokens in step $j$ and $N=\sum_k n_k$:

$$a_j=A_i\cdot\frac{N\,w_j}{n_j\sum_k w_k}$$

and every token in step $j$ gets $a_j$ in place of $A_i$.

The thing to actually remember is what this equalises. Step $j$'s **total** contribution is

$$n_j a_j = A_i N\frac{w_j}{\sum_k w_k}$$

which depends on the quality weight and **not on length**. A step earns influence by being judged good, not by being verbose. The $1/n_j$ factor just spreads that fixed total over however many tokens the step happens to contain. Summing over steps gives $\sum_j n_j a_j = A_i N$ — exactly the total push baseline GRPO would have applied to those same $N$ tokens.

> [!NOTE] Total-push conservation
> Reallocation never inflates or deflates a trajectory's overall influence. It moves existing influence onto the steps that earned it. The *scalar sum* is conserved; the gradient vector is not, which is the entire point. ^push-conservation

Because all $w_j\ge 0$, no $a_j$ ever flips sign relative to $A_i$. A wrong judge can misallocate but cannot invert a reinforce into a suppress. If all verdicts are unanimous, all weights are equal and the rule collapses to baseline GRPO. If a winner's weights all collapse to zero, the normaliser vanishes and the implementation falls back to uniform $A_i$.

The authors list seven exact properties (Appendix E.9): conservation, no-signal-equalises, sign preservation, monotone correctness, length independence, reward-scale invariance, winner/loser symmetry. Note what is *not* claimed: invariance to how finely turns are cut into steps. Splitting a step of weight $w$ into two adds $w$ to the normaliser and changes everyone's share. AppWorld fixes step boundaries by environment, so granularity is not a free knob here — but it would be in another domain.

### Training setup

- Policy: Qwen3.6-27B (all ablations) and Qwen2.5-32B-Instruct.
- [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]] rank 16, $\alpha=32$, all linear modules. Base weights frozen. So every reported gain is a rank-16 update.
- 16 prompts per step × $G=6$ = 96 rollouts. 100 steps (20 epochs), 8×H100.
- AdamW, lr $5\times10^{-5}$ **constant, no warmup**. One PPO epoch per batch, so updates are fully on-policy and the clip range never fires. [[KL Divergence|KL]] penalty in the loss, coefficient 0.01, Schulman's $k_3$ estimator.
- Judge: GPT-5.4, temperature 0.1, for every rubric operation. ~20 judge calls per group of 6.
- Trained **only** on the 90 AppWorld training tasks. τ-bench is zero-shot transfer.
- Checkpoints are *not* selected on held-out performance — that would leak implicit supervision. They report the mean over the final three checkpoints.

## Ablation Studies and Experiments

All numbers are $p^1$ (mean success over 3 runs) unless noted. $p^k$ = success in **all** $k$ trials (consistency), pass@$k$ = success in at least one (discovery).

### Main table, Qwen3.6-27B, AppWorld `test_normal` (168 tasks)

| Setting | TGC $p^1$ | SGC $p^1$ | TGC $p^3$ |
|---|---|---|---|
| Base (untrained) | 69.4 | 41.1 | 47.6 |
| Outcome reward (unit tests, GRPO) | 80.0 | 59.3 | 63.3 |
| Random reward | 74.0 | 50.0 | – |
| w/o dynamic & credit (static rubric) | 81.1 | 59.9 | 64.7 |
| w/o dynamic (static + credit) | 81.9 | 60.7 | 65.5 |
| w/o credit (dynamic rubrics) | 82.1 | 64.9 | 66.3 |
| **DRACO** | **85.3** | **70.6** | **72.8** |
| DRACO, self-judge | 81.1 | 62.7 | 65.5 |

`test_challenge` (417 tasks, unseen apps): base 49.7 → DRACO 61.5. τ-bench Banking: 15.8 → 20.4. Qwen2.5-32B-Instruct: 35.7 → 62.9 TGC, closing most of the gap to SALT (66.2), which *does* use the ground-truth reward.

### The ablation that actually matters

Neither component is worth much alone. Static rubric + credit: **+0.8** TGC. Dynamic rubrics, no credit: **+1.0** TGC. Both together: **+4.2** TGC and **+10.7** SGC over the static baseline, widening to +8.1/+14.3 at $p^3$.

On `test_challenge` the interaction changes *sign*: step credit on a fixed rubric **costs 3.7 TGC** at $p^3$; on per-trajectory rubrics it adds +1.4.

Appendix B.3 explains this exactly, and it is the best part of the paper. The credit rule only does anything when a trajectory's verdicts are **mixed**. Unanimous verdicts ⇒ equal weights ⇒ baseline GRPO. So you can count how often credit is inert. On the static rubric, the inert share climbs from 23.6% of rollouts (steps 1–10) to **91.9%** (steps 76–100) — because the policy learns to pass every fixed criterion. On DRACO it starts at 29.7% and only reaches 56.5%. Averaged over training: **76.4% inert (static) vs 48.1% (dynamic)**.

The static arm does not attribute badly. For most of training it does not attribute *at all*.

### Gains are in consistency, not discovery

TGC $p^3$ rises by 25.2 points; pass@3 rises by only 3.9. The untrained model could already solve most of these tasks *once*. Training made the successes repeatable. That is a different thing from "the model got smarter" and worth separating when you read agent RL results.

### The reward curve

Static rubrics hit the mid-90s pass rate by step 25 and sit there (94.8% / 95.8% mean) — saturated, dead signal. Dynamic rubrics sit at 66.9% / 74.2% and keep moving. This is by design: the generator is prompted for criteria the group will likely fail, and dropout deletes the rest.

Crucially, re-scoring DRACO's rollouts against the *static* criteria it never trained on recovers **91.3%**. So dynamic rubrics subsume the static ones rather than optimising something unrelated. Appendix B.2 confirms it from the generator's side: 84.4% of the 6,245 distinct criteria were written for a single task, but the 18 task-general ones cluster into exactly four families — secrets, pagination, stopping after completion, error recovery — which are four of the 21 hand-written static criteria, rediscovered.

The task-specific ones are what discriminate: **26.7%** of DRACO's applicable verdicts are failures vs **5.5%** for the static set, and `not applicable` drops from 10.7% to 1.9%.

### Self-judge: the policy grades itself

Replace GPT-5.4 with the policy model, thinking on, score each trajectory $k=3$ times, a criterion passes only if all three calls pass it. Judge cost for 100 steps drops from **$1607 to $316** (5.1×). It still beats the verifier-trained reference on `test_normal` (81.1/62.7 vs 80.0/59.3) and gets the **best τ-bench SR of any setting (21.1)**.

Appendix B.1 replays 60,689 archived judge calls through the policy checkpoint (single call, no thinking — a lower bound). Three findings:

- **Applying a rubric**: 89.4% agreement with GPT-5.4, against 72.0% for a judge that passes everything. Errors are asymmetric: it passes 30.4% of criteria the frontier judge failed, but fails only 1.3% of those it passed. Lenient ≈ 24× more likely than strict. Leniency *grows* over training, 22.1% → 34.9% false-pass rate.
- **Writing criteria**: recall 0.21 against a self-consistency ceiling of 0.43. It writes 3.8 criteria per rollout vs GPT-5.4's 5.1, and misses most of what the frontier judge writes.
- **Merging**: this is where it breaks. Recall 0.80 (keeps the right criteria) but precision 0.44, because it keeps **15.3 criteria per group where GPT-5.4 keeps 6.3**. The surplus are criteria every rollout passes, so the discriminative share drops from 46.5% to 31.3%.

Caveat the authors flag: the replay is pinned to archived prompts, so the self-model's generation *extends* GPT-5.4's proposals and its merge consumes GPT-5.4's candidate pool. A fully self-judged run would compound weaker candidates into a weaker merge, which is not measured.

### Things that did not work / did not move

- **Random rewards.** Reported elsewhere to help in some RLVR settings. Here: 74.0 TGC / 50.0 SGC, well below DRACO's 85.3/70.6. So the rubric signal is doing real work, not just acting as a noise-driven exploration bonus.
- **Step credit on a static rubric** hurt on `test_challenge` (−3.7 TGC at $p^3$).
- **Self-judge without thinking and with $k=1$**: 79.3/59.3, about 2 points below the $k=3$ variant across the board, and 6.7 TGC worse on `test_challenge`. The unanimity rule is specifically targeting the lenient-false-pass failure mode.
- **τ-bench turn counts went up** for every setting, not down. On a held-out benchmark the untrained model's shorter episodes reflect giving up earlier, not finishing sooner.

### Efficiency, which is a nice surprise

DRACO shortens AppWorld `test_normal` episodes from 18.7 to 14.7 turns and cuts evaluation cost from $10.77 to $8.27 while scoring 15.9 points higher. It is solving more tasks in fewer turns, not buying accuracy with longer rollouts.

Against frontier models on the same split: claude-opus-4-7 gets 93.5 TGC for $32.71; DRACO on a 27B model gets 85.3 for $8.27. It beats DeepSeek-V4-Flash (284B) by 14.5 TGC and gpt-oss-120b by 41.3.

Figure 3 shows *how* rollouts end. Early in training, rollouts die from length overflow, turn-budget exhaustion, server errors, or no action at all. Both dynamic settings converge on "the agent almost always submits an answer" — but dynamic-without-credit oscillates for the first 30 steps, sometimes below 30% submitted, while DRACO settles within 40.

## Worth Remembering

**The honest limitation, stated by the authors.** There is no independent signal against which to check whether the criteria describe the task faithfully. A judge that is internally consistent can be systematically wrong, and the experiments cannot separate the two. They report no chance-corrected agreement with human annotators. The same gap exists one level down: end-task performance does not prove credit landed on the right steps. A redistribution that credits the *wrong* steps can still improve the policy, and one that credits the right steps can fail to. Compare [[Troubling Trends in Machine Learning Scholarship|the mathiness critique]] — the seven formal properties are real and provable, but they constrain the *rule*, not the *judge*.

**The judge defines the objective, it does not estimate it.** This is their closing line and it is the right frame. Unlike a [[Training language models to follow instructions with human feedback|reward model]] trained to approximate human preferences, or [[Constitutional AI- Harmlessness from AI Feedback|RLAIF]] where the constitution is fixed, here the criteria are regenerated each step by the same process being trained against. There is no fixed target to converge to. Any bias in the judge is inherited with no verifier to catch it — a point they list explicitly under ethics.

**Training-time variance is not characterised.** Discriminative dropout makes the surviving criterion set a function of the sampled group, so the same prompt is scored against different criteria at different points in training. Their three runs are inference-time repeats of fixed checkpoints. Nobody re-ran training with a different seed.

**Thin evidence per step.** The median step is cited by exactly one criterion (mean 1.95); only 0.6% are cited by ten or more. So most cited steps take $Q_j\in\{0,1\}$ from a single verdict. The rule is deliberately literal about the judge's attribution rather than hedging toward neutral. That is a design choice with a cost.

**Practical caveats if you wanted to run this.**
- The step boundary is fixed by the environment (one emitted code block). In a domain with no natural turn boundary you would be choosing granularity, and P5 does *not* hold under re-cutting.
- Judge cost dominates. $1607 per 100 training steps for 96 rollouts/step with a frontier judge. The self-judge path is 5× cheaper and roughly as good on AppWorld, but the merge stage is where it degrades — you might use a frontier judge only for the merge.
- Rubrics saturate. If you write criteria once and freeze them, expect the reward to stop discriminating within ~25 steps and the credit component to silently switch itself off.
- The "winner" branch keys on the sign of $A_i$, i.e. group-relative reward, **not** task success. In the worked example a rollout with $R_i=-0.2$ (three of five criteria failed) is a "winner" simply because its group scored worse. The reward never observes whether the task passed.

**Connections.** The rubric-as-reward line runs through [[Small Language Models as Judges for Rubric-Based Reinforcement Learning]] and [[Constitutional AI- Harmlessness from AI Feedback]]. The credit-assignment problem is the same one that motivates [[High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)|GAE]] and the delayed-reward discussion in [[Markov Decision Process]] — except here the "value estimate" is replaced by a text judge's citations, which is cheaper and completely unverified. The "make successes reliable rather than discover new ones" result echoes what [[Is Next-Chunk Reasoning RL Really Better than SFT- Revisiting Training Strategies under no-CoT Data|entropy-not-difficulty]] and other RL-on-LLM papers keep finding: RL sharpens what the base model could already sometimes do.

**Open question worth chasing.** They say it themselves: how good does the judge have to be? The self-judge result suggests "less good than you'd think, if you force unanimity across $k=3$ calls" — but the leniency drift over training (22% → 35% false passes) hints at a reward-hacking pressure that a longer run would expose.

## Links

Related: [[Proximal Policy Optimization Algorithms]] · [[Training language models to follow instructions with human feedback]] · [[Small Language Models as Judges for Rubric-Based Reinforcement Learning]] · [[High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)]] · [[Markov Decision Process]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Constitutional AI- Harmlessness from AI Feedback]] · [[KL Divergence]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[Direct Preference Optimization (DPO)]] · [[Dynamic Important Example Mining for Reinforcement Finetuning]] · [[Troubling Trends in Machine Learning Scholarship]]

New topics worth writing: GRPO (Group Relative Policy Optimization), RLVR and verifiable rewards, AppWorld benchmark, τ-bench, LLM-as-a-judge reliability vs validity, temporal credit assignment in LLM agents, pass@k vs pass^k as agent metrics, reward hacking under learned judges
