---
title: "Do We Really Need KL Divergence for On-Policy Distillation of Large Language Models?"
authors: ["Wenze Lin", "Jiyuan Long", "Jiale Zhao", "Shenzhi Wang", "Xitai Jiang", "Ce Luo", "Rui Lan", "Qianli Ma", "Fukang Wen", "Hui Wu", "Liyuan Chen", "Shuoling Liu", "Jiangpeng Yan", "Gao Huang"]
year: 2026
arxiv: "2609.33791"
url: https://arxiv.org/abs/2609.33791
priority: Good-To-Read
read_on: 2026-09-30
tags: [paper, llm, rl]
---
## The Core Idea

On-policy distillation (OPD) teaches a small model (the **student**) by letting it write its own answers, then asking a bigger model (the **teacher**) what it would have said at every token. The standard loss is reverse [[KL Divergence|KL]]. This paper says: you do not need the KL. You only need its **sign**.

Replace the whole loss with a coin flip per token. If the teacher gives that token more probability than the student did, reward $+1$. If less, reward $-1$. Throw away the magnitude entirely. Train with plain [[Policy Gradient|policy gradient]]. Call this **BinaryOPD**. It matches or beats real OPD on seven student–teacher pairs across maths and code.

Then a sharper claim. Sort tokens by how much teacher and student disagree, $\ell_t = \log\frac{\pi_T(o_t)}{\pi_\theta(o_t)}$. Over 90% of tokens have $|\ell_t| \le 0.8$ — the teacher and student basically agree. Those tokens do not matter. You can reward them in the **wrong** direction and training still works. What matters is the tiny sliver with $|\ell_t| > 0.8$: fewer than 1.5% of tokens in one setup. Get their sign right and OPD works. Flip their sign and OPD collapses.

> [!NOTE] Directional sufficiency
> In OPD the student does not need to know *how far* to move toward the teacher, only *which way*. Magnitude is recoverable from repetition. ^directional-sufficiency

Why does discarding magnitude cost nothing? Because OPD is a closed loop, not a one-shot regression. If you saw a state once, you would need the size of the gap to know how far to step. But the student keeps revisiting the same region of state space, and each revisit gives a **fresh** sign. Once the student overshoots past the teacher, the sign flips on its own. The loop integrates the correction for you.

The authors back this with a measurement (Appendix A) that is arguably the most interesting part of the paper. They embed every visited state $s_t = (x, y_{<t})$ using the frozen teacher's last-layer hidden state, then look at nearest neighbours **from different prompts**. Mean nearest-neighbour cosine similarity is $0.893$. 18.5% of states have a different-prompt neighbour at cosine $\ge 0.99$. The 2,048 states from step 1 alone already cover 10.4% of the next 29 steps' states at that threshold; growing the library $29\times$ only lifts that to 18.4%. Training keeps producing token-level-novel contexts that are geometrically old. That redundancy is the mechanism that makes sign-only feedback sufficient.

What it unlocks: the loss becomes a **mask over tokens plus a sign**, which composes. You can now ask multiple teachers to vote on the sign, which is the paper's application (C-MOPD). You could not do that cleanly with KL, because you would have to decide how to average incomparable magnitudes from different teachers.

Why it did not exist before: [[Distilling the Knowledge in a Neural Network|Hinton's distillation]] framed the whole enterprise as matching soft targets — the "dark knowledge" in the teacher's full distribution *is* the point. OPD inherited that framing unexamined. Nobody checked whether the on-policy setting, which changed the data distribution, also changed what the loss needed to carry.

## The Methodology

### The baseline being replaced

Standard OPD minimises reverse KL on the student's own rollouts:

$$\min_{\theta}\ \mathbb{E}_{q,\,o\sim\pi_{\theta}(\cdot\mid q)}\left[\sum_{t=1}^{|o|}D_{\mathrm{KL}}\big(\pi_{\theta}(\cdot\mid q,o_{<t})\,\big\|\,\pi_{T}(\cdot\mid q,o_{<t})\big)\right]$$

Differentiate it and you get something that looks exactly like a policy gradient:

$$\nabla_{\theta}\mathcal{L} = \mathbb{E}\left[\sum_t \big(\log\pi_T(o_t) - \log\pi_\theta(o_t)\big)\,\nabla_\theta\log\pi_\theta(o_t)\right]$$

So each token already carries a dense reward, and that reward is the log-ratio $r_t = \log\frac{\pi_T(o_t \mid q, o_{<t})}{\pi_\theta(o_t \mid q, o_{<t})}$. Reverse KL is just policy gradient with a specific per-token reward. This reframing is what makes the paper's move obvious once stated.

### BinaryOPD

Keep the gradient shape, replace the reward with its sign:

$$r_t = \begin{cases} +1, & \pi_T(o_t \mid q, o_{<t}) > \pi_\theta(o_t \mid q, o_{<t}) \\ -1, & \pi_T(o_t \mid q, o_{<t}) < \pi_\theta(o_t \mid q, o_{<t}) \end{cases}$$

Exact ties get $0$ (they essentially never happen in float). Then maximise

$$\max_{\theta}\ \mathbb{E}_{q,\,o\sim\pi_{\theta}}\left[\sum_{t} r_t \log \pi_\theta(o_t \mid q, o_{<t})\right]$$

A $+1$ pushes that token's probability up, toward the teacher. A $-1$ pushes it down, also toward the teacher. That is the entire method. No temperature, no threshold, no tuning.

### The three-group partition

For the second experiment, pick a threshold $\epsilon > 0$ and split tokens by $\ell_t$:

| Group | Condition | Meaning | Share of tokens |
|---|---|---|---|
| **A** | $\ell_t > \epsilon$ | teacher wants it much more | $<1.5\%$ |
| **B** | $-\epsilon \le \ell_t \le \epsilon$ | they roughly agree | $>90\%$ |
| **C** | $\ell_t < -\epsilon$ | student wants it much more | $<5\%$ |

They run $\epsilon = 0.8$ as the headline, with $0.2$ and $0.5$ in the appendix.

Two mirror-image experiments. **Positive**: assign $+1$ uniformly to a chosen subset, $0$ to everything else. **Negative**: assign $-1$ uniformly to a chosen subset, $0$ elsewhere.

The crucial asymmetry: in the positive experiment, $+1$ on Group A moves the student **toward** the teacher (teacher already preferred it). But $+1$ on Group C moves the student **away** (the student already over-weighted it; pushing harder makes the gap worse). So "positive reward on A+B+C" is a deliberate sabotage of Group C.

### C-MOPD

Multi-teacher OPD (MOPD) routes each sample to one domain expert via $d(q)$ and does reverse KL against only that teacher. Problem: updating on a maths sample can quietly damage coding ability, because nobody asked the code teacher whether it agreed.

C-MOPD asks every teacher on every token. With $p_s = \pi_\theta(o_t)$, $p_k = \pi_{T_k}(o_t)$, $\ell_k = \log(p_k/p_s)$, domain teacher index $k^*$, and $d = \operatorname{sign}(\ell_{k^*})$:

$$r_t = \begin{cases} +1, & p_k > p_s\ \forall k \\ -1, & p_k < p_s\ \forall k \\ +1, & \text{conflict},\ d = +1,\ \text{and } \ell_k > -\epsilon\ \forall k \neq k^* \\ -1, & \text{conflict},\ d = -1,\ \text{and } \ell_k < \epsilon\ \forall k \neq k^* \\ 0, & \text{otherwise} \end{cases}$$

Read it as three rules. **Unanimous** → follow everyone. **Conflict but mild** (the dissenting teachers are inside the $\epsilon$ band, i.e. Group B for them) → follow the domain teacher, because Group B tokens tolerate being pushed the wrong way. **Conflict and strong** (a dissenter is in its own Group A or C) → reward $0$, do nothing. The $\epsilon = 0.8$ from the token study is reused directly here; that is the whole point of the connection.

### Setup

- Framework: Verl. LR $1\times10^{-6}$, batch 256, rollout temperature $1.0$, max response 8192 train / 16384 eval, eval temperature $0.7$, top-$p$ $0.95$.
- Teachers are all domain-specialised RL checkpoints, not generically bigger models. Qwen3-4B-Base-RL was made by running [[GRPO]] on DAPO-Math-17k for 3 epochs.
- Seven pairs. Maths: DeepSeek-Distill-Qwen-1.5B ← JustRL-1.5B; Qwen3-1.7B-Base ← Qwen3-4B-Base-RL; Llama-3.2-3B-Instruct ← GT-Llama3.2-3B-MATH; Qwen3-4B-Non-Thinking ← its RL-Math version; Qwen3-30B-A3B-Non-Thinking ← Qwen3-30B-A3B-Instruct-2507. Code: two pairs on Eurus.
- Benchmarks: AIME24/25, AMC, MATH500, Minerva, OlympiadBench for maths; LiveCodeBench v6, HumanEval, MBPP for code. All Avg@8.

## Ablation Studies and Experiments

### BinaryOPD vs OPD

Averages across the six maths benchmarks:

| Pair | Student | Teacher | OPD | BinaryOPD |
|---|---|---|---|---|
| DS-Distill-1.5B ← JustRL-1.5B | 41.0 | 56.5 | 54.7 | **54.9** |
| Qwen3-1.7B-Base ← 4B-Base-RL | 17.3 | 30.9 | 21.2 | **22.2** |
| Llama-3.2-3B-Inst ← GT-MATH | 11.3 | 18.8 | **18.2** | 17.9 |
| Qwen3-4B-NT ← RL-Math | 42.0 | 66.0 | 64.1 | **65.1** |
| Qwen3-30B-A3B-NT ← Instruct-2507 | 48.1 | 70.5 | 54.8 | **55.2** |

Code averages: Qwen3-4B pair 55.4 (OPD) vs 55.2 (Binary); 1.5B pair 51.0 vs 50.6. Effectively identical.

Four out of five maths pairs go to BinaryOPD, all by under a point. The honest reading is a **wash**, which is the finding — you deleted the magnitude and nothing happened. The training-dynamics curves (accuracy, [[Cross Entropy|entropy]], response length, gradient norm) sit on top of each other, so it is not two different paths to the same score; it is the same trajectory.

Worth noticing: the 30B pair recovers only 55.2 of a 70.5 teacher, versus the 4B pair recovering 65.1 of 66.0. Both methods stall equally there, so it is a property of that pair, not of the loss.

### The token-group ablation — the real result

Positive-reward experiment, Qwen3-4B-NT ← RL-Math, $\epsilon = 0.8$:

| Setting | Tokens kept | Maths avg | Verdict |
|---|---|---|---|
| OPD | all | 64.1 | works |
| BinaryOPD | all | 65.1 | works |
| ALL+1 (A) | $<1.5\%$ | 64.5 | **works** |
| ALL+1 (A+B) | $>90\%$ | 64.3 | works |
| ALL+1 (A+B+C) | ~100% | **40.3** | **collapses** |

Read the last two rows against each other. A+B and A+B+C differ only by Group C, under 5% of tokens. Adding them with the wrong sign takes 64.3 → 40.3 — below the 42.0 starting student. Reversing the direction of under 5% of tokens does not degrade training; it destroys it.

Read the third row on its own. **Less than 1.5% of tokens, each given a constant $+1$, reproduces full OPD.** The other 98.5% get exactly zero gradient. That is a remarkable statement about where the teaching signal lives.

The negative experiment mirrors it: C alone works; C+B works and beats GRPO, marginally worse than C alone; C+B+A fails.

### What did not work

- **Group B alone, either sign.** $+1$ on B alone improves but underperforms [[GRPO]] — counted as failure, since a distillation method that loses to reward-free RL has transferred nothing. $-1$ on B alone fails outright. The 90% majority of tokens carries no usable teacher signal in isolation.
- **Wrong sign on high-disagreement tokens.** A+B+C positive, C+B+A negative. Both collapse. This is the one thing you cannot get away with.
- **C-MOPD with $\epsilon = 0.0$.** The fully conservative version — zero out *any* token where teachers disagree at all — only matches MOPD (maths 59.8/52.8/86.2/92.8/34.6/58.9, code 28.2/85.7/53.8). Its Zero Reward Ratio climbs monotonically past 90%, so late in training almost nothing updates. The tolerance band is not a hedge; it is what makes the method work.
- **Verifier alignment, deliberately reversed.** Appendix B is the strangest experiment. They built **Verifier-Conflicted**: on trajectories with the *correct* final answer, keep only tokens the teacher likes *less* and reward them $-1$; on *incorrect* trajectories, keep only tokens the teacher likes *more* and reward $+1$. Correct answers receive only non-positive reward. Incorrect answers receive only non-negative reward. This is the exact inverse of [[Reward Function|RLVR]].

  It does not hurt. On the DeepSeek-1.5B pair it scores **55.1**, the best number in that block, ahead of OPD (54.7), BinaryOPD (54.9) and Verifier-Aligned (54.4). On code it gets the top LiveCodeBench v6 score (29.1). Verifier-Aligned shows no consistent gain anywhere. Training curves are indistinguishable. The many recent papers bolting verifier rewards onto OPD have, on this evidence, no measured mechanism.

### C-MOPD vs MOPD

Qwen3-4B-Non-Thinking student, two teachers (RL-Math, RL-Code), mixed corpus of 25,276 DeepMath + 9,639 CodeContests + 9,579 TACO + 3,462 APPS + 2,596 Codeforces.

| | AIME24 | AIME25 | AMC | MATH500 | Minerva | Olympiad | LCB v6 | HumanEval | MBPP |
|---|---|---|---|---|---|---|---|---|---|
| Student | 25.0 | 16.7 | 58.4 | 81.3 | 27.7 | 42.9 | 25.1 | 80.8 | 48.7 |
| Math teacher | 61.2 | 56.3 | 88.0 | 93.2 | 36.7 | 60.5 | — | — | — |
| Code teacher | — | — | — | — | — | — | 36.0 | 90.6 | 56.2 |
| MOPD | 60.0 | 52.5 | 86.4 | 92.9 | 34.3 | 58.8 | 27.9 | 85.9 | 53.5 |
| C-MOPD $\epsilon{=}0$ | 59.8 | 52.8 | 86.2 | 92.8 | 34.6 | 58.9 | 28.2 | 85.7 | 53.8 |
| C-MOPD $\epsilon{=}0.8$ | **61.3** | **53.9** | **88.0** | 93.0 | **35.5** | **60.4** | **32.9** | **87.8** | **54.3** |

The code side is where it pays: LiveCodeBench v6 goes 27.9 → 32.9, closing most of the gap to the 36.0 teacher, where MOPD had closed less than a third of it. Maths matches or beats the teacher on AMC (88.0) and nearly on OlympiadBench (60.4 vs 60.5). Consistent with the story: MOPD's maths updates were eroding code ability, and consensus gating stops that.

## Worth Remembering

**The cleanest reusable fact.** Reverse KL on the student's own rollouts *is already* policy gradient with per-token reward $\log(\pi_T/\pi_\theta)$. Once you see that, "does the magnitude matter?" becomes an obvious experiment nobody had run.

**The redundancy measurement is more valuable than the method.** The claim that OPD state space saturates almost immediately — step-1's 2,048 states covering 10.4% of the next 29 steps at cosine $\ge 0.99$, and the step-$i$/step-$j$ similarity matrix flat in $[0.138, 0.149]$ with no early/late separation — is a fact about on-policy training that generalises well past distillation. It says the student is not exploring; it is circling. The authors are careful to hedge it: this shows no *progressive separation into new regions under this fixed representation*, not that the state distribution is unchanged. Still, if you are budgeting rollouts for any on-policy method, this is the measurement to copy.

**Limitations they do not fully own.**
- All teachers are domain-RL checkpoints roughly the same size as, or one step up from, the student. Nothing here tests a genuinely large capability gap, where magnitude might carry information that repetition cannot recover.
- $\epsilon = 0.8$ is picked from one pair's dynamics and reused in C-MOPD without a sweep in the multi-teacher setting. The $\epsilon = 0.0$ failure shows the parameter matters a lot.
- The token-group study is reported on essentially one pair for the headline table, with a second pair and two other $\epsilon$ values in the appendix. The 1.5% figure is one measurement, not a law.
- C-MOPD is tested with $K=2$ teachers. The unanimity rule tightens as $K$ grows, so Zero Reward Ratio will rise — and $\epsilon = 0.0$ already showed what happens when it exceeds 90%.
- Only two domains, both verifiable. No test on anything where a teacher's preference is stylistic rather than correct.

**The verifier result deserves a follow-up question.** If Verifier-Conflicted works as well as Verifier-Aligned, the honest interpretation is that the trajectory-level correctness label is doing nothing at these scales, and the gains attributed to OPD+RLVR hybrids come from the OPD half. But a reversed mask also *removes* tokens, and both variants remove roughly half. A proper control would be a random mask of matched density. The paper does not run it.

**Practical caveats for use.**
- BinaryOPD removes the log-ratio from the loss but you still need teacher logprobs on every student token, so the teacher forward pass — the expensive part — does not go away. The saving is conceptual and in robustness, not compute.
- Discarding magnitude removes the natural gradient shrink as the student converges. The loop-feedback argument says the sign oscillation handles it, but this makes the [[Momentum|learning rate]] the only remaining brake. They run a flat $10^{-6}$.
- Good [[Chain of Thought|reasoning]]-relevant diagnostic: dump $\ell_t$ per token and look at where the Group A/C tokens land (their Figure 11 does exactly this). If your distillation is not moving, check whether high-disagreement tokens are being masked out by whatever length or formatting filters you have.
- This is a [[Credit Assignment|credit assignment]] result in disguise: the signal is sparse in token space even though the loss looks dense. Worth comparing against PACT's finding that credit is approximately sparse.

**Connections.** The "only a tiny subset of tokens matters" conclusion lands in the same place as [[1% of Tokens Can Be Enough- On Gradient Estimation in On-Policy Distillation|1% of Tokens Can Be Enough]], which reaches it through Fisher-geometry noise analysis rather than sign ablation — two independent routes to the same claim is much stronger than either alone. The binary reward is structurally a $\pm 1$ [[Policy Gradient#^baseline|advantage]] with no baseline, so it sits closer to [[Simple Statistical Gradient-Following Algorithms (REINFORCE)|REINFORCE]] than to [[PPO]]. And the whole thing is a counterexample to the [[Distilling the Knowledge in a Neural Network|dark knowledge]] premise: on-policy, the teacher's soft-target *shape* appears not to be what transfers.

## Links
Related: [[KL Divergence]] · [[Distilling the Knowledge in a Neural Network]] · [[Policy Gradient]] · [[GRPO]] · [[1% of Tokens Can Be Enough- On Gradient Estimation in On-Policy Distillation]] · [[On-policy Distillation with Verifiable Reward]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[Credit Assignment]] · [[PACT- From Credit Assignment to Critic Alignment]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[Distillation]] · [[Reward Function]] · [[On-Policy vs Off-Policy]]

New topics worth writing: sign-based gradient methods (signSGD and relatives), closed-loop vs one-shot supervision, state-space saturation in on-policy training, multi-teacher consensus gating, token-level reward masking, RLVR verifier signal necessity, teacher-student disagreement as a diagnostic
