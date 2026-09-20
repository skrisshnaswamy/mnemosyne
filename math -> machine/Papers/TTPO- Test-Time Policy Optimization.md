---
title: "TTPO: Test-Time Policy Optimization"
authors: ["Aozhe Wang", "Zhengxi Lu", "Jianze Wang", "Shangke Lv", "Ying Liu", "Weiming Lu", "Jun Xiao", "Yueting Zhuang", "Hua Yang", "Qianglong Chen", "Yongliang Shen"]
year: 2026
arxiv: "2608.27448"
url: https://arxiv.org/abs/2608.27448
priority: Good-To-Read
read_on: 2026-09-05
tags: [paper, llm, rl, vision]
---
## The Core Idea

You have a model, a pile of hard maths problems, and **no answers**. You want the model to get better on those exact problems, right now. That is *test-time training* (TTT). Every strong post-training recipe — RL with verifiable rewards, on-policy self-distillation — needs the ground-truth answer, either to score a rollout or to feed a teacher. So none of them apply.

The obvious patch is to vote: sample $K$ answers, take the most common one as a fake label. This is what TTRL does. It is fragile. On competition problems the vote is **wrong about 85% of the time** (Qwen3-1.7B on AIME 2026). And a wrong label hurts *much* more when your supervision is dense: a bad reward misleads once per trajectory; a bad teacher misleads at **every single token**.

The insight that rescues this is an asymmetry in how the errors land:

> [!NOTE] The asymmetry
> Even when the majority vote is **wrong**, ~79% of the rollouts that *disagree* with it are also wrong — they produce a third answer that is neither the vote nor the truth. So "this rollout is bad" is a reliable statement even when "the vote is right" is not. Disagreement is a label-free signal; agreement is not. ^disagreement-is-safe

This is the same trick as *negative learning from noisy labels*: saying what something **is not** stays true far more often than saying what it **is**.

TTPO builds two different losses on top of that.

- **Rollouts that agree with the vote** → dense token-level distillation from an answer-conditioned teacher. Safe, because the teacher is conditioned on *the very answer those rollouts already produced*. If the answer is wrong, the update does not steer toward a random error; it degenerates into distilling the model's *thinking* mode into its *non-thinking* mode. Still useful.
- **Rollouts that disagree** → a GRPO-style negative gradient. Safe, because it only uses "not in the majority cluster", never the content of the label.

What this unlocks: label-free training that **matches or beats label-supervised distillation**, and a self-improving loop — better model → better rollouts → better votes → better training signal — that ground-truth supervision cannot sustain on problems the model mostly fails.

## The Methodology

### Setup

For each problem $x$, sample $K=64$ trajectories $\{y_1,\dots,y_K\}\sim\pi_\theta(\cdot\mid x)$, extract final answers, cluster them by mathematical equivalence. Largest cluster = pseudo-label $\hat a$. Split:

- $\mathcal{P} = \{k : a_k \equiv \hat a\}$ — positives (agree)
- $\mathcal{N} = \{k : a_k \not\equiv \hat a\}$ — negatives (disagree)

### The teacher is the same model, given a hint

No separate teacher network. One model, two prompts, sharing the same completion tokens $y_k$:

$$q_t^{(\hat a)} = \pi_\theta(\cdot \mid [x;\hat a]_{\text{teacher}}, y_{<t}) \quad \text{(no grad)}$$
$$p_t = \pi_\theta(\cdot \mid x_{\text{student}}, y_{<t}) \quad \text{(with grad)}$$

The teacher prompt has thinking mode **on** and includes $\hat a$ as a "reference solution". The student prompt has thinking mode **off**. The teacher runs on frozen base weights (LoRA adapters disabled), so it does not drift. This is the [[Distilling the Knowledge in a Neural Network|distillation]] setup where privileged context, not a bigger model, is what makes the teacher better.

### Branch 1 — distil the positives

Forward [[KL Divergence|KL]], teacher first, with a per-token weight:

$$\mathcal{L}_{\text{OPSD}}(k)=\frac{1}{T_k}\sum_{t=1}^{T_k} w(t)\cdot \mathrm{KL}\!\left(q_t^{(\hat a)} \,\|\, p_t\right)$$

The weight combines student entropy $\hat H(t)$ and teacher–student divergence $\hat\Delta(t)=\widehat{\mathrm{KL}(q_t\|p_t)}$, both min-max normalised to $[0,1]$ per sample, joined by a **soft-OR**:

$$w(t)=\hat H(t)+\hat\Delta(t)-\hat H(t)\hat\Delta(t)$$

High weight if the student is *uncertain* **or** *confidently disagrees with the teacher*. Near zero only when both are low — the student already knows this token. In their case study, coordinate literals like "$(0,a)$" get near-zero weight; the geometric insight "the extension beyond $A$ is the line going downwards" gets high weight.

### Branch 2 — penalise the negatives

Binary reward $r_k = \mathbf{1}[a_k \equiv \hat a]$, group-relative advantage over all $K$ rollouts, so $A_k < 0$ for every $k\in\mathcal{N}$. With $|\mathcal{P}|/K = \bar r$:

$$A_k = -\sqrt{\frac{\bar r}{1-\bar r}}, \quad k \in \mathcal{N}$$

$$\mathcal{L}_{\text{GRPO}}(k)=-\frac{A_k}{T_k}\sum_{t=1}^{T_k} m(t)\cdot \log \pi_\theta(y_k^{(t)}\mid x, y_k^{(<t)})$$

Here is a real problem they had to fix. In vanilla GRPO, positive-advantage rollouts push *up* on tokens that also appear inside failed rollouts, cancelling unfair penalties. TTPO's RL branch touches **only** negatives, so that counterbalance is gone — every locally-correct reasoning step inside a failed trajectory gets punished. Their fix is a mask that keeps only the top 50% of tokens by

$$s(t) = -\log \pi_\theta(y_k^{(t)}\mid x,y_k^{(<t)}) \cdot (1-\hat H(t))$$
$$m(t) = \mathbf{1}\!\left[s(t) \geq \mathrm{median}(\{s\})\right]$$

Read it as: *low probability × high confidence* = the model confidently emitted something it thinks is unlikely = anomalous. Using the **unnormalised** $-\log p$ as the dominant term is deliberate: high-probability tokens (usually the correct routine arithmetic) drop out automatically.

### Combined

$$\mathcal{L}_{\text{TTPO}}=\frac{1}{|\mathcal{B}|}\left(\sum_{k\in\mathcal{P}}\mathcal{L}_{\text{OPSD}}(k)+\lambda\sum_{k\in\mathcal{N}}\mathcal{L}_{\text{GRPO}}(k)\right), \quad \lambda = 0.1$$

$\lambda$ is not cosmetic — the raw GRPO loss is about an order of magnitude larger than the forward-KL loss, so without it RL swamps distillation.

### Training details that mattered

- Qwen3-1.7B / 4B / 8B, [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]] $r=64$, $\alpha=128$, all linear layers.
- [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], lr $5\times10^{-6}$, grad-norm clip 0.1, batch 32, **100 steps**. bfloat16 + [[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention]]-2, 4×H20.
- Sample $K=64$, but only $K_{\text{train}}=8$ enter the gradient step — a **fixed 50/50** positive/negative split.
- Max generation 16,000 tokens (so answers can actually be extracted without truncation), but the gradient only touches the **first 1,024 completion tokens**.
- Full-vocabulary logit distillation, not top-k.

## Ablation Studies and Experiments

Five competition benchmarks: AIME 2025/2026, HMMT 2025/2026, BRUMO 2025. Avg@12, temperature 1.0, thinking mode on.

**Against label-supervised baselines** (OpenThoughts data; GRPO and OPSD use ground truth, TTPO does not):

| Model | Base | +GRPO† | +OPSD† | +TTPO |
|---|---|---|---|---|
| Qwen3-1.7B | 34.6 | 35.7 | 39.7 | **40.1** |
| Qwen3-4B | 56.0 | 57.5 | 58.4 | **58.6** |
| Qwen3-8B | 58.6 | 61.2 | 61.7 | **62.6** |

TTPO with *no labels* edges out OPSD *with labels* at all three scales. TTPO on 4B (58.6) equals the untrained 8B base (58.6).

**Pure TTT** (train directly on the test problems, nobody gets labels), average over AIME26 / HMMT26 / BRUMO25:

| Model | Base | +TTRL | +OPSD-TTT | +TTPO |
|---|---|---|---|---|
| 1.7B | 38.0 | 40.2 | 41.9 | **45.2** |
| 4B | 57.4 | 58.8 | 59.4 | **61.1** |
| 8B | 60.7 | 63.0 | 63.7 | **65.3** |

**Update-strategy ablation** (AIME26, 1.7B, TTT) — this is the one that carries the paper:

| pos / neg | Score |
|---|---|
| FKL / GRPO (**TTPO**) | **48.9** |
| FKL / — (positives only) | 46.7 |
| FKL / FKL (all rollouts) | 46.3 |
| — / FKL (negatives only) | 43.9 |
| GRPO / FKL (reversed) | 37.2 |

Distilling *only the disagreeing* rollouts (43.9) — which is what one prior paper does — is worse than doing nothing clever, because the conflict term $\Delta_{\text{conflict}}(t)$ actively drags a possibly-correct trajectory toward a possibly-wrong label. Putting GRPO on positives is catastrophic (37.2): it directly reinforces trajectories, so a wrong vote reverses the update with no mitigation.

**Token selection** (1.7B, OpenThoughts):

| Variant | AIME26 | HMMT26 | BRUMO25 |
|---|---|---|---|
| TTPO | 46.5 | 31.6 | 54.7 |
| w/o positive weighting | 43.3 | 30.6 | 52.8 |
| w/o negative masking | 45.4 | 29.5 | **50.0** |

Both help; the negative mask matters more on BRUMO (−4.7). Consistent with the "no positive advantage to cancel false penalties" argument.

**Privileged information** (1.7B, AIME26; parentheses = evaluated with thinking off):

| Privilege | Thinking teacher | Non-thinking teacher |
|---|---|---|
| None | 45.8 (36.1) | 41.7 (8.1) |
| Answer | **46.5 (39.8)** | 33.6 (6.7) |
| Full trajectory | 41.1 (8.9) | 40.8 (10.6) |

Giving the teacher a whole solution trajectory **hurts** (41.1). It stops reasoning and just completes the prefix. A bare answer is a light nudge that leaves the teacher's own reasoning intact.

### What did not work

- **Ground-truth labels are worse than pseudo-labels here.** Swapping $\hat a$ for the true answer inside the same asymmetric objective *lowers* performance. Reason: on AIME-hard problems $|\mathcal{P}_{\text{GT}}| \approx 0$, so the FKL branch has nothing to distil **and** $|A_k| = \sqrt{\bar r/(1-\bar r)} \approx 0$ kills the RL branch too. The loss curve for TTPO-w/-GT is nearly flat. The vote always produces a non-empty cluster, so both branches stay alive. Also, an AIME answer is a bare integer — too thin to shift the teacher.
- **Dynamic positive/negative ratio in $K_{\text{train}}$** (mirroring the true split) underperformed a fixed 50/50 (46.1 vs 46.5 on AIME26, 51.1 vs 54.7 on BRUMO). If positives dominate, shrinking the negative count cancels the amplified advantages those negatives just earned.
- **"Top signal" rollout selection** — pick the positives with largest teacher–student divergence and negatives with highest log-prob — lost to simply picking the **shortest** completions (45.8 vs 46.5 AIME26; 51.9 vs 54.7 BRUMO25). The authors are blunt: the intuition "bigger divergence = more to learn" has no theoretical grounding and does not hold empirically. Shortest wins because only the first 1,024 tokens get gradients, so short chains put the decisive reasoning inside the window.
- **$\lambda$ is sharp.** 0.01 → 41.4, 0.05 → 43.9, **0.10 → 46.5**, 0.15 → 44.7, 0.20 → 41.7 (AIME26).

## Worth Remembering

**The non-thinking result is the loudest number in the paper.** Evaluated with thinking mode disabled, Qwen3-8B goes 20.3 → **56.7** average (+36.4), while label-supervised OPSD only reaches 23.8 (+3.5). TTPO is, in effect, an extremely good thinking→non-thinking compressor. The authors credit the GRPO branch: pure distillation passively pulls the student toward the teacher, but the negative branch *actively suppresses* the student's own failure modes in its own generation mode.

**Self-evolution is real and measured.** On HMMT26, Avg@12 climbs toward the base model's Maj@12 — the collective vote knowledge gets absorbed into single samples — but Maj@12 **also rises**, so the ceiling moves up with the model. Training entropy stays *higher* with pseudo-labels than with ground truth, which the authors read as the vote-routing sustaining exploration.

**Compute cost is not free.** 64 rollouts per problem at up to 16k tokens, then a teacher and a student forward pass. This is heavy inference-time work for a 100-step LoRA update. Compare with the cheaper [[Prefix Sliding for efficient test-time scaling|test-time scaling]] family, where you spend compute at inference but never touch weights.

**Admitted limitations.** (1) If $K$ is small, or *no* rollout ever gets the right answer, both branches get pure noise — there is no fallback. (2) Only verifiable maths; code (needs execution) and open-ended text (needs a reward model) are untested. (3) The asymmetric objective is static; as $\hat a$ becomes more accurate over training, the ideal $\lambda$ probably should shift, and they do not do this.

**Open questions.** Does the 50/50 split still hold when the base model is strong enough that positives dominate at 90%? Does the "penalise disagreement" logic survive in domains where wrong answers cluster — e.g. a common misconception producing a large *wrong* majority, where the 79% figure could collapse? And the 1,024-token gradient window is an implementation compromise that quietly biases the whole method toward short reasoning chains.

## Links

Related: [[Distilling the Knowledge in a Neural Network]] · [[On-policy Distillation with Verifiable Reward]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[SecOPD- Mitigating Adaptive Prompt Injections by On-Policy Distillation]] · [[Proximal Policy Optimization Algorithms]] · [[Training language models to follow instructions with human feedback]] · [[KL Divergence]] · [[Cross Entropy]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Is Next-Chunk Reasoning RL Really Better than SFT- Revisiting Training Strategies under no-CoT Data]] · [[Prefix Sliding for efficient test-time scaling]] · [[Uncertainty]] · [[Mixed Precision Training]]

New topics worth writing: GRPO (Group Relative Policy Optimization), TTRL / test-time reinforcement learning, majority-vote self-consistency decoding, negative learning from noisy labels (NLNL), token-level credit assignment in RLVR, forward vs reverse KL in distillation, privileged-information teachers, thinking-mode toggles in Qwen3
