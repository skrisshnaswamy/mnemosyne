---
title: "Negative Self-Distillation: Learning to Reason by Avoiding Flaws"
authors: ["Rongcan Pei", "Zhepei Wei", "Shuyao Xu", "Xinyu Zhu", "Wei-Lin Chen", "Yu Meng"]
year: 2026
arxiv: "2609.11699"
url: https://arxiv.org/abs/2609.11699
priority: Good-To-Read
read_on: 2026-09-17
tags: [paper, llm, rl, vision, theory]
---
## The Core Idea

Normal self-distillation gives a model a cheat sheet — the gold answer — lets it write a clean solution while looking at that cheat sheet, then makes the same model (without the cheat sheet) copy that clean solution token by token. It works, except the cheat-sheet version writes suspiciously smooth reasoning. It never says "wait, that's wrong", never backtracks, never hedges. So the student learns to be confident and linear. On hard maths problems, confidence and linearity are exactly what you do *not* want. The student loses the ability to catch its own mistakes.

**Negative Self-Distillation (NSD)** flips the sign. Instead of a teacher that knows the answer, build a teacher that is deliberately *bad*. Ask the model itself to write a short instruction like "You are a student who trusts the first pattern they see and never checks whether the bound is attained." Feed that instruction to a frozen copy of the model. Now you have a **negative teacher** — a distribution over tokens that is tilted towards careless reasoning. Then train the student to move *away* from it.

No gold labels. No external teacher. No majority vote. The only supervision is "here is a bad version of me; be less like that."

> [!NOTE] Negative teacher
> The same frozen weights as the student, but conditioned on an extra prompt that induces a specific bad habit. It is not a worse model — it is the same model wearing a flawed persona. ^negative-teacher

The hard part is not the idea, it is the targeting. If you just push down every token the bad teacher likes, you push down commas, the word "the", and "step". Those appear in good and bad reasoning alike. Unlearning them wrecks the model's basic fluency. NSD's actual contribution is a **gate**: only penalise a token if the bad persona made that token *more likely than it was without the persona*. That difference is the signal. Everything else is language, and language is left alone.

What it unlocks: a training signal that is dense (per token, like [[Distilling the Knowledge in a Neural Network|distillation]]), label-free (like TTRL or Intuitor), and — unusually — *increases* self-correction instead of destroying it. Reflection tokens per response went from 3.6 (base Qwen3-4B) to 7.5 under NSD, while on-policy self-distillation dropped them to 2.2.

## The Methodology

Three models, all the same weights, differing only in what they are conditioned on:

- $\pi_\theta$ — the **student**, the only one being updated.
- $\pi_{\text{ref}}$ — **frozen**, sees only the problem $x_i$.
- $\pi_{\text{neg}}$ — **frozen**, sees the problem $x_i$ *plus* a negative condition prompt $n_i$.

**Step 1 — make the negative condition.** Default is online: sample an initial solution, then ask the model to write an attack prompt aimed at that solution's weak step.

$$y_{\text{init}} \sim \pi_\theta(\cdot \mid x), \qquad n \sim \pi_\theta(\cdot \mid x, y_{\text{init}})$$

The meta-prompt is quite specific (Appendix C.3). It demands four parts: a **persona** starting literally with "You are a student who…", a **trigger** that abstracts the problem class without naming the numbers, a **flawed execution** (an impulsive shortcut), and a **fatal omission** (explicitly forbid the verification step). Two to three sentences.

**Step 2 — roll out.** Sample one trajectory $y = (y_1,\dots,y_T) \sim \pi_\theta(\cdot\mid x_i)$. Just one. Max length 4096.

**Step 3 — the gate.** For each sampled token, compare how likely the bad teacher thought it was against how likely the plain reference thought it was:

$$G_t = \max\big(0,\; \pi_{\text{neg}}(y_t \mid x_i, n_i, y_{<t}) - \pi_{\text{ref}}(y_t \mid x_i, y_{<t})\big)$$

If the persona did not raise the token's probability, $G_t = 0$ and nothing happens to it. If it did, $G_t$ is the size of the boost, and that is also the penalty weight. Bigger boost, bigger push.

> [!NOTE] Token-level adaptive gating
> The penalty weight is the *increase in likelihood caused by the bad persona*, clipped at zero. Ordinary grammar tokens get similar probability under both contexts, so the difference is ~0 and they are protected automatically. ^adaptive-gating

**Step 4 — the bounded penalty.** Plain unlikelihood training (Welleck et al. 2020) minimises $-\log(1-\pi_\theta(y_t))$. That blows up as $\pi_\theta \to 1$. NSD squashes it through a sigmoid, which happens to have a clean closed form:

$$\mathcal{L}_{\text{GU}}^{(t)} = G_t \cdot \sigma\Big(-\log\big(1-\pi_\theta(y_t \mid x_i, y_{<t})\big)\Big) = G_t \cdot \frac{1}{2 - \pi_\theta(y_t \mid x_i, y_{<t})}$$

The reason this matters is visible in the [[Backpropagation|gradient]] with respect to the logit. Plain unlikelihood gives $G \cdot \pi_c$ — grows linearly with confidence, so the biggest updates land on punctuation. The sigmoid version gives

$$\frac{\partial \mathcal{L}_{\text{GU}}}{\partial z_c} = G \cdot \frac{\pi_c(1-\pi_c)}{(2-\pi_c)^2}$$

which goes to zero as $\pi_c \to 1$ and peaks around $\pi_c \approx 0.6$ — the ambiguous, reasoning-critical tokens. The mechanism is the same $\pi_c(1-\pi_c)$ softmax Jacobian term you know from [[Cross Entropy]]; here it is left in the numerator instead of being cancelled.

**Step 5 — the leash.** A single-sample [[KL Divergence|forward KL]] anchor to the reference, evaluated only on the sampled token (no full-vocabulary sum):

$$\mathcal{L}_{\text{KL}}^{(t)} = \pi_{\text{ref}}(y_t) \cdot \log \frac{\pi_{\text{ref}}(y_t)}{\pi_\theta(y_t)}$$

$$\mathcal{L}_{\text{NSD}}^{(t)} = \mathcal{L}_{\text{GU}}^{(t)} + \alpha \cdot \mathcal{L}_{\text{KL}}^{(t)}, \qquad \alpha = 0.01$$

Sum over tokens, backprop, done. The whole per-token loss needs exactly three scalars: $p_\theta$, $p_{\text{ref}}$, $p_{\text{neg}}$. No logit alignment over the vocabulary.

**Setup.** MATH training set with gold labels thrown away. Qwen3-1.7B / 4B / 8B, 2 epochs, batch 32, actor LR $1\times10^{-6}$, warmup ratio 0.1, 8×A100 (4 for actor, 2 for teacher). All training and eval in Qwen3 **non-thinking** mode (`enable_thinking=False`, empty `<think>` block). Eval: temperature 0.6, top-$p$ 0.95, top-$k$ 20, 32K output, Avg@8.

## Ablation Studies and Experiments

**Main table, Avg@8 (%) over 7 maths benchmarks — AIME 24/25/26, HMMT 25 Feb, AMC 23, OlympiadBench, MATH-500.** $\Delta$ Avg is absolute average gain over the untuned base.

| Model | Method | $\Delta$ Avg | 95% CI | $p$ |
|---|---|---|---|---|
| 1.7B | OPSD† | +1.1 | [−0.3, +2.4] | 0.06 |
| | Intuitor | −0.5 | [−1.8, +0.8] | 0.29 |
| | TTRL | +0.3 | [−1.0, +1.5] | 0.29 |
| | **NSD** | **+2.3** | [+0.7, +4.0] | 0.001 |
| 4B | OPSD† | +1.0 | [−0.3, +2.4] | 0.10 |
| | Intuitor | +1.3 | [−0.5, +3.1] | 0.05 |
| | TTRL | +0.2 | [−1.2, +1.7] | 0.33 |
| | **NSD** | **+7.5** | [+5.4, +9.5] | $<10^{-4}$ |
| 8B | OPSD† | +0.3 | [−1.3, +1.9] | 0.33 |
| | Intuitor | +1.9 | [+0.2, +3.4] | 0.02 |
| | TTRL | −0.1 | [−1.4, +1.3] | 0.57 |
| | **NSD** | **+6.0** | [+4.0, +7.9] | $<10^{-4}$ |

† needs gold labels. Concretely on 4B: AIME 2024 goes 23.8 → 35.8, AIME 2025 20.4 → 31.3, AIME 2026 17.9 → 29.2. On 8B, AIME 2024 28.8 → 39.6. Note every baseline's confidence interval crosses zero except Intuitor-8B; NSD's does not. That is the honest reading of this table — the baselines are noise, NSD is not.

**Pass@8** (is it finding genuinely new solutions, or just reranking?): 4B goes +8.3 average, 8B +9.7, against OPSD at +0.0 and −0.6. So the solution set is actually widening, not just being sharpened. Compare this to the concern in [[Locked at the Entrance, Open Inside- Where RLVR Narrows the Solution Space|RLVR narrowing the solution space]].

**Reflection tokens per response** (counting "wait", "actually", "hmm", "let me verify", "reconsider", full list in Appendix E), Qwen3-4B averaged over AIME 24/25 and HMMT 25:

| Method | Average |
|---|---|
| Base | 3.6 |
| OPSD | 2.2 |
| Intuitor | 0.8 |
| **NSD** | **7.5** |

Intuitor — which rewards the model's own confidence — nearly deletes reflection. This is the paper's central claim made visible: confidence-seeking objectives buy accuracy by removing doubt, and on hard problems doubt is the machinery.

**Ablation 1 — does the gate actually pick reasoning tokens?** They define a *style-task ratio* $R$ = mean weight on style tokens ÷ mean weight on task tokens (task = digits, operators, LaTeX, maths vocabulary; style = whitespace, punctuation, connectives, hedges, function words). Lower is better.

| NSD gate (wiki) | NSD gate (sol-aware) | NSD gate (question-only) | Entropy-OPSD | OPSD |
|---|---|---|---|---|
| 2.6× | 3.4× | 3.5× | 3.9× | 5.4× |

Every method still over-weights style tokens; NSD just over-weights them least. The entropy-weighting trick from the "80/20 rule" line of work (Wang et al.) barely beats vanilla OPSD here.

**Ablation 2 — remove the KL anchor.** NSD-noKL collapses mid-training. Forward KL to the reference oscillates sharply (learn-then-forget cycling), and around step 120 the gate activation ratio falls off a cliff. Once the student has drifted far from the reference, the gate — which is defined relative to *frozen* models — stops corresponding to anything the student is actually doing, and fires spuriously. The KL term is what keeps the gate meaningful, not just what keeps the weights from exploding.

**Ablation 3 — where does the negative condition have to come from?** Qwen3-4B, four datasets:

- Online solution-aware (default): +7.8
- Question-only, offline: **+7.3**
- Wiki-irr — just paste a random irrelevant Wikipedia passage as "context": competitive
- Offline solution-aware: clearly worse

This is the most deflating and most interesting result. **Pasting irrelevant Wikipedia text works almost as well as a carefully engineered persona attack.** The mechanism seems to be less "the teacher makes specific errors" and more "the teacher is degraded in some way, and we move away from degradation." The offline solution-aware variant loses because it conditions on a stale rollout from an earlier checkpoint — the mismatch hurts more than the extra specificity helps.

**What did not work / negative results:**

- **Naive unlikelihood** — unbounded, gradient scales as $G\cdot\pi_c$, blows up on punctuation, destroys fluency. Motivated the sigmoid.
- **No KL** — mid-training collapse (above).
- **Offline solution-aware conditioning** — worst of the four variants.
- **Policy-gradient reformulation** (Appendix D.3), setting $A_t = -\mathcal{L}_{\text{NSD}}^{(t)}$ and optimising $\sum_t A_t \log \pi_\theta(y_t)$: beats the direct loss on 1.7B (+5.0 vs +2.5) but *loses* on 4B and 8B. The authors' read is that treating the loss as a constant advantage truncates gradient information. Under this regime wiki-irr becomes the best strategy at every size, which reinforces the "generic degradation" reading.
- **Thinking mode, 4B** (Appendix D.2): NSD +3.0, OPSD +0.2, TTRL −1.9, **Intuitor −11.3** — Intuitor over-thinks so badly that responses blow past a 38K token limit and get truncated. NSD's OlympiadBench jump here is odd though: 45.9 → 57.9, far larger than anywhere else.
- **1.7B gains are small** (+2.3). The model is too weak to write a useful negative condition about itself.

**Efficiency** (wall-clock per step, A100s): NSD needs $n=1$ rollout vs $n=8$ for GRPO-based TTRL/Intuitor, cutting rollout time ~60%. $\pi_{\text{ref}}$ and $\pi_{\text{neg}}$ share weights and prefill in parallel. Loss needs 3 scalars, not top-$k$=128 logits like OPSD. Wiki-irr removes the online condition-generation rollout entirely: 68s → 54s per step.

## Worth Remembering

- **The gate is the paper.** The negative-teacher idea is old (unlikelihood training, RLCD, contrastive distillation). What makes it not destroy the model is $G_t = \max(0, \pi_{\text{neg}} - \pi_{\text{ref}})$ — a per-token, self-normalising way to separate "this token is a flaw" from "this token is English". Anywhere you want to unlearn a behaviour without unlearning the substrate, that difference-of-two-conditionings trick transfers.

- **The gradient direction is inverted relative to OPSD.** OPSD's gradient *grows* with token probability, so it teaches the student to imitate the teacher's high-confidence tokens — mostly style. NSD's gradient *shrinks* with token probability and peaks at $\pi_c \approx 0.6$. Two objectives that look superficially symmetric behave completely differently on the token distribution a language model actually produces.

- **The wiki-irr result should make you suspicious of the story.** If irrelevant Wikipedia text is nearly as good as a bespoke "careless reasoner" persona, then the framing ("avoid *flaws*") may be a post-hoc narrative over a simpler mechanism: distraction degrades the conditional distribution, and moving away from a degraded distribution sharpens it. Whether this is closer to a regulariser than to error-avoidance is not settled by any experiment here.

- **Limitations the authors admit:** the method needs the model to be capable enough to author its own negative conditions, which is why 1.7B barely moves. Online conditioning costs an extra rollout plus two forward passes.

- **Only maths, only Qwen3, only MATH as training data.** No code, no general reasoning, no cross-family check. The one-sided $p$-values are welcome but they are computed over 7 benchmarks with Avg@8, not over training seeds — run-to-run variance in RL-style post-training is not measured.

- **Practical caveats if you wanted to use this:** $\alpha=0.01$ is not swept and the KL ablation shows the method dies without it, so that is a load-bearing unswept hyperparameter. The gate compares *frozen* models, so any large student drift silently degrades the signal — monitor gate activation rate as your health metric, not the loss. Max generation length 4096 during training vs 32K at eval is a real train/test mismatch that goes unexamined.

- **Connection worth chasing:** the gating rule is structurally a per-token version of a difference-in-conditionals estimator — it asks "how much did the treatment (the negative prompt) raise this token's probability?" That is a contrast, not a reward, and it sidesteps the advantage-collapse problem that motivates the whole label-free RL line. Whether the same construction gives a usable per-token credit signal in ranking or bandit settings is an open and interesting question.

- **Follow-up question:** the reflection-token count is a proxy, and a gameable one. An objective that pushes away from a "never verifies" persona will mechanically raise the frequency of verification words whether or not the verification is doing work. Pass@8 gains are the stronger evidence; the reflection table alone would not convince.

## Links

Related: [[Distilling the Knowledge in a Neural Network]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[On-policy Distillation with Verifiable Reward]] · [[KL Divergence]] · [[Cross Entropy]] · [[Distillation]] · [[Mode Collapse]] · [[Uncertainty]] · [[Proximal Policy Optimization Algorithms]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[Locked at the Entrance, Open Inside- Where RLVR Narrows the Solution Space]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Regularization]] · [[Backpropagation]] · [[TTPO- Test-Time Policy Optimization]] · [[In Context Learning]]

New topics worth writing: Unlikelihood training, Reinforcement Learning with Verifiable Rewards (RLVR), GRPO, On-policy self-distillation, Label-free / intrinsic-reward RL (RLIF), Test-time reinforcement learning (TTRL), Privileged-information teachers, Token-level credit assignment in LLM post-training

**Tags:** `llm` `rl` `optimization`
