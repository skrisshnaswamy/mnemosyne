---
title: "SCOPD: Sparse-Context On-Policy Self-Distillation for Efficient Vision-Language Models"
authors: ["Ahmadreza Jeddi", "Enming Zhang", "Jasper Gerigk", "Hakki Karaimer", "Mozhgan Nasr Azadani", "Jiayun Luo", "Minh Ngoc Le", "Gholamali Aminian", "Hugo Buurmeijer", "Yongchao Chen", "Leonid Sigal", "Igor Gilitschenski", "Konstantinos G. Derpanis", "Marco Pavone", "Babak Taati"]
year: 2026
arxiv: "2609.34044"
url: https://arxiv.org/abs/2609.34044
priority: Good-To-Read
read_on: 2026-09-30
tags: [paper, llm, vision]
---
## The Core Idea

When a vision-language model (VLM) looks at an image, the image first becomes hundreds or thousands of "visual tokens" — little vectors the language model reads like words. That is expensive. So people **prune** them: throw away most visual tokens, keep the ones that look important, and run the model on what's left. Cheap, no retraining. But when you get aggressive — keep only 10% — accuracy falls off a cliff.

The standard explanation: you threw away the evidence. The picture information the question needed is gone. Nothing to be done except prune smarter.

This paper shows that explanation is **incomplete**, with a clean experiment. Take 500 questions the unpruned model gets right with greedy decoding. Prune each image **once** to 10% of tokens and *freeze* that pruned context. Now sample 64 different reasoning traces from that exact same frozen context.

- Greedy Pass@1: $53.2\%$
- Pass@64 from the identical pruned tokens: $79.6\%$
- Same thing with the image deleted entirely (language-only control): $2.8\% \to 4.2\%$

Nothing about the visual input changed between those 64 samples. So the recovery cannot be "we got lucky with the pruning". The evidence was *still in there*. The language model just failed to reach for it reliably.

> [!NOTE] Representation–utilization gap
> Two separate failures hide inside one accuracy drop. **Representation** failure: the needed pixels were pruned away, unrecoverable. **Utilization** failure: the needed pixels survived, but the language model's decoding does not consistently use them. Pruning research has treated the drop as all representation. Much of it is utilization, and utilization is fixable by training the language model — not by changing the pruner. ^representation-utilization-gap

What that unlocks: a post-training fix that is orthogonal to every pruning method. You do not touch the architecture, you do not add inference cost, you do not need labelled answers or reasoning traces. You just teach the language model to read sparse visual context. At 10% retention this moves the normalized aggregate over 13 benchmarks from $86.37\%$ of unpruned performance to $92.43\%$.

The mechanism is **on-policy self-distillation with a privileged teacher**. The student sees 10% of the tokens and writes a reasoning trace. The teacher is the *same model* seeing 100% of the tokens, and it grades that trace token by token. The teacher is smarter only because it can see more — not because it is a bigger model.

> [!NOTE] Privileged teacher
> A teacher with the same weights as the student, but given extra information the student will never have at deployment. Here: the full visual context. The student cannot copy the teacher's advantage, only learn to compensate for missing it. ^privileged-teacher

## The Methodology

### SCOPD — the base method

Notation. $x_i$ is the text (question), $Z_i$ is the full visual token sequence, $P_b$ is a pruning operator at retention budget $b$, and $Z_i^b = P_b(Z_i)$ is the pruned visual context.

Three steps per training example.

**1. Roll out on-policy, sparse.** The student generates a reasoning trajectory from the *pruned* context:

$$\hat{y}_i \sim p_\theta(\cdot \mid x_i, Z_i^b)$$

This is the whole point of "on-policy". Training happens at the states the deployed pruned model actually visits, not at states from some curated dataset.

**2. Score the same prefix twice.** At each decoding position $t$, take the student's own prefix $\hat{y}_{i,<t}$ and get two next-token distributions:

$$p^b_{i,t} = p_\theta(\cdot \mid x_i, Z_i^b, \hat{y}_{i,<t}) \qquad q_{i,t} = p_{\bar\theta}(\cdot \mid x_i, Z_i, \hat{y}_{i,<t})$$

Student with 10% of tokens, teacher with 100%. Same weights family, same prefix, different amount of picture.

**3. Match them.** Forward KL, averaged over the whole response:

$$\mathcal{L}^{\text{SCOPD}}_i = \frac{1}{T_i}\sum_{t=1}^{T_i} D_{\mathrm{KL}}\!\left(q_{i,t} \,\|\, p^b_{i,t}\right)$$

No ground-truth answer appears anywhere. No reference reasoning trace. The supervision is entirely "what would I have said if I could see the whole image".

The teacher $\bar\theta$ is an **exponential moving average** of the student, decay $0.9999$. This matters enormously — see the ablations.

### SCOPD+ — supervise only where vision matters

Dense KL has a flaw the authors identify sharply: **a big teacher–student disagreement is not necessarily a *visual* disagreement.** The teacher might prefer a different capitalisation, a synonym, a different phrasing of the same step. High KL, zero visual content. Training on those positions spends gradient on language-level noise.

So they measure visual dependence directly, by intervention. Build a slightly richer context:

$$b^+ = b + \delta, \qquad Z_i^{b^+} = P_{b^+}(Z_i)$$

with $\delta = 1\%$ by default. Score the **same frozen prefix** under this slightly-better-seeing student:

$$p^{b^+}_{i,t} = p_\theta(\cdot \mid x_i, Z_i^{b^+}, \hat{y}_{i,<t})$$

If one extra percent of visual tokens noticeably changes the next-token distribution here, this position is hungry for visual evidence. Score it with Jensen–Shannon divergence:

$$S_{i,t} = \tfrac{1}{2}D_{\mathrm{KL}}(p^b_{i,t}\|m_{i,t}) + \tfrac{1}{2}D_{\mathrm{KL}}(p^{b^+}_{i,t}\|m_{i,t}), \qquad m_{i,t}=\tfrac{1}{2}(p^b_{i,t}+p^{b^+}_{i,t})$$

JSD, not KL, on purpose: neither budget is "the truth", so the measure should be symmetric; and JSD is bounded, so one freak low-probability token cannot dominate the ranking.

Keep the top $\rho = 10\%$ of positions by $S_{i,t}$:

$$\mathcal{K}_i = \operatorname{TopK}_{t}(S_{i,t}, \lceil \rho T_i \rceil)$$

and apply the teacher KL only there:

$$\mathcal{L}_{\text{SCOPD+},i} = \frac{1}{|\mathcal{K}_i|}\sum_{t\in\mathcal{K}_i} D_{\mathrm{KL}}\!\left(q_{i,t}\,\|\,p^b_{i,t}\right)$$

Same rollout, same teacher, 10% of the backprop positions. **No extra rollout** — the intervention reuses the student's trace and the already-encoded visual features, costing one more forward pass.

> [!NOTE] Visual sensitivity
> How much a model's next-token prediction shifts when you give it slightly *more* visual evidence, holding the text prefix fixed. A direct test of "does this token depend on the image", as opposed to teacher–student disagreement, which conflates vision with wording. ^visual-sensitivity

### Training setup

| Item | Value |
|---|---|
| Base model | Qwen2.5-VL-7B-Instruct (also Qwen3-VL-4B) |
| Pruner | VisionZip, 5% dominant + 5% contextual tokens |
| Budget | $b = 10\%$ (also 20%, 5%) |
| Adaptation | LoRA on the LLM only, $r=16$, $\alpha=32$, no dropout |
| Frozen | Vision encoder, multimodal projector |
| LR | constant $2\times10^{-5}$ |
| Batch | effective 32 over 4 GPUs |
| Teacher | EMA of student, decay 0.9999 |
| $\delta$, $\rho$ | $1\%$, $10\%$ |
| Data | ~10K LLaVA-CoT examples (images + questions only for SCOPD) |
| Format | `<think>...</think>` then `<answer>...</answer>` |
| Cost | ~14 hours on 4× L40 |

VisionZip itself is worth knowing: it scores visual tokens by how much attention other visual tokens pay them, keeps the top "dominant" ones outright, then clusters the leftovers by key-feature similarity and merges each cluster into one "contextual" token. It is text-agnostic — it never sees the question — so the retained set stays fixed for the whole reasoning trace.

## Ablation Studies and Experiments

### Headline (13 image benchmarks, normalized to unpruned Vanilla = 100)

| Method | 100% tokens | 20% | 10% | 5% (8 benchmarks) |
|---|---|---|---|---|
| Vanilla | 100.00 | 93.42 | 86.37 | 75.99 |
| SFT | 99.17 | 95.22 | 88.82 | — |
| EPIC | 100.30 | 94.71 | 86.45 | — |
| GRPO | 100.84 | 93.95 | **85.73** | — |
| SCOPD | 100.67 | 96.80 | 90.49 | 83.63 |
| SCOPD+ | 100.02 | **97.25** | **92.43** | **83.66** |

Read the first column first. With the full visual context both methods sit at ~100 — they add nothing. The gain appears *only* as the budget shrinks. That is the cleanest possible evidence that this is sparse-context adaptation and not generic post-training polish.

Where SCOPD+ beats SCOPD hardest at 10%: LogicVista ($34.68 \to 38.26$), MathVerse ($30.46 \to 33.38$), VisOnlyQA ($42.09 \to 43.65$), HR4K ($63.75 \to 66.00$). Diagram reading, fine-grained perception, high-resolution detail — exactly the tasks where tokens carry irreplaceable visual content.

### What did not work

**GRPO actively hurt.** At 10% retention it scored $85.73$, *below* the untrained Vanilla at $86.37$. The diagnosis in Appendix E is the most instructive negative result here. With group size $G=4$, the base model already answers all four samples correctly on roughly $55\%$ of prompts. Those groups have zero within-group variance, so the [[GRPO|group-mean baseline]] gives them zero advantage and zero gradient. Only ~$35\%$ of groups are mixed. Over 2,500 steps and ~30 hours, accuracy ($\approx 76\%$), Pass@4 ($\approx 91\%$) and response length ($\approx 160$ tokens) were all flat. The only thing that improved was format compliance, $84.8\% \to 89.2\%$. Outcome-level reward at one bit per trajectory is simply too thin a signal here; dense token-level supervision is not a nicety, it is the difference between learning and not learning.

**EPIC barely moved at 10%** ($86.45$ vs Vanilla $86.37$), despite being the compression-aware baseline built for exactly this. Its progressive consistency distillation helps at moderate budgets, not aggressive ones.

**SFT beat both RL and EPIC** ($88.82$) but lost to SCOPD by 1.7 points — while needing reference reasoning traces that SCOPD does not.

**A teacher that shares the student's live weights collapses completely.** Avg₁₃ falls to $12.35$ — MMStar 9.07, MathVerse 0.00, VisOnlyQA 0.17. The model is destroyed. A *frozen* initial teacher is fine ($91.37$), EMA is best ($92.43$). This replicates the same finding from Vision-OPD, so treat it as established: in self-distillation the teacher must be regularised away from the current policy or the objective becomes degenerate self-agreement.

### Is visual sensitivity actually the right selector? (Avg₆, all selective variants use $\rho=10\%$)

| Selection | Avg₆ |
|---|---|
| Bottom sensitivity (negative control) | 89.13 |
| Random 10% | 93.56 |
| TIP | 94.21 |
| Dense SCOPD (all positions) | 94.44 |
| Top teacher–student KL | 94.51 |
| **Top visual sensitivity (SCOPD+)** | **95.25** |

This table carries the paper's second claim. Top-KL ($94.51$) barely beats dense ($94.44$) — confirming that disagreement magnitude is a weak signal. Visual sensitivity adds $+0.81$ over dense while using a tenth of the positions. And the negative control is decisive: deliberately picking the *least* visually sensitive tokens drops you to $89.13$, well below random. The ordering is real, not an artifact of sparse gradients.

### Generalisation

**Across pruners** (trained on VisionZip only, no re-adaptation, Avg₆):

| Pruner | Vanilla | SCOPD+ | Δ |
|---|---|---|---|
| VisionZip | 88.74 | 95.25 | +6.51 |
| DivPrune | 83.47 | 91.23 | +7.76 |
| Random | 82.06 | 88.85 | +6.79 |
| FastV | 81.47 | 86.03 | +4.56 |

FastV is the informative one: it prunes *inside* the language model layers, not before them. Transfer still works. So what is being learned is not "how to decode VisionZip's particular token set" but something closer to a general skill of reasoning under a thin visual context.

**Across model families.** Qwen3-VL-4B at 10%: Vanilla $75.29 \to$ SCOPD $81.60 \to$ SCOPD+ $82.92$. (Note the plumbing hack — Qwen3-VL stacks features from several vision layers, so they compute VisionZip's selection on the final layer and reuse those indices for the three intermediate streams.)

**Image training → video evaluation**, with no video-specific post-training at all. Avg₅ over VideoMME, TempCompass, MVBench, MLVU, Video-TT: Vanilla-pruned $90.88 \to$ SCOPD $95.70 \to$ SCOPD+ $96.60$.

### Hyperparameters and design

- $\rho$: peaks at $10\%$; denser supervision gives nothing back.
- $\delta$: $1\%$ best. Bigger interventions compare against a much richer context and start flagging positions that are not bottlenecked at the actual operating budget. The measurement should be *local* to where you deploy.
- Divergence: forward KL $92.43$, reverse KL $92.39$ — a wash. JSD as the *distillation* loss is slightly worse at $91.55$. (JSD is used for *selection*, KL for *training*.)

### Ground truth, if you happen to have it (Appendix F)

Hand the full-context teacher the correct answer while it grades the student's trace. Avg₁₃ at 10% goes $90.49 \to 92.86$, and at 100% context $100.67 \to 102.43$ — i.e. it beats the unpruned original. Kept out of the main results because it changes the supervision assumptions, but if you have labels, take the 2.4 points.

## Worth Remembering

**The fixed-context Pass@K protocol is the transferable artifact.** Freeze the degraded input, resample the output, compare against an input-ablated control. Any time a compression, quantization or pruning step hurts, this separates "information destroyed" from "information not used". The authors note ShortOPD found the same shape for structurally pruned LLMs — Pass@1 collapses, Pass@K survives. Expect this to keep showing up. It is a general diagnostic for efficiency work, not a VLM-specific trick.

**They checked that recovery was real reasoning, not lucky guessing.** Answers were graded by a judge model given the image, question, reference and full trace, rejecting answer-critical hallucinations and unsupported guesses, with uncertain cases counted as failures. They also report $\mathrm{Hit}_{\geq 4}@64 = 74.2\%$ — at least four of 64 samples valid — as a stricter repeatability measure. Without this the Pass@64 number would be unconvincing.

**Cost is negligible where it counts.** SCOPD+ adds $+21.4\%$ theoretical FLOPs over SCOPD but only $+1.9\%$ wall-clock ($17.21 \to 17.54$ s/sample) and $+0.4\%$ peak memory — the extra forward pass is not the bottleneck. At inference there is *zero* overhead: the teacher and the intervention exist only during training. Output length also stays honest: SCOPD 153.0 tokens, SCOPD+ 155.7, unpruned Vanilla 155.3. It is not buying accuracy with longer traces. (Interesting side note: SFT and EPIC generate notably *shorter* traces, 111 and 123 tokens — they seem to have partly learned to stop reasoning.)

**Limits the authors own.** One primary training-time pruner (VisionZip), fixed retention budgets, short-form image and video reasoning only. They flag dynamic/in-LLM pruners, learned compression policies, and long-horizon or embodied tasks as untested. At 5% retention the gap to unpruned is still ~16 points — utilization adaptation does not rescue you from genuine information loss, it only closes the part of the gap that was never information loss to begin with.

**Practical caveats if you want to use this.**
1. The EMA teacher is load-bearing. Shared weights → total collapse. Do not skip it.
2. Do not reach for outcome-reward RL here. With a strong base model and small group sizes most groups are saturated and contribute nothing; you will burn 30 GPU-hours for a format-compliance improvement.
3. LoRA on the LLM only, vision tower frozen — so the fix is cheap and stackable with whatever pruner you already ship.
4. Evaluate at your deployment budget. Every gain here is invisible at 100% context.

**Open questions.** Does visual sensitivity stay a good selector when the pruner is *query-aware* (so the retained tokens change with the question, and possibly mid-trace)? Could the sensitivity score be used at inference — spend more visual budget only at the positions that ask for it — rather than only as a training-time selector? And why does forward vs reverse KL not matter here, when in [[KL Divergence#Forward KL vs Reverse KL|mode-covering vs mode-seeking]] terms it usually does?

## Links

Related: [[Distillation]] · [[Distilling the Knowledge in a Neural Network]] · [[KL Divergence]] · [[GRPO]] · [[On-policy Distillation with Verifiable Reward]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[What Does Privileged Information Add to On-Policy Self-Distillation]] · [[Learning from Teacher Continuations at Student States]] · [[1% of Tokens Can Be Enough- On Gradient Estimation in On-Policy Distillation]] · [[Best Practice Critic Optimization]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[LoRA]] · [[Momentum#⚠️ The other momentum: momentum encoders|EMA weights]] · [[Sparse Attention]] · [[Test-Time Compute]] · [[Chain of Thought]] · [[Fine-Tuning]] · [[Evals]] · [[Cross Entropy]] · [[Mind2Dialogue- Training Human-Aware Language Models by Simulating User Mental States]] · [[Negative Self-Distillation- Learning to Reason by Avoiding Flaws]]

New topics worth writing: visual token pruning, VisionZip, Pass@K as a capability diagnostic, Jensen–Shannon divergence, vision-language model inference cost, privileged-information distillation, zero-advantage saturation in group-relative RL, prefill cost in multimodal models
