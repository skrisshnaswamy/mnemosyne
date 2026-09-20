---
title: "Drift-Constrained Optimization: Only Direction Matters in Fine-Tuning Instruct Models"
authors: ["Yuan et al."]
year: 2026
arxiv: "2609.13680"
url: https://arxiv.org/abs/2609.13680
priority: Good-To-Read
read_on: 2026-09-19
tags: [paper, rl, vision, theory]
---
## The Core Idea

When you fine-tune an already-good instruct model, two things happen at once. It gets better at your task. It also gets *different* — it stops behaving like the model you started from, and the skills you were not training (maths, code, general reasoning) quietly rot. The usual framing treats that second thing as damage to be measured afterwards, or as a penalty term bolted onto the loss.

This paper flips the order. Decide up front how much behaviour change you will tolerate. Then ask: given that fixed budget, how do I spend it best?

$$\min_\theta \mathcal{T}(\theta) \quad \text{s.t.} \quad D_{\text{ref}}(\theta) \le \delta$$

where $\mathcal{T}$ is the usual next-token loss on your task data, and $D_{\text{ref}}$ measures how far the tuned model's output distribution has moved from the frozen reference model, averaged over *general* inputs (not task inputs).

> [!NOTE] Behavioural drift
> $D_{\text{ref}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}_{\text{general}}} \mathrm{KL}(\pi_{\theta_0}(\cdot|x) \,\|\, \pi_\theta(\cdot|x))$ — the KL measured *from* the original model, on data it used to handle well. Note the direction: this is forward, mode-covering KL. It punishes you for dropping probability mass the reference model put somewhere. That is exactly what forgetting looks like. Reverse KL (the one PPO uses) punishes inventing new behaviour instead, which is a different worry. ^behavioural-drift

Now the geometric move. Expand that KL to second order around $\theta_0$ and you get the Fisher information matrix $F$:

$$D_{\text{ref}}(\theta_0 + \Delta\theta) \approx \tfrac{1}{2}\Delta\theta^\top F \Delta\theta$$

So the budget $\delta$ carves out an ellipsoid in parameter space. Every fine-tuning update, whatever the method, is an arrow from the same origin $\theta_0$ inside (or outside) that same ellipsoid. That gives you a shared coordinate system: split any update into $\Delta\theta = \rho v$, where $\rho = \sqrt{\Delta\theta^\top F \Delta\theta}$ is *how far* you moved in behaviour terms, and $v$ (with $v^\top F v = 1$) is *which way*.

If the budget fixes $\rho$, the only thing left to choose is $v$. Local task improvement factorises cleanly:

$$-g^\top \Delta\theta = \rho \cdot \frac{-g^\top u}{\sqrt{u^\top F u}}, \qquad g := \nabla_\theta \mathcal{T}(\theta_0)$$

> [!NOTE] Directional efficiency
> $\eta(u) = \dfrac{-g^\top u}{\sqrt{u^\top F u}}$ — task improvement per unit of behaviour spent. At a fixed drift budget, maximising task performance *is* maximising $\eta$. Its optimum is the natural-gradient direction $u^\star \propto -F^{-1}g$. ^directional-efficiency

The payoff of this framing is a reclassification. Full fine-tuning, LoRA, and freezing-some-layers are not "big change vs small change". They are **different sets of directions you are allowed to point in**, all measured against the same ruler:

- Full FT: $\mathcal{S} = \mathbb{R}^d$, all directions.
- LoRA: directions whose per-layer matrix has rank $\le k$, zero everywhere else.
- Parameter-subset tuning: directions that are zero outside a mask $M$.

And the concrete prediction that makes this testable: **at the same drift, swapping the available directions can flip fine-tuning from failure to success.** Not "make it a bit better" — flip the sign.

## The Methodology

**The stress test.** To make direction matter as much as possible, they pick a setting where naive fine-tuning clearly fails. Take a strong instruct model that earns its scores by writing out a reasoning trace. Fine-tune it on *only* question → answer pairs, with the reasoning stripped out. Then at inference, still let it (and require it to) generate reasoning. Supervision says "output the answer"; evaluation says "think, then answer". Figure 1(b) shows the naive version doing exactly what you would fear: drift climbs monotonically, downstream reasoning drops, and the task never meaningfully beats the starting point.

**The probe: Layer-Selective Tuning (LST).** Rather than solve for $-F^{-1}g$ (which nobody can do at 8B scale), they use a deliberately crude way of changing the direction family: freeze whole transformer layers. Searching all $2^L$ subsets is hopeless, so they restrict to two-segment masks:

$$M(l_1, l_2) = \{1,\dots,l_1\} \cup \{l_2,\dots,L\}$$

i.e. some bottom layers plus some top layers, middle frozen. Notation: `b16` = bottom 16 layers contiguous. `b4t16` = **two stages** — first fine-tune the bottom 4 layers, then from that checkpoint fine-tune the top 16, middle always frozen. (This is worth flagging: `b4t16` is a curriculum as well as a mask.)

**Measuring drift in practice.** To compare direction families under one ruler they freeze the embedding and the output head $(W_0, b_0)$ when *estimating* drift, reading out hidden states through the original head:

$$\hat{D}(\theta) = \mathbb{E}_{x}\, \mathrm{KL}\big(\mathrm{softmax}(h_{\theta_0}(x)W_0^\top + b_0) \,\|\, \mathrm{softmax}(h_\theta(x)W_0^\top + b_0)\big)$$

And the empirical efficiency they report everywhere is $\eta = \Delta\text{Task} / \sqrt{\text{KL}}$.

**Setup.**
- Models: Qwen3-8B, Qwen3-14B. Transfer check on Intern-S1-mini-8B. 8× H200.
- Science: 300K samples from SmolInstruct (chemistry), scored on a 14-task aggregate.
- Translation: ~2.8M pairs from Lego-MT covering 100+ languages; evaluated on FLORES-101 with xCOMET, averaged over four pivots (English, Chinese, Nepali, Cebuano) in both `lg→x` and `x→lg` directions.
- Capability preservation: `General Avg` $= \frac{1}{3}\left(\frac{\text{AIME25+AIME26}}{2} + \frac{\text{LCBv5+LCBv6}}{2} + \text{BBEH}\right)$.
- Analysis experiments: 300K MegaScience train / 50K held-out, scored by teacher-forced next-token accuracy (feed the gold prefix, check the next token) — quieter signal than free generation.
- Baselines: full fine-tuning (FFT), LoRA rank 64, and ASFT (anchored SFT — drift as a KL penalty in the objective with weight $\alpha$).

## Ablation Studies and Experiments

**Translation, Qwen3-8B** (reference: general 42.22, xCOMET 47.07 / 51.40):

| Method | KL | ΔTask (lg→x / x→lg) | General Avg | xCOMET |
|---|---|---|---|---|
| FFT | 0.32 | −5.03 / −4.38 | **4.42** | 42.05 / 47.02 |
| LoRA r=64 | 0.11 | −6.30 / −4.90 | 34.50 | 40.78 / 46.50 |
| ASFT α=0.2 | 0.09 | −0.23 / −0.50 | 31.25 | 46.84 / 50.90 |
| LST b16 | 0.10 | −9.99 / −8.75 | 39.13 | 37.09 / 42.65 |
| LST b4t8 | 0.12 | +3.27 / +2.27 | 42.32 | 50.35 / 53.67 |
| **LST b4t16** | 0.22 | **+5.59 / +4.20** | 38.85 | **52.66 / 55.60** |

Full fine-tuning is the headline disaster: general capability goes 42.22 → **4.42**. The model is effectively destroyed, and translation got *worse* too.

The matched-drift comparison is the real evidence. LoRA sits at KL 0.11, continuous `b16` at KL 0.10, split `b4t8` at KL 0.12. Same budget, wildly different outcomes: −6.30, −9.99, and **+3.27** respectively. Drift magnitude explains nothing here; direction explains everything.

At 14B, `b4t16` reaches xCOMET 56.51 / 58.31 vs the 51.85 / 55.21 reference. Both sizes beat the dedicated translation systems — Seed-X-PPO-7B (47.76 / 51.31), Tower-Plus-9B (46.95 / 52.44), Aya-Expanse-8B (36.44 / 43.30) — using only question→answer supervision.

**What did not work:**

- **KL as a penalty term (ASFT) just slides along the trade-off.** On SmolInstruct with Qwen3-8B: $\alpha=0.05$ gives task 23.76 but general collapses to **16.32**; $\alpha=0.5$ preserves general (41.06) but task falls to 17.21, *below* the 19.34 baseline. You can pick your point on the curve; you cannot leave the curve. LST does leave it — `b4t16` gets task 29.61 *and* general 44.03, above the untuned 42.22.
- **LoRA is inconsistent.** It behaves well on chemistry (general 42.65, essentially preserved) and badly on translation (task down 6.30 points). Low rank is a direction restriction, but not reliably an *efficient* one.
- **Continuous layer blocks are not the point — the split is.** `b16` updates roughly as many parameters as `b4t12` and does far worse on translation. It even loses to `b4t8`, which touches fewer parameters.
- **Training `lm_head` / input embeddings is pure waste.** Across all configs it raises drift and slightly *lowers* accuracy. Under a drift budget, touching the readout is an inefficient direction.

**The ablations that complicate the story.** Two findings pull in opposite directions and the paper is honest about it:

1. *At matched drift (KL ≈ 0.27), contiguous beats split in absolute accuracy* — `b24` > `b12+t16`.
2. *At matched trainable-layer count, split beats contiguous in accuracy per unit drift* — `b4+t20` gets higher accuracy at equal or lower drift than a contiguous 24-layer block.

The Pareto frontier reconciles them: **in the low-drift regime split configurations win on efficiency; as the budget grows, contiguous updates dominate on raw accuracy.** No single structure is universally best. A gap sweep (varying how many middle layers are frozen) shows the same trade-off continuously — bigger gap, better efficiency, worse final accuracy, presumably because bottom and top updates can no longer form coherent joint representations.

**Two more results worth having:**
- `b4t16` transfers to Intern-S1-mini-8B with no re-search of the layer selection, improving the task and holding general capability.
- The LST model is a better *starting point* for RL. `Qwen3-8B + LST` already beats `Qwen3-8B + RL` (the published WALAR model) on every translation direction, and running the same RL pipeline from the LST checkpoint gives larger gains than running it from base — best final numbers across the board.

## Worth Remembering

**The gap between the framing and the practice is large, and you should hold it in mind.** Nothing here actually enforces $D_{\text{ref}}(\theta) \le \delta$ during training. $F$ is never computed. $-F^{-1}g$ is never approximated. The drift budget is *measured after the fact* and the comparisons are post-hoc "these two runs happened to land at similar KL". The theory is a clean lens for interpreting results, not an algorithm. The actual algorithm is "freeze some middle layers and train in two stages", chosen by a small grid search.

**The empirical $\eta = \Delta\text{Task}/\sqrt{\text{KL}}$ is a nice cheap diagnostic** even if you ignore the rest. It gives you a single number for "how much capability am I burning per point of task gain", comparable across methods that are otherwise incomparable. You could compute this today on any fine-tuning run with a held-out general corpus and a frozen reference.

**`b4t16` is confounded with sequential training.** It is not just a mask — it is stage 1 (bottom 4) then stage 2 (top 16, from the stage-1 checkpoint). Nowhere do they run "train bottom 4 and top 16 jointly" as a control. Some of the win may be curriculum, not geometry.

**The forward/reverse KL distinction is the most portable idea in the paper.** Appendix C makes the case properly: both agree to second order at $\theta_0$, so the local metric is the same. The difference is (a) what they penalise globally — anchored/forward KL punishes losing the reference's support, i.e. forgetting; reverse KL punishes going off-distribution — and (b) whether you measure from a *fixed* origin or from the current iterate. Stepwise KL (as in PPO) uses $F(\theta_t)$, which drifts as you train, so equal KL budgets at different steps are not equal displacement. Anchoring at $\theta_0$ gives every method one shared ruler.

**Practical caveats:** everything is Qwen3-scale; no larger models, no multimodal, no other domains. The layer-subset search is combinatorial and they only explored two-segment masks, so there is no claim these are the *best* directions, only that effective ones exist and are not isolated points. And the SmolInstruct FFT result (task 19.34 → 59.65) is a reminder that if you genuinely do not care about general capability, full fine-tuning still wins the target metric by a mile.

**Open question:** the drift is measured over some $\mathcal{D}_{\text{general}}$ that is never specified in detail. Since the whole coordinate system is defined by that distribution, the choice of general corpus is a free parameter that could move every number in the paper.

## Links

Related: [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[LoRA]] · [[KL Divergence]] · [[Fine-Tuning]] · [[Trust Region Policy Optimization (TRPO)]] · [[Proximal Policy Optimization Algorithms]] · [[Old Optimizer, New Norm- An Anthology (Muon)]] · [[Continual Learning Mechanisms Compose for Long-Horizon Memorization]] · [[Training language models to follow instructions with human feedback]] · [[Regularization]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Derivative]] · [[Cross Entropy]]

New topics worth writing: Natural gradient descent and the Fisher information matrix, Anchored SFT and KL-penalised fine-tuning, Elastic Weight Consolidation and forgetting-aware regularisers, Layer-wise heterogeneity in transformers, xCOMET and reference-based MT evaluation
