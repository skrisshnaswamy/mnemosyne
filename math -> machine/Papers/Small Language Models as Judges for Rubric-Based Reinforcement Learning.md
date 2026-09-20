---
title: "Small Language Models as Judges for Rubric-Based Reinforcement Learning"
authors: ["Xie et al."]
year: 2026
arxiv: "2608.30005"
url: https://arxiv.org/abs/2608.30005
priority: Good-To-Read
read_on: 2026-09-06
tags: [paper, llm, rl]
---
## The Core Idea

Some tasks have no exact answer to check. "Write a good research report" has no unit test. So people write a **rubric** — a list of criteria with weights — and ask a big language model to grade each criterion. Add up the weighted grades, and you have a scalar reward you can run RL on.

That works, but it is expensive. Every RL step generates several responses, each response has ~8 criteria, and each criterion needs a judge call. If the judge is GPT-4o or an 8B model that writes out "Yes/No" verdicts, reward computation dominates the training budget.

The trick here: **you do not need the small model to *say* the verdict. You just need to read it out of the hidden state.** Freeze a Qwen3-1.7B. Feed it `(question, response, one criterion)`. Take the hidden vector at the last token. Train a single linear classifier on top to predict "does this response satisfy this criterion?". That is the whole judge.

Why this matters: a 1.7B model asked to *generate* a verdict on the science rubrics gets 0.443 macro-F1 — barely better than coin flipping. The **same frozen model**, read through a linear probe, gets **0.835**. The knowledge was in there. Generation was the bottleneck, not the representation. Small models know more than they can say.

And it works downstream. Used as the reward model for [[Proximal Policy Optimization Algorithms|GRPO]], the 1.7B probe trains a policy from 0.232 → **0.643** rubric score. An 8B generative judge reaches only **0.594** — and burns **10.7×** more judge compute to get there.

> [!NOTE] Pointwise rubric judging
> Given a prompt $q$, one candidate response $y$, and one criterion $c_j$, predict whether $y$ satisfies $c_j$. Contrast with *pairwise* judging, which only says "A is better than B" and throws away which criteria failed. Pointwise labels can reconstruct pairwise preferences; the reverse is impossible. ^pointwise-rubric

> [!NOTE] Probe judge
> A frozen LM backbone plus a lightweight linear head trained on hidden states. No LM weights change. The head outputs a satisfaction probability, which becomes the criterion-level reward. ^probe-judge

## The Methodology

**The scoring rule.** A rubric is $C(q) = \{(c_j, w_j)\}_{j=1}^{m_q}$ — criteria and weights. The scalar score is a weighted average:

$$\widehat{R}(q,y) = \frac{\sum_{j=1}^{m_q} |w_j|\, p_j(q,y)}{\sum_{j=1}^{m_q} |w_j|}$$

Absolute weights are used because RaR-Science has "pitfall" criteria with negative weights. Those get flipped into *avoidance* criteria — "the response avoids this pitfall" — so higher always means better.

**Three ways to read a verdict out of the same backbone.**

1. **Generative.** Prompt the model with one criterion, parse a Yes/No out of the generated text. Unparseable → counted wrong. One generation per criterion.

2. **Logprob.** Same prompt, stop before the answer token, read the next-token probabilities of the fixed words " Yes" and " No":
$$\Delta_j = \log P(\text{Yes} \mid q,y,c_j) - \log P(\text{No} \mid q,y,c_j)$$
Soft score $p_j = \sigma(\Delta_j)$; binary threshold at $\Delta_j = 0$. No parsing needed.

3. **Probe.** Same prompt, run forward with hidden states on, grab $h_\ell(q,y,c_j)$ at the **final non-padding token** of layer $\ell$. Standardise features using training-set mean/std only. Train a linear binary classifier with [[Cross Entropy|binary cross-entropy]], positive-class weighted. Full-batch [[Decoupled Weight Decay Regularization (AdamW)|AdamW]], 200 epochs, lr 0.01, weight decay 0.01. Pick the layer and the decision threshold on the **dev** split by macro-F1. Held-out is touched once.

They also test SFT as a fourth axis: [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]]-tune the backbone to emit the full verdict vector, then re-run all three readouts on the tuned backbone.

**The datasets, because none existed.** Pointwise rubric judging needs four things at once: explicit criteria, several responses under the *same* rubric, a label for every response–criterion pair, and a weighted aggregation rule. RewardBench has pairs but no criteria. HealthBench has weighted rubrics but one response. RubricEval has per-criterion labels but no weights. So:

- **PointRubric**, from OpenRubrics. 3,000 questions rewritten by GPT-4o into a fixed 6-criterion shape: 2 hard (weight 3) + 4 soft (weight 1). Four responses per prompt: a forced full-score one (10/10), a forced low-score one (≤3), a rubric-guided partial one, and an unguided one that never sees the rubric. Final: 1,042 questions, 4,168 responses, 25,008 labels. Split by *question* — 417 / 104 / 521.
- **RaR-Science-Static**, from RaR-Science. 1,500 science questions, each with the dataset reference answer plus a greedy Qwen3-4B answer. 6–11 criteria per rubric, mean 7.52. Split 1,000 / 200 / 300.

Labels come from GPT-4o. The authors are explicit that this is an **operational target**, not truth: the question is "can a small local judge replace an expensive pipeline", not "does it match humans". A human audit of 600 decisions gets 90.9% weighted agreement with GPT-4o and 96.0% pairwise agreement, with disagreement concentrated in soft criteria (completeness, clarity) rather than hard factual ones.

**The RL setup.** Qwen3-4B-Base actor, GRPO, VERL framework, 1,600 RaR-Science train prompts, 4 responses per prompt, batch 8, lr $1\times10^{-6}$, [[KL Divergence|KL]] coefficient 0.02 on the actor, 5 epochs, comparison at step 800. Max prompt 1024 tokens, max response 1536. **The only variable across runs is the reward judge.** Final scoring by GPT-4o on 100 reserved prompts.

The RL reward probe is trained on just **500 GPT-4o-labelled responses** (3,766 criterion labels, roughly **$2.43** of API spend). Once fitted it is frozen and reusable.

## Ablation Studies and Experiments

**Static judging (Table 3).** Weighted criterion accuracy on PointRubric, macro-F1 on RaR-Science-Static.

| Model | PointRubric Gen / Logprob / Probe | RaR-Sci Gen / Logprob / Probe |
|---|---|---|
| Qwen3-0.6B | 0.370 / 0.582 / **0.802** | 0.413 / 0.433 / **0.793** |
| Qwen3-1.7B | 0.518 / 0.766 / **0.875** | 0.443 / 0.449 / **0.835** |
| Qwen3-4B | 0.790 / 0.893 / **0.913** | 0.496 / 0.738 / **0.851** |
| Qwen3-8B | 0.888 / 0.907 / **0.914** | 0.609 / 0.756 / **0.864** |

Read the two benchmarks differently. PointRubric is a clean, controlled format — generation scales normally with size, and SFT is *very* effective (1.7B SFT-Generative jumps 0.518 → **0.902**, beating everything). RaR-Science-Static is the realistic downstream format, and there **generation collapses at every size**. An 8B generative judge gets 0.609. A 0.6B probe gets 0.793.

**Where the generative judge goes wrong.** Paired on 4,518 held-out decisions: 33.0% correct only under Probe, 7.8% correct only under Generative, 51.3% both right, 7.9% both wrong. The generative errors are **33.7% false positives** vs 4.2% false negatives — it says "yes" too easily. Concrete case: the criterion demands an explanation that fusion converts mass into energy; the response only says "conversion of hydrogen into helium via nuclear fusion". The topic is mentioned so Generative accepts it. Probe and GPT-4o reject it.

The probe is not free of failure: it marks a wavefunction-form criterion unsatisfied when the response literally gives the exponentially decaying wavefunction. Its errors split more evenly (6.7% FP, 9.0% FN).

**Probe design ablations (1.7B, RaR-Sci macro-F1).** Data: 50 q → 0.767, 100 → 0.805, 250 → 0.818, 500 → 0.828, 1000 → 0.834. **The curve is nearly flat after 100 questions.** Design: linear last-token 0.834; MLP-32 0.834; MLP-64 0.840; last-4-layer mean 0.839; **mean pooling 0.761** — the one clear loser. So: fancy heads buy ~0.006, and pooling across the sequence actively destroys the signal. The judgment lives in the last token.

Layer sweep is a broad late-layer plateau (layers 16–26 all sit 0.828–0.841, peak at 18). Not a lucky single layer.

**The main RL comparison (Table 5).** Same actor, same everything, only the reward judge changes.

| Reward | RaR-Science score |
|---|---|
| Base actor, no RL | 0.232 |
| 0.6B Probe | 0.562 |
| **1.7B Probe** | **0.643** |
| 4B Probe | 0.588 |
| 8B Probe | 0.506 |
| 8B Generative | 0.594 |

**This is the most interesting result in the paper and it is a non-monotonicity.** The 4B and 8B probes are *better* static judges (0.851, 0.864 macro-F1) but train *worse* policies. Why? Look at the step-800 rollout rewards:

| Reward judge | Mean | Within-group SD | Groups fully saturated |
|---|---|---|---|
| 1.7B Probe | 0.911 | 0.032 | 12.5% |
| 8B Probe | 1.000 | 0.0005 | **100%** |
| 8B Generative | 0.706 | 0.102 | 12.5% |

The 8B probe hands out ~1.0 to everything. GRPO computes advantages by normalising rewards *within* a group of rollouts — if all four rewards are identical, the advantage is zero and there is no gradient. The 8B probe is an accurate judge that is a **useless reward function**. A more accurate judge with no variance teaches nothing.

**Efficiency (Table 8).** Through step 800: 1.7B Probe used 2.33 judge-hours, 8B Generative used 24.98 — **10.7×**. On a validation pass over 50 responses / 385 criteria: 31.1s vs 492.0s — **15.8×** (0.08s vs 1.28s per criterion). Wall-clock is only **1.26×** (8h52m vs 11h12m) because reward calls are parallelised and rollout/optimisation/checkpointing stay in the critical path.

**Transfer.** Policy: the probe-reward policy improves GPQA-Diamond from 0.335 → 0.388 (4 runs with permuted answer orders), so the gain is not just rubric-format overfitting. Judge: a science-trained probe scores 0.718 macro-F1 on RaR-Medicine with zero medicine labels, vs 0.782 for the in-domain probe. Medicine→Science is 0.702 vs 0.754 in-domain.

**Response-generator shift.** Extend the held-out bank with Mistral-7B-Instruct and OLMo-2-7B responses (1,200 total). Unchanged 1.7B probe: **0.741** macro-F1. Generative: 0.394. Logprob: 0.424. Harder for everyone, ranking unchanged.

**What did not work.**
- **Mean pooling** for the probe representation — 0.761 vs 0.834. Averaging over the sequence dilutes the decision.
- **Bigger probes as rewards** — 8B probe saturates and kills GRPO's signal.
- **Rubric-RM as a baseline.** The existing pairwise rubric reward model frequently fails to emit parseable output: parse rates 0.396–0.727. Counting failures as wrong, its accuracy is 0.210–0.487 vs 0.888–0.924 for the 1.7B probe. On parseable outputs only it reaches 0.523–0.670, so it does hold real signal — it just cannot be plugged into an RL loop.
- **GPT-5 as the data constructor.** It refused or failed to produce deliberately low-quality responses, which the benchmark needs. They fell back to GPT-4o.
- **SFT on the downstream format.** SFT-Generative at 1.7B goes 0.443 → 0.708 on RaR-Science — a real gain, but still far below the untuned probe at 0.835. On PointRubric, SFT actually *hurts* the probe (0.875 → 0.845).

## Worth Remembering

**The single most transferable lesson:** static judge accuracy is the wrong model-selection metric for a reward model. What GRPO needs is *within-group reward spread*. A judge that is right 86% of the time but assigns 1.000 to every rollout produces zero advantage. Before adopting any learned reward model, log the within-group standard deviation of the rewards it emits on on-policy rollouts. This generalises well beyond rubrics — it is the same failure mode as a saturated [[Training language models to follow instructions with human feedback|reward model]] in RLHF, but here it is measured directly.

**Second:** the gap between what a small model represents and what it can generate is enormous, and it is *format-dependent*. On the tidy PointRubric format the 1.7B generative judge does fine (0.902 after SFT). On the messy real rubrics it collapses to 0.443 while its own hidden states carry 0.835. Do not conclude "small models cannot judge" from generative evaluations alone.

**Admitted limitations.**
- The supervision target is GPT-4o, not humans. The audits (90.9% weighted criterion agreement, 96.0% pairwise, plus a GPT-5 cross-check at 0.853 macro-F1 agreement) support the target but do not replace human evaluation. Any bias in GPT-4o's grading is baked into the probe.
- One actor family, one rubric-RL setup, one policy scale. GPQA and RaR-Medicine transfer is real but modest — this is not domain-invariant rubric judgment.
- Efficiency numbers are one implementation on one cluster. The 10.7× judge-time ratio is not a serving constant.
- The paper does not test adversarial reward hacking. The probe is a frozen linear classifier on a frozen backbone — a policy trained long enough could plausibly find inputs that light up the probe without satisfying the rubric. They show one artifact case (the 8B-probe policy emitting repeated assistant-prefix junk while its internal reward saturated) which is exactly what early reward hacking looks like.

**Practical caveats if you want to build this.**
- Budget: ~$2.43 of teacher labels for the RL reward probe. Data ablation says 100 questions already gets you 0.805 of an eventual 0.834. Start small.
- Use last-token, not mean-pooled. Use linear, not MLP.
- Select the layer on a dev split, but do not agonise — the plateau is wide.
- Pointwise scales better than pairwise in the RL loop: with a rollout group of $k$, pairwise needs $k(k-1)/2$ comparisons, pointwise needs $k$ scores. At $k=4$ that is 6 vs 4, and pointwise also tells you *which* criterion failed.
- Refit when the rubric format changes substantially. Moderate domain shift (science→medicine) costs ~0.06 macro-F1 without refitting.

**Open questions.** Does probe reward survive long RL runs, or does the policy eventually hack the linear head? Would an ensemble of probes at different layers restore variance for larger backbones? And why exactly does the 8B probe saturate — is it overconfidence from a better-separated representation, or a threshold-calibration artifact that a temperature fit would fix?

## Links
Related: [[Proximal Policy Optimization Algorithms]] · [[Training language models to follow instructions with human feedback]] · [[Direct Preference Optimization (DPO)]] · [[Cross Entropy]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Distilling the Knowledge in a Neural Network]] · [[KL Divergence]] · [[Constitutional AI- Harmlessness from AI Feedback]] · [[On-policy Distillation with Verifiable Reward]] · [[Sparse Readout Prism- Explaining Logit-Lens Scores in Features Instead of Tokens]] · [[Saliency]]

New topics worth writing: GRPO (Group Relative Policy Optimization), Linear probing of hidden states, RLVR (reinforcement learning with verifiable rewards), LLM-as-a-judge, Reward hacking and overoptimization, Macro-F1 and class-imbalanced metrics, Matthews correlation coefficient, Bootstrap confidence intervals, Rubrics as Rewards (RaR)
