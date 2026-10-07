---
title: "RoboFollow: Unveiling the Instruction Following Mirage in Embodied Agents"
authors: ["Guo et al."]
year: 2026
arxiv: "2609.25636"
url: https://arxiv.org/abs/2609.25636
priority: Good-To-Read
read_on: 2026-09-28
tags: [paper, vision]
---
## The Core Idea

A robot policy can score 98% on a manipulation benchmark and still be ignoring the words you gave it.

The reason is structural, not a modelling accident. In most manipulation benchmarks, once you look at the starting scene there is really only one sensible thing to do. A red block sits next to a bowl; the only trained behaviour is "put the block in the bowl". The instruction adds no information. A policy that maps pixels → actions and treats the sentence as a task ID will look perfect. This is [[Shortcut Learning in Deep Neural Networks#^shortcut|shortcut learning]] with language as the discarded input.

The paper gives this a number. **Scene entropy** is the conditional entropy of the training task label given the *task-independent* description of the scene:

$$H_{\text{scene}} = H(T \mid S) = -\sum_s p(s) \sum_t p(t\mid s)\log_2 p(t\mid s)$$

where $S$ is the scene spec (which objects, where they can spawn, initial state predicates) with all instructions and goals stripped out, and $T$ is the task label. If each scene supports one task, $H_{\text{scene}} = 0$ and language is provably redundant. If a scene supports 16 different trained tasks with equal demos, that scene contributes $\log_2 16 = 4$ bits of ambiguity that *only* language can resolve.

> [!NOTE] Scene entropy
> How much task uncertainty is left after you have seen the scene but not the instruction. Zero bits means vision alone determines the task, so a policy can score highly with no language understanding at all. ^scene-entropy

Measured under the same task-label definition, LIBERO's four suites average **0.880 bits**. RoboFollow, built by deliberately pairing one scene with many kinematically distinct task branches, is **3.782 bits**.

Two things follow. First, entropy is a *dataset design* knob, not a score — you raise it so that the benchmark can no longer be gamed. Second, once language is necessary, you can grade comprehension separately from motor skill. RoboFollow does this with **Intent** and **Execution** scores per stage, so "reached for the wrong cube" and "reached for the right cube and fumbled it" stop looking identical.

The headline result: nine current VLA and World-Action Models look excellent in-distribution and fall apart the moment object positions are swapped or the instruction is recombined from parts it already saw. Every mitigation the authors tried — a much stronger VLM backbone, QA co-training, LangForce, [[Classifier-Free Guidance|CFG]] — failed to close the gap. One made it worse.

## The Methodology

Built on the RoboTwin 2.0 bimanual simulator. Four scene families, 75 training task labels, 6 task-independent scene groups with $K_s = (16, 16, 16, 16, 7, 4)$ labels each, 50 demonstrations per task = 3,750 training episodes.

Per-episode noise is deliberately tiny: objects jitter 1–2 cm, and the instruction is sampled from 3 paraphrase templates. Paraphrases and jitter do **not** create new scene groups or task labels — that is what keeps the entropy computation honest.

**The four scene families**, each targeting a different kind of grounding:

| Scene | Tests | Example |
|---|---|---|
| 1 | Extrinsic spatial relations | "pick up the block to the right of the red ball" — candidate blocks are visually identical, so only the relation identifies the target |
| 2 | Intrinsic attributes + action choice | colour × size × shape × {pick, push, stack, place}; no single attribute ever suffices |
| 3 | Trajectory and orientation constraints | "move it *avoiding* the red slab, long side facing right" — the final state is not enough to grade |
| 4 | Elementary logic | "first A then B", "not A", "if A then B else C" |

**The L0–L3 protocol.** Not a difficulty ladder — four separate generalisation axes:

- **L0** — test distribution = train distribution. In-distribution execution.
- **L1** — *same instruction, swapped object positions.* Did the model bind "right of the red ball" to the red ball, or to a memorised image region?
- **L2** — *same layout, recombined instruction.* Every atom appeared in training ("red", "cube"), the combination did not ("red cube"). Crucially the **target action is unchanged** — in the worked example the target is still the same small red cube in the same place, only the words describing it are new.
- **L3** — both perturbed.

The authors are explicit that in every level the required action is inside the demonstrated repertoire. So a failure cannot be "it never learned that motion".

**Stage-wise scoring.** Each task splits into up to three stages: (1) grounding the source object or first contact, (2) the action-specific objective, (3) the post-manipulation outcome — retraction, final pose, orientation.

> [!NOTE] Intent Score vs Execution Score
> **Intent** = did the policy select the correct object / relation / waypoint / logical branch and move the semantically right way, even if it physically failed. **Execution** = was the subgoal physically completed. Separating them turns "it failed" into "it misunderstood" versus "it fumbled". ^intent-execution

A dedicated finishing stage is worth **20%** of the score: after completing the request the policy must stop. Continuing to fiddle costs you credit. Completion Rate (CR) is a third, action-dependent binary metric — pickup tasks need final IS and ES both $\geq 0.8$; others use target-layout or action-progress signals.

**Models evaluated**, all fine-tuned on the full 3,750 episodes to their own in-distribution plateau: $\pi_0$, $\pi_{0.5}$, GR00T N1.6, OpenVLA-OFT, X-VLA, ACoT-VLA, Lingbot-VLA (VLAs), plus Motus and FAST-WAM (world-action models). Steps range 8k–32k, batch 16–64.

## Ablation Studies and Experiments

**The main collapse.** Averaged Intent Score across all scenes and levels, best models: $\pi_{0.5}$ 55.7, $\pi_0$ 48.4, ACoT-VLA 53.5. But the averages hide the shape. Picking the sharpest cases:

| Model | Scene | L0 IS | L1 IS | L2 IS | L3 IS |
|---|---|---|---|---|---|
| $\pi_0$ | 1 (spatial) | 98.2 | **0.0** | 4.1 | 15.7 |
| $\pi_{0.5}$ | 1 (spatial) | 99.1 | 45.5 | 58.0 | 44.9 |
| $\pi_{0.5}$ | 2 (attributes) | 100.0 | 77.5 | 32.4 | 34.2 |
| GR00T N1.6 | 1 | 71.4 | 0.0 | 24.0 | 36.0 |
| FAST-WAM | 1 | 83.1 | 0.0 | 13.0 | 41.9 |
| ACoT-VLA | 1 | 100.0 | 8.7 | 60.9 | 48.0 |

Four models score **exactly 0.0** Intent on Scene 1 L1. Swap the red ball and the green ball and the policy still reaches for the same patch of image. That is about as clean a demonstration of coordinate memorisation as you could ask for.

Note the failure is in *Intent*, not Execution. Execution Score tracks Intent closely, so this is not a motor-control story.

**The sanity check that validates the benchmark.** They ran $\pi_{0.5}$ with the instruction blanked out. Average IS drops to **12.3** (from 55.7), CR to 5.2. On a low-entropy benchmark an empty-language policy would still often succeed. Here it cannot. This is the experiment that proves the scenes really do require language — without it the whole paper would be arguable.

**Is it the VLM's fault?** They built a 20-question visual QA probe on Scene 2 (object identities, colours, spatial relations). Accuracy:

- base PaliGemma: **1/20**
- pre-trained $\pi_{0.5}$ backbone: **2/20**
- fine-tuned $\pi_{0.5}$ backbone: **3/20**
- Qwen3-VL-4B, zero-shot: **19/20**

So the small backbones genuinely cannot see the scene. But — and this is the two-layer finding — even on the questions the fine-tuned backbone answers *correctly*, roughly **half** of those episodes still produce the wrong manipulation. Comprehension does not propagate to action. Motus shows the same disconnect despite a completely frozen VLM.

**So swap in the good VLM.** They built Qwen-GR00T: Qwen3-VL-4B backbone + GR00T diffusion action head, with visual QA pairs co-trained alongside trajectories to prevent [[Fine-Tuning#^catastrophic-forgetting|catastrophic forgetting]] of language ability. Scene-2-only fine-tune, Intent Score:

| Config | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| $\pi_{0.5}$ (baseline) | 93.6 | 75.7 | 37.8 | 45.8 |
| Qwen-GR00T | 67.9 | 6.3 | 31.1 | 14.8 |
| + COCO QA co-train | 60.1 | 3.5 | 26.4 | 18.9 |
| + Scene-2 QA co-train | 68.2 | **0.0** | 21.8 | **0.0** |
| + mixed QA | 66.6 | 2.9 | 30.8 | **0.0** |
| LangForce | 65.1 | 6.3 | 18.6 | **0.0** |
| $\pi_{0.5}$ + CFG 1.2 | 55.9 | 50.4 | 31.2 | 31.2 |
| $\pi_{0.5}$ + CFG 1.5 | 43.6 | 37.8 | 28.6 | 24.6 |

A backbone that answers 19/20 on the QA probe generalises *worse* than one that answers 3/20. QA co-training does not help and sometimes drives L1/L3 to zero. This is the most useful negative result in the paper: **VLM comprehension is not the bottleneck, and fixing it with more QA supervision does not touch the action head.**

**CFG is actively harmful here, for an interesting reason.** [[Classifier-Free Guidance]] extrapolates away from the unconditional prediction: $\hat{\epsilon} = \epsilon_\varnothing + w(\epsilon_c - \epsilon_\varnothing)$. On a low-entropy benchmark the unconditional branch is a stable, near-correct baseline because vision alone almost determines the task, so the residual $\epsilon_c - \epsilon_\varnothing$ is a clean semantic correction. On high-entropy scenes, dropping the language leaves the model guessing among 16 valid branches — the unconditional prediction is multimodal mush, and the residual is dominated by noise. Guidance scale 1.2 drops L0 IS from 93.6 to 55.9. Scale 1.5 drops it to 43.6.

**LangForce** — a method that strengthens the statistical correlation between instruction text and generated motion — barely moves anything. The authors' reading: correlation between text and action distribution is not the same as compositional semantics, and nothing in the objective supervises the latter.

**More data / more paraphrases / longer training** ($\pi_{0.5}$, Scene 2 only, Intent Score):

| Factor | Value | L0 | L1 | L2 | L3 |
|---|---|---|---|---|---|
| Demos per task | 25 → 50 | 88.4 → 93.6 | 69.8 → 75.7 | 33.5 → 37.8 | 39.8 → 45.8 |
| Instruction variants | 1 → 3 → 5 | 93.1 / 93.6 / 93.1 | 72.8 / 75.7 / **79.0** | 31.6 / 37.8 / 32.0 | 41.5 / 45.8 / **53.0** |
| Training steps | 2k → 4k → 6k | 86.4 / 92.4 / 93.6 | 79.2 / 77.6 / 75.7 | 31.5 / 35.7 / 37.8 | 38.9 / **47.6** / 45.8 |

Doubling demos moves the L0→L2 gap from 54.9 to 55.8 points — i.e. not at all. More paraphrases help L1 and L3 (the visual-swap axes) but L2 wobbles non-monotonically. Longer training improves L0 and L2 while *hurting* L1. At 6k steps the L0→L2 and L0→L3 gaps are still 55.8 and 47.8 points.

**Real-robot pilot.** $\pi_{0.5}$, right arm only, 8 training instructions and 8 held-out instructions, 5 trials each. Success 20/40 (50%) → 6/40 (15%). Split by action type: pick 14/20 vs 6/30, stack 6/20 vs 0/10. In one held-out rollout the policy went for the blue cube when asked for the red one.

## Worth Remembering

**What the Intent/Execution split actually buys.** It rules out the easy defence. When a model goes from 98.2 to 0.0, you cannot say "the arm was imprecise" — Intent was 0.0 too, meaning it did not even *aim* at the right object. Compare this to [[Agent Evaluation#^grade-the-route-too|grading the route rather than the answer]]: the final state is a lossy summary of what the policy was trying to do.

**The generalisation axes are not ordered.** L2 is often harder than L3, and L1 is sometimes catastrophic while L2 is fine ($\pi_0$ Scene 1: L1 = 0.0, L2 = 4.1, L3 = 15.7 — L3 is *better* than L1). So "L3 is the hardest" is wrong. They measure different things: L1 is entity binding, L2 is compositional semantics. A model can memorise coordinates while still parsing new adjective-noun pairs, or vice versa.

**Honest limitations the authors state.**

- The L0→L1/L2/L3 gap is measured *under their fine-tuning setup*. They point at $\pi_{0.7}$, which reports better instruction following from much broader data and richer context conditioning, and say their setting is different. This is a claim about fine-tuning on a narrow high-entropy dataset, not a claim about the ceiling of the architecture.
- They never tested **increased structural layout diversity during training** — the most obvious fix — because doing so would mean training on layouts that are currently held out as L1 test scenes, invalidating the split. This is a real hole: the paper's central negative result may partly be an artefact of training on exactly six scene layouts.
- The real-robot pilot compares **different instruction sets** with different pick/stack mixes, not matched pairs. 50% vs 15% is suggestive, not controlled. Wilson 95% intervals [35.2, 64.8] and [7.1, 29.1] barely separate, and even those assume independent trials.
- Scenes are deliberately simple: no contact-rich manipulation, no long horizons, no open vocabulary. That simplification is the point — it removes execution confounds — but it means the numbers are not manipulation-competence numbers.

**Where this sits in the literature.** LIBERO-PRO and LIBERO-Plus perturb *at test time* on a low-entropy training set. RoboFollow's contribution is building the ambiguity **into training**, so a shortcut is never available to learn in the first place. [[Breaking the Vision-Action Shortcut- Latent Interface Training for Generalizable Robotics Foundation Models#^vision-action-shortcut|the vision-action shortcut]] paper attacks the same disease from the architecture side; this one builds the diagnostic.

**The generalisable idea for anyone building an eval.** Before trusting a language-conditioned benchmark, compute $H(T \mid S)$ on your own training set, and run the empty-instruction control. If blanking the language barely dents the score, the benchmark is measuring perception, not instruction following — the same class of mistake as [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|weak baselines in recsys]] or [[Do ImageNet Classifiers Generalize to ImageNet|test sets that leak into the model selection loop]]. This transfers straight to ranking and recsys: if your "personalised" model scores the same with the user features zeroed out, the personalisation is a mirage.

**Practical caveat if you were going to use this.** The CR metric is hand-crafted per action type, with score fallbacks and a 20% "stop when done" term. It is defensible but not portable — if you extend the benchmark you will be writing bespoke stage definitions. IS and ES are the numbers to report.

**Open questions.** Where exactly does comprehension fail to reach the action head? The finding that a 19/20 backbone underperforms a 3/20 one suggests the flow-matching / diffusion action head is conditioning on language only weakly and cannot be forced to do more by upstream supervision. Does an explicit grounded alignment loss between instruction tokens and the object the gripper approaches fix it? Would more training layouts close the L1 gap while leaving L2 open — i.e. are entity binding and compositionality genuinely separate failures?

## Links

Related: [[Shortcut Learning in Deep Neural Networks]] · [[Breaking the Vision-Action Shortcut- Latent Interface Training for Generalizable Robotics Foundation Models]] · [[Classifier-Free Guidance]] · [[Classifier-Free Diffusion Guidance]] · [[Evals]] · [[Agent Evaluation]] · [[Grounding]] · [[Cross Entropy]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[Towards Quantifying Benchmark Optimization in ASR Models]] · [[Reward Hacking]] · [[Fine-Tuning]] · [[Flow Matching]] · [[Imitation Learning]] · [[Diffusion Policy]] · [[Think Like a World Model, Act Like a VLA- Distilling World-Model Representations into Compact Robot Policies]] · [[Grounded Action Model- 3D Grounding as a Foundation for Robotics]] · [[Transferring the Intelligence of VLMs to Robotic Control]]

New topics worth writing: Scene entropy as a benchmark design metric, LIBERO and LIBERO-PRO, RoboTwin 2.0, LangForce, compositional generalisation in language-conditioned policies, counterfactual input ablation as an eval sanity check, Wilson score intervals
