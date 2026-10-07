---
title: "PerturBot: Breaking Shortcut Priors in Vision-Language-Action Models with Perturbative Training"
authors: ["Mingyu Liu", "Chonghao Sima", "Tianjian Feng", "Hanqing Wang", "Cong Chen", "Hao Chen", "Chunhua Shen"]
year: 2026
arxiv: "2610.04616"
url: https://arxiv.org/abs/2610.04616
priority: Good-To-Read
read_on: 2026-10-06
tags: [paper, vision]
---
## The Core Idea

A robot policy can finish the task and still be reading the wrong thing.

The authors call this a **modality shortcut**. In a pile of successful demonstrations, three regularities almost always hold:

1. The target object fills the wrist camera, because the human operator was reaching for it.
2. Each object name shows up with only one verb — the bowl is always *picked up*, never *pushed*.
3. The gripper closing is always followed by a lift, because the expert never grasped air.

Each of these makes a cheap cue perfectly predictive of the expert action. [[Imitation Learning|Behaviour cloning]] only asks the policy to agree with the recorded action; nothing in the loss asks it to agree *for the right reason*. So the policy learns "lift when the gripper closes" instead of "lift when you can see something in the gripper".

> [!NOTE] Modality shortcut
> An easily-decoded visual, lexical, or motor cue that stands in for the evidence a decision actually requires, because successful demonstrations make the two agree. ^modality-shortcut

The three named failures:

- **Salience capture** — an object near the wrist camera displaces the instructed target.
- **Noun lock-in** — "push the bowl" produces a grasp, because *bowl* was only ever paired with *pick up*.
- **Motor inertia** — the gripper closes on nothing, and lifts anyway.

**Why this breaks scaling.** The field's default fix is more demonstrations. But if new trajectories carry the same correlations, the shortcut rule and the intended rule prescribe the same actions on every new example. More data cannot separate two rules that never disagree. Success rate goes up; brittleness stays.

The second half of the contribution is the measurement problem. Task success cannot distinguish a policy reading evidence from one reading shortcuts, because on the demonstration distribution both succeed. Worse, you can fake robustness to distractors by **ignoring vision entirely** — and the paper shows this empirically (removing the wrist camera gets the *best* distractor score and the *worst* success rate).

So: **GroundFscore**, an offline score that pairs edits which should change nothing with edits which should change something specific. Ignoring an input scores zero on both halves, so it cannot be gamed. No robot rollouts needed, so you can compute it at every checkpoint while scaling data.

The slogan: task success tells you *whether* a policy improves; GroundFscore tells you whether it improves *for the right reason*.

## The Methodology

### The information-theoretic framing

Policy $\pi_\theta(a \mid v, \ell, q)$ maps camera views $v$ (head + both wrists), instruction $\ell$, and motor context $q$ (proprioception, past actions) to an action chunk $a$.

For a decision $d$, shortcut $s$, and task evidence $e$, split the posterior:

$$p(d \mid s, e) = p(d\mid s)\cdot\frac{p(e \mid d,s)}{p(e\mid s)}, \qquad \mathbb{E}\left[\log \frac{p(d\mid s,e)}{p(d\mid s)}\right] = I(d;e\mid s) = I(d;e) - \mathcal{R}$$

where $\mathcal{R} = I(d;s) + I(d;e) - I(d;s,e)$ is the **redundancy** — the information about $d$ that the shortcut already duplicates.

A shortcut exists when $I(d;e\mid s) \approx 0$ on the demonstration data. Two distinct reasons:

- **Collinearity**: $\mathcal{R} \approx I(d;e)$. The evidence tells you nothing the shortcut didn't. This is salience capture and noun lock-in.
- **Degeneracy**: $H(d) \approx 0$. The decision never varies, so $I(d;e) \le H(d)$ is near zero regardless. This is motor inertia — nothing ever follows gripper closure but a lift.

This matters because the two need **different fixes**:

- To lower $\mathcal{R}$: insert a distractor off the grasp path. The expert action is unchanged, so the label is free.
- To raise $I(d;e)$: write captions that name the decision-relevant attributes.
- Under degeneracy, neither helps until you **record a missing branch** — a trajectory where the gripper reopens after an empty grasp. Relabelling existing data cannot invent an action that was never recorded. The paper is explicit and repeated about this limit.

### The three interventions

All three are training-data changes. Inference is untouched — no extra model, no extra forward pass.

**V — task-preserving wrist-view perturbation.** Edit the wrist image (with GPT-Image-2.5) to insert a plausible distractor object, varying in identity and placement, placed **off the grasp path**. It must not occlude the target, change contact geometry, or imply a new collision. If it would, the edit is rejected rather than kept with the old label. Only the wrist image changes; head view, proprioception, instruction, and action label stay exactly as recorded. Training distractors are disjoint from the distractors used in GroundFscore probes.

Lesson taught: a salient object is not necessarily the instructed one, but wrist evidence should still steer the motion.

**C — decision-relevant caption enrichment.** A VLM splits each recording into atomic skills (grasp, lift, move, put down) and writes each one in a fixed template: acting hand + open/closed state + verb + absolute and relative position + object state + attributes + object + grasp point.

So the grasp in "pick up the mug" becomes *"the left hand closes and grasps the upright blue mug left of the tray by its body"*.

Two rules that matter: attributes are added **only when needed** to disambiguate (this is about decision-relevant detail, not length), and a different verb appears **only if the data contain that action**. Rewriting a grasp as a push does not create a push demonstration.

Both demonstrations and auxiliary segments get the same brief/detailed treatment, so caption style doesn't leak which source a sample came from. Observations and actions are unchanged.

**R — caption-grounded trajectory expansion.** Collect 200 random-motion recordings (the operator sweeps the workspace with no task) and 200 deliberately failed-execution recordings (near misses, off-centre pushes, empty grasps, grasps that slip). Split them into atomic-skill segments with synchronised images, motor state, and actions. Discard unusable segments. Each retained segment gets a brief instruction for **what it actually does** plus a detailed caption from C.

A failed bowl-grasp recording may yield the segment *"move the empty gripper left and open it"* — labelled with that local behaviour, not with the original goal and not with a failure that hasn't happened yet.

The authors are careful here: R widens language-conditioned state–action coverage but does **not** teach recovery. Recovery needs recorded corrective continuations evaluated under the original instruction, and they don't have those.

### The objective

Nothing new. Standard $\pi_{0.5}$ [[Flow Matching|flow matching]] action loss, applied to a mixture.

With $\rho \in [0,1]$ the auxiliary fraction:

$$Q = (1-\rho)\,\mathcal{D}_{\text{demo}} + \rho\,\mathcal{D}_{\text{extra}}$$
$$\mathcal{L}(\theta) = \mathbb{E}_{(v,\ell,q,a)\sim Q}\left[\mathcal{L}_{\text{act}}(\pi_\theta; \tilde v, \tilde \ell, q, a)\right]$$

$\tilde v$ is the original or perturbed image, $\tilde \ell$ the brief or detailed instruction. Setting $\tilde v = v$ kills V, $\tilde \ell = \ell$ kills C, $\rho = 0$ kills R. Auxiliary samples **replace** a fraction of each batch rather than adding optimizer steps, so compute is matched.

### GroundFscore

For each shortcut family $c \in \{c_V, c_L, c_A\}$ (visual, lexical, motor), build paired edits of an observation $o$:

- A **null edit** $e \in \mathcal{E}^0_c$ changes the input but not the correct action — an off-path decoy, a paraphrase, a different approach to the same pose.
- A **causal edit** $e \in \mathcal{E}^1_c$ changes the correct action by a *known* amount $\Delta a^\star_e$ — the target moved by $\Delta p$, a different object named, a grasp that closed on nothing.

Two measurements:

$$R_c = \mathbb{E}_{o,e\sim\mathcal{E}^0_c}\left[\frac{\lVert \pi_\theta(e(o)) - \pi_\theta(o)\rVert}{\lVert a^\star\rVert}\right], \qquad S_c = \mathbb{E}_{o,e\sim\mathcal{E}^1_c}\left[\frac{\langle \pi_\theta(e(o)) - \pi_\theta(o),\ \Delta a^\star_e\rangle}{\lVert \Delta a^\star_e\rVert^2}\right]$$

$R_c$ is the **spurious response**: how much the policy moves when it shouldn't. $S_c$ is the **causal sensitivity**: what fraction of the demanded change it delivers ($1$ = exact compliance, $0$ = none). Both clipped to $[0,1]$, then combined as a harmonic mean:

$$\mathrm{GF}_c = \frac{2\,S_c\,(1-R_c)}{S_c + (1-R_c)}$$

This is an F-score with precision $= 1 - R_c$ (don't move when nothing should) and recall $= S_c$ (do move when something should). The harmonic mean is the whole trick: a policy that ignores the input has $R_c = 0$ (perfect precision) but $S_c = 0$, so it scores **zero**. Borrowed from HalFscore in PerturboLLaVA.

> [!NOTE] GroundFscore
> Harmonic mean of "doesn't move under null edits" and "moves correctly under causal edits". Measurable from paired forward passes only — no rollouts. Cannot be raised by ignoring an input. ^groundfscore

Measurement details that make it honest:
- Computed on 100 held-out General PnP observations, each given null + causal edits of all three families.
- $\Delta a^\star_e$ comes from the difference between **two expert recordings from the same state**, one under the original condition and one under the edited condition.
- Edited and unedited forward passes **share the flow-matching noise**, so sampling noise does not register as a response.
- $\lVert a^\star\rVert$ is floored at a tenth of its median; causal edits with $\lVert\Delta a^\star_e\rVert$ below a fifth of the median are discarded (tiny denominators would blow up the ratio).

### Setup

Backbone is $\pi_{0.5}$, fully fine-tuned. All conditions share initialisation, optimizer, step count, and batch size — only the data varies.

| Setting | Value |
|---|---|
| Optimizer | AdamW, $\beta = (0.9, 0.999)$, LR $2\times10^{-5}$ constant, no warmup |
| Weight decay | 0 |
| Grad clip | 1.0 |
| Batch | 256 (32/GPU) |
| Precision | BF16 |
| Images | $224\times224$ per view, 3 views (head + 2 wrists) |
| Chunk length $H$ | 16 (~1 s at 15 Hz) |
| Flow matching | 4 noise samples/example, 10 Euler steps at inference |
| Hardware | 8× A100 80GB, ~4 days per run |

Robot: dual-arm AgileX PiPER X, two 6-DoF arms with parallel grippers, RealSense D435 head camera + one wrist camera per arm, 30 FPS at $640\times360$.

Task: **General PnP**. Each layout draws 6 objects and 3 containers from a pool of 100 items, so combinations rarely repeat. 1,000 teleoperated demonstrations (main runs use a fixed 500). 50 evaluation layouts generated once by rule-based random placement, restored before every rollout, so every policy sees identical cases. Splits are made at the *recording* level **before** any segmentation, captioning, or augmentation.

## Ablation Studies and Experiments

### Main result — all eight component combinations

| Method | SR (%) | Prog. | $\mathrm{GF}_V$ | $\mathrm{GF}_L$ | $\mathrm{GF}_A$ | SC↓ | MI↓ |
|---|---|---|---|---|---|---|---|
| $\pi_{0.5}$ fine-tune | 42 | 2.71 | 0.28 | 0.24 | 0.19 | 68 | 82 |
| + V | 52 | 3.12 | **0.51** | 0.26 | 0.21 | 36 | 80 |
| + C | 54 | 3.20 | 0.30 | **0.47** | 0.22 | 62 | 78 |
| + R | 56 | 3.34 | 0.29 | 0.27 | **0.44** | 64 | 44 |
| + V + C | 64 | 3.62 | 0.55 | 0.50 | 0.24 | 30 | 76 |
| + V + R | 66 | 3.73 | 0.54 | 0.28 | 0.47 | 34 | 40 |
| + C + R | 68 | 3.82 | 0.31 | 0.51 | 0.48 | 58 | 38 |
| **PerturBot (V+C+R)** | **84** | **4.36** | **0.62** | **0.57** | **0.53** | 20 | **24** |

SC = salience-capture failure rate (off-path wrist distractor probe). MI = motor-inertia failure rate (grasp scripted to close on nothing). Both 50 trials, separate from the 50 task cases.

The pattern is clean and is the main evidence for the whole framing: **each component mostly moves the GroundFscore of its own family**. V lifts $\mathrm{GF}_V$ 0.28 → 0.51 and halves SC. C lifts $\mathrm{GF}_L$ 0.24 → 0.47. R lifts $\mathrm{GF}_A$ 0.19 → 0.44 and takes MI from 82% to 44%.

And they compound. Every pair beats both of its members; the full method doubles baseline success and wins every GroundFscore column. Removing any single component costs 16–20 points of success.

### The controls — and the one that matters most

| Control | SR | $\mathrm{GF}_V$ | SC↓ | MI↓ |
|---|---|---|---|---|
| Baseline | 42 | 0.28 | 68 | 82 |
| Noise / dropout (V's budget) | 44 | 0.33 | 60 | 82 |
| 2× demonstrations | 48 | 0.29 | 66 | 80 |
| Negative guidance (inference-time) | 46 | 0.41 | 48 | 62 |
| **Wrist camera removed** | **26** | **0.16** | **12** | 88 |

Camera removal is the headline negative result. It achieves the **best** salience-capture score in the whole paper — 12% failure vs 20% for PerturBot — by the simple expedient of not looking. It pays 16 points of success and has the worst $\mathrm{GF}_V$ of any condition. This is exactly the failure mode GroundFscore was designed to catch, and it catches it: the F-score collapses because $S_V$ goes to zero even though $R_V$ is tiny.

Everything else stays within 6 points of baseline. Doubling demonstrations buys 6 points of success and essentially nothing in GroundFscore.

### Budget-matched specificity controls

Each component adds data, so you have to rule out "more samples" and "generic regularisation".

| Condition | SR | $\mathrm{GF}_V$ | $\mathrm{GF}_L$ | $\mathrm{GF}_A$ | SC↓ | MI↓ |
|---|---|---|---|---|---|---|
| FT | 42 | 0.28 | 0.24 | 0.19 | 68 | 82 |
| Generic aug (noise/crop/dropout) | 44 | 0.33 | — | — | 60 | 82 |
| Random-placement decoys | **40** | 0.38 | — | — | 50 | 82 |
| **+ V** | 52 | **0.51** | — | — | 36 | 80 |
| Paraphrases | 44 | — | 0.28 | — | 66 | 82 |
| Length-matched irrelevant text | 44 | — | 0.30 | — | 66 | 80 |
| **+ C** | 54 | — | **0.47** | — | 62 | 78 |
| Equal-duration success demos | 46 | — | — | 0.20 | 66 | 80 |
| Random motion only | 50 | — | — | 0.34 | 66 | 60 |
| Failed executions only | 52 | — | — | 0.39 | 64 | 52 |
| **+ R** | 56 | — | — | **0.44** | 64 | 44 |

Three things worth noting:

- **Random-placement decoys score *below* baseline on success (40 vs 42).** Putting a distractor *on* the grasp path invalidates the recorded action, so you're training on wrong labels. The "off-path only, reject otherwise" rule in V is load-bearing.
- Paraphrases and length-matched filler move $\mathrm{GF}_L$ by at most 0.06. C moves it 0.23. So it is the decision-relevant *content* doing the work, not the fact of having longer text.
- Random motion and failed executions each lower MI, and together reach 44%. Neither alone is enough.

### Data scaling — the result that justifies the framing

Nested subsets of 125 / 250 / 500 / 1000 demonstrations at fixed optimizer steps, 3 seeds.

| Demos | FT: SR | FT: $\overline{\mathrm{GF}}$ | FT: MI | PB: SR | PB: $\overline{\mathrm{GF}}$ | PB: MI |
|---|---|---|---|---|---|---|
| $1/4\times$ | 30 | 0.21 | 86 | 48 | 0.36 | 46 |
| $1/2\times$ | 36 | 0.23 | 84 | 66 | 0.47 | 34 |
| $1\times$ | 42 | 0.24 | 82 | 84 | 0.57 | 24 |
| $2\times$ | 48 | 0.25 | 80 | 96 | 0.66 | 18 |
| **Slope/doubling** | +6.0 | +0.013 | −2.0 | **+16.2** | **+0.100** | −9.4 |
| **Spearman $\rho$(SR, $\overline{\mathrm{GF}}$)** | 0.41 | | | **0.93** | | |

This is the paper's sharpest claim. Over an 8× data range, plain fine-tuning gains **18 points of success while $\overline{\mathrm{GF}}$ moves 0.04** and motor-inertia failures barely budge (86% → 80%). More demonstrations bought proficiency and nothing else. PerturBot gains 48 points of success *and* 0.30 of GroundFscore, with failure rates falling at every budget.

PerturBot at $1/4\times$ (48% SR) already beats fine-tuning at $2\times$ (48% SR) — same number, **8× fewer demonstrations**. Compare to the [[Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning (MAGIC)|variance-reduction]] style of argument: the point isn't a better estimator, it's that the data was never going to identify the right rule.

The $\rho = 0.41$ vs $0.93$ contrast is also diagnostic: fine-tuning's $\overline{\mathrm{GF}}$ barely varies, so seed noise dominates its ranking.

### Sampling rate sensitivity

Varying $p_V$, $p_C$, $\rho$ one at a time, others held at validation-selected values:

| Level | $p_V$: SR | $p_C$: SR | $\rho$: SR |
|---|---|---|---|
| 0 | 68 | 66 | 64 |
| $1/2\times$ | 76 | 74 | 74 |
| $1\times$ | **84** | **84** | **84** |
| $2\times$ | 80 | 82 | 78 |

Mild interior optimum on all three. Pushing $p_V$ to $2\times$ buys a slightly better SC (16 vs 20) at the cost of 4 points of success — the robustness/conditioning trade-off is visible but shallow. Nothing cliff-edged.

### RoboTwin 2.0 — generality across 50 bimanual skills

Two settings. **Full** trains on clean + randomised demonstrations. **Clean2Random** trains on clean only, so randomised scenes (varied textures, lighting, clutter, table height, instruction wording) are unseen — the sharper test, since randomised clutter adds salient cues that clean training never contradicted.

| Method | Full: Clean | Full: Rand | C2R: Clean | C2R: Rand |
|---|---|---|---|---|
| $\pi_{0.5}$ baseline | 82.7 | 76.8 | 70.7 | 46.0 |
| PerturBot w/o V | 84.9 | 79.3 | 72.6 | 49.8 |
| PerturBot w/o C | 84.1 | 80.6 | 71.8 | 55.4 |
| PerturBot w/o R | 84.4 | 80.9 | 72.0 | 56.2 |
| **PerturBot** | **85.6** | **81.9** | **73.4** | **58.2** |

The gain concentrates exactly where predicted: **+12.2 on Clean2Random randomised scenes vs +2.7 on clean**. And removing V costs most on randomised scenes in both settings (−8.4 and −2.6), consistent with its role against salient distractors.

Context caveat the authors flag: Table 7 in the appendix lists systems reporting much higher numbers (InternW0 at 93.1/75.6, ABot-M0.5 at 94.1 Full avg). Those are different architectures and recipes. The $\pi_{0.5}$ comparison is the matched one.

### Verb-change probes — noun lock-in

General PnP has one operation, so it can't test verbs. They built 5 RoboTwin task pairs whose shared object admits two operations from the same state (press vs move the stapler), restored one task's initial state, and issued the *other* task's instruction. 100 trials per policy.

| Method | Followed↑ | Lock-in↓ | Other |
|---|---|---|---|
| $\pi_{0.5}$ FT | 36 | 52 | 12 |
| PerturBot w/o V | 64 | 26 | 10 |
| **PerturBot w/o C** | **45** | **43** | 12 |
| PerturBot w/o R | 66 | 24 | 10 |
| PerturBot | 71 | 19 | 10 |

Fine-tuning does the *familiar* operation more often than the instructed one. Removing C restores most of the lock-in; removing V or R costs at most 7 points. C is the component doing the lexical work — and since C never introduces a verb absent from the data, the gain is about *grounding* verbs the demonstrations already contained.

### Controllability — the "did you just go deaf?" check

| Condition | Target: instructed↑ | Target: distractor↓ | Lift: secured↑ | Lift: empty↓ |
|---|---|---|---|---|
| $\pi_{0.5}$ FT | 88 | 68 | 94 | 82 |
| No wrist camera | 48 | **12** | 90 | 88 |
| **PerturBot** | 90 | 20 | 92 | **24** |

Read row 2 carefully. Removing the camera cuts distractor-grabbing from 68% to 12%, but instructed-object selection collapses from 88% to 48%. PerturBot gets distractor-grabbing to 20% while *improving* instructed selection to 90%. Same story for lifting: PerturBot lifts secured objects nearly as often as baseline (92 vs 94) but lifts after empty grasps 24% vs 82%.

### Does GroundFscore actually predict rollout failures?

Spearman $\rho$ against negative online failure rate over 36 checkpoints (12 conditions × 3 seeds), with two-level cluster bootstrap intervals (resample conditions, seeds, and within-checkpoint evaluation units; all predictors share draws so differences are paired):

| Predictor | $\rho$ | 95% CI | GF − predictor | Ranks no-wrist below baseline? |
|---|---|---|---|---|
| Task success | 0.31 | [−0.06, 0.60] | 0.41 [0.12, 0.68] | yes |
| Stability ($1-R$) alone | 0.44 | [0.09, 0.69] | 0.28 [0.04, 0.52] | **no** |
| Responsiveness ($S$) alone | 0.38 | [0.02, 0.65] | 0.34 [0.08, 0.60] | yes |
| **GroundFscore** | **0.72** | [0.49, 0.86] | — | yes |

The ablation of the *score itself* is the good part. Stability alone gets $\rho = 0.44$ — not terrible — but it ranks the blind no-wrist policy **above** the baseline. That's the exact pathology. Responsiveness alone manages only 0.38. You need both terms, and the harmonic mean is what enforces it. Task success's CI includes zero.

### Detailed-instruction evaluation

Replay the same 50 cases with detailed instructions instead of brief ones:

| Method | Brief | Detailed | $\Delta$ |
|---|---|---|---|
| $\pi_{0.5}$ FT | 42 | 34 | −8 |
| + V | 52 | 46 | −6 |
| + C | 54 | 60 | +6 |
| + R | 56 | 46 | −10 |
| + C + R | 68 | 74 | +6 |
| PerturBot | 84 | 86 | +2 |

Without C, detailed instructions are out of distribution and cost 6–10 points. With C, they help a little. PerturBot's *brief*-instruction success already beats every other configuration under either style, so the method doesn't secretly depend on verbose prompts at test time.

## Worth Remembering

**The honesty about what relabelling can't do.** The degeneracy case ($H(d) \approx 0$) is called out as unfixable by data editing. If the demonstrations never contain a reopen-after-empty-grasp, no caption rewrite creates one. This is why R needs *recorded* random and failed trajectories, not synthesised ones. And even then, R gives behavioural support, not recovery skills — recovery would need corrective continuations evaluated under the original instruction, which they don't have. The paper repeats this caveat at least four times, which is unusual and good.

**The admitted limitation.** You need detailed captions and relabelled segments, plus a corpus of random and failed trajectories on top of demonstrations. Their one consoling observation: real-robot deployment generates failures for free, and "successful trajectories tend to resemble one another, while each failure goes wrong in its own way" — so failure data is cheap and information-dense.

**The negative result worth stealing.** Random-placement decoys *hurt* success (40 vs 42 baseline). If an edit invalidates the recorded action, you've created a label-noise problem, not an augmentation. Any counterfactual data editing scheme needs a rejection rule, and the rejection rule is where the work is.

**Where the measurement idea transfers.** GroundFscore is structurally the same move as a paired A/B design: hold everything fixed, change one thing, measure the response, and crucially measure both a null and a known-nonzero contrast. The precision/recall framing stops you from being fooled by a policy that is "robust" because it's inert. This generalises well beyond robotics — any system where you want to know whether a feature is *used* rather than *present* has the same problem, and "sensitivity to task-preserving perturbation" alone (used elsewhere for demonstration selection) is half a metric.

**Open questions.** $\Delta a^\star_e$ requires two expert recordings from the same state under different conditions — expensive, and only obtainable where you can reset state exactly. Fine in simulation and on a bolted-down table; unclear for mobile manipulation or anything stochastic. The score also lives in $\pi_{0.5}$'s normalised action space, so cross-architecture comparison of absolute GF values is probably meaningless; only within-recipe rankings are safe.

**Connections.** This is causal confusion / copycat in [[Imitation Learning|imitation learning]] with a multimodal framing, and it's the same disease as language-prior answering in VQA. The redundancy decomposition is the information-theoretic cousin of [[Shortcut Learning in Deep Neural Networks|shortcut learning]]. [[Breaking the Vision-Action Shortcut- Latent Interface Training for Generalizable Robotics Foundation Models|Breaking the Vision-Action Shortcut]] attacks the same problem architecturally (train a latent interface) rather than via data; worth reading as the contrasting approach. The harmonic-mean trick is lifted from PerturboLLaVA's HalFscore, by overlapping authors.

**Practical caveat.** Zero inference-time cost is a real selling point — V, C, R all vanish at deployment, and the policy takes ordinary brief instructions. But the training-time cost is an image-editing API over your whole wrist-view corpus plus a VLM captioning pass plus 400 extra teleoperated recordings. Budget that before assuming this is a cheap win.

## Links

Related: [[Imitation Learning]] · [[Shortcut Learning in Deep Neural Networks]] · [[Breaking the Vision-Action Shortcut- Latent Interface Training for Generalizable Robotics Foundation Models]] · [[Flow Matching]] · [[Think Like a World Model, Act Like a VLA- Distilling World-Model Representations into Compact Robot Policies]] · [[Grounded Action Model- 3D Grounding as a Foundation for Robotics]] · [[Counterfactual Reasoning and Learning Systems]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[Evals]] · [[Reward Hacking]] · [[Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning (MAGIC)]] · [[RoboFollow- Unveiling the Instruction Following Mirage in Embodied Agents]]

New topics worth writing: Causal confusion in imitation learning, Redundancy and synergy in multivariate mutual information, Counterfactual data augmentation for policies, Domain randomization, Metric gaming via input ablation, Paired-edit diagnostics for feature reliance
