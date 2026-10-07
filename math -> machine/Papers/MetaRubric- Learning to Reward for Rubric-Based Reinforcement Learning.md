---
title: "MetaRubric: Learning to Reward for Rubric-Based Reinforcement Learning"
authors: ["Yuxuan Fan", "Jaehong Yoon"]
year: 2026
arxiv: "2610.02824"
url: https://arxiv.org/abs/2610.02824
priority: Good-To-Read
read_on: 2026-10-05
tags: [paper, llm, rl, vision, theory]
---
## The Core Idea

Rubric-based RL is the current answer to "how do I do RL on tasks with no single right answer". You write a list of criteria for each prompt ("asks for the patient's location", "does not overstate the abstract"), each with a weight, and a language-model judge marks each one met or not met. The weighted score becomes the reward. See [[Small Language Models as Judges for Rubric-Based Reinforcement Learning]] and [[ImpossibleRubrics- Stress-Testing Generated Rubrics as Reward Signals]] for the surrounding literature.

This paper names a specific way that breaks. **Vacuous Credit**: the judge marks a criterion satisfied even though the thing the criterion asks for is *completely absent from the response*.

> [!NOTE] Vacuous Credit
> A judge gives a criterion a high score while the response contains none of the information or action that criterion requires. The response earns credit for words that gesture at the requirement — generic safety advice, hedging, "guidelines vary by region" — instead of doing it. ^vacuous-credit

The worked example: a patient asks how often to go for check-ups after a stent, and whether the schedule depends on where they live. The rubric says the model should *ask for the patient's location*. The model says only that guidelines differ between regions, and never gives a check-up frequency. The judge ticks the location criterion anyway.

Two things make this worse than "the judge is a bit noisy".

**First, it survives deletion.** The authors took 1,000 rollouts the judge had marked correct, then used a strong model to surgically delete the content that satisfied the target criterion. GPT-4o-mini still awarded 82.0% of those criteria. GPT-5.4-mini still awarded 69.8%. The award was never attached to the evidence in the first place.

**Second, it flips the sign of the gradient.** [[GRPO]] does not use raw rewards — it standardises each response's reward against the other responses sampled from the same prompt. So one bogus award is not a small bias. A response that skipped a required action, but got Vacuous Credit for it, can out-score a response that actually performed it, and therefore gets the *larger positive advantage*. The update pushes the policy toward the worse answer. Removing a single erroneous award can reverse a response's advantage from positive to negative. This is [[Reward Hacking]] at the level of a single criterion rather than a whole reward model.

The headline diagnostic: train Qwen3-8B on HealthBench with a judge, and the **training reward goes up while an independent three-judge panel says rubric satisfaction goes down**. The proxy and the goal move in opposite directions — [[Reward Hacking#^goodharts-law|Goodhart]] in its purest form.

The fix has two halves, and the paper's claim is that you need both.

1. **Make credit conditional on evidence.** A criterion cannot score higher than the degree to which the response actually contains the required content and can support it.
2. **Let the rubric itself learn.** Vague wording ("considers the patient's location") is the root cause of some Vacuous Credit, and no amount of reweighting fixes a vague sentence. So between training stages, rewrite the criterion text and reshuffle the weights based on which requirements the policy keeps failing.

What this unlocks: an evidence-grounded reward that cannot be satisfied by boilerplate, and a rubric that tracks the policy's moving error profile instead of being frozen at step 0. The paper's most interesting finding is that the *schedule* of rubric changes matters, not just the final rubric — more on that below.

## The Methodology

### The static rubric reward being replaced

For a prompt $x$ and response $y$, with criteria indexed by $k$, weights $w_k \ge 0$, and a binary judge verdict $m_k(x,y) \in \{0,1\}$:

$$R(x,y) = \frac{\sum_k w_k m_k(x,y) - B_x}{P_x}$$

$B_x = \sum_k w_k b_k$ is an offset and $P_x = \sum_k w_k - B_x$ a normaliser. The $b_k \in \{0,1\}$ is a reference value that handles *negative* criteria. Some criteria describe a thing you should avoid ("overstates the abstract"). For those, the judge scores the presence of the bad thing, and $b_k = 1$ flips it onto a common "fulfilment" scale:

$$u_k^d = b_k + (1-2b_k) a_k^d$$

So every score reads in the same direction — higher is better — regardless of whether the criterion is a reward or a penalty. Nice bit of bookkeeping.

### Counterfactual prompt pairs

Every training prompt $x$ gets a twin $x^{\text{cf}}$ in which **exactly one task-relevant fact changes** and everything else is held fixed. In the paper's figure, a stated patient age goes from 40 to 60.

Why: it forces the reward to separate requirements that depend on the facts from requirements that are generic. Each twin has its own rubric, and a response is scored only against its own prompt's rubric. A fact-dependent criterion then only gets credit when the response actually uses *those* facts.

Each criterion also gets a **correspondence label** $z$ saying how it behaves across the pair: `INVARIANT`, `TARGET_CHANGE`, `WEIGHT_CHANGE`, `DROPPED`, `ADDED`. These labels do double duty later — they define the pooling groups for reweighting, and they anchor rubric rewrites.

For images, the counterfactual is explicitly flagged as *hypothetical* ("suppose this observation were X instead") because you cannot edit the radiograph.

### The evidence-aware criterion score

This is the core mechanism. The judge gives each criterion **three** scores in $[0,1]$ instead of one verdict:

- $u_k^{\text{sat}}$ — **satisfaction**: is the requirement correctly fulfilled?
- $u_k^{\text{cov}}$ — **coverage**: is the required information or action actually *present* in the response?
- $u_k^{\text{sup}}$ — **support**: is that fulfilment backed by allowed evidence (the supplied source, or the prompt plus reliable domain knowledge when permitted)?

The criterion score is the weakest link:

$$\rho_k = \min\!\left(u_k^{\text{sat}},\, u_k^{\text{cov}},\, u_k^{\text{sup}}\right)$$

Why the minimum and not an average: an average lets a confident satisfaction score drag a zero-coverage criterion up to 0.5. The min makes any single failure fatal. And the three are genuinely not redundant — a response that states *an incorrect sample size* has full coverage (the number is there, you asked for it) but fails satisfaction.

> [!NOTE] Coverage vs satisfaction
> Coverage asks "did you say anything about this?". Satisfaction asks "was what you said right?". Vacuous Credit is the case where satisfaction is high and coverage is zero. Taking the min is what makes the deletion test bite. ^coverage-vs-satisfaction

### The full rubric reward

$$R^{\text{rub}} = \min\!\left(\operatorname{clip}_{[0,1]}\!\left(\frac{\sum_k w_k \rho_k - B_x}{P_x}\right),\; U^{\text{sup}}\right)\cdot \frac{1+Q}{2}$$

Two response-level guards bolted on:

- **Support cap** $U^{\text{sup}} \in [0,1]$: a single judgement of whether the response's claims *as a whole* are supported. This stops a response from ticking every box and then inventing nonsense in the gaps — unsupported claims outside the rubric still pull the ceiling down.
- **Quality factor** $Q \in [0,1]$: the minimum of relevance, coherence and concision. Multiplying by $\frac{1+Q}{2}$ means a $Q=0$ response keeps half its score, a $Q=1$ response keeps all of it. It penalises padding without zeroing the signal.

### The auxiliary QA reward — a judge-independent check

All of the above still runs through a judge, which is the thing under suspicion. So there is a second, mechanical reward that does not involve a judge at all.

Before training starts, a generator turns each criterion's reference facts into **four-option multiple-choice questions**. Then a small frozen reader — **Qwen3-1.7B, thinking off, temperature 0** — is asked each question *given only the policy's response*.

$$R^{\text{QA}}(x,y) = \frac{\sum_q r_q \,\mathbf{1}[a_q(y) = a_q^\star]}{\sum_q r_q}$$

The construction has a screen that matters: a question is only kept if the reader **cannot** answer it from the question and options alone. That guarantees success requires information the response supplied, rather than world knowledge or an answer-shaped cue in the option lengths. Questions that fail the screen go back for revision — the generator may rewrite the question and all three distractors, but the tested facts, the correct answer's meaning and the correct option's *position* are frozen.

The question bank is fixed before training and never changes during rubric adaptation, so it is a stable yardstick.

### Combined training reward

$$R(x,y;\Phi_t) = \lambda_{\text{rub}} R^{\text{rub}} + \lambda_{\text{QA}} R^{\text{QA}}, \qquad \lambda_{\text{rub}}=1.0,\ \lambda_{\text{QA}}=0.3$$

### GRPO with paired groups

Both prompts in a pair are sampled $G=8$ times, and all $2G = 16$ responses form **one** GRPO group:

$$A_j = \frac{R_j - \bar{R}}{\operatorname{std}(R) + \epsilon}, \qquad \epsilon = 10^{-6}$$

Mean and std over all 16. This means a response is implicitly competing against responses to the counterfactual too — a response that would score well under either set of facts is not rewarded relative to one that tracked the specific facts it was given.

The launcher **disables row shuffling** so pairs stay adjacent in the batch. Small detail, load-bearing.

### Outer loop, part 1: error-guided weight adaptation

Rubric criteria are instance-specific, so there are far too few observations per criterion to tune its weight. Solution: pool.

Each criterion has a **severity class** $h \in \{1,2,3,4\}$ — nice-to-have, should-have, must-have, contraindication — with base weights $(\omega_1, \ldots, \omega_4) = (4,5,6,8)$. Criteria sharing the same $(h, z)$ pair (severity × correspondence label) form a group $g$ with one shared log-adjustment:

$$w_k = \omega_h \exp(\tau_g)$$

At the end of each stage, a **separate grading pass** — independent of the training reward, using binary verdicts — measures which criteria the policy actually fails, over at most 96 responses (one per sampled prompt). The group error $L_g$ averages $1 - m_k$ first over criteria in the group, then over responses, then over cases. That three-level nesting is deliberate: it stops a case with 39 criteria from dominating a case with 1.

$$\tau_g \leftarrow \tau_g + \eta (L_g - \bar{L}), \qquad \eta = 0.1$$

Groups failing more than average gain weight; the rest lose it. Three constraints keep this from rewriting the rubric's priorities:

- **Severity ordering**: $\omega_h \exp(\tau_{(h,z)})$ must stay increasing in $h$ for each $z$. A frequently-missed nice-to-have must never outweigh a must-have.
- **Clipping**: $|\tau_g| \le c$ where $c = \log \min_h \omega_{h+1}/\omega_h = \log(6/5) \approx 0.1823$ — the smallest log-gap between adjacent severity levels. An adjustment can never jump a priority tier.
- **Centering**: mean adjustment stays at zero, so only *relative* emphasis moves, not the overall reward scale.

### Outer loop, part 2: semantically anchored rubric revision

Reweighting cannot repair bad wording. If the criterion reads "considers the patient's location" rather than "asks for the patient's location", a response that merely notes regional differences keeps earning Vacuous Credit at *any* weight.

But rewriting the rubric from the policy's own responses is obviously dangerous — you drift toward rewarding whatever the policy already does. Three defences:

1. **Semantic anchor.** The *original prompt's initial rubric* is fixed and is the authority on what each requirement means. Correspondence labels say how the anchored requirement reads under the counterfactual's facts. Revisions may clarify *wording*, never change *requirements*, polarity, severity, weights, or the fixed auxiliary questions.
2. **At most one revision per stage boundary.**
3. **Two-gate acceptance.** A revision is accepted only if it improves agreement with a reference panel by strictly more than $\delta = 0.02$ on 20 held-out validation prompt pairs, **and** a separate semantic review confirms it preserves the anchored requirement. The proposer never sees the held-out responses or their reference judgements. The panel judges before seeing the proposal.

If either gate fails, the old description is kept — but the updated weights carry over regardless.

### Stage alternation

$$\theta_{t+1} = \mathcal{U}_{\text{GRPO}}(\theta_t; \Phi_t), \qquad \Phi_{t+1} = \mathcal{A}(\Phi_t; \mathcal{B}_t, \mathcal{V}_t)$$

$\Phi_t = (\mathcal{E}_t, \bm{\tau}_t)$ holds the criterion descriptions plus correspondence labels, and the per-group log-weight adjustments. Fixed within a stage. $\mathcal{B}_t$ is the responses collected; $\mathcal{V}_t$ the held-out validation set.

### Training configuration (Qwen3-4B-Instruct-2507, HealthBench)

| Knob | Value |
|---|---|
| Thinking mode | disabled |
| Optimiser | AdamW, $\beta = (0.9, 0.999)$, wd 0.01 |
| Learning rate | $10^{-6}$, 4 warmup updates then constant |
| Prompt batch / PPO mini-batch | 18 / 18 (= 9 complete pairs) |
| Rollouts per prompt $G$ | 8 → 144 responses per update |
| PPO epochs per batch | 1 |
| PPO clip | 0.2 / 0.2, dual-clip 3.0 |
| KL penalty | **disabled** |
| Entropy coefficient | 0 |
| Loss aggregation | token mean |
| Sampling | $T=0.7$, top-$p$ 0.8, top-$k$ 20 |
| Max prompt / response | 2,048 / 1,024 tokens |
| Grad norm clip | 1.0 |
| Epochs / seed | 3 / 42 |
| Judge, grader, proposer, reviewer, panel | GPT-5.4-mini, temperature 0, separate calls |
| Hardware | 2 GPUs, FSDP2 + TP rollout, CPU offload, [[Flash Attention]] |

Note the KL penalty is off. No [[RLHF#^kl-leash|KL leash]] to the reference model — the evidence checks are doing that job instead.

Data: PubMedQA uses a six-criterion template (3 positive: correct decision, accurate use of abstract evidence, relevance; 3 negative: unsupported claims, contradiction/overstatement, missing or generic rationale). HealthBench uses the official physician-written criteria and signed points. MMOral-RL has 980 examples with 8,672 criteria — mean 8.85 per example, median 8, range 1–39.

## Ablation Studies and Experiments

### The diagnostic experiments (Section 3) — the best part of the paper

**Finding 1: reward up, satisfaction down.** Qwen3-8B on HealthBench, judged by GPT-4o-mini or GPT-5.4-mini. Evaluated on held-out data with a panel of three stronger judges (GPT-5.5, Gemini 3.5 Flash, DeepSeek-V4-Pro), counting a criterion satisfied only under **unanimous agreement**. Training reward rises; panel score falls. Auditor-flagged Vacuous Credit persists throughout training.

**Finding 2: the deletion test.** 1,000 rollouts the judge had marked correct. GPT-5.5 produces two paired edits of each:

| Reward judge | Retention after deleting **required content** | Retention after deleting **generic statements** (control) |
|---|---|---|
| GPT-4o-mini | 82.0% | 93.1% |
| GPT-5.4-mini | 69.8% | 95.0% |

Read this carefully. Deleting the required content *does* lower retention relative to the control — the judge is not completely blind. But the large majority of awards survive the removal of the only thing that justified them. The gap between 82.0% and 93.1% is the judge's entire sensitivity to whether the requirement was met.

### Main results

Four benchmarks: PubMedQA (accuracy, 500-question expert-labelled test set, with abstract context), HealthBench-Hard (accuracy-axis score, 1,000 conversations), MMOral-X (mean over Simple/Moderate/Complex, 300 questions), MMOral-OPG (overall, 578 questions). GPT-5.5 judges all model-based test evaluations.

| Model | Method | PubMedQA | HB-Hard | MMOral-X | MMOral-OPG |
|---|---|---|---|---|---|
| Qwen3-4B | baseline | 52.60 | 8.83 | 3.38 | 20.59 |
| | GRPO | 72.40 | 10.56 | 5.72 | 23.18 |
| | DAPO | 73.80 | 11.74 | 7.12 | 24.61 |
| | Dr. GRPO | 76.60 | **13.28** | 6.94 | 25.72 |
| | GSPO | 74.60 | 11.38 | 6.42 | 24.08 |
| | **MetaRubric** | **78.40** | 13.02 | **7.73** | **26.35** |
| Qwen3-8B | baseline | 44.20 | 4.24 | 3.50 | 26.28 |
| | GRPO | 70.00 | 7.15 | 7.08 | 27.35 |
| | Dr. GRPO | 76.40 | 9.72 | 8.58 | 29.63 |
| | **MetaRubric** | **76.80** | **10.34** | **9.21** | **30.44** |
| Gemma-e2b | baseline | 48.00 | 11.23 | 3.69 | 32.77 |
| | GRPO | 51.60 | 13.07 | 4.74 | 31.28 |
| | Dr. GRPO | 70.80 | 14.94 | 5.71 | **35.42** |
| | **MetaRubric** | **72.00** | **15.68** | **6.03** | 35.10 |

Beats static-judge [[GRPO]] in all 21 family×metric comparisons. Best overall in 19 of 21 — it loses HealthBench-Hard on Qwen3-4B to Dr. GRPO (13.02 vs 13.28) and MMOral-OPG on Gemma-e2b (35.10 vs 35.42).

PubMedQA gains over GRPO: +6.00 (4B), +6.80 (8B), **+20.40** (Gemma-e2b). The Gemma number is enormous and the paper does not explain it; note Gemma's GRPO baseline on PubMedQA was unusually weak (51.60, barely above the 48.00 untrained baseline), so a large part of that 20.40 is Gemma's GRPO run failing rather than MetaRubric succeeding. Treat it as the loosest number in the table.

Open-ended gains over GRPO: 1.29–3.82 points. More modest, and in the right ballpark for this kind of work.

Auxiliary QA accuracy rises 3.23–5.68 points over GRPO everywhere. The honest caveat the authors state themselves: auxiliary QA is *also a training objective*, so this is partly measuring that the objective was optimised. It leads on Aux. Acc. even in the two settings where Dr. GRPO wins the benchmark — which is a mild warning that the two can come apart.

### Component ablations (Qwen3-4B family)

All variants keep the quality multiplier, auxiliary QA reward and paired sampling. So these isolate the three *new* pieces, not the whole method.

| Removed | HealthBench-Hard Δ | MMOral-OPG Δ |
|---|---|---|
| Rubric revision (freeze descriptions + labels) | larger of the two | smaller |
| Weight adaptation (freeze weights) | smaller | larger |
| **Both** | **−3.17** | **−4.40** |
| Evidence checks (satisfaction only, no support cap) | −1.22 | −2.59 |

Three things to take from this:

1. **The two adaptation mechanisms are complementary, not substitutes.** Revision matters more on HealthBench-Hard; weight adaptation matters more on MMOral-OPG. Freezing both costs more than either alone. Clarifying *what* a requirement means and adjusting *how much it counts* are different controls.
2. **Evidence checks are the smaller contributor** — −1.22 and −2.59, against −3.17 and −4.40 for frozen adaptation. This is a slightly awkward result for the paper, since Vacuous Credit is the headline problem and the min-of-three is the direct fix. The likely explanation is that auxiliary QA and paired sampling (retained in that ablation) already catch a lot of the same failures. The paper does not run the clean "remove evidence checks *and* auxiliary QA" ablation that would separate these.
3. There is no ablation of counterfactual pairing, the support cap alone, or the quality factor alone. Those are the gaps.

### The result that matters most: is the final rubric enough?

Take MetaRubric's *final* learned rubric, and retrain from scratch with it, frozen, no further adaptation.

| Training rubric | HB-Hard | MMOral-OPG |
|---|---|---|
| baseline | 8.83 | 20.59 |
| **Final rubric, frozen** | 9.04 | 21.78 |
| **MetaRubric (adapting)** | **13.02** | **26.35** |

The final rubric recovers **5.0%** of the HealthBench gain and **20.7%** of the MMOral gain. Almost nothing.

This is the paper's most interesting negative result. The artefact is not the contribution — **the schedule is**. A requirement emphasised late in training, when the policy is making one class of error, gives a different and apparently much worse signal when applied from step 0 to a policy making entirely different errors. It is curriculum, not rubric engineering.

It also means you cannot shortcut this. You cannot run MetaRubric once, extract the good rubric, and hand it to your team. The alternating loop has to run.

### Revision statistics (Qwen3-4B)

| Dataset | Criteria | Revised ≥ once | Words before → after |
|---|---|---|---|
| HealthBench | 10,448 | 4,727 (45.2%) | 40.8 → 46.3 (+13.5%) |
| PubMedQA | 2,949 | 879 (29.8%) | 14.3 → 18.2 (+27.3%) |
| MMOral-RL | 8,672 | 5,259 (60.6%) | 16.9 → 22.5 (+33.1%) |

Revision is selective — 39.4% to 70.2% of criteria are never touched. And revision rate is not simply a function of initial length: MMOral-RL starts with *longer* descriptions than PubMedQA yet revises twice the fraction. The templated six-criterion PubMedQA rubric was apparently already specific enough; the auto-generated MMOral rubrics were not.

Descriptions grow. Nobody checks whether that growth is clarification or creeping specification, which is the thing the semantic-review gate is supposed to prevent.

### Case study

A PubMedQA prostate-bed motion question. One criterion asks the model to report the *absence* of motion differences; a paired penalty names unsupported left–right and superoinferior claims. In a stage with more overstatement errors than evidence-use errors, the evidence criterion's weight moves $+6 \to +5$ and the penalty $-6 \to -8$. Separately, the revision names *which* claims the evidence supports.

The point being made: raising the penalty alone leaves its interpretation unchanged; clarifying the wording alone leaves its priority unchanged. Two distinct controls over the reward.

## Worth Remembering

**The limitations the authors admit.** Two, both serious. **Single seed** per configuration — no variance across runs, so the 1.29–3.82 point open-ended gains have no error bars. Given that MetaRubric loses two of 21 comparisons by 0.26 and 0.32 points, a second seed could plausibly move several cells. **Comparison scope**: most related rubric-RL work has not released training code, so the baselines are generic RL variants (GRPO, DAPO, Dr. GRPO, GSPO) rather than competing rubric methods. There is no comparison against rubric dropout, RLR³, or Chasing the Tail — all cited as alternative Vacuous-Credit mitigations.

**The judge cost.** Count the GPT-5.4-mini calls per stage: inner-loop judging on every rollout (144 responses per update × $\mathcal{N}$ updates), outer-loop grading on up to 96 responses, one proposal, one semantic review, and a reference panel over 20 validation pairs. The inner-loop judge is now returning *four* numbers per criterion (sat, cov, sup, plus global support and three quality scores) rather than one bit. On MMOral-RL at 8.85 criteria per example that is a long structured output per response. This is not a cheap method.

**A circularity worth sitting with.** The problem is that judges are unreliable. The fix asks a judge for *three* scores instead of one, and asks further judges to grade, propose, review and provide reference panels. The defence is structural rather than empirical: decomposing into coverage and support asks narrower, more mechanical questions than "is this criterion met", and the min aggregation makes any one of them sufficient to deny credit. But the deletion test is never re-run on the *trained* evidence-aware judge. That experiment — does required-content retention drop from 69.8% to something near zero under the three-score prompt? — is the one I most want and it is not in the paper.

**The anchor is the load-bearing safety property.** Everything that prevents the rubric from drifting toward "whatever the policy already does" rests on the original prompt's initial rubric being frozen and authoritative. If that initial rubric is itself vague — and 60.6% of MMOral-RL criteria needed revision, suggesting it often was — the anchor permits a lot of movement. The semantic reviewer is the only thing checking that movement is clarification rather than relaxation, and it is the same model family as the proposer.

**Self-grading pressure on the auxiliary reward.** $R^{\text{QA}}$ is a training objective *and* a reported metric (Aux. Acc.). The question bank is frozen before training, which is the right call and genuinely prevents the worst version of this. But a policy optimised for 300+ steps against a fixed Qwen3-1.7B reader can learn to write in the style that reader parses well. The two cases where Dr. GRPO beats MetaRubric on the benchmark while MetaRubric leads on Aux. Acc. are exactly the shape you would expect if that were starting to happen.

**MMOral counterfactuals are a weaker instrument.** You cannot change a fact in a radiograph, so the counterfactual is a verbal "suppose this finding were X instead". The prompt template explicitly forbids claiming the image shows the hypothetical. That makes the image–text pair *inconsistent by construction* — the model is asked to reason about facts its own visual input contradicts. It is unclear whether this teaches fact-sensitivity or teaches ignoring the image. MMOral is also where weight adaptation mattered more than revision, which is consistent with the counterfactual signal being weaker there.

**Practical caveats if you wanted to use this.** The severity anchors $(4,5,6,8)$ set the clipping bound $c = \log(6/5)$, so with tighter severity gaps the weights can barely move at all; with a wider spread they move freely. That coupling is undocumented and you would need to pick anchors deliberately. The pairing requires disabling batch shuffling, so pair construction has to produce balanced, adjacent rows. And the KL penalty being off means you are relying entirely on the evidence checks and quality factor to hold the policy together — if you weaken those, expect drift.

**Connections.** The evidence-aware score is structurally what [[Grounding]] asks for, applied to the reward rather than the output: credit only for what the response can support. The counterfactual pairing is the reward-design analogue of what [[Counterfactual Reasoning and Learning Systems]] does for logged data — change one thing, see what the signal depends on. The "reward up, quality down" curve is the rubric-level version of [[Reward Hacking#^optimiser-as-adversary|optimiser-as-adversary]], and the final-rubric-insufficiency result rhymes with the curriculum findings in [[Dynamic Important Example Mining for Reinforcement Finetuning]] — what you emphasise matters less than *when*.

**Follow-up questions.** Does the deletion test retention rate actually fall under the three-score judge? What happens with a cheap open-weights judge instead of GPT-5.4-mini — does the min-of-three recover a weak judge, or does a weak judge fail all three checks together? Is there a version of this with a frozen rubric but an *annealed* weight schedule, to isolate "curriculum" from "learned curriculum"? And since the schedule is the contribution, could you just transplant the recorded sequence of rubric configurations from one run into another policy?

## Links
Related: [[GRPO]] · [[Reward Hacking]] · [[Small Language Models as Judges for Rubric-Based Reinforcement Learning]] · [[ImpossibleRubrics- Stress-Testing Generated Rubrics as Reward Signals]] · [[Reward Function]] · [[RLHF]] · [[Grounding]] · [[Evals]] · [[Preference Learning]] · [[Credit Assignment]] · [[Proximal Policy Optimization Algorithms]] · [[Counterfactual Reasoning and Learning Systems]] · [[Agent Evaluation]] · [[Training language models to follow instructions with human feedback]] · [[Dynamic Important Example Mining for Reinforcement Finetuning]]

New topics worth writing: Vacuous Credit, Rubric-Based Reinforcement Learning, Rubric Dropout, DAPO, Dr. GRPO, GSPO, Counterfactual Prompt Pairs, LLM-as-Judge Reliability, Reward Curriculum Scheduling, HealthBench, PubMedQA, Content-Deletion Testing for Reward Models
