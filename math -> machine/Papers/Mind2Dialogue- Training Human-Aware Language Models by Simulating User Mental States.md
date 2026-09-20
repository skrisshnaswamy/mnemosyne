---
title: "Mind2Dialogue: Training Human-Aware Language Models by Simulating User Mental States"
authors: ["Wang et al."]
year: 2026
arxiv: "2609.15972"
url: https://arxiv.org/abs/2609.15972
priority: Good-To-Read
read_on: 2026-09-19
tags: [paper, llm, vision]
---
## The Core Idea

Training an assistant to be "human-aware" — to answer the question a person *actually* has, given their unspoken goals, mood and constraints — runs into a data problem. The training examples you want are conversations where the helper already knows the person well. Those conversations are private, rare, and expensive to annotate. Public chat logs have scale but the user's inner state is invisible, so a model trained on them never learns to act on it.

The trick here: **generate the user's hidden state and the conversation together, then let the teacher read the hidden state while the student never sees it.**

Concretely. A simulator invents a persona and a structured "mental state" record $s_t$ that updates every turn. That same $s_t$ drives two things at once: what the fake user says, and what an **Oracle** assistant answers. So the Oracle is not guessing why the user asked — it is *told*. It writes a reply informed by the real reason. Then you throw the state away and train a student on only `(visible dialogue, Oracle reply)` pairs with plain [[Cross Entropy|cross-entropy]].

That asymmetry is the whole contribution. Earlier synthetic-dialogue work either used a static persona string (no evolution) or built realistic user simulators and let an off-the-shelf assistant reply — but that assistant has to *infer* the hidden state from the text, so its inference errors get baked into the labels. Here the label-maker has ground truth by construction, because the ground truth was made first.

> [!NOTE] Privileged distillation ^privileged-distillation
> Teacher sees $(x, s)$; student sees only $x$. The student cannot copy the teacher's *reasoning* — it has no access to $s$ — so it must learn a policy that hedges over all states consistent with $x$. Formally the student's optimum is the state-marginalised teacher $\bar q(y \mid x) = \mathbb{E}_{s \sim q(\cdot\mid x)}[q(y\mid x,s)]$.

What it unlocks: personalisation supervision you can mass-produce without a single human annotation per dialogue. 8,244 training examples, LoRA fine-tune, and preference-following generation jumps 26.6–40.9 points on three different 7–8B backbones.

## The Methodology

Three named artefacts: **M2D-Sim** (the generator), **M2D-Corpus** (the data), **M2D-Chat** (the trained student).

**The rollout loop.** At each turn $t$, three generation calls, all sharing a GPT-4o-mini backbone but with different prompts:

$$s_t \sim \pi_{\mathrm{state}}(\cdot \mid p, s_{t-1}, H_{<t})$$
$$m_t \sim \pi_{\mathrm{user}}(\cdot \mid p, s_t, H_{<t})$$
$$a_t \sim \pi_{\mathrm{oracle}}(\cdot \mid p, s_t, H_t)$$

where $p$ is the persona, $H_{<t}$ the history, $m_t$ the user message, $a_t$ the Oracle reply, $H_t = (H_{<t}, m_t)$. Note the next state update sees $a_t$ too — so if the assistant says something helpful, the simulated user's trust and mood move in response. The loop is closed.

**The state record.** $s_t = (c_t, z_t^{\mathrm{stable}}, z_t^{\mathrm{transient}})$:
- $c_t$ — turn index, unresolved goals, trust history, running summary.
- $z^{\mathrm{stable}}$ — values, background constraints, stance toward the assistant. Moves slowly.
- $z^{\mathrm{transient}}$ — mood, current worry, judgement of the last reply. Moves every turn.

This is a hand-designed schema written into a prompt, not a learned latent. The authors are explicit that these are control variables, not measurements of anything real.

**Scenario construction.** Each dialogue starts from a scenario generated from the persona, in three families: *Lifelong* (identity, long-term development), *High-Frequency* (everyday needs — debugging, recipes, homework), *Affective* (grief, uncertainty). Candidates are rejected if cosine similarity to an existing scenario exceeds $\theta_{\mathrm{sim}}$, if the abstraction level is wrong, or if they clash with the persona. The prompts (Appendix G) are remarkable — the Lifelong generator is instructed to anchor scenarios in Erikson stages, Bowen family systems, Stroebe's dual-process grief model, and to plant a "narrative echo" linking two scenarios by a shared psychological thread.

**Behaviour control.** A controller picks one of 16 modes per turn — 14 drawn from the Taxonomy of User Needs and Actions (TUNA) plus two fallbacks (`compound_request`, `default_behavior`). Six families: information seeking, information processing, procedural guidance, content creation, social interaction, meta-conversation. Modes differ in *cognitive delegation level* — `retrieval` is low (just look this up), `content_generation` is very high (write it for me). The controller is explicitly told to avoid repeating recent modes and to spread across families. Less behavioural steering at turn 1, more later.

**Filtering.** Six gates before a dialogue is kept: four programmatic (schema validity; turn count / role alternation / token bounds; state-trajectory completeness; profile binding) and two LLM-judge (persona consistency on a 1–5 Likert; profile-contradiction detection labelled `no_contradiction` / `unclear` / `contradicts`). The judge is a different model from the simulator and the Oracle.

**Two views of each trajectory.** The dialogue view pairs student-visible context with the Oracle reply. The QA view feeds the persona, the *saved state trace* and a dialogue excerpt to a question generator, producing multiple-choice persona-memory items, free-form preference items, and preference-classification items. The student sees only the question and visible context — never the state.

**Training mixture** — 8,244 examples total:

| View | Component | Count |
|---|---|---|
| Dialogue | multi-turn Oracle supervision | 3,312 |
| QA | PersonaMem-format MCQ | 2,052 |
| QA | open-ended preference | 1,442 |
| QA | preference classification | 1,438 |

**Loss.** Plain autoregressive [[Cross Entropy|cross-entropy]] on response tokens only:

$$\mathcal{L}_{\mathrm{SFT}}(\phi) = -\mathbb{E}_{(x,y)}\left[\sum_{j=1}^{|y|}\log \pi_\phi(y_j \mid x, y_{<j})\right]$$

Minimising this is equivalent to minimising $\mathbb{E}_x D_{\mathrm{KL}}(\bar q(\cdot\mid x) \,\|\, \pi_\phi(\cdot\mid x))$ — see [[KL Divergence#^kl-is-ce-minus-entropy|CE = entropy + KL]]. No on-policy term, no auxiliary state-prediction head, no distillation temperature. The Oracle is frozen.

**Training setup.** [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]] with 4-bit quantisation ([[QLoRA- Efficient Finetuning of Quantized LLMs|QLoRA]]), response-only masking, [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] with a cosine schedule. Identical recipe across all three backbones and all four data fractions ($1/8$, $1/4$, $1/2$, All). Backbones: Qwen2.5-7B-Instruct (primary), Llama-3.1-8B-Instruct, OLMo-3-7B-Instruct.

**Corpus stats.** 289 personas, 6,330 analysed multi-turn conversations, 47 scenario categories (2,882 lifelong / 2,020 high-frequency / 1,428 affective), 153 distinct persona specialisations, ~65k user turns. Long-tailed: the top 9 categories are ~52% of the data.

## Ablation Studies and Experiments

**Evaluation is deliberately two-domain.** Personalisation (PersonaMem-v1, PersonaMem-v2, PrefEval) tests whether the model *uses* what it knows about a user. Theory of Mind (ToMi, BigToM) tests whether it can *reason* about someone's beliefs and predict their actions. The justification is SimpleToM: a model can attribute a belief correctly and still fail to act on it, so either suite alone is misleading.

**Main personalisation table (all non-proprietary on Qwen2.5-7B, accuracy %):**

| Method | PM-v1 MCQ | PM-v2 MCQ | PrefEval Gen | PrefEval Cls |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct | 49.9 | 32.0 | 23.4 | 62.0 |
| PersonaVLM | 33.9 | 24.3 | 29.8 | 54.7 |
| HumanLM | 38.1 | 27.3 | 43.6 | 78.5 |
| LLMoPt | 35.3 | 22.2 | 41.5 | 75.1 |
| Mem0 | 54.3 | 39.8 | – | – |
| **M2D-Chat** | **56.4** | **42.0** | **56.8** | **78.9** |
| GPT-4o-mini | 48.6 | 37.3 | 21.1 | 84.6 |
| GPT-5-mini | 61.5 | 49.2 | 90.8 | 99.3 |

Three things to notice. First, three of the four task-specific baselines *hurt* the base model on PersonaMem — PersonaVLM drops v1 by 16 points. Second, the PrefEval-Classification margin over HumanLM is 0.4 points; the authors say outright this does not support a separation claim. Third, GPT-5-mini beats M2D-Chat everywhere, by 34 points on PrefEval-Gen — the teacher here was GPT-4o-mini, which the student *does* beat in three of four columns.

**Scaling.** For every backbone, the full mixture is best on every personalisation metric. Full-mixture gains over base:

| Backbone | PrefEval Gen | PM-v1 MCQ | PM-v2 MCQ | ToMi | BigToM FB | BigToM FA |
|---|---|---|---|---|---|---|
| Qwen2.5-7B | +33.4 | +6.5 | +10.0 | +1.8 | +13.0 | +7.5 |
| Llama-3.1-8B | +40.9 | +6.3 | +13.6 | +7.0 | +24.8 | +17.8 |
| OLMo-3-7B | +26.6 | +5.6 | +10.2 | +2.2 | **−8.2** | **−7.0** |

**The OLMo result is the honest part.** Same data, same recipe, and OLMo's belief-attribution *falls* from 30.2 to 22.0 while its preference-following rises 26.6 points. The authors do not have an explanation and say so. This is precisely why the two-suite evaluation earns its keep: a personalisation-only paper would have reported three clean wins.

**Generation and selection scale differently.** On Qwen, going from $1/4$ to $1/2$ of the data moves PrefEval classification +8.5 but generation only +0.7. The final step to full data moves generation +18.9 and classification +3.1. Endpoint numbers alone would hide this. On Llama, PersonaMem-v2 generation is essentially saturated at $1/8$ (44.0 → 45.6 at full), while MCQ keeps climbing (34.2 → 37.5).

**The supervision-view ablation (Llama-3.1-8B) — the most informative table:**

| Training | PM-v2 MCQ | ToM macro | PrefEval Gen | PM-v2 Gen |
|---|---|---|---|---|
| Base | 23.9 | 40.5 | 24.5 | 33.9 |
| Dialogue only | 24.8 | 46.3 | **66.0** | **45.8** |
| QA only | **38.6** | 52.3 | 22.7 | 31.7 |
| Full mixture | 37.5 | **57.1** | 65.4 | 45.6 |

**QA-only training makes generation worse than the untrained model** — 22.7 vs 24.5 on PrefEval, 31.7 vs 33.9 on PersonaMem-v2. Training only on multiple-choice answers teaches the model to pick options and degrades its ability to write a helpful reply. Symmetrically, dialogue-only barely moves MCQ (+0.9). Neither view alone does the job, and the ToM benefit is a *joint* effect: 46.3 and 52.3 separately, 57.1 together. Caveat the authors flag: this ablation changes data *volume* as well as content, so it is not a clean isolation.

**Simulator ablation (Table 8, persona specificity 1–5 at turn 20):**

| Condition | State | Behaviour | $t{=}0$ | $t{=}20$ | $d$ vs M2D-Sim |
|---|---|---|---|---|---|
| Vanilla | ✗ | ✗ | 3.31 | 3.20 ↓ | −0.66** |
| Profile-only Oracle | ✗ | ✓ | 3.35 | 3.28 ↓ | −0.49* |
| Stateful, no controller | ✓ | ✗ | 3.26 | 3.35 | −0.15 |
| M2D-Sim | ✓ | ✓ | 3.32 | 3.40 | ref. |

The state document is doing the work; the behaviour controller adds a little. **Both stateless conditions get *less* persona-specific as the conversation goes on** — the persona washes out. Both stateful conditions improve. The gap is invisible at turn 0 (all ~3.3) and only opens with length, which is the point: a static persona string is fine for one turn and decays after that. Topic-depth effect sizes grow from $d{=}0.16$ over five turns to $d{=}1.22$ over thirty. Correlation of effect size with turn index: $r = 0.90$.

A nice control: Vanilla's user messages are **2.4× longer** (97.6 vs 40.6 words). So the stateless simulator is more verbose and *less* persona-specific — verbosity does not explain the gap. Breaking the effect down by field, it is concentrated in `goal` ($d{=}1.03$) and near zero for `identity` ($d{=}{-}0.05$) and `communication` ($d{=}{-}0.08$). Static personas get the *who* right; they lose the *why*.

**Contamination audit.** Exact 13-gram matching against all five benchmarks: zero matches. But the PersonaMem QA component was built to match PersonaMem's four-option interface, so those MCQ gains are partly format familiarity — stated plainly in §F.6.

**Human audit.** 1,240 uniformly sampled post-filter conversations, binary rubric (persona consistency, trajectory coherence, state-response consistency, response relevance). 1,216 passed = 98.06%. Most failures were "prompt seed capture" — a generic scenario displacing the assigned persona.

**Alignment check (Fig. 9).** Assistant text is closer in cosine similarity to the *scenario* than to the *static persona*, $p<0.001$ by Wilcoxon in all three families. The Oracle is responding to the situation more than to the profile — consistent with the design, and a warning that "personalisation" here may be partly scenario-following.

## Worth Remembering

**The teacher is GPT-4o-mini.** Both the user simulator and the Oracle. Everything the corpus knows about human mental states is what a small, cheap 2024 model believes about them. The authors list this first among limitations and do not test a stronger teacher, an ensemble, or a different family. The students do beat GPT-4o-mini on most personalisation metrics, which is the usual distillation-of-a-weak-teacher surprise, but the ceiling is visible: GPT-5-mini is 34 points ahead on PrefEval-Gen.

**"Privileged" is doing less work than it sounds.** The Oracle and simulator share a backbone. The privilege is entirely in what appears in the prompt. There is no separate trained privileged policy, no on-policy regularisation, no joint objective — contrast with Penaloza et al.'s Privileged Information Distillation, which does use a shared-parameter conditioned teacher. Here the frozen Oracle plus vanilla SFT is a feature, not a limitation: it means anyone with an API key and a LoRA script can reproduce it.

**The state schema is hand-written and untested against alternatives.** No sensitivity analysis on individual fields, no comparison of schemas. The whole method rests on a prompt template someone wrote, and it might encode distinctions that do not exist in real users.

**No validation against real people.** Everything is synthetic conversations, public benchmarks, LLM judges and a human audit *of the synthetic data*. Nobody checked whether M2D-Chat is actually a better assistant for a real human over weeks.

**Connections worth chasing.** The structure — an agent acting under partial observability of a hidden state that evolves in response to its own actions — is exactly a [[Markov Decision Process|POMDP]], and PUMA (cited) formulates it that way with explicit [[Beliefs|belief]] tracking. Mind2Dialogue instead sidesteps belief tracking entirely: the student never represents $s_t$, it just learns a marginalised policy. That is cheaper and works, but it means the model has no place to put an uncertainty estimate about the user, and no mechanism to ask a clarifying question when the state is genuinely ambiguous. If you care about [[Uncertainty#^epistemic|epistemic uncertainty]] over the user, this architecture gives you nowhere to store it.

The [[Distillation|distillation]] framing is worth pausing on — the student is trained on the teacher's hard outputs only, no logits, no [[Distilling the Knowledge in a Neural Network#^softmax-temperature|temperature]], so none of the "dark knowledge" machinery applies. It is behaviour cloning of a better-informed policy.

**Practical caveats if you wanted to use this.** (1) Never train on the QA view alone — it degrades generation below baseline. (2) Expect backbone-dependent results; budget for testing on your own model rather than trusting the headline. (3) If you copy the format of your evaluation benchmark into your training data, say so — the PersonaMem MCQ gains are partly interface familiarity. (4) The dataset and code are released, so the corpus is directly reusable.

**Open questions.** Why does OLMo regress on BigToM? Does the gain survive a strong teacher, or is it mostly recovering ground GPT-4o-mini already had? Would predicting $s_t$ as an auxiliary task beat marginalising it away? And the one the authors raise: could you mine *real* conversations for moments where the assistant clearly used unspoken context, and use those as a reality anchor for the simulator?

## Links

Related: [[Distillation]] · [[Distilling the Knowledge in a Neural Network]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[QLoRA- Efficient Finetuning of Quantized LLMs]] · [[Cross Entropy]] · [[KL Divergence]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Fine-Tuning]] · [[Instruction Tuning]] · [[Beliefs]] · [[Markov Decision Process]] · [[Uncertainty]] · [[Evals]] · [[Best Practice Critic Optimization]] · [[Training language models to follow instructions with human feedback]] · [[Embeddings]]

New topics worth writing: Learning Using Privileged Information (Vapnik & Vashist), generalized distillation, LLM user simulators as an evaluation substrate, Theory-of-Mind benchmarks (ToMi, BigToM, FANToM), persona-conditioned synthetic data generation, benchmark contamination auditing via n-gram matching, Cohen's d and effect sizes for LLM-judge comparisons, POMDP formulations of dialogue assistance, Taxonomy of User Needs and Actions (TUNA)
