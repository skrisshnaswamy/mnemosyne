---
title: "The Tasteful Agent: Measuring and Improving Taste in Long-Horizon Tasks"
authors: ["Wenbo Pan", "Zhichao Liu", "Shujie Liu", "Jingying Zeng", "Chin-Yew Lin", "Xianfeng Tang", "Yan Lu", "Qi He", "Xiaohua Jia"]
year: 2026
arxiv: "2609.25804"
url: https://arxiv.org/abs/2609.25804
priority: Good-To-Read
read_on: 2026-09-27
tags: [paper, llm]
---
## The Core Idea

An agent working a long task makes lots of small choices. Which fix to try. Which hypothesis to test. Which implementation to build on. Most of these choices look fine at the moment you make them — and the bad one only shows its cost an hour later, after the budget is gone.

The paper calls the ability to pick well at these moments **taste**, and points out that nobody measures it. Benchmarks like SWE-bench only say whether the agent finished. They say nothing about whether the route it took was any good.

> [!NOTE] Taste
> The ability to choose the better direction at a point in a trajectory where the outcome of each direction is not yet visible. Measured, not vibes: pick between two branches, get scored against what actually happened afterwards. ^taste-def

The obstacle is labels. To grade a decision you need to know which direction was better, and that needs either an expert or a crystal ball. Experts are expensive and do not scale across domains.

The trick that unlocks it: **the rest of the trajectory is the label**. Agents already run the same task many times. Those attempts often share an identical opening and then split. One passes, one fails. That split is a free, outcome-labelled comparison — nobody had to annotate it.

> [!NOTE] Decision fork
> A point where two attempts at the same task, sharing an equivalent prefix, diverge into different directions. The recorded outcome of each branch names the better direction. ^decision-fork

So: freeze the trajectory at the fork, delete everything after it, show a model the two directions, ask it to pick. This is `Taste-Bench` — 502 questions built this way.

Three results. Best frontier model gets **59.7%** (random guessing on their strict metric is 25%). Accuracy collapses to near-chance when the deciding evidence lives far in the future. And more thinking tokens do **not** help — which is the interesting one, because it suggests this is not a reasoning-effort problem but a missing-information problem.

Then they show taste is trainable: distil a teacher that was shown the answer into a student that was not, and end-to-end success on held-out SWE-bench Pro tasks goes from 14.6% to 33.7%.

The connection to your world: this is [[Off-Policy Evaluation|off-policy evaluation]] wearing engineering clothes. Two branches share a prefix, sampling randomness assigns the direction, the realised outcome estimates $\mathbb{E}[U \mid h_t, c_i]$. That is a randomised comparison mined out of logs. It is also [[Credit Assignment|credit assignment]] — attributing a final outcome to one step in the middle.

## The Methodology

### The formal setup

Two attempts at a task $q$ share a prefix

$$h_t = (o_0, a_0, \ldots, o_t)$$

where $o$ is an observation and $a$ an action. At step $t$ they split into candidate directions $c_1$ and $c_2$. Each branch runs to the end, producing evidence $E_i$ (test results, research scores), and an outcome function $U$ turns that into a number.

The label is simply whichever branch scored higher:

$$y = \arg\max_{i \in \{1,2\}} U(E_i)$$

The question shown to the model is $x = (q, h_t, c_1, c_2)$ — everything after $t$ is hidden. Taste is estimated as the fraction of questions where $\pi(x) = y$.

The argument for why this is fair: the prefix is the same on both sides, and sampling randomness — not any deliberate design — decided which branch got which direction. So the outcome gap is mostly attributable to the direction. It is the same logic as [[AB Testing|randomisation giving you causation]], just discovered after the fact rather than designed in advance.

### Two ways to mine a fork

**Parallel forks.** Take two attempts at the same task with opposite outcomes. Find where they diverge. The passing attempt's prefix becomes $h_t$; the two divergent actions become the candidates, rewritten in neutral parallel wording. These capture mistakes the agent *never noticed* — it ran the bad direction to completion and failed.

**Detour forks.** Take one trajectory where the agent went wrong and then corrected itself. The generator must cite four excerpts in strict order: commitment to the bad direction, the observed failure that killed it, commitment to the recovery, and evidence the recovery worked. The fork is placed *just before* the bad commitment. These test whether a model can see the failure coming earlier than the acting agent did.

The two are complementary: one gives you uncaught errors, the other gives you caught-but-late errors.

Three mechanical checks reject detour candidates: every quote must appear verbatim in the cited step, the four steps must be in order, and the "wall" must contain a real failure signal (non-zero exit, traceback, failing assertion). A self-critical remark is not a wall.

### The source pools

| Domain | Source | Size |
|---|---|---|
| Engineering | Their own GPT-5.4/5.5 runs on SWE-bench Pro | 2,677 graded rollouts, 517 tasks, 11 repos |
| Research | MALT, METR's public transcripts (RE-Bench + HCAST research subset) | 1,132 runs, 47 tasks |

### Filtering — the part that makes it a benchmark rather than a pile

A generator (GPT-5.6 Sol, high reasoning) proposes candidate forks under a long rubric. Then a panel of four judges — Kimi K2.5, GPT-4.1, Llama 4 Maverick, Mistral Large 3, none of them the generator — runs two screens:

1. **Trivial screen.** Each judge sees *only the two candidate texts*, no trajectory. If all four answer correctly, the question is thrown out — the answer was leaking through the wording. Example killed this way: one candidate literally described itself as hard-coding three answers and returning empty strings for everything else. No judgment needed.

2. **Undecidable screen.** Each judge sees the full record — task, prefix, both complete continuations, recorded outcomes. The question survives only if all four agree with the mined label. This catches cases where the label is an artifact rather than a consequence: one removed example had the "wrong" branch showing 80% attack success in the prefix and only losing because it later timed out, with no time budget stated in the task.

Yield: 4,657 candidates in, **502 out — 10.8%**. That aggressive rejection is the point.

Composition is a $2\times2$: parallel/detour × engineering/research. 390 engineering, 112 research.

**Human check.** 100 sampled questions, two reviewers, two-stage interface (choose blind, then see outcomes and judge which was better). Of 172 explicit A/B judgments, 170 matched the mined label — **98.8% agreement**, Cohen's $\kappa = 0.973$ between reviewers.

### Scoring, and the position-bias fix

Two-choice questions suffer badly from position bias — many models just prefer whatever is labelled A. So every question is asked twice, once in a seeded order and once exactly reversed.

**A question counts as correct only if both orders are answered correctly.** This makes random guessing score $0.5 \times 0.5 = 25\%$, and a model that always picks the same slot scores **0%**. They also report mean accuracy over the two orders as a softer number.

All 14 models get the same prompt, same 65,536-token output budget. Unparseable responses count as wrong.

### The distillation recipe

Base model is Qwen3.6-27B; training touches only [[LoRA]] adapters (rank 16, $\alpha=32$, 79.7M trainable params, all attention and MLP projections).

**Folds.** The 390 engineering questions split by *task* into two folds of 195. No shared question, task, source trajectory or prefix. Each student trains on one fold, is evaluated on the other.

**Why not just fine-tune on the labels?** Each question gives one binary bit and no reasoning. A model fitted on the bits memorises answers rather than learning judgment. So they distil *reasoning traces* instead.

**Privileged teacher.** Teacher and student are the *same frozen base model* with different contexts. The teacher's context has the question plus a **demonstration** — a short note naming the supported candidate. Knowing the answer, the teacher reliably reasons its way to it (328/328 and 310/311 kept traces). Training then aligns those same tokens under the *student* context, which contains only what the benchmark shows: task, prefix, two shuffled candidates.

This is [[Distillation|context distillation]] with privileged information — the same family as STaR and LEAP.

**Loss.** SDPO-style token-level distillation with a forward [[KL Divergence|KL]]:

- At each reasoning position: forward KL over the student's top-100 tokens plus one extra bucket holding the remaining mass, so nothing is dropped. Long traces subsample to at most 512 positions.
- At the answer position: forward KL restricted to the two answer tokens.
- Both terms weight one.

**The KL is computed on continuations sampled from the teacher, not the student.** Their reasoning: the two contexts differ exactly on the tokens where the demonstration changes the teacher's prediction, and the student would almost never produce those tokens on its own — so a student-sampled KL would never see the disagreement it is supposed to fix. This is a deliberate step *away* from standard [[On-Policy vs Off-Policy|on-policy]] distillation.

**Leakage control is structural, not textual.** The answer is a closed A/B choice and the teacher sequence stops at the answer, so no output span can be a copy of the demonstration text.

**Calibration pass.** After distillation, the student generates its own free traces on the training fold and the answer position alone is trained with [[Cross Entropy|cross-entropy]] on the supported label (LR $10^{-5}$, one epoch, softmax temperature 2.0). This nudges only the final choice under the student's own reasoning.

Distillation: LR $10^{-4}$, 2 epochs, AdamW, [[On the difficulty of training Recurrent Neural Networks|gradient clipping]] at norm 1.0. ~2 hours per fold on one A100 80GB, peak 71GB.

### Advice injection

To test whether better judgment helps *doing*, not just *judging*: for each held-out task, the student answers every mined fork question, and its choice is written as a note of advice — the situation, the candidate to avoid, the candidate to take — placed before the problem statement in the executor's first message.

A separate fixed executor (SWE-agent 1.1.0 on Qwen3.6-27B, thinking off, temp 0.7, 75 model calls, 4,800s budget) then does the task. Student and executor never interact.

The advice template is worth noting: it loudly tells the executor the notes are *not evidence about its own patch*, and imposes a "submission contract" requiring the agent to actually run a check and see the output before submitting.

## Ablation Studies and Experiments

### Headline: nobody is good at this

14 frontier models. Best is **GPT-5.6 Sol at 59.7% Average** (the 1:1 mean of research and engineering subsets). GPT-5.5 at 59.5%. Random is 25%.

| Model | Average | Eng | Res | Mean acc | Unparsed |
|---|---|---|---|---|---|
| GPT-5.6 Sol | 59.7 | 56.9 | 62.5 | 65.2 | 0 |
| GPT-5.5 | 59.5 | 55.6 | 63.4 | 64.6 | 0 |
| Claude Opus 5 | 55.5 | 46.7 | 64.3 | 60.3 | 0 |
| Grok 4.5 | 54.6 | 52.1 | 57.1 | 62.0 | 10 |
| GLM-5.2 | 53.9 | 47.9 | 59.8 | 64.0 | 17 |
| DeepSeek V4 Flash | 43.3 | 38.5 | 48.2 | 54.9 | 2 |
| Grok 4.20 Reasoning | 15.7 | 22.6 | 8.9 | 29.0 | 459 |

Grok 4.20's 15.7% is mostly a formatting failure — 459 of 1,004 responses unparseable. Worth remembering that the strict both-orders metric punishes instruction-following problems as hard as it punishes bad judgment.

### Construction matters more than domain

Mean over 14 models, per cell:

| Cell | Mean accuracy |
|---|---|
| Parallel engineering | 58.1% |
| Parallel research | 50.9% |
| Detour research | 50.8% |
| **Detour engineering** | **35.9%** |

Research is *not* uniformly harder than engineering. The construction gap beats the domain gap. Parallel-engineering is easiest because the two branches have cleanly separated recorded outcomes; detour-engineering is hardest because it demands you spot the mistake before the acting agent did.

### The time-horizon result — the most useful finding

Every fork is annotated by a judge (GPT-5.5) with how far into the future you must see before the right answer is justified:

- **in prefix** — a decisive fact already visible rules out one candidate
- **inferable** — no single decisive fact, but the hints together justify it
- **next step** — the first observation after the fork settles it
- **more work** — needs a completed local check or substantial later work

(Five scores collapsed to four; the top two merged because only 13 questions hit the highest.)

Mean accuracy over 14 models:

| Level | $n$ | Mean accuracy |
|---|---|---|
| In prefix | 158 | 62.3% |
| Inferable | 219 | 42.9% |
| Next step | 56 | 31.5% |
| More work | 69 | **21.0%** |

At the "more work" level, the *average frontier model is below random guessing*. Even GPT-5.6 Sol only manages 21.7% there, against 79.7% on in-prefix questions.

This is the finding with teeth. Whatever models are doing well at, it is reading evidence that is already on the page — not predicting consequences.

### The negative result: reasoning budget does nothing

Three reasoning-effort settings each for two models, 6,024 responses, everything else held fixed.

| Model | Setting | Accuracy | More-work level |
|---|---|---|---|
| GPT-5.6 Sol | low | 56.2 | 20.3 |
| GPT-5.6 Sol | xhigh | 55.6 | 21.7 |
| GPT-5.6 Sol | max | 56.0 | 23.2 |
| GPT-5.6 Luna | none | 43.4 | 13.0 |
| GPT-5.6 Luna | high | 45.8 | 18.8 |
| GPT-5.6 Luna | max | 45.6 | 15.9 |

Low → high moves Sol by $-0.2$ points, Luna by $+2.2$. Bootstrap intervals: $[-3.2, +2.8]$ and $[-1.6, +6.0]$. A logistic model with a budget×horizon interaction finds nothing ($p = 0.80$, $p = 0.82$). No response hit the token limit, so this is not truncation.

The diagnostic detail: **both models spend the most reasoning tokens at the more-work level** — the level where they score worst. They know these questions are hard. They think longest on them. They still cannot answer them.

The reading: this is not a [[Test-Time Compute|test-time compute]] problem. The evidence genuinely is not in the context, and no amount of staring at it will conjure it. Contrast with [[Chain of Thought|chain-of-thought]], where extra tokens buy real compute on problems whose inputs are all present.

One domain split worth noting: Luna's *research* subset does improve with budget, 43.8% → 54.5%, while engineering stays flat. Small $n$, so treat as suggestive.

### Is this just SWE-bench in disguise?

Against public SWE-bench Verified scores (Vals AI leaderboard, one shared harness), 11 models after dropping three with >9% unparseable rate:

- Average vs SWE-bench Verified: $r = +0.63$, so $R^2 = 0.39$
- **Engineering subset alone: $r = +0.37$** — even though this subset is mined from SWE-bench Pro tasks

The engineering subset being *less* correlated than the mixed Average is the counter-intuitive bit, and it argues the benchmark measures something separate rather than a noisier restatement.

The top of the leaderboard separates: the four highest SWE-bench Verified models sit within 4.0 points of each other there, but 10.7 points apart on Taste-Bench. Two models swap hard — DeepSeek V4 Flash is 5th on SWE-bench and 10th here (with only 2 unparsed responses, so not a formatting artifact), GPT-5.5 is 8th there and 2nd here.

**Caveats the authors state:** reasoning-effort settings are not matched between the two evaluations, and at $n=11$ the correlation interval is wide — $[+0.04, +0.89]$ for the Average ($p=0.04$), $[-0.30, +0.79]$ for engineering. The engineering correlation is not statistically distinguishable from zero *or* from strong. Treat "the benchmarks are different" as **suggestive, not established**.

### Distillation transfer

On the *training* fold, single-order accuracy jumps 48.6% → 92.9% ($p \approx 8\times10^{-9}$). That number proves nothing on its own — it is consistent with memorisation. The task-disjoint fold is what separates the two.

On the held-out 390 questions, strict both-orders scoring:

| | Accuracy | Mean accuracy |
|---|---|---|
| Base Qwen3.6-27B | 30.0% | 42.7% |
| Distilled student | **47.9%** | **62.4%** |

$+17.9$ points on tasks never seen in training. Flip analysis: 104 questions fixed, 34 broken. Placed in the engineering ranking, the student lands equal to GLM-5.2 (47.9%), above Claude Opus 5 (46.7%), below GPT-5.6 Sol (56.9%). The base model sits just below GPT-5.4 Nano.

A 27B model with LoRA adapters, beating Claude Opus 5 on this slice.

### End-to-end: does judgment turn into success?

41 held-out SWE-bench Pro tasks, 98 forks, official evaluation:

| Setting | Success rate |
|---|---|
| No advice | 14.6% |
| **Student advice** | **33.7%** |
| Correct advice (oracle ceiling) | 39.0% |

Oracle advice buys 24.4 points (exact McNemar $p \le 0.004$). The student captures 19.1 of those 24.4 — about 78% of the available headroom. Its raw fork accuracy on these tasks was 77/98.

The oracle setting is the honest framing: it is the ceiling on what *any* advisor could deliver through this channel, so the student's number is measured against a real bound rather than against nothing.

### What did not work, or is quietly fragile

**Fine-tuning on labels directly** — rejected upfront on the grounds that one binary bit per question with no intermediate reasoning produces memorisation. Not empirically compared against, which is a gap; you have to take the argument on faith.

**Standard on-policy distillation** — explicitly abandoned. The teacher-sampled KL is a deliberate departure, justified by the argument that the student would never generate the divergent tokens on its own. Also not ablated.

**The judges are not good judges blind.** In the parallel-engineering cell, judges on the released questions score 34.7%–63.7% from candidate text alone, with pairwise agreement of only 51.7% — barely above chance. That is by construction (the filter kept exactly the questions where at least one judge failed blind), but it means the released set is selected for judge disagreement, and the panel's composition shapes what survives.

**Two rejected-example categories** are documented and worth reading as failure taxonomy: trivial questions where a candidate self-incriminates in its wording, and undecidable ones where the prefix evidence actually favours the *losing* branch and the label comes from an unstated constraint like a time budget.

**Truncation.** One question exceeds the 65,536-token prompt limit for Claude Opus 5 and Claude Sonnet 5. Negligible, but the rendering scheme (head + tail, middle elided, command outputs clipped at 700 chars) is a real information loss that is not ablated.

## Worth Remembering

**The mining idea generalises past this paper.** Any system that runs parallel attempts and records outcomes is producing labelled decision data for free. That is [[Experience Replay|a replay buffer]] of judgments. Every one of your A/B infrastructure instincts applies: is the prefix truly equivalent, is the assignment truly random, is the outcome truly downstream of the decision. The 10.8% pass rate is what it costs to be honest about those three questions.

**But the assignment is only *quasi*-random.** The claim is that sampling randomness assigns directions to branches. That is weaker than deliberate randomisation — the branches came from the same model with the same prompt, so any systematic bias in the model's sampling appears on both sides in correlated ways. The generator rubric and judge panel are patching this by hand rather than by design. Contrast with [[Unbiased Offline Evaluation of Contextual-bandit-based News Article Recommendation Algorithms|replay evaluation]], where the logging policy's randomisation is known and the correction is exact.

**No propensities are logged.** There is no $\mu(a \mid x)$ here, so none of the [[Doubly Robust Policy Evaluation and Learning|doubly-robust]] or [[The Self-Normalized Estimator for Counterfactual Learning (SNIPS) (NeurIPS)|self-normalised]] machinery applies. The estimator is a raw win-rate over matched pairs. Clean, but it means the benchmark cannot be reweighted to a different target policy — it measures agreement with *these* trajectories' outcomes.

**The "long horizon is hard" result is the durable one.** Accuracy 62.3% → 21.0% as the deciding evidence moves later is a clean dose-response, and it survives the reasoning-budget control. This is a statement about what is *in the context*, not about how hard the model thinks. It rhymes with [[Credit Assignment#The post-match analysis ⚽|delay being the difficulty]] and with why [[High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)|GAE]] exists — long-delay credit is intrinsically noisy, whether you are a critic or an LLM.

**The horizon annotation is itself a model judgment.** One judge (GPT-5.5) assigns all four levels, with no inter-annotator check on this specific labelling. The headline finding leans on a single LLM's taxonomy. **Current best understanding**, not established fact.

**Advice as an interface is interesting separately from the taste result.** The student model and the executor never share weights, gradients or a context window — the entire channel is a block of text prepended to the task. That makes advice a clean, swappable component: you can improve the advisor without retraining the executor. Related to the advisor-model line of work they cite. Practically it also means advice injection is *only* available for tasks you have already run — the forks come from prior attempts. It is not a general capability boost, it is a replay-and-improve loop.

**The advice prompt is doing defensive work.** It explicitly warns the executor that these notes say nothing about the current patch, and imposes a "run a check and see the output before submitting" contract. Some fraction of the 19.1-point gain may be coming from that contract rather than from the judgments. Not disentangled — a no-advice-but-with-contract arm is the missing baseline.

**Practical caveats for using it:**
- Strict scoring conflates bad judgment with bad instruction-following. Check the unparsed column before reading a model's score.
- The engineering half is mined from SWE-bench Pro rollouts by GPT-5.4/5.5. Models from the same family may be advantaged in ways that have nothing to do with taste.
- The research half is only 112 questions across 30 tasks. Per-cell numbers there are noisy.
- Model names throughout (GPT-5.6, Claude Opus 5, Grok 4.5, DeepSeek V4) are from a 2026 preprint — verify availability before trying to reproduce.

**Open questions worth chasing:**
1. Can the fork label be made an outcome *margin* rather than a binary, so the benchmark measures graded judgment? The research pool already has scalar scores.
2. Does taste transfer across domains? Training is engineering-only; nobody tested the student on research forks.
3. Would a value function trained on trajectory prefixes — literally $V(h_t)$ — beat prompting a frontier model? The more-work failure mode looks exactly like something a learned critic should handle better than a language model, and the paper never tries it.
4. Could this become a process reward signal for [[GRPO]]-style training rather than just advice text?

## Links

Related: [[Off-Policy Evaluation]] · [[Credit Assignment]] · [[Distillation]] · [[Distilling the Knowledge in a Neural Network]] · [[LoRA]] · [[KL Divergence]] · [[On-Policy vs Off-Policy]] · [[AB Testing]] · [[Agent Evaluation]] · [[Agentic Workflows]] · [[Evals]] · [[Test-Time Compute]] · [[Chain of Thought]] · [[What Does Privileged Information Add to On-Policy Self-Distillation]] · [[Best Practice Critic Optimization]] · [[DRACO- Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training]] · [[An Empirical Study of Harness Design for Coding Agents]] · [[Doubly Robust Policy Evaluation and Learning]] · [[Value Function]]

New topics worth writing: process reward models, SWE-bench Pro, position bias in LLM-as-judge, context distillation, privileged-information training (LEAP / STaR), SDPO, advisor models, Cohen's kappa, McNemar's test, agent trajectory mining
