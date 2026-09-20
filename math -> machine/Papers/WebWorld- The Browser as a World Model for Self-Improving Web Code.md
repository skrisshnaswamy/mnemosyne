---
title: "WebWorld: The Browser as a World Model for Self-Improving Web Code"
authors: ["Jiajun Wu", "Jian Yang", "Yaxin Du", "Wei Zhang", "Haowen Wang", "Junhang Cheng", "Yuxuan Zhang", "Tuney Zheng", "Xianglong Liu", "Ming Zhou"]
year: 2026
arxiv: "2608.30530"
url: https://arxiv.org/abs/2608.30530
priority: Good-To-Read
read_on: 2026-09-05
tags: [paper, llm, vision]
---
## The Core Idea

A model that writes a web page and then judges its own web page from a screenshot will fool itself. The screenshot shows a submit button; it does not show that the button does nothing. So the usual self-improvement loop — vision-language model (VLM) looks at a render, complains, rewrites, keeps the rewrite — mostly generates noise. The proposer and the judge are the same network, and both live in pixel space, while the actual defects are behavioural.

The fix here is not a better critic. It is to hand the judging job to something the model cannot argue with: **a real browser**. A browser is already a deterministic, executable simulator of what an HTML file does when you click, type, drag, and press keys. In [[Mastering Diverse Domains through World Models (DreamerV3)|world model]] language, it *is* the world model for web code — except nobody has to train it. It ships with Chromium. The open problem is not learning the dynamics; it is designing the **interface** between a learned prior (the VLM, which is good at guessing plausible repairs) and this free, exact simulator (which is good at proving them).

WebWorld is that interface. Three pieces:

1. **Interaction contract** — the VLM's free-form complaint gets compiled into a typed, checkable claim: *this predicate must become true, under this exact replay action, without breaking this list of things that already worked.*
2. **Acceptance certificate** — the browser re-runs the candidate under the contract and either issues a proof or a typed refusal. Only a certificate promotes a candidate to the new baseline.
3. **Quality ratchet** — the set of verified behaviours only grows. A later patch that fixes a new bug but kills an old working control is rejected.

The unit of supervision stops being "a good final HTML file" and becomes **a certified state transition**: (broken page, observed failure, contract, fixed page, proof). Fine-tune on those and the model gets meaningfully better at making pages that actually work.

The reason this matters beyond web code: it is the same trick SWE-bench gets for free from unit tests, applied to a domain that has no unit tests. [[Agentic Game Development as a Verifiable Trajectory Data Engine for Scaling World Models#^unverifiability-tax|The unverifiability tax]] is paid by finding a verifier that already exists in the environment.

> [!NOTE] Hypothesis–proof separation
> The generator may only *propose*. A separate, non-learned oracle decides what enters the training pool. No claim reaches supervision without re-execution. ^hypothesis-proof

## The Methodology

**The loop, one round.** The runtime opens the current HTML artifact $x_t$ in headless Chromium/Playwright, drives it through a fixed catalogue of 14 generic interaction probes (click, drag, keyboard, pointer hold, form submit, hover reveal, SVG animation, canvas focus, …), and records an observation $o_t$: screenshots, DOM facts, console output, probe pass/fail, and the executed action trace.

**Critique.** The VLM sees $o_t$ — *not the source code* — and emits a structured critique $c_t$: which issue family it sees, which piece of evidence in $o_t$ supports that, the affected region, the success condition, what must be preserved, and a suggested repair skill. "Make it more polished" is invalid by construction; every field must point at recorded evidence.

**Contract.** A deterministic planner compiles $c_t$ into contract $r_t$, a triple:
- **target predicate** over post-repair page state,
- **replay** — a concrete action sequence anchored to stable CSS selectors that the verifier can re-run,
- **preserve set** $\mathcal{P}_t$ — predicates inherited from all earlier certified rounds,
- plus an **impact scope** so edits that wander into unrelated regions fail the contract itself.

This compilation is the first filter. Critiques anchored to broad containers, or that are purely perceptual, or that assert DOM state with no screenshot-observable consequence, are rejected before any repair is attempted — they could never be certified later, so there is no point spending budget.

**Repair.** A router picks one of 20 typed repair skills (15 VLM lanes: gameplay, control wiring, SVG, structural completion, visual, maintenance, plus a bounded holistic rewrite; 5 deterministic patch lanes for mechanical fixes like a missing dependency). The output diff is bounded by the contract's impact scope. A pre-execution filter throws away candidates that obviously cannot advance the contract. All of this is *proposer-side* — it sharpens the guess, it proves nothing.

**Verification.** The verifier is

$$\mathcal{V}(q, x_t, r_t, \hat{x}_{t+1}, \mathcal{P}_t) \rightarrow \{\textsc{reject}, \textsc{accept}(e_t)\}$$

and accepts only when all three hold:

$$\textsc{target}(r_t, x_t) = 0,\quad \textsc{target}(r_t, \hat{x}_{t+1}) = 1,\quad p(\hat{x}_{t+1}) = 1 \ \ \forall p \in \mathcal{P}_t$$

In words: the target was genuinely broken before, it is genuinely fixed after, and **every** previously certified capability still passes. That third clause is stricter than "don't regress on tests" — it is a monotone memory that grows with the trajectory:

$$\mathcal{P}_{t+1} = \mathcal{P}_t \cup \textsc{verified}(e_t)$$

**Proof levels**, tried in priority order, and the certificate records only which one won:
1. **Same-trace replay** — the contract's exact action now produces the target state.
2. **Capability gain** — a probe macro that used to fail now passes.
3. **Target-issue progress** — the predicate is true under an equivalent replay.
4. **Localized visual evidence** — diff and visual change both lie inside the impact scope.
5. **Static structural repair** — non-interactive content addition with low preserve risk.

**IterStep vs QualityStep.** Any safe runtime step (a re-render, a probe run, a refused candidate) is an `IterStep` — it keeps exploration moving and its typed rejection reason ("partial progress", "visual-only", "preserve-risk", "stale replay") feeds the *next* critique. Only a certificate-backed transition is a `QualityStep`, and only `QualityStep`s reach the export:

$$\mathcal{D}_{\text{SFT}} = \{(q, o_t, c_t, r_t, x_{t+1}, e_t) \mid \textsc{QualityStep}(z_t) = 1\}$$

Refused candidates are useful debugging signal but never become training targets. That separation is the whole ratchet.

**Data.** Up to $T_{\max}=10$ rounds per source artifact. About 60K candidates survived contract filtering and reached the browser. Re-execution **refused 24,350** of them. The certified pool is **42,860** transitions, split by winning proof level: same-trace replay 42.3%, capability gain 22.5%, target progress 16.4%, localized visual 11.5%, static structural 7.3%. The SFT corpus is a uniform random 32,800-example subsample (seed 42) of that pool, sized to match the Raw baseline budget exactly.

**Training.** Qwen3.5 backbones at 4B / 9B / 27B. AdamW ([[Decoupled Weight Decay Regularization (AdamW)]]), lr $2\times10^{-5}$, weight decay 0.01, 3% linear warmup, effective batch 128, sequence length 16,384, 3 epochs. 64×H100 for 4B/9B, 128×H100 for 27B. Single seed (42) everywhere. Critique decoding is greedy; patch generation is temperature 0.7, one candidate per contract. Backbone family, prompt template, optimiser, and eval runner are held fixed across every row — **only the supervision source changes**.

## Ablation Studies and Experiments

**Benchmarks.** HTMLBench-400 is the primary test: 400 prompts scored on five dimensions summing to 100, where Functionality dominates and is a weighted pass rate over 6,000 frozen deterministic browser test cases (TC pass). MiniAppBench-Val is the out-of-distribution transfer check.

**Main result** (Raw = same recipe, same budget, supervised on raw HTML):

| Model | HTMLBench | TC pass % | Func. | MiniApp avg |
|---|---|---|---|---|
| WebWorld-4B-Raw | 43.3 | 26.8 | 18.9 | 50.3 |
| WebWorld-4B | **46.7** | 39.3 | 22.3 | **60.8** |
| WebWorld-9B-Raw | 44.5 | 34.1 | 20.3 | 52.9 |
| WebWorld-9B | **49.3** | 40.6 | 26.1 | **66.3** |
| WebWorld-27B-Raw | 47.4 | 33.5 | 21.7 | 70.6 |
| WebWorld-27B | **52.7** | 43.2 | 28.0 | **85.5** |

The 27B gain is +5.3 HTMLBench, +9.7 TC pass, +14.9 MiniApp. The gap widens with capacity: +3.4 at 4B, +4.8 at 9B, +5.3 at 27B. For context (not a controlled comparison), 52.7 sits above Kimi-K2.6 at 49.8 and GPT-5.4 at 49.2 on HTMLBench.

**The gain is entirely in the dimensions the gate constrains.** TC pass and Functionality move hard. Rendering, Visual, and Code barely move — and where they move, they often move *down*: at 9B, Visual drops from 9.1 (Raw) to 8.0; at 27B, 9.3 → 8.9. Interactivity also nudges down (0.7 → 0.6). The paper frames this as "the gate does not optimise for screenshot polish, and does not have to." Fair, but the 9B Visual drop is a full 1.1 points, larger than the "less than a point" the text claims.

**The ablation that carries the paper.** All at 9B, same budget, same recipe, only the acceptance rule changes:

| Acceptance rule | Score | TC | Func. | MiniApp | Δ vs Raw-9B |
|---|---|---|---|---|---|
| Raw-9B | 44.5 | 34.1 | 20.3 | 52.9 | — |
| No certificate | 44.9 | **31.0** | 21.3 | 55.9 | **+0.4** |
| No preserve | 46.0 | 31.9 | 22.0 | 59.7 | +1.5 |
| VLM-only | 48.0 | 33.4 | 23.1 | 62.7 | +3.5 |
| Score gate | 48.1 | 34.0 | 23.4 | 61.4 | +3.6 |
| WebWorld (full) | **49.3** | **40.6** | **26.1** | **66.3** | **+4.8** |

**What did not work, in order of interest:**

- **"No certificate" is worse than useless.** This is the faithful re-implementation of the naive critique-and-rewrite loop: keep the VLM critique, keep the skill routing, drop the browser-issued proof. It gains 0.4 points overall — twelve times less than the full gate — and its **TC pass falls 3.1 points below the Raw baseline** (31.0 vs 34.1). Removing the oracle does not just fail to help; it actively injects supervision that damages functionality. Everything the loop calls progress was noise.
- **Partial substitutes buy score without buying behaviour.** VLM-only and score-gate rules recover ~3.5 points of headline score by accepting more visually-plausible candidates, but they trail the full gate by 6.6–7.2 TC-pass points. They are optimising a proxy. This is a clean, quantified instance of [[Shortcut Learning in Deep Neural Networks#^shortcut|shortcut learning]] in the *data-selection* stage.
- **Dropping only the preserve set costs 3.3 points** (46.0 vs 49.3), concentrated in TC pass and Functionality — exactly where silent regressions hide. Progress-only acceptance lets a patch fix layout while breaking keyboard control.

**Depth diagnostic** (training budget frozen at 5,000 examples, varying minimum certified trajectory depth):

| Min depth | Score | TC | MiniApp |
|---|---|---|---|
| ≥1 | 43.4 | 27.3 | 57.8 |
| ≥3 | 46.8 | 32.8 | 61.1 |
| ≥5 | **48.8** | 36.1 | 58.2 |
| ≥8 | 48.5 | 35.5 | 61.5 |

**5,000 depth-≥5 examples come within 0.5 points of the full 32,800-example model.** That is a ~6.5× data reduction for near-equal performance, and it is the ratchet's prediction made concrete: a trajectory that cleared the gate five times in a row has five layers of accumulated preserve predicates, so each of its examples carries more verified capability than an average certified sample. Score plateaus at depth ≥8 — deeper is not free signal forever.

Accepted repairs by round peak in rounds 4–6, then fade. One-shot repair misses the hard interaction bugs entirely; unbounded looping churns past zero marginal yield. The 10-round cap is doing real work.

**Leakage audit.** Five explicit disjointness checks (item IDs and prompt hashes, prompt templates, test cases, probe macros vs item-specific assertions, repair skills vs task families) with empty intersections. The HTMLCure evaluation runner is held out entirely from construction and training. Worth noting given how easily this kind of pipeline could quietly optimise a benchmark — see [[Towards Quantifying Benchmark Optimization in ASR Models]].

## Worth Remembering

- **The generalisable claim is about *where* the verifier lives, not about web code.** Any domain that has a free, deterministic, executable simulator can run this loop. Web code happens to be an unusually clean case because the simulator is fully specified by a standard and installed on every machine. Compare [[Mastering Chess and Shogi by Self-Play (AlphaZero)]] where the game rules are the oracle, and [[On-policy Distillation with Verifiable Reward]] where the verifier is a math checker.
- **Contrast with the RLHF family.** [[Training language models to follow instructions with human feedback#^reward-model|Reward models]] and [[Direct Preference Optimization (DPO)|DPO]] learn a judge from preferences, which is exactly the modality-bias failure this paper diagnoses. [[Constitutional AI- Harmlessness from AI Feedback#^rlaif|RLAIF]] makes the judge the model itself, which is the failure mode taken to its limit. WebWorld goes the other direction: hard-code the judge, learn only the proposer.
- **Refusals are kept in the loop but out of the export.** The typed rejection ("stale replay", "visual-only") feeds the next round's critique. This is a nice design pattern: exploration signal and supervision signal are different objects with different admission bars.
- **Limitations the authors own.** Single-file HTML only — multi-file or framework-heavy projects add dependency replay and scaffolding that the runtime does not handle. Genuinely perceptual defects (aesthetic taste, brand identity, accessibility nuance) have *no* executable certificate and fall back to the noisy VLM proxy. The verifier kills false positives but does not make critiques correct: a wrong critique still burns repair budget, it just does not poison the data.
- **Caveats for anyone reproducing.** Every number is a single seed-42 run, with no variance reported anywhere. The ablation deltas of 1.5 vs 3.5 points are probably not separable at that evidence level; the 0.4-vs-4.8 gap almost certainly is. The comparison against Kimi-K2.6 and GPT-5.4 is benchmark context, not a controlled experiment — different training data, different everything.
- **Practical takeaway if you build this.** The verification pass costs real compute (interaction + critique + patch + re-execution per round, 10 rounds, ~60K candidates, 40% of which get refused). The depth result says you can amortise that: filter aggressively for deep trajectories and train on ~15% of the pool. Noisy positives are far more expensive than refusals when the output is SFT data.
- **Open question.** Everything here is SFT on certified transitions. The certificate is a hard binary reward signal sitting right there — why not use it as an RL reward directly? The paper never says. Presumably because the browser round-trip is too slow for on-policy rollouts, but that is a guess.

## Links

Related: [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Agentic Game Development as a Verifiable Trajectory Data Engine for Scaling World Models]] · [[AgentMercury- Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale]] · [[On-policy Distillation with Verifiable Reward]] · [[Training language models to follow instructions with human feedback]] · [[Constitutional AI- Harmlessness from AI Feedback]] · [[Shortcut Learning in Deep Neural Networks]] · [[What Makes Good Agentic Data- An ACE Lens on Data Generation for LLM Agents]] · [[SWE Refactor Bench- Can Coding Agents Complete a Long-Horizon, Whole-Repository Stack Migration]] · [[Best Practice Critic Optimization]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Towards Quantifying Benchmark Optimization in ASR Models]] · [[The Bitter Lesson (essay)]]

New topics worth writing: Execution-grounded verification for LLM training data, SWE-bench and test-suite-as-oracle, Self-Refine and self-repair loops, Reflexion and verbal feedback agents, HTMLBench-400, Playwright/headless browser automation for ML pipelines, Data curation by trajectory depth, Typed rejection taxonomies in agent loops
