---
title: "Chain-of-Experience for Continual LLM Improvement"
authors: ["Haoqin Tu", "Yunhao Fang", "Yizhong Wang", "Cihang Xie", "Shen Yan"]
year: 2026
arxiv: "2608.18027"
url: https://arxiv.org/abs/2608.18027
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, llm, theory]
---
## The Core Idea

A deployed language model is frozen. Every question it answers is an isolated event. It solves a problem, forgets, and starts the next one from zero. Humans do not work this way — we try, we see what went wrong, we try again with that memory in hand.

**Chain-of-Experience (CoE)** is the setting where the model keeps its whole solving history in context and keeps re-attempting the *same* problem, with feedback after each attempt. Not one shot. Twenty shots, each one seeing all previous attempts and all previous feedback.

Formally, a normal LLM samples $a \sim P(a \mid Q)$. CoE turns this into a sequential process:

$$a_t \sim P\big(a_t \mid Q,\ e_0, e_1, \ldots, e_{t-1}\big), \qquad e_i = (a_i, f_i)$$

where $e_i$ is the $i$-th **experience** — the attempt $a_i$ paired with the feedback $f_i$ the environment gave for it. Feedback is drawn from an environment distribution $f_i \sim P'(F \mid Q, a_i)$, which can be a Python interpreter, a critic model, or a ground-truth checker.

> [!NOTE] Chain-of-Experience
> Iterative test-time problem solving where the model's *entire* history of attempts and feedback stays in context, forming a growing conditioning signal. No weights are updated — the "learning" is contextual. ^chain-of-experience

Why this did not exist as a clean idea before: prior work had pieces of it. Self-Refine and Reflexion do self-critique loops. Self-Debug adds execution signals. Tree-of-Thoughts and majority voting search in parallel — but they collapse everything into a single answer and throw the rest away. Dynamic CheatSheet and ACE build a memory *across different problems*. Nobody had systematically asked: what if you just keep the whole raw trail for the *same* problem, and what kind of feedback actually matters?

What it unlocks, concretely: with only self-generated feedback, average accuracy across six benchmarks goes from 66.8% (no feedback) to **71.0%**, and with the best feedback channel to **79.3%** — while costing *19% less* in API dollars than the no-feedback loop. The cheaper part is the surprise. Feedback makes models stop rambling.

## The Methodology

No training. This is entirely a prompting / interaction protocol.

**The loop.** Model $\mathcal{M}$ sees question $Q$ plus all prior $(a_i, f_i)$ pairs, emits attempt $a_t$, environment $\mathcal{E}$ returns $f_t$, repeat. Default is 20 iterations. Score reported is best accuracy reached within those 20.

**The four feedback types** (this is the actual axis of study):

1. **None.** $f_i = \varnothing$. The model sees only its own past attempts. Any improvement must come from self-reflection on the raw trail.
2. **Execution feedback.** For code: run the candidate, hand back stack traces, error messages, public test-case outcomes. $f_i = \mathcal{E}(Q, a_i)$. A weaker "binary executor" variant returns only pass/fail.
3. **Model feedback (self-feedback).** An auxiliary LM — usually the *same* model — acts as critic: $f_i = \mathcal{M}_{\text{fb}}(Q, a_i)$, returning free-text critique. This is the realistic, deployable one.
4. **Correctness feedback.** $f_i = \mathbf{1}\{a_i \text{ is correct}\} \in \{0,1\}$ from an oracle. Unrealistic in production; included as an upper bound.

**Models.** Eight reasoning LLMs: GPT-5, GPT-5-mini, o3, o3-mini, o4-mini, Gemini-2.5 Pro, Claude 4.5 Sonnet. Temperature 1.0 for OpenAI, 0.2 for Gemini/Claude. Three runs each, mean ± std reported.

**Benchmarks.** AIME 2025 (30 math problems), OmniMath (200 sampled olympiad problems), LiveCodeBench V6 (175), LiveBench-Code (128), GPQA Diamond (198 PhD-level MCQ), EvaLearn (648, explicitly a learning-ability benchmark). Grading: exact match for AIME/GPQA, LLM-as-judge for OmniMath, Python interpreter for code.

**Baselines** — two families:
- *Built-in reasoning depth*: OpenAI `reasoning_effort` low/high; Claude thinking disabled vs. 10k token budget.
- *Cross-task experience*: few-shot [[In Context Learning|ICL]] retrieving the $k$ most similar past solved problems via `text-embedding-3-large`; **Dynamic CheatSheet (DC)**, which distils reusable strategies into a persistent cheatsheet; **ACE (Agentic Context Engineering)**, which maintains an evolving "playbook" via generate→reflect→curate. All swept over $k \in \{1,5,8,12,15,20\}$.

**Improving-capability metric.** To compare models with different starting points, they normalise by headroom:

$$\Delta_{\mathcal{M}} = \frac{S_{\max} - S_{\text{base}}}{1 - S_{\text{base}}}$$

$S_{\text{base}}$ is zero-shot accuracy, $S_{\max}$ is peak under CoE with self-feedback. This asks "what fraction of the remaining gap did you close?"

## Ablation Studies and Experiments

**Main table (averaged best-of-20 across all eight models):**

| Method | AIME 25 | LCB (V6) | LiveBench-Code | OmniMath | GPQA-D | EvaLearn |
|---|---|---|---|---|---|---|
| ICL | 71.8 | 62.5 | 65.5 | 53.1 | 78.5 | 41.0 |
| ACE | 72.0 | 66.9 | 69.4 | 50.3 | 76.6 | 42.5 |
| DC | 73.3 | 63.6 | 68.6 | 48.6 | 79.6 | 42.7 |
| w/o feedback | 77.8 | 72.6 | 60.2 | 65.2 | 80.0 | 44.9 |
| reasoning-high | 69.1 | 70.6 | 55.5 | 61.8 | 76.2 | 39.6 |
| reasoning-low | 60.5 | 61.0 | 55.4 | 50.6 | 72.9 | 29.3 |
| **Self feedback** | **82.2** | **75.7** | **69.9** | **67.5** | **81.0** | **51.7** |
| Correctness/Executor | 89.1 | 74.5 | 75.8 | 79.6 | 99.5 | 57.1 |

Averages: ICL 62.1%, ACE 64.0%, DC 62.7%, no-feedback 66.8%, self 71.0%, best-feedback 79.3%.

**The loudest negative result: the cross-task memory methods lose to doing nothing.** ICL, ACE and DC all score *below* the plain no-feedback CoE baseline. These are widely-cited "test-time learning" methods and under modern reasoning models they do not scale. Compare with [[On the Difficulty of Evaluating Baselines]] and [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] — same flavour of finding.

**Cranking up built-in reasoning effort is also worse than iterating.** `reasoning-high` averages below the no-feedback loop on every benchmark. More thinking inside one attempt < more attempts with memory.

**Cost.** Self-feedback on coding gets 73.4% vs 75.0% for executor feedback, at 13.4% fewer API calls ($70.7 vs $81.6), and 20% cheaper than *no-feedback* CoE. On AIME 2025 self-feedback costs $4.6 vs $8.8 for no-feedback — a 47.3% reduction — while scoring 4.4 points higher. Token-level (Table 5): on AIME, correctness feedback = 84.6% at 108K tokens; no-feedback = 74.1% at 106K tokens. Same budget, ten points better. DC uses only 11K tokens but lands at 74.7%.

**Base ability predicts learning ability.** Pearson $r$ between zero-shot score and $\Delta_{\mathcal{M}}$: LiveBench-Code 0.97, LiveCodeBench 0.83, AIME 0.33, OmniMath 0.24. Average across five benchmarks $\approx 0.50$. Stronger models digest feedback better.

**Spurious feedback.** Feed the model a constant lie — always "correct" or always "incorrect". Damage averages 7.6% on AIME and 2.6% on GPQA. GPT-5-mini loses only 2.5 / 0.6 points; o4-mini loses 12.8 / 4.6. Adding **Selective Majority Voting** (SelMV-$n$: majority-vote the first $n$ valid attempts) recovers 1.2 points on AIME and 2.3 on GPQA. Weirdest result: on GPQA, SelMV under always-"incorrect" feedback (80.3%) *beats* honest self-feedback (79.4%). Constant "incorrect" forces re-derivation; constant "correct" breeds overconfidence and the model stops checking.

**Dual feedback** (Claude 4.5 Sonnet, model + correctness/executor combined):

| Setting | AIME 25 | LiveBench-Code | OmniMath |
|---|---|---|---|
| Dual | **76.7** (best @ R19) | **81.2** (R15) | 73.5 (R17) |
| Correctness/Executor | 70.0 (R13) | 78.1 (R15) | **74.5** (R17) |
| Model only | 60.0 (R6) | 57.8 (R17) | 50.5 (R9) |
| Model + DC | 50.0 | 51.6 | 46.0 |
| Model + SimpleMem | 56.7 | 54.7 | 49.5 |

Two things here. Dual helps a lot on AIME and LiveBench-Code, but on hard OmniMath it is *worse* than correctness alone — when the primary signal is already strong, extra critique is noise. And **compressing the experience trail hurts**: bolting DC or SimpleMem onto CoE *within the same task* loses 3–10 points versus keeping the raw full history. Aggressive summarisation throws away intermediate reasoning that the model actually reuses.

**Where do the gains come from?** They took 6,630 incorrect→correct flips and had GPT-5 attribute each to one of four causes (validated against two human annotators, overall Cohen's $\kappa = 0.768$, "substantial"). 47.7% of improvements are genuinely feedback-driven. In coding, 30.0% come from *specification recall* — the model just re-reading the output format requirement. Model-generated feedback produces a higher feedback-attributed share than external sources (58.7% vs 41.1%), suggesting self-critique is more contextually aligned than a foreign signal.

**Diminishing returns are sharp.** Extending from 20 to 50 iterations: on AIME the first 20 iterations give 16.7% average gain, iterations 20–50 give 2.2%. OmniMath: 21.2% then 3.5%. Almost all the value is in the first few rounds — note the "Best R" column above, where model-only feedback peaks at round 6–9.

**Feedback strength.** Cross-model feedback pairing GPT-5 and GPT-5-mini: on AIME, GPT-5-mini reaches 94.4% with GPT-5's feedback, *beating* the 93.0% from oracle correctness feedback. But on harder OmniMath, GPT-5-mini (62.5% zero-shot) cannot produce useful feedback for GPT-5 — GPT-5 gets 82.8% with correctness feedback but only 74.5% with mini's critique. **Model-as-judge is only competitive when the judge is strong on that specific task.**

**Where it fails outright.** On BrowseComp-Plus (deep-research browsing, 200 sampled), self-feedback makes most models *worse* than no feedback. The task needs external knowledge the model does not have. Reflecting harder on a fact you never knew does not conjure it.

## Worth Remembering

- **The headline practical takeaway is the cost, not the accuracy.** Self-feedback CoE is simultaneously more accurate *and* cheaper than the same loop without feedback. Feedback makes the model terse and directed instead of verbose and flailing. On AIME that is a 47% cost cut.
- **Gemini-2.5 Pro is the exception** on the cost–accuracy frontier: its self-feedback is verbose and gains little. The authors read this as weak self-evaluation ability. A reminder that "self-critique" is a real capability that varies by model, not a free property of LLMs.
- **Best round matters for deployment.** Model-only feedback peaks around round 6–9; correctness/executor feedback pushes the peak to round 13–19. Richer signal means slower, longer improvement. If you budget 5 rounds, self-feedback is the right choice.
- **Authors' admitted limitations.** (1) Only math / code / knowledge — no long-horizon agentic tasks like SWE-bench or $\tau$-bench, where the paradigm *should* extend but is untested. (2) **No weight updates.** This is contextual reuse, not learning. Close the tab and everything is gone. They flag internalising experience into parameters as the obvious next step — connects to [[Inject, Align, Recover- Staged Post-Training for Retrieval-Free Document Knowledge Internalization]].
- **Peak-of-20 is a generous metric.** "Best accuracy over 20 iterations" is not the same as "final answer after 20 iterations" — you need an oracle or a selector to know which round was best. SelMV is their answer to this, and it is the honest deployable version.
- The correctness-feedback numbers (99.5% on GPQA) are near-degenerate — with an oracle telling you yes/no on a 4-way multiple choice for 20 rounds, you are basically brute-forcing. Read that column as a ceiling, not a method.
- Open question worth chasing: does the gain survive if the trail is truncated to a sliding window? They showed *summarisation* hurts, but never tested plain truncation, which is what any real context-limited system would do.
- Conceptually this sits between [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] (reasoning inside one pass) and [[Proximal Policy Optimization Algorithms|RL post-training]] (reasoning baked into weights). It is RL's outer loop run entirely in the context window, with the environment reward read as text. Very much in the spirit of Silver & Sutton's "era of experience" and of [[The Bitter Lesson (essay)]].

## Links

Related: [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[In Context Learning]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Training language models to follow instructions with human feedback]] · [[Proximal Policy Optimization Algorithms]] · [[The Bitter Lesson (essay)]] · [[Mastering the game of Go with deep neural networks (AlphaGo)]] · [[Scaling Laws for Neural Language Models]] · [[On the Difficulty of Evaluating Baselines]] · [[Inject, Align, Recover- Staged Post-Training for Retrieval-Free Document Knowledge Internalization]] · [[Foundation Models]] · [[Markov Decision Process]]

New topics worth writing: Self-Refine, Reflexion, Tree-of-Thoughts, Self-Consistency / majority voting, Dynamic CheatSheet, Agentic Context Engineering, LLM-as-a-judge, process reward models, test-time compute scaling, Cohen's Kappa, EvaLearn benchmark, GPQA Diamond, LiveCodeBench
