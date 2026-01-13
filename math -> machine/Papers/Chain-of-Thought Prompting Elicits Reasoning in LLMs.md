---
title: "Chain-of-Thought Prompting Elicits Reasoning in LLMs"
authors: ["Jason Wei", "Xuezhi Wang", "Dale Schuurmans", "Maarten Bosma", "Brian Ichter", "Fei Xia", "Ed Chi", "Quoc Le", "Denny Zhou"]
year: 2022
arxiv: "2201.11903"
url: https://arxiv.org/abs/2201.11903
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, llm, theory]
---
## The Core Idea

Take a big language model, show it eight worked examples in the prompt, and instead of writing `Q: ... A: 8`, write `Q: ... A: <the reasoning, step by step> The answer is 8`. That's the whole method. No fine-tuning, no gradient updates, no new architecture. On GSM8K (grade-school math word problems), PaLM 540B goes from **17.9% → 56.9%** solve rate. That beats the previous state of the art, which was a fine-tuned GPT-3 175B with a separately trained answer verifier (55%).

Why this did not exist before: two separate lines of work each had a wall in front of them.

1. **Rationale-based training.** People knew that making a model write out intermediate steps helps. But they got it by training on thousands of hand-written rationales. Expensive.
2. **Few-shot prompting** ([[Language Models are Few-Shot Learners (GPT-3)|GPT-3]]). Cheap, but it was known to be flat on reasoning tasks — Gopher showed scaling the model barely moved the needle on arithmetic.

The trick is that you only need *eight* rationales, written by hand in an afternoon, and you put them in the prompt instead of in the training set. That combines the cheapness of prompting with the benefit of intermediate steps.

The second, bigger finding is that this is an **emergent ability of scale**. Below roughly 10B parameters, chain of thought makes performance *worse* than plain prompting. Small models write fluent nonsense — grammatically fine reasoning chains with broken logic. The gains only appear around ~100B parameters. This means the standard way of measuring what a model can do (plain few-shot prompting) is only a *lower bound* on its capability. A 540B model already knew how to do multi-step arithmetic; nobody had asked it in the right format.

> [!NOTE] Chain-of-thought prompting
> Few-shot exemplars given as $\langle$input, chain of thought, output$\rangle$ triples rather than $\langle$input, output$\rangle$ pairs, where the chain of thought is a series of natural-language intermediate reasoning steps. The model then imitates that format at test time and generates its own steps before its answer. ^chain-of-thought

## The Methodology

There is no training loop. Everything is inference on frozen, off-the-shelf models.

**The prompt.** Eight hand-written exemplars, prepended to the test question. One of them, verbatim:

> Q: Olivia has \$23. She bought five bagels for \$3 each. How much money does she have left?
> A: Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8. The answer is 8.

The same eight exemplars were reused across four of the five math datasets — they were not tuned per dataset. AQuA (multiple choice algebra) got its own four exemplars. Prompts fit in 1024 tokens.

**Decoding.** Greedy, one sample. (Self-consistency — sample many chains, take the majority answer — came later and improves on this.)

**Answer extraction.** String-match on "The answer is X".

**Models.** Five families, deliberately spanning scale so the emergence curve is visible:
- GPT-3 / InstructGPT: 350M, 1.3B, 6.7B, 175B (`text-davinci-002`)
- LaMDA: 422M, 2B, 8B, 68B, 137B
- PaLM: 8B, 62B, 540B
- UL2 20B, Codex (`code-davinci-002`)

**Tasks.** Three families:
- *Arithmetic*: GSM8K, SVAMP, ASDiv, AQuA, MAWPS (which splits into SingleOp / SingleEq / AddSub / MultiArith by difficulty).
- *Commonsense*: CSQA, StrategyQA, BIG-bench Date Understanding, BIG-bench Sports Understanding, SayCan (map an instruction to a robot action sequence like `find(coke), pick(coke), find(trash), put(coke)`).
- *Symbolic*: last-letter concatenation ("Amy Brown" → "yn") and coin-flip state tracking. Both have an in-domain test set (2 items, same as the exemplars) and an out-of-domain one (3 and 4 items, longer than anything shown).

**Optional add-on: external calculator.** Because many chains had correct logic but botched arithmetic, they ran every generated equation through Python `eval` and propagated results forward by string matching. LaMDA 137B on GSM8K: 14.3 → 17.8. PaLM 540B on MAWPS: 93.3 → 93.5 (already saturated).

## Ablation Studies and Experiments

**Headline arithmetic numbers** (accuracy %, standard → chain of thought):

| Model | GSM8K | SVAMP | ASDiv | AQuA | MAWPS |
|---|---|---|---|---|---|
| LaMDA 137B | 6.5 → 14.3 | 29.5 → 37.5 | 40.1 → 46.6 | 25.5 → **20.6** | 43.2 → 57.9 |
| GPT-3 175B | 15.6 → 46.9 | 65.7 → 68.9 | 70.3 → 71.3 | 24.8 → 35.8 | 72.7 → 87.1 |
| Codex | 19.7 → 63.1 | 69.9 → 76.4 | 74.0 → 80.4 | 29.5 → 45.3 | 78.7 → 92.6 |
| PaLM 540B | 17.9 → 56.9 | 69.4 → 79.0 | 72.1 → 73.9 | 25.2 → 35.8 | 79.2 → 93.3 |

**The scale story, sharply.** PaLM on GSM8K: 8B gets 4.9 → **4.1** (CoT hurts), 62B gets 9.6 → 29.9, 540B gets 17.9 → 56.9. LaMDA 420M on MAWPS: 3.2 → **0.9**. Same for GPT 350M/1.3B/6.7B — chain of thought is *net negative* at every one of those scales.

**Gains scale with problem hardness.** MAWPS stratified (PaLM 540B): SingleOp 94.1 → 94.1 (nothing), AddSub 93.9 → **91.9** (worse), but MultiArith 42.2 → 94.7. One-step problems get nothing; multi-step problems get everything.

**The three ablations** (LaMDA 137B, GSM8K, standard = 6.5, CoT = 14.3). These are the important part of the paper because each kills a competing explanation:

| Variant | GSM8K | What hypothesis it tests |
|---|---|---|
| Equation only | **5.4** | "It just helps to write the equation" |
| Variable compute only (emit `…` dots equal in length to the equation) | 6.4 | "It just gets more forward passes / tokens of compute" |
| Reasoning *after* the answer | 6.1 | "It just activates relevant pretraining knowledge" |
| Chain of thought | 14.3 | — |

All three ablations land at baseline. So: it is not extra compute, it is not knowledge priming, and it is not the equation. It is specifically **sequential natural-language reasoning generated before the answer**. The "reasoning after answer" ablation is the cleanest — the tokens are identical, only the order changed, and the effect vanishes entirely. This is a token-level [[Auto-regressive models|autoregressive]] causality result: the answer token has to be able to attend to the reasoning tokens ([[Causal Attention|causal attention]] means order is everything).

Equation-only *does* help on shallow datasets (SVAMP 29.5 → 35.1, ASDiv 40.1 → 45.9) but not GSM8K. The authors' explanation, with an example: for *"Mike scores 4 points, then 25% more points"*, equation-only writes `(4 + 20 * 0.25) = 6` — it cannot compress the semantics into one line. Chain of thought lets it handle each clause separately and gets 9.

**Robustness.** Three different co-authors independently wrote the chains of thought. LaMDA 137B GSM8K: A 14.3, B 15.5, C 17.6, A-concise 11.1. Three sets of eight exemplars pulled at random from the GSM8K *training set* (written by crowdworkers with no ML background): 12.6, 12.7, 12.6. All well above 6.5. Also robust to exemplar order (small std) and to number of exemplars (1 through 8).

**Commonsense** (PaLM 540B, standard → CoT): StrategyQA 68.6 → 77.8 (beats prior SOTA 69.4), Sports 80.5 → 95.4 (beats an unaided sports enthusiast at 84%), Date 49.0 → 65.3, SayCan 80.8 → 91.7. CSQA 78.1 → 79.9 — basically nothing.

**Symbolic and length generalisation** (PaLM 540B). Last-letter concat, 2 words: 7.6 → 99.4. On 4 words, never seen in the prompt: 0.0 → **63.0**. Coin flip, 4 flips: 54.8 → 90.2. Standard prompting is at chance out of domain; chain of thought carries the *procedure* forward to longer inputs. But 62B on 4-word concat only gets 13.4, so length generalisation is itself scale-gated.

### What did not work

- **AQuA on LaMDA 137B: 25.5 → 20.6.** CoT actively hurt.
- **GPT-3 175B on CSQA: 79.5 → 73.5**, and StrategyQA 65.9 → 65.4. The *same prompts* that helped PaLM hurt GPT-3 on these two. Prompts do not transfer cleanly across model families — a real limitation the authors flag.
- **Reversing a list of five items.** Two co-authors could not write a chain-of-thought prompt that solved it at all, despite trying. A third could, and it worked perfectly. So "prompt engineering does not matter" is false; the arithmetic results just happen to be forgiving.
- **Annotator variance can be huge on some tasks.** Coin flip: Annotator A 99.6%, Annotator C 71.4%. Both beat the 50% baseline, but that is a 28-point spread from writing style alone.
- **More exemplars does not rescue standard prompting.** Going from 8 to 16 plain exemplars did not close the gap.

## Worth Remembering

**Manual error analysis is the most useful part of the appendix.** Of 50 LaMDA 137B chains on GSM8K with the *right* answer, 49 had genuinely correct logic — only one was right by luck. Of 50 with the *wrong* answer:

- 8% — arithmetic slip only (fixable by a calculator)
- 16% — symbol mapping error (wrote `15 x 30` when it meant `15 x 50`; words correct, numbers wrong)
- 22% — one reasoning step missing
- 54% — real semantic-understanding or incoherence failures

Scaling PaLM 62B → 540B fixed a large chunk of *all three* categories, especially semantic understanding and one-step-missing. So "why does scale help?" is genuinely multi-causal: semantic parsing, symbol manipulation, staying on topic, and raw arithmetic all improve at once.

**The correctness analysis has a hole the authors admit.** On free-response math, getting the right answer by accident is unlikely, so correct answer ≈ correct reasoning. On binary or multiple-choice tasks (all the commonsense benchmarks) the model can trivially be right for a wrong reason. Do not read commonsense accuracy as evidence the reasoning is faithful. Example from StrategyQA: *"Harry Potter is a fictional character. Thus, Harry Potter can do anything."* Fluent, confident, wrong.

**Limitations the authors state plainly.** (1) They do not claim the network is "reasoning" — chain of thought imitates a human thought process, that is all. (2) There is no guarantee the reasoning path is correct, and improving factuality is open. (3) Emergence at ~100B makes this expensive to serve. Inducing reasoning in small models is left to future work — which is what a whole literature on reasoning [[Distillation|distillation]] then went and did.

**Practical caveats if you want to use it.**
- Only worth it when three conditions hold: the task needs multiple steps, the model is big, and the scaling curve is flat. If your model already scores 90%+, expect nothing or a small regression.
- Bolt a calculator on. Almost free, and 8% of failures are pure arithmetic.
- Validate your prompt on your own model. Cross-model transfer is not guaranteed.
- Greedy decoding is the weakest version of this. Self-consistency (sample $k$ chains, majority-vote the final answer) came out one month later and is strictly better.

**Connections.** This is a clean instance of the [[The Bitter Lesson (essay)|bitter lesson]] applied to prompting — no task-specific machinery, no symbolic solver, just scale plus a format change. It also sharpens what [[In Context Learning|in-context learning]] means: the exemplars are not teaching the *task*, they are teaching an *output format*, and the format is what unlocks the capability. Related to [[Scaling Laws for Neural Language Models|scaling laws]] by contrast — scaling laws predict smooth loss curves, but chain of thought is discontinuous in scale and could not be extrapolated from small models.

**Open questions.** Could you generate the chains automatically instead of hand-writing them (the authors float this — STaR and zero-shot "Let's think step by step" both did it shortly after)? Why exactly does 8B fail on last-letter concatenation when the full procedure is spelled out in the prompt? Does chain of thought help on tasks that are not obviously step-by-step — translation, summarisation?

## Links

Related: [[Language Models are Few-Shot Learners (GPT-3)]] · [[In Context Learning]] · [[Scaling Laws for Neural Language Models]] · [[The Bitter Lesson (essay)]] · [[Attention Is All You Need]] · [[Causal Attention]] · [[Auto-regressive models]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[Training language models to follow instructions with human feedback]] · [[Distillation]] · [[Shortcut Learning in Deep Neural Networks]] · [[Foundation Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]]

New topics worth writing: Emergent abilities of large language models, Self-consistency decoding, Zero-shot chain of thought ("Let's think step by step"), STaR (bootstrapping rationales), GSM8K benchmark, Prompt engineering and prompt sensitivity, Faithfulness of model-generated explanations, Verifier models for math reasoning, Scratchpads for intermediate computation, Length generalisation
