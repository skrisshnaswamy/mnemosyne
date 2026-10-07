---
title: "Capable yet Parsimonious: Extracting and Characterizing Hidden Chain-of-Thought in Frontier Models"
authors: ["Xiaoyu Luo", "Tao Ren", "Wenrui Yu", "Xiao Li", "Qiongxiu Li", "Johannes Bjerva"]
year: 2026
arxiv: "2609.26637"
url: https://arxiv.org/abs/2609.26637
priority: Good-To-Read
read_on: 2026-09-28
tags: [paper, llm, vision]
---
## The Core Idea

Frontier labs hide the raw chain of thought. You get a final answer, maybe a sanitised summary, and a token count. So you can measure *what* a model solves but not *how*, and you cannot tell a correct answer that came from sound reasoning from one that came from a lucky guess.

The trick here is embarrassingly simple. Register a fake tool over the normal API — one function, one argument, a free-form string described as "scratchpad for working through the problem". Use the API's `tool_choice` control to **force** the model to call that tool first. Whatever it writes into the string argument is now in your logs. Reply with a content-free `"Received"`, hand control back to automatic tool choice, and let the model keep calling the scratchpad until it answers.

Crucially, native reasoning is turned **off** while this happens. So the reasoning-effort dial that providers expose does not actually stop reasoning-like text from coming out — it just stops it coming out of the field they chose to hide. The reasoning and its designated output channel are not welded together.

> [!NOTE] Forced-Reasoning protocol
> Force a single custom "scratchpad" tool on the first API call, replay its arguments back into the conversation, then release tool choice. The forced part is only the *selection* of the tool; the content is whatever the model chooses to write while genuinely solving the task. ^forced-reasoning

Why this matters beyond the security angle: it gives you a **measuring instrument** for closed models. With traces in hand, the authors compare four frontier models on length, redundancy, step types, and branching structure. The headline finding is that the most efficient model (called GPT-6 Astra in the paper) does not perform *different* reasoning operations — it performs the same ones and simply writes far less of them down. It skips elementary arithmetic expansions, uses recalled facts without restating them, and walks a near-straight path to the answer instead of branching.

And that has a consequence nobody was measuring: **compressed traces are worse teaching material for weak students.** Hand Astra's trace to a small model as context and the small model sometimes gets the question wrong *even though the correct answer is literally written in the trace it was given*. Hand the same trace to a strong model and almost no accuracy is lost. The value of a reasoning trace is relative to whoever reads it.

## The Methodology

**The tool.** `forced-reasoning(reasoning: string)`. Schema requires exactly that one string, `additionalProperties: false`. Description and parameter description are identical: *"Scratchpad for working through the problem."* The tool computes nothing. Its only job is to make text visible and keep it in the message history.

The loop, in plain pseudocode:

```
messages = [system, user]
reasoning = {effort: "none"}
response = chat(messages, tools=[forced_reasoning], tool_choice="forced-reasoning")
while response has tool calls:
    record json(call.arguments)["reasoning"]
    messages += [assistant_tool_call, tool_result("Received")]
    response = chat(messages, tools=..., tool_choice="auto")
return response.visible_text
```

No demonstrations, no shadow model, no prompt search. Standard API features only. Everything ran through OpenRouter.

**Three conditions compared.** `None` (native reasoning off, no tool), `Native` (native reasoning at high effort), `Forced` (the protocol). For Astra, which cannot have native reasoning disabled, they used the *lowest* available effort instead.

**Benchmarks.**
- MATH: 80 competition problems — 47 APEX Shortlist + all 33 HMMT Feb 2026, from MathArena.
- LiveCodeBench: 100 hard problems (all 80 hard in v6, plus 20 sampled from v5, seed 2026).
- HLE (Humanity's Last Exam): 100 problems, no images, no Math category, balanced across seven subject buckets, seed 2026.

**Validating the traces on open models first.** The obvious worry: maybe the scratchpad is a post-hoc story, not the actual reasoning. So they ran the same protocol on DeepSeek-V4-Flash and GLM-5.2, where the real CoT *is* visible, and checked three levels at once.

- **Performance.** DeepSeek: 30.5% (no reasoning) → **73.3% forced** vs 70.0% native. GLM-5.2: 21.4% → **84.3% forced** vs 89.9% native.
- **Lexical.** Compare a forced trace against a native trace (N–F) and use two independent native traces (N–N) as the ceiling. DeepSeek ROUGE-1: 0.613 (N–N) vs 0.538 (N–F). GLM: 0.588 vs 0.465. Forced traces are also similar in length — median forced/native length ratio 1.42× and 1.39×.
- **Structural.** Segment each trace into units and label each with one of seven Schoenfeld-style episode types — `Read`, `Analyze`, `Plan`, `Implement`, `Explore`, `Verify`, `Monitor` (plus `Other`). Native and forced conditions show the same repertoire in broadly similar proportions.

The argument is that agreement at any one level could be luck; agreement at all three is hard to fake without actually doing the reasoning.

**Closed-model numbers** (accuracy %, `N/A` = Astra cannot disable reasoning):

| Benchmark | Method | Opus 4.8 | Sonnet 5 | GPT-5.6 Sol | GPT-6 Astra |
|---|---|---|---|---|---|
| MATH | None | 72.5 | 42.5 | 25.0 | N/A |
| | Native | 86.3 | 77.5 | 97.5 | 97.5 |
| | **Forced** | 85.0 | 81.3 | 91.3 | 93.8 |
| HLE | None | 24.0 | 17.0 | 13.0 | N/A |
| | Native | 33.0 | 21.0 | 25.0 | 35.0 |
| | **Forced** | 27.0 | 22.0 | 23.0 | 32.0 |
| LCB | None | 58.0 | 52.0 | 47.0 | N/A |
| | Native | 83.0 | 77.0 | 90.0 | 92.0 |
| | **Forced** | 82.0 | 77.0 | 90.0 | 89.0 |

Forced lands close to native everywhere and crushes the no-reasoning baseline. That is the behavioural licence to treat these traces as a proxy.

**The four measurements on frontier traces.**

1. **Length and redundancy.** Mean output tokens (completion minus native reasoning tokens), plus lossless zlib compression ratio at level 9 on the scratchpad text only. A *higher* zlib ratio means *harder to compress*, i.e. less repeated boilerplate.

2. **Local granularity.** Hand-matched excerpts where all models do the same check on the same problem, read side by side.

3. **Episode composition.** The seven-category taxonomy again, annotated by Claude Opus 5 subagents on deterministically pre-segmented units (paragraph breaks and sentence delimiters; code fences and display math kept atomic). Model identity, condition, benchmark and correctness were hidden from the annotator.

4. **Reasoning trees.** Using LCoT2Tree, adapted. A "thought" is a run of consecutive units, with a new thought starting at each `Explore`, `Verify` or `Monitor`. A judge model (DeepSeek-V4-Flash, reasoning off) extracts an ordered *sketch* of the solution, then for each thought separates the steps it **works on** ($W_i$) from steps it merely **refers to** ($R_i$). That split is the key adaptation — citing an earlier result no longer counts as returning to that step. Tree metrics: width $p$ = max nodes at one sketch level, depth $q$ = furthest sketch step reached, size $N$ = total non-root nodes. Revisiting a step makes a *new* node, so backtracking is preserved rather than merged.

**Transfer experiment.** Take a donor model's extracted trace, strip the final answer, hand it to a recipient model as prior context. Recipient answers in one call with native reasoning off and no tools. Recipients were deliberately chosen to span a wide range of standalone Native-high accuracy on MATH, including Claude Haiku 4.5, GPT-5.4 Nano and DeepSeek-V4-Flash.

## Ablation Studies and Experiments

**Is the tool doing anything, or is it just the prompt wording?** This is the cleanest ablation in the paper, and the answer is that the tool channel matters a lot. GPT-5.6 Sol on MATH:

| Setting | Accuracy | Mean tokens |
|---|---|---|
| no-tool, default prompt | 25.0 | 442 |
| no-tool, "reason step by step in your response" | 36.3 | 906 |
| no-tool, maximal-deliberation wording | 36.3 | 670 |
| Native low | 78.8 | 1,599 |
| Native medium | 92.5 | 3,403 |
| Native high | 97.5 | 5,526 |
| **Tool, forced, default description** | **81.3** | 1,940 |
| **Tool, forced, maximal description** | **91.3** | 3,481 |

Simply *asking* for step-by-step reasoning in the visible response gets 36.3%. Putting the exact same maximal-deliberation words into the response prompt instead of the tool description gets you nothing extra — still 36.3%. The forced tool call gets 81–91%. On HMMT alone the gap is 60.6% (asking nicely) vs 93.9% (forced tool with strong wording). The authors also note the no-tool outputs read like tidy Markdown answers, not chain of thought.

Interpretation: the scratchpad-shaped channel is not just a permission slip, it is a different *mode*. `Forced` also does not map onto a single fixed native effort level — default wording sits near native-low, maximal wording near native-medium.

**The wording ablation reverses sign between models.** This is the most interesting negative result. Strong "write out absolutely everything" wording *increased* Sol's externalisation. On Astra it did the opposite. Astra on MATH, all at low native effort:

| Tool description | Accuracy | Output tokens | Native reasoning tokens | % runs with zero native tokens |
|---|---|---|---|---|
| default | 90.0 | 849 | 887 | 27.5 |
| maximal deliberation | 97.5 | 594 | 1,386 | 11.25 |
| "think here, don't work it out elsewhere" | 95.0 | 1,755 | 13.65 | **93.75** |

Demanding an exhaustive account made Astra write *less* in the tool and keep more in its hidden channel. Inviting it to continue its train of thought in the scratchpad moved almost all of it across — 93.75% of runs reported literally zero native reasoning tokens. Practical lesson: for stronger models, redirect the reasoning rather than demand a report of it.

**Turning native reasoning back up breaks clean extraction.** With native effort at high/xhigh on 47 APEX problems, accuracy is near-perfect and scratchpads get long (Sol 15.0k → 19.7k tokens; Astra 5.8k → 12.0k). But *whole-interaction* zero-native (no native tokens on any call, including the final answer) collapses for Astra: 18/47 at high, 7/47 at xhigh. The leak is mostly on the final-answer call. So you can get more visible reasoning or cleaner attribution, not both.

**Length and compressibility.** Mean forced output tokens vs native reasoning tokens:

| | MATH native / forced | HLE native / forced | LCB native / forced |
|---|---|---|---|
| GPT-6 Astra | 2,903 / 1,752 | 2,313 / 652 | 1,962 / 939 |
| GPT-5.6 Sol | 5,526 / 3,925 | 3,293 / 2,214 | 4,058 / 2,665 |
| Opus 4.8 | 24,087 / 16,283 | 5,385 / 3,627 | 14,849 / 12,696 |
| Sonnet 5 | 27,846 / 27,851 | 10,604 / 10,705 | 25,128 / 26,416 |

Astra is roughly an order of magnitude shorter than the Claude models while scoring the same or better. It also has the **highest zlib ratio on all three benchmarks** — shortest *and* least compressible. Short traces are not short because they repeat themselves less; there is simply less padding in them at all.

**What gets skipped, concretely.** Same HMMT problem, same division check:

- Astra: `verify 420261/101 = 4161 yes.`
- Sol: `Verify division: 420261/101: 101*4161 = 416100+4161=420261 exactly.`
- Opus: `101*4000=404000, 101*161=16261, sum=420261. Yes. Good.`

And a valence-electron count on an HLE chemistry item: Astra writes `48+18+36=102` without ever stating that C, H and O contribute 4, 1 and 6. Sol and Opus state the per-element values first. The elementary step happens; it just is not verbalised.

**Episode composition says the repertoire is the same.** All four models are dominated by `Analyze` and `Implement`, with non-trivial `Read`, `Plan`, `Explore`, `Verify`, `Monitor`. Mean characters *per episode unit* are comparable across Astra, Sol and Opus (only Sonnet writes long individual units). But total characters *per trace per category* differ several-fold. So Astra is not compressing sentences — **it is surfacing fewer steps.** The coarse temporal shape survives too: `Read` clusters at the start, `Verify` at the end, `Implement` in the middle, in every model.

**Reasoning trees say the difference is branching.** MATH, median [25th, 75th]:

| Model | Width $p$ | Depth $q$ | Nodes $N$ |
|---|---|---|---|
| GPT-6 Astra | **5.0** [3.0, 9.0] | 11.0 [8.8, 15.0] | **27.0** [17.8, 44.0] |
| GPT-5.6 Sol | 12.0 [5.0, 19.3] | 11.0 [10.0, 13.0] | 60.0 [29.8, 90.3] |
| Claude Opus 4.8 | 22.0 [5.0, 44.0] | 12.0 [9.0, 14.0] | 96.0 [28.5, 220.3] |
| Claude Sonnet 5 | 28.5 [9.0, 49.3] | 13.0 [10.0, 15.0] | 136.0 [44.8, 234.8] |

Same depth, five times narrower, five times fewer nodes than Sonnet. Astra reaches a comparably deep solution with far less lateral search, revisiting and dead-ending. It picks a good trajectory earlier. (Caveat from the appendix: by *means* rather than medians Astra's depth is actually higher — 14.9 vs 11.2–12.4 — so "comparable depth" is a median claim.)

**Transfer: the result that matters most.** Traces from Sol and Opus transfer to every recipient with little loss. Astra's traces transfer almost losslessly to strong recipients and *partially* to weak ones. The biggest shortfalls: Claude Haiku 4.5 and GPT-5.4 Nano.

The damning sub-analysis: Astra explicitly states the correct answer in its scratchpad on **73/80** MATH problems (91.25%). Among the recipient's *wrong* answers, the fraction where the correct answer was sitting right there in the input:

| Recipient | Wrong answers with correct answer exposed in input |
|---|---|
| Haiku 4.5 | 15/18 (83.3%) |
| GPT-5.4 Nano | 11/16 (68.8%) |
| DeepSeek-V4-Flash | 11/15 (73.3%) |

One example: Astra's trace ends `conditional 24/60 = 2/5`. Nano answers $2/3$; DeepSeek answers $4/7$. Another: Astra computes `2+6+3+6+14+18+10 = 59`; two recipients answer $0$ and one answers $129$. On a Hamiltonian-path count where Astra writes `total per start 1023 = 2^10-1 times 20 = 20460`, Nano keeps only one of the two cases and outputs $20\cdot 511 = 10220$.

So compression does not merely make a trace harder to read — below some reader capability it puts the trace **out of reach entirely**. The shortfall is relative, not absolute: weak recipients still improve a lot over unaided performance.

**Where the protocol fails outright.** Claude Opus 5, Fable 5 and Fable 5.1: **0% extraction success**. Opus 5 refused reasoning-tool use across every description and parameter name tried, while still happily producing visible reasoning when asked directly on the same benign maths question. Fable 5/5.1 returned explicit errors when thinking was disabled and rejected required/named tool choice even at low reasoning effort. The authors are careful: this is a failure boundary, and they cannot tell whether the block lives in the model or in provider-side request handling.

**Prefix echo (Appendix M).** Injecting the first 1% of a donor's scratchpad into a recipient's open thinking channel, then scoring only the newly generated continuation against the donor's remaining text. Opus 4.8 → Kimi-K3 shows a consistent rise on all three metrics (ROUGE-L $0.204 \to 0.243$; 5-gram Jaccard $0.0077 \to 0.0213$; ROUGE-L up on 27/33 HMMT problems). Opus → DeepSeek and Sol → either recipient show **no consistent improvement** — two of four pairs get slightly worse. So the echo effect is real but strongly model-pair dependent.

## Worth Remembering

**The honest limitation, stated by the authors.** For closed models they have no ground-truth CoT, so "similar to native" and "useful downstream" cannot establish that the scratchpad text *is* the model's internal computation. It may be a byproduct of forcing intermediate text through a different channel. The open-model validation is the only place identity is even partially testable.

**Practical scope.** You need an endpoint that supports custom tools, forced selection of a *named* tool, and ideally disabling native reasoning. Nothing else. No model internals, no shadow model, no gradient access. That is what makes this both cheap to reproduce and awkward for providers to patch without breaking legitimate planning tools.

**The distillation implication is the part I'd carry forward.** This offers a mechanism for the capacity gap in [[Distillation]] — the long-observed fact that the strongest teacher does not always produce the best student. The strongest models write the *most compressed* traces, and a compressed trace omits exactly the routine steps a weak student cannot fill in itself. Trace **explicitness** is a variable separate from teacher **strength**. As frontier models get optimised harder for token efficiency, their traces get better for humans paying per token and worse as supervision for small models.

They are explicit that they did **not** test this. The experiment measures in-context reuse, not fine-tuning, and donor traces differ in correctness, content and style, so compression is not isolated as a cause. They state the falsifiable prediction: *expanding a compressed trace into its implicit intermediate steps should restore most of its usefulness to weak recipients while barely affecting strong ones.* That is a cheap experiment and nobody has run it.

**Measurement caveats on the tree analysis.** Width, depth and size all depend on the segmentation rule and the judge's step assignment. A smaller tree means fewer *represented* nodes — it does not prove fewer internal computations or a shorter necessary proof. The whole analysis measures externalised reasoning, which is the point, but it is easy to over-read as a claim about internal search.

**A judge is in the loop twice.** Episode labels come from Claude Opus 5 subagents; sketch-step assignment comes from DeepSeek-V4-Flash. The prompt-development pilot measured 90% agreement with manual review on 200 units (up from 83% with the original minimal prompt). Model identity and correctness were blinded to the annotator, which is the right control, but every structural conclusion inherits that 10% label noise.

**Reasoning-effort dials are not a confidentiality boundary.** Astra's Forced runs reported **zero** native reasoning tokens from the provider while producing coherent multi-step reasoning in a tool argument. If your safety or IP story depends on the reasoning channel being hidden, hiding one field is not enough. The authors reported this to OpenAI and Anthropic security teams with reproduction code before publishing, and suggest mitigations: restrict free-form string tool arguments when native reasoning is off, limit forced tool selection, or classify tool arguments for deliberative content — all with the obvious false-positive cost on genuine scratchpad tools.

**Open question I'd want answered.** The transfer shortfall was *not* strictly ordered by recipient capability — GPT-5.4 Nano and DeepSeek-V4-Flash have similar standalone accuracy but recover different amounts from Astra's traces. So "capability" is the wrong single variable. Scale, tokeniser, and training recipe presumably all matter, and nobody has separated them.

## Links

Related: [[Chain of Thought]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Chain-of-Thought Faithfulness of Reasoning Models Varies with Where and How Preference Cues Are Delivered]] · [[Distillation]] · [[Distilling the Knowledge in a Neural Network]] · [[Test-Time Compute]] · [[Tool Use]] · [[Structured Output]] · [[Evals]] · [[Every Coin Has Two Sides- On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models]] · [[1% of Tokens Can Be Enough- On Gradient Estimation in On-Policy Distillation]] · [[Cost and Latency]] · [[Observability and Tracing]] · [[Prompt Injection]] · [[Guardrails]]

New topics worth writing: Schoenfeld episode theory for reasoning traces, LCoT2Tree reasoning-tree construction, capacity gap in knowledge distillation, reasoning-trace extraction attacks (REP, EchoCoT, Trace Inversion), overthinking and CoT redundancy, zlib compressibility as a text-redundancy probe, LLM-as-a-judge annotation protocols and blinding, reasoning-token accounting across API channels
