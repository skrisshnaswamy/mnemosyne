---
title: "Sample Count Is Not Enough: Candidate-Generation Strategy Shapes the Energy and Performance of LLM Test-Time Scaling"
authors: ["Kashaniyan et al."]
year: 2026
arxiv: "2609.19499"
url: https://arxiv.org/abs/2609.19499
priority: Good-To-Read
read_on: 2026-09-27
tags: [paper, transformers, llm, rl]
---
## The Core Idea

When people talk about "test-time scaling" — spending more compute at inference to get better answers — they describe the budget with one number: $N$, the number of candidate answers sampled. Self-consistency generates $N$ reasoning traces and takes the majority answer. Best-of-$N$ scores $N$ traces and picks the winner.

The point of this paper is that $N$ is not a description of a workload. It says how many candidates exist, not **how they were run**.

Eight candidates can be produced as:

- one generation call with batch size 8, or
- eight generation calls with batch size 1, or
- anything in between.

Same prompt, same decoding settings, same voting rule, same accuracy in expectation. Wildly different cost. On A100s, eight serial single-candidate calls burn **4.64–4.86× the GPU energy** and have **5.77–6.12× the P95 latency** of one batched call of eight.

So the authors give the missing variable a name. Call the **generation schedule**

$$S = (b_1, \dots, b_C), \qquad \sum_{c=1}^{C} b_c = N$$

where $C$ is the number of sequential calls and $b_c$ is the batch size of call $c$. The notation $a \times b$ means $a$ calls of $b$ candidates each. $N$ fixes the sum; it says nothing about $C$.

> [!NOTE] Generation schedule
> The partition of a fixed candidate budget $N$ into $C$ sequential generation calls with batch sizes $b_1 \dots b_C$. Two runs with identical $N$ and identical accuracy can differ by ~5× in energy purely because they differ in $S$. ^generation-schedule

Why did this not exist before? Because the test-time-scaling literature and the serving literature barely talk to each other. Serving papers — [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)|vLLM]], ORCA, Sarathi-Serve — have known for years that batching dominates inference efficiency. Test-time-scaling papers report $N$ and accuracy and stop. The authors audit six representative papers (Self-Consistency, Adaptive-Consistency, Universal Self-Consistency, Large Language Monkeys, Scaling Test-Time Compute, Difficulty-Adaptive SC): **all six report $N$, none report calls per query, none report candidates per call, none report energy.**

What it unlocks is small and useful. Candidate generation is a scheduling problem, not a count. And it explains a reproducibility hole: two labs can publish the same method, same model, same dataset, same $N$, and have systems costs 5× apart, with nothing in either paper to reveal why.

A second, subtler lesson: **average power is a trap as an efficiency metric.** The serial schedule draws *less* power per instant (177.8 W → 139.8 W for Phi-3) while consuming far more total energy, because it keeps the GPU alive ~6× longer. Watts are not joules.

## The Methodology

Two models, both small: **Phi-3-mini-4k-instruct** and **Qwen2.5-1.5B-Instruct**. Plain Hugging Face `transformers` generation (2.4.1 / 4.57.6), not a serving engine — so no continuous batching, no paged KV. Each job reserves one GPU exclusively.

Sampling: temperature 1.0, top-$p$ 0.95. Max new tokens 512 on GSM8K, 64 on SciQ.

**Aggregation.** Plain plurality vote — no verifier, no reward model, no token-level confidence. Answers are extracted from GSM8K by trying, in order: the requested final-answer delimiter, `\boxed{}`, explicit answer statements, a trailing numeric expression, the last non-empty line. Then normalised. SciQ: match one of A–D, case-insensitive. Extraction failures stay in the vote pool and count as wrong if they win. Ties go to whichever answer appeared first among the candidates.

That tie rule matters more than it sounds — see the accuracy results.

**Token accounting**, to prove the schedules are not secretly generating different amounts of text. Input volume per query is trivially fixed at $N P_q$ ($P_q$ = prompt length) since every candidate reuses the same prompt. Generated volume is

$$T^{\text{logical}}_{\text{gen}}(q, S) = \sum_{c=1}^{C} \sum_{j=1}^{b_c} L_{q,c,j}$$

Measured spread across schedules: **0.8%** for Phi-3, **1.0%** for Qwen. So the cost differences are not a length artefact.

**The measurement boundary**, which is the part to copy if you ever do this yourself:

1. GPU synchronise.
2. Read NVML cumulative energy counter → $E_{\text{start}}$.
3. Run everything: prompt processing, prefill, decode, *all* generation calls, answer extraction, plurality vote.
4. GPU synchronise again.
5. Read NVML → $E_{\text{end}}$.

$$E_{\text{gross}} = E_{\text{NVML,end}} - E_{\text{NVML,start}}$$

Excluded: model loading, warm-up (two throwaway generations), token counting for reporting, final grading. **Idle power is not subtracted**, and this is whole-*device* energy, not whole-*node* — no CPU, no DRAM, no cooling, no PSU loss. Average power is derived as $E_{\text{gross}} / \text{latency}$, not measured directly.

**Five experiments.**

| Study | Workload | Design |
|---|---|---|
| Accuracy scaling | GSM8K, 500 prompts | one pool of 8 candidates; score $N \in \{1,2,3,4,8\}$ by taking prefixes |
| Batched scaling | GSM8K, 100×3 | batched $N \in \{1,2,4,8\}$ plus serial $N{=}8$ |
| Schedule sweep | GSM8K, 100×3 | fixed $N{=}8$: $1{\times}8$, $2{\times}4$, $4{\times}2$, $8{\times}1$ |
| Cross-node check | GSM8K, 100×3 | repeat the two endpoints on two more A100 nodes |
| Short-output check | SciQ, 500×3 | full $N{=}8$ sweep on V100 |

The accuracy study uses **prefix evaluation** — generate 8 once, then score $N{=}3$ using the first 3. That gives paired per-prompt comparisons across budgets, which is what the paired bootstrap needs. Those generations are used *only* for accuracy; systems numbers come from separate runs.

Hardware: primary schedule + cross-node = A100-SXM4 80 GB, 500 W cap, graphics clock pinned at 1275 MHz. SciQ = V100 PCIe 32 GB, 250 W cap, 1230 MHz. **Clocks are pinned**, which is the right call — otherwise boost behaviour confounds everything.

**Confound control worth noting.** All four schedules run inside the same A100 job. Two repetitions use the order $1{\times}8 \to 2{\times}4 \to 4{\times}2 \to 8{\times}1$; the third runs it reversed, to catch thermal drift and ordering effects. GSM8K test split shuffled with seed 42, first 100 prompts for systems work, prompt order frozen across every schedule, repetition, job and node.

**Statistics.** Systems values are mean ± SD over three repetitions. P95 latency is computed within a repetition then summarised across them. Accuracy intervals: 10,000 prompt-level bootstrap resamples, **paired** when comparing budgets. The fixed-$N$ ratios use a *hierarchical* paired bootstrap — resample repetitions first, then prompts within each chosen repetition, preserving the schedule pairing, and recompute P95 inside every bootstrap sample. That is the correct structure for this design and is more careful than most systems papers bother with.

## Ablation Studies and Experiments

### Accuracy: yes, more candidates help

GSM8K, 500 prompts, $N{=}1 \to 8$:

| Model | $N{=}1$ | $N{=}8$ | Gain | 95% paired CI |
|---|---|---|---|---|
| Phi-3-mini | 81.4% | 89.8% | +8.4 pp | [5.8, 11.2] |
| Qwen2.5-1.5B | 51.4% | 69.8% | +18.4 pp | [14.8, 22.0] |

The weaker model gains more, which is the usual self-consistency story — there is more headroom between "can sometimes get it right" and "usually gets it right".

**The curiosity in the curve:** $N{=}1$ and $N{=}2$ give *identical* accuracy. That is the tie rule, not a measurement bug. With two candidates that disagree, each gets one vote, and the first-appearing answer wins — which is just $N{=}1$. Gains only start at $N{=}3$, when a majority can actually form. A useful reminder that **even-numbered voting budgets waste a sample.**

The authors check whether tie handling or extraction failure is doing the work:

| Model | $N$ | Tie rate | Primary | Random tie-break | Cond. on extraction |
|---|---|---|---|---|---|
| Phi-3 | 2 | 25.0% | 81.4 | 81.9 | 82.1 |
| Phi-3 | 8 | 3.4% | 89.8 | 90.1 | 89.8 |
| Qwen | 2 | 63.0% | 51.4 | 50.5 | 52.1 |
| Qwen | 8 | 18.2% | 69.8 | 71.2 | 70.5 |

Max swing 1.4 pp. So the gain is real, not a voting-rule artefact. Note Qwen's 63% tie rate at $N{=}2$ — it disagrees with itself constantly, which is exactly why it gains 18 points from voting.

### Systems cost of raising batched $N$

Phi-3 on V100, Qwen on A100 here — **do not compare the two columns**, only read trends within each.

| $N$ | Phi-3 J/query | Phi-3 J/token | Qwen J/query | Qwen J/token |
|---|---|---|---|---|
| 1 | 631 ± 19 | 2.634 | 596 ± 13 | 2.222 |
| 2 | 795 ± 15 | 1.607 | 670 ± 19 | 1.265 |
| 4 | 974 ± 36 | 1.004 | 802 ± 6 | 0.758 |
| 8 | 1286 ± 49 | 0.655 | 975 ± 20 | 0.459 |

The scissors: **energy per query roughly doubles, energy per token falls ~4–5×.** Batching is enormously efficient per unit of work, and you still pay more in total because you asked for 8× the work.

Latency $N{=}1 \to 8$: 5.06 → 8.91 s (Phi-3), 5.26 → 7.64 s (Qwen). Only ~1.5×, for 8× the candidates — that is the batching win. GPU-hours per 1,000 queries: 1.41 → 2.47 and 1.46 → 2.12.

### The main result: fixed $N{=}8$, vary only the schedule

All values normalised to $1{\times}8$. Lower is better.

| Schedule | Energy (Phi-3) | Energy (Qwen) | P95 latency (Phi-3) | P95 latency (Qwen) |
|---|---|---|---|---|
| $1{\times}8$ | 1.00 | 1.00 | 1.00 | 1.00 |
| $2{\times}4$ | 1.63 | 1.66 | 1.81 | 1.97 |
| $4{\times}2$ | 2.71 | 2.86 | 3.21 | 3.57 |
| $8{\times}1$ | 4.64 | 4.86 | 5.77 | 6.12 |

The **monotone, smooth** progression matters. If only the $8{\times}1$ endpoint were bad, you would suspect some special-case pathology in batch-size-1 kernels. Instead every halving of batch size costs you, gradually. This is a continuous effect of exposed parallel work, not a cliff.

Bootstrap CIs on the $8{\times}1 / 1{\times}8$ energy ratio: Phi-3 **4.64× [4.48, 4.79]**, Qwen **4.86× [4.71, 5.02]**. P95 latency 5.77× [5.38, 5.99] and 6.12× [5.70, 6.75]. Throughput retained: **16.7%** and **17.9%**.

In wall-clock terms, per 1,000 queries: 2.09 → 12.49 GPU-hours (Phi-3), 2.13 → 11.76 (Qwen).

**The power/energy dissociation.** Phi-3 mean power *drops* from 177.8 W ($1{\times}8$) to 139.8 W ($8{\times}1$) — the serial run looks gentler on the GPU at any instant. But mean latency rises 5.97×, so $E = P \cdot t$ goes up sharply anyway. A monitoring dashboard showing watts would tell you the serial schedule is the greener option. It is ~4.6× worse.

### Cross-node robustness

Three independently scheduled A100 jobs per model, the two endpoints repeated:

| Metric | Phi-3 range | Qwen range |
|---|---|---|
| Gross J/query ratio | 4.43–4.64 | 4.85–4.88 |
| Mean latency ratio | 5.85–5.97 | 5.50–5.53 |
| P95 latency ratio | 5.53–5.77 | 6.06–6.12 |
| Throughput retained | 16.7–17.1% | 17.9–18.0% |

Ranges are tiny next to the ~5× effect. Not a node fluke.

### Length stratification — the "is it just long outputs?" check

The 100 systems prompts are split into four groups of 25 by mean candidate length (measured from the separate accuracy generations).

- Phi-3: energy ratio 4.40–4.77, latency ratio 5.45–6.16
- Qwen: energy ratio 4.51–5.09, latency ratio 5.15–5.88

**Non-monotonic in response length.** The penalty is not concentrated in the long-output group — it shows up everywhere. The authors flag this as descriptive only: 25 prompts per bin, and the schedules sampled candidates independently.

### Short-output validation: SciQ on V100

SciQ answers are ~1.71 tokens (Qwen) and ~3.46 tokens (Phi-3) on average. Almost pure prefill, essentially no decode.

| Model | Schedule | Rel. energy | Rel. latency | Rel. throughput |
|---|---|---|---|---|
| Phi-3 | $2{\times}4$ | 1.29 | 1.37 | 0.74 |
| Phi-3 | $4{\times}2$ | 1.75 | 1.97 | 0.51 |
| Phi-3 | $8{\times}1$ | **2.57** | 2.88 | 0.36 |
| Qwen | $2{\times}4$ | 1.38 | 1.59 | 0.63 |
| Qwen | $4{\times}2$ | 2.04 | 2.59 | 0.39 |
| Qwen | $8{\times}1$ | **3.34** | 4.42 | 0.23 |

Same direction, same monotonicity, **smaller magnitude** (2.57–3.34× instead of 4.64–4.86×). Tempting to conclude "shorter outputs → smaller penalty", and the authors explicitly refuse to: SciQ ran on V100 while GSM8K ran on A100, so GPU architecture and output length are confounded. The claim they actually make is narrow — the effect survives in both regimes. Good discipline; the over-claim was right there.

### What did not work, and what is missing

**They do not decompose the mechanism.** This is the paper's own biggest admission. The 4.6× is an end-to-end number. Candidate causes are listed — per-call framework and synchronisation overhead, repeated prompt processing / prefill per call, insufficient parallel work exposed to the SMs — but no profiling separates them. Since every candidate shares one prompt, my strong suspicion is that repeated prefill is a large slice: the $8{\times}1$ schedule pays the prompt eight times with zero batching amortisation, whereas $1{\times}8$ prefills once across a batch. Left unquantified. **Open question.**

**Candidates are sampled independently across schedules.** So $1{\times}8$ and $8{\times}1$ are not generating the *same eight strings*. Token volumes are matched to within 1%, which is good enough for the systems claim, but it is a matched comparison rather than a controlled one.

**Average power is the metric that fails.** Not a failed experiment exactly, but the clearest negative result: it moves in the *opposite* direction to total energy.

**No verifier or reward model.** Plurality vote only, so nothing here about whether better-of-$N$ selection changes the picture.

**The scaling extrapolation is not a projection and the authors say so.** Linearly applying the per-query gap to a million queries: +1.37 MWh and +10,401 GPU-hours for Phi-3, +1.00 MWh and +9,631 GPU-hours for Qwen. They label these "linear illustrations of the measured configurations". Treat them as a sense of scale, nothing more.

## Worth Remembering

**The recommended heuristic**, which is about as simple as guidance gets. Let $b_{\max}$ be the largest candidate count that fits GPU memory and other constraints. Then

$$b = \min(N, b_{\max}), \qquad C = \left\lceil \frac{N}{b} \right\rceil$$

with the last call taking the remainder. Fewest calls, largest feasible batch. If all 8 fit, use $1{\times}8$; if only 4 fit, use $2{\times}4$.

**When the heuristic does not apply** — their own table, and this is the honest half:

| Situation | What to do |
|---|---|
| All candidates fit | one call, batch $N$ |
| Memory-limited | largest feasible batch, fewest calls |
| Candidates depend on earlier outputs | sequential is forced |
| Continuous-serving environment | coordinate with the serving scheduler |
| Extra latency / resource limits | pick any feasible schedule; among feasible ones, fewest calls |

That third row is the big carve-out. Anything with a dependency chain — self-refine, tree search, [[Monte Carlo Tree Search|MCTS]]-style rollouts, sequential-revision test-time scaling — *cannot* be batched into one call. The paper's guidance covers the independent-samples family only: self-consistency, best-of-$N$, repeated sampling.

**The scope limits, stated plainly.** Two small models (3.8B and 1.5B). Two GPU generations. Plain Hugging Face `generate`, single GPU. No continuous batching, no [[Quantization|quantisation]], no [[Speculative Decoding|speculative decoding]], no tensor or pipeline parallelism. Under a real serving engine like vLLM, per-call overhead is amortised very differently and the ratios would almost certainly shrink — the authors say as much. **Do not port 4.64× into a vLLM deployment.**

**Why the HPC framing is the right one.** In a batch-job world — Slurm, allocated GPU-hours, a finite job rather than a continuous server — the schedule is genuinely yours to choose, and reasons to serialise pile up fast: cleaner logging, deterministic per-call seeding, control logic between calls, intermediate analysis, adaptive stopping. Each of those is a plausible engineering reason to write a `for` loop around `generate`, and each costs ~5× on A100. Anyone using [[Slurm]] and thinking in [[HPC|allocated GPU-hours]] should read this as: the loop you wrote for convenience is the dominant line item.

**The tension with adaptive stopping is unresolved.** Adaptive-consistency and difficulty-adaptive self-consistency *deliberately* serialise so they can stop early on easy prompts — sample a few, check agreement, stop. That saves candidates and costs calls. This paper shows calls are expensive. Nobody has done the joint accounting: does adaptive stopping still win once you charge it for its schedule? One of the authors has a 2026 adaptive-sampling paper, so presumably they know this. **Open question, and it's the most interesting one here.**

**What to actually measure, if you take one thing away.** Report $N$, calls per query, candidates per call, batching mode, hardware, latency, throughput, GPU-hours, and *the energy measurement boundary*. That last item is doing a lot of work: "gross device energy over the query interval, idle not subtracted" and "whole-node energy minus idle baseline" can differ by a large factor, and papers reporting a bare joule figure are not comparable.

**Sanity anchor.** Phi-3 at $N{=}8$ batched costs ~1286 J/query, roughly 0.36 Wh. Serialised, ~5970 J, about 1.66 Wh. At a million queries that is the difference between ~0.36 MWh and ~1.7 MWh, on GPU device power alone, for arithmetic word problems.

**Follow-up questions I would want answered:**

1. How much of the 4.6× is repeated prefill? A prefix-cache experiment ([[Prompt Caching|prompt caching]] across calls) would isolate it almost immediately.
2. Where does the curve sit for a 70B model, where a single sequence already saturates the SMs? Likely much flatter.
3. Under [[Continuous Batching|continuous batching]], does the effect vanish entirely — does the server just merge your eight calls back together?
4. What is the accuracy-per-joule frontier? Two axes exist here (how many candidates, how grouped) but they are only ever varied one at a time.

**Epistemic tier.** The measurements themselves are solid and well-controlled for what they are — pinned clocks, reversed ordering, three independent nodes, hierarchical paired bootstrap. That is **established** for these two models on these two GPUs under Hugging Face `generate`. The direction of the effect (fewer, bigger calls are cheaper for independent samples) is **current best understanding** and follows straightforwardly from known batching physics. The *magnitudes* are **specific to this setup** and should not be reused. The mechanism split is **open**.

## Links

Related: [[Chain of Thought]] · [[Test-Time Compute]] · [[Continuous Batching]] · [[Prefill and Decode]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]] · [[Cost and Latency]] · [[KV Cache]] · [[Sampling Parameters]] · [[HPC]] · [[Slurm]] · [[GPU processing]] · [[Prompt Caching]] · [[Speculative Decoding]] · [[Quantization]] · [[Evals]] · [[Troubling Trends in Machine Learning Scholarship]] · [[On the Difficulty of Evaluating Baselines]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Prefix Sliding for efficient test-time scaling]]

New topics worth writing: Self-consistency and plurality voting over reasoning paths, Best-of-N selection and verifier-based reranking, Adaptive-consistency and early-stopping sampling budgets, NVML energy measurement methodology and boundary definitions, Energy-per-token vs energy-per-query as inference metrics, ORCA and Sarathi-Serve iteration-level scheduling, Reporting standards for inference systems metrics, Hierarchical paired bootstrap for repeated-measures systems experiments, GSM8K, SciQ
