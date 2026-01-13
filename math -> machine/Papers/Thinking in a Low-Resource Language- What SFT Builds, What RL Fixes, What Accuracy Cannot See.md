---
title: "Thinking in a Low-Resource Language: What SFT Builds, What RL Fixes, What Accuracy Cannot See"
authors: ["Ayoub Kirouane", "Christos Petrocheilos"]
year: 2026
arxiv: "2608.17744"
url: https://arxiv.org/abs/2608.17744
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, transformers, llm, rl]
---
## The Core Idea

Fine-tune a big model so it *thinks out loud in Greek*. Then measure what changed. On the accuracy benchmark: nothing. Best fine-tuned arm scores **76.5** against the base's **77.2** on a 1,000-item Greek probe. A null result.

But the null is fake in a specific way. The authors re-ran **one configuration with only the random seed changed** and the score moved **7.7 points** (76.2 / 68.7 / 76.4, sd = 4.4 pp). Every data effect, corpus version, and selection method they had spent weeks interpreting was *smaller than the noise from doing nothing*. Five written-up conclusions died.

So the paper becomes two things at once:

1. **A measurement paper.** If accuracy is noise at this scale, what *isn't*? Answer: behaviour. Across those same three seeds, the fraction of traces written in Greek was $1.00 / 1.00 / 1.00$ and median trace length was $132 / 148 / 152$ words. The behavioural dimensions sit still while accuracy swings 7.7 points. That asymmetry is what makes the rest of the paper measurable.

2. **A result about what supervised fine-tuning actually installs.** The base models — Qwen3.6-35B-A3B, Gpt-OSS-20B, two Nemotron hybrids — reason in Greek on **0 of 1,000** traces even when the question is Greek. Median Greek-character ratio 0.33: mixed script with English scaffolding ("*We need to answer a multiple-choice question in Greek…*"). The user gets a right answer produced by reasoning they cannot read, audit, or correct. After SFT, every checkpoint reasons in the question's language on **97.4–98.1%** of items.

> [!NOTE] Language fidelity ^language-fidelity
> $g(t) = |\mathrm{GR}(t)| / (|\mathrm{GR}(t)| + |\mathrm{LA}(t)|)$ — the share of alphabetic characters that are Greek, after stripping code and LaTeX (or every technical trace reads as Latin). Report median $g$ and the fraction of traces with $g \geq 0.9$.

The third thread: SFT cannot fix its own defects. A quarter of answers skip the requested output format, answers leak into the reasoning channel, and "think in English" is obeyed under half the time. A pre-registered round of **RLVR** (reinforcement learning where the reward is a deterministic check, not a learned model or a judge) fixes the first two outright: format fallback $24\% \to 2.5\%$, answer-channel leak $3.5\% \to 0.0\%$, both against a **flat random-reward control** that moved neither.

The unlock: a template for evaluating any low-resource-language fine-tune where the accuracy number is uninformative.

## The Methodology

**Models.** Three sparse mixture-of-experts families, four checkpoints, chosen so that *active* parameters per token are nearly identical (3.58–3.97B) while total size varies $1.7\times$:

| lab | base | total | active | routing |
|---|---|---|---|---|
| Alibaba | Qwen3.6-35B-A3B | 36.0B | 3.97B | 8 of 256 |
| OpenAI | Gpt-OSS-20B | 20.9B | 3.60B | 4 of 32 |
| NVIDIA | NemotronH-30B-A3B | 31.6B | 3.58B | 6 of 128 |
| NVIDIA | Nemotron-3.5-Lightning-30B-A3B | 31.6B | 3.58B | 6 of 128 |

The two Nemotrons are Mamba/MoE hybrids — state-space layers replace most [[Attention Is All You Need|attention]]. Active parameters is the axis they treat as real, because it is what serving costs.

**Adaptation.** [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]] at $r=32$, $\alpha=64$, one epoch, effective batch 32, learning rate $2\times10^{-4}$. One DGX B200 node ($8\times$ B200). "Expert stride 3" means adapters go on every third MoE layer; the shared expert (on every token's path) always included. Why the stride: when experts are individually-materialised `nn.Linear` modules, PEFT attaches **one adapter per expert** — 4,188 adapter tensors on NemotronH against 28 on Qwen, at *fewer* trainable parameters (325M vs 688M). The cost is kernel launches and gradient all-reduces, not FLOPs; it trains $3.7$–$7.6\times$ slower.

**Corpus.** 118,092 Greek rows, two roughly equal halves.

- *Reasoning half* (59,107): explicit trace in a separate field, **98.5% synthetic**. Questions and gold answers translated from public English sets; traces **generated, not translated**. Math 38.2%, science 18.7%, world knowledge 10.0%, logic 8.7%, medical 8.3%, commonsense 6.6%.
- *Direct half* (58,985): no traces, reused unchanged from the Sophea-Titan-1 instruction mix. 91% Greek, 9% English kept deliberately as **replay** against catastrophic forgetting.

**Where traces come from.** Ask a translator to render a chain of thought and you get a tidy summary. So they prompt a frontier model to solve each question afresh *in Greek* and keep the trace only if the final answer matches gold — the answer-gating rule from STaR. Wrong traces are discarded, not repaired. Yield 60–95%.

They score trace *genre* with a corpus diagnostic:

$$S = 0.40\,b + 0.25\,v + 0.25\,p + 0.10\,\ell$$

$b$ = backtracking markers, $v$ = verification, $p$ = fraction that is flowing prose rather than a numbered list, $\ell$ = length. Backtracking dominates because a write-up never has it. A public Greek ECQA set with human-written justifications scores $S = 0.27$ with **0%** of traces above 0.5; regenerating traces for the *same questions* gives $S = 0.68$ with **97%** above 0.5.

**Three recipes.** *One-Phase* = one pass over both halves. *Two-Phase* = reasoning-only, then continue the same adapter on the hybrid mix. *Reasoning-only* = the reasoning half alone.

**Benchmark.** 5,156 Greek items: 1,345 math (250 human-translated MGSM-el + 1,095 machine-translated GSM8K test), 3,267 commonsense (HellaSwag-el + WinoGrande-el), 544 decontaminated ProofWriter-el logic reported as **macro-recall**. Greedy decoding.

**The six metrics.** Accuracy is one of six, not a summary.

- **M1 Correctness** — *anchored* accuracy: parse the requested answer line first, free text only if absent. Logic is macro-recall because the class split is 891/270/165 and a majority answerer scores 67% raw, 33% macro.
- **M2 Language fidelity** — as above.
- **M3 Reasoning budget** — median words, and **words per correct answer** $\sum_i w_a(i)/|\{i: a\text{ correct}\}|$.
- **M4 Termination** — share of rows hitting the generation cap with no answer.
- **M5 Reasoning steps** — count of *intermediate conclusions* (a numeric result asserted, an option evaluated). Explicitly **not** sentence count, which is $w/12$ in disguise.
- **M6 Budget overrun** — with $f(i)$ the fraction of arms correct on item $i$ and $\tilde{w}(i)$ the median trace length across arms on $i$:

$$\mathrm{OV}_a = \frac{|\{i : f(i) \geq 0.8 \wedge w_a(i) \geq 3\tilde{w}(i)\}|}{|\{i : f(i) \geq 0.8\}|}$$

Overspending on items nearly everyone gets right, where difficulty cannot excuse it. Both ingredients are model-independent.

> [!NOTE] The length gate ^length-gate
> Any candidate behavioural metric whose correlation with trace length is $|r| \geq 0.6$ gets **redefined or dropped**, never reported with a caveat. This removed 3 of their original 7 dimensions. The fix for a length-caused metric is not normalisation — it is a comparison where length is held constant by design (hence conditioning on the item in M6).

Everything is reported in **words**, not tokens, because Greek costs $2.3$–$2.5\times$ the tokens per word of English on all three tokenizers (measured fertility $2.32$, $2.39$, $2.51$). A uniform `--max-new` budget is a $2.4\times$ tighter ceiling in Greek.

**The RLVR round.** GRPO on the released Qwen checkpoint. 8 completions per prompt, group-relative advantage, **no KL term** ($\beta = 0$), lr $10^{-6}$, temperature 0.7, completion budget 1,536 tokens. LoRA $r=32$, **dropout fixed at 0** — under a policy gradient, dropout makes the update policy differ stochastically from the rollout policy and corrupts the importance ratio. Prompts arrive in language-matched pairs. Five deterministic reward terms: correctness (locale-aware numeric parser), language consistency (redirected to the *instructed* language when an instruction is present, so the two terms cannot both be satisfied by ignoring the instruction), format, termination, override obedience. Two gates: language terms need a minimum letter count after stripping code/LaTeX; every behavioural term requires the trace to contain **work** — an intermediate value that is not the answer restated, deliberately not a length test.

A **28-attack adversarial fuzz suite** must pass before any rollout: trace elision, fluent Greek filler with no computation, answer shotguns, code-block laundering, the locale exploit, truncation mid-number, obeying/disobeying override pairs. It caught three exploits in their own draft reward, including a fluent no-computation trace scoring full marks. Four arms, identical data/steps/seed/hardware, varying only reward terms — including a **random-reward control**.

## Ablation Studies and Experiments

**The seed control (the finding that reorganised the paper).**

| seed | mean | logic | fallback | trace Greek |
|---|---|---|---|---|
| 42 | 76.2 | 56.2 | 10% | 1.00 |
| 43 | 68.7 | 35.6 | 41% | 1.00 |
| 44 | 76.4 | 54.0 | 3% | 1.00 |

sd 4.4 pp, range 7.7 pp. Cost: one extra training run. Note this reads less like symmetric jitter than an occasional failure mode — one run in three lands ~8 pp low, taking logic *and* instruction-following down together.

The power calculation is the sharp bit. For $\Delta$ detectable at $\alpha = 0.05$ with 90% power against $\sigma \approx 4.4$:

$$n \gtrsim 2\left(\frac{(z_{\alpha/2}+z_\beta)\sigma}{\Delta}\right)^2$$

≈ **11 seeds per arm** for their largest effect (6.3 pp), ≈ **40** for the 3.2 pp one. They could afford neither, and say so.

Also: this is not the $\pm3.4$ pp *sampling* error a 1,000-item benchmark implies. That is item-draw noise. Training-run variance is roughly twice as large. Papers reporting one number per config are accounting for the smaller of the two sources.

Honest caveat they include: with $n=3$, the 95% chi-square interval on $\hat\sigma = 4.4$ is $[2.3, 27.7]$ pp. The 7.7 number is a point estimate on one family. What transfers is the qualitative conclusion.

**Data selection does nothing.** *Subset* = top 15,607 rows by trace-structure score (mean structure 0.836). Control = 15,607 drawn uniformly at random (mean structure 0.587).

| rows | selection | mean |
|---|---|---|
| 62,562 | all | 69.5 |
| 15,607 | top-by-structure | 69.5 |
| 15,607 | uniform random | 69.0 |

Selection buys $-0.5$ pp ($0.31\sigma$). A $4\times$ smaller pool costs nothing. Pre-registered prediction was selection wins by >2 pp.

**What did move.** At *equal* item-pooled accuracy (72.9 vs 72.9), the base spends median **1,010 words** per trace and **1,454 words per correct answer**; the reasoning-only fine-tune spends **150** and **239**.

But the token story flips sign by family. Retokenizing every trace with each family's own tokenizer:

- Qwen release: **$3.0\times$ fewer** tokens than base (586 vs 1,788)
- NemotronH: **parity** (638 vs 681)
- Gpt-OSS: **$1.6\times$ more** (640 vs 396)

Greek's fertility tax must be repaid by shortening before the word-level saving becomes a serving saving. On Gpt-OSS the Greek fine-tune is *more expensive to serve* than its English-reasoning base.

**Budget discipline.** The base overruns on **98.0% of easy items** — and its rate on hard items is 98.4%. It has **no representation of question difficulty**. Fine-tunes: 0.2–12% on easy, and range $1.9$–$60\times$ between easy and hard in direct mode.

**Length is a marker of struggle, not effort.** Within every arm, longer traces are *less* accurate (base: 82.5% under 400 words vs 72.3% over). Conditioning on trace length, the base is at least as accurate as the fine-tune. The fine-tune's advantage is that it does not enter the long unproductive regime, not that its tokens are worth more. Both stratifications are endogenous — no causal claim available.

**The one accuracy effect that survives.** Reasoning-only vs two-phase, across **15 independently trained arms**: 73.6 ($n=6$) vs 66.7 ($n=9$), 52 of 54 pairwise comparisons favouring reasoning-only, exact permutation $p = 0.0008$. A *single* A/B on this comparison is $+6.3$ pp and sits inside the floor — uninterpretable. The aggregate is the correct unit.

Its confound, stated: arms are historical runs and corpus version varies within both groups. Stratifying the permutation by corpus version leaves 3 usable strata and 16 total permutations; the observed split ranks second ($p = 0.125$, attainable floor 0.0625). Supportive, almost no resolution.

Mechanism: `fallback%` (never emitting the requested answer line) is 2–12% for reasoning-only, 33–40% for two-phase. Training on ~59k non-reasoning rows degrades instruction-following. Part of what looked like a reasoning regression is a model that still reasons but no longer answers in the requested form.

**Three recipes, three different failures:**

| recipe | mean | empty trace | fallback |
|---|---|---|---|
| One-Phase | 66.1 | 23.6% | 12% |
| Two-Phase | 64.4–69.9 | 0.0–1.3% | 33–40% |
| reasoning-only | 73.6 | 0.0% | 2–12% |

One-Phase **collapses the reasoning switch**: asked to think, it returns an empty trace on 23.6% of items. This is coverage-independent — it counts empty traces, not answers. Two-phase fixes the switch and breaks instruction-following.

Reverses on Gpt-OSS: 62.5 vs 68.1, one run per condition. Both statements go in the record.

### Six ways the instruments lied

This is the best section of the paper.

1. **Un-anchored scorer.** The prompt requests "write on a new line: The answer is `<letter>`". Their scorer scanned the whole response and let the last option-mention win — so a model that answers, then explains why the others are wrong, is scored on its explanation. One-directional bias against verbose arms. Produced a **$+29.8$ pp ($18\sigma$)** artifact favouring the fine-tune. Anchoring removes it entirely.

2. **A capability gap that was a format artifact.** A $-3.0$ pp commonsense deficit appeared in **12 of 12** fine-tuned arms — textbook catastrophic forgetting. Re-asking the same 300 items with a letter-only prompt reverses it to **$+1.7$ pp**. And format must be chosen per axis: letter-only is valid for commonsense but invalid for logic, where it drops the base to 29.6% against a 33% chance baseline.

3. **Contamination a standard check missed.** The logic axis was **38.9% contaminated** — 175 of 450 items shared a 13-gram with the training pool; math and commonsense were clean at 0%. Cause was structural: benchmark and training slice drawn from the same ProofWriter pool, which the build script assumed was eval-only. Removing items moved arms by 0.7 to 22.2 pp — *unevenly*, which is what differential memorisation looks like.

4. **Translating the benchmark with the family you trained on.**

| arm | human-transl. | machine-transl. | $\Delta$ |
|---|---|---|---|
| Base | 95.6 | 90.7 | $-4.9$ ($-3.14\sigma$) |
| Subset | 94.0 | 95.1 | $+1.1$ |
| Reasoning | 93.6 | 95.3 | $+1.7$ |
| Two-Phase | 87.6 | 90.4 | $+2.8$ |

The translation model *is* the model that generated the math traces. The fine-tunes are adapted to its Greek register; the base is not. Include the machine-translated items and the fine-tune appears to beat base by 0.6 pp; on the human subset the base leads by 1.1. **No contamination check finds this — the items are novel, only the register is shared.** A 250-item human control caught a $3.1\sigma$ artifact.

5. **A scorer that cannot read Greek numbers.** Greek writes $17.500$ for seventeen-thousand-five-hundred and $3{,}5$ for three-point-five — separators reversed from English. Their extractor strips commas and calls `float()`. Six answers on the released Qwen checkpoint are in Greek thousands form; **five are scored wrong while being right**. The mirror is worse: $3{,}5$ becomes $35$, scoring a wrong answer correct. Effect on the paper: 0.45 pp on one axis, no number changes. It generalises: invisible to any English test suite, systematic, and it penalises exactly the answers written in the target language's own convention. **Locale is part of the instrument.**

6. **A default that changed which lane was measured.** A same-day baseline read 1.1% fallback where the frozen number was 24%. Rescoring the *original* dump reproduced 24.1% exactly — acquitting the scorer. The generations differed: without an explicit flag, the chat template rendered with the reasoning trace **disabled**. Four GPUs spent four hours generating the direct lane under a filename that said think lane. Nothing crashed, nothing warned, every number was internally plausible.

**The repetition metrics that had to be thrown away.** Three definitions of degenerate looping, all length proxies: $|r|$ with word count of 0.44–0.85 for a per-token rate, 0.86–0.95 for a fixed-window rate. A "the fine-tune loops $29\times$ less" claim was pure artifact.

The asymmetry underneath is genuinely interesting though. Applying a repetition penalty costs the base **$-0.4$ pp** (inside noise) and the fine-tune **$-9.7$ pp**. The base loops when it has nothing to say; the fine-tune's repeated spans are **load-bearing** — part of its argument, not its style.

**Is Greek reasoning worse reasoning?** An early run showed $-17.5$ pp for Greek vs the same items in English. Run the *same* items, force the *same* answer language, vary only trace language: the effect collapses to **$+1.4$ pp ($0.65\sigma$)**. The $-17.5$ was a different, easier question mix in the English lane expressing itself as a language effect.

**Forgetting (E2), on the Titan-1 suite** — 9 Greek + 5 English NLU benchmarks, scored by log-likelihood with no generation:

| release | Greek macro $\Delta$ | English macro $\Delta$ |
|---|---|---|
| Sophea-Qwen3.6-v1 | $-0.01$ | $+0.08$ |
| Sophea-OSS-v1 (repaired) | $-3.2$ | $+1.0$ |
| Sophea-Nemo-3-Nano-v1 | $+3.8$ | $-2.0$ |
| Sophea-Nemo-3.5-Lightning-v1 | $+1.7$ | $-1.1$ |

The Gpt-OSS story is the interesting one: pre-repair it lost $-7.3$ Greek and $-7.7$ English with a 70% fallback rate. Adding a **2% dose of trace-less anchored answer-closing twins** recovered 4.1 Greek points, pushed English *above* base, and cut fallback $70\% \to 26\%$. So the loss was format-closing behaviour, fixable by data.

Fluency probes (LLM judge at temp 0, small $n$, reported as counts): **grammaticality improves on all four families**, most where the base is weakest — NemotronH $13/58 \to 27/57$, Nemotron-3.5 $12/58 \to 29/58$, Gpt-OSS $32/57 \to 40/58$, Qwen $41/58 \to 42/58$. Register control retained or gained everywhere.

One more instrument failure hidden here: the first-pass Gpt-OSS grammar score was $23/58$ — a spurious regression from a response-splitting bug that prefixed a stray channel marker. The judge read the marker as ungrammatical Greek. Fixing the splitter flipped it from worst fine-tune to above base.

**The language lock (E1).** One-directional training produces a model that ignores the instruction entirely:

| model, instruction | English traces | Greek traces |
|---|---|---|
| base, "reason in English" | 100.0% | 0.0% |
| base, "reason in Greek" | 72.0% | 0.0% |
| one-dir. ft, "reason in English" | **0.0%** | 98.7% |
| one-dir. ft, "reason in Greek" | 0.0% | 98.7% |
| matched ft, "reason in English" | 44.8% | 24.8% |

The base is *partially* locked the other way — told to reason in Greek it still emits English 72% of the time. Asymmetric locking is not unique to fine-tuning; the firmness is.

The fix is **language-matched pairs**: each Greek problem and trace kept beside its English original, gated so a drifting trace never enters training. This restores the question-following default on *every* family (100% English traces on English questions). But the *instruction* channel only re-opens on two of four:

| release | "in English" (Greek Q) | "in Greek" (English Q) |
|---|---|---|
| Sophea-Qwen3.6-v1 | 44.8% | 83.7% |
| Sophea-Qwen3.6-v1.1 (RLVR) | 53.9% | 85.7% |
| Sophea-OSS-v1 | 62.5% | 93.3% |
| Sophea-Nemo-3.5-Lightning-v1 | **0.0%** | 92.7% |
| Sophea-Nemo-3-Nano-v1 | **0.0%** | 87.5% |

Same corpus, same recipe, same adapter config, same serving budget, agreement on every default-behaviour metric — and they split $62.5/44.8/0/0$ on this one axis. Tracks neither architecture nor recipe. Unexplained.

**The RLVR outcomes, against rules frozen before the first optimizer step:**

| arm | Greek fid. % | acc | fallback % | leak % | override el→EN % |
|---|---|---|---|---|---|
| SFT checkpoint (before) | 97.98 | 73.7 | 24.1 | 3.53 | 44.8 |
| **random reward (control)** | 98.06 | 74.0 | **22.1** | **3.61** | 44.1 |
| correct+format+term. | 98.22 | 77.2 | **2.5** | **0.00** | 48.8 |
| + language | 98.02 | 77.2 | 5.6 | 0.04 | 45.9 |
| + override | 98.27 | 77.0 | 2.8 | 0.02 | **53.9** |

The control row is the whole admissibility argument — random rewards have been shown to recover most of an RLVR gain on this model family, so any axis where the control matches a trained arm is *elicitation, not learning*, and the pre-registered rule withdraws that result. It matched on nothing.

Cost: ~78 GPU-hours per arm (39 h wall on two GPUs), 2,000 prompts.

**What did not clear its bar.** Override obedience moved $+9.1$ pp (control-adjusted $+9.8$) while holding every gate — but the frozen rule required $\geq +15$ pp reaching $\geq 60\%$ to call it *trainable*. Middle branch. They report it as "reward-responsive, bar missed" rather than rewording the criterion.

**The collapse that did not come.** The prior said an accuracy-only GRPO gradient erodes cross-lingual consistency to zero. The unprotected arm ended at **98.22%** fidelity — above its start, zero in-question switches. Stated as bounded: contradicts the dense-scale prior at *this* operating point (MoE, LoRA, 16,000 completions, property installed by SFT not RL), not everywhere.

Odd result with no mechanism: the **+language arm gains least** on override (45.9 vs 48.8 without the term), even though the reward redirects that term on instructed rows by construction.

**Per-domain accuracy is unreadable.** Math 87.6–95.6, commonsense 70.7–82.3, logic 40.5–46.3 — spreads of 8.0, 11.6, 5.8 pp against a 7.7 pp seed range. Two exceed the range itself. Naming a per-domain winner selects the top of a noisy draw.

## Worth Remembering

**No fine-tuned arm beats its own base on the pooled Greek reasoning benchmark, in any family.** They print this plainly. And an earlier draft got caught reading a Qwen "gain" on commonsense ($63.2 \to 75.8$) that was actually a cross-family comparison — the $63.2$ was the *Gpt-OSS* base. Against its own base's $82.4$, the Qwen release is $-6.6$. The final tables are restructured so every release is indented under its own base, making the cross-base reading structurally hard.

**Format compliance eats most of the apparent deficit.** Conditioning on rows that emit the requested answer line: Qwen release reads $95.8/78.8/39.5$ against base $96.0/82.8/42.7$. Gpt-OSS pre-repair goes $74.6 \to 92.2$ on math. Report `fallback%` beside every accuracy number or you are measuring two things at once.

**The per-family gotchas, all of which produced plausible numbers rather than crashes:**

- Qwen's fused-expert LoRA adapter **cannot be loaded back** by `PeftModel.from_pretrained` — must merge to dense weights before eval. Lost an entire evaluation round.
- Qwen's base is **truncation-crippled**: at 1,536 tokens it scores 20.8 with 670/1000 rows never reaching an answer; at 4,096 it scores 77.2. Any comparison at a standard budget measures the budget.
- Gpt-OSS does not use `<think>` — reasoning arrives in harmony channels (`analysis`/`assistantfinal`). It emits an analysis channel on **1000/1000 rows even at `reasoning_effort=none`**. Split on markers present, not on having asked.
- NemotronH needs `ddp_find_unused_parameters=True` (routing leaves experts gradientless, DDP aborts mid-run), `model_accepts_loss_kwargs=False` (otherwise loss and gradients scale with accumulation count), and `trust_remote_code=False` at eval (its `prepare_inputs_for_generation` indexes a `cache_position` that `generate()` passes as `None` — fails only at inference, after five hours of successful training).
- Installing `mamba-ssm` changed step time by **0%**, despite transformers logging a fallback warning on every run. Worth recording because the obvious diagnosis was wrong.

**Limitations they own.** Everything is LoRA on MoE — no dense control, and they cannot separate "LoRA SFT is seed-sensitive at this scale" from "sparse routing amplifies seed sensitivity", since routing decides which experts get gradient at all. They call a dense control the single cheapest experiment that would sharpen the paper. Greek is *mid*-resource with a dedicated open-model ecosystem; transfer to genuinely low-resource languages is untested. **No human read the traces** — apart from one annotator hand-labelling ~150 traces for switching, with no guidelines or agreement measurement, every fidelity number is automatic. So "auditable by the user" is measured as script identity, not fluency or followability, and it inherits the standing assumption that a chain of thought is faithful to the computation it narrates. Two cheaper alternatives never compared: post-hoc translating the base's English trace, and few-shot Greek exemplars (their "prompting cannot reach it" evidence covers bare instructions only). And the zero-switch target treats all code-switching as a defect, though Greek technical register routinely borrows English terms.

**The checklist worth stealing wholesale.** Run the seed control *before* the ablations — one extra run told them more than five corpus versions. Keep a human-translated subset of any machine-translated benchmark. Test numeric normalisation in the target language's conventions, both directions. Gate metrics on length correlation. Run a random-reward control arm with any RLVR claim. Probe trace-language steerability **per checkpoint** — it is not predictable from recipe, family, or any other metric they measured.

**Follow-ups I would want.** Does the length gate ($|r| < 0.6$) generalise as a metric-design rule, or is 0.6 arbitrary? What makes the instruction channel open on two families and shut on two others after identical training — is it a base-model property measurable before fine-tuning? And does the $-9.7$ pp cost of a repetition penalty on the fine-tune mean SFT traces contain genuine repeated-verification structure, or just that the model has learned a stylistic tic the reward now depends on?

## Links

Related: [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Proximal Policy Optimization Algorithms]] · [[Training language models to follow instructions with human feedback]] · [[On the Difficulty of Evaluating Baselines]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Towards Quantifying Benchmark Optimization in ASR Models]] · [[Distilling the Knowledge in a Neural Network]] · [[Attention Is All You Need]] · [[Shortcut Learning in Deep Neural Networks]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[Regularization]] · [[Uncertainty]]

New topics worth writing: Mixture-of-Experts routing, GRPO and RLVR, STaR self-taught reasoning, tokenizer fertility in multilingual models, pre-registration in ML experiments, benchmark contamination detection, Mamba and state-space models, catastrophic forgetting, permutation tests, statistical power for ML ablations
