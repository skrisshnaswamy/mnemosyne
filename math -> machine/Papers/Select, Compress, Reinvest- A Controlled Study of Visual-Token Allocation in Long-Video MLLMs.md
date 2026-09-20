---
title: "Select, Compress, Reinvest: A Controlled Study of Visual-Token Allocation in Long-Video MLLMs"
authors: ["Prakhar Khatri"]
year: 2026
arxiv: "2609.03820"
url: https://arxiv.org/abs/2609.03820
priority: Good-To-Read
read_on: 2026-09-16
tags: [paper, llm, vision, theory]
---
## The Core Idea

A video language model never sees the whole video. Decode an hour at one frame per second and you have 3,600 images; the model is given eight, or thirty-two. Everything it will ever know about that video squeezes through those few frames. The rule that picks them is usually written off as preprocessing.

This paper's move is not a new method. It is a **controlled experiment**. Every published frame-selector paper changes four things at once: the scorer that ranks frames, the text handed to that scorer, the resolution policy, and the answering model. So when selector A beats selector B by 5 points, you have compared two experiments, not two ideas. Here everything is frozen except one decision at a time:

1. **Select** — change *which* timestamps you keep, holding frame count and resolution fixed.
2. **Compress** — hold the timestamps, shrink the pixels per frame.
3. **Reinvest** — spend the tokens you just saved on *more* timestamps instead of sharper ones.

Each comparison is **paired**: the same benchmark question answered twice, once per policy, everything else identical. So the difference is attributable to that one change.

Two results are worth carrying around. First, selection substitutes for frame count: on hour-long LongVideoBench videos, **8 query-chosen frames beat 16 evenly-spaced ones by 6.9 points** ($p{=}.0011$). Half the input, better answers. Second, the selector that does this is **Orthogonal Matching Pursuit**, a greedy sparse-approximation algorithm from 1993, used unmodified and untuned, with zero video-specific machinery. It matches or comes within a point of every purpose-built modern selector tested.

> [!NOTE] Orthogonal Matching Pursuit (OMP)
> A greedy algorithm for picking a small set of "atoms" (here: frames) that best explain a target vector (here: the question embedding). Each step takes the atom most aligned with what is *left over*, then subtracts everything already picked from the target. Relevance plus anti-redundancy, in two lines. ^omp-selector

The reason that result is interesting is not "old algorithm good". It is a measurement of **where the gains in recent selector papers actually live**. If a 1993 algorithm with no tuning sits inside one point of a 2026 purpose-built rule, then the subset-selection rule is not where those papers won — the scorer, the prompt, and the pipeline are. The paper makes this uncomfortably concrete: two carefully controlled harnesses running the *same published rules* at the *same budget* with the *same encoder* disagree by 0.07 to 3.74 points, which covers most of the gaps published selector comparisons are built on.

## The Methodology

**The shared scorer.** Videos decoded at 1 fps. Every candidate frame and the question stem encoded once with LongCLIP, embeddings cached. Every selector reads the same cache, so no rule gets a better encoder than another. This is the whole point — see [[On the Difficulty of Evaluating Baselines]].

**The prompt boundary, and what it costs.** By convention (following the official AKS code) the scorer sees only the *question stem*, not the multiple-choice options. They measured the price. Scoring against question+options changes 53.3% of OMP's picks on the 600 s bin and gives **.6699 vs .6311** ($+3.88$, $p{=}.033$). A pre-registered replication on the 3600 s bin agrees in sign ($+2.30$, $p{=}.18$). A control arm swapping in *another video's* options — content-irrelevant, length-matched — recovers 56% of that gain, so the effect is not about the options being correct. Reading: the text you hand the scorer is an uncontrolled axis worth 2–4 points, and papers inconsistent about it are measuring the prompt, not the selector.

**OMP, exactly.** Let $E \in \mathbb{R}^{N\times d}$ be L2-normalised frame embeddings and $q_0$ the normalised stem embedding. At step $r$:

$$b_r = \operatorname*{arg\,max}_{i\notin B_{r-1}} e_i^\top q_{r-1}, \qquad q_r = q_0 - \operatorname{Proj}_{\operatorname{span}\{e_j : j\in B_r\}}(q_0)$$

Pick the frame most correlated with the residual, then project away the span of everything already picked. In signal processing the residual is supposed to shrink as you go. Whether that survives in a contrastive image–text embedding space is treated as an empirical question — and §5.5 shows it does not (below).

**The six rules compared** at $k{=}8$: uniform sampling, cosine top-$k$, AKS, FOCUS\*, OMP, LDDR-select. Two caveats the paper flags itself: AKS and FOCUS natively score with BLIP-ITM, so those rows are *subset rules under a substituted scorer*; FOCUS\* additionally replays the published clip-bandit schedule against dense LongCLIP scores rather than the original budgeted online process. LDDR-select is the only row run with the encoder its authors used, and it is stage 1 (the Linear-DPP selector) only.

**Compression.** Frames resized *before* the processor, with Qwen3-VL's internal resize disabled so theirs is the only one. The main arm is a residual-proportional schedule at a mean spatial fraction of ~0.53 ("D@53"). Two controls: a flat split at the same mean (does OMP's *ranking* make a good pixel-priority?), and a reconstruction of LDDR's stage-2 Group-DPP importance on identical timestamps.

**Reinvestment.** $k{=}16$ timestamps with a ~50% per-frame cap, compared against $k{=}8$ at full resolution. Crucially they **audited the tokens** rather than assuming the resize ratio transfers: a script reproduces Qwen's smart-resize rules from each video's own dimensions. The 16-frame arm costs **0.996×** and **0.984×** the 8-frame arm on the 600 s and 3600 s bins. Both under one, so the winning arm is also the cheaper one.

**Setup.** LongVideoBench ($n{=}1337$, stratified into 15/60/600/3600 s bins), Video-MME ($n{=}2700$), LVBench ($n{=}1549$). Primary answerer Qwen3-VL-8B-Instruct via lmms-eval, greedy decoding, temperature 0, subtitles off. Transfer checks on InternVL3-2B/8B and GPT-5-mini.

**Statistics.** Exact two-sided McNemar tests on paired correctness. Where they claim two arms are *interchangeable*, they use two one-sided tests (TOST) with 90% CIs, because failing to find a difference is not evidence of none.

> [!NOTE] TOST (two one-sided tests)
> An equivalence test. Instead of asking "is the difference nonzero?", it asks "is the difference provably inside $\pm\delta$?" You need it whenever your claim is "these are the same", which a non-significant $p$-value never establishes. ^tost

## Ablation Studies and Experiments

**Selection (Table 2, $k{=}8$, Qwen3-VL-8B).** Accuracy, with the gap from OMP:

| Rule | LongVideoBench | Video-MME | LVBench |
|---|---|---|---|
| Uniform | .5654 ($-5.69$) | .5637 ($-5.85$) | .3454 ($-11.81$) |
| Top-$k$ | .6028 ($-1.95$) | .5704 ($-5.18$) | .4319 ($-3.16$) |
| **OMP** | .6223 | .6222 | .4635 |
| AKS | .5916 ($-3.07$) | .6059 ($-1.63$) | .4287 ($-3.48$) |
| FOCUS\* | .5819 ($-4.04$) | .5578 ($-6.44$) | .3983 ($-6.52$) |
| LDDR-select | **.6320** ($+0.97$) | .6193 ($-0.29$) | **.4693** ($+0.58$) |

OMP vs uniform is decisive on Video-MME ($p{=}5.1\times10^{-10}$) and LVBench ($p{=}9.5\times10^{-17}$). LDDR-select is the only rule that keeps pace, with no detectable paired difference from OMP on two of three benchmarks.

**When selection matters at all.** On 15 s clips, uniform, top-$k$ and OMP give *exactly the same* accuracy (.7249) — eight frames out of fifteen candidates cover the video however you pick. The gain is $+7.8$ at 600 s and $+7.5$ at 3600 s. It **switches on** when the candidate pool exceeds the budget; it does not keep growing with duration. (Honest caveat: the bins differ in content too, so this is observational, not a controlled truncation.)

**Scorer swap (§5.2).** Replace LongCLIP with SigLIP-so400m, hold everything else. This changes **67–84%** of the frames the answerer sees (the two scorers agree on 1.32 of 8 frames for OMP). Yet the ordering is untouched: uniform < top-$k$ < OMP under both. Top-$k$ lands on .6068 under *both* scorers; OMP moves .6311 → .6408 ($p{=}.73$). The test is not powerful — the 90% interval on the scorer effect spans $[-3.6, +5.6]$ — so this bounds *large* scorer effects rather than proving invariance.

**The statistical trap they explicitly refuse.** OMP beats top-$k$ significantly under SigLIP ($+3.40$, $p{=}.049$) and not under LongCLIP ($+2.43$, $p{=}.33$), which invites "OMP suits SigLIP". The interaction test rejects it: difference of gains 0.97 points, $p{=}.68$. Appendix B records the same error made on question categories. *A significant result next to a non-significant one is not an interaction.*

**Compression (§5.3).** Holding timestamps fixed and halving the pixel budget moves Qwen accuracy by at most **0.44 points** on any of the three benchmarks. Bounded properly: on pooled LongVideoBench long bins ($n{=}976$) the compressed arm is $+0.72$ points with 90% CI $[-0.60, +2.03]$ — inside $\pm3$ (TOST $p{=}.0022$) but failing $\pm2$ ($p{=}.054$). InternVL3-8B under a harsher cut (24 tiles → 8) gives $-0.10$ with CI $[-1.66,+1.45]$, clearing $\pm2$ ($p{=}.022$).

**What did *not* work:** the residual-proportional pixel schedule is indistinguishable from a flat 50% split at the same mean. OMP's ranking is a great priority for *choosing* frames and a useless priority for *distributing pixels* among them. A reconstructed LDDR stage-2 Group-DPP importance signal *does* beat the flat split by 1.84 points ($p{=}.0198$), so a better importance signal exists — it just is not OMP's rank order. Also failed: MMR underperforms; query-blind DPP ($\beta{=}0$) collapses toward uniform sampling; query-weighted DPP, residual floors and partial orthogonalisation all sit in a $\pm1.5$-point band around OMP with no significant winner.

**Reinvestment (§5.4).** Eight full-res frames vs sixteen compressed, at audited equal-or-lower token cost:

| Setting | $k{=}8$ full | $k{=}16$ compressed | $\Delta$ |
|---|---|---|---|
| Qwen / LongVideoBench | .6223 | .6447 | $+2.24$ |
| Qwen / LVB long pool | .5820 | .6055 | $+2.36$ ($p{=}.0346$) |
| Qwen / Video-MME | .6222 | .6378 | $+1.56$ |
| Qwen / LVBench | .4635 | .4939 | $+3.04$ ($p{=}.0009$) |
| GPT-5-mini / LVBench | .4900 | .5139 | $+2.39$ ($p{=}.039$) |

Does reinvestment *need* OMP? They ran all four cells of the selector × budget design in one environment. Uniform 8→16 gives $+1.13$ ($p{=}.37$); OMP gives $+2.56$. The interaction is 1.43 points and **not** significant ($p{=}.26$). With 203 discordant pairs this only excludes *large* selector-specific effects. Defensible statement: reinvestment works with OMP, points the same way under uniform, selector-specificity open at the one-point scale.

**The mechanism check that broke the story's own premise (§5.5).** OMP is supposed to reconstruct the query. It does not. The residual norm is **0.972** of its starting value after one pick and **0.967** after eight — almost none of the query is ever explained. What collapses is the residual's correlation with the frame being picked: **.233 at pick 1 → .004 at pick 8**, ~0 by 16. The query-directed signal is exhausted early. The cause is the **modality gap**: the text query sits nearly orthogonal to the cone of frame embeddings, so with an effectively low-rank frame dictionary the residual cannot shrink whatever you select. What OMP actually does here is *suppress directions already covered*. Relevance plus decorrelation, not sparse reconstruction.

**Failure audit.** 93 inspected OMP failures at $k{=}8$. Dominant pattern (20/41 at 600 s, 35/52 at 3600 s): **off-topic novelty** — picks going to visually distinctive but irrelevant material, unrelated chapters, title cards, dark transitions. Unconstrained diversity rewards being *different*, not being *useful*. The mirror failure also shows up: dense top-$k$ clusters on one relevant moment and misses a second required scene on cross-scene tracking questions. Design target: a **relevance floor**, so spread is explored inside plausible evidence regions. (Sample is failure-conditioned, category-balanced, model-assisted, unblinded — exploratory only.)

**Answerer transfer (§5.6).** Identical timestamps handed to InternVL3. OMP minus uniform: LVBench $+9.04$ (2B) and $+10.14$ (8B); LongVideoBench $+4.71$ / $+6.51$. Video-MME refuses: the 8B model gains $+5.44$ on short videos but $+0.33$ medium and $+0.11$ long, despite receiving demonstrably different frames. Selection gains are a property of the **answerer–benchmark pair**, not of the timestamp set.

## Worth Remembering

**The bug is the best part of the paper.** Their own AKS port had a padding branch: the recursive partition gives each leaf at depth $d$ a quota of $\lfloor k/2^d \rfloor$ frames, so at $k{=}8$ with the published depth of 5, every leaf gets **zero**, the recursion returns nothing, and the pad silently filled the whole selection from global cosine top-$k$. The AKS row was a duplicate of the top-$k$ row on every benchmark. Correcting it changed **~99.5% of selected frames** — and moved LVBench accuracy by **0.07 points**. A reader comparing accuracies alone would see two-hundredths of a point between a faithful implementation and one selecting entirely different frames on 99.5% of questions.

What caught it was not a per-arm sanity check. It was a **cross-arm overlap check on selected frame indices**: 98.5% agreement between two rules that optimise different objectives means at least one is not implementing its objective. Steal this. It generalises to any table whose rows are supposed to be distinct methods. Compare [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] and [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]].

**Harness disagreement is the second measurement.** Table 7 puts their numbers beside LDDR's published ones — same answerer family, same budget, same encoder. Their gains are smaller in 8 of 9 comparable cells, by 0.07 to 3.74 points. The uniform baselines themselves differ (LVBench: 28.08 published vs 34.54 theirs), so the two harnesses are not measuring from the same floor. That range covers most published selector deltas. Their own argument's numbers included.

**Honest caveats the authors volunteer:**
- Equivalence margins ($\pm2$, $\pm3$, $\pm4$) were chosen **after** seeing the data. So the result answers "which margin does this sample clear", not "does it fall inside a justified bound". Tests uncorrected for multiplicity.
- FOCUS\* is a schedule replay, not the FOCUS pipeline. LDDR-select is stage 1 only.
- MDP3 and Q-Frame are not run, because neither has a published number at this exact cell (Qwen3-VL-8B / 8 frames / LongCLIP), so neither port could be validated — the exact condition that let the AKS bug survive.
- Mechanism evidence is encoder-specific. Both LongCLIP and SigLIP share a contrastive image–text objective; an ITM head or an MLLM-attention scorer could reorder the table.
- Subtitles disabled throughout, which quietly depresses all arms on subtitle-anchored questions and may explain the unstable temporal-category result in Appendix B.
- Some secondary results ran on a second GPU stack; every arm of those contrasts was re-run there, and the drift is quantified (0.36 pt on one baseline, 0.71 on another) rather than assumed small.

**Practical ordering for anyone building this:** fix selection first, treat resolution as the slack variable, and *spend* the savings rather than pocketing them. Compression on its own buys nothing in accuracy — though it still buys latency and memory. The selection-stage cost (1 fps decode + encoding the whole pool with LongCLIP) is **not** counted in the token accounting, and a one-query-per-video deployment never amortises it.

**Open question worth chasing:** OMP works for a reason its derivation does not predict. If the win is really "relevance plus decorrelation" under a large modality gap, then a scorer with a *smaller* gap should change the picture entirely — and the residual trace gives you a cheap pre-check on whether a candidate scorer has more query signal to offer before you run a full evaluation. Connects to [[Representation Degeneration Problem in Training NLMs]] and [[How Contextual are Contextualized Word Representations]] on embedding cones.

## Links
Related: [[Keep-or-Drop- Adaptive Tokenizer for Compact Video Representation]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Attention Is All You Need]] · [[On the Difficulty of Evaluating Baselines]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Troubling Trends in Machine Learning Scholarship]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[Representation Degeneration Problem in Training NLMs]] · [[How Contextual are Contextualized Word Representations]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[A Simple Framework for Contrastive Learning (SimCLR)]] · [[LatentPress- Context Compression Beyond Text and Vision]] · [[Dense Passage Retrieval (DPR)]] · [[Calibrated Recommendations (RecSys)]]

New topics worth writing: Orthogonal Matching Pursuit, modality gap in contrastive image–text embeddings, maximal marginal relevance (MMR), determinantal point processes, McNemar's test, TOST / equivalence testing, LongCLIP, SigLIP and the sigmoid pairwise loss, LongVideoBench / Video-MME / LVBench, cross-arm overlap checks as an implementation-bug detector, harness variance in ML evaluation
