---
title: "Last Translation Benchmark"
authors: ["Vilém Zouhar", "Niyati Bafna", "Mukund Choudhary", "Maike Züfle", "Sara Rajaee", "Pinzhen Chen", "Jannis Vamvas", "Sara Papi", "Ona de Gibert", "Bhavitvya Malik", "Eliya Habba", "Orfeas Menis Mastromichalakis", "Patrícia Schmidtová", "Michelle Wastl", "Sheriff Issaka", "Leshem Choshen", "Stella Biderman", "Antonis Anastasopoulos", "Jan Niehues", "Rico Sennrich"]
year: 2026
arxiv: "2609.04173"
url: https://arxiv.org/abs/2609.04173
priority: Good-To-Read
read_on: 2026-09-16
tags: [paper, llm, rl, vision]
---
## The Core Idea

Machine translation benchmarks have run out of headroom. On standard test sets like WMT or FLORES, the best systems now score close to the ceiling, and the differences between the top few systems are smaller than the noise in the measurement. When a test cannot separate the leaders, it stops being a test.

Two things break at once when that happens:

1. **The data is too easy.** Sentences pulled from news articles are exactly the kind of text these models were trained on. They are not where the models fail.
2. **The ruler is bent.** Automatic metrics — BLEU counts overlapping word n-grams, COMET is a neural model trained to predict human quality scores — give you one number with no explanation. Worse, a neural metric is a learned scorer, and anything learned can be gamed. You can raise a COMET score without raising translation quality. That is reward hacking, the same failure that shows up in [[Training language models to follow instructions with human feedback|RLHF]].

Human evaluation is the usual fallback, but it is slow, expensive, and two annotators often disagree about the same sentence. So it does not reproduce.

The Last Translation Benchmark attacks both problems with one design choice: **build the test set adversarially, and ship each example with its own hand-written grading rules.**

The adversarial part means every example was written or found by a human who first checked that current top models *get it wrong*. If a leading model translates it correctly, it does not go in. This is the same recipe as "Humanity's Last Exam" — the benchmark is defined by the frontier's failures, not by a corpus sample.

The verification-rule part is the more interesting half.

> [!NOTE] Verification rule
> A concrete, checkable statement attached to one example that describes a specific way a translation can be wrong. Not "is this good, 1–5?", but something closer to "the output must not render *X* as *Y*", or "the second occurrence of the pronoun must be feminine". Grading becomes a set of pass/fail checks rather than a subjective score. ^verification-rule

That swap is what makes the evaluation both *reproducible* (two people applying the same rule agree) and *actionable* (a failure tells you which linguistic phenomenon broke, not just that the number went down). A score of 62 tells you nothing. "Failed 8 of 9 gender-agreement rules and 1 of 12 idiom rules" tells you what to fix.

The benchmark is also **multimodal** — text, images, audio, and video — because translation in the real world includes subtitles, signage, and speech, and because a model that reads a menu photo has more ways to fail than one that reads a sentence.

Finally, it is a **live dataset**. Contributions stay open, and releases are cut by date. LTBv1 is everything accepted before 1 September 2026. This is a deliberate hedge against the benchmark itself saturating — when models catch up, you add harder examples rather than writing a new paper.

## The Methodology

> [!WARNING] What is actually available
> The source text here is the arXiv abstract page only. The full PDF holds the concrete counts, language pairs, model names, and rule taxonomy. What follows is the pipeline as the abstract describes it; treat the finer mechanics as unverified until you read the PDF.

The construction loop, as stated:

**Step 1 — a human authors a candidate example.** Source material in one of four modalities: a text passage, an image, an audio clip, or a video. Contributors are drawn from the paper's author list, which is enormous (roughly 350 names) and spans a very wide set of languages and institutions. That breadth is the point — you cannot build a hard multilingual test with a small, monolingual team. The crowd *is* the method.

**Step 2 — the example must break a leading model.** The contributor runs current state-of-the-art translation systems on it. If they all handle it, it is not interesting and is discarded. This is a filter on difficulty, applied before any human review of quality.

**Step 3 — the contributor writes verification rules.** For that specific example, they enumerate the ways the translation can go wrong. These are handcrafted, example-specific, and describe *concrete failure cases*. This is the labour-intensive part and the reason the benchmark cannot be scraped or auto-generated.

**Step 4 — peer review.** Every submission is reviewed before acceptance. This is the quality gate that separates this from a plain crowdsourced collection: someone other than the author checks that the example is genuinely hard, that it is correct, and that the rules are well-formed.

**Step 5 — versioned release.** Accepted contributions are frozen at a cutoff date. LTBv1 = accepted before 2026-09-01. More releases follow as the pool grows.

Evaluating a new system means: run it on the examples, then check its outputs against each example's rules. The score is the fraction of rules satisfied. Because the rules are explicit, you can aggregate them by category and read off *which* phenomenon a model is weak on.

The structural comparison worth holding in your head: this is **rubric-based evaluation**, the same shape as the pointwise rubrics in [[Small Language Models as Judges for Rubric-Based Reinforcement Learning]] and the dynamic rubrics in [[DRACO- Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training]], except the rubrics here are written by humans per example and never learned. That is a deliberate trade: no scale, but no reward model to hack.

## Ablation Studies and Experiments

This is a dataset-and-protocol paper, not a modelling paper, so there is nothing to ablate in the usual sense — there is no architecture, no loss, no training run. What would normally be the experiments section is instead the *construction statistics* and the *model leaderboard*, and neither survives into the abstract.

What the abstract does claim, and what you should verify against the PDF:

- **Leading MT models fail on these examples.** This is true by construction — Step 2 filters for it — so it is not evidence of anything. The real question is *how badly*, and whether models fail on the same examples or different ones. If failures are uncorrelated across systems, the benchmark is measuring idiosyncratic gaps; if correlated, it is measuring a shared blind spot in the training distribution.
- **Rule-based verification is more reliable than metrics or human scores.** For this to be more than an assertion you want inter-annotator agreement on rule application versus agreement on direct quality scores. That number is the load-bearing evidence for the whole design, and it is not in the abstract.
- **Metrics are vulnerable to reward hacking.** Asserted as motivation, not demonstrated here. The supporting literature is elsewhere.

What is missing and matters: no baseline comparison against simply *hard-filtering* an existing test set (take FLORES, keep only the sentences the top model gets wrong, evaluate with COMET). That is the cheap version of this idea, and the argument for hand-written rules has to beat it. Also missing from the abstract: how many examples, how many languages, what the modality split is, and how many rules per example.

The honest read is that the contribution is the **protocol plus the human effort**, and the empirical validation of the protocol is in the PDF or not at all.

## Worth Remembering

- **Adversarial construction has a known cost.** When you keep only the examples the current frontier fails, you build in a dependency on *which* models you filtered against. Next year's model may fail on a different set. This is the same shape as the concern in [[Do ImageNet Classifiers Generalize to ImageNet]] — the test set carries the fingerprint of the process that made it. The live-dataset design is the mitigation, not a fix.

- **Per-example rules do not scale, and that is the trade.** Every example needs a human to imagine its failure modes. You get reproducibility and diagnosis; you give up ever having 100k examples. Contrast with automatically generated verifiable environments like [[AgentMercury- Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale]], which scale but only in domains where correctness is machine-checkable. Translation is not one of those domains — there is no compiler for a Czech sentence.

- **This is the "unverifiability tax" argument applied to MT.** [[Agentic Game Development as a Verifiable Trajectory Data Engine for Scaling World Models]] makes the point that domains without an automatic checker progress slower. Translation has been paying that tax for two decades. Handwritten rules are an attempt to manufacture a checker by hand, one example at a time.

- **Saturation is a benchmark-lifecycle problem, not an MT problem.** The same story ran in recommender systems ([[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]]), in speech ([[Towards Quantifying Benchmark Optimization in ASR Models]]), and in baselines generally ([[On the Difficulty of Evaluating Baselines]]). A benchmark that everyone optimises against eventually measures optimisation pressure instead of capability.

- **Practical caveat if you want to use this.** A rule-based score is not comparable to a BLEU or COMET number, and it is not comparable across versions of the benchmark either, since v2 will contain harder examples than v1. Report the version. Treat it as a diagnostic instrument — "where does my system break" — rather than a single leaderboard number.

- **Open question.** Can the verification rules be applied by an LLM judge rather than a human? If yes, the benchmark becomes cheap to run and the human cost is a one-time authoring cost. If no — if judges disagree with the rule authors — then the reproducibility claim only holds for human graders, and the scalability argument collapses. The rubric-judge literature suggests this is plausible but fragile.

- **Nearly 350 authors** is itself a finding about how multilingual evaluation has to be built now. No single lab has native speakers of enough languages. The paper is closer to a consortium than a study.

## Links

Related: [[Attention Is All You Need]] · [[Seq2Seq models]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[On the Difficulty of Evaluating Baselines]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[Towards Quantifying Benchmark Optimization in ASR Models]] · [[Troubling Trends in Machine Learning Scholarship]] · [[Small Language Models as Judges for Rubric-Based Reinforcement Learning]] · [[DRACO- Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training]] · [[Training language models to follow instructions with human feedback]] · [[Agentic Game Development as a Verifiable Trajectory Data Engine for Scaling World Models]] · [[AgentMercury- Your Agent Can Synthesize Verifiable Environments for Business Scenarios at scale]] · [[Shortcut Learning in Deep Neural Networks]] · [[SWE Refactor Bench- Can Coding Agents Complete a Long-Horizon, Whole-Repository Stack Migration]] · [[Thinking in a Low-Resource Language- What SFT Builds, What RL Fixes, What Accuracy Cannot See]] · [[Foundation Models]]

New topics worth writing: BLEU, COMET and learned MT metrics, Benchmark saturation, Adversarial benchmark construction, Humanity's Last Exam, WMT shared task, FLORES-200, Inter-annotator agreement, Reward hacking, Multimodal translation, Speech translation
```
