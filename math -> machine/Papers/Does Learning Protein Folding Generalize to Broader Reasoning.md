---
title: "Does Learning Protein Folding Generalize to Broader Reasoning?"
authors: ["Yong Liu", "Zhanpeng Shi", "Yizhou Dang", "Zhongyue Zhang", "Xiaoliang Shi", "Zhijian Wei", "Shuangjia Zheng"]
year: 2026
arxiv: "2609.38879"
url: https://arxiv.org/abs/2609.38879
priority: Good-To-Read
read_on: 2026-10-05
tags: [paper, llm, vision]
---
## The Core Idea

Language models learn from text people wrote. Text usually states conclusions, not the spatial logic behind them. So where do you get training data that is *about* 3D structure and is *automatically checkable*?

The answer here: solved protein structures. One protein in the Protein Data Bank is a list of 3D atom coordinates. From those coordinates you can compute, with a short program and zero human labelling, thousands of true/false facts — "do residues 23 and 147 touch?", "is residue 10 closer to the protein's centre than residue 88?", "is this four-atom arrangement left- or right-handed?". Every answer is exact. No annotator, no model, no judge.

The claim being tested is stranger than "train on proteins, get better at proteins". It is: **train a general model on protein geometry questions, then throw away every protein-specific part, and the model gets better at reasoning tasks that have nothing to do with biology.** Graph questions. Image-based spatial puzzles. Chemistry multiple choice. BIG-Bench Hard.

The headline: macro-average over 10 general benchmarks goes from **45.09% → 48.33%** (+3.23 percentage points), with a positive mean change on all 10.

Why this did not exist before: people building protein–LLM systems were trying to make the LLM *good at proteins*. Nobody detached the protein interface afterwards and measured what was left. And the data-centric post-training literature mostly makes synthetic logic puzzles or code — nobody treated a solved scientific archive as a generic supervision mine.

What it unlocks, if it holds: a solved science problem becomes a post-training corpus. Text is finite; the PDB is a different, non-linguistic, exactly-verifiable source.

> [!NOTE] FoldingCorpus
> A question–answer dataset built by running 12 deterministic programs over known protein backbone coordinates. 1,200 proteins × 12 questions = 14,400 records. Every answer is recomputed from the coordinates, so the labels are auditable rather than trusted. ^foldingcorpus

> [!NOTE] Structural decodability
> Whether 3D structure can be *read out* of a model's internal representations by a fixed decoder — not whether the model can predict structure well. Fold2Reason's TM-score of 0.169 is far below a real folding system; the number is a diagnostic, not a folding claim. ^structural-decodability

The honest version of the finding, which the authors state themselves: this is **behavioural transfer**, with model and task dependence. Whether it is new reasoning machinery, better *invocation* of machinery the base model already had, or partly answer-format adaptation, is unresolved.

## The Methodology

### The data: 12 programs over coordinates

Source is the OpenFold monomer short-protein set. For a protein of length $L$, let $Y \in \mathbb{R}^{L\times 4\times 3}$ be the backbone coordinates (N, CA, C, O per residue) and $c_i$ the $C\alpha$ position of residue $i$.

**Coordinates are never an input.** They only generate labels and losses. The model sees the amino-acid sequence, plus optionally an MSA (a stack of related sequences) or template hints.

The 12 operators, with their actual rules:

| Operator | Rule | Sampling |
|---|---|---|
| CONTACT_SHORT/MEDIUM/LONG/ANY | A iff $d(i,j) < 8$ Å | separation 6–11 / 12–23 / ≥24 / ≥6 |
| DISTANCE_ORDER_1/2 | A iff $d(i,j) < d(k,l)$ | pairs drawn from lowest and highest distance deciles |
| SEGMENT_ORIENTATION | A if $\cos(u,v)\ge 0.5$; B if $\le -0.5$; else C | segment starts ≥8 residues apart |
| NEARER_CENTER | A iff $\lVert c_i-\bar c\rVert < \lVert c_j-\bar c\rVert$ | two valid residues |
| LOCAL_FRAME_DIRECTION | A iff $(c_j-c_i)^\top(C_i - CA_i) \ge 0$ | separation ≥6 |
| CA_CHIRALITY | A iff $[(c_2-c_1)\times(c_3-c_1)]^\top(c_4-c_1)\ge 0$ | four consecutive $C\alpha$ |
| MULTI_CONSTRAINT | pick nearer candidate only if its distance $<8$ Å, else C | one anchor, two candidates |
| RETRIEVAL_32 | pick the right structural summary out of 32 | target + 31 hard negatives |

Nine are binary, two are three-way, one is 32-way. **Every answer option is a single vocabulary token** — verified against the tokenizer. So one protein's pack supervises exactly 12 answer tokens plus EOS.

The 32-way summary fingerprint is: sequence length, radius of gyration, dominant secondary-structure class, and eight long-range contacts quantised onto a $32\times32$ grid. Hard negatives are nearest neighbours in that feature space.

Anti-leakage details that matter: the intended answer class is chosen by a **salted hash**, then valid arguments are sampled for that class, then option order is independently permuted. So answer position is not a function of the operator. Observed label balance is near 50/50 everywhere (e.g. CONTACT_LONG: 447 A / 553 B).

Splits: 1,000 train / 100 dev / 100 frozen test, in **disjoint MMseqs2 clusters** (30% identity, 80% coverage). Residues with pLDDT < 70 are dropped from supervision in train/dev.

### The architecture

Base is Qwen3.5-9B, frozen. [[LoRA]] rank 16, alpha 32, dropout 0.05 → 43.28M trainable adapter parameters.

**Step 1 — protein forward.** The prompt has exactly one marker token per residue. Pull out those marker hidden states:

$$H = f_\theta(x)_{\text{res}} \in \mathbb{R}^{L\times 4096}$$

**Step 2 — the workspace.** A small 3.90M-parameter module $W_\phi$ squeezes $H$ to width 256, passes messages over pairs of residues, and returns two things:

$$(E, M) = W_\phi(H), \quad E\in\mathbb{R}^{L\times 4096},\quad M\in\mathbb{R}^{16\times 4096}$$

$E$ is residue-aligned. $M$ is 16 "evidence tokens" made by 16 learned queries pooling the residue set — a [[LoRA|prefix]]-style summary, in the spirit of Perceiver and prefix-tuning.

Pair selection matters: sequence offsets 1–4 (local backbone) plus evenly spaced longer-range indices, capped at 2,048 pairs. **Target contacts do not choose the edges** — otherwise the graph would leak the answer. Pair features are absolute differences and elementwise products of the reduced states, averaged at incident residues, then residual-updated.

**Step 3 — two readouts from the same $E$/$M$.**

Readout A, the discrete one. Prepend $M$ to the packed 12-question sequence $q$, and supervise only the 12 answer tokens and EOS:

$$\mathcal{L}_{\text{qa}} = -\frac{1}{|\mathcal{S}|}\sum_{t\in\mathcal{S}}\log p_{\theta,\phi}(y_t \mid M, q, y_{<t})$$

This goes through the model's **native language head**. Nothing new is bolted on.

Readout B, the continuous one. $E$ goes into a **frozen** coordinate + distogram decoder $g_\psi$ (3.31M parameters), giving

$$\mathcal{L}_{\text{geo}} = \mathcal{L}_{\text{coord}} + \mathcal{L}_{\text{pair}} + \mathcal{L}_{\text{contact}} + \mathcal{L}_{\text{dist}} + \mathcal{L}_{\text{local}} + \mathcal{L}_{\text{torsion}} + \mathcal{L}_{R_g}$$

Total objective: $\mathcal{L} = \mathcal{L}_{\text{qa}} + \mathcal{L}_{\text{geo}}$.

### The detail that makes the geometry loss do any work

Freezing $\psi$ means **excluding it from the optimiser, not detaching $E$**. Gradients still flow through the decoder into the workspace and the LoRA weights. If the decoder were trainable, a fresh coordinate head would absorb the whole geometry objective and the shared representation would learn nothing. The decoder is pretrained in a "Phase-0" head-only stage on the same 1,000 proteins with the base model and LoRA frozen.

Each training example therefore does **two forwards through the same LoRA-adapted Qwen** — the protein forward producing $H$, and the answer forward consuming $[M; \text{embed}(q)]$ — with the computation graph retained between them.

### Training and the evaluation trick

Three epochs, 375 optimizer steps, 4× A800-80G, ~32 minutes per seed. Four workers × 2 microsteps of one protein = 8 proteins per update. LoRA LR $10^{-4}$, workspace LR $3\times10^{-4}$, weight decay 0.01, 5% warmup, cosine decay, grad-norm clip 1.0, [[Decoupled Weight Decay Regularization (AdamW)|AdamW]]. Seeds 20260729 / 20260803 / 20260804. Checkpoint always epoch 3 / step 375 — **no benchmark-based selection**.

Then the key move: **at transfer evaluation, $W_\phi$ and $g_\psi$ are thrown away.** No protein input, no workspace, no decoder. Just base Qwen + the LoRA adapter on its normal benchmark interface. Any gain has to live in the adapter.

### The benchmarks

General-10 is an unweighted macro over FTB-Core (their own 12,000-question text benchmark for 3D reasoning), SpatialViz (images), VSI (video), GraphQA Easy and Hard, BBH, ChemBench, ChemBench4K, Lab-Bench, SciBench. 76,725 examples total. Greedy decoding throughout, no few-shot.

$$\Delta_{\text{G10}} = \frac{1}{10}\sum_{d=1}^{10}\left[s_d(\text{adapter}) - s_d(\text{Base})\right]$$

The unit of replication is **an independently trained adapter**, not an evaluation run. Three seeds, mean ± sample SD.

## Ablation Studies and Experiments

### The main number and its spread

45.09% → 48.33%, SD 0.15pp. Per-seed: +3.19, +3.12, +3.40pp. Biggest movers: GraphQA Hard +6.83, SpatialViz +6.10, GraphQA Easy +4.56, ChemBench4K +4.37.

Robustness reaggregations, all post-hoc:
- Drop both GraphQA sets → +2.62pp over 8 datasets.
- Drop FTB-Core (their own benchmark) → +3.14pp on "External9".
- Drop FTB-Core *and* SpatialViz → +2.77pp over the remaining 8.

So it is not one benchmark carrying everything.

### The controls — this is the most important table

Three matched Pure-LoRA controls, same base model, same 12-question packing, same 375 steps, same seeds:

| Arm | What it changes | G10 |
|---|---|---|
| Base | — | 45.09 |
| **Hidden Geometry** | real protein prompts, answers computed from an *unobserved random* point structure (IID cloud / polymer chain / clustered shape) matched on length, mask and radius of gyration | +0.92 |
| **Format Copy** | donor answer codes appended, model trained to copy them | **−0.09** |
| **Fixed Shuffle** | real prompts, fixed donor labels matched by operator and vocabulary | **−0.27** |
| Fold2Reason | real structure, both losses | **+3.23** |

Read this carefully. Copying valid answer codes buys nothing. Learning a fixed wrong input→label mapping buys nothing. Internally *coherent but input-independent* geometry buys about a quarter of the effect. Real structure buys 3.5× the strongest control.

Caveat the authors flag: Fixed Shuffle preserves label marginals rather than forcing disagreement, so **42.85% of its "shuffled" labels happen to coincide with the true answer**. And Format Copy is really a format + copying + answer-prior control, not a pure format control.

### Component ablation — which loss does what

| Arm | G10 | 3D macro | FTB | SpatialViz | VSI | Text G7 |
|---|---|---|---|---|---|---|
| Base | 45.09 | 40.44 | 36.18 | 26.61 | 58.53 | 47.09 |
| w/o FoldingCorpus (geometry-dominant) | +0.68 | +1.03 | +2.26 | +0.79 | +0.05 | +0.52 |
| w/o Geometry | **+2.93** | +2.86 | +2.81 | +5.20 | +0.54 | **+2.96** |
| Full | +3.23 | +3.94 | +4.08 | +6.10 | +1.63 | +2.93 |

The discrete question-answering is doing nearly all the broad transfer. Geometry adds +0.30pp overall — and **the per-seed differences are +0.57, +0.49, −0.15pp**, so the sign flips. That increment is honestly not resolved.

But it is not spread evenly. Relative to w/o Geometry, Full gains +1.26pp on FTB-Core, +0.90 on SpatialViz, +1.08 on VSI (3D macro +1.08pp), while Text G7 moves −0.03pp. **Geometry buys 3D-specific behaviour, not general reasoning.**

Structural readout goes the other way. Full vs w/o Geometry: lDDT-$C\alpha$ +0.0097, Contact F1 +0.0063, but TM-score −0.0021 and $C\alpha$ MAE **worse** by 0.148 Å. Local structure becomes more recoverable; global topology does not. Per-protein: lDDT improves for 75.4% of proteins, Contact F1 for 78.7%.

Folding scores in absolute terms (×100): Fold2Reason 16.88 TM / 25.32 lDDT / 6.90 Contact F1, versus the unadapted Qwen3.5-9B asked to emit coordinates directly at 4.55 / 9.38 / 3.05. That is the 2.7–3.5× in the abstract. It is also nowhere near a real folding system — [[Mastering the game of Go with deep neural networks (AlphaGo)|this is not AlphaFold]], and the paper says so.

### Model scale and family

| Model | G10 gain | per-seed |
|---|---|---|
| Qwen3.5-2B | +5.41 | all positive |
| Qwen3.5-4B | +4.84 | all positive |
| Qwen3.5-9B | +3.22 | all positive |
| InternVL3.5-8B | +1.53 | all positive |
| **Gemma-4-12B-IT** | **+0.07** | −0.69, +0.51, +0.40 |

Smaller models gain more. InternVL proves it is not a Qwen artefact. **Gemma is flat and its t-interval spans zero** — real family dependence, unexplained.

### Data scaling

Seven nested subsets, 50 → 4,000 proteins, all trained 3 epochs (so steps scale with data: 21 → 1,500):

| Proteins | 50 | 100 | 250 | 500 | 1K | 2K | 4K |
|---|---|---|---|---|---|---|---|
| G10 Δ | 0.49 | 0.61 | 1.37 | 2.92 | 3.31 | **3.70** | 3.02 |
| FC acc | .478 | .486 | .485 | .492 | .501 | **.546** | .418 |
| Contact F1 | .0666 | .0671 | .0666 | .0675 | .0694 | .0739 | **.0768** |

Peaks at 2,000. Note the dissociation at 4K: one seed's answer-token calibration **collapses** (held-out FoldingCorpus accuracy 0.128 vs 0.568 for another seed) and yet its General-10 gain is still +3.14pp. Downstream transfer is not determined by source-task accuracy. They kept the broken seed in the mean, which is the right call.

### What did not work

**Label density is flat.** Hold proteins at 1,000 and steps at 375, vary labels per protein between 3, 6, 12: gains are +3.39, +2.95, +3.31pp. Slope against $\log_2 q$ is −0.040pp per doubling, 95% interval $[-0.814, 0.734]$. **Asking more structural questions about the same proteins does not help.** Breadth of proteins matters; density of labels does not. (Though the protein-count curve scales coverage *and* compute together, so that mechanism isn't isolated either.)

**Pure LoRA beats Full on the 3D macro** — 44.54% vs 44.38%. The workspace and geometry decoder, in aggregate, do not pay for themselves. The 3D-specific geometry benefit is *conditional on keeping the workspace*.

**Lab-Bench and ChemBench barely move**, and at 4,000 proteins both go slightly negative (−0.39, −0.12). VSI route planning −1.89pp and absolute distance −0.85pp get worse.

### The audit that nearly undoes the spatial claim

This is the most intellectually honest appendix I have read in a while.

FTB-Core's +4.08pp is **74.98% one task**: sparse-constraint candidate selection, 32.50% → 69.17%. Remove it and the other 11 tasks average +1.11pp. SpatialViz's +6.10pp is **54.17% one task**: CubeAssembly, 1.25% → 50.00%.

So: is this parser recovery? They checked.

- FTB-Core: 12,000/12,000 valid outputs for Base *and* every Full seed. Zero parse failures either side. The sparse-selection gain is not recovered formatting.
- But Base picks A or B on 946/1,000 sparse-selection questions despite balanced labels. **Answer-selection bias is a live alternative explanation.**
- SpatialViz: Base has 20 parse failures (1.69%), Full has zero. On the 1,160 commonly-valid items the gain is still +5.46pp; the 20 recovered items account for only 12.04% of the headline gain.
- CubeAssembly: Base picks D on 79/80 items, and **no reference answer is D**. 2DRotation: Base picks D on 74/80, scores 0.00%. Those are valid wrong answers, not rejected strings. A more permissive sensitivity parser recovers zero extra correct answers.
- The 20 Base failures all re-encode to the 128-token cap — length-limit truncation, concentrated in MechanicalSystem (11), CubeCounting (8), CrossSection (1).

Their own conclusion: output-format recovery, correction of a response bias, and improved visual inference must not be conflated. **They do not claim to have separated them.**

### Checkpoint timing

At step 5, the gain is +0.016pp. At step 125 it is +3.12pp, at step 375 +3.31pp. Continuous growth in every seed of both arms; no instant plateau. That rules out "the model instantly learned the prompt format". It does not rule out *gradual* format adaptation, and the authors say so.

## Worth Remembering

**Leakage they found and did not hide.** Six scaling-training proteins contain template constraints sourced from FoldBench targets (8ec3, 8ey3, 8g64, 8t9n, 8uds, 9etn). A Smith–Waterman audit at ≥30% identity / ≥80% coverage found 12 train–dev, 14 train–test, 3 train–FoldBench, 3 dev–test, 1 dev–FoldBench matches. They did **not** rerun after discovering this and explicitly refuse to call the FoldBench curves homology-disjoint generalisation. The core-training vs FoldBench screen is only exact-sequence + 5-mer Jaccard (max 0.02381) — full MMseqs2, deposition-date, CATH and SCOP isolation are unverified.

**FTB-Core is their own benchmark.** Program-audited answers, deterministic generator, zero exact-prompt overlap between splits — but "prompt clarity lacks separately logged human validation", and semantic near-duplicates under a shared generator remain possible. Hence the External9 number (+3.14pp) is the one to quote.

**The workspace is doing something protein-specific.** Swap in another protein's workspace and TM-score drops 0.0104. The geometry decoder is reading residue-level information, not a generic prior.

**Two evaluation contracts exist and the base scores differ.** The canonical study scores VSI over all 5,130 questions; the scaling study uses VSI's official 8-task macro. GraphQA Easy base is 65.14 in one and 67.71 in the other. This is why the 1,000-protein scaling endpoint is +3.31pp while the canonical run is +3.23pp. Every gain is computed inside its own contract — which is correct, but means cross-table arithmetic is not safe.

**Gemma needed response seeding.** Its SpatialViz and VSI wrappers prefill the assistant turn with `<answer>` and `Final answer:` because the native chat interface kept writing explanations. The prefixes contain no answer content and apply to Base and adapters alike — but there is no with/without ablation, and Gemma is also the one neutral family. Possibly related, possibly not.

**Practical caveats if you wanted to use this.** The whole thing is 32 GPU-minutes on 4×A800 for a 9B model with rank-16 LoRA — astonishingly cheap for a +3pp macro. But: 2,000 proteins is the sweet spot, more labels per protein is wasted effort, and whether it works on *your* base model is a coin flip you can only resolve by running it. Smaller models gain more, which is consistent with "this is making latent capability more reliably accessible" rather than "this is installing new capability" — though that reading is my inference, not a measured result.

**Open questions worth chasing.** (1) Fixed-choice decoding and balanced option permutations would separate reasoning from answer-selection bias on the spatial tasks — the authors name this as the required next experiment. (2) Why does Hidden Geometry get +0.92pp at all? Internally-consistent-but-fictional geometry transferring is a strange result and deserves its own paper. (3) The sign-flipping geometry increment needs more than three seeds. (4) Nobody has tried this with [[Proximal Policy Optimization Algorithms|RL against verifiable rewards]] instead of plain cross-entropy, which is the obvious next move given the labels are exactly checkable.

**Where this sits.** It is the data-centric sibling of [[Chain-of-Thought Prompting Elicits Reasoning in LLMs|CoT]]-era work asking what supervision builds general capability — closer to synthetic-logic-corpus papers than to protein-LLM alignment work like ProteinGPT or 3D-MoLM, which measure protein ability and keep the protein interface. The honest framing the authors land on: structure-dense scientific data *can* improve broad reasoning, in a model- and task-dependent way, and a solved scientific problem is a usable post-training resource. They deliberately do not claim to have found reusable reasoning computation.

## Links

Related: [[LoRA]] · [[Fine-Tuning]] · [[Attention Is All You Need]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Instruction Tuning]] · [[Exploring the Limits of Transfer Learning (T5)]] · [[Distilling the Knowledge in a Neural Network]] · [[Shortcut Learning in Deep Neural Networks]] · [[Troubling Trends in Machine Learning Scholarship]] · [[Evals]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Cross Entropy]] · [[Chain of Thought]] · [[The Bitter Lesson (essay)]]

New topics worth writing: verifiable-reward post-training, AlphaFold and protein structure prediction, MMseqs2 sequence clustering, TM-score and lDDT, distogram prediction, Perceiver-style learned-query pooling, benchmark contamination auditing, answer-selection bias in multiple-choice evaluation, parser-failure audits for LLM benchmarks, synthetic logic corpora for reasoning transfer, cross-domain generalisation tax in post-training
