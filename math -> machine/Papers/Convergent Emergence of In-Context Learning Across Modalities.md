---
title: "Convergent Emergence of In-Context Learning Across Modalities"
authors: ["Breslow et al."]
year: 2026
arxiv: "2609.14011"
url: https://arxiv.org/abs/2609.14011
priority: Good-To-Read
read_on: 2026-09-19
tags: [paper, llm, optimization, vision]
---
## The Core Idea

Few-shot in-context learning (ICL) — showing a model a handful of input→output pairs in the prompt and having it infer the rule — is treated as a language-model thing. This paper asks whether that is an accident of English text or a general property of next-token prediction on any rich sequence data.

The trick is to build **one abstract task suite** and then re-encode it into six different worlds. The abstract task is always a function on 8-bit strings, $f:\{0,1\}^8 \to \{0,1\}^8$ — reverse the bits, rotate left by one, swap the halves, and so on. Then each modality gets an encoding map $\phi_M$ that turns a bitstring into something that modality's model can eat: random digits for a language model, random nucleotides (A/C/G/T) for a genome model, random amino acids for a protein model, colour-cluster pixels for an image model, signed half-sine lobes for a time-series forecaster.

Because the *function* is held fixed and only the *costume* changes, you can ask a new question: **do the same abstract tasks get easy or hard in the same places across modalities?** That is the Convergent Emergence Hypothesis.

> [!NOTE] Convergent Emergence Hypothesis
> If one shared mechanism drives ICL regardless of training corpus, then the per-task benefit from having correct input→output pairings should be *correlated across modalities* — the same abstract functions help everywhere. ^convergent-emergence

Why this could not exist before: earlier work showed ICL in one non-language modality at a time (vision sequences, genomes), each with its own bespoke task set. You cannot compare "genome ICL" to "protein ICL" if they were measured on different tasks. The shared suite is the whole methodological contribution.

What it unlocks, if true: prompting techniques built for LLMs transfer to protein, genome and forecasting models — domains where you *cannot* write an instruction in words, so few-shot examples are the only steering wheel you have. This is the same "condition the model without gradient updates" idea as [[In Context Learning]], but for models that have no language to be instructed in.

The second, sharper finding is a **separation of emergence from convergence**. ImageGPT clearly does ICL (25.2 point gap over control), but its per-task profile does *not* line up with the others. Getting the capability does not mean getting the same shape of the capability.

## The Methodology

**The abstract tasks.** 100 functions on 8-bit strings: 30 single primitives (`identity`, `reverse_bits`, `rotl1`, `swap_halves`, `majority`, `minority`, `parity_fill`, `flip_bits`, `center_mask`, …) and 70 compositions, where $f \to g$ means $g(f(x))$. Input and output space are both $\{0,1\}^8$, so $|\mathcal{I}| = 256$.

**The prompt.** For $n$ shots the model sees

$$\text{prompt}_M = \phi_M(I_1)\circ\phi_M(O_1)\circ\psi_M \circ \cdots \circ \phi_M(I_n)\circ\phi_M(O_n)\circ\psi_M\circ\phi_M(I_{n+1})$$

with $O_i = f(I_i)$ and $\psi_M$ a modality-specific separator. The model generates the continuation, which is decoded back to 8 bits with $\phi_M^{-1}$. Scoring is **exact match on all 8 bits** — a malformed or wrong-length output is simply wrong. $I_{n+1}$ is held out, never in the demonstrations.

**The control, which is the heart of the paper.** A model can score well without inferring anything. If $f(x) = x \bmod 2$, guessing "0" always gets 50%. So every cell is run twice: clean ($P_n$) and **deranged** ($Q_n$), where the outputs are permuted across the demonstrations:

$$Q_n = \big((I_1, O_{\rho(1)}), \dots, (I_n, O_{\rho(n)})\big)$$

The derangement is not a uniform random permutation. They sample up to 256 permutations and keep the one minimising $\sum_i \mathbf{1}[O_i = O_{\rho(i)}]$ — the one that breaks the most pairings — stopping early at zero. This preserves the *multiset* of outputs (so the output distribution, the format, the label space are all unchanged) while destroying the *mapping*.

> [!NOTE] Deranged control
> Shuffle which output goes with which input, keeping the same set of outputs. Anything the model gains from clean-minus-deranged is genuinely from the input→output relationship, not from noticing "the answers around here tend to be all-ones". The headline metric is $\Delta_n = \frac{1}{|F|}\sum_f (X_{f,\text{clean},n} - X_{f,\text{deranged},n})$. ^deranged-control

**Output identifiability.** A second confound: what if the demonstrations genuinely do not pin down the answer, so the model is punished for a consistent guess? Appendix B derives a union bound. For a rival $g$ agreeing with $f$ on all $n$ demos but differing on the query,

$$\Pr(A_{f,g,n}) = \frac{d(f,g)}{m}\cdot\frac{\binom{m - d(f,g)}{n}}{\binom{m-1}{n}}$$

with $m = 256$ and $d(f,g)$ the Hamming disagreement count over the input space. Mean pairwise distance is ~244/256, so this decays geometrically. Identifiability is $\geq 0.871$ at $n=4$, $\geq 0.991$ at $n=8$, effectively 1 from $n=16$. So a missing clean-deranged gap is the model's failure, not the task's ambiguity.

**The six modalities and their encodings.**

| Modality | Model | Encoding $\phi_M$ | Separator $\psi_M$ |
|---|---|---|---|
| Language | Qwen3 base, 0.6B–14B | random digit 0–9 for 1, another for 0 | a third disjoint digit |
| Genome | Evo2, 1B–40B | random nucleotide for each bit | a third nucleotide |
| Integer seq. | NextTerm 47M / 440M (trained here on OEIS) | random digit 1–9 per bit | literal comma |
| Image | ImageGPT small/medium/large, 76M–1.4B | random colour-cluster token per bit, 16-pixel span per demo | left-padded 16-pixel span of a third colour |
| Time series | TimesFM 2.5, 200M | bit = signed 16-step half-sine sub-lobe, $+s_{16}$ for 1, $-s_{16}$ for 0; two per 32-step patch | one all-zero patch |
| Protein | ProGen2, 151M–6.4B | random canonical amino acid per bit | a third residue |

The symbol assignment is **re-sampled every trial** and only consistent within a trial. That is deliberate: encoding language bits as literal `0`/`1` would let Qwen3 lean on memorised bit-twiddling from pretraining. Random digits make the task out-of-distribution for everyone.

**Decoding details that matter.** ImageGPT and ProGen2 use *constrained* decoding — at each output position only the two trial-specific bit-symbol logits are compared, the winner is fed back autoregressively. This is the same move as [[Structured Output|constrained decoding]]: do not beg the model for valid output, restrict the sampler. TimesFM averages the forecast for the context with the sign-corrected forecast for its negation, enforcing sign-flip invariance; each 16-step span is then projected onto $\pm s_{16}$.

**Evaluation protocol.** $T = 128$ trials per (task, model, shot count, condition). Shot sweep $N = \{1,2,4,8,16,32,64\}$, reduced for context-limited models (ImageGPT tops out at 31, ProGen2 at 48, TimesFM at 48). Error bars are task-cluster standard errors over the 100 task accuracies. Significance is an **exact one-sided within-task condition-swap test** — under the null, clean and deranged labels are exchangeable within a task, so enumerate all $2^m$ sign assignments (after dropping zero differences) and count how many beat the observed mean. Six modality-level tests, Holm-adjusted.

## Ablation Studies and Experiments

**Main result — ICL emerges in all six.** At each modality's max shot count:

| Modality | Model | $n_{\max}$ | Clean | Deranged | Gap (pp) [95% CI] |
|---|---|---|---|---|---|
| Language | Qwen3-14B | 64 | 30.0% | 16.1% | 13.9 [10.8, 17.1] |
| Genome | Evo2-40B | 64 | 33.2% | 15.2% | 18.0 [15.0, 21.0] |
| Integer seq. | NextTerm-440M | 64 | 48.1% | 16.7% | 31.4 [25.6, 37.6] |
| Image | ImageGPT-large | 31 | 43.5% | 18.3% | 25.2 [19.5, 31.3] |
| Time series | TimesFM-2.5 | 48 | 32.6% | 14.0% | 18.5 [15.1, 22.0] |
| Protein | ProGen2-base | 48 | 26.4% | 16.5% | 9.9 [8.2, 11.7] |

All six survive Holm adjustment with $p \leq 1.6\times10^{-19}$. Clean accuracy climbs with shot count; deranged stays substantially flatter.

**The convergence result.** Build a 100-dimensional profile per modality (per-task clean-minus-deranged at max shots), then Spearman-correlate the profiles pairwise. The five non-image modalities correlate at $\rho = 0.35$–$0.89$. Genome and protein are the tightest at $\rho = 0.89$ — which is a nice sanity check, since DNA and amino-acid sequences are literally two codings of the same biology.

**What breaks convergence: images.** ImageGPT correlates with genome, protein, integer sequences and time series at only $\rho = 0.06$–$0.15$, though moderately with language at $\rho = 0.44$. So ICL emerged but the difficulty profile is idiosyncratic. This is the finding the authors flag as most interesting, and it is the reason the hypothesis is only *partially* supported.

**Semantic clusters.** The 100 tasks are grouped into $k=7$ behavioural families by clustering on *truth tables*, not syntax. Directional variants (`rotl1` vs `rotr1`, `left_half` vs `right_half`) are merged into "semantic orbits" first, collapsing 99 functions into 62 orbits; distance is a Gaussian kernel on a custom edit distance $\delta$ (substitute a bit, circular shift either way, complement the string), then classical MDS and $k$-means. The clusters: Global Broadcast (30), Parity Masking (17), Local Shifts (14), Half Segments (13), Bit Inversion (10), Half Swaps (9), Bit Reversal (7).

The per-cluster profiles differ sharply. ImageGPT dominates Bit Reversal (0.66) and Local Shifts (0.61) — plausibly because its raster encoding gives it spatial priors for reversing and shifting a pixel row. NextTerm, Evo2 and ProGen2 peak on Half Segments; Qwen3 peaks on Global Broadcast; TimesFM peaks on Local Shifts.

**Scale.** Within a family, the paired gap generally grows with parameter count — but not reliably. Monotone for ImageGPT and NextTerm. Qwen3 rises through 8B then plateaus at 14B. **Evo2 peaks at 7B, not 40B.** **ProGen2 peaks at its 764M `base` config and then *declines* for `large` and `xlarge`.** Scale helps; scale does not guarantee. This is a useful counterweight to the usual [[Scaling Laws for Neural Language Models|power-law]] story — the capability is not a smooth function of parameters.

**What did not work — chess.** ChessGPT-50M (a 16-layer Lichess-trained model), with 4-bit strings encoded as two-bit knight-move signals followed by deterministic reset moves so the board returns to its start. Accuracy *does* rise with shot count, but the clean-minus-deranged gap at $n=8$ is $-0.0016$, $p = 0.640$. Shuffling the pairings changes nothing. The model is adapting to the output distribution, not the mapping — exactly the confound the derangement control was built to catch. They also screened one-bit knight toggles, one-bit signal-reset moves, explicit separator moves, and layouts that reset the game between examples; none produced separation.

**What barely worked — music.** Musicroll-50M, a 50M byte-level model trained for 1B tokens on ~24,000 solo-piano MIDI performances (MAESTRO v3, GiantMIDI-Piano, ATEPP), with bits encoded as pitches in 125 ms piano-roll slices. Early shots are non-significant ($p \geq 0.098$). At $n = 32$: clean 0.1732 vs deranged 0.1625, gap 0.0107, $p = 0.00349$, Holm-adjusted 0.0174. Statistically real, practically nothing — $9.3\times$ smaller than the *weakest* primary modality (ProGen2's 0.0991).

**The ablations' real message.** The clean/deranged split is doing all the epistemic work. Without it, chess looks like it has ICL (accuracy rises with shots!) and you would report a false positive. Raw accuracy is the wrong instrument; the paired gap is the instrument.

## Worth Remembering

**Absence of evidence.** The authors are explicit: you can prove ICL emerges in a modality, but you cannot prove it does not. A failed elicitation might just be a bad prompt format. Chess and music are weak evidence, not refutations. Their suggested fix — put a prior on emergence per modality and update on failed elicitations — is left as future work. This asymmetry haunts any elicitation-based claim, and it is worth internalising for anything that measures "can model X do Y".

**You cannot rank modalities.** NextTerm-440M's 48.1% clean accuracy does not mean integer sequences are "better at ICL" than protein at 26.4%. Each model's pretraining priors bias which encodings suit it — ImageGPT's raster layout is basically a gift for bit-reversal tasks. Priors are partially controlled *within* a modality (same corpus, same architecture) and not at all *across*. The one attempt at control is encoding language bits as random digits rather than `0`/`1`, to stop Qwen3 recalling bit-twiddling from pretraining. Generalising that — "encodings that prevent contaminated prior leakage" — is flagged as an important open problem.

**Meta-ICL is explicitly excluded.** Training a model on few-shot prompts and then testing few-shot prompts is a different phenomenon. It is task-specific: a model meta-trained on linear and cosine functions fails on their composition. This paper is only about ICL falling out of plain next-token prediction, which is why the modality criterion demands naturally occurring pretraining data. Two modalities partially violate this — TimesFM includes synthetic series, and OEIS obviously contains function-induction-flavoured data — acknowledged but judged acceptable.

**Theoretical bite.** Explanations of ICL that lean on properties specific to human language — "parallel structures" in text, linguistic compositionality — are directly challenged by a shared effect structure across genomes, proteins and time series. Accounts based on properties that many natural sequence distributions share (burstiness, compositionality generally) survive.

**Relation to the Platonic Representation Hypothesis.** That line argues internal representations converge across modalities with scale. This is the external-behaviour version: task *difficulty profiles* converge. Complementary, and notably the image case diverges here in a way the representation work would not predict.

**Unexplained anomalies the authors flag.**
- ProGen2's non-monotonic scale curve is much sharper than any other family's.
- Deranged accuracy itself behaves differently across architectures: rising monotonically for NextTerm, but rising through 8 demos and then *receding* for Evo2. Possibly an artefact of their non-uniform derangement selection, untested.
- NextTerm-47M vs 440M differ in scale, vocabulary (15 vs 16 symbols), architecture, context length (2048 vs 40,960) and training corpus simultaneously — so that gap is not a controlled scale comparison and should not be read as one.

**Practical caveats if you wanted to use this.** Everything is deterministic bitstring manipulation; real few-shot problems are noisy and stochastic, and noisy regression/classification is untouched. Task difficulty is treated as a bag-of-functions with no explicit difficulty scale, so "which tasks are hard" is correlational not causal. And there is no theory that predicts *at what scale* or *for which modality* ICL will appear — the practically valuable thing, since it would tell you how long to train and on what.

**Honest disclosure.** Appendix A states LLMs were used heavily in implementation and drafting, and that "the vast majority of code released for reproducibility is LLM-generated". Worth knowing before building on the released repo.

## Links
Related: [[In Context Learning]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Auto-regressive models]] · [[Foundation Models]] · [[Scaling Laws for Neural Language Models]] · [[Structured Output]] · [[Embeddings]] · [[Prompt Engineering]] · [[Evals]] · [[Tokenization]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Context Window]] · [[Shortcut Learning in Deep Neural Networks]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Sampling Parameters]]

New topics worth writing: Platonic Representation Hypothesis, meta-ICL and learning-to-learn, induction heads, permutation and randomisation tests, Holm correction and multiple comparisons, Spearman rank correlation, genomic language models (Evo2), protein language models (ProGen2), time-series foundation models (TimesFM, Chronos), ImageGPT and pixel-level autoregression, OEIS integer-sequence modelling, function identifiability and union bounds, classical multidimensional scaling, elicitation-based capability evaluation
