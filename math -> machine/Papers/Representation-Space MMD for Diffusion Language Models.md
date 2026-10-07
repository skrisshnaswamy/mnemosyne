---
title: "Representation-Space MMD for Diffusion Language Models"
authors: ["Ilya Drobyshevskiy", "Ilia Sudakov", "Maksim Semenov", "Denis Kuznedelev", "Maksim Ignatov", "Pavel Temirchev", "Nikita Balagansky", "Viacheslav Meshchaninov", "Nikita Gushchin", "Dmitry Baranchuk"]
year: 2026
arxiv: "2610.06648"
url: https://arxiv.org/abs/2610.06648
priority: Good-To-Read
read_on: 2026-10-07
tags: [paper, llm, rl, diffusion]
---
## The Core Idea

Diffusion language models (DLMs) are trained with a loss that only looks at **one token at a time**. A masked DLM predicts, for each hidden position separately, a distribution over which token should go there. A continuous DLM predicts the mean of the clean latent. Both are *marginal* objectives: they get each position roughly right on average, but nothing in the loss ever asks "is this whole sequence, decoded in one shot, any good?"

That is exactly why few-step sampling is bad. If you reveal many tokens at once, you are sampling them independently from marginals that were never trained to agree with each other. Take many small steps and the model can patch things up as it goes; take a few big steps and you get word salad.

The fix here: add a **distribution-level** loss. Generate a batch of sequences, take a batch of real sequences, and push the two *distributions* together using Maximum Mean Discrepancy.

> [!NOTE] Maximum Mean Discrepancy (MMD)
> A way to measure how far apart two distributions are using only samples, no densities. Pick a kernel $k$ (a similarity score between two points). Then compare the average within-group similarity to the average across-group similarity. If the two groups are drawn from the same distribution, those averages match and MMD is zero. ^mmd-def

$$\operatorname{MMD}^2_k(P,Q)=\mathbb{E}_{x,x'\sim P}[k(x,x')]+\mathbb{E}_{y,y'\sim Q}[k(y,y')]-2\,\mathbb{E}_{x\sim P,y\sim Q}[k(x,y)]$$

The problem with MMD on raw text is that there is no good kernel on token strings, and raw latent distances in high dimensions mean very little. So the real contribution is **where** MMD is computed: in the hidden states of a *frozen copy of a pretrained DLM*. Feed real and generated sequences through that frozen model, grab layer-6-ish activations, compute the kernel there.

Two reasons this is clever and cheap:

1. **No auxiliary model to train.** Competing distribution-matching methods (DiDi-Instruct, IDLM, D-MMD) all need a second network trained *alongside* the generator to track its drifting output distribution. Here the feature extractor is frozen and is just the pretrained model you already have.
2. **One forward pass gives many samples.** Instead of pooling a sequence into one embedding, they keep the hidden state at *every* scored position. A sequence of length 512 with 300 masked positions yields 300 feature vectors from one extractor pass. MMD estimates are sample-hungry; this is where the samples come from. It also means you can train with **one reference sequence per prompt**, which matters because real datasets give you exactly one answer per question.

What it unlocks: on 16B DMax models, **tokens-per-forward goes up 10–16%** on math with equal-or-better accuracy, and code accuracy rises by 2.4–3.8 points — after **400 training steps, ~1.7 GPU-hours on 8×H100**. That is a post-training stage cheap enough to be an afterthought.

## The Methodology

### The feature space

A frozen pretrained DLM $\phi$ maps a sequence to per-position hidden states:

$$\phi(\mathbf{x}) = (\phi_1(\mathbf{x}),\dots,\phi_L(\mathbf{x})) \in \mathbb{R}^{L\times D}$$

Features are taken from **clean** inputs (no diffusion noise) at a single intermediate layer — layer 4 for OWT masked, layer 7 for TinyGSM masked, layer 6 for ELF, layer 10 for 16B DMax. Corrupting the extractor input did not help.

The kernel is Gaussian RBF, $k(a,b)=\exp(-\|a-b\|^2/2\sigma^2)$. Bandwidth $\sigma$ is tuned per setup and varies wildly: 160 (OWT masked), 90 (GSM8K masked), 10–16 (DMax), 3330 (ELF on GSM8K). This is the main fragile hyperparameter.

$S$ = the set of scored positions. Masked positions for masked DLMs; response positions for prompted generation; all positions for unconditional latent generation.

### Kernel between two sequences

Average the token-level kernel over all cross-position pairs:

$$\kappa_\phi(\mathbf{x},\mathbf{x}') = \frac{1}{|S||S'|}\sum_{i\in S}\sum_{j\in S'} k(\phi_i(\mathbf{x}),\phi_j(\mathbf{x}'))$$

This is an inner product of empirical kernel mean embeddings, so it is itself a valid positive-semidefinite kernel on sequences.

### The estimator, and one subtle correctness point

MMD needs **independent** draws inside each expectation. Tokens inside the same sequence are *not* independent — they were generated together and the extractor saw them in context. So the within-group sums exclude all same-sequence pairs, including pairs at different positions:

$$\widehat{\mathcal{L}}_{\mathrm{MMD}} = \frac{1}{B(B-1)}\sum_{b\neq b'}\big[\kappa_\phi(\mathbf{x}^r_b,\mathbf{x}^r_{b'}) + \kappa_\phi(\mathbf{x}^g_b,\mathbf{x}^g_{b'})\big] - \frac{2}{B^2}\sum_{b,b'}\kappa_\phi(\mathbf{x}^r_b,\mathbf{x}^g_{b'})$$

So $B$ sequences × $|S|$ positions give you $B|S|$ observations without pretending they are $B|S|$ independent draws.

Reading the three terms:
- **real–real**: no $\theta$ in it, dropped during optimisation. This is why one reference per prompt is enough.
- **real–generated** (coefficient $-2$): attraction. Be similar to real features.
- **generated–generated** (coefficient $+1$): **repulsion**. Be dissimilar to your own other samples. This is the anti-[[Mode Collapse|mode-collapse]] term and it needs $B\geq 2$.

The conditional training loss (one reference, $B$ generations) is:

$$\widehat{\mathcal{L}}_{\mathrm{train}}(\theta\mid c)=\frac{1}{B(B-1)}\sum_{b\neq b'}\kappa_\phi(\mathbf{x}^g_b,\mathbf{x}^g_{b'})-\frac{2}{B}\sum_b \kappa_\phi(\mathbf{x}^r,\mathbf{x}^g_b)$$

### Discrete DLMs: policy gradients

Tokens are sampled categorically, so you cannot differentiate through them. Instead treat $-\widehat{\mathcal{L}}_{\mathrm{train}}$ as a reward and use [[Simple Statistical Gradient-Following Algorithms (REINFORCE)|REINFORCE]].

The training loop for one step:
1. Take a clean sequence, corrupt it to $\mathbf{x}_t$, pick the scored set $S$.
2. **One** denoiser forward pass gives $q_\theta(\mathbf{x}_i \mid \mathbf{x}_t)$ for all $i\in S$.
3. Draw $G$ independent *groups*, each of $B$ sequences, all from that one forward pass. So $G\times B$ sequences, one network call.
4. Each group $g$ gets reward $r_g = -\widehat{\mathcal{L}}_{\mathrm{train}}(\mathbf{x}^r, \mathbf{X}_g \mid \mathbf{x}_t)$.
5. Leave-one-out baseline across groups: $A_g = r_g - \frac{1}{G-1}\sum_{h\neq g} r_h$.
6. Surrogate loss:

$$\mathcal{L}_{\mathrm{PG}} = -\frac{1}{G}\sum_{g=1}^{G} A_g \sum_{b=1}^{B}\sum_{i\in S}\log q_\theta(\mathbf{x}_{g,b,i}\mid \mathbf{x}_t)$$

Note the structure: the reward is a property of the *whole group*, so it multiplies the summed log-prob of every sequence in that group. This is the same leave-one-out trick as [[GRPO]] / RLOO — the other groups are the baseline, which is valid because they are independent given the shared reference and $\mathbf{x}_t$.

The key efficiency point: **no sampling trajectory is ever unrolled**. One denoiser pass plus one extractor pass per training step.

Two instantiations:
- **MDLM-MMD** — masked-style [[Discrete Diffusion|discrete diffusion]]. $S$ = masked positions only.
- **DMax-MMD** — hybrid masked + uniform diffusion. MMD replaces *both* the masked loss and the "prediction" loss (where the model conditions on its own previous token predictions and must revise them).

Hyperparameters: $G=4$, $B=2$, AdamW, lr $10^{-4}$ (OWT) / $10^{-5}$ (TinyGSM), global batch 512, 3,250–7,000 steps. For 16B DMax: $G=8$, lr $2\times10^{-6}$, 400 steps.

### Continuous DLMs: differentiate straight through

With continuous latents you do not need [[Policy Gradient|policy gradients]] — the samples are differentiable. They start from a pretrained **ELF** (Embedded Language Flows) model, which runs [[Flow Matching]] in the frozen latent space of a text encoder (T5-small or GPT-2), and turn it into a one-step generator:

$$\mathbf{x}^g = G_\theta(\mathbf{z}, t=0, \mathrm{SC}=\mathbf{0}),\qquad \mathbf{z}\sim\mathcal{N}(\mathbf{0},\mathbf{I})$$

Features come from the frozen ELF evaluated at the clean end of the schedule, with the latent fed into *both* its inputs:

$$\phi(\mathbf{x}) = \phi_{\mathrm{ELF}}(\mathbf{x}, t=1, \mathrm{SC}=\mathbf{x})$$

Gradients flow through both of those inputs back into $G_\theta$. No REINFORCE, no variance problem.

**Multi-step sampling via self-conditioning.** ELF has a self-conditioning input $\mathrm{SC}$ — a previous estimate of the clean latent. They exploit it as a refinement loop at *fixed* diffusion time $t=0$:

$$\hat{\mathbf{x}}^{(k)} = G_\theta(\mathbf{z}^{(k)}, t=0, \mathrm{SC}=\hat{\mathbf{x}}^{(k-1)}),\quad k=1,\dots,K$$

starting from $\hat{\mathbf{x}}^{(0)}=\mathbf{0}$. Noise is either held fixed or resampled each iteration. One final pass at $t=1$ with null SC converts the latent to logits; argmax gives tokens. Total $K+1$ network calls. $K=1$ is the pure one-step model.

**Bootstrapping.** At inference step 3 the model sees a *predicted* latent as SC; during naive training it only ever saw $\mathbf{0}$. Classic train/test mismatch. Fix: sample the number of warm-up refinements uniformly from $\{0,\dots,n-1\}$, run them with fixed noise and **no gradient**, then apply MMD to one more pass and backprop only through that. Costs $n$ forward passes but only one backward.

**IRD (Iterative Refinement Distillation).** Optional stage *after* MMD. Freeze the trained ELF-MMD as teacher, roll it out $K$ steps with fixed noise, and train a student to hit the final output in one shot:

$$\mathcal{L}_{\mathrm{IRD}}(\theta)=\mathbb{E}_{\mathbf{z}}\big[\|G_\theta(\mathbf{z},t{=}0,\mathrm{SC}{=}\mathbf{0}) - \bar{\mathbf{x}}^{(K)}\|_2^2\big]$$

Plain [[Distilling the Knowledge in a Neural Network|knowledge distillation]], and they report that fancier alternatives ([[Consistency Models|consistency distillation]], FMLM⋆'s objective) gave no noticeable gain.

Also: a decoder fine-tuning head runs alongside, reconstructing tokens from perturbed latents with [[Cross Entropy|cross-entropy]], costing ~20% more batch. Optimiser is [[Old Optimizer, New Norm- An Anthology (Muon)|Muon]], lr $5\times10^{-5}$.

## Ablation Studies and Experiments

### Unconditional, OpenWebText

Metric is generative perplexity under GPT-2 Large, reported at matched unigram entropy. The entropy matching matters: you can always lower gen-PPL by lowering temperature until the output is boring, so they interpolate curves at the **data entropy $H\approx5.43$** rather than treating higher entropy as better.

| Masked, 8 steps | gPPL | Ent. |
|---|---|---|
| MDLM (base) | 282.5 | 5.45 |
| DiDi-Instruct | 149.0 | 5.44 |
| IDLM | 56.2 | 5.44 |
| IDLM-REINFORCE | 52.9 | 5.41 |
| **MDLM-MMD** | **43.5** | 5.43 |

At 16 steps: 26.8 vs IDLM's 33.8. At 32: 21.5 vs 29.3. Roughly **17–21% lower gen-PPL than IDLM** across budgets.

The IDLM-REINFORCE row is the cleanest control in the paper: it uses IDLM's log-ratio reward with *this paper's* policy-gradient optimiser. It lands between IDLM and MDLM-MMD, so the gain is **mostly the objective, not the optimiser**.

Continuous side, T5 encoder: ELF-MMD+IRD at 8 steps gets gPPL 47.3 / entropy 5.39, against ELF-PD's 63.7 / 5.39. At 4 steps IRD cuts ~30 gPPL points off ELF-MMD (110.1 → 78.1).

### Conditional, GSM8K (trained on TinyGSM)

Masked: MDLM-MMD reaches **~54% accuracy at ~49 average steps**, above IDLM, DiDi-Instruct and MDLM at moderate-to-high budgets. (Confidence-threshold decoding makes step count adaptive, so they sweep the threshold and report average steps.)

Continuous, GPT-2 Small latents:

| Steps | 4 | 8 | 16 | 64 |
|---|---|---|---|---|
| ELF | 6.0 | 13.8 | 23.3 | 31.6 |
| ELF-PD | 16.0 | 23.5 | 27.2 | 28.5 |
| ELF-GAN | 9.7 | 20.3 | 29.1 | 34.4 |
| ELF-MMD | 14.2 | 27.5 | 34.2 | 35.2 |
| **ELF-MMD + IRD** | **20.8** | **32.5** | **35.6** | **36.3** |

Note ELF-PD *degrades* past 16 steps (27.2 → 28.5 → 28.5) while MMD keeps climbing. And the one-step column is the one place MMD loses: ELF-PD gets 2.24% at a single step vs ELF-MMD's 0.71%. Progressive distillation is still better at the absolute extreme.

MMD also transfers onto other continuous backbones: ELF⋆-MMD beats ELF⋆ at every budget, FMLM+-MMD beats FMLM+ at every budget (e.g. 9.1 vs 5.9 at 4 steps).

### 16B scale

Starting from released DMax-Math and DMax-Coder (both tuned from LLaDA2.0-Mini), 400 steps, ~1.7–2.5 GPU-hours.

| | GSM8K TPF | MATH500 TPF | Minerva-Alg TPF | ASDIV TPF |
|---|---|---|---|---|
| LLaDA-2.0-mini | 2.04 | 2.58 | 3.01 | 2.03 |
| DMax-Math | 5.48 | 5.94 | 7.03 | 5.62 |
| **DMax-Math-MMD** | **6.15** | **6.84** | **8.19** | **6.20** |

Accuracy over those four: 92.1 / 76.0 / 92.1 / 92.9 vs DMax's 92.1 / 75.4 / 91.5 / 92.5 — equal or slightly better, with 10–16% more tokens per forward.

Code is a bigger win on accuracy: HumanEval-Instruct 85.9 (from 83.5) with TPF 8.07 (from 7.36); MBPP-Instruct 83.0 (from 79.2) with TPF 6.10 (from 5.86). Both axes improved simultaneously.

Worth noting dParallel SFT actively *hurts* code accuracy (84.2 → 76.8) while buying parallelism. MMD does not pay that tax.

### The loss ablation — what is actually doing the work

Four variants compared against token-level RBF:

1. **Linear kernel** — equivalent to MSE between generated and reference feature *means*. This is essentially the EBFT / feature-matching objective. Worse than RBF everywhere.
2. **Sequence-level RBF** — mean-pool features over positions, then one RBF per sequence. Consistently worse than token-level. So keeping the per-token observations is not just a sample-count convenience; it provides a genuinely better signal.
3. **Attraction-only** — drop the generated–generated repulsion term. Noticeably worse for discrete models, and for continuous models it **collapses outright** on OWT. This is the sharpest negative result: the repulsive term is load-bearing, exactly as [[Mode Collapse]] theory would predict. An attraction-only objective in feature space has a trivial optimum — emit the same good-looking sequence forever.
4. **Feature regression** — MSE between the full intermediate feature tensors at corresponding positions. Low accuracy, badly so for continuous models. Matching features *pointwise* is not the same as matching their distribution.

Ordering on OWT at every budget: token RBF > sequence RBF > linear ≫ attraction-only, feature regression.

### Representation-space ablation

- Discrete: frozen-DLM hidden states beat (a) raw vocabulary embeddings — which carry no context at all — and (b) features from a frozen **autoregressive** LM trained on the same data.
- Continuous: frozen-ELF features beat (a) the raw encoder latents themselves ($\phi = \text{identity}$) and (b) a separately trained masked-objective Transformer on ELF latents.

So the gain is specifically from *diffusion-model* contextual features, not from "any contextual features".

### Other ablations

- **Group size $G$**: the only meaningful jump is $1\to2$, which is when the leave-one-out baseline becomes available. $G\in\{2,4,8\}$ are indistinguishable. Variance reduction is the whole benefit; more samples is not.
- **Bootstrap steps**: one step helps (T5 OWT, GSM8K); more steps give limited or inconsistent gains, and GPT-2 results are mixed. Each step costs a forward pass.
- **Model size**: ELF-M-MMD > ELF-M at every budget, so the method survives scaling B→M. But the smaller **ELF-B-MMD beats the larger plain ELF-M at 4–16 steps** — MMD post-training buys more than one model-size step at low budgets.
- **Shifted timestep schedule** (shift 32 for ELF, 128 for ELF-PD) was needed to make the continuous baselines competitive at all. The authors applied it to the baselines, which is good practice.

## Worth Remembering

**Why this is the interesting framing.** The marginal/joint gap is the central pathology of parallel decoding — see `^parallel-tokens-are-independent` in [[Discrete Diffusion]]. Most fixes are architectural or scheduling tricks. This one changes the *loss* to score joint samples, and does it by delegating "what makes a sequence coherent" to a frozen network's internal representations. It is the text analogue of perceptual loss in vision.

**What the authors admit they do not have.** The identifiability argument is honest but incomplete. A characteristic kernel through an *injective* feature map gives zero population MMD iff the original distributions agree (the MMD-GAN argument), and there is recent work suggesting causal transformer representations are almost-surely injective. But they explicitly state this does **not** establish that matching their *token-feature* distributions pins down the sequence distribution. Treat the method as a good training signal, not a consistent estimator. **Open research question.**

**Bias they knowingly accept.** Keeping same-sequence token pairs in the within-group sums would bias the estimator, so they exclude them — but they note those comparisons "may still be useful in practice". Unexplored.

**Practical caveats if you wanted to use this:**

- **RBF bandwidth $\sigma$ is the hyperparameter that will cost you a week.** It ranges over three orders of magnitude across their setups (10 to 3330) with no stated selection rule. Feature-layer choice (4, 6, 7, 10) is also hand-picked per model.
- $B=2$ is tiny for an MMD estimate. It works because token-level features multiply the observation count, but the *sequence-level* independence structure is still only pairwise. This is probably where remaining noise lives.
- The repulsion term is not optional. If you simplify to an attraction-only feature-matching loss because it is cleaner to implement, continuous models collapse.
- **400 steps, 1.7 GPU-hours on a 16B model.** The cost structure here is nothing. If you run a DLM in production, this is a near-free post-training stage worth trying before anything more elaborate.
- Multiple benchmark numbers are averaged over 5 seeds with standard deviations given — rare and good. The 16B table reuses the original DMax paper's baseline numbers, but they state they reproduced them with the official implementation.

**Connections.** The discrete update is structurally identical to [[GRPO]]/RLOO with a non-verifiable reward, which raises the obvious [[Reward Hacking]] question: the reward is a frozen network's kernel similarity, and nothing stops the generator from finding inputs that look right to layer 7 of MDLM without being good text. The entropy-matched reporting is a partial guard against this; it is not a proof. The ELF side is cleaner because gradients are exact — no reward-model intermediary, just differentiation through a frozen feature map, the same shape as [[Distillation]] with a perceptual target.

**Follow-ups the authors flag.** Combining features across layers or noise levels; an ensemble of extractors; combining MMD with other discrete objectives (motivated by IRD stacking cleanly after ELF-MMD).

**A question they do not answer.** The method needs reference sequences, so this is distribution matching against *data*, not against a *teacher*. For the 16B runs the "data" is LLaDA-2.0-mini's own generated responses to math and code prompts. So in practice it was self-distillation on model output, not matching the real data distribution — which makes the gains somewhat easier to believe and somewhat less exciting.

## Links

Related: [[Discrete Diffusion]] · [[Flow Matching]] · [[Diffusion Models]] · [[Mode Collapse]] · [[GRPO]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[Policy Gradient]] · [[Distillation]] · [[Distilling the Knowledge in a Neural Network]] · [[Consistency Models]] · [[Perplexity]] · [[Cross Entropy]] · [[KL Divergence]] · [[Old Optimizer, New Norm- An Anthology (Muon)]] · [[Reward Hacking]] · [[Generative Adverserial Network]] · [[Evaluating Generative Models]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Rectified Flow]] · [[Score-Based Generative Modeling through SDEs]] · [[Latent Variable Models]]

New topics worth writing: Maximum Mean Discrepancy, kernel mean embedding, kernel two-sample test, characteristic kernels, MDLM (masked diffusion language models), LLaDA and large-scale diffusion LLMs, self-conditioning in diffusion, progressive distillation, flow map matching, distribution matching distillation (DMD), generative moment matching networks, perceptual / feature-matching losses, generative perplexity as a metric and its failure modes, leave-one-out baselines for policy gradients, confidence-threshold parallel decoding, tokens-per-forward as an efficiency metric
