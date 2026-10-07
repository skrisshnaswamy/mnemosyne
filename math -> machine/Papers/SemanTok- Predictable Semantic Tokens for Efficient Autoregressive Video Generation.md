---
title: "SemanTok: Predictable Semantic Tokens for Efficient Autoregressive Video Generation"
authors: ["Mikhail Dereviannykh", "Vikram Voleti", "Simon Donne", "Mallikarjun Byrasandra Ramalinga Reddy", "Shimon Vainer", "Mark Boss"]
year: 2026
arxiv: "2610.00686"
url: https://arxiv.org/abs/2610.00686
priority: Good-To-Read
read_on: 2026-10-05
tags: [paper, transformers, llm, diffusion, vision]
---
## The Core Idea

Video generation here is a two-stage job. A **tokenizer** turns a clip into a short list of discrete symbols. An **autoregressive model** then predicts those symbols one by one, like a language model predicting words, and a diffusion decoder paints them back into pixels.

Most tokenizers give every clip the same fixed grid of tokens. A still shot of a wall and a car chase both cost the same. **Flexible-length** tokenizers fix this: the tokens form one ordered list, coarse first, fine later, and you can cut the list anywhere. Cutting at $k$ tokens per latent frame still decodes to a watchable video.

> [!NOTE] Flexible-length coarse-to-fine tokenizer
> A tokenizer whose output is a single ordered sequence, where every prefix of that sequence is independently decodable. Early tokens set the broad content, later tokens add detail. The budget $k$ is chosen at inference time by the application, not fixed at training time. ^flexible-coarse-to-fine

The question this paper asks: **what should live in those first few tokens?**

The prior system, VideoFlexTok, tried to push semantics early using [REPA](https://arxiv.org/abs/2410.06940) — a loss that makes an early decoder layer match frozen [[DINOv2- Learning Robust Visual Features|DINOv2]] features. The flaw is simple and the paper's best observation. That decoder layer sees **two** inputs: the token prefix *and* a partially noised video latent. At low noise levels, the noised latent already looks like the clip. So the decoder can satisfy the DINO target from the latent and largely ignore the tokens. The loss gets minimised; the tokens learn nothing. Measured: at noise $\sigma{=}0.25$, VideoFlexTok's readout moves from $0.716$ at $k{=}1$ to $0.719$ at $k{=}256$. Two hundred and fifty-five extra tokens bought $0.003$ cosine. The tokens were not the source.

SemanTok removes the escape route. It adds heads that read **only the retained token prefix** — no noised latent anywhere — and must reconstruct the DINO features from it. Now the only way to lower the loss is to put semantics in the tokens.

What this unlocks: the early prefix becomes a cheap, semantic, *predictable* code. A 201M AR model on SemanTok tokens matches or beats a VideoFlexTok AR model $3.4\times$ larger. At $k{=}16$ the prefix costs $32\%$ fewer bits per token to predict. The content of the clip is settled early and cheaply; pixel detail is deferred to the tail and partly filled in by the generative decoder.

The trade is explicit: SemanTok reconstructs *worse* in PSNR (about 2.7 dB lower at $k{=}16$ on uCO3D) but generates *better*. That is the [compression–generation trade-off](https://arxiv.org/abs/2412.16326) applied at the level of a prefix rather than a whole code.

## The Methodology

Both systems share everything except the semantic supervision. Same backbone, same codebook, same decoder, same AR recipe, same data, same compute. This is a clean A/B.

### The frozen front end

A frozen VidTok VAE maps an RGB clip $x$ (17 frames, $128\times128$) to latents $z \in \mathbb{R}^{T\times h \times w \times C_z}$ with $T=5$, $h=w=C_z=16$. So 17 RGB frames become 5 latent frames; the VAE compresses time causally by $4\times$.

Each latent frame has $P = h \times w = 256$ patch positions.

### VideoFlexTok's encoder (the shared base)

Each latent patch is lifted to encoder width $d_e = 1152$:

$$\mathbf{e}_{t,p} = W_{\text{in}}\mathbf{z}_{t,p} + \mathbf{b}_{\text{in}}$$

The encoder also holds $K = 256$ **learnable register tokens** $\mathbf{r}_{t,i}$. Index $i$ is the coarse-to-fine position. Each frame packs patches first, then registers:

$$\mathcal{S}_t = (\mathbf{e}_{t,0},\ldots,\mathbf{e}_{t,P-1}, \mathbf{r}_{t,0},\ldots,\mathbf{r}_{t,K-1})$$

Attention is [[Causal Attention|time-causal]] — frame $t$ sees only frames $\le t$. Inside a frame, patches attend freely, and register token $i$ reads all patches plus registers $j \le i$. That triangular rule inside the registers is what makes the ordering meaningful.

The patch embeddings are then **thrown away**. Only the register outputs survive. Each is projected to $\mathbf{u}_{t,i} \in \mathbb{R}^{6}$ and quantised with FSQ ([Finite Scalar Quantization](https://arxiv.org/abs/2309.15505)) — $\tanh$ to bound each of the six dims, then round onto the lattice $[8,8,8,5,5,5]$:

$$\mathbf{q}_{t,i} = \operatorname{FSQ}(\mathbf{u}_{t,i})$$

Vocabulary $V = 8\cdot8\cdot8\cdot5\cdot5\cdot5 = 64{,}000$. So one token carries at most $\log_2 64000 \approx 16$ bits. That number matters later.

> [!NOTE] Nested dropout
> Per clip, sample one $k$ uniformly from $\{1,2,4,\ldots,256\}$ and replace every token at position $i \ge k$ with a learned mask token. The decoder must therefore work at any budget. No loss tells any token what to hold — the ordering *emerges* because early tokens are present in every sample and late ones only sometimes. Same family as [[Matryoshka Representation Learning|Matryoshka]] and the original nested-dropout work. ^nested-dropout

The surviving prefix conditions a time-causal **rectified-flow decoder** running on noised VAE latents. Training objective:

$$\mathcal{L}_{\text{base}} = \mathcal{L}_{\text{Flow}} + \lambda\,\mathcal{L}^{\text{dec}}_{\text{REPA}}, \qquad \lambda = 1$$

$\mathcal{L}_{\text{Flow}}$ is [[Flow Matching|flow matching]]; the REPA term aligns an early decoder layer with frozen DINOv2 patch features.

### What SemanTok adds

Three changes, all purple in the paper's Figure 3. A frozen DINOv2-L teacher supplies, per latent frame, a $16\times16$ grid of patch features $\mathbf{d}_{t,p} \in \mathbb{R}^{1024}$ aligned with the latent positions, plus a class token $\mathbf{c}_t$.

**1. DINO features into the encoder input.** Concatenate before projecting:

$$\mathbf{e}_{t,p} = W_{\text{in}}[\mathbf{z}_{t,p}; \mathbf{d}_{t,p}] + \mathbf{b}_{\text{in}}$$

**2. Class token into register zero**, through a zero-initialised projection so it starts as a no-op:

$$\widetilde{\mathbf{r}}_{t,0} = \mathbf{r}_{t,0} + W_{\text{cls}}\mathbf{c}_t$$

**3. Two prefix-only readout heads.** This is the part that does the work. For frame $t$ at budget $k$, the shared context is

$$\mathcal{C}_{t,k} = \{\mathbf{q}_{s,i} : 1 \le s \le t,\ 0 \le i < k\}$$

Note the temporal restriction $s \le t$ — it matches the tokenizer's causal path, so the head never cheats with the future.

Each head is two [[Cross Attention|cross-attention]] layers, width $d_h = 768$, 12 heads, independently parameterised. The **Dense DINO head** $h_\phi$ has one learned readout query $\mathbf{a}_p$ per spatial position. The **Class DINO head** $h_\psi$ has a single query $\mathbf{a}_{\text{cls}}$ (distinct from register token $\mathbf{r}_{t,0}$).

$$\widehat{\mathbf{D}}_t = h_\phi(\mathcal{C}_{t,k}), \qquad \widehat{\mathbf{c}}_t = h_\psi(\mathcal{C}_{t,k})$$

Both losses are cosine distance:

$$\mathcal{L}_{\text{dense}} = \frac{1}{TP}\sum_{t=1}^{T}\sum_{p=0}^{P-1}\!\left(1 - \cos(\widehat{\mathbf{d}}_{t,p}, \mathbf{d}_{t,p})\right)$$

$$\mathcal{L}_{\text{cls}} = \frac{1}{T}\sum_{t=1}^{T}\!\left(1 - \cos(\widehat{\mathbf{c}}_t, \mathbf{c}_t)\right)$$

$$\mathcal{L} = \mathcal{L}_{\text{base}} + 0.5\,\mathcal{L}_{\text{dense}} + 0.5\,\mathcal{L}_{\text{cls}}$$

The crucial property, stated plainly: **these heads never see the noised latent, so only the tokens can lower their losses.** That is the whole mechanism. Nested dropout keeps resampling $k$, so every prefix length must independently support both DINO predictions.

And note what is *not* changed: the decoder's reconstruction target is still the VAE latent $\mathbf{z}$, never DINO features. SemanTok is not a [RAE](https://arxiv.org/abs/2510.11690)-style "diffuse in DINO space" model. DINO is a teacher, not the output.

No loss assigns a particular DINO feature to a particular token. The ordering falls out of the shared prefix constraint.

### The AR stage

Each $\mathbf{q}_{t,i}$ maps to an index $\tau_{t,i} \in \{0,\ldots,63999\}$. With the tokenizer frozen, a LLaMA-style causal decoder (RMSNorm, [[Gated Activation|SwiGLU]]) trains on all $TK = 1280$ indices in **time-first** order:

$$(\tau_{1,0},\ldots,\tau_{T,0},\ \tau_{1,1},\ldots,\tau_{T,K-1})$$

All five frames at coarse position 0, then all five at position 1, and so on. Any prefix is a valid budget, so one trained model serves every $k$.

Sizes: 49M / 85M / 201M / 393M / 679M / 1.33B / 2.29B non-embedding params. Width $= 64d$ at depth $d$, with $d$ heads. AdamW, $\beta=(0.9,0.95)$, weight decay 0.05, clip 1.0, bf16, 2.5% warmup then cosine decay to 1% of peak. Peak LR scales as $0.512/\text{width}$ (uCO3D) or $1.024/\text{width}$ (Kinetics-600) — a width-inverse rule in the spirit of [[Tensor Programs V- Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (muP)|μP]].

Tokenizer budgets: 131B tokens on Kinetics-600, 66B on uCO3D. Both well under the ~400B of the released VideoFlexTok checkpoint, which the authors could not use anyway (its decoder was fine-tuned for bidirectional attention, deviating from the paper), so they retrained the baseline themselves under identical conditions.

Sampling: temperature 1.0, no top-$k$/top-$p$. Decoder 50 flow steps, [[Classifier-Free Guidance|CFG]] 3.0. AR guidance on Kinetics-600 is budget-dependent (3.0 for $k\in\{1,4\}$, 2.0 for $k\in\{8,16,32\}$, 1.0 above), copied from VideoFlexTok and **not tuned for either arm**.

## Ablation Studies and Experiments

Two datasets: Kinetics-600 (class-conditioned, 600 action labels) and uCO3D (text-conditioned objects, split into in-distribution and held-out out-of-distribution classes).

Metrics split into two axes the paper keeps rigidly separate:

- **Fidelity** — does it look like real video. gFVD, gFID. (See [[Evaluating Generative Models]] for why FVD/[[LAION-BVD- A 10-Million-Hour Open Video Dataset for Multimodal Pre-training#^fid|FID]] see what they see.)
- **Semantic alignment** — does it show the thing you asked for. UMT-L top-1 class accuracy on Kinetics-600; nearest-class-mean in InceptionV3 space on uCO3D (which stays defined for unseen classes, where a closed-set classifier cannot go); ViCLIP text–video cosine; ClipV video–video cosine against ground truth.

### Headline: smaller models, better videos

With each arm at its best $k$, SemanTok wins at **every one of seven AR sizes** on both datasets.

| Dataset | gFVD | gFID | class acc. |
|---|---|---|---|
| Kinetics-600 | $-11$ to $-24\%$ | $-5$ to $-17\%$ | $+25$ to $+61\%$ |
| uCO3D | $-2$ to $-13\%$ | $-5$ to $-8\%$ | $+22$ to $+30\%$ |

The semantic gap is enormous and the fidelity gap is solid. Concretely on Kinetics-600: the **smallest** SemanTok model (49M) beats the **largest** VideoFlexTok model (2.29B) on class accuracy, ClipV and ViCLIP. 2.29B SemanTok reaches 0.631 class accuracy against 0.560 for 2.29B VideoFlexTok; 49M SemanTok already clears 0.560 territory at its best $k$.

At a fixed $k{=}16$, an 85M SemanTok model beats VideoFlexTok of *any* size and *any* budget on gFID (both datasets) and gFVD (Kinetics-600).

There is also a degradation point worth knowing. More tokens eventually *hurt* gFVD, because the AR model's own prediction errors compound down the sequence. VideoFlexTok degrades after $k{=}16$; SemanTok holds to $k{=}32$.

### What scaling buys, and what it does not

This is the sharpest finding in the paper.

**Fidelity scales. Semantics do not.** Bigger or longer-trained AR models mostly trim compounding rollout error, which is a fidelity problem. But *what the clip contains* is largely decided by the first few tokens, and those come from the frozen tokenizer. No amount of AR scale fixes a tokenizer whose prefix is semantically thin.

For the 1.33B model on Kinetics-600, swept from 1.3B to 65.5B training tokens:

- SemanTok's gFVD lead shrinks from $28\%$ → $9\%$ by 26B tokens, then holds at $11$–$14\%$.
- SemanTok's class-accuracy lead stays at $30\%$ after 65.5B tokens.

So $5\times$ more training buys VideoFlexTok some fidelity back and **no** semantics. Same pattern across model size: at $k{=}64$ the gFID gain halves from 49M to 2.29B, while the class-accuracy gain does not close.

### Out-of-distribution classes

Tested on uCO3D, where the OOD object classes were never in training. With a 201M model at $k{=}16$, SemanTok raises generated class accuracy by $24\%$ on ID and $29\%$ on OOD. ClipV improves on both. The $k{=}16$ ID/OOD class-accuracy table: 0.34/0.23 for SemanTok vs 0.27/0.18 for VideoFlexTok.

The DINO teacher generalises, and so does what it taught the tokens. ViCLIP is the one holdout — it favours SemanTok only from $k{=}16$, not at $k{=}4$.

### The decoder-REPA probe — the mechanism check

This is the experiment that proves the diagnosis rather than just the fix. Read out the decoder's REPA projection with no further training, sweeping the decoder's input noise level $\sigma$ (same noise seed for both arms). At $\sigma{=}1$ the decoder input is pure noise, so any DINO content *must* have come from the tokens. $\sigma{=}1$ is also the state generation actually starts from.

Pure-noise readout at $k{=}256$, DINOv2 cosine:

| | uCO3D | Kinetics-600 |
|---|---|---|
| VideoFlexTok | 0.683 | 0.631 |
| SemanTok | 0.752 | 0.707 |

And the dependence on the latent, measured as the gap between $\sigma{=}0.25$ and $\sigma{=}1$ at $k{=}256$:

| | uCO3D | Kinetics-600 |
|---|---|---|
| VideoFlexTok | 0.035 | 0.054 |
| SemanTok | 0.003 | 0.016 |

Over $11\times$ more reliance on the latent for VideoFlexTok. SemanTok's prefix alone gives the decoder essentially everything the clean latent would have added. At $k{=}32$ on uCO3D, SemanTok's **pure-noise** readout (0.721) matches what VideoFlexTok needs a 75%-clean latent to reach (0.716).

A nice side-note: SemanTok's own two-layer dense head, reading tokens with no decoder at all, beats VideoFlexTok's full pure-noise *decoder* readout from $k{=}8$ on uCO3D (0.662 vs 0.641 at $k{=}16$).

The honest limitation of this probe, which the authors state: it does not separate the three token-side changes from each other. Encoder DINO input, class-token injection, and prefix DINO targets are tested as a bundle.

### Reconstruction vs generation — the realization gap

VideoFlexTok reconstructs better. SemanTok generates better. Specifics at $k{=}16$:

| Kinetics-600, $k{=}16$ | rFVD↓ | PSNR↑ | class acc.↑ |
|---|---|---|---|
| VideoFlexTok | **148.1** | **14.64** | 0.230 |
| SemanTok | 156.0 | 11.33 | **0.396** |

At $k{=}256$ on Kinetics-600: VideoFlexTok PSNR 19.66 vs SemanTok 18.53; SSIM 0.668 vs 0.634. On uCO3D the PSNR gap is wider — 23.88 vs 21.46 at $k{=}256$.

But VideoFlexTok *loses* more going from reconstruction to generation. SemanTok has the lower gFVD at every AR size at both $k{=}64$ and $k{=}128$, so its realization gap is narrower. A latent that reconstructs well is not automatically a latent that is easy to model — the point [LARP](https://arxiv.org/abs/2410.21264) and CRT also make.

### Where the gain lives: teacher forcing + bits

Two measurements localise the advantage to the early tokens.

**Teacher forcing.** Force the first $m$ ground-truth tokens per frame, then free-run to $k{=}256$. At 201M with nothing forced, SemanTok's gFVD is $23\%$ lower. Force just the first 16–64 tokens and **the lead vanishes** — after which VideoFlexTok's better-reconstructing tail actually edges ahead. The entire advantage is in predicting the first handful of tokens correctly.

**Bits.** [[Cross Entropy|Cross-entropy]] per token at $k{=}16$, 201M model: SemanTok 8.8 bits vs VideoFlexTok 12.9 bits — $32\%$ fewer. (Recall the ceiling is ~16 bits/token.) The marginal entropy of a position is only about one bit lower for SemanTok, so **most of the saving comes from context**: the prefix is more predictable from what came before, not just lower-entropy in isolation.

Per-position: SemanTok is cheaper for the first 128 positions and *more expensive* over 129–192. It front-loads cheap semantics and pushes the hard, high-entropy pixel detail into the tail.

And the saving is not degenerate repetition — the authors checked. Over all 256 positions, SemanTok duplicates $8.5\%$ of tokens within a frame vs VideoFlexTok's $12.0\%$, and copies the previous frame's token at the same position $0.05\%$ vs $0.42\%$. SemanTok repeats *less*.

The cost accounting is stark: SemanTok's first **4** tokens cost 37 bits/frame and nearly match the class accuracy of VideoFlexTok's first **32** tokens at 405 bits/frame. Roughly $11\times$ cheaper for the same semantic content.

### What did not work — the $k{=}1$ failure

Every statistically significant SemanTok loss sits on uCO3D at $k \le 4$, mostly gFID and ClipV at $k{=}1$. On Kinetics-600 SemanTok is never significantly worse at any budget.

The decoder-REPA probe shows the same wall. At $k{=}1$ the pure-noise lead is only $0.01$–$0.02$ cosine (vs $0.04$–$0.08$ from $k{=}16$), and with a mostly clean latent ($\sigma{=}0.25$) SemanTok is *behind* up to $k{=}4$ on uCO3D and $k{=}8$ on Kinetics-600.

The explanation is a budget argument and it is convincing: one token holds ~16 bits. That cannot simultaneously serve flow matching, decoder REPA, and two DINO reconstruction targets. Something has to give, and at $k{=}1$ what gives is pixel fidelity.

The oddity the authors flag as surprising: **class accuracy still favours SemanTok at every budget, including $k{=}1$** — 0.188 vs 0.074 in reconstruction on Kinetics-600, more than double. So even one token is more semantic; it is the *fidelity* metrics that lose at $k{=}1$.

### Other negative and null results

- **SigLIP 2 as teacher** was tried in early ablations and rejected in favour of DINOv2.
- **Either semantic pathway alone** lost to using both — encoder input *and* prefix targets together.
- **uCO3D overfits past 66B tokenizer tokens.** Continuing to 98B made reconstruction worse for *both* arms (VideoFlexTok PSNR 18.66→18.07, SemanTok 15.96→15.56). Small dataset; they report the 66B checkpoints.
- **Kinetics-600 was still improving at 131B** for both arms, so neither is converged.
- **No tokenizer checkpoint reverses the ordering.** Six uCO3D checkpoints (6.6B–66B) with a fresh 49M probe, and four Kinetics-600 checkpoints (26B–131B) with the full 201M recipe: SemanTok leads at every one, and no gap closes. On Kinetics-600 at **66B** — half the budget — SemanTok already beats VideoFlexTok's 131B result on all three metrics at every $k \ge 4$.

### Statistics

Paired bootstrap, 1000 replicates, both arms conditioned on the same prompts so each replicate resamples jointly. At $k{=}16$ they additionally pool six sampling seeds. Semantic-alignment gains clear zero from $k{=}2$; fidelity gains clear zero from $k{\approx}16$ on uCO3D and $k{=}4$–$32$ on Kinetics-600 depending on size. This is more careful than most generative-model papers manage.

## Worth Remembering

**The transferable lesson has nothing to do with video.** If an auxiliary loss is computed from a module that sees two inputs, and one of those inputs can satisfy the target on its own, the loss does not constrain the other input. REPA on a decoder that also sees a near-clean latent is a leaky objective. The fix — a head that reads *only* the thing you want to constrain — is the general move. Worth holding next to [[Shortcut Learning in Deep Neural Networks|shortcut learning]]: the model found the easy path, and the easy path was a different input.

**The limitation the authors state plainly.** Because SemanTok prioritises semantics over reconstruction, it cannot recover exact colours and appearance detail at low budgets. If your application needs pixel-faithful reconstruction at small $k$, this is the wrong tokenizer. The PSNR gaps are 2–3 dB and real.

**No loss ever assigns a feature to a token.** The coarse-to-fine ordering is entirely emergent from nested dropout plus the shared-prefix constraint. That is elegant and also means you cannot directly control *what* lands where.

**Semantic alignment is a tokenizer property; fidelity is an AR property.** This is the most actionable split in the paper. If your generations show the wrong thing, scaling the AR model is close to wasted money — go fix the tokenizer. If they show the right thing badly, scale the AR model. A useful diagnostic before spending on compute.

**Both arms are undertrained.** 131B vs the released VideoFlexTok's ~400B. The checkpoint sweeps make a reversal unlikely but do not rule out the gap narrowing further at full budget — the Kinetics-600 fidelity gap *does* shrink with AR training, after all. Treat the gFVD numbers as **current best understanding** rather than settled, and the class-accuracy numbers as robust.

**Guidance was not tuned.** Both arms use VideoFlexTok's budget-dependent AR guidance schedule. Fair, but it is tuned for VideoFlexTok's token statistics, and SemanTok's are measurably different (32% fewer bits). Tuning might move either arm.

**The teaser of a follow-up.** The authors note that the early prefix is a compact, semantic, causally-ordered state — a natural candidate for **action-conditioned world models**. A policy conditioning on 4–16 tokens per frame rather than 256 is a very different compute story. Also flagged: optimising predictability directly rather than getting it as a side effect, and using video-native or language-aligned teachers instead of an image model.

**A caveat on the DINO frames.** The VAE compresses 17 RGB frames into 5 latent frames as groups $\{0\}, \{1..4\}, \{5..8\}, \{9..12\}, \{13..16\}$. The dense DINO target *averages* DINOv2 over each group. Averaging features across four frames of motion is lossy, and is a plausible reason the method helps appearance more clearly than it helps motion.

**Open question worth chasing.** Why does SemanTok keep its class-accuracy lead at $k{=}1$ while losing on fidelity? A single ~16-bit token holding better class information than VideoFlexTok's is almost a free lunch; the paper calls it surprising and leaves it there.

## Links

Related: [[DINOv2- Learning Robust Visual Features]] · [[Matryoshka Representation Learning]] · [[Flow Matching]] · [[VQ-VAE]] · [[Video Diffusion]] · [[Diffusion Models]] · [[Causal Attention]] · [[Cross Attention]] · [[Auto-regressive models]] · [[Tokenization]] · [[Evaluating Generative Models]] · [[Cross Entropy]] · [[Classifier-Free Guidance]] · [[Latent Diffusion]] · [[Shortcut Learning in Deep Neural Networks]] · [[JEPA]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Tensor Programs V- Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (muP)]] · [[Embeddings]] · [[Rectified Flow]] · [[Mastering Diverse Domains through World Models (DreamerV3)]]

New topics worth writing: REPA (representation alignment for diffusion training), Finite Scalar Quantization, FVD and video generation metrics, nested dropout, VideoFlexTok and FlexTok, compression–generation trade-off in visual tokenizers, perception–distortion trade-off, teacher-forcing as an attribution tool, bits-per-token as a tokenizer diagnostic, token-based world models
