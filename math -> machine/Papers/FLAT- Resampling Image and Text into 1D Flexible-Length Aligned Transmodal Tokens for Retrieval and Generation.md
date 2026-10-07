---
title: "FLAT: Resampling Image and Text into 1D Flexible-Length Aligned Transmodal Tokens for Retrieval and Generation"
authors: ["Sun et al."]
year: 2026
arxiv: "2609.16591"
url: https://arxiv.org/abs/2609.16591
priority: Good-To-Read
read_on: 2026-09-22
tags: [paper, llm, self-supervised, diffusion, vision]
---
## The Core Idea

Today's multimodal stack is two separate jobs glued together. First you train an encoder — [[CLIP]] or similar — to make good embeddings. Then you freeze it and hang a generator off it: a language decoder for captioning, a diffusion model for text-to-image. The generator never gets to change the embedding. Whatever the contrastive objective threw away is gone forever.

FLAT trains both halves at once, and the thing in the middle is the point.

One shared encoder eats either an image or a piece of text and spits out the same kind of output: an ordered sequence of up to 256 continuous vectors, each 64 numbers wide. That sequence is simultaneously (a) the thing you do nearest-neighbour search against and (b) the conditioning signal a rectified-flow image decoder and an autoregressive text decoder read from. Three losses pull on it at the same time — contrastive alignment, caption cross-entropy, flow-matching velocity regression.

Two things fall out that are worth the read even if you never touch image generation.

**The sequence is nested, so you can cut it short.** During training they sample a keep-length $K$ from $\{1, 4, 16, 64, 256\}$ and throw away every token past position $K$. This is [[Dropout- A Simple Way to Prevent Overfitting|nested dropout]] applied to a sequence rather than to dimensions, the same trick as [[Matryoshka Representation Learning]] but one rung up the ladder. At serving time you pick $K$ per query.

**Retrieval barely notices.** On MS-COCO, one token (64 floats) gets I2T R@1 of 63.00. All 256 tokens (16,384 floats) get 63.54. A 256× reduction in embedding width costs half a point. Matryoshka, sparse autoencoder and contrastive-sparse-coding baselines all degrade sharply as you shrink; FLAT's curve is flat.

> [!NOTE] Flexible-length representation
> An embedding stored as an *ordered sequence* of token vectors, trained so that any prefix of length $K$ is itself a usable embedding. Truncation is the compression knob, and it is free at query time. ^flexible-length

The generation side tells the opposite story, and that asymmetry is the most interesting empirical fact in the paper. A single token is enough to *find* the right image but nowhere near enough to *draw* it. GenEval on colour binding is 0.03 at $K{=}1$ and 0.67 at $K{=}16$. Global semantics live in the first token; composition and spatial relations need more.

> [!NOTE] Retrieval and generation want different amounts of information
> Discriminating "which of 5,000 images is this" is a much lower-bandwidth task than reconstructing the image. One vector suffices for the first. It does not for the second. ^retrieval-vs-generation-bandwidth

## The Methodology

**Encoder.** A Qwen3.5-2B VLM with frozen base weights and a LoRA adapter ($r{=}16$, $\alpha{=}32$, dropout 0.05, on Q/K/V/O and the MLP projections). Total trainable across adapters and projections: 11.4M parameters. See [[LoRA- Low-Rank Adaptation of Large Language Models]].

Given an input $x$ of either modality, append $N = 256$ learnable **register tokens** $R$, run the whole thing through the encoder with the system prompt *"Represent the user's input."*, take the hidden states at the register positions, and linearly project:

$$\mathbf{z} = W_{\text{lat}}\, f_{\text{enc}}(x, R) \in \mathbb{R}^{256 \times 64}$$

Images and text share the registers, the encoder and the projection. That is the whole reason the two modalities land in the same space.

**Truncation.** One $K$ is drawn per optimisation step and broadcast to all 64 ranks, so every loss in that step sees the same prefix. Only $\mathbf{z}_{:K}$ survives. They sample uniformly over the ladder $\{1,4,16,64,256\}$ — a geometric ladder, not consecutive integers, because adjacent lengths like 200 and 201 are indistinguishable and you would waste steps on them.

**Contrastive loss.** Not a single pooled vector. They score register-to-register at matching positions and average — this is exactly [[ColBERT- Efficient and Effective Passage Search via Late Interaction|late interaction]], but position-aligned rather than max-over-all-pairs:

$$s_{ij}^{(K)} = \frac{1}{K}\sum_{n=1}^{K} \frac{(z^{\text{img}}_{i,n})^\top z^{\text{txt}}_{j,n}}{\lVert z^{\text{img}}_{i,n}\rVert_2 \lVert z^{\text{txt}}_{j,n}\rVert_2}$$

Then symmetric [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)|InfoNCE]] in both directions, temperature 0.07, negatives gathered across all devices (global batch 1024).

**I2T decoder.** A second LoRA adapter on the same Qwen backbone. The image representation $\mathbf{z}^{\text{img}}_{:K}$ goes through an MLP + RMSNorm into the decoder's hidden width, is prepended as soft tokens after the prompt *"Describe the input."*, and standard next-token cross-entropy is applied to the caption.

**T2I decoder.** SANA-1.6B, fully fine-tuned, conditioned on the text soft tokens $\mathbf{e}^{\text{txt}}$ via [[Cross Attention|cross-attention]]. Standard [[Rectified Flow|rectified-flow]] / [[Flow Matching|flow-matching]] objective on [[Variational Autoencoder|VAE]] latents: with $\hat{x}_t = (1-t)\hat{x} + t\epsilon$,

$$\mathcal{L}_{\text{img}} = \mathbb{E}_{\hat x, \epsilon, t}\left[\lVert v_\phi(\hat x_t, t, \mathbf{z}^{\text{txt}}_{:K}) - (\epsilon - \hat{x})\rVert_2^2\right]$$

[[Classifier-Free Guidance|CFG]] dropout 0.1, logit-normal $t$ sampling, shift 3.0.

**Total:** $\mathcal{L} = \lambda_{\text{align}}\mathcal{L}_{\text{align}} + \lambda_{\text{txt}}\mathcal{L}_{\text{txt}} + \lambda_{\text{img}}\mathcal{L}_{\text{img}}$, all three weights set to 1.0. No tuning reported.

**Scale.** 65.7M image–text pairs, 135k steps, 8 nodes × 8 H200, global batch 1024, AdamW at $1\times10^{-4}$ with 500-step warmup then cosine to zero, bf16, 512×512 images. Data mix: ~30M long-caption, ~33M short-caption, ~4M prompt-style, ~0.1M curated.

**Fine-tuning** is per-task and touches different parameters each time: T2I updates only the image decoder (8k steps on 120K clean pairs); captioning updates only the text-decoder LoRA (5k steps on COCO); retrieval updates the encoder LoRA, the registers and the latent projection (5k steps COCO, 320 steps Flickr30k).

## Ablation Studies and Experiments

**Headline numbers.** Pre-trained only, no task fine-tuning: GenEval 71.1, zero-shot COCO R@5 69.1 (I2T) / 64.6 (T2I). After task fine-tuning: GenEval **83.1** without prompt rewriting (MetaQuery-L gets 0.78 under the same no-rewrite rule); COCO captioning **40.5 BLEU-4, 138.6 CIDEr**; COCO R@5 **86.8 / 75.8**; Flickr30k R@5 **98.3 / 93.6**; ImageNet linear probe on frozen features **81.8** at $K{=}64$, and **73.3** using a single 64-dim token, which already beats DREAM's 72.7 at 1024 dims.

**The loss ablation (40k steps, $K{=}256$).** All seven non-empty combinations:

| Losses | I2T R@1 | T2I R@1 | B@4 | CIDEr | GenEval |
|---|---|---|---|---|---|
| All three | 52.2 | 56.3 | 40.2 | 137.5 | 0.327 |
| w/o contrastive | 13.6 | 6.7 | 40.7 | 139.2 | 0.297 |
| w/o captioning | 51.5 | 53.6 | 33.9 | 117.1 | 0.329 |
| w/o image gen | 51.2 | 54.5 | 40.0 | 137.8 | **0.003** |
| contrastive only | 51.0 | 57.0 | 34.2 | 118.0 | 0.003 |
| captioning only | 39.4 | **4.0** | 40.8 | 139.7 | 0.004 |
| image-gen only | 0.6 | 1.1 | 33.1 | 110.8 | 0.323 |
| untrained | 0.2 | 0.0 | 30.8 | 105.0 | 0.000 |

Read it honestly and it is mostly *"each task needs its own loss."* Drop the contrastive term and retrieval collapses to 13.6/6.7. Drop image generation and GenEval is 0.003. The genuinely interesting cells are the cross-effects: captioning-only gives **T2I retrieval of 4.0** — a generative objective alone destroys the retrieval geometry, while contrastive-only gives a perfectly fine 57.0. And at $K{=}1$ the full objective beats every subset on both retrieval directions (57.8 / 62.4 vs 50.7 / 57.2 for contrastive-only), so the mutual reinforcement is real but concentrated at short prefixes.

**The control the paper deserves credit for (Appendix D).** Same architecture, same data, same steps, but no FLAT pre-training:

| | $K{=}1$ | $K{=}4$ | $K{=}16$ | $K{=}64$ | $K{=}256$ |
|---|---|---|---|---|---|
| GenEval, task-only | 0.60 | 0.61 | 0.61 | 0.60 | 0.59 |
| GenEval, FLAT | 0.49 | 0.77 | **0.83** | 0.83 | 0.82 |
| CIDEr, task-only | 80.8 | 91.5 | 104.3 | 105.3 | 105.0 |
| CIDEr, FLAT | 122.0 | 131.0 | 136.2 | 138.1 | 138.6 |
| T2I R@1, task-only | **48.62** | 48.54 | 48.49 | 48.53 | 48.43 |
| T2I R@1, FLAT | 47.76 | 47.93 | 47.94 | 47.73 | 47.98 |

Generation gains enormously. Retrieval gains essentially nothing — text-to-image R@1 is *worse* with pre-training by 0.4–0.9 points. If you only want embeddings, in-domain contrastive fine-tuning learns almost the whole thing on its own. The joint pre-training buys you a representation that a decoder can use, and that is the only place it pays.

Note also the task-only GenEval is flat at ~0.60 across $K$. Without joint pre-training the decoder cannot exploit extra tokens at all.

**Things that did not work.**

- **Null padding.** [[Diffusion Transformer|FlexTok]]-style fixed-length sequences with learned null embeddings past position $K$ caused gradient-norm spikes around 33k steps and the contrastive loss collapsed to $\ln(1024) \approx 6.93$ — exactly random chance for the global batch. Plain truncation trained stably. (Caveat: the two runs used their own original recipes, so this is qualitative.)
- **No sampling schedule wins everywhere.** Geometric sampling (weighting long prefixes, $\beta = 1.3$) gives GenEval 50.2 at $K{=}256$ versus uniform's 32.7 at 40k steps — a huge gap — but costs retrieval (I2T R@1 51.8 vs 52.2, T2I 55.5 vs 56.3) and is worse at $K{=}1$ (26.0 vs 33.0 GenEval). "Replay" (short prefixes first, then full) is best for retrieval. They shipped uniform.
- **Zero-shot captioning looks bad and mostly isn't.** The pre-trained checkpoint gets 63.8 CIDEr versus 138.6 fine-tuned. The diagnosis: 27.1% of zero-shot captions open with a quotation mark (title style), averaging 10.8 words. Overlap metrics punish style, not semantics. Much of the "fine-tuning gain" is format alignment.
- **Composed retrieval with counting.** Latent arithmetic on CIRR reaches Hit@1 44.0 at $K{=}1$, but edits that change object count while preserving category fail.

**Compute (Appendix F.2).** The encoder is causal, so register $n$'s hidden state does not depend on registers after it — you can encode just the first $K$ and skip the rest. Going from $K{=}256$ to $K{=}1$ cuts encoder FLOPs by 50% and I2T decoder FLOPs by 71%, but T2I only by 13.8%, because denoising and VAE decoding dominate there.

## Worth Remembering

**The modality gap closes.** PCA of COCO pairs shows CLIP and SigLIP2 with clearly separated image and text clouds. FLAT overlaps them: one register token halves CLIP's centroid distance, 256 tokens quarters it. This is what makes the interpolation and arithmetic demos work — $z_\alpha = (1-\alpha)z_1 + \alpha z_2$ decodes sensibly through *both* heads, and $z_1 - z_2 + z_3$ edits a concept while preserving scene context, with no editing supervision.

**Composed retrieval for free.** Turn an edit instruction into add/remove clauses, compute $\mathbf{z}_q = \mathbf{z}_{\text{ref}} + \alpha[\sum_{a}(\mathbf{z}_a - \mathbf{z}_0) - \sum_r (\mathbf{z}_r - \mathbf{z}_0)]$ where $\mathbf{z}_0$ is the neutral phrase *"a photo"*, re-normalise, score with the late-interaction formula. $K{=}1$ wins (Hit@1 44.0, $\alpha = 3$), which is another data point that global semantics live in token zero. $\alpha$ had to be swept over $\{0.25 \ldots 8\}$ per $K$ — the scale is not self-calibrating.

**Practical caveats if you wanted to use this.**

- Late interaction means your index stores $K$ vectors per item and scoring is $K$ dot products, not one. At $K{=}1$ this is just a normal 64-dim [[Vector Database|ANN]] index. At $K{=}256$ it is a ColBERT-style problem with all the serving cost that implies. The paper's own numbers say use $K{=}1$ for retrieval.
- Nothing here is evaluated on hard negatives, and the contrastive setup is plain in-batch negatives ([[Dense Passage Retrieval (DPR)]] style) with no mining. The Flickr30k retrieval fine-tune is 320 steps, which is tiny — suspiciously easy to overfit, and Flickr T2I R@1 actually *drops* at $K{=}256$ (76.26) relative to $K{=}4$ (77.40).
- Encoder and captioner are two LoRA adapters on the *same* frozen 2B backbone. Serving both simultaneously means adapter swapping or two copies.
- The pre-training data is proprietary (65.7M pairs, Meta-internal mix). The 120K "high-quality" T2I fine-tune set is unspecified.
- The image decoder is fully fine-tuned (1.6B params); only the language side is parameter-efficient.

**Open question I'd want answered.** The retrieval-flat-across-$K$ result is the commercially valuable one, and Appendix D says the generative pre-training contributed roughly nothing to it. So: is the flatness coming from joint training at all, or just from nested dropout over an ordered register sequence? A contrastive-only model with the same register + truncation scheme would settle it, and the ablation table's "contrastive only" row is at $K{=}256$ only for retrieval — it reports 51.0/57.0, competitive with the full 52.2/56.3. That is close to an answer, and the answer looks like "nested dropout is doing the work."

**Where it connects.** The nesting idea is [[Matryoshka Representation Learning]] moved from dimensions to sequence positions. The scoring is [[ColBERT- Efficient and Effective Passage Search via Late Interaction]]. The modality-gap finding sits next to the anisotropy line of work in [[How Contextual are Contextualized Word Representations]] and [[Whitening Sentence Representations]]. The "one embedding serving both retrieval and a downstream decoder" ambition is the same bet as CoCa and BLIP, with the generative half upgraded from captioning to bidirectional.

## Links

Related: [[Matryoshka Representation Learning]] · [[ColBERT- Efficient and Effective Passage Search via Late Interaction]] · [[CLIP]] · [[Embeddings]] · [[Dense Passage Retrieval (DPR)]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Flow Matching]] · [[Rectified Flow]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Latent Diffusion]] · [[Classifier-Free Guidance]] · [[Cross Attention]] · [[Vector Database]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Dropout- A Simple Way to Prevent Overfitting]] · [[Reranking]] · [[Sentence-BERT]] · [[Variational Autoencoder]]

New topics worth writing: nested dropout, register tokens, 1D visual tokenisation (TiTok / FlexTok), modality gap in dual encoders, composed image retrieval (CIRR), GenEval, CoCa, adaptive-dimension embeddings (CSR / sparse autoencoder retrieval), Shapley attribution for loss ablations
