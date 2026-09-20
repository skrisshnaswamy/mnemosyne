---
title: "NeoMME: A Single-Tower Multimodal-Native Multilingual Foundation Encoder for Efficient Fine-Tuning and Inference"
authors: ["Aurélien Lac", "Tony Wu"]
year: 2026
arxiv: "2609.01657"
url: https://arxiv.org/abs/2609.01657
priority: Good-To-Read
read_on: 2026-09-06
tags: [paper, transformers, llm, self-supervised, diffusion, vision, theory]
---
## The Core Idea

Almost every multimodal retriever today is a Frankenstein. You take a vision tower (SigLIP, CLIP) that was trained separately, bolt it onto a language model that was trained separately, and then fine-tune the pair to score query–document pairs. ColPali does this with a 3B generative VLM. ModernVBERT does it with a 250M bidirectional encoder plus a frozen-ish SigLIP2 tower. In both cases you inherit architecture that was designed for *generating* text, and you pay its parameter and compute cost for a job that never generates anything.

**NeoMME throws away both towers.** One bidirectional Transformer. Text tokens go in through an embedding table. Raw $32\times32$ RGB pixel patches go in through a tiny 2-layer MLP. After those two input projections, *every layer is shared*. No pretrained vision encoder, no causal decoder, no patch-merger module. The whole thing is trained from random initialisation.

The second unusual choice is the pretraining objective. Instead of BERT's fixed 15% masking (see [[BERT- Pre-training of Deep Bidirectional Transformers]]), NeoMME uses **masked discrete diffusion**: for each document you sample a corruption rate $\rho \sim \mathcal{U}(0,1)$ and mask that fraction of tokens. It is BERT's [[BERT- Pre-training of Deep Bidirectional Transformers#^masked-language-model|masked LM]] generalised over every noise level at once, which is what [[Denoising Diffusion Probabilistic Models|diffusion]] does for continuous data. For image–text pairs, the image patches stay fully visible and only the text is corrupted — the image is *always* conditioning, never a target.

> [!NOTE] Single-tower multimodal encoder
> Text and image patches are projected into the same hidden space by cheap modality-specific stems, then processed by one shared bidirectional Transformer. Contrast with dual-tower ([[A Simple Framework for Contrastive Learning (SimCLR)|CLIP]]-style, no cross-modal interaction) and VLM-style (pretrained vision tower feeding a causal decoder). ^single-tower

What it unlocks: a 260M-parameter model that scores **0.523 nDCG@10 on ViDoRe v3**, within 0.2 points of the 3.75B ColQwen2.5 — a $14.4\times$ parameter reduction — while encoding $2048\times2048$ pages at **51.3 pages/second on an L40S, $1.97\times$ ColModernVBERT's throughput**. And because there is no vision tower dictating a fixed pixel budget, NeoMME can eat full-resolution pages: 16,384-token context, enough for two 4K UHD images.

The key trick that keeps this affordable is the **32-pixel patch**. Everyone else uses 16. At the same resolution, 32-pixel patches give you a quarter as many image tokens. That is a $16\times$ reduction in attention cost, and it is the single reason a small model can look at a 2048-pixel page.

---

## The Methodology

### Input paths

**Text.** ALBERT-style factorised embedding to save parameters: a $256$-dimensional lookup table $E \in \mathbb{R}^{V\times d_e}$ ($V = 131{,}072$) followed by a [[Linear Projection|projection]] $P \in \mathbb{R}^{d \times d_e}$ up to model width:

$$\bm{h}^{(0)}_i = P E_{x_i,:}^\top$$

The output head reuses *both* factors, $\bm{\ell}_i = E P^\top \bm{h}_i$, so the masked-token decoder adds zero new parameters.

**Images.** RGB, cut into non-overlapping $32\times32$ patches. Each patch is $3\times32\times32 = 3072$ raw numbers → [[Layer Normalization|LayerNorm]] → 2-layer MLP → model width. That is the whole vision stack. Dynamic resolution: for each training example they sample a longest-side cap uniformly between 1024 and 2048 pixels, preserve aspect ratio, downsample only. Structural tokens `[IMG]` and `[ROW]` mark the image start and each patch-row boundary.

**Position.** 2D [[RoFormer- Enhanced Transformer with Rotary Position Embedding|RoPE]]: consecutive rotary frequency pairs alternate between two axes. Text token $i$ gets coordinate $(i,i)$ — the diagonal, which recovers ordinary 1D ordering. A patch at row $r$, column $c$ gets $(b+2+r,\, b+2+c)$ where $b$ is the image's coordinate base. Global layers use *partial* RoPE (only 25% of each head is rotated) with base $10^6$; sliding-window layers use full RoPE with base $10^4$.

### Backbone

17 layers (260M) or 20 layers (800M), hidden width 1024 / 1792. Bidirectional attention throughout. Long context is made affordable by mixing attention types:

- Every sixth layer, plus the final layer, is **global** (3 global layers in the 260M, 4 in the 800M).
- All other layers use **symmetric sliding window**, alternating half-windows of 256 and 1024 tokens.
- [[GQA- Training Generalized Multi-Query Transformer Models|Grouped-query attention]] everywhere: 16 query / 4 KV heads (260M), 28 / 7 (800M), head dim 64.
- QK normalisation: queries and keys are RMS-normalised independently *before* RoPE.

Each block: parameter-free RMS pre-norm, squared-ReLU MLP, and a query-dependent elementwise sigmoid gate on the attention output (see [[Gated Activation]]). Both residual branches are scaled by $(2L)^{-1/2}$ — the branch-scaling piece of Depth-$\mu$P.

Two less common additions, both borrowed from modded-nanogpt:

- **Token-indexed value embeddings.** At the first and last global layers, a learned table indexed by *token id* is added to the value vectors. Related to value-residual learning but with its own table.
- **Residual mixing.** Each block forms a learned scalar mix of the current residual stream and the original normalised input before its attention. Initialised at $(1, 0)$ — i.e. pure residual stream at step 0.

**Initialisation.** Attention output and MLP down projections are set to **zero**, making every block an exact identity at init (the Fixup idea). Word table drawn from $\mathcal{N}(0, d_e^{-1})$ with $d_e=256$, which keeps the tied factorised decoder's logits at order-one scale. Compare [[Delving Deep into Rectifiers (He init, PReLU)]] and [[Understanding the difficulty of training deep feedforward networks (Xavier init)]] — this is a different philosophy: kill the branches rather than balance the variance.

### Pretraining objective

For each real document segment $s$ in a packed sequence, draw a corruption rate. Text-only: $\rho_s \sim \mathcal{U}(0,1)$. **Multimodal: $\rho_s \sim \mathcal{U}(0.30, 1)$** — the higher floor is deliberate, to stop the model solving the task from surrounding text alone. Each eligible text position is independently replaced by `[MASK]` with probability $\rho_s$. Unlike BERT there is no 80–10–10 mixture; selected positions are *always* masked, and loss is computed only there. Image patches, padding, and structural markers are never masked.

$$\mathcal{L}_j = \frac{\sum_{i\in\mathcal{M}_j} w_i \operatorname{CE}\!\left(f_\theta(\tilde{\bm{x}}_j, \bm{p}_j)_i,\, x_i\right)}{\sum_{i\in\mathcal{M}_j} w_i}, \qquad w_i = \frac{1}{\max(r_{s(i)},\, 0.05)}$$

The inverse-rate weight $w_i$ is the standard absorbing-mask correction: low-corruption samples are rare and informative, so they get up-weighted, but the weight is capped at 20 to keep the loss scale stable. See [[Cross Entropy]] for `CE`. Losses are averaged over data-parallel ranks.

**There is no image loss at all.** No pixel reconstruction, no masked image modelling, no image-level contrastive term. The image only earns its keep by helping predict masked text.

### Training scale

| | 260M | 800M |
|---|---|---|
| Nodes / H100s | 2 / 16 | 4 / 32 |
| Packed tokens per step | 1,048,576 | 1,048,576 |
| Steps | 500,000 | 500,000 |
| Total tokens | ~524B | ~524B |
| NorMuon peak LR | 0.012 | 0.010 |
| AdamW peak LR | 0.0013 | 0.00075 |

Mixture is 55% text-only / 45% multimodal by packed token. Text draws from FineWeb-Edu, FineWeb2-HQ (20 languages), FinePDFs, Cosmopedia-v2, Wikipedia, FineMath, StarCoderData. Multimodal draws from FineVision (50%), PDFA+LightOnOCR (30%), PixelProse (10%), DocAtlas (7%), synthetic multilingual OCR (3%).

**Optimiser routing** is worth noting. Matrix-valued parameters use **NorMuon** — [[Old Optimizer, New Norm- An Anthology (Muon)|Muon]]'s orthogonalised matrix update plus per-neuron second-moment normalisation. Embedding tables and 1D parameters use a custom MasterAdamW with full-precision master weights while the model itself stays in bfloat16 (cf. [[Mixed Precision Training]], [[Decoupled Weight Decay Regularization (AdamW)]]). NorMuon uses *cautious* weight decay: decay is applied only to coordinates where the parameter sign and the update sign agree. Schedule is warmup–stable–decay: 300 warmup steps, flat, then linear decay over the final 10% to 1% of peak.

Systems detail that mattered: at this scale the **data pipeline, not the GPU, is the bottleneck**. Documents are packed into 16,384-position streams with variable-length attention boundaries keeping them isolated. Images decode and patchify in background workers; batches prefetch one step ahead. [[FlashAttention- Fast and Memory-Efficient Exact Attention|FlashAttention-3]], Liger fused linear cross-entropy, and `torch.compile` on top.

### Retrieval fine-tuning

One backbone pass, **two heads**:

- **Late-interaction head.** Linear projection of every final hidden state to 128 dims, L2-normalised. One vector per text token, one per image patch. Exactly [[ColBERT- Efficient and Effective Passage Search via Late Interaction|ColBERT]]'s shape. Queries get 10 learned `[MASK]` expansion tokens appended, ColBERT-style.
- **Dense head.** Mean-pool the final hidden states, L2-normalise. Parameter-free. Trained with [[Matryoshka Representation Learning|Matryoshka]] at widths 128/256/512/1024 (and 1792 for the 800M).

Scoring uses **MeanMaxSim** rather than plain MaxSim — MaxSim divided by query length:

$$s_{\mathrm{late}}(q,d) = \frac{1}{L_q}\sum_{s=1}^{L_q}\max_{1\le t\le L_d}\langle \bm{q}_s, \bm{d}_t\rangle$$

Loss is plain [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)|InfoNCE]] at $\tau = 0.02$ on both heads, summed with weight 1 each:

$$\mathcal{L}_{\mathrm{retrieval}} = \mathcal{L}_{\mathrm{late}} + \mathcal{L}_{\mathrm{dense}}$$

The dense term averages the InfoNCE loss over all Matryoshka widths $k \in \mathcal{K}$, each renormalised after truncation. No teacher logits, no [[Distilling the Knowledge in a Neural Network|distillation]].

20,000 steps on 8 H100s. Batch mixture 1 text : 2 image. GradCache keeps the full in-batch corpus for [[Dense Passage Retrieval (DPR)#^in-batch-negatives|in-batch negatives]] while recomputing encoder activations in chunks of 24 (260M) or 12 (800M) — ordinary gradient accumulation would silently shrink the negative pool and change the objective.

**Hard negatives are self-mined in two passes.** Pass 1 trains with in-batch negatives only; that checkpoint then mines a 32-candidate window per visual query; pass 2 retrains with 7 negatives sampled from the window. Negatives scoring $\ge 0.98\times$ the positive are dropped as likely false negatives. Compare [[Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)]] — same bootstrap idea, one round instead of continuous refresh.

### Late-Interaction Kernels (LIK)

A naive MaxSim materialises the full $L_q \times L_d$ similarity tensor. At a $2048\times2048$ page that is a 4,162-vector document, scored against the whole in-batch corpus — the tensor dominates memory. LIK tiles the computation and keeps only a running per-query-token maximum, storing the *winning document-token index* so the backward pass can route the gradient to it. Exact, not approximate. This is [[FlashAttention- Fast and Memory-Efficient Exact Attention#^io-aware|the FlashAttention trick]] applied to MaxSim.

At 4,096 document tokens: memory peak drops 672 MB → 193 MB, backward 1.82 ms → 0.54 ms. At 8,192 tokens the naive version OOMs and LIK runs in 321 MB.

---

## Ablation Studies and Experiments

### ViDoRe v3 (nDCG@10, 8 public tasks)

| Model | Params | v3 | v2 (@5) | v1 (@5) |
|---|---|---|---|---|
| ColSmol-256M | 256M | 0.207 | 0.348 | 0.797 |
| ColModernVBERT | 250M | 0.261 | 0.407 | 0.806 |
| **NeoMME-260M** | 260M | **0.523** | **0.522** | **0.860** |
| ColSmol-500M | 500M | 0.340 | 0.455 | 0.825 |
| Vultron Flash | 850M | 0.565 | 0.604 | 0.882 |
| **NeoMME-800M** | 800M | 0.556 | 0.559 | 0.874 |
| ColPali v1.3 | 2.92B | 0.430 | 0.547 | 0.848 |
| ColQwen2.5-v0.2 | 3.75B | 0.524 | 0.601 | 0.895 |

The 260M number is the headline: **+26.1 nDCG points over the best sub-300M model on v3**. It beats ColQwen2.5 (3.75B) on v3 while losing on v1 — v1 is saturated, v3 is the harder and better-constructed benchmark, so this is a real signal about which benchmark you trust.

Scaling 260M → 800M buys +3.3 on v3, +3.7 on v2, +1.5 on v1. Diminishing, and it costs you half your throughput.

### Text retrieval (BEIR-15, [[NDCG|nDCG@10]])

| Model | Params | Score |
|---|---|---|
| ColBERTv2 | 110M | 0.486 |
| LateOn | 149M | 0.572 |
| NeoMME-260M (late) | 260M | 0.488 |
| NeoMME-800M (late) | 800M | 0.513 |
| DenseOn | 149M | 0.562 |
| NeoMME-260M (dense) | 260M | 0.306 |
| NeoMME-800M (dense) | 800M | 0.369 |

This is where the model is clearly *not* state of the art. Late-interaction beats the dense head by **18.3 points** at 260M and **14.4** at 800M — a huge gap, larger than typical. The dense head at 0.306 is poor. The authors attribute it to supervision scale: NeoMME sees ~430K pure-text query examples; mLateOn sees ~660M contrastive pairs plus 16M hard-negative pairs. Plausible, but untested.

### The dual-head ablation (the one real controlled experiment)

Same init, same seed, same data order, same negatives, same schedule — only the active losses differ.

| Architecture | ViDoRe v3 late | ViDoRe v3 dense | BEIR late | BEIR dense |
|---|---|---|---|---|
| LI head only | 0.5088 | – | 0.4774 | – |
| Dense head only | – | 0.3906 | – | 0.3240 |
| Dual-head | **0.5226** | 0.3907 | **0.4881** | 0.3055 |

Joint training helps late-interaction (+1.38 on v3, +1.07 on BEIR) and does essentially nothing for dense (+0.01 on v3). A paired test over all 14,514 judged queries gives the late gain a 95% CI of [1.11, 1.67], $p = 10^{-4}$; the dense change has CI $[-0.26, 0.32]$, $p = 0.81$ — indistinguishable from zero. **And on BEIR the dense head gets 1.85 points *worse* under joint training.** So this is one-directional transfer, dense → late, not a free lunch. Single seed, so run-to-run variance is unmeasured.

### Compression — the most practically useful result

A $2048\times2048$ page produces 4,162 vectors × 128 dims × float32 ≈ **1.5 MB per page**. That is fatal at corpus scale. Two orthogonal fixes:

**Hierarchical token pooling** (cluster similar document vectors, replace each cluster with its mean). At pool factor 7 both models keep >99% of baseline nDCG@10. At factor 10: 99.2% (260M) / 98.9% (800M). At factor **20** they still keep 97.8% / 98.0%. For comparison, the original text-retrieval paper found factor 3 was the near-lossless boundary and factor 6 already cost ~10%. Page images are far more redundant than text — many patches are whitespace.

**Asymmetric quantization** (documents cheap, queries precise, because you store $N$ documents and encode one query):

| Query | Doc | nDCG@10 (260M) | Storage |
|---|---|---|---|
| fp32 | fp32 | 0.5226 | 1537 kB |
| int8 | int8 | 0.5224 (−0.02) | 390 kB |
| int8 | binary | 0.5068 (−1.58) | 48 kB |
| binary | binary | 0.4960 (−2.66) | 48 kB |

int8 documents are essentially free ($3.9\times$ smaller, ≥99.96% quality retained). Binary documents cost ~1.6 points for $32\times$. Quantising the *query* too costs another point for zero storage benefit — don't do it.

**Combined**: pool factor 8 + int8 queries + binary documents → 1536.7 kB down to **6.0 kB, a $255\times$ compression, retaining 95.19%** of baseline nDCG@10. A gentler point: factor 10 + int8/int8 → 39.0 kB ($39.4\times$) at 99.16%. Compare [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)]] for the classical version of this trade-off.

### Resolution

Dropping the cap 2048 → 1536 costs at most 1.8% quality and uses ~57% the vectors. 1024 costs 3.0–12.7%. 768 costs 11.7–36.1%. **Upsampling small pages *up* to 2048 does not help either head** — so the right policy is downscale-only with a 2048 cap.

### Throughput (median pages/sec, largest effective square)

| Model | Square | Vectors | H100 | L40S | M5 Pro |
|---|---|---|---|---|---|
| NeoMME-260M | 2048 | 4162 | 76.8 | 51.3 | 3.2 |
| NeoMME-800M | 2048 | 4162 | 40.4 | 21.2 | 1.4 |
| ColModernVBERT | 2048 | 1149 | 53.3 | 26.0 | 2.9 |
| Vultron Flash | 1344 | 1775 | 35.2 | 20.2 | – |
| ColQwen2.5 | 756 | 740 | 42.7 | 16.1 | 1.1 |

Note the *vectors* column: NeoMME produces $3.6\times$ more vectors per page than ColModernVBERT and is still $2\times$ faster. Query latency (batch 1) is 21.0 ms on L40S, 78.3 ms on a 128-core CPU, 15.9 ms on an M5 Pro — second only to ColModernVBERT everywhere.

### What did not work

- **48-pixel patches with a linear stem and learned coordinate embeddings** (a Gemma-4-inspired variant): weakened text-reading ability at the 260M scale. The guess is that each token had to compress too large a page region. 32 pixels + MLP won.
- **Frozen natural-image representations are bad.** 16-shot frozen probes across 10 natural-image classification tasks average **13.2% accuracy**. Not a typo. Because there is no image-side loss, the model never learns a global image representation — only whatever is needed to help recover masked transcripts. Fine-tuning recovers some ground: 77.1% Food101, 63.9% Oxford Pets, 46.8% Stanford Cars.
- **Document images transfer much better than natural images.** Frozen first-token probe on RVL-CDIP reaches 51.6%; fine-tuning on 6,000 examples (<2% of the training set) hits 81.5%.
- **Language transfer is decent but not competitive.** 17-task GLUE/SuperGLUE/multilingual mean of 75.3 vs 79.3 for the same-size text-only LFM2.5-Encoder-230M. It beats that peer on PAWS-X, MASSIVE Intent, SeaHorse, MRPC, WSC — but CoLA Matthews correlation is only 46.4, a clear weakness on grammatical acceptability.
- **LIK gives memory, not speed, in this recipe.** At NeoMME-260M's training shape, throughput with LIK is 383,614 tok/s vs 377,439 without — 1.6%, inside run-to-run noise. The value is being able to fit the batch at all.
- **Tokenizer coverage collapses outside the target languages.** On the 14 target languages NeoMME emits 44.4% fewer tokens than ModernBERT and 6.3% fewer than mmBERT. Across all 204 FLORES-200 pairs it emits **65.29% *more* tokens than mmBERT**, worst in Tibetan, South Asian, and Southeast Asian scripts. A whitespace-unconstrained SuperBPE vocabulary trained on 21 languages is a narrow tool.

### Image sensitivity probe

They check the model actually uses the image with a cross-modal ablation: run the same masked text twice, once with the real patches $I_t$, once with a zero tensor, and diff masked-token accuracy.

$$G_t = A_t(I_t) - A_t(\bm{0})$$

Positive at every masking rate for both models, and it grows as text context is removed: **+38.4 points at 90% masking for the 260M, +40.5 for the 800M**. Image gain starts near zero and rises during training. This is the load-bearing check that the $\rho \ge 0.30$ floor for multimodal segments did its job.

---

## Worth Remembering

**The masking-rate floor is the quiet load-bearing hyperparameter.** At low corruption, masked tokens are recoverable from neighbouring text, so the image is dead weight and the gradient never asks for it. Raising the floor to 0.30 for multimodal segments forces the image into the prediction. This is a known failure mode of multimodal masked LM, and the fix is simply "mask more."

**No image target is the paper's central bet, and it visibly costs them.** The 13.2% frozen natural-image probe is the price. For visual *document* retrieval it barely matters — pages are mostly text and layout, and masked-transcript recovery is a great proxy task. For anything requiring a global semantic image representation, this backbone is the wrong tool without adaptation.

**Late interaction is doing most of the work, not the backbone.** The gap between the dense and late heads on the *same* forward pass is 13–18 nDCG points. That is consistent with the theoretical result that single-vector embeddings have bounded capacity — a fixed dimension limits which top-$k$ sets can be separated by a score margin. If you take one thing to production from this paper, it is the two-head design: one backbone pass gives you a dense vector for [[Efficient and robust approximate nearest neighbor search using HNSW|ANN]] first-stage retrieval and a multi-vector representation for reranking, with the dense head costing literally only mean-pooling.

**Authors' own admitted limits.** (1) ~524B tokens total vs ModernBERT's ~2T text tokens — roughly $7\times$ less text. (2) **No architecture ablations at all.** Zero-init, token-indexed value embeddings, residual mixing, exclusive self-attention, the 256/1024 window alternation, NorMuon — none of these are isolated. The paper is a recipe, not a study. (3) They trained on mixed text+image retrieval so that text chunks and page images share one embedding space, but never evaluated that setting (UniDoc-Bench and MixBench exist). (4) Visual retrieval data covers only 6 languages, versus 13+ on the text side.

**Practical caveats.** The step-20,000 retrieval checkpoints are what all numbers use. Predecay pretraining checkpoints are released so you can anneal on your own domain — genuinely useful if your documents are not English/French/German/Italian/Spanish PDFs. Everything is Apache 2.0 with a day-zero Hugging Face Transformers implementation. The binary-quantized index only pays off in latency if you write XOR + popcount kernels that work on packed bits; dequantising before scoring throws the advantage away, and the authors did not build those kernels.

**Similarity maps show emergent OCR.** For the query token `hour`, the highest-similarity patches land on the chart title, the x-axis label, and the numbered hours — with no OCR pipeline anywhere in training. Caveat the authors raise: some background-patch responses may be the high-norm "register" tokens known to appear in [[An Image is Worth 16x16 Words (ViT)|ViTs]], not real evidence. Treat the map as a diagnostic, not an explanation.

**Open questions.** Does the diffusion objective actually beat fixed-rate masking here, or is the win entirely from the corruption floor and the data mixture? Would adding *any* image-side loss (contrastive, pixel, or token) rescue natural-image transfer without hurting document retrieval? And would distilling 800M → 260M close the 3.3-point v3 gap for free — the authors flag it as untried.

---

## Links
Related: [[ColBERT- Efficient and Effective Passage Search via Late Interaction]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[An Image is Worth 16x16 Words (ViT)]] · [[RoFormer- Enhanced Transformer with Rotary Position Embedding]] · [[GQA- Training Generalized Multi-Query Transformer Models]] · [[Matryoshka Representation Learning]] · [[Product Quantization for Nearest Neighbor Search (IEEE TPAMI)]] · [[Dense Passage Retrieval (DPR)]] · [[Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Denoising Diffusion Probabilistic Models]] · [[Old Optimizer, New Norm- An Anthology (Muon)]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Mixed Precision Training]] · [[Layer Normalization]] · [[Sentence-BERT]] · [[NDCG]] · [[Attention Is All You Need]] · [[Efficient and robust approximate nearest neighbor search using HNSW]] · [[Cross Entropy]] · [[ELECTRA- Pre-training Text Encoders as Discriminators]]

New topics worth writing: masked discrete diffusion language models (MDLM / MD4 / LLaDA), ModernBERT, ColPali and visual document retrieval, SigLIP and sigmoid contrastive loss, hierarchical token pooling for multi-vector indexes, asymmetric quantization for retrieval, GradCache, NorMuon, Depth-μP and branch scaling, Fixup initialization, sliding-window vs global attention schedules, SuperBPE whitespace-unconstrained tokenization, ViDoRe benchmark family, PLAID multi-vector indexing, MUVERA fixed-dimensional encodings, vision transformer registers
