---
title: "GAE: Learning a Geometry-Native Latent Space for 3D-Consistent World Generation"
authors: ["Lu et al."]
year: 2026
arxiv: "2609.24981"
url: https://arxiv.org/abs/2609.24981
priority: Good-To-Read
read_on: 2026-09-27
tags: [paper, transformers, rl, diffusion, vision]
---
## The Core Idea

A video generator can make frames that each look photorealistic while the scene they depict quietly falls apart — walls bend, a table moves, and the camera does not go where you asked. The usual fix is to bolt geometry on: add a camera-control input, add a depth output, add a geometry reward. This paper says the problem is one level lower. It is in the **latent space** — the compressed code the generator actually learns to move through.

Think about what a [[Latent Diffusion|latent diffusion]] model evolves. With a pixel VAE, the latent is about appearance. With a representation autoencoder, the latent is about semantics. In neither case is "how far away is this surface" or "where is the camera" something you can simply *read off* the latent. Geometry has to be inferred from appearance, or injected from outside.

Meanwhile, a different family of models — geometry foundation models like DUSt3R, VGGT and DA3 — already builds internal features from which depth, camera rays and 3D point maps can be read directly. So the trick: **make the generator's latent be a geometry model's features**, compressed. Then perception and generation share one state. Generating RGB and generating 3D are the same act of generation, decoded two ways.

> [!NOTE] Geometry-native latent
> A compressed code that a *frozen* geometry model's own decoder head can still read as depth, camera, and point maps — so 3D structure is directly present in the generated state, not inferred from it afterwards. ^geometry-native-latent

Why did this not already exist? Because geometry foundation models do not have one clean latent. DA3 hands its geometry decoder a **hierarchy of four feature levels**, each doing a different job (fine detail and cross-view matching early, semantic context later). That is fine for perception and awful for generation:

- Model all four levels → you need a cascade of coupled flow models (this is what GLD does).
- Pick one level → you throw away information the geometry head expects.
- Either way the raw features are horrible to generate in: each level has **3,072 channels but only ~11 effective dimensions**, with covariance condition numbers between $10^8$ and $10^{16}$. Almost all channels are dead; the live ones sit at wildly different scales.

So the geometry is already there. Its *parameterisation* is just unfit for a generative model. GAE (geometry-native autoencoder) rewrites that parameterisation into 64 or 128 channels that a single, ordinary [[Diffusion Transformer|DiT]]-style [[Flow Matching|flow]] model can handle.

The payoff, measured with the generator and training recipe held fixed and only the latent swapped: FVD drops **12.7%** on RealEstate10K and **23.1%** on DL3DV, and camera-trajectory error is roughly **halved** on RealEstate10K.

## The Methodology

Two stages. Stage 1 builds the latent. Stage 2 trains a flow model in it.

### Stage 1 — the codec

Start from a **frozen** DA3-GIANT encoder $\mathcal{E}$. For each of $V$ views it emits four levels of patch tokens $\mathbf{f}_{v,\ell} \in \mathbb{R}^{N \times C_\ell}$.

**Fuse the hierarchy.** Normalise each level by fixed per-channel training-set statistics, reshape tokens back onto their spatial grid, and concatenate along channels:

$$\mathbf{X}_v = \big\|_{\ell=0}^{3} \mathrm{reshape}\!\left(\frac{\mathbf{f}_{v,\ell} - \mathbf{m}_\ell}{\mathbf{s}_\ell + \epsilon}\right) \in \mathbb{R}^{C_x \times h_p \times w_p}$$

The per-level normalisation is not cosmetic — without it a few high-variance channels drown the rest.

**Compress the channels only.** A convolutional pyramid with spatial self-attention encodes $\mathbf{X}_v$ to a grid latent $\mathbf{z}_v \in \mathbb{R}^{C_z \times 18 \times 18}$ with $C_z \in \{64, 128\}$. It is a [[Variational Autoencoder|VAE]]-style bottleneck — it outputs $(\bm{\mu}_v, \log\bm{\sigma}_v^2)$ and samples during codec training — but with a *small* [[KL Divergence|KL]] weight. The KL is there to keep the space tidy, not to force a heavily regularised generative prior. For everything downstream they use the deterministic posterior mean $\bm{\mu}_v$.

The patch grid is preserved. Only channels shrink: 3,072 → 128 is a 24× cut.

**Decode twice from the same code.**

$$\hat{\mathbf{I}}_v = \mathcal{D}^{\mathrm{rgb}}_\theta(\mathbf{z}_v), \qquad \hat{\mathbf{G}}_v = \mathcal{H}_{\mathrm{DPT}}\!\left(\mathrm{Dec}_\phi(\mathbf{z}_v)\right)$$

The decoder $\mathrm{Dec}_\phi$ rebuilds **all four feature levels in one pass**, and DA3's original dense-prediction head $\mathcal{H}_{\mathrm{DPT}}$ — kept frozen — reads depth, camera rays and point maps out of them. A separate *learned* head renders RGB straight from the latent.

Freezing $\mathcal{H}_{\mathrm{DPT}}$ is the load-bearing design choice. If the geometry head could be fine-tuned, it would learn to compensate for whatever the bottleneck destroyed, and you would never know the geometry was gone. Frozen, the reconstructed hierarchy must stay readable through the *original* interface.

**The codec loss:**

$$\mathcal{L}_{\mathrm{codec}} = \mathcal{L}_{\mathrm{feat}} + \lambda_{\mathrm{kl}}\mathcal{L}_{\mathrm{kl}} + \mathcal{L}_{\mathrm{rgb}} + \mathcal{L}_{\mathrm{geo}} + \mathcal{L}_{\mathrm{repr}}$$

- $\mathcal{L}_{\mathrm{feat}}$ — weighted per-level feature reconstruction.
- $\mathcal{L}_{\mathrm{rgb}}$ — pixel plus perceptual.
- $\mathcal{L}_{\mathrm{geo}}$ — depth and ray supervision, but against **pseudo-targets**: run the frozen DA3 head on the *original* features and match that, not dataset ground truth.
- $\mathcal{L}_{\mathrm{repr}}$ — the interesting one.

### The two-part representation loss

Reconstruction losses say *what* the latent must keep. They say nothing about *how* it is arranged, and arrangement is what makes a space easy or hard for a flow model to transport through. A reconstruction-only codec here was compact and well-conditioned but still had weak transport smoothness and weak semantic neighbourhoods.

**Token-wise term.** Align each projected latent token to the co-located C-RADIO feature (a [[Distillation|distillation]]-style cosine match to a frozen teacher):

$$\mathcal{L}_{\mathrm{tok}} = \frac{1}{VN}\sum_{v,i}\left[1 - \left\langle \widehat{g_\eta(\bm{\mu}_{v,i})}, \hat{\mathbf{c}}_{v,i} \right\rangle\right]$$

**Relational term.** Match the latent's pairwise cosine-similarity structure to DINOv2's:

$$\mathcal{L}_{\mathrm{struct}} = \frac{1}{VN(N-1)}\sum_v \sum_{i \neq j}\left(\langle \hat{\bm{\mu}}_{v,i}, \hat{\bm{\mu}}_{v,j}\rangle - \langle \hat{\mathbf{d}}_{v,i}, \hat{\mathbf{d}}_{v,j}\rangle\right)^2$$

$$\mathcal{L}_{\mathrm{repr}} = \lambda_{\mathrm{repa}}\left(\mathcal{L}_{\mathrm{tok}} + \lambda_\mu \mathcal{L}_{\mathrm{struct}}\right), \quad \lambda_{\mathrm{repa}} = 0.25,\ \lambda_\mu = 8.0$$

Note $\lambda_\mu = 8$ — the relational term needs a big weight to matter. And unlike REPA, both terms shape the **codec latent itself**, not an intermediate denoiser feature. Teachers and projector are thrown away after Stage 1.

### Stage 2 — the flow model

Freeze everything. Standardise each latent channel by training-set statistics, $\bar{\mathbf{z}}_v = (\mathbf{z}_v - \mathbf{m}_z)/(\mathbf{s}_z + \epsilon)$. Then plain conditional flow matching on the linear path $\bar{\mathbf{z}}_t = (1-t)\bar{\mathbf{z}}^{\mathrm{gt}} + t\bm{\epsilon}$:

$$\mathcal{L}_{\mathrm{flow}} = \mathbb{E}\left[\rho(t)\left\|\hat{\mathbf{u}}_\psi(\bar{\mathbf{z}}_t, t, \mathcal{C}) - (\bm{\epsilon} - \bar{\mathbf{z}}^{\mathrm{gt}})\right\|_2^2\right]$$

The network actually predicts the **clean latent** (following RAEv2), converted to velocity by $\hat{\mathbf{u}}_t = (\bar{\mathbf{z}}_t - \hat{\bar{\mathbf{z}}}^{\mathrm{gt}})/\max(t, 0.05)$. One transformer models all target views jointly, so cross-view interaction happens directly in the geometry-native state.

**Architecture:** 28 encoder blocks of width 768, six decoder blocks of width 2048, ~0.93B trainable params, $1\times1$ latent tokens, cross-view self-attention, frame-axis [[RoPE|rotary embeddings]] for temporal models.

**Three conditioning signals, all kept out of the ODE state:**

1. **Reference views.** Geometry encoders are set-conditioned — encode an image alongside the targets and you get different features than encoding it alone. So if you stuff a "full-set" reference latent into the flow state during training, you are using information that does not exist at test time. Fix: encode the $K$ observed references using *only each other* as context, give those clean latents timestep $t=0$, prepend them as tokens that everything attends to, and strip them before the prediction head. Every output slot — even one at a reference camera pose — stays an ordinary noisy flow variable. One plain Euler sampler, no clamping.
2. **Cameras.** Per-latent-cell world-space Plücker rays $(\mathbf{d}, \mathbf{m})$, decomposed into direction $\mathbf{d}$, normalised moment $\hat{\mathbf{m}}$, and a log-scale channel $s = \log(\|\mathbf{m}\| + \epsilon)$ that retains **metric** translation magnitude. These modulate query/key [[Attention|self-attention]] (SCoPE-style), so pairwise token interaction depends on the actual camera geometry.
3. **Text.** Frozen Qwen3-0.6B features through [[Cross Attention|cross-attention]].

Independent condition dropout (text 0.10, camera 0.05, reference 0.05) gives one checkpoint that does text-to-image, camera-controlled video, and reference-conditioned novel view synthesis — and supplies the unconditional branch for [[Classifier-Free Guidance|CFG]].

**Training:** BF16, [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] with $\beta = (0.9, 0.95)$, **zero** weight decay, gradient clip 1.0, EMA decay 0.9995, 2k warmup to $10^{-4}$ then cosine to $10^{-6}$ over 100k steps. A single-view text-to-image batch interleaved every three multi-view updates. Eight GPUs per controlled run.

## Ablation Studies and Experiments

**The controlled setup is the strongest thing about this paper.** Seven latents are dropped into the *same* flow architecture, data mixture (equal RealEstate10K + DL3DV), optimiser schedule and sampling protocol. Only the encoder/decoder and the input/output projections change. Evaluation: 64 held-out scenes, nine views at $252^2$, one reference, 50 Euler steps, CFG 2.

**Generation quality (FVD ↓ / FID ↓):**

| Latent | RE10K FVD | RE10K FID | DL3DV FVD | DL3DV FID |
|---|---|---|---|---|
| SD-VAE (pixel, image) | 258.6 | 36.4 | 373.2 | 57.1 |
| WAN2.1 VAE (pixel, video) | 362.9 | 41.0 | 596.7 | 77.3 |
| RAEv2 (semantic) | 379.4 | 31.4 | 453.4 | 49.2 |
| Raw DA3 L0 | 298.6 | 35.5 | 376.5 | 60.2 |
| Raw DA3 L3 | 488.9 | 50.0 | 584.8 | 80.6 |
| **GAE-128** | 233.4 | **25.6** | 345.2 | 44.2 |
| **GAE-64** | **225.7** | 27.1 | **287.0** | **41.3** |

**3D consistency, judged by VGGT (a model that shares no backbone with any of these latents):** camera ATE on RealEstate10K goes 0.0072 (SD-VAE) → **0.0034** (GAE-64), a 52.8% cut. On DL3DV, 0.0073 → 0.0056, 23.3%. MEt3R (feature disagreement between reprojected view pairs) best controlled result 0.1208 on RE10K and 0.1347 on DL3DV.

**Reconstruction sanity check.** GAE-128 gets PSNR 28.76 / LPIPS 0.036 single-view, slightly *beating* raw DA3 L0 (28.64 / 0.038) while using 24× fewer channels. Geometry reconstruction: camera ATE 0.006 vs raw-L0's 0.068 — an order of magnitude — because the raw baseline must propagate L0 back through the frozen backbone to recover the other levels, whereas GAE reconstructs all four directly.

### What the ablations actually reveal

**The token loss on its own is a trap.** This is the most instructive table in the paper:

| Objective | $\rho$ ↓ | LNC@5 ↑ | LDS ↑ | SRSS ↑ |
|---|---|---|---|---|
| None | 0.820 | 0.310 | 0.240 | 0.300 |
| $\mathcal{L}_{\mathrm{tok}}$ | 0.700 | 0.490 | **0.020** | **0.030** |
| $\mathcal{L}_{\mathrm{tok}} + \mathcal{L}_{\mathrm{struct}}$ | 0.674 | 0.498 | 0.444 | 0.553 |

Token-wise alignment improves transport smoothness and semantic neighbourhoods — and **destroys spatial structure**, dropping LDS from 0.240 to 0.020 and SRSS from 0.300 to 0.030. Aligning every token independently is perfectly compatible with wrecking every relationship *between* tokens. The pairwise DINOv2 term puts that back without giving up the token-level gains. A clean lesson for anyone doing representation-alignment losses: per-item supervision does not constrain relational geometry, and relational geometry is what "3D consistent" means.

**Conditioning ablations (RealEstate10K):**

| Setting | FVD ↓ | PSNR ↑ | Diagnostic ↓ |
|---|---|---|---|
| Reference latent inside ODE state | 257.1 | 18.91 | 0.1016 (ref–tgt point-cloud gap) |
| Clean conditioning token | **225.7** | **20.02** | **0.0318** |
| Per-scene normalised pose | 350.7 | 16.28 | 0.0346 (SE(3) ATE, no Sim(3)) |
| Metric pose | **225.7** | **20.02** | **0.0093** |
| No T2I co-training | 472.9 | 16.77 | 0.199 (LPIPS) |
| With T2I co-training | **225.7** | **20.02** | **0.143** |

Three separate findings. (1) Putting the reference latent in the ODE state triples the reference-to-target point-cloud gap — the set-encoding train/test mismatch is real and expensive. (2) Normalising camera poses per scene, which is standard practice, throws away translation scale and costs **125 FVD points**; conditioning on metric Plücker rays instead cuts scale-sensitive ATE by 3.7×. (3) Interleaving text-to-image batches more than halves FVD — the multi-view data alone does not supply a broad enough appearance prior.

**$\mathcal{L}_{\mathrm{repr}}$ transfers:** GIANT-128 without it scores FVD 266.4; with it, 233.4. Latent-space diagnostics predicted downstream generation quality here, which is not always true.

**Backbone capacity is not the main lever.** DA3-LARGE-128 with the full objective gets FVD 270.5 — worse than GIANT-128's 233.4 — though LARGE keeps a small MEt3R edge (0.1192 vs 0.1213). Representation shaping and backbone size gave complementary, not interchangeable, gains.

**What did not work / what stayed weak**

- **Raw DA3 L3 was a disaster across the board**: FVD 488.9, PSNR 19.90, LPIPS 0.219, LDS 0.054. The intuition that "deeper features are more semantic and therefore better to generate" is wrong here — L3 has lower $\rho$ (smoother transport) but its condition number is $6.6 \times 10^{16}$ and it has lost the spatial and cross-view structure. Lower transport cost bought with destroyed structure is not a bargain.
- **Raw DA3 L0 kept the best cross-view correspondence of anything tested** (xLNC* 0.642, above GAE's 0.591). Compression cost them a little here and they do not hide it.
- **GAE-64 beats GAE-128 at generation while losing at reconstruction** (PSNR 27.30 vs 28.76, rFVD 9.1 vs 4.3). Another data point for the now-familiar result that reconstruction fidelity does not determine generative quality.
- **Gen3R, an external system, still wins some depth metrics** on RealEstate10K — but it uses a separate geometry latent, which is exactly the design GAE argues against.

## Worth Remembering

**The honest framing.** This is a representation paper dressed as a world-model paper. The claim is narrow and well-supported: *hold the generator fixed, swap the latent, and 3D consistency moves*. That controlled-comparison discipline is worth copying regardless of whether you care about 3D video.

**Geometry targets are pseudo-targets.** $\mathcal{L}_{\mathrm{geo}}$ supervises against the frozen DA3 head applied to the original features — not against dataset depth. So GAE inherits DA3's errors by construction. Its ceiling is DA3, in exactly the way a latent diffusion model's ceiling is its VAE.

**The evaluation is carefully de-confounded, and they say so.** Depth and point maps are compared against Pi3 run on the *real* target frames, not against geometry estimated from each method's own generated images. Camera metrics come from VGGT (independent) with DA3-GIANT reported separately as a "backbone-overlap cross-check". This is the right way to evaluate a system whose latent *is* a perception model — and easy to get wrong.

**The effective-rank finding stands alone.** A 3,072-channel feature spanning ~11 effective dimensions with $\kappa \sim 10^{16}$ is worth internalising as a general caution: channel count tells you nothing about how many directions carry variance. If you are ever tempted to diffuse in a pretrained backbone's raw feature space, measure the spectrum first. This is the same anisotropy story as the sentence-embedding literature, just in a geometry backbone.

**Practical caveats if you wanted to use this.** Stage 1 needs three frozen teachers (DA3, C-RADIO, DINOv2) plus a perceptual loss — expensive, though the teachers are discarded afterwards. The showcased 81-view, $672\times378$ rollouts come from a *separately trained* model on 80 GPUs with seven datasets; those results are explicitly qualitative and not part of the ranking. The long-rollout point clouds are just 81 unprojected depth maps concatenated, with no cross-view fusion — impressive precisely because nothing stitches them.

**Open question the paper raises but does not settle.** They frame the latent as "a shared interface between perception and generation." If that is true, the interesting follow-up is the other direction: can a generator trained in this space *improve* the perception model, rather than just borrowing from it? Nothing here tests that.

## Links

Related: [[Latent Diffusion]] · [[Flow Matching]] · [[Diffusion Transformer]] · [[Variational Autoencoder]] · [[Video Diffusion]] · [[Conditional Generation]] · [[Classifier-Free Guidance]] · [[Denoising Objective]] · [[Rectified Flow]] · [[Neural ODE]] · [[Curse of Dimensionality]] · [[Distillation]] · [[KL Divergence]] · [[Evaluating Generative Models]] · [[JEPA]] · [[Self-Supervised Learning from Images with I-JEPA]] · [[Understanding Dimensional Collapse in Contrastive Learning]] · [[Whitening Sentence Representations]] · [[Representation Degeneration Problem in Training NLMs]] · [[Cross Attention]] · [[RoPE]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Latent Variable Models]]

New topics worth writing: geometry foundation models (DUSt3R / VGGT / DA3 lineage), representation autoencoders for generation (RAE, RAEv2), REPA and representation-alignment losses, latent diffusability diagnostics (transport ratio $\rho$, effective rank, LDS, SRSS), Plücker ray camera conditioning, FVD and MEt3R as multi-view consistency metrics, point maps and Sim(3) trajectory alignment (ATE / RPE), novel view synthesis, world models for video
