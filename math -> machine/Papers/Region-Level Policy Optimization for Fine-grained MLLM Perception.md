---
title: "Region-Level Policy Optimization for Fine-grained MLLM Perception"
authors: ["Shi et al."]
year: 2026
arxiv: "2609.19745"
url: https://arxiv.org/abs/2609.19745
priority: Good-To-Read
read_on: 2026-09-27
tags: [paper, transformers, llm, rl, vision]
---
## The Core Idea

Multimodal models get better at small details when you feed them more pixels, and more pixels means more visual tokens. Tokens are expensive twice over: once in the vision encoder, once in the language model's [[Prefill and Decode|prefill]]. But the answer usually lives in a tiny patch of the image, so uniform high resolution is mostly paying to encode background.

The new observation is that the job splits into two steps with **different resolution needs**:

1. **Localize** — find where the evidence is.
2. **Recognize** — read what it says.

The authors measure this instead of asserting it. On ZoomBench with Qwen3.5-4B, they take 238 questions the model already answers correctly through a predict-a-box-then-read-the-crop pipeline. Then they degrade one stage at a time. Shrink the *scene* the box is predicted from, and 82.4% of cases still come out right at 6× downsampling. Freeze the box and shrink the *crop* instead, and only 51.3% survive at 6×. At matched survival, localization tolerates roughly **3–4× stronger token compression** than recognition.

> [!NOTE] Resolution-decoupled inference
> Find the region of interest from a cheap, blurry view of the whole image; spend your token budget only on that region at high resolution. ^resolution-decoupled

That shifts the problem onto the region-of-interest (RoI) predictor. Existing options are all unsatisfying. Letting the model itself decode box coordinates is trainable from answers but costs a full model pass per query. A small proposal head distilled from the model's own [[Attention|attention]] maps (SD-RPN) is fast — but it is trained on attention as a stand-in for truth, so it keeps distracting regions and drops weakly-attended evidence, and nothing ever checks whether the proposed region actually helped the answer.

Why can't you just train the head on the answer? Because the path from RoI map to answer is **not differentiable**. The map is binarized, connected regions are extracted, one crop is chosen, and that crop is re-encoded. There is no gradient thread from the answer back to the heatmap. And you cannot supervise it geometrically either, because in fine-grained VQA there is no single ground-truth box — the useful region is whatever the question happens to need.

**Vision-RL²** fixes this by treating coherent image regions as *actions* in a reinforcement learning problem. A frozen copy of the MLLM acts as a reader and scores each region by how much removing it hurts the likelihood of the gold answer. That score becomes a reward. Only the small proposal head is trained — no region labels, no sampled responses, no reasoning traces.

## The Methodology

### Starting point

The predictor is SD-RPN: reuse the first $B$ frozen MLLM blocks as a backbone ($B=21$ for Qwen3.5, 18 for Qwen2.5-VL-7B, 27 for Gemma-4-12B), attach **three trainable blocks**, and produce a dense RoI logit map from the last prefill token:

$$Z_\theta = G\!\left(Q_{\text{RoI}} K_{\text{RoI}}^\top\right), \qquad P_\theta = \sigma(Z_\theta)$$

where $Q_{\text{RoI}}$ comes from the last prefill token's hidden state, $K_{\text{RoI}}$ from the refined visual features, and $G$ reshapes onto the $H_g \times W_g$ visual-token grid. RL starts from the already-trained SD-RPN checkpoint.

### Regions as actions

Treating each grid cell as an independent binary action gives $2^{H_g W_g}$ masks and credits isolated pixels — useless, because cropping needs spatially connected evidence. So the action unit is a **region**: smooth the map with a Gaussian, threshold at 30% of the peak, take connected components.

A set of kept regions becomes a mask $M$ (cell-wise union). $I[M]$ is the image with everything outside the mask replaced by the per-channel mean colour.

### The reward: does removing this region break the answer?

The frozen reader $p_\phi$ is fed $I[M]$ and teacher-forced on the gold answer $y^\star$. Score = geometric-mean token probability, pushed through a logit:

$$P_\phi(M) = \exp\!\left(\tfrac{1}{T}\sum_{t=1}^{T}\log p_\phi(y^\star_t \mid I[M], q, y^\star_{<t})\right), \qquad h_\phi(M) = \mathrm{logit}\,P_\phi(M)$$

This is just the answer's [[Perplexity|per-token likelihood]] under a masked image, on an unbounded additive scale. A region's value is its **leave-one-out contribution**:

$$\Delta_\phi(R \mid M_{\text{ref}}) = h_\phi(M_{\text{ref}}) - h_\phi(M_{\text{ref}} \ominus R)$$

clipped to $[-5, 5]$. Positive means removal hurt, so the region carried evidence. Cost is $n+1$ reader passes for $n$ regions — no subset search. This is the same move as leave-one-out feature attribution in [[A Unified Approach to Interpreting Model Predictions (SHAP)|SHAP]], repurposed as a reward instead of an explanation.

> [!NOTE] Functional contribution
> A region is scored by what it *does* to the answer, not by how well it matches a box someone drew. ^functional-contribution

### Subtractive group — prune bad proposals

Take the top $K \le 6$ predicted regions ranked by mean logit $z_\theta(R) = \frac{1}{|R|}\sum_{u \in R} Z_\theta(u)$. The full prediction is the reference mask $M^p$; each region's contribution is $\Delta^p_k = \Delta_\phi(R^p_k \mid M^p)$.

Problem: the reader's likelihood wobbles a bit even when you remove genuinely irrelevant pixels, and the size of that wobble varies per image and per model. So they measure the noise floor directly. Grow two **control regions** in the background (where $P_\theta < 0.02$, area matched to the median predicted region), *add* them to the mask, and see how much the score moves:

$$b = \min\!\left(\kappa \max_{R \in \mathcal{R}^c}\left|h_\phi(M(\mathcal{R}^p \cup \{R\})) - h_\phi(M^p)\right|,\; 1\right)$$

with $\kappa = 1.25$ (4B) or $1.0$ (others).

The differentiable channel is a softmax over removal actions. Removing region $k$ scores $-z_k$; keeping everything scores $0$:

$$\pi^{\text{sub}}_\theta(a_k \mid x) = \frac{\exp(-z_k)}{1 + \sum_{j \in \mathcal{K}}\exp(-z_j)}$$

Nothing is sampled — all removals are evaluated. This policy exists purely to route credit into the dense map. The advantage and loss:

$$A^p_k = \frac{b - \Delta^p_k}{s^p + 1}, \qquad \mathcal{L}_{\text{sub}} = -\frac{1}{|\mathcal{K}|}\sum_{k \in \mathcal{K}} A^p_k \log \pi^{\text{sub}}_\theta(a_k \mid x)$$

$s^p$ is the standard deviation of the contributions in the group — so it is group-relative normalisation, the same variance-reduction trick as in [[GRPO]], with the group being *regions in one image* rather than samples of one prompt. If $\Delta^p_k > b$, removal gets negative credit and $z_k$ is pushed up (keep it). Otherwise confidence drops.

### Additive group — recover missed evidence

Pruning alone cannot add back evidence the head never proposed. So candidates are drawn from frozen response-to-image attention maps at layers 7, 11, 15, 19, 23, 27 (spread across depth, because the informative layer varies per sample). Regions outside $M^p$ are merged across layers; the top $J \le 4$ become $\mathcal{R}^s$. Reference mask is now $M^{\text{aug}} = M(\mathcal{R}^p \cup \mathcal{R}^s)$, and credit flows through mean inclusion log-likelihood $\ell^+_\theta(R) = \frac{1}{|R|}\sum_{u \in R}\log P_\theta(u)$:

$$A^s_j = \frac{\Delta^s_j}{s^s + 1}, \qquad \mathcal{L}_{\text{add}} = -\sum_{j=1}^{J} A^s_j\, \ell^+_\theta(R^s_j)$$

No noise margin here, on purpose: excluding a worthless candidate is already the zero-gradient default, so a margin would only suppress weak-but-real recoveries.

Note the deliberate asymmetry in credit structure. Removals are mutually exclusive, so they compete through a shared softmax normaliser. Additions are independent yes/no decisions, so each is judged alone.

### Stabilisation and full objective

$$\mathcal{L}_i = \mathcal{L}_{\text{KL},i} + w_i\left(\mathcal{L}_{\text{sub},i} + \mathcal{L}_{\text{add},i}\right) + \mathbb{1}[K_i=1]\,\mathcal{L}_{K1,i}$$

- $\mathcal{L}_{\text{KL}}$: per-cell Bernoulli [[KL Divergence|KL]] to a frozen copy of the initial SD-RPN map — a leash, exactly as in [[RLHF]].
- $w_i$: detached attainability weight. If the reader cannot get the gold answer under *any* tested mask, within-group normalisation would amplify pure noise, so scale the loss by the best attained likelihood divided by its EMA (decay 0.99), clipped to $[0,3]$.
- $\mathcal{L}_{K1}$: when only one region exists there is no relative removal, so a BCE term holds the proposal in place.

Only $\theta$ (the three attached blocks) updates. The MLLM is frozen throughout.

### Sparse visual encoding

An inference-time addition, orthogonal to the RL. Take the bounding box of the predicted foreground, measure its foreground occupancy $\rho_{fg}$, and re-encode the crop at zoom $\eta = \min(\sqrt{1/\rho_{fg}}, \eta_{\max})$ — then **drop the background tokens before they reach the encoder**. Crop area grows by $\eta^2$, so retained token count $\rho_{fg}\eta^2 N_{\text{crop}} \approx N_{\text{crop}}$: same budget, evidence viewed $\eta\times$ sharper.

The detail that makes it work: position embeddings (both vision encoder and LLM) are assigned on the *full* bbox grid before dropping, so the sparse token set is positionally identical to a dense encoding of the same crop. Multiple regions get one enclosing box rather than separate crops, so their spatial relations survive. Source-image tokens are not re-encoded — their [[KV Cache|KV cache]] from the proposal pass is reused in some LLM layers.

### Training data

7K QA pairs from the VisualCoT corpus: 5K InfographicVQA, 1K TextVQA, 1K DocVQA. From 10K candidates per split, samples are ranked by the **standard deviation of their region-removal rewards** under the initial SD-RPN and drawn from the top half — i.e. keep the images where region choice actually matters. One epoch, AdamW, batch 32, peak LR $1.5\times10^{-5}$, cosine, 20% linear warmup, source images capped at 576 visual tokens during training.

## Ablations Studies and Experiments

### Main table (16,384-token source limit, six benchmarks)

| Model | Avg | V*Bench | ZoomBench |
|---|---|---|---|
| Gemini-3.1-Pro | 79.3 | 88.0 | 61.2 |
| Qwen3.5-397B | 77.4 | 88.0 | 57.2 |
| Qwen3.5-9B base | 74.6 | 83.8 | 54.9 |
| Vision-OPD-9B (full finetune) | 78.7 | 90.6 | 65.1 |
| **Vision-RL² 9B** | **80.1** | **95.3** | **68.4** |
| Qwen3.5-4B base | 70.1 | 85.9 | 51.5 |
| Vision-OPD-4B | **76.0** | 90.6 | 59.5 |
| Vision-RL² 4B | 75.3 | 91.1 | 65.1 |

The 9B model tops the table while training only the attached head; everything it is beating (DeepEyes, ZwZ, P2R, Vision-OPD) fully finetunes the MLLM. On Qwen2.5-VL-7B it hits 71.0 avg vs ZwZ's 69.9. On encoder-free Gemma-4-12B: base 63.3 → SD-RPN 69.0 → 72.8.

**Where it loses, consistently:** the MME-RealWorld splits. 4B gets 65.8/65.3 vs Vision-OPD's 74.2/70.6. The authors are blunt about why — full finetuning can improve the *reader*, and a frozen reader cannot fix recognition failures. This is the price of the design, not a tuning issue.

### Efficiency

Under the training-aligned protocol, Vision-RL² sits above base, SD-RPN and Vision-OPD at **every** token budget on all six benchmarks. Concretely:

- 4B at 576 tokens beats 4B base at 4,096 tokens by >3 points, on ~¼ the tokens.
- Matches SD-RPN's 4,096-token accuracy with **4.2× fewer** visual tokens (4B); within half a point with 2.5× fewer (9B).
- 4B at 576 tokens beats Vision-OPD-4B at its largest budget while responding **2.6× faster**.

Latency breakdown at the 4,096-token source limit (RTX A6000), 6-benchmark avg / seconds per sample:

| | 4B | 9B |
|---|---|---|
| Base | 67.8 / 0.62s | 70.9 / 0.92s |
| SD-RPN | 71.1 / 0.86s | 76.1 / 1.40s |
| **Ours** | **75.3 / 0.95s** | **78.7 / 1.47s** |
| ViCrop (attention routing) | 71.6 / 2.04s | 74.6 / 2.98s |
| Self-decoded box | 69.1 / 2.22s | 75.4 / 3.05s |

The routing interfaces that need a full model pass spend 1.1–1.8s per query purely on localization. The RPN head routes in **31–49ms** — roughly 30× cheaper — because it rides on the answer call's own prefill.

### Ablations (Qwen3.5-4B, 576-token limit, 6-benchmark avg)

| Variant | Avg | $\Delta$ |
|---|---|---|
| Base model | 56.2 | — |
| SD-RPN | 66.6 | −3.0 |
| **+ region-level RL** | **69.6** | — |
| + sparse encoding | **71.1** | +1.5 |
| Reward → generation accuracy | 67.3 | −2.3 |
| Reward → raw mean log-prob | 68.1 | −1.5 |
| Actions → singleton *and* pair removals | 69.7 | +0.1 |
| Actions → cell-level keep-sets | 68.4 | −1.2 |
| No control-region margin ($\kappa=0$) | 68.5 | −1.1 |
| No additive group | 68.7 | −0.9 |

What this tells you:

- **The reward shape is the single most load-bearing choice.** Binary correctness (−2.3) fails because when every action in a group produces the same right-or-wrong outcome, there is no relative credit at all — a graded likelihood always separates the regions. This is the same variance problem that makes [[Reward Function|sparse rewards]] hard everywhere.
- The logit transform matters too (−1.5 for raw log-prob): without it, samples where the reader is already saturated dominate the gradient scale.
- **Region granularity is real.** Budget-matched cell-level keep-sets, sampled from the map logits and scored as complete masks, lose 1.2. Coherent connected regions are the right action unit.
- **Pair removals are wasted compute.** +0.1 for substantially more reader passes, so leave-one-out stays.
- Both credit-assignment pieces earn their keep, but modestly: noise margin −1.1, additive recovery −0.9.
- RL and sparse encoding are complementary — one changes *which* evidence is proposed, the other *how the fixed crop budget is spent on it*.

### Versus token pruning (Qwen2.5-VL-7B)

The honest comparison against the other route to token efficiency. At a 25% token budget on text-dense tasks, VisionZip retains 56.5% of base OCRBench accuracy and 73.3% of InfoVQA; Vision-RL² holds 82.1% and 99.1%. But at 25% on **ChartQA** pruning wins (79.1% retention vs 71.7%) — charts need global layout, and at one-sixth source resolution uniform token selection preserves more structure than region re-allocation. Also worth noting: pruning still pays the full vision-encoder bill and only saves LLM tokens; routing saves both.

### The diagnostic's own controls

Worth reading, because the naive version of that probe is confounded. Along the localization ladder the re-predicted box *grows* (median 7.4× area at $r=6$), and a bigger box buys a bigger native-resolution crop. So they re-ran it with the crop hard-capped at 128 tokens: localization still holds 69.5% vs recognition's 51.3% at $r=6$ — an 18.2-point gap. They also filtered out questions answerable from text alone (via a first-token margin) to stop the language prior from carrying the answer as image information vanishes.

Box behaviour under compression is instructive: median IoU against the anchor box collapses from 0.73 to 0.11, but median ground-truth *coverage* rises from 0.44 to 1.00. Localization does not stay precise — it fails gracefully by getting sloppy and wide, which downstream cropping absorbs.

## Worth Remembering

**The structural insight is transferable.** When a discrete choice sits between your differentiable module and your loss, you can replace the missing gradient with a frozen evaluator's measured response to interventions. Here the intervention is masking a region and the measurement is gold-answer log-odds. Nothing about that recipe is vision-specific — it applies to any component that decides *where to spend compute*.

**This is a bandit-shaped problem, not a trajectory-shaped one.** One step, a handful of actions, all of them evaluated exhaustively rather than sampled. There is no rollout, no [[Credit Assignment|temporal credit assignment]], no value function. The "policy" softmax exists only as a differentiable conduit for credit. Calling it RL is defensible but the machinery it actually borrows is group-relative advantage normalisation plus a [[KL Divergence|KL]] anchor.

**The noise-floor calibration is the part most people would skip and shouldn't.** Measuring how much the reader's likelihood moves when you add *irrelevant* background, per sample, and requiring a region to beat that — it is worth 1.1 points and it is the difference between a reward and a coin flip on borderline regions.

**Limitations the authors state:**
- The frozen reader is a hard ceiling. If recognition is the failure, this method cannot help. MME-RealWorld shows exactly that.
- The compression-asymmetry probe is one backbone, one benchmark. Treat it as a measurement, not a law.
- Training needs question–answer pairs (to compute gold likelihood) but *not* region annotations. That is a real reduction in labelling cost, not the elimination of it.

**Practical caveats if you wanted to use this:**
- Training cost per sample is $n+1$ reader forward passes for the subtractive group, plus more for the additive group — cheap compared to [[Fine-Tuning|full finetuning]] with sampled rollouts, but not free.
- You need the SD-RPN supervised checkpoint first; this is a refinement stage, not a from-scratch recipe.
- The sparse-encoding position-embedding trick (assign on the full grid, *then* drop) is easy to get wrong and would silently degrade quality if you dropped first.
- Sample selection by reward standard deviation is doing quiet work — training on images where region choice is irrelevant produces no gradient signal.

**Open question:** the reward is gold-answer likelihood under teacher forcing, which is a proxy for the model actually answering correctly. The generation-accuracy ablation shows the proxy is *better* than the true objective here (because it is graded), but it also invites the usual [[Reward Hacking|Goodhart]] risk — a region that raises answer likelihood without containing the evidence. Nothing in the paper probes for that.

## Links
Related: [[Attention]] · [[Saliency]] · [[A Unified Approach to Interpreting Model Predictions (SHAP)]] · [[GRPO]] · [[Policy Gradient]] · [[Simple Statistical Gradient-Following Algorithms (REINFORCE)]] · [[KL Divergence]] · [[RLHF]] · [[Reward Function]] · [[Credit Assignment]] · [[Distillation]] · [[Distilling the Knowledge in a Neural Network]] · [[Fine-Tuning]] · [[Prefill and Decode]] · [[KV Cache]] · [[Cost and Latency]] · [[Perplexity]] · [[Cross Entropy]] · [[Sparse Attention]] · [[Test-Time Compute]] · [[Reward Hacking]] · [[Evals]]

New topics worth writing: Region Proposal Networks, visual token pruning (VisionZip / DART), thinking-with-images RL, privileged-view distillation, straight-through and score-function estimators for discrete choices, leave-one-out attribution as a reward signal, dynamic-resolution vision encoders (NaViT / Qwen-VL native resolution)
