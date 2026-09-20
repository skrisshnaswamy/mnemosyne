---
title: "Why Gated DeltaNet Survives 4-Bit Quantization: NVFP4 W4A4 for the Recurrent Half of a Hybrid 27B LLM"
authors: ["Sergii Kozyrev", "Davyd Maiboroda"]
year: 2026
arxiv: "2609.04098"
url: https://arxiv.org/abs/2609.04098
priority: Good-To-Read
read_on: 2026-09-06
tags: [paper, transformers, llm]
---
## The Core Idea

Modern big models are **hybrids**: most layers are not softmax attention but a cheap "linear attention" layer that keeps a fixed-size memory matrix and updates it token by token. In Qwen3.8-27B, 48 of the 64 layers are **Gated DeltaNet** (GDN) and only 16 are real attention.

Everyone who shipped a 4-bit version of this model refused to touch the GDN block. The reasoning felt obvious: a recurrence carries its state across 32,000 tokens, so a rounding error made at token 5 should still be sitting in the state at token 30,000, and errors in the *gates* — the numbers that decide how much to forget and how hard to write — should compound worst of all. So every public recipe left GDN at 8 or 16 bits and only crushed the MLPs.

This paper shows that intuition is exactly backwards. They quantized **all 496 linear layers**, gates included, to NVFP4 W4A4 (4-bit weights *and* 4-bit activations), calling the result *Minima*. It matches BF16 within seed noise on six benchmarks, is the smallest checkpoint (17.5 GiB vs 50.1 GiB), and has the fastest prefill. More interestingly, the perplexity gap to BF16 **shrinks as the context grows**: $+0.081$ nats in the first half of a 32K window, $+0.011$ in the second half, and *negative* in the last 2K tokens.

The mechanistic story is the real contribution, and it is four links long:

1. NVFP4 gives every 16 numbers their own scale, so a giant outlier only ruins its 15 neighbours instead of the whole tensor.
2. The gates are computed in log space through `softplus` and `exp`/`sigmoid`. These squash an $\sim11\%$ error in the raw matrix multiply down to $\sim2\%$ error in the layer's output. The gate projections turn out to be the **least** sensitive tensors in the layer.
3. The delta rule does not just accumulate. Each write *overwrites* the state along the direction of the current key, so old errors get deleted key-by-key rather than merely decayed. A 1% state error injected at token 1024 dies to $1/10$ within ~2,200 steps, even in layers whose decay gate implies a 62,000-token memory horizon.
4. So end-to-end, the 4-bit cost is a *per-token* cost that a filled state absorbs, not a debt that piles up.

> [!NOTE] Gated delta rule
> The GDN state update is $S_t = \alpha_t S_{t-1} + \beta_t k_t (v_t - S_{t-1}^\top k_t)^\top$. The term $v_t - S_{t-1}^\top k_t$ is a *prediction error*: what the state currently says about key $k_t$, versus what it should say. The write corrects that error rather than blindly adding $v_t$. This self-correcting write is why injected noise is erased, not accumulated. ^gated-delta-rule

## The Methodology

**The architecture being quantized.** Qwen3.8-27B, hidden size 5120, 48 GDN layers interleaved with 16 full-attention layers. Each GDN layer has five weight matrices:

- `in_proj_qkv` — produces $q_t, k_t, v_t$ (through a depthwise causal conv and SiLU)
- `in_proj_z` — the output gate
- `in_proj_a` — the decay gate
- `in_proj_b` — the write-strength gate
- `out_proj` — mixes back into the residual stream

The gates are parameterised in log space:

$$g_t = -\exp(A_{\log})\,\operatorname{softplus}(a_t + \text{dt\_bias}), \qquad \alpha_t = e^{g_t} \in (0,1), \qquad \beta_t = \sigma(b_t) \in (0,1)$$

Per-head state $S_t \in \mathbb{R}^{128\times128}$, output $o_t = S_t^\top (q_t/\sqrt{K})$, then RMSNorm modulated by $z$ (see [[Layer Normalization]], [[Gated Activation]]).

> [!NOTE] NVFP4
> Values stored as E2M1 (4 bits: 1 sign, 2 exponent, 1 mantissa), with one E4M3 8-bit scale per **16-element block** (set to $\text{blockmax}/6$), plus one FP32 scale per tensor. W4A4 = both sides of the GEMM in this format, so it runs on native 4-bit tensor cores. Two facts matter: an outlier only inflates the scale of its own block of 16, and inside a block $\max/\text{RMS} \le \sqrt{16} = 4$. ^nvfp4

**The recipe.** llm-compressor NVFP4 W4A4 on every linear layer — 240 GDN, 64 attention, 192 MLP = 496 total. Left in BF16: embeddings, `lm_head`, the GDN `conv1d`, all norms, and the scalar parameters $A_{\log}$ / `dt_bias`. Calibrated on a frozen set of 128 samples × 32K tokens. Post-training only — no fine-tuning, no [[Distilling the Knowledge in a Neural Network|distillation]].

**Serving.** vLLM 0.27.1 ([[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]]), TP=1, one RTX PRO 6000 96 GB, FP8 KV cache, one fixed regime for every number in the paper.

**How the mechanism study was done.** They captured the real inputs of all 48 GDN layers as the BF16 model read eight 32K-token documents. Then they re-implemented one GDN layer standalone in pure PyTorch (verified to $6\times10^{-3}$ median relative difference against the reference kernel, i.e. BF16 rounding noise). On this they ran **fake quantization**: quantize to NVFP4, dequantize back, keep computing in high precision. That injects exactly the 4-bit rounding error and nothing else — no kernel differences, no dtype confounds. Three probes:

- *Per-projection sensitivity*: quantize exactly one of the five matrices at a time, 96 replays over layers and sequences.
- *Lockstep recurrence*: run one clean FP32 trajectory and eleven perturbed ones on identical inputs for 32K tokens, on five layers spread over depth, and track relative state error $\mathrm{relS}(t)$.
- *Positional NLL decomposition*: split the per-token loss of a 32K perplexity run into 2K-token bins.

## Ablation Studies and Experiments

**Main table** (BF16 / Minima / Unsloth / RadixArk, same harness, FP8 KV):

| | BF16 | Minima | Unsloth | RadixArk |
|---|---|---|---|---|
| PPL @4K / @32K | 6.95 / 10.35 | 7.67 / 10.84 | 7.16 / 9.91 | 7.35 / 9.95 |
| MMLU-Pro | 80.4 | 79.7 | 78.9 | 79.1 |
| GSM8K | 95.5 | 95.5 | 95.4 | 95.7 |
| AIME'25 (pass@1, 4 seeds) | 86.7 | 86.7 | 87.5 | 84.2 |
| GPQA-Diamond | 86.5 | 85.1 | 85.0 | 85.4 |
| LiveCodeBench v6 | 79.0 | 78.5 | 79.9 | 79.6 |
| 5-task avg | 85.62 | 85.10 (−0.52) | 85.34 (−0.28) | 84.80 (−0.82) |
| VRAM weights | 50.13 GiB | **17.53** | 20.23 | 18.83 |
| TTFT @32K | 6.90 s | **4.03 s** | 4.49 s | 4.39 s |

No pair of models is separated by confidence intervals on any task. RULER retrieval is 100 for all four at 32K and 64K. Minima matched BF16's AIME score exactly (26/30 on all four seeds) with the whole recurrent half at 4 bits, and did not start "thinking longer" (mean generation 14,531 vs 14,532 tokens).

**Ablation 1 — are GDN inputs easier? No.** GDN reads the same residual stream as attention, with the same ugly statistics: median-layer $\max/\text{RMS} = 63.5$, kurtosis ~1,560, 10.6% of 16-element blocks dominated by one value. Yet the measured activation quantization error is *uniform across all layer roles*, 7.5–9.2%, because block scaling contains each outlier. Weight error (10.5–11.9%) actually **exceeds** activation error everywhere. So the robustness is not in the data — it is in what the layer does with the error.

**Ablation 2 — per-projection sensitivity.** This is the table that inverts the community's precision map. Median relative errors in %:

| quantized | GEMM err | err on $1-\alpha$ | err on $\beta$ | state $S$ | output $y$ |
|---|---|---|---|---|---|
| `a` (decay gate) | 11.0 | 7.5 | — | 3.6 | **2.1** |
| `b` (write gate) | 8.5 | — | 5.2 | 3.2 | **2.6** |
| `ab` (the pair everyone protects) | — | 7.5 | 5.2 | 5.2 | 3.6 |
| `qkv` | 10.6 | — | — | 12.1 | 10.4 |
| `z` | 8.3 | — | — | 0 | 9.9 |
| `out` | 12.7 | — | — | 0 | 12.7 |
| all | — | 7.5 | 5.2 | 12.6 | 19.2 |

The two tensors every public recipe keeps in BF16 are the two smallest contributors. The errors from the five projections are also statistically **independent** — summing the single-projection $y$ errors in quadrature gives 19.4%, and the all-at-once measurement is 19.2%. Nothing grows along the sequence (first quarter 19.5%, last quarter 19.7%).

**Ablation 3 — does the state error accumulate? No.** With full Minima quantization injected at every step, $\mathrm{relS}$ is 12.96% at token 256 and 12.31% at token 32,768. Plateau 12.6%, max 14.9%. It reaches equilibrium immediately — forgetting balances injection — and holds it for the whole window.

**The impulse test.** A single 1% state perturbation at $t_0=1024$ decays to $1/e$ in 80–1,382 steps and to $1/10$ in ~2,200–2,900 steps. The decay gates alone would predict horizons $1/(1-\alpha)$ of **44,000–62,000 tokens** for those same layers. The extra 20× erasure is the delta rule overwriting the state along each new key.

**What *does* break it — the synthetic noise arm.** Apply 0.1% multiplicative noise **directly to $\alpha$** and the state error hits 22%. Apply 1% and it hits 43%. This is the real fragility: with $\alpha \approx 1$, a tiny $\delta\alpha$ is a huge relative change to the horizon $1/(1-\alpha)$. Quantizing `a` produces only 3.6% state error from an 11% GEMM error *because the noise lands on the pre-activation*, where softplus and the exponential compress it first. The log-space parameterisation was chosen for training stability; it happens to be a quantization shield. Noise on $\beta$ is harmless outright (1% noise → 0.4% state error), because a mis-scaled correction gets corrected by later writes.

**Ablation 4 — the positional decomposition.** Weight cost (Minima − BF16, matched KV): $+0.081$ nats first half, $+0.011$ second half, $-0.053$ in the final 2K bin. The KV-cache cost behaves the *opposite* way — small, rising with position, and ~3× larger for the quantized model — which is the signature of an attention-path effect, not a weight effect.

**The KV ablation.** FP8 KV (scale 1.0) is free on tasks — no score moves outside seed spread across two models, four seeds, six suites — and gives 1.8–1.9× more cacheable tokens. Its one cost is $+0.13$ PPL at 32K for BF16 and $+0.41$ for Minima. Adding calibrated static per-tensor FP8 scales (32 tensors on the 16 attention layers) drops PPL@32K from 10.84 to 10.50, recovering **83%**; the residual $+0.07$ is *below* BF16's own uncalibrated penalty. Throughput identical within 0.4%.

**What did not work / what nearly faked a result.** Four serving-stack bugs, each of which silently corrupted a number first:

- **Fused-GEMM scale mismatch.** llm-compressor calibrates one FP32 global scale per *module*; vLLM serves `qkv`+`z` as one fused GEMM and `b`+`a` as another, taking the **max** of the constituent global scales without rescaling the local ones. The paired scales differed by 1.82× and 2.75× in *all 48 layers*, so the served model computed its gates with mis-scaled weights. The corrupted model is deceptively plausible: AIME drops to 80.8, but PPL@32K becomes a flat **6.86 — better than BF16's 10.84** — because a broken forget gate makes the state hold everything, which happens to help next-token prediction on WikiText. Fixed checkpoint-side by rewriting each fused group to a shared global scale and folding the ratio into the per-block E4M3 scales.
- **Multimodal composite serving path.** The hub checkpoint is a composite; serving it makes vLLM take a multimodal position-encoding path even for pure text, changing PPL@32K by 0.18. All models here served from text-only extractions.
- **Raw-completion harnesses are invalid for thinking models.** lm-eval's `local-completions` sends no chat template, so "thinking disabled" never reaches the model. It opened `<think>` on 25–48 of 50 sampled MMLU-Pro questions and got truncated, producing ±40–60 point per-subject swings in *both* directions and an invalid headline score of 66.3 for BF16 (true value 80.4).
- **Context inversion in the base model.** BF16 Qwen3.8-27B scores the same tokens *worse* inside a 32K request (10.35) than in isolated 4K windows (6.95), deterministically, in both vLLM and the reference implementation, while retrieval stays 100% at 64K. Not a quantization artefact — but it means "PPL@32K" only means anything within one fixed serving path.

## Worth Remembering

- **The practical recipe is one line:** quantize everything including GDN, serve FP8 KV, ship calibrated KV scales.
- **The general lesson** is that a nonlinearity between the GEMM and the thing that matters acts as a quantization firewall. Softplus + exp on the decay pre-activation compresses 11% into 3.6%. If you are deciding what to protect in a new architecture, ask *where the error lands relative to the squashing function*, not *how important the tensor sounds*.
- **The fragility is real but the parameterisation hides it.** Direct 0.1% noise on $\alpha$ is catastrophic. A recurrent mixer with a *linearly*-parameterised decay would not enjoy this protection. Do not port the conclusion blindly to other SSM-style layers.
- **Decode is weight-bandwidth-bound**, so all three quantized models land within 4% of each other on decode tokens/s despite different sizes. Minima actually *trails* RadixArk by 2–4% on decode, attributed to small-batch NVFP4 activation-quantization overhead — a kernel artefact, not a recipe property. The win from quantizing GDN shows up in **prefill** (+14–19% at 8K) and in VRAM.
- **The fused-scale bug is a warning about the whole calibrate-then-serve pipeline.** The calibration tool and the serving kernel disagreed about module boundaries, and the failure mode *improved* one headline metric. It only bites recipes that quantize GDN, which is why nobody had hit it.
- **Limitations the authors admit:** one model, one format, tested to 32K perplexity and 64K retrieval — 128K behaviour is extrapolation. The `Minima+scales` task scores are inherited, not re-measured. A concurrent quantization-aware-training checkpoint (QUASAR) reaches the same configuration by learning the 4-bit weights via distillation from the BF16 teacher; this paper's point is that the training is *not necessary* — plain post-training calibration gets there.
- **Open question:** the GDN block is 5.5B parameters, ~23% of decode weight bytes. If the recurrent half is the easy half, does the sub-4-bit frontier live entirely in the MLPs? The authors' follow-up work says yes — it takes MLPs to 3-bit codebooks while keeping GDN and attention **sealed at 4-bit**.

## Links

Related: [[Mixed Precision Training]] · [[QLoRA- Efficient Finetuning of Quantized LLMs]] · [[Efficient Memory Management for LLM Serving with PagedAttention (vLLM)]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[GQA- Training Generalized Multi-Query Transformer Models]] · [[Fast Transformer Decoding- One Write-Head is All You Need (MQA)]] · [[Long Short-Term Memory (Neural Computation)]] · [[Gated Activation]] · [[Layer Normalization]] · [[Attention Is All You Need]] · [[ML Infrastructure]] · [[GPU processing]] · [[Distilling the Knowledge in a Neural Network]] · [[Mixed Precision training]]

New topics worth writing: Gated DeltaNet, the delta rule for linear attention, NVFP4 and microscaling data formats, post-training quantization (GPTQ / AWQ / SmoothQuant), KV-cache quantization (KVQuant, KIVI), Mamba2 and state-space duality, RULER long-context benchmark, quantization-aware training, activation outliers in LLMs, softplus parameterisation for gate stability
