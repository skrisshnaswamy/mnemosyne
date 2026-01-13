---
title: "Mixed Precision Training"
authors: ["Paulius Micikevicius", "Sharan Narang", "Jonah Alben", "Gregory Diamos", "Erich Elsen", "David Garcia", "Boris Ginsburg", "Michael Houston", "Oleksii Kuchaiev", "Ganesh Venkatesh", "Hao Wu"]
year: 2017
arxiv: "1710.03740"
url: https://arxiv.org/abs/1710.03740
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, llm, optimization, vision]
---
## The Core Idea

Train the whole network in 16-bit floating point (FP16) instead of 32-bit (FP32), and lose nothing. Half the memory, and on hardware with FP16 matrix units, 2–8× the arithmetic throughput. No change to the model, no change to the learning rate, no change to any hyper-parameter.

The reason this had not worked before is **range**, not precision. FP16 has 5 exponent bits and 10 mantissa bits. Normalised values live roughly in $[2^{-14}, 65504]$. Anything smaller than $2^{-24}$ flushes to exactly zero. Real training gradients sit *far* down in that basement — in the Multibox SSD detector, 67% of activation gradient values were already zero in FP16, and a big chunk of the rest lived in $[2^{-34}, 2^{-24})$, i.e. below the floor. So naive FP16 training silently deletes most of the backward signal.

Earlier work on low precision attacked the wrong end: binarise the weights, quantise activations to 2/4/6 bits, but keep gradients in FP32. That saves inference cost, not training cost, and still lost accuracy on large CNNs. Here every tensor in forward and backward is FP16 — weights, activations, gradients — and accuracy matches FP32 on ImageNet classification, object detection, speech, translation, language modelling and GANs.

Three tricks make it work, and they are all about keeping numbers inside the representable window:

1. **FP32 master weights** — a shadow copy of the weights in full precision that the optimiser actually updates.
2. **Loss scaling** — multiply the loss by a constant $S$ before backprop, so all gradients shift up into FP16 range; divide it back out before the update.
3. **FP32 accumulation** — FP16 multiplies, but the running sum inside a dot product is FP32.

> [!NOTE] Loss scaling
> Multiply the loss by a constant $S$ before backpropagation. By the chain rule every gradient in the network is then scaled by exactly $S$, for free — no extra ops. Unscale the weight gradients by $1/S$ before the optimiser touches them, so learning rate, weight decay and gradient-clipping thresholds all stay unchanged. ^loss-scaling

## The Methodology

**The training loop, one iteration.** Keep the master weights $W_{32}$ in FP32. Then:

1. Cast $W_{32} \to W_{16}$.
2. Forward pass in FP16 with $W_{16}$. Activations stored FP16.
3. Multiply the loss by $S$.
4. Backward pass in FP16. Activation gradients and weight gradients are FP16.
5. Unscale the weight gradients by $1/S$ (immediately, before clipping or anything else).
6. Optimiser step on $W_{32}$ in FP32 (momentum, Adam state, weight decay — all FP32).

Memory: you now hold weights twice (FP32 master + FP16 copy), which is $1.5\times$ FP32 weight memory. But weights are not where training memory goes — activations dominate, because of large batches and because every layer's output is cached for [[Backpropagation]]. Activations are FP16, so total memory still roughly halves.

**Why master weights are needed — two separate failure modes.**

*Failure 1: the update itself underflows.* The update is $\eta \cdot g$. In the Mandarin speech run, about 5% of weight gradients had exponents below $-24$. Multiply by a learning rate $< 1$ and they become exactly zero in FP16. Those updates vanish permanently.

*Failure 2: the update underflows during the addition.* Even if $\eta g$ is representable, $W + \eta g$ can equal $W$. FP16 has 10 mantissa bits, so once $|W| / |\eta g| \gtrsim 2048$, aligning binary points right-shifts the update off the end of the mantissa and it contributes nothing. This one bites even when nothing is denormal.

Both disappear if the accumulation happens in FP32.

**Choosing $S$.** They used constants from 8 to 32K; many networks needed none. Rule: pick $S$ so that $S \times (\text{max } |g|) < 65504$. There is **no penalty for picking $S$ too large** except overflow, and overflow is cheap to detect — inf/NaN shows up in the unscaled weight gradients. If you see one, skip the update and move on. (This is exactly the dynamic loss scaler that later shipped in Apex and `torch.amp`; the paper flags it as future work.)

**Arithmetic precision, by operation type.**

- *Dot products* (conv, FC, the matmuls inside recurrent cells): FP16 inputs, FP32 accumulator, result rounded to FP16 on the memory write. Volta Tensor Cores do exactly this. Without FP32 accumulation, some models did not match baseline.
- *Large reductions* — softmax denominators, [[Batch Normalization]] statistics: do the arithmetic in FP32, read/write FP16 from memory. Free, because these layers are memory-bandwidth bound anyway, not arithmetic bound.
- *Point-wise ops* — activations, element-wise products: memory bound, precision is irrelevant to speed. Either works.

**Experimental setup.** FP32 baselines ran on Maxwell/Pascal. Mixed-precision ran on Volta V100. The speech experiments used "pseudo FP16" on Maxwell — FP16 storage, FP32 math — which emulates Tensor Core behaviour, and they checked it matched real Volta runs.

## Ablation Studies and Experiments

**ImageNet classification (top-1, single crop).** No loss scaling needed for any of these.

| Model | FP32 | Mixed |
|---|---|---|
| AlexNet | 56.77 | 56.93 |
| VGG-D | 65.40 | 65.43 |
| GoogLeNet | 68.33 | 68.43 |
| Inception v2 | 70.03 | 70.02 |
| Inception v3 | 73.85 | 74.13 |
| ResNet-50 | 75.92 | 76.04 |

**Detection (mAP, Pascal VOC 2007 test) — the ablation that matters.** This is where loss scaling is isolated:

| Model | FP32 | MP, no loss scale | MP, loss scale |
|---|---|---|---|
| Faster R-CNN | 69.1 | 68.6 | 69.7 |
| Multibox SSD | 76.9 | **diverges** | 77.1 |

SSD is the clean demonstration: without scaling it does not train at all. A scale of just **8** (shifting exponents by 3) fixes it. That tells you something precise about the model — activation gradients below $2^{-27}$ were irrelevant, but values in $[2^{-27}, 2^{-24})$ were load-bearing.

**Master-weight ablation (Mandarin speech, 800h, 20 epochs).** Three runs: FP32; FP16 forward/backward with FP32 master weights; FP16 forward/backward updating FP16 weights directly. The last one suffered **80% relative accuracy loss**. This is the sharpest ablation in the paper — it separates "storage precision" from "update precision" and shows the update is where it breaks.

**Speech recognition (DeepSpeech 2, CER).** English 115M params on 6000h, Mandarin 215M on 2600h. Nesterov SGD, identical hyper-parameters.

| | Baseline | Mixed |
|---|---|---|
| English (WSJ '92) | 2.20 | **1.99** |
| Mandarin (internal) | 15.82 | **15.01** |

Mixed precision was 5–10% *better*. The authors' guess: FP16 storage acts as a regulariser — see [[Regularization]]. Treat this as an observation, not an established mechanism.

**Machine translation.** 3-layer and 5-layer encoder/decoder, 1024 LSTM cells per layer, attention, WMT15 EN→FR, SGD. Loss-scaled MP matched FP32. No loss scaling gave a *slight* degradation — not divergence, just worse. Important detail: three separate FP32 runs with identical settings gave noticeably different curves, so run-to-run variance is comparable to the effect being measured.

**Language modelling (bigLSTM).** Two layers of 8192 LSTM cells, 1024-dim projection, 793K vocab, sampled softmax with 8K negatives, Adagrad, 50 epochs on the 1B-word benchmark. Without loss scaling the perplexity curve tracks FP32 fine and then **diverges after 300K iterations**. Scale of 128 fixes it. The delayed failure is the practical lesson: a short smoke test would have looked healthy.

**DCGAN.** 7 fractionally-strided conv layers in the generator, 6 conv + 2 FC in the discriminator, batch norm, Adam, 100K iterations on CelebA at 128×128. No loss scaling needed. Only qualitative evaluation — they show uncurated samples, which is unusual and honest, since GANs had no accepted quality metric at the time. See [[Generative Adversarial Networks]].

**What the ablations reveal.** The three techniques are not equally important, and their importance is task-dependent:
- FP32 master weights: needed essentially everywhere. The speech ablation shows catastrophic failure without it.
- Loss scaling: needed only when the gradient distribution sits low. All six ImageNet CNNs needed none; SSD, bigLSTM and translation did.
- FP32 accumulation in dot products: "some networks require" it — the paper never names which, which is the weakest-supported of the three claims.

## Worth Remembering

- **Speedups are on kernels, not end-to-end.** DeepBench ops on Volta show 2–6× versus FP32 *when memory- or arithmetic-bandwidth bound*. Latency-bound ops gain much less. The authors explicitly say full-network speedups depend on library and framework work that had not happened yet in 2017. Do not read "2–6×" as wall-clock training speedup.

- **65504 is the number to remember.** Max FP16 value. Overflow produces inf/NaN in weight gradients, which after one update irreversibly destroys the weights. Detect and skip, do not try to recover.

- **Unscale early.** Right after the backward pass, before clipping, before weight decay. Otherwise every gradient-dependent hyper-parameter is off by a factor of $S$ and you are silently tuning a different problem.

- **The dynamic loss scaler is a footnote here.** "Loss-scaling factor could be dynamically increased or decreased by inspecting the weight gradients for overflow" — one sentence in future work, and it became the default in every framework. Standard policy: start high (e.g. $2^{15}$), halve on overflow and skip the step, double every 2000 clean steps.

- **BF16 later made most of this optional.** BF16 has 8 exponent bits — the same range as FP32 — so loss scaling is unnecessary. It pays for that with 7 mantissa bits instead of 10, so it is *less* precise. The FP32 master weight and FP32 accumulator survive into BF16 training unchanged. See [[Mixed Precision training]].

- **Denormals matter.** Part of FP16's usable low range is denormal. If your hardware or compiler flushes denormals to zero, your effective floor rises from $2^{-24}$ to $2^{-14}$ and you will need much more aggressive loss scaling than the paper's factor of 8.

- **The optimiser state is quietly FP32 too.** Momentum buffers, Adam's $m$ and $v$ ([[Adam- A Method for Stochastic Optimization]]) live alongside the master weights. So "mixed precision halves memory" is a claim about *activation-dominated* training. For a model with small activations and huge parameter count, the arithmetic is less favourable — which is what [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models]] later attacks from the other side.

- **Open question the paper leaves.** Why do gradients cluster so far below 1 in the first place? Normalisation layers, initialisation scale ([[Delving Deep into Rectifiers (He init, PReLU)]]) and depth all push the distribution around. A network initialised differently might need a completely different $S$ — which is precisely why the constant-factor approach did not survive.

## Links

Related: [[Mixed Precision training]] · [[Backpropagation]] · [[Batch Normalization]] · [[Adam- A Method for Stochastic Optimization]] · [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models]] · [[Long Short-Term Memory (Neural Computation)]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Generative Adversarial Networks]] · [[Regularization]] · [[Megatron-LM- Training Multi-Billion Parameter Models Using Model Parallelism]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Delving Deep into Rectifiers (He init, PReLU)]]

New topics worth writing: IEEE 754 floating point formats (FP32/FP16/BF16/FP8), denormal numbers and flush-to-zero, dynamic loss scaling, gradient underflow, Tensor Cores and GPU arithmetic throughput, quantisation-aware training, activation checkpointing, roofline model / memory-bandwidth-bound vs compute-bound kernels
