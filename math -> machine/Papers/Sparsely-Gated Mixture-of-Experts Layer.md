---
title: "Sparsely-Gated Mixture-of-Experts Layer"
authors: ["Noam Shazeer", "Azalia Mirhoseini", "Krzysztof Maziarz", "Andy Davis", "Quoc Le", "Geoffrey Hinton", "Jeff Dean"]
year: 2017
arxiv: "1701.06538"
url: https://arxiv.org/abs/1701.06538
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, transformers, llm, vision, scaling]
---
## The Core Idea

A neural network learns more when it has more parameters. But normally every parameter is used on every example, so doubling the parameters doubles the compute. This paper breaks that link.

The trick: put a layer in the network that contains thousands of small feed-forward networks ("experts"), plus a tiny router ("gating network") that picks only **4** of them for each token. The other 4092 sit idle and cost nothing. So the layer holds billions of parameters but only does a few million multiply-adds per token.

> [!NOTE] Conditional computation — only part of the network runs for any given input, chosen by a learned router. The parameter count and the FLOP count become independent knobs. ^conditional-computation

The idea of mixtures of experts is from 1991 (Jacobs, Jordan, Hinton). What did **not** exist before is a version that actually runs fast on a GPU cluster and actually beats state-of-the-art. Four practical walls had blocked it:

1. **Shrinking batches.** If a batch of $b$ examples is split across $n$ experts with $k$ picked each, each expert only sees $\frac{kb}{n}$ examples. GPUs are terrible at tiny matrix multiplies.
2. **Network bandwidth.** Experts live on different machines. You have to ship activations back and forth.
3. **Load imbalance.** The router converges to loving a handful of experts. Those get more gradient, get better, get picked more. A self-reinforcing rich-get-richer loop.
4. **Not enough data.** Prior work tested on 600k images. You cannot train a billion parameters on that.

They fix all four and get **>1000× more capacity** for roughly the same compute. On the 1-billion-word language benchmark they beat the previous best (Jozefowicz et al. 2016, 34.7 perplexity) with **34.1 perplexity using 6% of the compute**. This is the paper that made every later MoE model — Switch Transformer, Mixtral, DeepSeek-V3 — possible.

## The Methodology

**The layer.** Given input $x$, with $n$ experts $E_1 \dots E_n$ and gate $G$:

$$y = \sum_{i=1}^{n} G(x)_i \, E_i(x)$$

When $G(x)_i = 0$ you skip computing $E_i(x)$ entirely. That is the whole saving.

Each expert here is a plain 2-layer feed-forward net: $512 \to 1024$ ReLU $\to 512$. About 1M parameters each. Identical shape, separate weights.

**Noisy Top-K gating.** The router is one weight matrix, plus a second one that controls noise:

$$H(x)_i = (x \cdot W_g)_i + \mathcal{N}(0,1) \cdot \text{Softplus}((x \cdot W_{noise})_i)$$

$$G(x) = \text{Softmax}(\text{KeepTopK}(H(x), k))$$

`KeepTopK` sets everything outside the top $k$ to $-\infty$, so softmax gives them exactly 0. Two things to notice:

- The gate values for the surviving $k$ experts have real gradients, so the router trains by ordinary [[Backpropagation]] alongside everything else. No REINFORCE, no straight-through estimator. This is why $k > 1$ matters — with $k=1$ softmax over a single value is constant and the router gets no signal from the gate weights.
- The noise is not regularisation. It exists so that "how likely is expert $i$ to get picked" is a smooth, differentiable quantity. More on that below.

**Fixing the shrinking batch: mix data and model parallelism.** Run $d$ devices *synchronously*. The LSTMs and the router are replicated (data-parallel). The experts are **sharded** — exactly one copy of each expert, sitting on one device. Every device sends its routed tokens to whichever device owns the chosen expert. Now each expert sees $\frac{kbd}{n}$ examples instead of $\frac{kb}{n}$. A factor of $d$ recovered. Add more GPUs → more experts → same per-device memory, same step time.

**Convolutional application.** The same MoE layer is applied at every time step between two LSTM layers. Since layer $\ell-1$ finishes all time steps before layer $\ell$ starts, you batch all unrolled time steps together into one MoE call. Another big multiplier on batch size. (Note this breaks if you put the MoE *inside* the recurrence — they flag this as future work.)

**Fixing bandwidth.** Only activations cross the wire; experts stay put. The ratio of compute to communication for an expert equals its **hidden layer size**. So you buy efficiency by making experts wider, not by clever networking.

**Fixing load imbalance: two auxiliary losses.**

*Importance* is the batch-wise sum of gate values for each expert:

$$\text{Importance}(X) = \sum_{x \in X} G(x), \qquad L_{importance} = w_{imp} \cdot CV(\text{Importance}(X))^2$$

where $CV$ is the coefficient of variation (std / mean). Squared CV is zero when all experts get equal total gate mass.

But equal *importance* is not equal *load*. One expert could get 3 examples with weight 0.9; another 100 examples with weight 0.03. Same importance, wildly different memory and compute. So a second loss targets the actual count. Counting is discrete and non-differentiable — which is where the noise pays off. Define $P(x,i)$ as the probability that $x$ would route to expert $i$ if you redrew the noise on component $i$ only:

$$P(x,i) = \Phi\!\left(\frac{(x \cdot W_g)_i - \text{kth\_excluding}(H(x), k, i)}{\text{Softplus}((x \cdot W_{noise})_i)}\right)$$

$\Phi$ is the standard normal CDF. Sum over the batch to get a smooth $\text{Load}(X)_i$, then apply the same squared-CV penalty. The noise turns a hard counting problem into a Gaussian tail probability you can differentiate.

**Initialisation detail worth stealing:** set $W_g$ and $W_{noise}$ to **all zeros**. This gives pure noise at step 0 — perfectly uniform routing — so nothing OOMs before the balancing losses kick in.

**Hierarchical MoE.** With thousands of experts the router's own softmax gets expensive. Split into $a$ groups of $b$. A primary gate picks groups, secondary gates pick within group:

$$y_H = \sum_{i=1}^{a}\sum_{j=1}^{b} G_{primary}(x)_i \cdot G_i(x)_j \cdot E_{i,j}(x)$$

They use $k=2$ at each level, so 4 experts total, same as flat $k=4$. First-level branching factor was set to 16 — the number of GPUs.

**Memory tricks for the 137B-parameter run.** (a) Do not store expert hidden activations; recompute them on the backward pass. (b) Modify [[Adam- A Method for Stochastic Optimization|Adam]] for expert parameters: set $\beta_1 = 0$ to drop the first-moment buffer entirely, and replace the full second-moment matrix with the outer product of its row-means and column-means divided by the overall mean. This is the ancestor of Adafactor.

**Architecture and training.** LM models: embed(512) → LSTM(512) → MoE → LSTM(512) → softmax, with [[Deep Residual Learning for Image Recognition (ResNet)|residual connections]] and [[Dropout- A Simple Way to Prevent Overfitting|dropout]] on every layer output. Adam, linear warmup for 1000 steps then $1/\sqrt{t}$ decay. Importance-sampled softmax (793k vocab). Batches of ~300k words. 16–32 K40 GPUs. $w_{importance} = w_{load} = 0.1$.

Translation models: GNMT cut down to 3 encoder / 2 decoder LSTM layers, MoE inserted between encoder layers 2–3 and decoder layers 1–2. 2048 experts, each $512 \to 2048 \to 512$ (2M params), ~8B total. $w_{importance} = w_{load} = 0.01$.

## Ablation Studies and Experiments

**Capacity at fixed compute (1B Word Benchmark).** All models pinned to ~8M ops/timestep. Test perplexity after 10 epochs:

| Model | Test PPL | Params (excl. embed/softmax) |
|---|---|---|
| LSTM-2048-512 (baseline) | 44.7 | 9.4M |
| 4xLSTM-512 | 46.0 | 8.4M |
| MoE-1-Wide (single 4096 expert) | 46.1 | 8.4M |
| MoE-1-Deep (single 4-layer expert) | 45.7 | 8.4M |
| **MoE-4** (all 4 always active) | 45.0 | 8.4M |
| MoE-32 | 39.7 | 37.8M |
| MoE-256 | 35.7 | 272.9M |
| MoE-1024-h | 34.6 | 1079M |
| MoE-4096-h | **34.1** | 4303M |

The **MoE-4 row is the load-bearing ablation**. Four experts, all always used — no sparsity, same compute, same architecture. It scores 45.0, i.e. identical to the dense baseline. Every point of improvement comes from the parameters you *added but did not use*, not from the mixture structure itself. 24% lower perplexity at 4096 experts.

**More compute on top of high capacity.** MoE-34M → 31.3 PPL; MoE-143M → 28.0 PPL. Big capacity does not saturate the benefit of more FLOPs. The 143M-op model matches the previous SOTA's compute budget and training time while being 18% better on perplexity.

**Where it breaks (100B Word Google News).** Perplexity after 100B words keeps dropping through 65536 experts (68.9B params, 28.9 PPL, 39% below baseline), then **gets worse at 131072 experts** (29.2). The authors' guess: too much sparsity. Also note computational efficiency collapses at that scale — 0.72 TFLOPS/GPU at 65k experts, but **0.30 at 131k**, versus ~1.1 for dense baselines. Part of that is they did not scale batch size with GPU count.

The 1B-word corpus showed diminishing returns past ~1B MoE parameters; the 100B corpus did not. Capacity only pays if you have data to fill it.

**Balancing loss ablation (Table 6, MoE-256).** This one is clean:

| $w_{imp}$ | $w_{load}$ | PPL | CV(Imp) | CV(Load) | max/mean Load |
|---|---|---|---|---|---|
| 0.0 | 0.0 | 39.8 | 3.04 | 3.01 | **17.80** |
| 0.2 | 0.0 | 35.6 | 0.06 | 0.17 | 1.47 |
| 0.0 | 0.2 | 35.7 | 0.22 | 0.04 | 1.15 |
| 0.1 | 0.1 | 35.6 | 0.06 | 0.05 | 1.14 |
| 1.0 | 1.0 | 35.7 | 0.03 | 0.02 | 1.07 |

Two readings. **For quality, either loss alone is enough** — 35.6/35.7 across the board, versus 39.8 with neither. The exact weight barely matters over two orders of magnitude. **For systems, $L_{load}$ is what you need**: without it the worst expert gets 17.8× the average load, which OOMs a real cluster. So the load loss is an engineering fix, not a modelling one.

**Translation.** WMT'14 En→Fr: **40.56 BLEU** vs GNMT+RL's 39.92, and 2.63 perplexity vs 2.79 — with 85M ops/timestep against GNMT's 214M. En→De: **26.03 BLEU** vs 24.91, in one day on 64 K40s. Google production En→Fr: +1.01 test BLEU in one sixth the training time.

**Multilingual (12 language pairs, one model).** Multilingual GNMT loses to 12 separate models. The MoE version gets 19% lower dev perplexity than multilingual GNMT, beats it on **11 of 12** pairs (up to +5.84 BLEU on Ko→En), and beats the 12 *separate* monolingual models on **8 of 12**. The failure: En→Ko, −1.79 BLEU, which they attribute to severe overtraining because rare pairs were heavily oversampled.

**Things that did not work / were worked around:**
- Making a single expert wider (MoE-1-Wide) or deeper (MoE-1-Deep) at the same FLOPs: no gain over baseline. The win is genuinely from sparse capacity.
- 131072 experts: worse than 65536.
- Recurrent MoE (MoE inside the LSTM cell): breaks the batching trick, not attempted.
- Noisy top-K did not fit their infrastructure for single-pair translation, which required exactly equal per-expert batches. They swapped in a **batchwise mask** — keep the top $m = \frac{k|X|}{n}$ values *per expert across the batch* instead of top $k$ per example. This makes training depend on the batch, so at inference they learn a per-expert threshold vector $T$ with an extra loss that matches the threshold mask to the batchwise mask.
- The GNMT attention function was replaced with a cheaper factored form $A(x_i,y_j) = \sum_d V_d \tanh((x_iU)_d)\tanh((y_jW)_d)$ — "little difference in quality", purely for matmul efficiency.

## Worth Remembering

- **Experts specialise semantically, unprompted.** Table 9: expert 381 fires on innovation/research words; expert 752 on the article "a" when it introduces a direct object in a leadership/importance phrase ("plays *a* core", "assume *a* leadership"); expert 2004 on speed words (rapidly, swift, drastically, fastest). Nobody supervised this.
- **The efficiency is real but not free.** Dense baselines ran 1.07–1.29 TFLOPS/GPU; low-compute MoE models ran 0.74–0.90. You pay roughly 25–35% in hardware utilisation for 500× the parameters. The largest model got 1.56 TFLOPS/GPU because bigger matrices amortise better — bigger experts are more efficient experts.
- **The experts are only 37–46% of total FLOPs** even in these models. Most of the compute is still the LSTMs and the softmax.
- **"Theoretically scary discontinuities"** — the authors' phrase. Top-K is not continuous; a tiny input change can flip which expert runs. They just say they never saw it hurt. This is still an unresolved wart in MoE training.
- The stated goal — "train a trillion-parameter model on a trillion-word corpus" — was written in 2017 and reached within about five years.
- **Practical caveat:** this design assumes a synchronous cluster where each expert lives on exactly one device and activations are shipped to it. If your setup is asynchronous data-parallel with parameter servers, none of the batch-size arithmetic works. The systems design is not separable from the model design.
- Applied here between LSTM layers; the modern move is to swap the feed-forward block of a [[Attention Is All You Need|Transformer]] for an MoE layer, which fits the "convolutional over positions" trick even more naturally. Shazeer is an author on both papers.
- Open question the paper leaves: is the right route the sparsity, or would learned soft mixing at the same FLOPs do as well? MoE-4 says dense mixing of 4 experts gains nothing — but it does not test dense mixing of 4096.

## Links

Related: [[Attention Is All You Need]] · [[Long Short-Term Memory (Neural Computation)]] · [[Adam- A Method for Stochastic Optimization]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Megatron-LM- Training Multi-Billion Parameter Models Using Model Parallelism]] · [[ZeRO- Memory Optimizations Toward Training Trillion Parameter Models]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Dropout- A Simple Way to Prevent Overfitting]] · [[Backpropagation]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Seq2Seq models]] · [[The Bitter Lesson (essay)]] · [[Gated Activation]] · [[Random variable]]

New topics worth writing: Switch Transformer, Mixtral / modern sparse LLMs, Expert Choice routing, Adafactor and factored second moments, coefficient of variation as a balance penalty, gradient estimation through discrete choices (straight-through vs REINFORCE vs noise smoothing), activation recomputation / gradient checkpointing, BLEU score, GNMT and wordpiece tokenisation, importance-sampled softmax
