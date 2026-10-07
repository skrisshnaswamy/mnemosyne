---
title: "Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer (muP)"
authors: ["Yang et al."]
year: 2022
arxiv: "2203.03466"
url: https://arxiv.org/abs/2203.03466
priority: Good-To-Read
read_on: 2026-09-25
tags: [paper, transformers, llm, theory]
---
## The Core Idea

Hyperparameters do not survive a change of model size. Tune the learning rate on a 100M-parameter Transformer, use it on a 7B one, and you get a diverged run or a mediocre one. So everybody re-tunes at every scale, which for large models means you cannot tune at all — one sweep costs more than the model.

This paper says the problem is not "big models are different". The problem is that the standard way of writing a neural network is **badly scaled**, and once you fix the scaling, the optimal hyperparameters stop moving.

The fix is a set of per-layer rules for how initialisation variance, learning rate, and multipliers should change as width grows. It is called **Maximal Update Parametrization**, or **μP**. Under μP, the optimal learning rate of a width-256 Transformer is also the optimal learning rate of a width-8192 Transformer. So you tune the small one and copy the numbers over. No tuning of the big model at all. They call the procedure **μTransfer**.

> [!NOTE] μP (Maximal Update Parametrization)
> A rule for how to *change* initialisation scale, per-layer learning rate, and per-layer multipliers when width changes, chosen so that (a) every activation vector keeps coordinates of size $\Theta(1)$ at every step of training, (b) the network output stays $O(1)$, and (c) **every** weight matrix still changes by a meaningful amount in the infinite-width limit. ^maximal-update
>
> Note the word *change*. μP does not tell you what learning rate to use at width 1024. It tells you what happens to it when 1024 becomes 4096.

Why this was not already known: the guess "small model HPs roughly work for big models" has always been folklore, and it has always been hit-or-miss. It is hit-or-miss because people reparametrise only the learning rate, and leave initialisation scale, embedding multipliers, and output-layer learning rate alone. As Appendix C shows, if you get *one* hyperparameter's scaling wrong, the others are forced to compensate for it, and their optima blow up with width anyway. You have to fix all of them together.

The intuition is just the Central Limit Theorem. Suppose you want to minimise

$$F_n(c) = \mathbb{E}_{x_1,\dots,x_n}\, f\big(c(x_1+\cdots+x_n)\big)$$

over the scalar $c$, where $x_i$ are iid zero-mean unit-variance. Written in $c$, the optimum shifts as $n$ grows. Written as $c = \alpha/\sqrt{n}$, the objective converges to $\mathbb{E} f(\mathcal{N}(0,\alpha^2))$, and the optimal $\alpha$ stops moving. Same optimum, different coordinates. μP is that change of coordinates for a neural network: $x_i$ are the random weights, $c$ is the learning rate, and $f$ is the loss after training.

What it unlocks, concretely: they tuned a 13M-parameter proxy and beat published Megatron BERT-large (350M), with total tuning cost equal to *one* BERT-large pretraining run. And they tuned a 40M proxy and beat GPT-3 6.7B, with tuning cost 7% of pretraining.

A second, less-advertised payoff: under μP, **wider is always better**, at every point in training. Under standard parametrization that is false — the width-8192 Transformer in Figure 1 is worse than the width-4096 one even after its learning rate is tuned.

## The Methodology

### What breaks in standard parametrization

Take a one-hidden-layer linear net $f(x) = V^\top U x$ with $V, U \in \mathbb{R}^{n\times 1}$. Standard parametrization (SP) initialises $V_\alpha \sim \mathcal{N}(0, 1/n)$, $U_\alpha \sim \mathcal{N}(0,1)$, which makes $f(x) = \Theta(|x|)$ at init. One SGD step with learning rate 1 gives $V' = V + \theta U$, $U' = U + \theta V$ for some $\Theta(1)$ scalar $\theta$. Now

$$f(x) = (V^\top U + \theta\, U^\top U + \theta\, V^\top V + \theta^2 U^\top V)\,x$$

and $U^\top U = \Theta(n)$ by the law of large numbers. **The output blows up linearly in width after a single step.** Figure 5 shows exactly this in a real Transformer: logits and attention logits grow with width after one Adam step.

In μP the same net has $V_\alpha \sim \mathcal{N}(0, 1/n^2)$, $\eta_V = 1/n$, $\eta_U = n$, giving

$$f(x) = (V^\top U + \theta n^{-1} U^\top U + \theta n\, V^\top V + \theta^2 U^\top V)\,x = \Theta(1).$$

You might think you can rescue SP by shrinking the global learning rate. You cannot. The word embeddings in a Transformer update by a **width-independent** amount per step (Figure 5, right), so if you shrink the global learning rate enough to stop the logits exploding, the embeddings stop learning. Some layers are moving too fast and others too slow, and one global knob cannot fix two problems in opposite directions. That imbalance is the whole story.

### The scaling table

The rules depend only on which of a tensor's dimensions grow with width. Purple in the paper = differs from SP; here I give SP in brackets.

| | Input weights & all biases | Output weights | Hidden weights |
|---|---|---|---|
| Init. variance | $1/\mathrm{fan\_in}$ | $1/\mathrm{fan\_in}^2$  (SP: $1/\mathrm{fan\_in}$) | $1/\mathrm{fan\_in}$ |
| SGD LR | $\mathrm{fan\_out}$  (SP: $1$) | $1/\mathrm{fan\_in}$  (SP: $1$) | $1$ |
| Adam LR | $1$ | $1/\mathrm{fan\_in}$  (SP: $1$) | $1/\mathrm{fan\_in}$  (SP: $1$) |

Plus one architectural change for Transformers: attention logits are $q^\top k / d$, **not** $q^\top k/\sqrt{d}$.

"Input weights" means anything mapping a *finite* dimension (vocab size, context length, the constant 1 for a bias) to an *infinite* one (width). "Output weights" is the reverse. "Hidden weights" have width on both sides — so in a Transformer, $W^q, W^k, W^v, W^o$ and both MLP matrices are all hidden weights, and get Adam LR $\eta / \tilde{d}_{model}$.

[[Layer Normalization|Layernorm]] gains and biases count as input weights: initialise to 1 and 0 as usual, constant learning rate.

### Why those exponents

This is the part worth internalising, because it lets you derive μP for a new architecture yourself. Everything follows from one fact about matrices times vectors, where the vector is **correlated** with the matrix (as activations always are with weights during training):

| $A$ is… | standard Gaussian $n\times n$ | a sum of outer products $n\times n$ | a vector $1 \times n$ |
|---|---|---|---|
| entry size of $Av$, given $v$ has $\Theta(1)$ entries | $\Theta(\sqrt n)$ | $\Theta(n)$ | $\Theta(n)$ |

A Gaussian matrix meets a vector through the Central Limit Theorem, so you get $\sqrt n$. A gradient update — which *is* a sum of outer products $\sum_i u^i (v^i)^\top$ — meets a correlated vector through the Law of Large Numbers, so you get $n$.

Consequence: the initial weights $W_0$ want entries of size $\Theta(1/\sqrt n)$, but the **update** $\Delta W$ wants entries of size $\Theta(1/n)$. Those are different scalings for the same tensor. SP uses one number for both, which is precisely the bug.

Adam makes this easy to arrange, because Adam normalises coordinatewise: its update has entries of size $\Theta(\eta)$ regardless of gradient magnitude. So you just set $\eta = \Theta(1/n)$ for hidden weights. (For [[Adam- A Method for Stochastic Optimization|Adam]], the update is not literally an outer product but a "nonlinear tensor product" $A_{\alpha\beta} = \psi(u^1_\alpha,\dots,v^k_\beta)$; the same LLN argument goes through.)

### Setting the constants

The table gives scalings, not values. To keep backward compatibility you pick a **base width** $n_0$ and insert ratios $\tilde n = n/n_0$, so that at $n = n_0$ the model is byte-for-byte standard. For the MLP:

$$W^3 \sim \mathcal{N}(0, 1/(n\tilde n)),\quad \eta_{W^1} = \eta_{b} = \eta\tilde n,\quad \eta_{W^2} = \eta,\quad \eta_{W^3} = \eta\tilde n^{-1}.$$

They use $n_0 = 128$ in the toy experiments. The released `mup` package stores a `p.infshape` on each tensor recording its base dimension and whether each dimension is "infinite" (scales) or "finite" (fixed, e.g. vocab size), then divides learning rates by `width_mult() = fan_in / base_fan_in`.

### The experiments

**IWSLT14 De-En.** Target = the default fairseq post-LN Transformer, $d_{model}=512$, 40M params. Proxy = the same thing with all widths cut 4×, 4M params. Random search over three HPs: learning rate, output multiplier $\alpha_{output}$, attention-key multiplier $\alpha_{attn}$. Each whole tuning trial repeated 25 times to get percentiles.

**WMT14 En-De.** Target = the big Transformer from [[Attention Is All You Need]], 211M. Proxy shrinks $d_{model}$ 1024→256, $d_{ffn}$ 4096→256, $n_{head}$ 16→4, giving 15M.

**BERT.** "BERT-prototype": 10 layers, $d_{model}=d_{ffn}=256$, 8 heads with $d_{head}=32$, 13M params. One proxy for *both* [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]]-base and BERT-large, because depth is scaled too. Six HPs tuned: Adam LR, embedding LR, $\alpha_{output}$, $\alpha_{attn}$, layernorm-gain multiplier, bias multiplier. 256 random samples × $10^5$ steps ≈ one BERT-large pretraining in FLOPs. Baseline is [[Megatron-LM- Training Multi-Billion Parameter Models Using Model Parallelism|Megatron]] BERT.

**GPT-3 6.7B.** Proxy = width 256, 40M params, **168× smaller**. 467 proxy runs: 350 trained on 4B tokens, 117 on 16B tokens (the target gets 300B). Search space: LR $\sim 10^{U(-4,-1)}$, init scale $\sim 10^{U(-1,1)}$, attention temperature, output temperature, embedding multiplier, relative-position-embedding multiplier. Chosen: LR 0.006, init scale 2.5, embedding multiplier 10, rest left at 1. Tuning cost as a fraction of pretraining:

$$\frac{s(t_1 N_1 + t_2 N_2)}{ST} = \frac{40\text{M}(4\text{B}\cdot 350 + 16\text{B}\cdot 117)}{6.7\text{B}\cdot 300\text{B}} \approx 0.07.$$

## Ablation Studies and Experiments

### Does the optimum actually stop moving

Figure 1 is the headline. Transformers of increasing $d_{model}$ on wikitext-2 with Adam: in SP the optimal learning rate slides left by roughly an order of magnitude from width 256 to 8192 and wider models *underperform* narrower ones; in μP the curves nest cleanly and the minimum stays put. Same story for an MLP on CIFAR-10 (Figure 3) with widths 256→8192.

Figure 4 sweeps four HPs — learning rate, $\alpha_{output}$, init standard deviation, and LR schedule (linear decay, two StepLR variants, cosine, constant, inverse-sqrt) — across width and depth, 5 seeds. Figure 19 does batch size, sequence length, and training steps.

### The headline numbers

**IWSLT14**, equal total tuning compute:

| Setup | #samples | 25th | 50th | 75th | best |
|---|---|---|---|---|---|
| fairseq default | — | — | — | — | 35.40 |
| tune the 1x model directly | 5 | 33.62 | 35.00 | 35.35 | 35.45 |
| naive transfer from 0.25x (SP) | 64 | diverged | | | |
| **μTransfer from 0.25x** | 64 | **35.27** | **35.33** | **35.45** | **35.53** |

Note the 25th percentile: the worst quarter of μTransfer trials beat the *median* direct-tuning trial. The compute–performance Pareto frontier (Figure 6) dominates conventional tuning everywhere.

**WMT14**: fairseq default 26.40 BLEU; direct tuning with 3 samples gave a diverged run and a best of 25.69; μTransfer with 64 proxy samples gave worst 25.94 / median 26.34 / best 26.42.

**BERT**:

| Model | Method | Test loss | MNLI (m/mm) | QQP |
|---|---|---|---|---|
| base | Megatron default | 1.995 | 84.2/84.2 | 90.6 |
| base | naive transfer | diverged | | |
| base | μTransfer | **1.970** | 84.3/84.8 | 90.8 |
| large | Megatron default | 1.731 | 86.3/86.2 | 90.9 |
| large | μTransfer | **1.683** | 87.0/86.5 | 91.4 |

Model-level speedup 4× (base) and 22× (large); 40× and 220× including the shortened training horizon.

**GPT-3 6.7B**: validation cross-entropy 1.98 with μP vs 2.03 for a from-scratch re-run with the original HPs. PTB perplexity 11.39 vs 13.00, WikiText-103 8.56 vs 9.13, LM1B 20.51 vs 21.70. HellaSwag zero-shot 72.0 vs 66.7 (published 6.7B: 67.4; published 13B: 70.9). Across the eval suite the μP 6.7B is roughly **comparable to the twice-as-large 13B model**.

### What did not work

This is the honest half.

- **Regularization hyperparameters do not transfer.** Dropout probability, weight decay. The reason is structural, not fixable by tuning: how much regularization you need depends on the ratio of model size to *data* size, and μP only knows about model size. So finetuning a pretrained model on a small dataset is outside what this buys you.
- **Initialisation standard deviation does not transfer across depth** (Figure 4, row 2, column 3), even though it is stable across width. Their practical workaround in the BERT runs was to freeze init std and transfer everything else.
- **Depth transfer only works for pre-LN Transformers.** For post-LN it is "fragile". Width, batch size, sequence length and training time still transfer for post-LN (Figure 17). This connects to [[On Layer Normalization in the Transformer Architecture|the pre-LN/post-LN literature]].
- **Minimum scales are required.** Below roughly width 256, depth 4, batch size 32, sequence length 128, and 5000 steps, the proxy is too noisy to read.
- **Squashing activations degrade transfer.** With `tanh` (Figure 9), narrow networks saturate the nonlinearity more than wide ones, which biases the gradients small and distorts the landscape. μP still beats SP but converges to the stable optimum much more slowly. Recommendation: use `ReLU`, whose derivative depends only on the sign.
- **Small $d_k$ makes the landscape noise.** Figure 10: with $d_k = d_{model}/n_{head}$ shrunk to 8 in the proxy, the optimum for $\alpha_{attn}$ is noisy and transfers badly. Decoupling $d_k$ from $d_{model}$ and pegging it at 32 (or 128) while shrinking everything else fixes it, at almost no speed cost because of CUDA kernel behaviour.
- **Non-Gaussian (e.g. uniform) init** sometimes makes wider models worse. Gaussian does not. Universality says they agree in the limit; non-Gaussianity just slows convergence to it.
- **FP16 broke.** The μP GPT-3 run diverged repeatedly from underflow in the backward pass and had to be trained in FP32. The authors suspect μP picked more aggressive per-layer learning rates on the linear weights. If you use this, budget for [[Mixed Precision Training|loss-scaling]] trouble.
- **Weight decay is incompatible with μP Adam**, but is automatically correct under [[Decoupled Weight Decay Regularization (AdamW)|AdamW]]. Use AdamW.
- One sample-efficiency caveat (Figure 6, right): given the *same number of random HP samples*, tuning the target directly is slightly better than tuning the proxy. The proxy is a noisy estimator of the target's landscape. The gap closes as you take more samples — which you can afford, because each one is cheap.

### What the ablations tell you about the mechanism

Three things carry the result, and they are separable.

1. **Output-layer scaling.** Init variance $1/\mathrm{fan\_in}^2$ and LR $1/\mathrm{fan\_in}$. Without it the logits blow up.
2. **Hidden-weight Adam LR $\propto 1/\mathrm{fan\_in}$.** Without it, $\Delta W$ has $\Theta(1)$ entries and the preactivations grow like $n$.
3. **Input/embedding LR held constant (Adam).** Without it, shrinking the global LR to tame (1) and (2) silently freezes the embeddings.

The paper's own argument that μP is *uniquely* correct rests on the Dynamical Dichotomy Theorem from Tensor Programs IV: any stable parametrization gives either a feature-learning limit or a kernel limit. Kernel-limit parametrizations (NTK parametrization, SP with $\Theta(1/n)$ LR) can't transfer because the limit doesn't learn features — Word2Vec in the NTK limit has trivial performance, and wider is not better. Other feature-learning parametrizations differ from μP only in that some parameter tensors are effectively stuck at initialisation in the limit, which makes *those* tensors' learning rates meaningless in the limit and therefore mismatched against finite width.

## Worth Remembering

**The debugging trick is arguably the most immediately useful thing here.** §I: take the hyperparameters that made your large model diverge, **reverse**-μTransfer them down to a small model, and reproduce the instability there. They did this twice — once tracing a width-8192 divergence to exploding attention logits on a width-256 model, once reproducing repeated blow-ups of a separate 6B production model on a 100M model, then fixing it with a small random search. Figure 21 shows the LR-vs-loss curve shape is identical between real width and "simulated width": a learning rate destabilises a wide model iff it destabilises the narrow model it was transferred back to. The implied claim is that a good fraction of "large Transformers are just fickle" is actually mis-scaled hyperparameters.

**Coordinate checking** is the gradient-check of μP. Implementing the table by hand is error-prone. Plot the average coordinate size of every activation vector over a few training steps, across a wide sweep of widths. A correct implementation keeps them flat; a wrong one shows a vector growing or shrinking with width, exactly like the top row of Figure 5. Bundled in the `mup` package.

**Zero-initialise the output layer and the query projection.** Not for performance — it doesn't help or hurt — but because in μP the network at init is a Gaussian process with variance $\Theta(\sigma^2/\text{width})$, which is nearly zero for the target model and decidedly not zero for the small proxy. That mismatch in initial conditions shifts the optimum. Setting $\sigma=0$ removes it. Same argument for attention logits: at init they are $\mathcal{N}(0, \Theta(1/d_{head}))$, which is 0 in the limit but not in a narrow proxy; zero-init $W^q$ makes them exactly 0 at both scales.

**An accidental explanation for T5.** Optimizers that normalise the update to have the same Frobenius norm as the weight — LARS, Adafactor, Lamb, Nero, Fromage — implicitly produce update entries of size $\Theta(1/\sqrt n)$. That is still $\sqrt n$ too big (you want $\Theta(1/n)$), so they will eventually blow up, but it is much closer than Adam's $\Theta(1)$. This may be why [[Exploring the Limits of Transfer Learning (T5)|T5]] with Adafactor trained 220M→11B on one fixed HP set while [[Language Models are Few-Shot Learners (GPT-3)|GPT-3]] with Adam had to decay its learning rate with model size. §B.3 also notes that Adam's $\epsilon$, if not negligible, must scale as $1/\mathrm{fan\_in}^2$ (added before the sqrt) or $1/\mathrm{fan\_in}$ (after) — a detail that silently breaks things if you left $\epsilon = 10^{-8}$ and went very wide.

**The open theoretical puzzle (§9) is genuinely interesting.** For μTransfer to be *useful*, the proxy width $N$ must be large enough that the HP optimum has converged, but small enough that the loss has *not* converged — otherwise why train the big model. That such an $N$ exists means the HP optimum is a coarse, fast-converging statistic while the learned function is a fine, slow-converging one. Nobody knows why. Open research question.

**What is not addressed:** transfer with respect to *test* loss when regularization is the binding constraint. Everything here targets training loss; it lands on test loss too only because none of their settings overfit. Also, depth is empirical only — no theory — and scaling depth changes the parameter count, so you must re-create the base model shape at each new depth.

**Follow-up.** [[Let's Scale Step by Step- Compute-Efficient Hyperparameter Transfer for Large-Scale Mixture-of-Experts|The MoE μP work]] extends this to mixture-of-experts with multi-head latent attention and Muon, and combines it with a token-budget scaling law — the natural next read if you care about how this holds up in a modern training stack. See also [[Old Optimizer, New Norm- An Anthology (Muon)]] for the norm-based view of why per-layer scalings look the way they do.

## Links

Related: [[Let's Scale Step by Step- Compute-Efficient Hyperparameter Transfer for Large-Scale Mixture-of-Experts]] · [[Accurate, Large Minibatch SGD- Training ImageNet in 1 Hour]] · [[Adam- A Method for Stochastic Optimization]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Understanding the difficulty of training deep feedforward networks (Xavier init)]] · [[Delving Deep into Rectifiers (He init, PReLU)]] · [[On Layer Normalization in the Transformer Architecture]] · [[Attention Is All You Need]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[Old Optimizer, New Norm- An Anthology (Muon)]] · [[Mixed Precision Training]] · [[Layer Normalization]] · [[Multi-Head Attention]] · [[Practical Bayesian Optimization of Machine Learning Algorithms]] · [[Bayesian Optimization]] · [[GPU processing]] · [[Regularization]] · [[Exploring the Limits of Transfer Learning (T5)]] · [[Megatron-LM- Training Multi-Billion Parameter Models Using Model Parallelism]]

New topics worth writing: Maximal Update Parametrization as a standalone concept note, Tensor Programs and the infinite-width feature-learning limit, Neural Tangent Kernel parametrization and the kernel-vs-feature-learning dichotomy, coordinate checking as a debugging procedure, per-layer learning rates, Adafactor and Frobenius-normalised optimizers, reverse-μTransfer for instability diagnosis
