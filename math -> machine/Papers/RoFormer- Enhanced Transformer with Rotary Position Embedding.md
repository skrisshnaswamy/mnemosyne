---
title: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
authors: ["Jianlin Su", "Yu Lu", "Shengfeng Pan", "Ahmed Murtadha", "Bo Wen", "Yunfeng Liu"]
year: 2021
arxiv: "2104.09864"
url: https://arxiv.org/abs/2104.09864
priority: Must-Read
read_on: 2026-08-24
tags: [paper, transformers, llm, vision, theory]
---
## The Core Idea

A transformer's [[Attention Is All You Need#^self-attention|self-attention]] is blind to word order. It sees a bag of vectors. So you have to inject position somehow. Every method before this one **added** something: the original transformer added a fixed sine/cosine vector $\boldsymbol{p}_i$ to the word embedding, BERT added a learned vector per slot, Shaw/T5/DeBERTa added learned bias terms into the attention score based on the gap $m-n$.

RoPE (Rotary Position Embedding) does something different: it **rotates** the query and key vectors instead of adding to them. Token at position $m$ gets its query vector spun by an angle proportional to $m$. Token at position $n$ gets its key spun by an angle proportional to $n$. Then when you take the dot product $\boldsymbol{q}_m^\intercal \boldsymbol{k}_n$, the two rotations partly cancel and what survives is a rotation by $m-n$.

That is the whole trick. **Absolute position goes in, relative position comes out, for free, inside the dot product.**

> [!NOTE] Rotary Position Embedding (RoPE)
> Multiply the query and key by a rotation matrix $\boldsymbol{R}^d_{\Theta,m}$ whose angle depends on the token's absolute index $m$. Because $(\boldsymbol{R}_m)^\intercal \boldsymbol{R}_n = \boldsymbol{R}_{n-m}$, the attention score only ever depends on the *difference* of positions. ^rotary-embedding

Why this did not exist before: people were stuck in the "decompose $(\boldsymbol{x}_m+\boldsymbol{p}_m)^\intercal W^\intercal W (\boldsymbol{x}_n+\boldsymbol{p}_n)$ into four terms and hack the terms" framing (Equation 6 in the paper). RoPE instead starts from the *requirement* — "I want $\langle f_q(\boldsymbol{x}_m,m), f_k(\boldsymbol{x}_n,n)\rangle = g(\boldsymbol{x}_m,\boldsymbol{x}_n,m-n)$" — and solves for $f$. In 2D, the solution is forced: it must be a rotation.

What it unlocks, and why every modern LLM (LLaMA, PaLM, GPT-NeoX, Mistral, Qwen) uses it:

1. **No position embedding table.** Nothing to look up, nothing bounded by a max length $L$. You can feed longer sequences than you trained on (badly, but you *can* — and later work like NTK-scaling and YaRN exploits exactly this).
2. **It composes with linear attention.** Additive relative-position schemes break linear attention, because linear attention never materialises the $N\times N$ score matrix where you would add the bias. Rotation does not need that matrix — you rotate $\phi(\boldsymbol{q})$ and $\varphi(\boldsymbol{k})$ before the associativity trick.
3. **Norms are preserved.** A rotation is orthogonal, so $\|\boldsymbol{R}\boldsymbol{x}\| = \|\boldsymbol{x}\|$. Nothing blows up or shrinks; position is pure phase, never magnitude.
4. **Built-in long-range decay.** With the frequencies set as in the sinusoidal scheme, attention scores naturally weaken as $|m-n|$ grows.

## The Methodology

**The 2D case.** Take $d=2$. Treat the vector as a complex number. The claim is

$$f_q(\boldsymbol{x}_m, m) = (\boldsymbol{W}_q\boldsymbol{x}_m)e^{im\theta}, \qquad f_k(\boldsymbol{x}_n, n) = (\boldsymbol{W}_k\boldsymbol{x}_n)e^{in\theta}$$

and then the inner product is $\operatorname{Re}[(\boldsymbol{W}_q\boldsymbol{x}_m)(\boldsymbol{W}_k\boldsymbol{x}_n)^{*}e^{i(m-n)\theta}]$ — depends on $m-n$ only. As a real matrix, this is a plain rotation applied *after* the [[Linear Projection|Q/K projection]]:

$$f_{\{q,k\}}(\boldsymbol{x}_m,m)=\begin{pmatrix}\cos m\theta & -\sin m\theta\\ \sin m\theta & \cos m\theta\end{pmatrix}\boldsymbol{W}_{\{q,k\}}\boldsymbol{x}_m$$

**The derivation** (§3.4.1, worth knowing because it shows rotation is not a design choice but the *only* answer). Write $f_q = R_q e^{i\Theta_q}$ in polar form. Two boundary conditions: the inner product must equal $g(\cdot,\cdot,n-m)$, and $f(\boldsymbol{x},0)$ must give back the plain unpositioned $\boldsymbol{q}$. Setting $m=n$ forces $R_q(\boldsymbol{x},m)=\|\boldsymbol{q}\|$ — the *magnitude cannot depend on position*. Setting $n=m+1$ forces $\phi(m+1)-\phi(m)$ to be a constant, so the angle $\phi(m)=m\theta+\gamma$ is an arithmetic progression. Set $\gamma=0$ and you are done.

**Scaling to $d$ dimensions.** Chop the $d$-dim vector into $d/2$ independent 2D pairs (dims 1&2, 3&4, …). Each pair gets its own rotation frequency. The full matrix $\boldsymbol{R}^d_{\Theta,m}$ is block-diagonal with $d/2$ 2×2 rotation blocks, and

$$\Theta = \{\theta_i = 10000^{-2(i-1)/d}\,,\ i \in [1,\dots,d/2]\}$$

Same frequency ladder as the original sinusoidal encoding: the first pair spins fast (once per token), the last pair spins glacially slowly (period ~60k tokens). Fast pairs encode fine local ordering, slow pairs encode coarse global position. That 10000 is the "base" that later long-context work turns into a tuning knob.

**How you actually implement it.** Never build the $d\times d$ matrix — it is 99% zeros. Use the elementwise form (Eq. 34):

$$\boldsymbol{R}^d_{\Theta,m}\boldsymbol{x} = \boldsymbol{x}\otimes\begin{pmatrix}\cos m\theta_1\\ \cos m\theta_1\\ \cos m\theta_2\\ \vdots\end{pmatrix} + \begin{pmatrix}-x_2\\ x_1\\ -x_4\\ x_3\\ \vdots\end{pmatrix}\otimes\begin{pmatrix}\sin m\theta_1\\ \sin m\theta_1\\ \sin m\theta_2\\ \vdots\end{pmatrix}$$

Two elementwise multiplies and an add, plus a cheap "swap-and-negate adjacent pairs" shuffle. Cost is $O(Nd)$, negligible.

**Where it goes in the architecture.** Only on [[Query, Key, and Value (QKV)|Q and K]], applied per head, applied *inside every layer* — not once at the embedding. Values $\boldsymbol{v}_n$ are untouched. There is no additive position embedding at the input at all.

**Long-term decay** (§3.4.3). Group the dot product into $d/2$ complex terms $h_i e^{i(m-n)\theta_i}$, apply Abel summation, and you get the bound

$$\Big|\sum_i h_i e^{i(m-n)\theta_i}\Big| \le \big(\max_i |h_{i+1}-h_i|\big)\sum_i |S_{i+1}|, \quad S_j = \sum_{i<j} e^{i(m-n)\theta_i}$$

The factor $\frac{1}{d/2}\sum|S_i|$ is a function of $m-n$ alone, and numerically it decays as $|m-n|$ grows. So the *upper bound* on attention shrinks with distance. Note this is a bound, not a guarantee — the model can still attend far if it wants.

**RoPE + linear attention.** Linear attention replaces $\exp(\boldsymbol{q}^\intercal\boldsymbol{k}/\sqrt d)$ with $\phi(\boldsymbol{q})^\intercal\varphi(\boldsymbol{k})$ for non-negative feature maps, letting you compute $K^\intercal V$ first and get $O(N)$ cost. RoPE drops in as:

$$\operatorname{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})_m = \frac{\sum_n (\boldsymbol{R}^d_{\Theta,m}\phi(\boldsymbol{q}_m))^\intercal(\boldsymbol{R}^d_{\Theta,n}\varphi(\boldsymbol{k}_n))\boldsymbol{v}_n}{\sum_n \phi(\boldsymbol{q}_m)^\intercal\varphi(\boldsymbol{k}_n)}$$

Note the honest hack: the **denominator is left unrotated**. Rotating it could make it zero or negative and you would divide by zero. So the weights are no longer a proper probability distribution — the numerator can contain negative terms. The authors just assert this is fine ("we kindly argue that the computation can still model the importance of values").

**Training setups used.**
- MT: fairseq, WMT14 EN-DE (4.5M pairs), 37k joint BPE vocab, Adam $\beta_1{=}0.9,\beta_2{=}0.98$, LR warmed $1e{-}7 \to 5e{-}4$ then inverse-sqrt decay, label smoothing 0.1, average last 5 checkpoints, beam 4, length penalty 0.6.
- Pre-training: bert-base-uncased architecture on BookCorpus + Wikipedia, batch 64, seq len 512, 100k steps, [[Decoupled Weight Decay Regularization (AdamW)|AdamW]] at LR 1e-5.
- Everything on **two servers with 4×V100 each**. This is a small-lab paper.

## Ablation Studies and Experiments

**Machine translation, WMT14 EN-DE (BLEU):**

| Model | BLEU |
|---|---|
| Transformer-base (Vaswani et al.) | 27.3 |
| RoFormer | **27.5** |

+0.2 BLEU. Barely anything, and no variance reported.

**MLM pre-training loss.** RoFormer's masked-LM loss falls faster than BERT's over 100k steps (Figure 3, left). This convergence-speed result is the paper's most-cited empirical claim.

**GLUE fine-tuning** (3 epochs, seq 512, batch 32, LR swept over 2/3/4/5e-5, best dev result reported):

| Model | MRPC | SST-2 | QNLI | STS-B | QQP | MNLI (m/mm) |
|---|---|---|---|---|---|---|
| BERT | 88.9 | **93.5** | **90.5** | 85.8 | 71.2 | **84.6/83.4** |
| RoFormer | **89.5** | 90.7 | 88.0 | **87.0** | **86.4** | 80.2/79.8 |

This table is **not a clean win** and the paper's phrasing ("significantly outperform BERT in three out of six") is generous. RoFormer *loses* on SST-2 (−2.8), QNLI (−2.5), and MNLI (−4.4 / −3.6). It wins hugely on QQP (+15.2), which is such a strange jump that it smells like a baseline problem rather than a real effect. Treat this table with suspicion.

**Performer + RoPE, Enwik8 char-level LM.** 12 layers, 768 dim, 12 heads, LR 1e-4, batch 128, seq 1024. Adding RoPE gives faster convergence and lower loss (Figure 3, right). This is the ablation that actually matters conceptually — it is the one thing competing relative-position schemes structurally *cannot* do.

**Chinese long-text (the strongest result).** RoFormer built by swapping WoBERT's absolute embeddings for RoPE, pre-trained on ~34GB of Chinese wiki/news/forum data in six stages with varying max length. Table 4 shows accuracy tracking sequence length: stage 1 (len 512) → 65.0%, stage 2 (len 1536) → 66.8%, stage 4 (len 128) → 63.4%, stage 5 (len 1536) → 67.4%. Longer context, better accuracy.

CAIL2019-SCM legal case matching (documents mostly >512 chars), accuracy:

| Model | Validation | Test |
|---|---|---|
| BERT-512 | 64.13 | 67.77 |
| WoBERT-512 | 64.07 | 68.10 |
| RoFormer-512 | 64.13 | 68.29 |
| RoFormer-1024 | **66.07** | **69.79** |

At 512 tokens RoPE is a wash — 68.29 vs 68.10. The whole gain comes from being *able to run at 1024*, which lifts test accuracy +1.5 over WoBERT. That is the honest headline: RoPE's value is length flexibility, not per-token quality.

**What did not work / what is absent:**
- **No frequency ablation.** They inherit $\theta_i = 10000^{-2i/d}$ from the sinusoidal paper and never test a different base or a learned $\Theta$. Given how much later work (position interpolation, NTK-aware scaling, base 500k in LLaMA-3) hinges on that constant, this is a big hole.
- **No head-to-head against Shaw / T5 bias / DeBERTa.** The only baselines are absolute-position BERT and vanilla Transformer. So the paper never shows RoPE beats other *relative* schemes on quality — only that it is cheaper and composable.
- **No GPT-style autoregressive LM experiment** on English. Ironic, since that is where RoPE ended up living.
- The unrotated denominator in linear attention is a known-ugly compromise that is never measured.

## Worth Remembering

**The authors admit they cannot explain their own results.** Two limitations stated verbatim: (1) no explanation for *why* it converges faster than other positional schemes; (2) no faithful explanation for *why* it does better on long text despite having the same long-term-decay property as sinusoidal encoding. So the mechanism is theoretically clean but the empirical benefit is unexplained.

**The paper's own experiments are the weakest part of its legacy.** +0.2 BLEU, a mixed GLUE table, one legal-matching dataset. RoPE won because thousands of people downstream found it just worked — LLaMA, PaLM, GPT-NeoX, Mistral, Qwen, DeepSeek all use it — not because this paper proved it. A useful lesson about how ideas actually propagate.

**Practical caveats if you implement it:**
- Apply per-head, on Q and K only, at *every* layer. A common bug is applying it once at the embedding.
- Two conventions exist for pairing dimensions: adjacent pairs $(x_1,x_2),(x_3,x_4)\dots$ as in the paper, versus "half-and-half" $(x_1,x_{d/2+1})$ as in most HF/GPT-NeoX code. They are equivalent up to a permutation of the head dims, but weights are **not** interchangeable across conventions.
- Precompute the $\cos$ and $\sin$ tables once and cache them. Under [[Mixed Precision training|mixed precision]], compute them in fp32 — bf16 phase angles at position 100k are visibly wrong.
- It interacts fine with [[Flash Attention]] because rotation happens before the kernel sees Q and K.
- KV-caching in [[Causal Attention|causal decoding]]: rotate the key *before* you store it in the cache, so the stored key already carries its absolute position.

**Extrapolation is oversold.** Because RoPE has no lookup table, people assumed it extrapolates to arbitrary lengths. It does not — attention degrades sharply past the training length, because the model has never seen those phase angles. The whole cottage industry of Position Interpolation, NTK-aware scaling, YaRN, and LongRoPE exists to patch this, all by rescaling $\theta_i$ or the position index. If you take one follow-up question away: *how do you change the frequency base at fine-tune time without destroying what the model learned?*

**Connection to Fourier thinking.** The frequency ladder is literally a [[Fourier Series Decomposition|Fourier basis]] over position — the model gets position written as a set of clock hands ticking at geometrically spaced rates, and it can read off relative offsets by comparing phases. Same intuition as the original sinusoidal encoding, but moved from *addition to the embedding* into *phase of the dot product*, which is where it belongs.

**The one-line summary to keep:** additive position encoding puts position into the *content* of a vector; RoPE puts position into the *angle*, leaving magnitude untouched — and angles subtract when you take dot products.

## Links

Related: [[Attention Is All You Need]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Query, Key, and Value (QKV)]] · [[Linear Projection]] · [[Causal Attention]] · [[Flash Attention]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Fourier Series Decomposition]] · [[Seq2Seq models]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[Foundation Models]] · [[Mixed Precision training]]

New topics worth writing: Positional Encoding (survey of absolute vs relative vs rotary), Linear Attention and the Performer / kernel-feature trick, ALiBi attention bias, Position Interpolation and YaRN for context extension, T5 relative position bias, DeBERTa disentangled attention, Orthogonal matrices and rotations in linear algebra, KV cache mechanics, Abel summation.
