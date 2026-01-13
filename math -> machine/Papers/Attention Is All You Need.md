---
title: "Attention Is All You Need"
authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Lukasz Kaiser", "Illia Polosukhin"]
year: 2017
arxiv: "1706.03762"
url: https://arxiv.org/abs/1706.03762
priority: Must-Read
read_on: 2026-08-21
tags: [paper, transformers]
---
## The Core Idea

Before this paper, the best translation systems read a sentence one word at a time. A recurrent network keeps a hidden state $h_t$ that depends on $h_{t-1}$, so word 50 cannot be processed until words 1–49 are done. That is a hard wall: you cannot parallelise inside a single sentence, and information from word 1 has to survive 49 hops of squeezing to reach word 50. Long-range links are learned badly because the gradient path is long.

The trick here: throw out recurrence entirely and let every word look at every other word directly, in one shot, using **attention**. A word at position 50 reads position 1 in a single operation. The path length between any two tokens drops from $O(n)$ to $O(1)$, and the whole layer is one big matrix multiply, so all positions compute at once on a GPU.

> [!NOTE] Self-attention
> Each token builds a query vector, and every token offers a key and a value vector. The token's new representation is a weighted average of all the values, where the weights come from how well its query matches each key. All queries, keys and values come from the *same* sequence — hence "self". ^self-attention

Why it did not exist before: attention was already invented (Bahdanau 2014) but always bolted on *top* of an RNN, used only to link decoder to encoder. Nobody had tried making it the entire model. The paper's title is the whole claim — you do not need the recurrence, the attention was doing the work.

What it unlocks: training that scales with hardware rather than sequence position. The big model hit **28.4 BLEU on WMT14 English→German** (previous best, including ensembles, was 26.36) and **41.8 BLEU on English→French**, in 3.5 days on 8 P100 GPUs — roughly $\tfrac{1}{4}$ the compute of the previous single-model record. The base model trained in **12 hours** and still beat everything published.

## The Methodology

An encoder–decoder [[Seq2Seq models|seq2seq]] shape, but every recurrent layer is replaced by attention plus a small MLP.

### Scaled dot-product attention

Pack queries, keys and values into matrices. Then

$$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$$

$QK^\top$ is a matrix of dot products — one similarity score for every (query, key) pair. Softmax turns each row into weights that sum to 1. Multiplying by $V$ takes the weighted average.

The $\sqrt{d_k}$ is not cosmetic. If query and key entries are roughly independent with unit variance, their dot product has variance $d_k$. With $d_k=64$ the scores spread out, softmax saturates, and the [[Derivative#Gradient|gradient]] through it becomes tiny. Dividing by $\sqrt{d_k}=8$ keeps the scores in a sane range. This is the difference between dot-product attention working and not working at scale. See [[Query, Key, and Value (QKV)]].

### Multi-head attention

Instead of one attention over 512 dimensions, run $h=8$ attentions over 64 dimensions each:

$$\mathrm{head}_i=\mathrm{Attention}(QW_i^Q, KW_i^K, VW_i^V),\qquad \mathrm{MultiHead}=\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h)W^O$$

Each $W_i$ is a learned [[Linear Projection]] down to $d_k=d_v=64$. Because $8\times 64 = 512$, the total cost matches single-head attention at full width — you get eight different "views" for free. The reason you want them: a single softmax-weighted average blurs everything together. Eight heads can each specialise; the appendix shows heads that track verb-object links and one that appears to resolve pronouns ("its" attends sharply to its antecedent).

### The three places attention is used

1. **Encoder self-attention** — every position sees every position.
2. **Decoder self-attention, masked** — position $i$ may only see positions $\le i$. Implemented by setting the illegal softmax inputs to $-\infty$ before the softmax, so their weights become exactly 0. This preserves the [[Auto-regressive models|autoregressive]] property during teacher-forced training. See [[Causal Attention]].
3. **Encoder–decoder attention** — queries from the decoder, keys and values from the encoder output. This is the classic Bahdanau alignment, now just one more instance of the same operation.

### The rest of the block

Each sub-layer is wrapped as $\mathrm{LayerNorm}(x + \mathrm{Sublayer}(x))$ — residual connection then layer norm. Every sub-layer and the embeddings output $d_{\text{model}}=512$ so the residual addition always type-checks.

After attention, a position-wise feed-forward net applied to each token independently:

$$\mathrm{FFN}(x)=\max(0, xW_1+b_1)W_2+b_2$$

with inner width $d_{ff}=2048$ (4× expansion). Same weights across positions within a layer, different across layers. Equivalently two convolutions of kernel size 1.

Stack: $N=6$ encoder layers, $N=6$ decoder layers. Decoder layers have three sub-layers (masked self-attention, encoder–decoder attention, FFN).

### Positional encoding

Attention is permutation-invariant — shuffle the input and the output shuffles the same way. It has no idea what "order" is. So position information is *added* to the embeddings:

$$PE_{(pos,2i)}=\sin\!\left(pos/10000^{2i/d_{\text{model}}}\right),\qquad PE_{(pos,2i+1)}=\cos\!\left(pos/10000^{2i/d_{\text{model}}}\right)$$

Wavelengths form a geometric progression from $2\pi$ to $10000\cdot 2\pi$ — a bank of sinusoids at many frequencies, much like a [[Fourier Series Decomposition]] of position. The reason for sinusoids specifically: for any fixed offset $k$, $PE_{pos+k}$ is a fixed linear function of $PE_{pos}$, so a head can learn "attend 3 to the left" as a linear operation.

### Training setup

- **Data:** WMT14 EN-DE, 4.5M sentence pairs, byte-pair encoding, ~37k shared source/target vocabulary. EN-FR, 36M pairs, 32k word-pieces.
- **Batching:** by approximate sequence length, ~25000 source + 25000 target tokens per batch.
- **Optimiser:** Adam, $\beta_1=0.9$, $\beta_2=0.98$, $\epsilon=10^{-9}$, with the now-famous warmup schedule
$$lrate = d_{\text{model}}^{-0.5}\cdot\min\left(step^{-0.5},\; step\cdot warmup^{-1.5}\right)$$
with $warmup=4000$. Linear ramp up, then $1/\sqrt{step}$ decay. Without warmup this model diverges — an early large step wrecks the attention softmax.
- **[[Regularization]]:** dropout $P_{drop}=0.1$ on every sub-layer output *before* the residual add, and on the embedding+positional sum. Label smoothing $\epsilon_{ls}=0.1$.
- **Weight tying:** input embedding, output embedding and pre-softmax projection share one matrix; embeddings scaled by $\sqrt{d_{\text{model}}}$.
- **Inference:** beam search, beam 4, length penalty $\alpha=0.6$. Checkpoint averaging — last 5 for base, last 20 for big.
- Base: 100K steps, 0.4 s/step, 12 h. Big ($d_{\text{model}}=1024$, $d_{ff}=4096$, $h=16$, $P_{drop}=0.3$): 300K steps, 1.0 s/step, 3.5 days.

## Ablation Studies and Experiments

Main results (newstest2014):

| Model | EN-DE BLEU | EN-FR BLEU | Train FLOPs |
|---|---|---|---|
| GNMT + RL | 24.6 | 39.92 | $2.3\cdot 10^{19}$ |
| ConvS2S | 25.16 | 40.46 | $9.6\cdot 10^{18}$ |
| ConvS2S ensemble | 26.36 | 41.29 | $7.7\cdot 10^{19}$ |
| **Transformer base** | **27.3** | 38.1 | $3.3\cdot 10^{18}$ |
| **Transformer big** | **28.4** | **41.8** | $2.3\cdot 10^{19}$ |

The base model beats every prior *ensemble* on EN-DE at ~1/23 the training compute of the ConvS2S ensemble.

### Model variations (Table 3, newstest2013 dev, base = 4.92 PPL / 25.8 BLEU)

**Number of heads, compute held constant.** $h=1$: 25.9 PPL... actually 5.29 PPL, **24.9 BLEU** — 0.9 BLEU worse. $h=4$: 25.5. $h=16$: 25.8. $h=32$: 25.4. So more heads help up to a point and then *hurt*. Too many heads means $d_k$ per head shrinks to 16, and the head loses the capacity to compute a meaningful match. Multi-head is real, but it is a Goldilocks knob, not a "more is better" knob.

**Shrinking $d_k$ alone** (rows B): $d_k=16$ gives 25.1 BLEU, $d_k=32$ gives 25.4 — both worse than 64. The authors' own reading: dot product may be too crude a compatibility function, and something more expressive could help. Nobody has convincingly beaten the dot product since, which is itself interesting.

**Size** (rows C): $N=2$ layers collapses to 23.7 BLEU / 6.11 PPL. $d_{\text{model}}=256$ drops to 24.5. $d_{\text{model}}=1024$ gives 26.0 with 168M params; $d_{ff}=4096$ gives 26.2. Bigger is straightforwardly better — an early signal of the scaling story.

**Dropout** (rows D): turning it off ($P_{drop}=0$) costs 1.2 BLEU (24.6). $P_{drop}=0.2$ gives 25.5, slightly below the 0.1 baseline. Overfitting is real at this data scale.

**Label smoothing** off: 25.3 BLEU with a *better* perplexity (4.67 vs 4.92). Smoothing makes the model less confident — worse [[Cross Entropy|cross-entropy]] — but better BLEU. A clean example of the loss you optimise not being the metric you want ([[Loss, Objectives, and Business Alignment]]).

**What did not matter:** learned positional embeddings instead of sinusoids gave 4.92 PPL / 25.7 BLEU — essentially identical. The sinusoids were kept only on the hope they extrapolate to longer sequences than seen in training. (Later work found this hope is largely unfounded, which is why RoPE and ALiBi exist.)

### Generalisation test: constituency parsing

A 4-layer Transformer, $d_{\text{model}}=1024$, on 40K WSJ sentences — a small-data regime where RNN seq2seq models historically failed. WSJ-only: **91.3 F1**, beating the BerkeleyParser (90.4) and everything except Recurrent Neural Network Grammars (91.7). Semi-supervised with 17M sentences: **92.7 F1**, beating all prior semi-supervised results. Almost no task-specific tuning — only dropout, learning rate and beam size were touched.

## Worth Remembering

**The cost that was traded away.** Table 1 is the honest accounting:

| Layer | Complexity | Sequential ops | Max path |
|---|---|---|---|
| Self-attention | $O(n^2 d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k n d^2)$ | $O(1)$ | $O(\log_k n)$ |

Self-attention is cheaper than recurrence only when $n < d$ — true for sentences (~40 tokens vs 512 dims), false for documents. The $n^2$ term is the entire reason [[Flash Attention]] and every long-context method since exists. The authors already flag restricted/local attention over a window $r$ (giving $O(rnd)$ cost and $O(n/r)$ path) as future work.

**Averaging blurs.** They admit the "reduced effective resolution due to averaging attention-weighted positions" — softmax averaging destroys information a convolution would keep. Multi-head attention is explicitly the patch for this, not an independent good idea.

**Inference is still sequential.** Training parallelises beautifully; generation still emits one token at a time because of the causal mask. The conclusion names "making generation less sequential" as an open goal. It still mostly is.

**Attention maps are suggestive, not proof.** The appendix figures show heads that look like they do anaphora resolution and syntactic attachment. Reading these as explanations is the same trap as reading [[Saliency]] maps as explanations — pretty pictures, weak evidence.

**Practical caveats if you reimplement it.** The warmup schedule matters more than almost anything; skipping it gives divergence, not slow learning. This paper uses *post-norm* (`LayerNorm(x + Sublayer(x))`), which is why warmup is needed — nearly every modern implementation moved to pre-norm, which is far more forgiving. Checkpoint averaging over the last 5–20 checkpoints is worth ~0.5 BLEU and costs nothing. And note the EN-FR big model used $P_{drop}=0.1$, not 0.3 — with 36M sentences you need less regularisation.

**The unreported implication.** Everything downstream — [[Foundation Models]], [[In Context Learning]], the whole LLM era — is this architecture with the encoder deleted and the parameter count multiplied by $10^4$. Nothing in the paper anticipates this; the largest model here is 213M parameters and the task is translation.

## Links

Related: [[Causal Attention]] · [[Query, Key, and Value (QKV)]] · [[Flash Attention]] · [[Seq2Seq models]] · [[Auto-regressive models]] · [[Linear Projection]] · [[Cross Entropy]] · [[Regularization]] · [[Backpropagation]] · [[Deep Learning]] · [[Fourier Series Decomposition]] · [[Foundation Models]] · [[In Context Learning]] · [[Loss, Objectives, and Business Alignment]] · [[Derivative]] · [[Saliency]]

New topics worth writing: Layer Normalization, Residual Connections, Adam and learning-rate warmup, Label Smoothing, Byte-Pair Encoding, BLEU score, Beam Search, Positional Encoding (RoPE / ALiBi), Pre-norm vs post-norm Transformers, Dropout
