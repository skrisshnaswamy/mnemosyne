---
title: "Representation Learning with Contrastive Predictive Coding (CPC / InfoNCE)"
authors: ["Aaron van den Oord", "Yazhe Li", "Oriol Vinyals"]
year: 2018
arxiv: "1807.03748"
url: https://arxiv.org/abs/1807.03748
priority: Must-Read
read_on: 2026-08-24
tags: [paper, transformers, rl, self-supervised, vision, theory]
---
## The Core Idea

To learn good features without labels, one old trick is: predict the future. If a model can guess what comes next, it must have understood something. The problem is that predicting the *actual raw future* — the next audio samples, the next image pixels — is a bad job to give a network. An image has thousands of bits of detail. The thing you care about (the object, the phoneme, the speaker) is maybe ten bits. A generative model that reconstructs every pixel burns nearly all its capacity on texture and noise, and can afford to ignore the context entirely.

Contrastive Predictive Coding (CPC) sidesteps this. **Do not predict the future. Predict a compressed code of the future, and only well enough to tell it apart from random other codes.**

Concretely: encode each chunk of the signal into a vector $z_t$. Summarise the past into a context vector $c_t$. Now, instead of asking "what is $x_{t+k}$?", ask a multiple-choice question: "here are $N$ candidate future codes, one is the real one and $N-1$ are random draws from elsewhere in the batch — which is real?" Train with plain [[Cross Entropy|cross-entropy]] over those $N$ choices.

Why this is not just a heuristic: the optimal scoring function for that classification problem is forced to be the **density ratio** $\frac{p(x_{t+k}\mid c_t)}{p(x_{t+k})}$ — up to a constant. That ratio is exactly the quantity inside mutual information. So the loss, which they name **InfoNCE**, is a lower bound:

$$I(x_{t+k}; c_t) \;\ge\; \log N - \mathcal{L}_N$$

Minimising the loss maximises a bound on how much the present tells you about the future. You never have to normalise anything, never evaluate $p(x)$, never build a decoder.

> [!NOTE] InfoNCE
> A loss that classifies one true sample against $N-1$ noise samples. Its optimum recovers the density ratio $p(x|c)/p(x)$, so it lower-bounds mutual information by $\log N - \mathcal{L}_N$. The bound is capped at $\log N$ — with a batch of 256 negatives you can never certify more than ~5.5 nats. ^infonce

> [!NOTE] Slow features
> Information that persists over many timesteps — phonemes, speaker identity, the object in a scene — as opposed to local detail that changes every millisecond. Predicting *far* ahead forces the model to keep only slow features, because local detail is unpredictable at that range. ^slow-features

What it unlocks: one recipe, four modalities. Speech, images, text, RL — same loss, same idea, only the encoder changes. This paper is the direct ancestor of SimCLR, MoCo, and the whole contrastive self-supervised wave.

## The Methodology

Three pieces.

**1. Encoder $g_{\text{enc}}$.** Maps raw observation $x_t \to z_t$. Usually downsamples in time.

**2. Autoregressive summariser $g_{\text{ar}}$.** Maps all past latents to a context: $c_t = g_{\text{ar}}(z_{\le t})$. They used a GRU; they note masked convolutions or [[Attention Is All You Need|self-attention]] would work too. Cf. [[Auto-regressive models]].

**3. Scoring function.** A log-bilinear form, one weight matrix per prediction distance $k$:

$$f_k(x_{t+k}, c_t) = \exp\!\big(z_{t+k}^\top W_k\, c_t\big)$$

So predicting $k$ steps ahead is just a [[Linear Projection|linear map]] $W_k c_t$ compared against the true future code by dot product. Cheap — this is the entire "predictor".

**The loss**, over a set $X$ containing one positive and $N-1$ negatives:

$$\mathcal{L}_N = -\mathbb{E}_X\left[\log \frac{f_k(x_{t+k}, c_t)}{\sum_{x_j \in X} f_k(x_j, c_t)}\right]$$

This is softmax cross-entropy where the "logits" are $z_j^\top W_k c_t$. Negatives come for free from the other examples in the minibatch — the same trick as [[Distributed Representations of Words and Phrases (negative sampling)|negative sampling]] in word2vec.

Either $z_t$ or $c_t$ can be the downstream representation. Use $c_t$ when history matters (speech), $z_t$ when it does not. For whole-sequence tasks, mean-pool.

**Audio setup (LibriSpeech, 100h, 251 speakers).** Encoder is 5 strided conv layers straight on 16kHz waveform: strides [5,4,2,2,2], filters [10,8,4,4,4], 512 units, ReLU. Total stride 160, so one $z$ per 10 ms — matching the phone label rate from Kaldi forced alignment. GRU with 256 hidden units. Predicts 12 steps (120 ms) ahead. Adam, lr 2e-4, 8 GPUs × batch 8, ~300k updates. Training windows of 20480 samples (~1.28 s).

**Vision setup (ImageNet).** Take a 256×256 image, cut it into a 7×7 grid of 64×64 crops with 32 px overlap. Encode each crop with a [[Deep Residual Learning for Image Recognition (ResNet)|ResNet-v2-101]] (no BatchNorm), take the third residual block output, spatially mean-pool → 1024-d per patch → a 7×7×1024 tensor. A PixelCNN-style autoregressive model scans **top to bottom** and predicts patch latents in the rows below, up to 5 rows down. Contrastive loss applied per patch. Augmentation: random crop from 300×300, 50% horizontal flip, convert to greyscale, plus a random 60×60 subcrop of each 64×64 patch padded back. Adam 2e-4, 32 GPUs × batch 16. Evaluation: freeze, mean-pool to 1024-d, train a linear classifier (SGD momentum 0.9, lr 0.1/0.01/0.001 for 50k/25k/10k steps, batch 2048).

**Text setup (BookCorpus).** Sentence encoder = 1D conv + ReLU + mean-pool → 2400-d. GRU with 2400 units predicts up to 3 future *sentence* embeddings.

**RL setup (DeepMind Lab).** Take a batched A2C agent (conv encoder + LSTM), add the CPC loss as an **auxiliary** term on top of the existing policy loss. Unroll 100 steps, predict up to 30 steps ahead. Only extra parameters are the linear $W_k$ maps. No replay buffer, so predictions must track a shifting policy.

## Ablation Studies and Experiments

**Speech, linear probe on frozen $c_t$ (Table 1):**

| | Phone (41 cls) | Speaker (251 cls) |
|---|---|---|
| Random init | 27.6 | 1.87 |
| MFCC | 39.7 | 17.6 |
| **CPC** | **64.6** | **97.4** |
| Fully supervised | 74.6 | 98.5 |

Speaker ID at 97.4% from a *linear* classifier is the headline: the representation keeps both what is being said and who is saying it. Swapping the linear probe for a single hidden layer lifts phone accuracy 64.6 → **72.5**, near the 74.6 supervised ceiling — so most of the information is there, just not linearly laid out. Worth remembering when you judge a method by linear-probe numbers.

**Ablation A — how far ahead to predict (phone accuracy):**

| steps | 2 | 4 | 8 | 12 | 16 |
|---|---|---|---|---|---|
| acc | 28.5 | 57.6 | 63.6 | **64.6** | 63.8 |

Predicting only 2 steps ahead is *catastrophic* — 28.5%, barely above random init. At 2 steps (20 ms) the answer is local smoothness of the waveform, which teaches nothing. This is the single most informative number in the paper: **the difficulty of the prediction task is the mechanism**. Past 12 steps it plateaus and slightly degrades.

**Ablation B — where negatives come from (all at 12 steps):**

| strategy | acc |
|---|---|
| Mixed speaker | 64.6 |
| Same speaker | 65.5 |
| Mixed speaker (excluding current sequence) | 57.3 |
| Same speaker (excluding current sequence) | 64.6 |
| Current sequence only | 65.2 |

Negatives drawn from *within the same utterance* are the useful ones (65.2 alone). If you exclude the current sequence and use cross-speaker negatives, you drop to 57.3 — because telling apart different speakers is easy, so the model can solve the task on voice timbre alone and never learn phonetic content. **Hard negatives, meaning ones sharing the nuisance factors, are what force the feature to be about content.**

**ImageNet linear probe.** Top-1: CPC **48.7** vs previous best 39.6 (Colorization, ResNet-v2) — +9 absolute. Top-5: CPC **73.6** vs 69.3 for a *combination* of Motion Segmentation + Exemplar + Relative Position + Colorization. One self-supervised task beating an ensemble of four.

**NLP (transfer from BookCorpus, logistic regression on frozen features):**

| | MR | CR | Subj | MPQA | TREC |
|---|---|---|---|---|---|
| Paragraph-vector | 74.8 | 78.1 | 90.5 | 74.2 | 91.8 |
| Skip-thought | 75.5 | 79.3 | 92.1 | 86.9 | 91.4 |
| Skip-thought + LayerNorm | 79.5 | 82.6 | 93.4 | 89.0 | – |
| CPC | 76.9 | 80.1 | 91.2 | 87.7 | **96.8** |

Roughly a tie with skip-thought, and a loss to skip-thought+LN on 4 of 5 — but CPC needs no LSTM word-level decoder, so it trains far faster. Honest caveat from the authors: **better BookCorpus modelling did not translate to better transfer scores**, so this benchmark is a weak signal. Also, more sophisticated sentence encoders "did not significantly improve the results" — bag-of-words is already strong on these tasks.

**RL.** On 5 DeepMind Lab tasks with 1B frames, the CPC auxiliary loss significantly beats the A2C baseline on 4. On `lasertag_three_opponents_small` it neither helps nor hurts — the authors' explanation is that the task needs no memory, so a purely reactive policy suffices and there is nothing for a future-prediction loss to add. See [[Markov Decision Process]], [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)|Dreamer]] for the world-model version of this idea.

**What did not work: MINE.** Appendix shows InfoNCE is a lower bound on the MINE estimator. Substituting MINE directly gave *identical* performance when the task was hard, but became **very unstable when the target was easy to predict** — e.g. predicting one step ahead where the target window overlaps the context. InfoNCE's stability comes from the positive term appearing in its own denominator (the $\log$-sum includes $e^{F(x,c)}$), which bounds the score from blowing up.

## Worth Remembering

- **The bound is capped at $\log N$.** With $N = 64$ negatives you cannot certify more than $\log 64 \approx 4.16$ nats of mutual information, no matter how good the model is. This is why later work (MoCo, SimCLR) fights so hard for large negative counts. It also means a low InfoNCE loss does *not* prove high MI — later papers (Poole et al., Tschannen et al.) showed the MI interpretation is loose, and CPC probably works because of the specific inductive biases of the encoder and the negatives, not because it maximises MI per se.
- **Context length mattered a lot.** The authors flag that the 20480-sample window (~1.28 s) capped GRU context and that "longer segments would give better results." A hyperparameter that was not fully explored.
- **No BatchNorm in the vision encoder.** Deliberate. BatchNorm leaks information across the batch, and your batch *is* your negative set — the model could cheat.
- The greyscale conversion in vision augmentation exists to stop the model matching patches by colour histogram. Every augmentation in that list is destroying a shortcut.
- $f_k$ is unnormalised. This is the whole reason there is no partition function to compute. Cf. how [[Efficient Estimation of Word Representations (word2vec)|word2vec]] escaped the softmax over the vocabulary.
- Ancestry: CPC → CPC v2 (Hénaff et al., 63% ImageNet top-1) → SimCLR/MoCo (drop the autoregressive part entirely, use augmentation pairs as the positive) → CLIP (positives are image–text pairs). The InfoNCE loss survives all of them essentially unchanged. Compare [[LeJEPA- Provable and Scalable Self-Supervised Learning|LeJEPA]] for the modern non-contrastive alternative that avoids needing negatives at all.
- Practical caveat: if your positive and your context overlap in time, the task collapses to trivial and the representation learns nothing (the 2-step result: 28.5%). Always check that your prediction gap is large enough to be hard.
- Open question the paper raises but does not answer: is the density-ratio form $z^\top W_k c$ doing real work, or would any bilinear/MLP scorer do? They say non-linear or recurrent predictors "could be used" but report no comparison.

## Links

Related: [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Efficient Estimation of Word Representations (word2vec)]] · [[Cross Entropy]] · [[KL Divergence]] · [[Auto-regressive models]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[LeJEPA- Provable and Scalable Self-Supervised Learning]] · [[Dream to Control- Learning Behaviors by Latent Imagination (Dreamer)]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Foundation Models]] · [[Attention Is All You Need]]

New topics worth writing: Mutual Information, Noise-Contrastive Estimation, SimCLR, MoCo, Linear Probing as evaluation, Slow Feature Analysis, MINE (Mutual Information Neural Estimation), PixelCNN, GRU, Hard Negative Mining, Auxiliary losses in RL
