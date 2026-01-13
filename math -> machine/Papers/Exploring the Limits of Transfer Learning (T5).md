---
title: "Exploring the Limits of Transfer Learning (T5)"
authors: ["Colin Raffel", "Noam Shazeer", "Adam Roberts", "Katherine Lee", "Sharan Narang", "Michael Matena", "Yanqi Zhou", "Wei Li", "Peter J. Liu"]
year: 2019
arxiv: "1910.10683"
url: https://arxiv.org/abs/1910.10683
priority: Good-To-Read
read_on: 2026-08-28
tags: [paper, transformers, optimization, vision]
---
## The Core Idea

Every text task is the same task: read a string, write a string.

That is the whole trick. Translation, sentiment classification, summarisation, question answering, even a *regression* task where you must output a number between 1 and 5 — all of them get turned into "here is some text, produce some text". You tell the model which job to do by sticking a prefix on the input: `translate English to German: That is good.` → `Das ist gut.` Or `mnli premise: ... hypothesis: ...` → `entailment`. The similarity score 2.57 becomes the literal string `"2.6"` (they round to the nearest 0.2, turning regression into 21-way classification).

Why this matters: before T5, every task needed its own head, its own loss, its own decoding rule. [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]] bolts a classifier onto a special `[CLS]` token; span prediction needs a start/end pointer; translation needs a decoder. That variety made it impossible to compare research choices fairly, because changing the objective also meant changing the architecture, which meant changing the evaluation. One unified format kills all those confounds. The same [[Cross Entropy|cross-entropy]] loss, same optimiser, same greedy decoding, everywhere.

With that testbed in hand, the actual contribution is a giant, careful, coordinate-by-coordinate ablation of the entire transfer-learning pipeline: architecture, objective, data, fine-tuning strategy, multi-task mixing, and scaling. Then they take every winning choice, scale it to 11 billion parameters and 1 trillion tokens, and get state of the art on 18 of 24 tasks.

The second contribution is **C4** — the Colossal Clean Crawled Corpus, 745 GB of filtered English web text, released publicly. Before this, everyone pre-trained on private or ad-hoc corpora, so "did the data or the method cause the gain?" was unanswerable.

> [!NOTE] Text-to-text framework
> Cast every NLP problem as string-in, string-out, with a natural-language task prefix telling the model what to do. One model, one loss, one decoder. ^text-to-text

## The Methodology

**Architecture.** A standard encoder–decoder Transformer, essentially unchanged from [[Attention Is All You Need]]. Baseline: 12 blocks in the encoder, 12 in the decoder, $d_\text{model}=768$, $d_\text{ff}=3072$, 12 heads, $d_{kv}=64$. About **220M parameters** — roughly twice BERT-Base because there are two stacks.

Three small deviations from the original Transformer:
- [[Layer Normalization|LayerNorm]] with **no additive bias** (rescale only), placed *outside* the residual path (pre-norm).
- [[Dropout- A Simple Way to Prevent Overfitting|Dropout]] 0.1 everywhere — in the FFN, on the [[Deep Residual Learning for Image Recognition (ResNet)|skip connection]], on attention weights, at stack input and output.
- **Simplified relative position embeddings.** Instead of a vector per position, each "embedding" is a single scalar added to the attention logit, chosen by the (key − query) offset. 32 buckets, log-spaced up to offset 128; everything beyond 128 shares one bucket. Shared across all layers, different per head. This is not [[RoFormer- Enhanced Transformer with Rotary Position Embedding|RoPE]] — it is the cheaper Shaw-style scalar bias.

**The pre-training objective (span corruption).** Take a sentence. Randomly pick 15% of tokens to corrupt. Replace each *consecutive run* of corrupted tokens with one unique sentinel token (`<X>`, `<Y>`, ...). The target is only the dropped spans, each prefixed by its sentinel, ending with a final sentinel.

```
input:  Thank you <X> me to your party <Y> week .
target: <X> for inviting <Y> last <Z>
```

The target is short, which is the whole point — the decoder does not have to reconstruct the full sentence, so training is cheaper.

**Training.** AdaFactor (not [[Adam- A Method for Stochastic Optimization|Adam]] — it saves optimiser memory by factorising the second-moment matrix). Inverse-square-root learning-rate schedule $1/\sqrt{\max(n, k)}$ with $k=10^4$ warm-up steps, so LR is constant at 0.01 for 10k steps then decays. Pre-train $2^{19} = 524{,}288$ steps, batch of 128 sequences of length 512 (packed), $\approx 2^{16}$ tokens per batch, so **$2^{35} \approx 34$B tokens total**. That is $\tfrac14$ of BERT's token budget and $\tfrac{1}{64}$ of [[RoBERTa- A Robustly Optimized BERT Pretraining Approach|RoBERTa]]'s — a deliberately modest budget so they could afford dozens of experiments.

Fine-tune $2^{18}$ steps, constant LR 0.001, checkpoint every 5000 steps, pick the best on validation. Teacher forcing throughout. Greedy decoding at test time (beam search with width 4, length penalty $\alpha=0.6$, only for the final translation and summarisation runs).

**Vocabulary.** SentencePiece, 32k wordpieces, trained on 10 parts English C4 to 1 part each German, French, Romanian. Shared between input and output. Consequence: the model can *only ever* handle those four languages.

**C4 construction.** From one month (April 2019) of Common Crawl, apply these heuristics:
- keep only lines ending in terminal punctuation
- drop pages with fewer than 3 sentences; drop lines under 5 words
- drop any page containing a word from the "bad words" list
- drop any line containing "Javascript"; any page containing "lorem ipsum"
- drop any page containing a curly brace `{` (a cheap code detector)
- strip Wikipedia citation markers; drop boilerplate cookie/ToS lines
- deduplicate: keep only one copy of any repeated three-sentence span
- `langdetect` English probability ≥ 0.99

**Reproducibility hygiene.** They trained the baseline 10 times from scratch to get a standard deviation, then treated that as the noise floor for every other experiment. Baseline GLUE 83.28 ± 0.235; SQuAD EM 80.88 ± 0.343. Anything within two standard deviations of the best is bolded, not declared a winner. This is the kind of discipline [[On the Difficulty of Evaluating Baselines]] and [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] argue for.

## Ablation Studies and Experiments

Reference point: baseline is GLUE 83.28 / CNNDM ROUGE-2 19.24 / SQuAD EM 80.88 / SuperGLUE 71.36 / EnDe 26.98 / EnFr 39.82 / EnRo 27.65. With **no pre-training at all**: 66.22 / 17.60 / 50.31 / 53.04 / 25.86 / 39.77 / 24.04. Note EnFr barely moves — that dataset is big enough that pre-training buys nothing.

**Architecture (Table 2).** They control for compute, not parameters. An $L{+}L$ encoder–decoder has $2P$ parameters but the *same* FLOPs as an $L$-layer decoder-only model, because the encoder only runs on the input and the decoder only on the output.

| Setup | Params | Cost | GLUE | SQuAD | SGLUE |
|---|---|---|---|---|---|
| Encoder–decoder, denoising | $2P$ | $M$ | **83.28** | **80.88** | **71.36** |
| Enc-dec, weights shared | $P$ | $M$ | 82.81 | 80.63 | 70.73 |
| Enc-dec, 6 layers each | $P$ | $M/2$ | 80.88 | 77.59 | 68.42 |
| Decoder-only LM | $P$ | $M$ | 74.70 | 61.14 | 55.02 |
| Prefix LM | $P$ | $M$ | 81.82 | 78.94 | 68.11 |

Three readings. (1) Encoder–decoder wins. (2) **Sharing encoder and decoder weights costs almost nothing** — halve the parameters, lose 0.5 GLUE. (3) A pure [[Improving Language Understanding by Generative Pre-Training (GPT-1)|decoder-only]] LM with [[Causal Attention|causal masking]] is badly hurt (SQuAD 61.14 vs 80.88), because the model's representation of the *input* can only look backwards. Fix the mask so the prefix is fully visible (prefix LM) and most of that gap closes. The remaining gap between prefix LM (81.82) and shared enc-dec (82.81) is the value of explicit encoder–decoder cross-attention.

Also confirmed: denoising beats plain LM pre-training in every architecture (enc-dec: 83.28 vs 79.56 GLUE).

**Objectives (Tables 4–7).** A four-step funnel:

1. *High-level approaches.* BERT-style masking 82.96 GLUE > prefix LM 80.69 > **deshuffling 73.17**. Shuffling the words and asking for the original order is a bad objective. Dead end.
2. *Simplifying BERT.* BERT-style (with 10% random-token swaps) 82.96, MASS-style (mask only, reconstruct whole sequence) 82.32, replace-spans-with-sentinels 83.28, drop-corrupted-tokens 84.44. All within noise except CoLA: dropping tokens entirely scores **60.04** on CoLA vs baseline 53.84, plausibly because detecting a missing word is close to judging grammatical acceptability. But dropping tokens hurt SuperGLUE (68.67 vs 71.36). Sentinels win on balance and give short targets.
3. *Corruption rate.* 10% / 15% / 25% / 50% → 82.82 / 83.28 / 83.00 / 81.27 GLUE. Only 50% clearly hurts. They keep 15% out of inertia.
4. *Span length.* Mean span 2 / 3 / 5 / 10 → SQuAD 82.09 / 81.84 / 82.05 / 81.84, SuperGLUE 72.20 / 72.53 / 72.23 / 70.44. Mean length 3 is the pick, and it's faster.

**The honest conclusion the authors draw: all denoising objectives are basically the same.** Pick whichever gives the shortest targets, because that is a speed decision, not a quality decision. They explicitly say further tinkering in this space is unlikely to pay, and point at [[ELECTRA- Pre-training Text Encoders as Discriminators|ELECTRA]] as the kind of *different* idea that might.

**Data (Table 8).** Same model, different pre-training corpus:

| Corpus | Size | GLUE | SQuAD | SGLUE |
|---|---|---|---|---|
| C4 | 745 GB | 83.28 | 80.88 | 71.36 |
| C4 unfiltered | 6.1 TB | 81.46 | 78.78 | 68.04 |
| RealNews-like | 35 GB | 83.83 | 80.39 | 72.38 |
| WebText-like | 17 GB | 84.03 | 81.42 | 71.40 |
| Wikipedia | 16 GB | 81.85 | 81.29 | 68.01 |
| Wikipedia + Toronto Books | 20 GB | 83.65 | 82.08 | **73.24** |

Removing the heuristic filters hurts *everything*, despite giving 8× more data. And in-domain data helps, sometimes dramatically: Wikipedia+TBC lifts SuperGLUE to 73.24 almost entirely from MultiRC Exact Match going **25.78 → 50.93**, because MultiRC's passages come from fiction books and TBC is fiction books. RealNews lifts ReCoRD EM 68.16 → 73.72; ReCoRD is news. Satisfying, but unhelpful if you want one general model.

**Dataset size and repetition (Table 9).** Truncate C4, but keep the token budget at $2^{35}$, so the data repeats:

| Tokens | Repeats | GLUE | SGLUE |
|---|---|---|---|
| full | 0 | 83.28 | 71.36 |
| $2^{29}$ | 64 | 82.87 | 72.03 |
| $2^{27}$ | 256 | 82.62 | 69.97 |
| $2^{25}$ | 1024 | 79.55 | 64.76 |
| $2^{23}$ | 4096 | 76.34 | 59.29 |

Training loss goes *down* as the dataset shrinks — the model memorises. 64 repeats is fine; 1000+ is poison. This is why they stuck with C4 for the final 1T-token run instead of the better-scoring but tiny WebText-like set.

**Fine-tuning methods (Table 10).** What did *not* work:
- **Adapter layers** (extra dense-ReLU-dense blocks after each FFN, only those + LayerNorm updated). At $d=32$: GLUE 80.52 but translation collapses — EnDe **13.84** vs 26.98, EnFr **17.88** vs 39.82. Even $d=2048$ only reaches EnDe 25.64. Adapters need capacity proportional to task size; on high-resource generation tasks they simply cannot fit. Compare with [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]], which came later and works far better at the same parameter budget.
- **Gradual unfreezing** (unfreeze one more layer every $2^{18}/12$ steps, top-down, in encoder and decoder in parallel). GLUE 82.50, SQuAD 79.17 — a small loss everywhere, in exchange for faster fine-tuning.

Full fine-tuning of all parameters wins. Expensive, but it wins.

**Multi-task learning (Table 11).** In this framework, multi-task just means mixing datasets in a batch. The whole problem is the mixing proportions.
- **Equal mixing: 76.13 GLUE.** Disaster — low-resource tasks overfit, high-resource tasks starve.
- **Examples-proportional** with a cap $K$ on effective dataset size: sample task $m$ with probability $r_m = \min(e_m, K)/\sum_n \min(e_n, K)$. Best around $K=2^{18}$–$2^{19}$ (GLUE 81.42–81.67). Too small and high-resource tasks starve; too big and unlabelled data swamps everything.
- **Temperature mixing**: raise $r_m$ to the power $1/T$ and renormalise, with $K=2^{21}$. $T=2$ gives GLUE 81.90, the best of the multi-task runs.

**Nothing beats pre-train-then-fine-tune (83.28).** But — and this is what they ship — **multi-task pre-training *followed by* per-task fine-tuning matches it**: GLUE 83.11, SGLUE 71.03, EnRo 28.07. So you get the practical benefit of watching downstream metrics during pre-training, at no cost.

Two more variants: **leave-one-out** (pre-train on all tasks except the target, then fine-tune on it) scores 81.98 GLUE / 71.68 SGLUE — barely worse, so task interference is mild and the model genuinely adapts to unseen tasks. **Supervised-only multi-task pre-training** (no denoising at all, the computer-vision recipe) drops to 79.93 GLUE / 65.36 SGLUE, but is fine on translation (EnFr 40.13). Unsupervised pre-training is doing real work.

**Scaling (Table 13).** The question: *you get 4× more compute, how do you spend it?*

| Strategy | GLUE | SQuAD | SGLUE | CNNDM |
|---|---|---|---|---|
| Baseline | 83.28 | 80.88 | 71.36 | 19.24 |
| 1× size, 4× steps | 85.33 | 82.45 | 74.72 | 19.33 |
| 1× size, 4× batch | 84.60 | 82.52 | 74.64 | 19.42 |
| 2× size, 2× steps | **86.18** | **84.18** | 77.18 | 19.66 |
| 4× size, 1× steps | 85.91 | 83.86 | **78.04** | 19.73 |
| 4× ensemble | 84.77 | 83.09 | 71.74 | **20.10** |
| 4× ensemble, fine-tune only | 84.05 | 82.36 | 71.55 | 19.57 |

Everything helps. Bigger beats longer, slightly. 2× bigger for 2× longer ≈ 4× bigger for 1× as long, so size and time are roughly interchangeable and complementary — a hand-rolled version of what [[Scaling Laws for Neural Language Models]] and [[Training Compute-Optimal Large Language Models (Chinchilla)]] formalise. Ensembling is *orthogonal* and wins outright on CNN/DM and translation, but does nothing for SuperGLUE. Ensembling four models fine-tuned from one shared pre-trained checkpoint is cheaper and still beats a single model, but is worse than four fully separate models.

**The final models (Table 14).** Span corruption (mean length 3, 15%), 1M steps at batch $2^{11}$ × length 512 = **~1 trillion tokens**, multi-task pre-training with a per-size artificial unlabelled dataset limit (710k examples for Small, up to 133M for 11B), then per-task fine-tuning with a *small* batch of 8 sequences and checkpoints every 1000 steps to avoid overfitting small GLUE tasks. Sizes: Small 60M, Base 220M, Large 770M, 3B, 11B (11B scales $d_\text{ff}$ to 65,536 with 128 heads, because TPUs love giant dense matmuls).

- **GLUE 90.3** (previous best 89.4). WNLI 94.5, RTE 92.8.
- **SuperGLUE 88.9**, from 84.6. Human performance is 89.8. They *exceed* humans on MultiRC and ReCoRD — which they read as a sign the metrics are biased toward machines — but humans get 100% on COPA and WSC where T5 gets 94.8 and 93.8.
- **SQuAD 91.26 EM / 96.22 F1** (previous 90.1 / 95.5).
- **CNN/DM ROUGE-2 21.55** (previous 20.30).
- **Translation: no SOTA anywhere.** EnDe 32.1 vs 33.8, EnFr 43.4 vs 43.8, EnRo 28.1 vs 38.5. The EnRo gap is enormous.

**The most important table is Table 15**, the one that separates scale from insight:

| | GLUE | SQuAD | SGLUE | EnDe |
|---|---|---|---|---|
| Baseline (34B tokens) | 83.28 | 80.88 | 71.36 | 26.98 |
| Baseline-1T (same model, 1T tokens) | 84.80 | 83.01 | 73.90 | 27.46 |
| T5-Base (1T tokens + all the study's choices) | **85.97** | **85.44** | **75.64** | **28.37** |

So scale alone gets you from 83.28 to 84.80; the non-scaling choices (span corruption, multi-task pre-training, per-task fine-tuning details) get you the further jump to 85.97 — a bigger step than scale gave. Scale is not the only thing, which is a mild pushback against the strong reading of [[The Bitter Lesson (essay)]].

## Worth Remembering

- **The methodology is the contribution.** One fixed baseline, change one coordinate at a time, report a 10-run standard deviation, bold anything within 2σ. Most papers cannot tell you whether their gain is real. This one can. Note the authors' own caveat: coordinate ascent misses second-order effects, e.g. an objective that only shines at larger scale.
- **The English-only vocabulary is a hard limit.** They trained SentencePiece on English + de/fr/ro, so the model physically cannot encode other languages. The translation failure is partly this, partly the lack of backtranslation (the SOTA methods use it), partly monolingual pre-training. mT5 fixed this later.
- **Adapters failing on translation is the underrated result.** Parameter-efficient fine-tuning is not free — it caps how much a task can change the model. If your downstream task is far from pre-training (English text → German text), a low-capacity adapter is not enough. [[LoRA- Low-Rank Adaptation of Large Language Models|LoRA]]'s later success suggests *where* you insert the extra capacity matters more than how much.
- **"Just filter your data" is a real gain.** Unfiltered C4 loses ~2 GLUE points despite being 8× bigger. Cheap, boring heuristics — terminal punctuation, curly braces, three-sentence dedup — are doing measurable work.
- **In-domain pre-training data helps a lot** (MultiRC EM nearly doubles with Toronto Books) but forces you into a small corpus, which then repeats, which then memorises. There is a real tension here and they do not resolve it.
- **Encoder–decoder was the right call in 2019 and the field went the other way anyway.** The T5 result is clear: at equal FLOPs, enc-dec > prefix LM > causal LM. [[Language Models are Few-Shot Learners (GPT-3)|GPT-3]] shipped a year later and decoder-only won on the strength of [[In Context Learning|in-context learning]] and simpler serving, not on this benchmark.
- **Practical caveat:** T5's task prefixes are just hyperparameters. The authors say wording changes had "limited impact" and did not investigate. Everything we now call prompt engineering lives in that one unexamined sentence.
- **Open questions they raise.** Denoising 1T tokens is probably an inefficient way to install knowledge — can we do better (they point at ELECTRA)? Can we formalise task similarity so we can *choose* pre-training data instead of guessing? Can we get strong performance from cheap models, via [[Distilling the Knowledge in a Neural Network|distillation]] or parameter sharing (recall: sharing encoder/decoder weights cost only 0.5 GLUE)?

## Links
Related: [[Attention Is All You Need]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[RoBERTa- A Robustly Optimized BERT Pretraining Approach]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[ELECTRA- Pre-training Text Encoders as Discriminators]] · [[Seq2Seq models]] · [[Causal Attention]] · [[Layer Normalization]] · [[Dropout- A Simple Way to Prevent Overfitting]] · [[Cross Entropy]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[The Bitter Lesson (essay)]] · [[LoRA- Low-Rank Adaptation of Large Language Models]] · [[Distilling the Knowledge in a Neural Network]] · [[On the Difficulty of Evaluating Baselines]] · [[Foundation Models]]

New topics worth writing: AdaFactor, SentencePiece and WordPiece tokenisation, relative position embeddings (Shaw et al.), C4 and web-corpus filtering, GLUE and SuperGLUE benchmark design, multi-task mixing rates (temperature-scaled sampling), adapter layers (Houlsby), mT5 and language-agnostic pre-training, backtranslation for NMT, ROUGE and BLEU as metrics, teacher forcing, beam search with length penalty
