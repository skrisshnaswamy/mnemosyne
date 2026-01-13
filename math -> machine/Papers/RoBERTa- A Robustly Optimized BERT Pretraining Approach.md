---
title: "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
authors: ["Yinhan Liu", "Myle Ott", "Naman Goyal", "Jingfei Du", "Mandar Joshi", "Danqi Chen", "Omer Levy", "Mike Lewis", "Luke Zettlemoyer", "Veselin Stoyanov"]
year: 2019
arxiv: "1907.11692"
url: https://arxiv.org/abs/1907.11692
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, transformers, llm, theory]
---
## The Core Idea

BERT was not broken. It was undertrained.

In 2018–2019 a stream of papers claimed to beat [[BERT- Pre-training of Deep Bidirectional Transformers|BERT]] with new pretraining objectives — XLNet's permuted autoregressive modelling, span prediction, multi-task tricks. Each one also happened to use more data, bigger batches, and longer training. Nobody had separated the two. Was the new objective doing the work, or was it just more compute?

This paper holds the architecture and the objective fixed — same transformer, same masked language model loss — and turns only the boring knobs: batch size, number of steps, amount of data, how the mask is generated, whether the next-sentence loss is there at all. The result is that plain BERT, retuned, beats every model published after it.

Concretely, the same $L=24$, $H=1024$, $A=16$, 355M-parameter architecture as BERT-large goes from 86.6 to **90.2** on MNLI-m dev, from 81.8 to **89.4** F1 on SQuAD 2.0 dev, and from 93.7 to **96.4** on SST-2. Nothing about the model changed. Only the recipe.

The unlock is partly a warning: **when training is expensive, hyperparameter sweeps are small, and unswept baselines get published as if they were ceilings.** Every later paper compared against a BERT that had been left on the table. This is the same disease diagnosed in [[On the Difficulty of Evaluating Baselines]] and [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]], now applied to language model pretraining.

> [!NOTE] Undertrained baseline ^undertrained-baseline
> A published result that is not the best the method can do, because the authors could not afford to tune it. It becomes the reference number everyone else beats, so the field credits new architectures for gains that were really just compute and tuning.

## The Methodology

Five changes to BERT's recipe. Each was tested separately at BERT-base scale first ($L=12$, $H=768$, $A=12$, 110M params), then stacked.

**1. Dynamic masking.** Original BERT picked which 15% of tokens to mask *once*, during data preprocessing. They duplicated the corpus 10 times so each sentence got 10 different masks, and trained 40 epochs — so the model saw each exact mask 4 times. RoBERTa generates a fresh mask every time a sequence is fed to the model. The 80/10/10 split (mask / leave / random token) is unchanged, and the loss is still [[Cross Entropy|cross-entropy]] on the masked positions only.

**2. Drop Next Sentence Prediction.** BERT had a second head predicting "did segment B follow segment A in the original document?", a binary loss. RoBERTa deletes it and changes how inputs are packed. Four formats were compared:

- `segment-pair+nsp` — original BERT: two multi-sentence segments, total < 512 tokens, NSP loss on.
- `sentence-pair+nsp` — two single natural sentences, NSP on. Much shorter inputs, so batch size raised to keep tokens/batch constant.
- `full-sentences` — pack contiguous sentences up to 512 tokens, crossing document boundaries with a separator. No NSP.
- `doc-sentences` — same but never crossing a document boundary; batch size dynamically raised when a sequence ends short.

**3. Big batches.** BERT used $B=256$ sequences for $S=1{,}000{,}000$ steps. RoBERTa uses $B=8192$. Holding total epochs constant, this is 31K steps at batch 8K. Peak learning rate is scaled up with the batch — 1e-4 at 256, 7e-4 at 2K, 1e-3 at 8K.

**4. Byte-level BPE.** BERT used a 30K character-level BPE vocabulary built after heuristic tokenisation. RoBERTa uses the GPT-2 style 50K byte-level BPE, which needs no pre-tokenisation and never emits an unknown token, since every byte sequence is representable. Costs ~15M extra params at base, ~20M at large.

**5. More data.** BERT trained on BookCorpus + Wikipedia, 16GB. RoBERTa uses 160GB:

| Corpus | Size | Source |
|---|---|---|
| BookCorpus + Wikipedia | 16GB | original BERT data |
| CC-News (new, collected here) | 76GB | 63M English news articles, Sept 2016–Feb 2019 |
| OpenWebText | 38GB | Reddit-linked pages with ≥3 upvotes |
| Stories | 31GB | CommonCrawl filtered to Winograd-like narrative style |

**Optimisation details that mattered.** [[Adam- A Method for Stochastic Optimization|Adam]] with $\beta_1 = 0.9$ but $\beta_2 = 0.98$ (not 0.999) — the lower $\beta_2$ shortens the memory of the second-moment estimate and improved *stability* at batch 8K. $\epsilon = 1\text{e-}6$, and they report training being "very sensitive" to it. Weight decay 0.01, dropout 0.1 everywhere, GELU activations, linear warmup then linear decay. Peak LR 4e-4 for large, 6e-4 for base. Warmup 30K steps for large. Gradient clipping off. [[Mixed Precision training|Mixed precision]] on DGX-1 nodes; RoBERTa-large's first 100K steps took about one day on **1024 V100 GPUs**.

Also: BERT randomly injected short sequences and trained on reduced-length sequences for the first 90% of updates. RoBERTa does neither — only full 512-token sequences, the whole way through.

Fine-tuning is deliberately plain. GLUE: batch ∈ {16, 32}, LR ∈ {1e-5, 2e-5, 3e-5}, 10 epochs, 6% warmup, early stopping on dev, median of 5 seeds. SQuAD: only the provided SQuAD data, no augmentation, same LR for all layers (XLNet used a layer-wise schedule). For SQuAD 2.0, an answerability classifier is trained jointly by summing its loss with the span loss. RACE: concatenate each of the four candidate answers with question and passage, encode all four, push each `[CLS]` through one linear layer, softmax over the four scores.

## Ablation Studies and Experiments

**Static vs dynamic masking** (BERT-base, medians over 5 seeds):

| Masking | SQuAD 2.0 F1 | MNLI-m | SST-2 |
|---|---|---|---|
| published BERT reference | 76.3 | 84.3 | 92.8 |
| their reimpl., static | 78.3 | 84.3 | 92.5 |
| their reimpl., dynamic | 78.7 | 84.0 | 92.9 |

The honest read: **dynamic masking barely matters.** It is "comparable or slightly better". Their static reimplementation already beat the published BERT by 2 F1 on SQuAD — so a chunk of the headline gain is just careful reimplementation, before any recipe change. They keep dynamic masking mostly because it removes the need to duplicate the corpus 10×.

**Input format and NSP** (base, 1M steps, batch 256, BookCorpus+Wiki):

| Format | SQuAD 1.1/2.0 F1 | MNLI-m | SST-2 | RACE |
|---|---|---|---|---|
| segment-pair + NSP | 90.4 / 78.7 | 84.0 | 92.9 | 64.2 |
| sentence-pair + NSP | 88.7 / 76.2 | 82.9 | 92.1 | 63.0 |
| full-sentences, no NSP | 90.4 / 79.1 | 84.7 | 92.5 | 64.8 |
| doc-sentences, no NSP | 90.6 / 79.7 | 84.7 | 92.7 | 65.6 |
| published BERT-base | 88.5 / 76.3 | 84.3 | 92.8 | 64.3 |

Two findings. First, **sentence-pair is clearly worse** — single sentences are short, so the model never sees long-range dependencies. This is the ablation that explains the original BERT paper's conclusion: Devlin et al. found removing NSP hurt, but they likely removed only the *loss term* while keeping the short segment-pair input format. The damage came from the input format, not from losing NSP.

Second, **removing NSP entirely matches or slightly beats keeping it**, once you also feed the model long contiguous text. `doc-sentences` (never crossing document boundaries) is the best of the four, but it produces variable batch sizes, so RoBERTa ships `full-sentences` for comparability — a small deliberate sacrifice.

**Batch size** (base, equal epochs, equal compute, LR tuned per row):

| Batch | Steps | LR | held-out ppl | MNLI-m | SST-2 |
|---|---|---|---|---|---|
| 256 | 1M | 1e-4 | 3.99 | 84.7 | 92.7 |
| 2K | 125K | 7e-4 | 3.68 | 85.2 | 92.9 |
| 8K | 31K | 1e-3 | 3.77 | 84.6 | 92.8 |

Note what this actually shows: 2K is the sweet spot on both perplexity and MNLI, and 8K is *worse* than 2K on every column here. They ship 8K anyway, because it parallelises better across 1024 GPUs. The end-task deltas are within noise. Large batches are a systems decision here, not an accuracy one.

**Byte-level BPE** — this one **did not work**, and they say so. "Early experiments revealed only slight differences between these encodings, with the [GPT-2 byte-level] BPE achieving slightly worse end-task performance on some tasks." They adopt it anyway for the engineering property of never producing an unknown token. No table is given. This is the clearest example in the paper of a component being kept for reasons other than measured accuracy.

**The stacked recipe** (RoBERTa-large, each row cumulative):

| Config | Data | Steps | SQuAD 1.1/2.0 | MNLI-m | SST-2 |
|---|---|---|---|---|---|
| Books+Wiki | 16GB | 100K | 93.6 / 87.3 | 89.0 | 95.3 |
| + all data | 160GB | 100K | 94.0 / 87.7 | 89.3 | 95.6 |
| + longer | 160GB | 300K | 94.4 / 88.7 | 90.0 | 96.1 |
| + even longer | 160GB | 500K | 94.6 / 89.4 | 90.2 | 96.4 |
| BERT-large | 13GB | 1M | 90.9 / 81.8 | 86.6 | 93.7 |
| XLNet-large + data | 126GB | 500K | 94.5 / 88.8 | 89.8 | 95.6 |

The first row is the punchline. **Same data as BERT, same architecture, same objective, only 100K steps — and SQuAD 2.0 F1 jumps from 81.8 to 87.3.** That 5.5-point gap is pure recipe. Everything after is diminishing returns from data (+0.4 F1 for 10× the data) and from time (+1.7 F1 for 5× the steps). Time buys more than data here.

**Headline results.** GLUE single-task dev: state of the art on all 9 tasks. GLUE test leaderboard (ensembles of 5–7 single-task models, no multi-task fine-tuning): average 88.5, edging XLNet's 88.4, new SOTA on MNLI, QNLI, RTE, STS-B. SQuAD 2.0 dev: 86.5 EM / 89.4 F1, beating XLNet by 0.4 EM / 0.6 F1 — and RoBERTa uses no external QA data while XLNet does. RACE test: 83.2 overall vs XLNet's 81.7.

**Two task-specific hacks** they are upfront about. QNLI test submissions use a pairwise ranking formulation (mine candidate answers, rank them) which "significantly simplifies the task" — they report dev numbers with plain classification for fair comparison to BERT. WNLI uses the SuperGLUE reformatted data with a margin ranking loss over spaCy-extracted noun phrases, which throws away over half the training examples because only positives are usable.

**RTE, STS-B, MRPC** are fine-tuned starting from the MNLI-fine-tuned model, not the raw pretrained one. These are the small GLUE datasets; this is transfer stacking, and it is doing real work on RTE (86.6 dev).

## Worth Remembering

- **"Even our longest-trained model does not appear to overfit our data and would likely benefit from additional training."** 500K steps × 8K sequences × 512 tokens and still no saturation. This sentence is the seed of [[Scaling Laws for Neural Language Models]] and [[Training Compute-Optimal Large Language Models (Chinchilla)]] — and it also means the reported numbers are a floor, not a limit.

- The masked language model objective is **not** inferior to XLNet's permuted autoregressive objective. Once compute is matched, the objectives are competitive. Architecture and objective mattered far less than "mundane details like dataset size and training time." This is a strongly [[The Bitter Lesson (essay)|Bitter Lesson]]-shaped result.

- Ablations were run at **base** scale and the recipe applied at **large** scale. Batch 8K was chosen at base scale where it lost to 2K on MNLI. There is no evidence in the paper that the ablation conclusions transfer up. Standard practice, still a caveat.

- $\beta_2 = 0.98$ and a tuned Adam $\epsilon$ are the unglamorous stability fixes. If you are training at batch ≥ 2K and seeing loss spikes, this is the first place to look. Related: the second-moment estimate in [[Adam- A Method for Stochastic Optimization]], and the weight-decay coupling issue in [[Decoupled Weight Decay Regularization (AdamW)]] — RoBERTa uses BERT's L2-style decay of 0.01, not decoupled decay.

- CC-News (76GB, 63M articles) was released alongside, which is much of why this paper is reproducible where XLNet's data was not.

- **Practical caveat:** RoBERTa has no NSP head and no `token_type_ids` trained for sentence-pair discrimination in the BERT sense. If you were relying on BERT's `[CLS]` NSP-trained representation for anything, it is not there. For sentence embeddings you still want something like [[Sentence-BERT]] or [[SimCSE- Simple Contrastive Learning of Sentence Embeddings]] on top — a raw RoBERTa `[CLS]` is a poor sentence vector, for reasons explored in [[How Contextual are Contextualized Word Representations]] and [[Representation Degeneration Problem in Training NLMs]].

- Open question the authors leave: LAMB-style training at 32K batch (You et al.) was untested here. And the byte-level BPE regression was never properly measured — "a more detailed comparison is left to future work."

## Links

Related: [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Attention Is All You Need]] · [[Adam- A Method for Stochastic Optimization]] · [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[On the Difficulty of Evaluating Baselines]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[The Bitter Lesson (essay)]] · [[Mixed Precision training]] · [[Cross Entropy]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Sentence-BERT]] · [[Decoupled Weight Decay Regularization (AdamW)]] · [[Foundation Models]]

New topics worth writing: Byte-level BPE and subword tokenisation, GLUE benchmark, SQuAD, XLNet and permutation language modelling, large-batch training and linear LR scaling, LAMB optimizer, GELU activation, learning-rate warmup schedules
