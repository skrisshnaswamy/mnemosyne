---
title: "The Embedder's Dilemma: LLMs Are Better, but at What Cost?"
authors: ["Adnan El Assadi", "Niklas Muennighoff", "Jinhyuk Lee"]
year: 2026
arxiv: "2608.12875"
url: https://arxiv.org/abs/2608.12875
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, llm, self-supervised, vision]
---
## The Core Idea

The question is blunt and practical: **should you throw away your embedding model and just ask an LLM?**

An embedding model turns text into a fixed vector. You compare vectors with cosine similarity. That is the whole pipeline for search, clustering, similarity scoring, and nearest-neighbour classification. These models are built with expensive machinery — contrastive training, hard-negative mining, multi-stage [[Distilling the Knowledge in a Neural Network|distillation]].

An LLM does none of that. You just prompt it: "is this sentence similar to that one, 0 to 5?" or "here are 300 documents, which answers this query?"

The finding: **on aggregate quality, the two are tied.** Best LLM (Gemini 3.1 Pro) scores 77.6 on a 37-task benchmark. Best embedding model (Octen-8B, 7.6B params) scores 77.2. A paired bootstrap test puts that 0.4-point gap squarely inside noise ($p = 0.85$, 95% CI $[-2.4, +3.1]$).

The catch, and the title of the paper: **one benchmark pass costs $154.14 with the LLM and $0.11 with the embedding model. That is 1,431×.** On the same H100 GPU, embedding models push tokens 2.5× to 736× faster.

So the interesting result is not "who wins" but *where* each one wins:

- **Retrieval: LLMs win by +8.5 points** (64.5 vs 56.0). This is the only real gap.
- **Classification: embedding models win by −5.6 points** (90.8 vs 85.2).
- **Clustering, STS, pair classification: statistical ties.** No paradigm advantage at all.

And there is a clean structural reason, which is the part worth remembering in a year. The whole thing reduces to one question: **how many documents does the model get to read jointly with the query?**

- Bi-encoder ([[Sentence-BERT|embedding model]]): zero. Documents are encoded offline, alone. Query is encoded alone. Only vectors meet.
- Cross-encoder reranker: one document at a time, with the query.
- LLM listwise reranker: the top-$k$ shortlist at once.
- LLM corpus-in-context: the *entire corpus*, in one forward pass.

Quality rises along that list. Cost rises along that list too, because everything after the bi-encoder is *query-specific compute you cannot precompute*. That is the trade, and it has been true since [[Sentence-BERT]] — this paper just shows it survives at frontier-LLM scale.

> [!NOTE] Corpus-in-context retrieval
> Instead of embedding documents into an index, you paste the whole corpus into the prompt with numbered IDs and ask the model which IDs answer the query. Works only for tiny corpora (here: 82–415 documents). Cost scales with corpus size $N$ *per query*, not once. ^corpus-in-context

> [!NOTE] The thinking-token tax
> Reasoning ("thinking") tokens are billed at the output rate. They account for **28–81% of total LLM inference cost** on these tasks. Cutting them by 54–96% preserves or *improves* retrieval quality for most models. You are paying most of the bill for compute that buys nothing here. ^thinking-token-tax

## The Methodology

**MTEB(LLM)** — a 37-task benchmark carved out of [MTEB](https://arxiv.org/abs/2210.07316). Each task is a fixed random subset (seed 42) of the original MTEB task, released on Hugging Face under `mteb/llm-eval-*`. Both paradigms are scored on exactly the same subset. Subsets are used because generative evaluation is orders of magnitude more expensive — a full MTEB pass would be hundreds of dollars *per LLM*.

**Models.** 10 LLMs across 6 families: Gemini 3.1 Pro, Gemini 3 Flash, Gemini 3.1 Flash-Lite (non-reasoning baseline), DeepSeek-R1, DeepSeek-V4-Flash, Qwen3.6-27B (dense), Qwen3.6-35B-A3B (MoE), GLM-4.7, Kimi-K2.6, MiniMax-M2.7. And 26 embedding models from 118M (mE5-small) to 14B (F2LLM-14B) parameters.

**The five task categories, and how each paradigm is asked to do them:**

| Category | Embedding pipeline | LLM pipeline | Metric |
|---|---|---|---|
| Classification (8) | kNN over embeddings of the **full labelled train split** | zero-shot prompt with label names | Accuracy |
| STS (10) | cosine similarity | "rate similarity on this scale, return a float" | Spearman $\rho$ |
| Clustering (9) | $k$-means on embeddings | all docs in one prompt, told the true $k$, return JSON assignments | V-measure |
| Pair classification (4) | cosine + threshold swept post-hoc on test set | binary yes/no per pair | Avg. precision |
| Retrieval (6) | cosine ranking | corpus-in-context, prompt caching amortises the corpus prefix | Recall@1 |

Note the two asymmetries the authors flag themselves: the kNN classifier sees the **whole labelled training set**, the LLM sees none. And embeddings get a threshold tuned on the test set in pair classification, with no LLM equivalent.

**Cost accounting.** This is the part the paper is really about, and it is done carefully.

Embedding cost = tokens processed divided by measured throughput, times GPU rental:
$$\text{cost} = \frac{\text{tokens}}{10^6} \times \frac{r_{\text{GPU}}}{T}$$
with $r_{\text{GPU}} = \$2.49$/hr (H100 80GB spot, Lambda Labs, March 2026) and $T$ = sustained tokens/hour at sequence length 512, largest batch that fits.

LLM cost from the provider's own `usage_stats`:
$$\text{Cost} = (\text{input} - \text{cached}) \cdot r_{\text{in}} + \text{cached}\cdot r_{\text{cache}} + (\text{total} - \text{input})\cdot r_{\text{out}}$$
with $r_{\text{cache}} = r_{\text{in}}/10$. The key move: **every non-input token is billed at the output rate**, so thinking tokens get charged like output tokens — which is what providers actually do.

A detail that matters: they tokenise with each model's *own* tokeniser. Token counts vary 4.4M–5.5M across vocabularies. Using one proxy tokeniser (GPT-2, 6.75M) would overstate embedding cost by 22–54%.

**Throughput** is measured separately and fairly: both paradigms on the *same* H100. The two open LLMs that fit on one card (Qwen3.6-27B, Qwen3.6-35B-A3B) run under vLLM in BF16, 256 concurrent requests, 200 in / 100 out tokens. Embedding models run batched on the same card.

## Ablation Studies and Experiments

**Headline table (category means, 0–100):**

| Model | Cls | Clust | STS | PairCls | Retr | Overall | Cost |
|---|---|---|---|---|---|---|---|
| Gemini 3.1 Pro | 85.2 | 66.6 | 88.5 | 83.2 | **64.5** | **77.6** | $154 |
| Gemini 3 Flash | 84.1 | 65.7 | 87.6 | 86.3 | 52.3 | 75.2 | $56 |
| Qwen3.6-27B | 84.8 | 53.6 | 84.9 | 83.6 | 62.4 | 73.9 | $103 |
| Gemini 3.1 Flash-Lite (no reasoning) | 82.9 | 21.7 | 85.1 | 83.7 | 49.0 | 64.5 | $7 |
| Octen-8B (emb, 7.6B) | 90.1 | 65.1 | 88.7 | 86.1 | 56.0 | 77.2 | **$0.11** |
| Jina-v5-Nano (emb, **212M**) | 89.6 | 60.4 | 87.0 | 85.0 | 47.4 | 73.9 | **$0.01** |
| mE5-small (emb, 118M) | 69.8 | 47.7 | 78.5 | 83.9 | 35.8 | 63.1 | $0.001 |

A 212M-parameter embedding model ties Qwen3.6-27B overall for one cent versus $103.

**Per-task view (Figure 6) — the cleanest statement of the whole paper.** Some embedding model matches the best LLM on **7 of 8** classification tasks and **7 of 10** STS tasks, but on only **1 of 6** retrieval tasks.

**Bootstrap significance (10,000 resamples, task-level resampling):**

| Comparison | $\Delta$ | 95% CI | $p$ |
|---|---|---|---|
| Overall | +0.3 | $[-2.4, +3.1]$ | 0.85 |
| Retrieval | **+8.5** | $[+0.2, +16.8]$ | $<0.05$ |
| Clustering | −0.2 | $[-5.6, +5.1]$ | 0.96 |
| STS | −0.3 | $[-2.2, +1.8]$ | 0.75 |
| PairCls | −3.9 | $[-13.6, +5.9]$ | 0.50 |
| Classification | **−5.6** | $[-9.2, -2.4]$ | $<0.01$ |

Under Bonferroni at $\alpha = 0.01$, only classification survives. Retrieval's confidence interval nearly touches zero — six tasks is thin evidence.

### Retrieve-then-rerank (the production-realistic experiment)

Four first-stage retrievers × cross-encoder and LLM listwise rerankers, on BEIR (semantic) and BRIGHT (reasoning-heavy). nDCG@10 — see [[NDCG]]:

| First stage | Alone | +Qwen3-RR-4B (cross-enc) | +Qwen3.6-27B (LLM listwise) |
|---|---|---|---|
| **BRIGHT** BM25 | 10.2 | 22.0 | 24.2 |
| **BRIGHT** Qwen3-E-8B | 22.3 | 26.4 | **35.1** |
| **BEIR** BM25 | 39.8 | 53.2 | 52.1 |
| **BEIR** Qwen3-E-8B | **63.1** | 60.3 | 58.9 |

On reasoning-heavy BRIGHT, an LLM reranker takes a strong embedding from 22.3 → 35.1. On semantic BEIR, **the embedding alone beats every reranked configuration** (63.1 vs 60.3 best). Reranking makes semantic retrieval *worse*.

Cost: reranking a top-100 shortlist is $10–30 per benchmark, versus $154 for reading the whole corpus in context. The MoE Qwen3.6-35B-A3B gets 33.6 on BRIGHT for $10 where the dense 27B gets 35.1 for $30.

### Ablation 1 — reduce the thinking

Gemini 3 Flash with `reasoning_effort=low`, and open models with reasoning disabled at the serving layer.

| Retrieval task | Flash default | Flash low | $\Delta$ | Thinking cut |
|---|---|---|---|---|
| AILAStatutes | 5.7 | 12.0 | +6.3 | — |
| FQuAD | 88.0 | 92.0 | +4.0 | 54% |
| HC3Finance | 60.0 | 66.0 | +6.0 | 87% |
| ConsumerContractsQA | 79.0 | 83.0 | +4.0 | 85% |
| PublicHealthQA | 49.0 | **66.0** | **+17.0** | 94% |
| TwitterHjerne | 32.4 | 33.4 | +1.1 | — |

**All six retrieval scores improved with less reasoning.** Across five families, 4 of 6 models preserved or improved retrieval with 54–96% fewer generated tokens; only the two Qwen models lost ground. Classification moved by $|\Delta| < 1.0$ on every task tested.

The hypothesis: [[Chain-of-Thought Prompting Elicits Reasoning in LLMs|chain-of-thought]] causes second-guessing. The model talks itself out of the obviously relevant document. This echoes Lu et al. (2025), who found CoT reranking consistently underperforms direct-output reranking.

### Ablation 2 — few-shot classification (this one failed hard)

Give Flash 5 [[In Context Learning|in-context]] examples and see if the classification gap closes.

| Task | Classes | Zero-shot | 5-shot | $\Delta$ |
|---|---|---|---|---|
| IMDB | 2 | 0.976 | 0.974 | −0.002 |
| ToxicConversations | 2 | 0.900 | 0.838 | −0.062 |
| TweetSentiment | 3 | 0.700 | 0.682 | −0.018 |
| **Banking77** | **77** | **0.831** | **0.165** | **−0.666** |

Five examples for 77 labels is worse than no examples at all — accuracy collapses to 16.5%. The five shown labels apparently anchor the model onto a tiny slice of the label space. Meanwhile the kNN classifier is using the *entire* labelled train split. This is not a fair fight, and the authors say so.

### What the qualitative errors show

The classification gap is largely **dataset-convention mismatch**, not reasoning failure:

- Banking77, "How do I locate my card?" — Pro reasons "locate implies the card is missing" → predicts `lost_or_stolen_card`. Gold is `card_arrival` (delivery tracking). Flash, with *less* reasoning, gets it right. kNN just matches "Where is my card?" from the train set and gets it right.
- AmazonCounterfactual, "Would be great to have on my tablet" — Pro says `counterfactual` (it's hypothetical). Dataset convention labels forward-looking wishes `not-counterfactual`.
- STS, "Islamist identity" vs "Islamic identity" — Pro scores 4.5 (political vs religious adjective). Gold says 5.0. A linguistically defensible distinction that costs points.

The kNN classifier is not smarter. It has read the annotation guidelines by proxy, through 3,000 labelled examples.

**Where reasoning actively hurts retrieval:** AILAStatutes (match case facts to legal statutes). Pro returns 14 documents spanning contract law, employment law, due process — plausible, associatively linked, wrong. Gold has 4 specific statutes. Octen-8B scores 23.2, Pro 14.5. Cosine similarity produces a narrower ranking precisely because it *cannot* free-associate.

### Cost sensitivity

The 1,431× figure is not fragile:

| Scenario | Emb. cost | Ratio |
|---|---|---|
| H100 spot $2.49/hr | $0.108 | 1,431× |
| H100 on-demand $3.99/hr | $0.173 | 893× |
| L4 spot $0.49/hr (3× slower) | $0.064 | 2,424× |
| Commercial embedding API $0.10/MTok | $0.457 | 338× |

The floor is 338×.

**Token breakdown.** Qwen3.6-27B generated 26.1M reasoning tokens against 2.7M output tokens — 81% of its $103 bill was thinking. Gemini 3.1 Pro: 8.6M reasoning, 1.8M output, $124 of $154 in the "output + think" column.

## Worth Remembering

**The cost gap reported is a lower bound, and the authors are explicit about why.** Corpora here are 82–415 documents. At real scale you cannot put the corpus in the prompt at all — you need an index, which means you need embeddings anyway. Also: document embeddings are computed **once** and reused forever; corpus-in-context reading repeats **per query**, only partly offset by prompt caching. And embedding *training* is a one-time cost amortised across every user of a public checkpoint, while LLM inference cost recurs in full on every call.

**Two models fail catastrophically and it is worth asking why.** E5-Mistral-7B scores 52.2 overall and **20.6** on retrieval; [GritLM-7B](https://arxiv.org/abs/2402.09906) scores 51.2 overall and **6.1** on retrieval — near random. Both are LLM-backbone embedding models. On STS22v2, E5-Mistral scores $-21.4$, i.e. *anti*-correlated with human judgment. The paper never explains this. My guess is instruction-prefix mismatch: these models expect a specific task instruction format and were run without it. Do not read those rows as evidence about LLM-derived embeddings in general.

**The clustering result is the strangest.** Gemini 3.1 Pro gets 66.6 on clustering by being handed all documents in one prompt *plus the ground-truth number of clusters $k$*, and returning a JSON array of assignments. It beats every embedding model on RedditClusteringP2P (93.9 vs 80.8). But non-reasoning Flash-Lite collapses to 21.7 and DeepSeek-R1 to 31.3. Clustering is where LLM capability variance is largest — the spread is 21.7 to 66.6 across LLMs versus 45.9 to 66.7 across embeddings.

**Limitations the authors admit:**
- Ten LLMs, six families, one snapshot. GPT-5 and Claude Opus 4.6 were not evaluated.
- The classification comparison is between *deployment pipelines*, not model ceilings. A classification-post-trained LLM would likely close the gap; none existed among the frontier models tested.
- Leading embedding models are contrastively trained on data overlapping these exact domains and label spaces. Part of the 5.6-point classification gap is in-domain fit, not representational superiority. Compare [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)|the RecSys replication paper]] and [[Towards Quantifying Benchmark Optimization in ASR Models]] on this exact hazard.
- Aggregate scores weight all 37 tasks equally. Borda count (as MMTEB uses) might flip the paradigm ordering.
- Reasoning controls differ by provider — Gemini's `reasoning_effort=low` is a soft hint, open models can genuinely disable it. Not the same intervention.

**Practical takeaways if you are building something:**

1. Default to an embedding model for classification, clustering, STS, and pair classification. Not close on cost, tied on quality.
2. For semantic retrieval (BEIR-like), a strong bi-encoder alone beats reranking. Adding an LLM reranker made it *worse* (63.1 → 60.3).
3. For reasoning-heavy retrieval (BRIGHT-like), retrieve-then-rerank on a top-100 shortlist. $10–30, not $154.
4. **Turn reasoning down.** It is the single largest line item and it does not help these tasks.
5. Prefer MoE for reranking: Qwen3.6-35B-A3B got 33.6 nDCG@10 for $10 vs the dense 27B's 35.1 for $30. 4% quality for 3× cost.
6. Report Pareto frontiers, not just accuracy. Two systems at 77 points can differ by three orders of magnitude in cost — a leaderboard column hides that entirely. Same argument as [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)|the technical-debt paper]] makes about hidden system costs.

**Open questions:** How does the retrieval advantage decay as corpus size grows from 400 to 4M documents? Nobody varied it. And if the classification gap is really annotation-convention mismatch rather than capability, would a few hundred labelled examples in a fine-tune close it entirely — making the whole comparison a statement about supervision, not architecture?

## Links

Related: [[Sentence-BERT]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Embedding-based Retrieval in Facebook Search]] · [[NDCG]] · [[Chain-of-Thought Prompting Elicits Reasoning in LLMs]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[In Context Learning]] · [[A Simple Framework for Contrastive Learning (SimCLR)]] · [[SimCSE- Simple Contrastive Learning of Sentence Embeddings]] · [[Representation Learning with Contrastive Predictive Coding (CPC - InfoNCE)]] · [[Understanding Contrastive Learning through Alignment and Uniformity]] · [[Distilling the Knowledge in a Neural Network]] · [[Distributed Representations of Words and Phrases (negative sampling)]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[On the Difficulty of Evaluating Baselines]] · [[Towards Quantifying Benchmark Optimization in ASR Models]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Recommender Systems - Evolution]] · [[Attention Is All You Need]] · [[FlashAttention- Fast and Memory-Efficient Exact Attention]] · [[Scaling Laws for Neural Language Models]]

New topics worth writing: MTEB (Massive Text Embedding Benchmark), BEIR, BRIGHT reasoning-intensive retrieval, cross-encoder reranking, ColBERT and late interaction, corpus-in-context prompting (LOFT), V-measure, Spearman correlation as an evaluation metric, prompt caching economics, vLLM serving, mixture-of-experts inference cost, paired bootstrap significance testing, Green AI and inference carbon accounting, Pareto frontier reporting for model selection, GritLM and unified generation-representation models, test-time compute budgets and overthinking
