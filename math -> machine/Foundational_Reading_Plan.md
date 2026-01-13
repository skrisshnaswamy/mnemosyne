---
tags: [reading-plan, papers, foundational]
taxonomy_revision: 1
built: 2026-08-22
source_lists: [Reading_Queue.md, Foundational_Reading_Tracker (Notion)]
---

# Foundational Reading Plan

Built by merging your two existing queues, applying `taxonomy` rev 1 honestly, and filling the gaps. New rows are numbered from **69** so this appends directly to the bottom of `Reading_Queue.md`.

---

## Diagnosis — what your two lists say about you

**Finding 1: Your lists contain almost nothing about what you actually do.**

Across 68 rows in `Reading_Queue.md` and 29 in the Notion tracker, there is not one paper on recommender systems, ranking, retrieval, embeddings-as-a-product, or bandits. Zero. Your day job is merchant representation learning and downstream ranking; your stated destination is decision-making under uncertainty. The reading lists are optimised for a different person — someone tracking the generative-AI frontier.

This isn't a criticism of the papers on the lists. It's that the lists were seeded from "canonical modern ML" and then fed by an HF-daily firehose, and neither of those sources knows what you work on. **The single highest-value change is adding Phases 3 and 4 below.**

**Finding 2: The generative track is over-weighted for your goals.**

Between the two lists you have VAE, GAN, VQ-VAE, DALL·E, DDPM, Improved DDPM, Latent Diffusion, Score-SDE, a diffusion visualiser, HF Diffusers, and a diffusion-LLM paper. That's ~11 items — the largest single block. Diffusion is genuinely important and you should understand it, but for a ranking/representation MLE moving toward decision intelligence, it's context, not core. I've kept the load-bearing four and demoted the rest.

**Finding 3: The `hf-daily` intake is crowding out the foundations.**

Rows 38-68 are 31 papers from two days of ingestion, ~70% correctly marked Low-Priority. At that rate you'll add ~450/month against a foundational backlog of ~30 unread seed papers. The triage is working; the queue discipline isn't. Suggestion at the end.

---

## Priority corrections to rows already in your queue

Applying the taxonomy definitions as written, not as previously assigned.

| Row | Paper | Yours | Mine | Reasoning |
|---|---|---|---|---|
| 15 | Adam | Good-To-Read | **Must-Read** | "Training recipe that got widely adopted" is the taxonomy's own signal. Adam is the default optimiser for essentially everything you'll train. Also short and readable. |
| 14 | Batch Normalization | Good-To-Read | **Must-Read** | Origin point of the entire normalisation line, and its *stated* explanation was later shown wrong — which makes it a case study in how the field self-corrects. Read it with Santurkar (row 100 below). |
| 24 | LeJEPA | Must-Read | **Good-To-Read** | Published Nov 2025. The taxonomy's Must-Read bar is "changed how the field thinks" — that's a verdict history renders, not one you can assign on release. Promote it in a year if it holds. Right now it's a bet, and labelling bets as Must-Read is how a queue loses its signal. |
| 30+31 | Dreamer + DreamerV3 | Both Must-Read | **V3 Must-Read, V1 Good-To-Read** | V3 supersedes V1 and is self-contained. Reading both is redundant unless you want the lineage. |
| 33 | Distributional RL (C51) | Must-Read | **Must-Read — agree** | Worth affirming: modelling the full return distribution rather than its mean is precisely the "decisions under uncertainty" framing you said you want. Good call. |
| 25 | DINOv2 | Low-Priority | **Low-Priority — agree** | Excellent paper, wrong domain for you. |
| 7 | LLaMA | Low-Priority | **Low-Priority — agree** | It's an engineering/recipe paper. Chinchilla already gave you the scaling insight. |
| — | DALL·E (Notion wk4) | — | **Low-Priority** | VQ-VAE is the durable contribution; DALL·E-1's specific recipe is superseded. Keep VQ-VAE, drop DALL·E. |
| — | Improved DDPM (Notion wk5) | — | **Low-Priority** | Incremental over DDPM. Read DDPM then jump to Score-SDE for the unifying view. |

---

## Phase 0 — The actual classics (missing from both lists)

You said "old papers I definitely should have read." These are those. Every one is short by modern standards, and several are more readable than their descendants. If you want research taste — the "AI scientist" instinct — this is where it comes from: seeing how the field's central ideas looked before they were polished.

Read in order. Two evenings total.

| #   | Paper | Authors | Year | Priority | Status | Source | Added | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 69 | Learning representations by back-propagating errors (*Nature*) | Rumelhart, Hinton & Williams | 1986 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 0. Four pages. The origin of everything. |
| 70 | [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) (*Neural Computation*) | Hochreiter & Schmidhuber | 1997 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 0. Read §3-4 for the constant error carousel; skim the rest. Pairs with Q26 in the bank. |
| 71 | [Gradient-Based Learning Applied to Document Recognition](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf) (LeNet) | LeCun et al. | 1998 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 0. Read §1-3. The inductive-bias argument still matters for your tabular work. |
| 72 | [A Neural Probabilistic Language Model](https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) (*JMLR*) | Bengio et al. | 2003 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 0. **The origin of learned embeddings.** Direct ancestor of everything you do with merchant vectors. |
| 73 | [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) (essay) | Sutton | 2019 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 0. 800 words. Read it twice a year. |
| 74 | [Understanding the difficulty of training deep feedforward networks](https://proceedings.mlr.press/v9/glorot10a.html) (Xavier init) | Glorot & Bengio | 2010 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 0. The variance-propagation argument behind Q6. |
| 75 | [Delving Deep into Rectifiers](https://arxiv.org/abs/1502.01852) (He init, PReLU) | He et al. | 2015 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 0. Short. Read alongside Xavier. |
| 76 | [Dropout: A Simple Way to Prevent Overfitting](https://jmlr.org/papers/v15/srivastava14a.html) | Srivastava et al. | 2014 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 0. |
| 77 | [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) (AdamW) | Loshchilov & Hutter | 2017 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 0. **Missing from your list.** This is the optimiser you actually use. The L2-vs-decay distinction is a classic interview probe (Q10). |

---

## Phase 1 — Core deep learning mechanics

Mostly already in your queue. Order matters here: AlexNet → BatchNorm → ResNet → Adam/AdamW → LayerNorm → Attention. Read them as one arc about *making deep networks trainable*, because that's what they collectively are.

Already queued: rows 12 (AlexNet), 13 (ResNet), 14 (BatchNorm), 15 (Adam), 16 (LayerNorm), 1 (Attention ✓), 3 (BERT ✓), 2 (GPT-1 ✓), 21 (Distillation).

| #   | Paper | Authors | Year | Priority | Status | Source | Added | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 78 | [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) | Liu et al. | 2019 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 1. The cleanest demonstration that the *recipe* beat the architecture. Kills NSP. |
| 79 | [ELECTRA: Pre-training Text Encoders as Discriminators](https://arxiv.org/abs/2003.10555) | Clark et al. | 2020 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 1. Best answer to "how would you improve MLM's 15% signal problem" — relevant to your masked-event modelling. |
| 80 | [Exploring the Limits of Transfer Learning (T5)](https://arxiv.org/abs/1910.10683) | Raffel et al. | 2019 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 1. Skim §3 — it's a systematic ablation study, which is the format your own work takes. |
| 81 | [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | Su et al. | 2021 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 1. RoPE is the default everywhere now. Q43. |
| 82 | [Train Short, Test Long (ALiBi)](https://arxiv.org/abs/2108.12409) | Press et al. | 2021 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 1. Read with RoPE for the contrast. |

---

## Phase 2 — Representation learning & embeddings

**This is your job, and it is almost entirely absent from both lists.** Your queue has SimCLR, I-JEPA and LeJEPA; it's missing the entire line that produced modern embedding practice — the contrastive objective's origin, the collapse/anisotropy literature, and everything about turning a language model into a usable vector space.

Read in this order. The anisotropy cluster (86-89) is four short papers that together explain why "just take BERT's CLS token" fails, which is the single most useful thing in this phase for your merchant work.

| #   | Paper | Authors | Year | Priority | Status | Source | Added | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 83 | [Efficient Estimation of Word Representations](https://arxiv.org/abs/1301.3781) (word2vec) | Mikolov et al. | 2013 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 2. **Missing from your list.** The origin of the ID-embedding paradigm you use daily. |
| 84 | [Distributed Representations of Words and Phrases](https://arxiv.org/abs/1310.4546) (negative sampling) | Mikolov et al. | 2013 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 2. The negative-sampling paper. Direct ancestor of your in-batch negatives. |
| 85 | [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) (CPC / InfoNCE) | van den Oord et al. | 2018 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 2. **The InfoNCE loss originates here.** Everything contrastive descends from this. |
| 86 | [Representation Degeneration Problem in Training NLMs](https://arxiv.org/abs/1907.12009) | Gao et al. | 2019 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 2. Anisotropy cluster 1/4. Why LM embeddings form a cone. |
| 87 | [How Contextual are Contextualized Word Representations?](https://arxiv.org/abs/1909.00512) | Ethayarajh | 2019 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 2. Anisotropy cluster 2/4. Measures the cone empirically across BERT/GPT-2/ELMo. |
| 88 | [Understanding Contrastive Learning through Alignment and Uniformity](https://arxiv.org/abs/2005.10242) | Wang & Isola | 2020 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 2. Anisotropy cluster 3/4. The decomposition that gives you embedding *health metrics*. Directly applicable to emb-bench. |
| 89 | [Whitening Sentence Representations](https://arxiv.org/abs/2103.15316) | Su et al. | 2021 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 2. Anisotropy cluster 4/4. Shows simple whitening matches BERT-flow. Q64. |
| 90 | [Sentence-BERT](https://arxiv.org/abs/1908.10084) | Reimers & Gurevych | 2019 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 2. Why CLS pooling fails and mean pooling + a contrastive objective works. Q57. |
| 91 | [SimCSE: Simple Contrastive Learning of Sentence Embeddings](https://arxiv.org/abs/2104.08821) | Gao et al. | 2021 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 2. Dropout-as-augmentation. Almost embarrassingly simple and it works. |
| 92 | [Momentum Contrast (MoCo)](https://arxiv.org/abs/1911.05722) | He et al. | 2019 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 2. The negative-queue idea — how to get many negatives without a huge batch. |
| 93 | [Bootstrap Your Own Latent (BYOL)](https://arxiv.org/abs/2006.07733) | Grill et al. | 2020 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 2. Collapse-avoidance without negatives. |
| 94 | [Exploring Simple Siamese Representation Learning (SimSiam)](https://arxiv.org/abs/2011.10566) | Chen & He | 2020 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 2. The ablation that isolates what actually prevents collapse. |
| 95 | [Barlow Twins](https://arxiv.org/abs/2103.03230) | Zbontar et al. | 2021 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 2. Decorrelation as an explicit anti-collapse objective. |
| 96 | [VICReg](https://arxiv.org/abs/2105.04906) | Bardes, Ponce & LeCun | 2021 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 2. The most interpretable of the family: write down what you want the space to look like, regularise toward it. Read before LeJEPA. |
| 97 | [Understanding Dimensional Collapse in Contrastive Learning](https://arxiv.org/abs/2110.09348) | Jing et al. | 2021 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 2. The singular-spectrum diagnostic you should be logging during training. Q68. |
| 98 | [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) | Kusupati et al. | 2022 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 2. Nested embeddings — truncate to any dimension at serving time. Q73. |

---

## Phase 3 — Recommenders, ranking & retrieval

**Zero coverage in either list.** This is the literature of your actual job. If an interviewer for a senior ranking role asks what you've read in the space, right now the honest answer is "nothing," which is a strange thing to be true of someone who's built these systems.

The four Must-Reads here (100, 102, 103, 108) are the ones I'd do first — they're the papers whose ideas you're already using without having read the source.

| #   | Paper | Authors | Year | Priority | Status | Source | Added | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 99 | [Matrix Factorization Techniques for Recommender Systems](https://ieeexplore.ieee.org/document/5197422) (*IEEE Computer*) | Koren, Bell & Volinsky | 2009 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 3. The Netflix Prize synthesis. Short, non-technical, still the right mental model for latent factors. |
| 100 | [BPR: Bayesian Personalized Ranking from Implicit Feedback](https://arxiv.org/abs/1205.2618) | Rendle et al. | 2009 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 3. **Why implicit feedback needs a pairwise objective.** The foundational argument for ranking losses over pointwise ones. |
| 101 | [Wide & Deep Learning for Recommender Systems](https://arxiv.org/abs/1606.07792) | Cheng et al. | 2016 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 3. The memorisation-vs-generalisation framing behind your hybrid ID+content design (Q56). |
| 102 | [Deep Neural Networks for YouTube Recommendations](https://research.google/pubs/pub45530/) (RecSys) | Covington, Adams & Sargin | 2016 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 3. **The most practically useful recsys paper ever written.** Candidate generation + ranking cascade, sampled softmax, example age feature. Read it twice. |
| 103 | [Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations](https://research.google/pubs/pub48840/) (RecSys) | Yi et al. | 2019 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 3. **The logQ correction.** You are almost certainly affected by this problem in the merchant model. Q110. |
| 104 | [Deep Learning Recommendation Model (DLRM)](https://arxiv.org/abs/1906.00091) | Naumov et al. | 2019 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 3. The systems view — embedding tables as the memory bottleneck. Plays to your infra background. |
| 105 | [Modeling Task Relationships with Multi-gate MoE (MMoE)](https://dl.acm.org/doi/10.1145/3219819.3220007) (KDD) | Ma et al. | 2018 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 3. Multi-task ranking. Q78. |
| 106 | [Progressive Layered Extraction (PLE)](https://dl.acm.org/doi/10.1145/3383313.3412236) (RecSys) | Tang et al. | 2020 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 3. Fixes MMoE's negative transfer. Read only if multi-task is live for you. |
| 107 | [From RankNet to LambdaRank to LambdaMART: An Overview](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) | Burges | 2010 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 3. How to optimise a non-differentiable metric. Still the best explanation of NDCG-aware gradients. Q113. |
| 108 | [Unbiased Learning-to-Rank with Biased Feedback](https://arxiv.org/abs/1608.04468) | Joachims, Swaminathan & Schnabel | 2016 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 3. **Position bias and IPS.** The bridge between your causal-inference work and ranking. Q77. |
| 109 | [Dense Passage Retrieval (DPR)](https://arxiv.org/abs/2004.04906) | Karpukhin et al. | 2020 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 3. The canonical two-tower retrieval recipe. |
| 110 | [ColBERT: Efficient and Effective Passage Search via Late Interaction](https://arxiv.org/abs/2004.12832) | Khattab & Zaharia | 2020 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 3. The middle of the bi-encoder/cross-encoder spectrum. Q117. |
| 111 | [Approximate Nearest Neighbor Negative Contrastive Learning (ANCE)](https://arxiv.org/abs/2007.00808) | Xiong et al. | 2020 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 3. Hard negative mining done properly. Q66. |
| 112 | [Efficient and robust approximate nearest neighbor search using HNSW](https://arxiv.org/abs/1603.09320) | Malkov & Yashunin | 2016 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 3. You depend on this in production; worth understanding the graph construction. Q119. |
| 113 | [Product Quantization for Nearest Neighbor Search](https://inria.hal.science/inria-00514462/) (*IEEE TPAMI*) | Jégou, Douze & Schmid | 2011 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 3. Compression for billion-scale indexes. |

---

## Phase 4 — Decision under uncertainty: bandits, off-policy evaluation, causal

**This is your career bridge, and it is completely absent from both lists.** Your Notion tracker jumps straight from RL Basics to PPO/RLHF, with causal inference as two YouTube lectures at the very end. That ordering has it backwards for you: you already *do* causal inference professionally, and bandits — not deep RL — are the natural next step from where you stand.

Read Phase 4 **before** Phase 5. Bandits are the special case of RL where actions don't change state, and understanding that boundary is what lets you argue about when RL is warranted (Q99). It's also the honest answer to "you have no RL experience" — you have adjacent experience, and this phase is what converts adjacent into legible.

If you only read six papers from this entire document, read 116, 117, 119, 121, 124 and 126.

| #   | Paper | Authors | Year | Priority | Status | Source | Added | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 114 | [Finite-time Analysis of the Multiarmed Bandit Problem](https://link.springer.com/article/10.1023/A:1013689704352) (UCB1) | Auer, Cesa-Bianchi & Fischer | 2002 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 4. The optimism-under-uncertainty principle and where regret bounds come from. Skim the proofs. |
| 115 | [A Contextual-Bandit Approach to Personalized News Article Recommendation](https://arxiv.org/abs/1003.0146) (LinUCB) | Li et al. | 2010 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 4. Contextual bandits applied to exactly your problem shape. Also introduces the replay-based offline evaluator. |
| 116 | [A Tutorial on Thompson Sampling](https://arxiv.org/abs/1707.02038) | Russo, Van Roy, Kazerouni, Osband & Wen | 2018 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 4. **Start here.** The single best entry point to the whole area, and it directly formalises what you built in the BAO/Tree-CNN project. |
| 117 | [An Empirical Evaluation of Thompson Sampling](https://papers.nips.cc/paper/2011/hash/e53a0a2978c28872a4505bdb51db06dc-Abstract.html) (NeurIPS) | Chapelle & Li | 2011 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 4. Why TS beats UCB in practice, including under **delayed and batched feedback** — the regime you'd actually deploy in. |
| 118 | [Doubly Robust Policy Evaluation and Learning](https://arxiv.org/abs/1103.4601) | Dudík, Langford & Li | 2011 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 4. The DR estimator. Q100. |
| 119 | [Counterfactual Reasoning and Learning Systems](https://arxiv.org/abs/1209.2355) | Bottou et al. | 2013 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 4. **The most underrated paper on this list for you specifically.** Causal reasoning applied to a real production ad system, by people who ran one. It is the intellectual bridge between your experimentation platform and policy learning. Long; worth every page. |
| 120 | [Counterfactual Risk Minimization](https://arxiv.org/abs/1502.02362) | Swaminathan & Joachims | 2015 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 4. Learning (not just evaluating) policies from logged bandit feedback. The variance problem and the self-normalised fix. |
| 121 | [Offline Reinforcement Learning: Tutorial, Review, and Perspectives](https://arxiv.org/abs/2005.01643) | Levine, Kumar, Tucker & Fu | 2020 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 4. **Learning policies from logged data you didn't collect** — i.e. every dataset you have ever had. The most directly applicable RL paper for someone with your background. |
| 122 | [Conservative Q-Learning for Offline RL](https://arxiv.org/abs/2006.04779) | Kumar et al. | 2020 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 4. Read after 121 if offline RL becomes real for you. |
| 123 | [Practical Bayesian Optimization of Machine Learning Algorithms](https://arxiv.org/abs/1206.2944) | Snoek, Larochelle & Adams | 2012 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 4. **You built infrastructure for this without reading the paper.** Acquisition functions, GP surrogates, the explore/exploit trade in continuous spaces. Q101. |
| 124 | [Gaussian Process Optimization in the Bandit Setting (GP-UCB)](https://arxiv.org/abs/0912.3995) | Srinivas et al. | 2010 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 4. Makes the BayesOpt↔bandit equivalence explicit and gives it regret bounds. |
| 125 | [Improving the Sensitivity of Online Controlled Experiments (CUPED)](https://exp-platform.com/cuped/) (WSDM) | Deng, Xu, Kohavi & Walker | 2013 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 4. **You run 1,200+ metric pipelines; you should have read this.** Variance reduction using pre-experiment data. Q140. |
| 126 | [Trustworthy Online Controlled Experiments](https://experimentguide.com/) (book — Kohavi, Tang, Xu) | Kohavi et al. | 2020 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 4. Not a paper, but the definitive reference for what you already do. Read Ch. 3, 17-19, 22. SRM, interference, switchbacks, novelty effects. |
| 127 | [Double/Debiased Machine Learning for Treatment and Structural Parameters](https://arxiv.org/abs/1608.00060) | Chernozhukov et al. | 2016 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 4. How to use ML models inside causal estimation without inheriting their bias. Cross-fitting. |
| 128 | [Recursive Partitioning for Heterogeneous Causal Effects](https://arxiv.org/abs/1504.01132) | Athey & Imbens | 2015 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 4. Causal trees / heterogeneous treatment effects — "who does this help" rather than "does this help." |

---

## Phase 5 — Reinforcement learning proper

Now the destination. Your queue's RL selection is good; I'm mostly reordering it and adding the two papers that make it coherent (GAE, and DPO for the alignment line).

**Read Sutton & Barto alongside these, not instead of them.** Chapters 3-6 (MDPs, Monte Carlo, TD learning) and 13 (policy gradients) are the scaffolding the papers assume. Skipping the textbook and reading only papers is how people end up able to describe PPO but not able to say what a value function is for.

Already queued: rows 26 (DQN), 27 (AlphaGo), 28 (PPO), 29 (SAC), 30/31 (Dreamer), 32 (DDPG), 33 (C51), 8 (InstructGPT). Notion adds REINFORCE.

| #   | Paper | Authors | Year | Priority | Status | Source | Added | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 129 | [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book-2nd.html) (book, 2nd ed.) | Sutton & Barto | 2018 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 5. Ch. 1-6 and 13. The scaffolding. Free PDF. |
| 130 | [Simple Statistical Gradient-Following Algorithms (REINFORCE)](https://link.springer.com/article/10.1007/BF00992696) | Williams | 1992 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 5. In your Notion list. The policy-gradient theorem in its original form. |
| 131 | [Trust Region Policy Optimization (TRPO)](https://arxiv.org/abs/1502.05477) | Schulman et al. | 2015 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 5. Read the intro and §2-3 only — it's the *motivation* PPO simplifies. Skipping it makes PPO look arbitrary. |
| 132 | [High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)](https://arxiv.org/abs/1506.02438) | Schulman et al. | 2015 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 5. The bias-variance knob (λ) inside every PPO implementation. Read with PPO. |
| 133 | [Mastering Chess and Shogi by Self-Play (AlphaZero)](https://arxiv.org/abs/1712.01815) | Silver et al. | 2017 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 5. Cleaner than AlphaGo — no human data, no handcrafted features. Read instead of row 27 if short on time. |
| 134 | [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) | Rafailov et al. | 2023 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 5. **Missing from both lists.** The reparameterisation that removed the RL loop from alignment. Now the default. Read right after InstructGPT. Q162. |
| 135 | [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) | Bai et al. | 2022 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 5. How preference data scales past human labelling. |

---

## Phase 6 — Generative modelling

Keeping the load-bearing four. Your two lists between them have ~11 generative items; for your trajectory that's over-invested. The point of this phase is fluency, not depth — you should be able to explain the latent-variable/score/denoising connections, not implement a diffusion model.

Already queued: rows 17 (VAE), 18 (GAN), 19 (DDPM), 37 (Score-SDE), plus Notion's VQ-VAE, Latent Diffusion, Improved DDPM, DALL·E.

**Read: VAE → GAN → DDPM → Score-SDE.** That arc gives you the whole picture. VQ-VAE and Latent Diffusion if the discrete-latent idea interests you (it's relevant to tokenising continuous features, which touches your tabular work). Skip DALL·E and Improved DDPM.

| #   | Paper | Authors | Year | Priority | Status | Source | Added | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 136 | [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) | Ho & Salimans | 2022 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 6. Two pages, and it's the trick behind every conditional diffusion model in production. |

---

## Phase 7 — Systems, scaling & efficiency

Your strongest existing area, and the phase where you should read *least* per unit of understanding gained — you already know most of this operationally. But the papers give you the vocabulary to argue about it, and several are directly relevant to your 75%-training-time-reduction story.

Already queued: rows 10 (FlashAttention), 5 (Scaling Laws ✓), 6 (Chinchilla ✓), 11 (LoRA), 34 (Muon), 35 (Lion).

| #   | Paper | Authors | Year | Priority | Status | Source | Added | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 137 | [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | Rajbhandari et al. | 2019 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 7. The stage-1/2/3 sharding decomposition behind FSDP. Q80. Directly relevant to your multi-A100 work. |
| 138 | [Megatron-LM: Training Multi-Billion Parameter Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) | Shoeybi et al. | 2019 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 7. Tensor parallelism. Q81. |
| 139 | [Mixed Precision Training](https://arxiv.org/abs/1710.03740) | Micikevicius et al. | 2017 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 7. Loss scaling, master weights, what must stay fp32. Q14. |
| 140 | [Fast Transformer Decoding: One Write-Head is All You Need (MQA)](https://arxiv.org/abs/1911.02150) | Shazeer | 2019 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 7. Three pages. The KV-cache insight. |
| 141 | [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) | Ainslie et al. | 2023 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 7. The MHA/MQA compromise everyone now uses. |
| 142 | [Efficient Memory Management for LLM Serving with PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180) | Kwon et al. | 2023 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 7. A pure systems insight beating a modelling one — virtual memory for KV cache. Very much your kind of paper. Q131. |
| 143 | [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | Dettmers et al. | 2023 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 7. Read after LoRA (row 11). NF4, double quantisation, paged optimisers. |

---

## Phase 8 — Tabular & classical ML

**Absent from both lists**, despite tabular/event data being what you actually train on. The Grinsztajn paper in particular is the one to have read before any conversation about why you chose a deep model.

| #   | Paper | Authors | Year | Priority | Status | Source | Added | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 144 | [XGBoost: A Scalable Tree Boosting System](https://arxiv.org/abs/1603.02754) | Chen & Guestrin | 2016 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 8. Second-order split criterion, sparsity-aware splitting, the systems design. Q145. |
| 145 | [LightGBM: A Highly Efficient Gradient Boosting Decision Tree](https://papers.nips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) (NeurIPS) | Ke et al. | 2017 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 8. Histogram binning, GOSS, EFB, leaf-wise growth. |
| 146 | [CatBoost: Unbiased Boosting with Categorical Features](https://arxiv.org/abs/1706.09516) | Prokhorenkova et al. | 2018 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 8. **Ordered target statistics** — an elegant fix to a leakage problem you will hit with high-cardinality merchant IDs. |
| 147 | [Why do tree-based models still outperform deep learning on tabular data?](https://arxiv.org/abs/2207.08815) | Grinsztajn, Oyallon & Varoquaux | 2022 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 8. **Read before defending any deep-tabular choice.** The three structural reasons. Q86. |
| 148 | [Revisiting Deep Learning Models for Tabular Data (FT-Transformer)](https://arxiv.org/abs/2106.11959) | Gorishniy et al. | 2021 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 8. The honest benchmark — most deep-tabular claims don't survive proper tuning of the GBDT baseline. |
| 149 | [TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](https://arxiv.org/abs/2207.01848) | Hollmann et al. | 2022 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 8. In-context learning for tables; approximating Bayesian inference over a synthetic prior. Q88. |
| 150 | [A Unified Approach to Interpreting Model Predictions (SHAP)](https://arxiv.org/abs/1705.07874) | Lundberg & Lee | 2017 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 8. Shapley values, the axioms, TreeSHAP. Q147. |

---

## Phase 9 — Understanding, failure, and research taste

The "AI scientist" phase. These aren't about building anything — they're about knowing what's actually established, how the field corrects itself, and what breaks. If you want research instincts rather than practitioner instincts, this is the phase that produces them.

Row 151 is the one I'd push hardest. It is the paper most relevant to a senior engineer with twenty years of systems experience, and almost nobody in ML has read it.

| #   | Paper | Authors | Year | Priority | Status | Source | Added | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 151 | [Hidden Technical Debt in Machine Learning Systems](https://papers.nips.cc/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html) (NeurIPS) | Sculley et al. | 2015 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 9. **Nine pages, no maths, and it is your entire career thesis stated by Google in 2015.** CACE, entanglement, pipeline jungles, glue code. Cite it when you argue for infrastructure. |
| 152 | [Understanding Deep Learning Requires Rethinking Generalization](https://arxiv.org/abs/1611.03530) | Zhang et al. | 2016 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 9. Networks fit random labels perfectly. Broke the classical capacity story and forced the field to admit it didn't know why deep learning generalises. |
| 153 | [Reconciling Modern ML Practice and the Bias-Variance Trade-off](https://arxiv.org/abs/1812.11118) | Belkin et al. | 2018 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 9. Double descent. Q137. |
| 154 | [Deep Double Descent: Where Bigger Models and More Data Hurt](https://arxiv.org/abs/1912.02292) | Nakkiran et al. | 2019 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 9. The empirical version, including epoch-wise double descent. |
| 155 | [The Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635) | Frankle & Carbin | 2018 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 9. Q123. Note the later "late resetting" caveat at scale. |
| 156 | [How Does Batch Normalization Help Optimization?](https://arxiv.org/abs/1805.11604) | Santurkar et al. | 2018 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 9. **Read immediately after the BatchNorm paper (row 14).** The original explanation was wrong; this is what replaced it. A model of how to falsify a mechanism claim. Q24. |
| 157 | [Shortcut Learning in Deep Neural Networks](https://arxiv.org/abs/2004.07780) | Geirhos et al. | 2020 | Must-Read | Yet to Read | foundational | 2026-08-22 | Phase 9. The unifying account of why models pass your benchmark and fail in production. Q95. |
| 158 | [Do ImageNet Classifiers Generalize to ImageNet?](https://arxiv.org/abs/1902.10811) | Recht et al. | 2019 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 9. Rebuilt the test set from scratch; everything dropped. A lesson about benchmark overfitting that applies to your offline eval. |
| 159 | [Inherent Trade-Offs in the Fair Determination of Risk Scores](https://arxiv.org/abs/1609.05807) | Kleinberg, Mullainathan & Raghavan | 2016 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 9. The impossibility theorem. Why "make it fair" isn't well-posed. Q165. |
| 160 | [Troubling Trends in Machine Learning Scholarship](https://arxiv.org/abs/1807.03341) | Lipton & Steinhardt | 2018 | Good-To-Read | Yet to Read | foundational | 2026-08-22 | Phase 9. Explanation vs speculation, mathiness, misuse of language. The best calibration tool for reading the HF-daily firehose critically. |

---

## What I deliberately left out, and why

- **Anything with a 2608.x arXiv ID from your hf-daily rows.** Not because they're bad — several look interesting — but because none has been load-bearing for long enough to earn a place in a *foundational* list. Let them age.
- **DALL·E, Improved DDPM, HF Diffusers, the Desmos visualiser.** Superseded, incremental, or tooling.
- **CLIP, Whisper, SAM, DINOv2.** Excellent, wrong domain for you.
- **Most of the agent/self-improvement papers.** The area is moving too fast for anything to be foundational yet.
- **Graph neural networks.** A defensible omission to challenge me on — GNNs are genuinely relevant to marketplace/fraud problems at Grab. I left them out because they're not on your stated path, but if fraud or supply-graph work is anywhere in your future, add GraphSAGE (Hamilton et al. 2017) and PinSage (Ying et al. 2018) to Phase 3 as Good-To-Read.
- **Mamba / state-space models.** Deliberately Good-To-Read at most. Real, but the verdict isn't in, and for your sequence lengths transformers are fine.

---

## Suggested reading order, compressed

If you want one linear path rather than nine phases:

**Weeks 1-2 (foundations you skipped):** 69 → 72 → 70 → 73 → 77 → row 15 (Adam) → row 14 (BatchNorm) → 156 → row 13 (ResNet)

**Weeks 3-6 (your actual job — the biggest gap):** 83 → 84 → 85 → 86 → 87 → 88 → 90 → 91 → 97 → 100 → 102 → 103 → 107 → 108

**Weeks 7-10 (your career bridge):** 116 → 117 → 115 → 119 → 125 → 118 → 121 → 123 → 126 (selected chapters)

**Weeks 11-14 (the destination):** 129 (Ch. 1-6, 13) → 130 → row 26 (DQN) → 131 → row 28 (PPO) → 132 → row 33 (C51) → row 31 (DreamerV3) → row 8 (InstructGPT) → 134

**Then, as topics arise:** Phases 6-9 opportunistically. 151 tonight — it's nine pages and it will change how you argue for your own work.

---

## A note on queue hygiene

Your `Reading_Queue.md` header says the agent picks from the top and new papers append at the bottom. With `hf-daily` adding ~15/day, the foundational backlog at rows 7-37 will never surface again — it's structurally buried, and the auto-triage can't fix that because it's a queue-ordering problem, not a classification problem.

Three options, in increasing order of effort:

1. **Separate the queues.** `Reading_Queue.md` for the firehose, this file for foundations. Read from both on a fixed ratio — say four foundational papers to one daily.
2. **Add a `Phase` column** and have the agent pick the lowest unread phase first, so foundations drain before new intake.
3. **Cap the intake.** Have the agent append at most the top 3 Good-To-Read-or-better papers per day and discard the rest. Most of rows 38-68 would not survive that filter, correctly.

Whatever you pick, the failure mode to avoid is the one you're currently in: a queue where the papers you most need to read sit at position 30 behind 400 papers you'll never read.
