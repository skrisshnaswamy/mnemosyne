---
tags: [reading-plan, papers, recsys, ranking]
taxonomy_revision: 1
built: 2026-08-22
supersedes: "Phase 3 of Foundational_Reading_Plan.md"
---

# RecSys & Ranking Canon — Phase 3, Expanded

Numbered from **161**, continuing after `Foundational_Reading_Plan.md`. Rows 99-113 of that file stay valid; this expands them into the full literature.

**On your list:** YouTube DNN (#102) and Wide & Deep (#101) I had. **DIN was a real gap** — the behaviour-sequence-attention line is the one closest to your merchant work and I skipped it. MMoE I had (#105); MoE's own origin I didn't, and the lineage matters.

Priorities below are assigned for *your* trajectory — merchant representation learning, sequence modelling, marketplace ranking — not for a generic recsys engineer. Several papers that are canonical in the field are Good-To-Read here because they solve problems you don't have.

---

## A. Pre-deep foundations

Skip these and you'll keep rediscovering their conclusions. The implicit-feedback paper in particular defines the data regime you actually live in — you have clicks and orders, never ratings.

| #   | Paper | Authors | Year | Priority | Status | Source | Added | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 161 | [Amazon.com Recommendations: Item-to-Item Collaborative Filtering](https://www.cs.umd.edu/~samir/498/Amazon-Recommendations.pdf) (*IEEE Internet Computing*) | Linden, Smith & York | 2003 | Good-To-Read | Yet to Read | recsys | 2026-08-22 | Five pages. Ran e-commerce for a decade. The precompute-item-similarity-offline insight is still the shape of every retrieval system you build. |
| 162 | [Collaborative Filtering for Implicit Feedback Datasets](http://yifanhu.net/PUB/cf.pdf) (ICDM) | Hu, Koren & Volinsky | 2008 | Must-Read | Yet to Read | recsys | 2026-08-22 | **The implicit-feedback formulation.** Confidence weighting, treating absence as weak negative rather than true negative. This is your data. Foundational for everything in section D. |
| 163 | [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf) (ICDM) | Rendle | 2010 | Must-Read | Yet to Read | recsys | 2026-08-22 | **The bridge from matrix factorisation to feature-rich models.** Models all pairwise feature interactions in linear time via shared latent vectors. Everything in section B is a descendant. |
| 164 | [Field-aware Factorization Machines for CTR Prediction](https://www.csie.ntu.edu.tw/~cjlin/papers/ffm.pdf) (RecSys) | Juan et al. | 2016 | Low-Priority | Yet to Read | recsys | 2026-08-22 | Won several Kaggle CTR competitions. Read only if you're doing classical CTR work. |
| 165 | [Ad Click Prediction: a View from the Trenches](https://research.google/pubs/pub41159/) (KDD) | McMahan et al. | 2013 | Good-To-Read | Yet to Read | recsys | 2026-08-22 | FTRL-Proximal, but the value is §4-6: memory-saving feature management, calibration, confidence estimation, and a section on what *didn't* work. Rare and honest. |
| 166 | [Practical Lessons from Predicting Clicks on Ads at Facebook](https://research.facebook.com/publications/practical-lessons-from-predicting-clicks-on-ads-at-facebook/) (ADKDD) | He et al. | 2014 | Must-Read | Yet to Read | recsys | 2026-08-22 | **GBDT features into a linear model, negative downsampling, and the calibration correction you must apply afterward.** The downsampling-then-recalibrate recipe is used everywhere and rarely cited properly. Q93, Q79. |

---

## B. Feature interaction & CTR architectures

The "how do we make a DNN learn feature crosses" line. Honest assessment: this is the most crowded and least differentiated area in recsys — a dozen papers each claiming a marginally better interaction module. Read Wide & Deep (#101), then DCN-V2, then stop unless you're specifically working on CTR architecture.

| #   | Paper | Authors | Year | Priority | Status | Source | Added | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 167 | [Deep & Cross Network for Ad Click Predictions](https://arxiv.org/abs/1708.05123) | Wang et al. | 2017 | Good-To-Read | Yet to Read | recsys | 2026-08-22 | Explicit bounded-degree feature crossing as a network layer. |
| 168 | [DCN V2: Improved Deep & Cross Network](https://arxiv.org/abs/2008.13535) | Wang et al. | 2020 | Good-To-Read | Yet to Read | recsys | 2026-08-22 | **Read this instead of #167 if choosing one.** Low-rank cross layers, production lessons from Google, and an honest comparison against the whole family. |
| 169 | [DeepFM: A Factorization-Machine based Neural Network](https://arxiv.org/abs/1703.04247) | Guo et al. | 2017 | Low-Priority | Yet to Read | recsys | 2026-08-22 | FM + DNN sharing embeddings. Widely deployed, conceptually thin once you know #163. |
| 170 | [xDeepFM: Combining Explicit and Implicit Feature Interactions](https://arxiv.org/abs/1803.05170) | Lian et al. | 2018 | Low-Priority | Yet to Read | recsys | 2026-08-22 | Compressed interaction network. Diminishing returns territory. |
| 171 | [AutoInt: Automatic Feature Interaction Learning via Self-Attention](https://arxiv.org/abs/1810.11921) | Song et al. | 2018 | Good-To-Read | Yet to Read | recsys | 2026-08-22 | Self-attention over *feature fields* rather than sequence positions. Relevant to your tabular transformer work — it's the recsys sibling of FT-Transformer. |

---

## C. Sequential & behaviour modelling — **your core line**

This is the section that matters most for you, and it's the one you correctly identified as missing. Your merchant model is a masked sequence model over event streams; **BERT4Rec is literally that, published in 2019.** Reading it will tell you which of your design decisions the field converged on and which you did differently.

Read in this order — it's a clean intellectual arc from RNN → attention-over-history → transformer → masked transformer → long sequences → entity embeddings.

| #   | Paper | Authors | Year | Priority | Status | Source | Added | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 172 | [Session-based Recommendations with RNNs (GRU4Rec)](https://arxiv.org/abs/1511.06939) | Hidasi et al. | 2015 | Good-To-Read | Yet to Read | recsys | 2026-08-22 | The paper that started sequential recsys. Session-parallel minibatching and ranking losses for sequences. Historical but short. |
| 173 | [Deep Interest Network for CTR Prediction (DIN)](https://arxiv.org/abs/1706.06978) | Zhou et al. | 2018 | Must-Read | Yet to Read | recsys | 2026-08-22 | **You named it, and you were right.** Attention over user behaviour history, weighted by relevance to the *candidate item* — so the user representation is query-dependent rather than fixed. That's a genuinely important idea and it's in direct tension with precomputed entity embeddings. Worth thinking hard about for your merchant vectors. |
| 174 | [Deep Interest Evolution Network (DIEN)](https://arxiv.org/abs/1809.03672) | Zhou et al. | 2019 | Good-To-Read | Yet to Read | recsys | 2026-08-22 | Adds GRU-based interest evolution and an auxiliary loss on the sequence. The auxiliary-loss trick is the transferable part. |
| 175 | [Self-Attentive Sequential Recommendation (SASRec)](https://arxiv.org/abs/1808.09781) | Kang & McAuley | 2018 | Must-Read | Yet to Read | recsys | 2026-08-22 | **Causal transformer for recommendation.** The standard baseline for every sequential recsys paper since. Short and clean. |
| 176 | [BERT4Rec: Sequential Recommendation with Bidirectional Transformer](https://arxiv.org/abs/1904.06690) | Sun et al. | 2019 | Must-Read | Yet to Read | recsys | 2026-08-22 | **Read this first in this section.** Masked item prediction over behaviour sequences — your exact approach, five years earlier. Compare their masking rate and objective design against yours (Q89). Note the later replication controversy: several groups found SASRec matches it under fair tuning, which is itself worth knowing. |
| 177 | [Behavior Sequence Transformer for E-commerce (BST)](https://arxiv.org/abs/1905.06874) | Chen et al. | 2019 | Good-To-Read | Yet to Read | recsys | 2026-08-22 | Alibaba's production transformer over behaviour, with positional encoding based on **time gaps** rather than index. Directly relevant to Q47. |
| 178 | [Search-based User Interest Modeling (SIM)](https://arxiv.org/abs/2006.05639) | Pi et al. | 2020 | Good-To-Read | Yet to Read | recsys | 2026-08-22 | Lifelong sequences (thousands of events) via a two-stage retrieve-then-attend design. The answer to "what if the merchant has 50,000 orders." |
| 179 | [PinnerFormer: Sequence Modeling for User Representation at Pinterest](https://arxiv.org/abs/2205.04507) | Pancha et al. | 2022 | Must-Read | Yet to Read | recsys | 2026-08-22 | **The closest published analog to your job.** How to train a *single precomputed user embedding* from a behaviour sequence that serves many downstream tasks — including the batch-vs-realtime trade-off and their dense all-action loss. Read alongside #185. |
| 180 | [Multi-Interest Network with Dynamic Routing (MIND)](https://arxiv.org/abs/1904.08030) | Li et al. | 2019 | Good-To-Read | Yet to Read | recsys | 2026-08-22 | One vector per entity is a strong assumption — a merchant may serve several distinct demand patterns. This represents an entity with *several* vectors. Worth reading for the framing even if you don't adopt it. |

---

## D. Retrieval & candidate generation

Complements rows 103, 109-113 in the main plan. The Airbnb paper is the standout for you — marketplace entity embeddings with cold start, written by people solving your exact problem.

| #   | Paper | Authors | Year | Priority | Status | Source | Added | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 181 | [Real-time Personalization using Embeddings for Search Ranking at Airbnb](https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb) (KDD, best paper) | Grbovic & Cheng | 2018 | Must-Read | Yet to Read | recsys | 2026-08-22 | **The single most relevant paper on this entire list for your merchant work.** Listing embeddings from session sequences in a two-sided marketplace, cold-start handling by geographic/type averaging, congregated search adaptations, and an honest account of what failed. If you read one thing here, this. |
| 182 | [Embedding-based Retrieval in Facebook Search](https://arxiv.org/abs/2006.11632) | Huang et al. | 2020 | Must-Read | Yet to Read | recsys | 2026-08-22 | End-to-end production two-tower: hard negative mining strategies, ANN parameter tuning, **and how embedding retrieval interacts with boolean filtering** — the failure mode in Q121 #9. Unusually practical. |
| 183 | [Learning Tree-based Deep Model for Recommender Systems (TDM)](https://arxiv.org/abs/1801.02294) | Zhu et al. | 2018 | Low-Priority | Yet to Read | recsys | 2026-08-22 | Alternative to ANN: learn a tree index jointly with the model, enabling arbitrary (non-dot-product) scoring at retrieval. Elegant, rarely adopted outside Alibaba. |
| 184 | [Graph Convolutional Neural Networks for Web-Scale Recommender Systems (PinSage)](https://arxiv.org/abs/1806.01973) | Ying et al. | 2018 | Good-To-Read | Yet to Read | recsys | 2026-08-22 | The GNN entry point I flagged earlier. Random-walk sampling and producer-consumer minibatching make it a systems paper as much as a modelling one. |

---

## E. Multi-task, multi-objective & bias correction

Complements rows 105-108. ESMM and the YouTube multi-task paper are the two that address problems I know you have — delayed sparse conversions, and position bias in a production ranker.

| #   | Paper | Authors | Year | Priority | Status | Source | Added | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 185 | [Recommending What Video to Watch Next: A Multitask Ranking System](https://daiwk.github.io/assets/youtube-multitask.pdf) (RecSys) | Zhao et al. | 2019 | Must-Read | Yet to Read | recsys | 2026-08-22 | **MMoE plus a shallow position-bias tower, in production at YouTube.** The two-tower debiasing trick from Q77 is described here concretely. Also candid about conflicting objectives and engagement-vs-satisfaction. |
| 186 | [Entire Space Multi-Task Model (ESMM)](https://arxiv.org/abs/1804.07931) | Ma et al. | 2018 | Must-Read | Yet to Read | recsys | 2026-08-22 | **Sample selection bias in conversion modelling.** CVR is trained on clicked items but served on all impressions. Solves it by modelling CTR and CTCVR over the entire space. If you predict order-given-impression anywhere, this is your paper. |
| 187 | [Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) | Shazeer et al. | 2017 | Good-To-Read | Yet to Read | recsys | 2026-08-22 | The MoE lineage you asked about. Conditional computation, load-balancing loss, the gating network. MMoE (#105) is this idea applied per-task. |
| 188 | [Recommendations as Treatments: Debiasing Learning and Evaluation](https://arxiv.org/abs/1602.05352) | Schnabel et al. | 2016 | Must-Read | Yet to Read | recsys | 2026-08-22 | **IPS applied to recommendation, framed explicitly as causal inference.** This is the paper that connects your experimentation background to your ranking work in one move. Read with #108. |
| 189 | [Calibrated Recommendations](https://dl.acm.org/doi/10.1145/3240323.3240372) (RecSys) | Steck | 2018 | Good-To-Read | Yet to Read | recsys | 2026-08-22 | Not probability calibration — *distributional* calibration: if a user is 30% dessert, the slate should be ~30% dessert. A clean formalisation of the diversity problem. |
| 190 | [The Use of MMR for Reordering Documents (MMR)](https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_LTMIR_1998.pdf) (SIGIR) | Carbonell & Goldstein | 1998 | Low-Priority | Yet to Read | recsys | 2026-08-22 | Two pages, still the default diversity re-ranker. Read it once so you stop reinventing it. |

---

## F. Evaluation, replication & skepticism

**Read this section early — before section B or C.** It will change how you read everything else, and it's the part of the recsys literature most people skip.

The field went through a genuine replication crisis between 2019 and 2021. A large fraction of published neural recsys improvements turned out to be artifacts of weakly-tuned baselines. Knowing this is what lets you evaluate a new architecture claim properly, and it's an unusually strong thing to bring up in an interview.

| #   | Paper | Authors | Year | Priority | Status | Source | Added | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 191 | [Are We Really Making Much Progress? A Worrying Analysis](https://arxiv.org/abs/1907.06902) (RecSys, best paper) | Dacrema, Cremonesi & Jannach | 2019 | Must-Read | Yet to Read | recsys | 2026-08-22 | **Read this before anything else in this document.** 18 neural methods; 7 reproducible; 6 of those beaten by simple, properly-tuned baselines. |
| 192 | [On the Difficulty of Evaluating Baselines](https://arxiv.org/abs/1905.01395) | Rendle, Zhang & Koren | 2019 | Must-Read | Yet to Read | recsys | 2026-08-22 | Shows a well-tuned baseline on Movielens 10M beating years of reported improvements. Nine pages, quietly devastating. |
| 193 | [Neural Collaborative Filtering vs. Matrix Factorization Revisited](https://arxiv.org/abs/2005.09683) | Rendle et al. | 2020 | Good-To-Read | Yet to Read | recsys | 2026-08-22 | A properly-tuned dot product beats the learned similarity function that replaced it. A useful prior about learned-vs-fixed similarity in your own two-tower work. |

---

## G. Production systems & embedding infrastructure

Your home territory. These are the papers that treat embedding tables as the engineering problem they actually are.

| #   | Paper | Authors | Year | Priority | Status | Source | Added | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 194 | [Compositional Embeddings Using Complementary Partitions (QR trick)](https://arxiv.org/abs/1909.02107) | Shi et al. | 2020 | Good-To-Read | Yet to Read | recsys | 2026-08-22 | Quotient-remainder hashing — represent millions of IDs with far fewer rows while keeping every ID's representation unique. The principled version of Q54. |
| 195 | [Mixed Dimension Embeddings](https://arxiv.org/abs/1909.11810) | Ginart et al. | 2019 | Good-To-Read | Yet to Read | recsys | 2026-08-22 | Frequency-proportional embedding dimensions. Directly the Q53 answer, with a theoretical justification. |
| 196 | [Monolith: Real Time Recommendation System With Collisionless Embedding Table](https://arxiv.org/abs/2209.07663) | Liu et al. (ByteDance) | 2022 | Good-To-Read | Yet to Read | recsys | 2026-08-22 | Cuckoo-hashed collisionless embeddings, online training, and an explicit treatment of the fault-tolerance-vs-freshness trade. Very much your kind of paper. |

---

## H. The current frontier

Where the field is heading. Not foundational yet — but #198 is the most interesting recsys result of the last few years and the direction most likely to reshape what your job looks like.

| #   | Paper | Authors | Year | Priority | Status | Source | Added | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 197 | [Recommender Systems with Generative Retrieval (TIGER)](https://arxiv.org/abs/2305.05065) | Rajput et al. | 2023 | Good-To-Read | Yet to Read | recsys | 2026-08-22 | **Semantic IDs**: quantise item embeddings via RQ-VAE into discrete code tuples, then *generate* the item ID autoregressively instead of retrieving by dot product. Sidesteps the ANN index entirely and gives cold-start items a meaningful ID by construction. |
| 198 | [Actions Speak Louder than Words: Generative Recommenders (HSTU)](https://arxiv.org/abs/2402.17152) | Zhai et al. (Meta) | 2024 | Must-Read | Yet to Read | recsys | 2026-08-22 | **Reformulates ranking and retrieval as sequential transduction over a unified user action sequence, and demonstrates scaling laws for recommendation.** Deployed at Meta scale. The strongest evidence yet that recsys is heading where language modelling went — which is directly the bet your merchant representation work is making. |

---

## If you read only eight

Ordered:

1. **#191 Are We Really Making Much Progress?** — recalibrates how you read everything else
2. **#181 Airbnb Embeddings** — closest published analog to your job
3. **#176 BERT4Rec** — your approach, already published
4. **#179 PinnerFormer** — precomputed entity embeddings serving many consumers
5. **#173 DIN** — the one you named; query-dependent user representation
6. **#166 Facebook Practical Lessons** — downsampling and calibration, which you're probably doing
7. **#186 ESMM** — sample selection bias in conversion modelling
8. **#198 HSTU** — where this is all going

That's roughly two weeks of evenings and it would take you from "no recsys reading" to genuinely well-read in the area.

---

## Where your named papers land

| You said | Where it sits | Priority |
|---|---|---|
| YouTube DNN | #102, main plan | Must-Read — the practical cascade blueprint |
| Wide & Deep | #101, main plan | Good-To-Read — the memorisation/generalisation framing survives; the architecture doesn't |
| MoE | #187 (Shazeer 2017) | Good-To-Read — read for the MMoE lineage |
| MMoE | #105, main plan | Good-To-Read — but read #185 instead if picking one, it's MMoE in production with position debiasing |
| DIN | #173 | **Must-Read** — you were right, and I'd missed it |

The instinct behind your list was sound. The gap in it is the same gap as in your queue: it's a **ranking/CTR** list, and nothing in it covers *entity representation* — which is what you actually build. That's sections C and D, and specifically #181, #179 and #176.
