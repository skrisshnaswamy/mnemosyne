---
aliases:
  - Normalized Discounted Cumulative Gain
  - nDCG
tags:
  - metrics
  - ranking
  - recsys
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A ranking metric that rewards putting **relevant items high** — and normalises so scores are comparable across lists.
> **Metaphor:** Marking a search results page. Right answer at #1 scores full marks; same answer at #10 scores a fraction.
> **Where it bites:** The default offline metric for search and recsys. **Caveat:** it needs the ideal ranking, so it's offline-only and inherits your relevance labels' biases.

---
---
aliases:
  - Normalized Discounted Cumulative Gain
---
At its core, NDCG answers a simple question: **How good is my model at ranking the most relevant items at the top of the list?** It does this by considering two key factors:

- **Relevance:** Highly relevant items are more valuable than somewhat relevant items, which are, in turn, more valuable than irrelevant ones.
    
- **Position:** Relevant items that appear higher up in a list are more useful than those that appear further down.

If you were to break it down into component metrics, you can see
1. Cummulative Gain (CG) - basically tells us **Is the good stuff there?**
	- It essentially asks **whether the recommended items are infact relevant; _without considering the order_**
	- Very similar to a grocery list. You need to know whether to get an item or not, not the order in which you should pick them up from the market.
	- You calculate the CG by simply adding up all the relevance score for each recommended items.
	- But this also means CG has a major flaw - ranking [1, 2, 3] and [2, 1, 3] would have the same CG
2. Discounted Cummulative Gain (DCG) - asks **Is the good stuff at the top?**
	- The **discounting** takes care of the position. It does this by **discounting** items that are relevant but appear lower in the list / ranked lower.
		- The idea is that users are less likely to scroll, so a highly relevant item at position 10 is less valuable than the same item at position 1.
	- So we **penalize** the relevance score for each item by the position they're at - $$
	DCG = \sum_{i = 1}^{p} (\frac{relevance_i}{log_2(i + 1)}) $$ where $p$ is the number of items in the list. The penalty applied is actually $log_2(i+1)$ where $i$ is the position. **Logarithm** is used because user attention drops off rapidly at first and then more slowly, making the penalty less severe for deeper ranks. And $i + 1$ is used to handle the first position correctly and **avoid a division-by-zero** error.
3. And finally NDCG asks **How good is the ranking compared to the best possible ranking?**
	- This is basically normalizing the metric. It is done for the very trivial reason why normalization is done everywhere - the $DCG$ although is an improvement already by adding the penalty for positions, but then longer list or the relevance score values themselves can vary DCG and same with shorter lists. Normalizing fixes that.
	- How we do this is by computing an **Ideal discounted cummulative gain -  IDCG** and divide $NDCG$ by it.
	- Which also means, $NDCG$ is an offline metrics. We **need** $IDCG$ to be able to compute $NDCG$ and the ideal ranking (perfect ranking / ground-truth) is only available after the fact, no?

---
# The three-step build, in one place

| Step | Question | Flaw it leaves |
|---|---|---|
| **CG** | Is the good stuff *there*? | Order-blind — [1,2,3] scores the same as [3,2,1] |
| **DCG** | Is the good stuff *at the top*? | Not comparable across lists of different length |
| **NDCG** | How close to the *best possible* ranking? | ✅ comparable, bounded $[0,1]$ |

$$\text{NDCG@k} = \frac{DCG@k}{IDCG@k}$$

$IDCG$ is the DCG of the **perfect** ranking — sort every item by true relevance and score that. So NDCG asks *"what fraction of the achievable ranking quality did we capture?"* — 1.0 is perfect, and the score means the same thing whether the list has 5 items or 500. 📏

> [!NOTE] The graded-relevance variant
> You'll often see the exponential form:
> $$DCG = \sum_i \frac{2^{rel_i} - 1}{\log_2(i+1)}$$
> Same discount, but the numerator now grows *exponentially* with relevance — so a "perfect" match is worth much more than two "okay" ones. Use it when relevance is graded (0–4 say) rather than binary. Always state which form you used; the numbers are not comparable. ^ndcg-exponential

---
# Your closing question — is it offline-only?

> [!SUCCESS] Yes, and you spotted exactly why
> ~={blue}You need $IDCG$, which needs the true relevance of **every** item — including the ones you never showed.=~ In production you only observe feedback on what you actually displayed, so the ideal ranking is unknowable.
>
> That's not a flaw in the metric; it's the definition of an offline metric. NDCG is what you compute against a **labelled test set**, not against live traffic. ^ndcg-is-offline

Which leads to the thing that actually bites teams:

> [!WARNING] Position bias poisons the labels
> If your relevance labels come from **click logs**, they're contaminated. Users click the top result because it *was on top*, not necessarily because it was best. Train on that and you get a model that learns to reproduce your existing ranker — a feedback loop, not an improvement. 🔁
>
> This is what **unbiased learning-to-rank** exists to fix: estimate a propensity for each position and reweight the clicks by its inverse. Without it, your offline NDCG can improve steadily while the live product doesn't move at all.

---
# Choosing between ranking metrics

| Metric | Handles graded relevance? | Position-aware? | Use when |
|---|---|---|---|
| **Precision@k** | ❌ binary | ❌ | Simple, interpretable |
| **MRR** | ❌ binary | ✅ (first hit only) | There's **one** right answer — QA, known-item search |
| **MAP** | ❌ binary | ✅ | Several relevant items, binary labels |
| **NDCG@k** | ✅ | ✅ | Graded relevance — the general-purpose default |

> [!WARNING] NDCG is not differentiable
> It depends on sort order, so it has no useful gradient and you cannot train on it directly. Models optimise a surrogate — pairwise (RankNet), listwise (LambdaRank/LambdaMART, which cleverly weights pairs by *how much swapping them would change NDCG*) — and NDCG is used for **evaluation** only. Same gap as accuracy vs [[Cross Entropy]]: the thing you care about and the thing you can optimise are rarely the same. ^ndcg-not-differentiable

---
# ⁉️
NDCG is how you *score* a ranker. How rankers themselves evolved — CF, matrix factorization, two-tower, generative — is [[Recommender Systems - Evolution]].
