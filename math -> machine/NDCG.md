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