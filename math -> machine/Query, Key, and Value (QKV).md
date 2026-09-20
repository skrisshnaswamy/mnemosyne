---
aliases:
  - QKV
  - Q K V
tags:
  - fundamentals
  - transformers
  - attention
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Every token produces **three** views of itself — what it's *looking for* (Q), what it *offers* (K), and what it actually *hands over* (V).
> **Metaphor:** A library search. Your question is the Query, book titles are the Keys, the book's contents are the Value.
> **Where it bites:** Self- vs cross-attention, why we divide by $\sqrt{d_k}$, multi-head, and the KV cache that dominates inference memory.

---
The three roles a piece of data plays in an [[Attention]] mechanism.
Think of it like a library search:
- your **Query** is your question
- you match it against the **Keys** (book titles) to find the right book
- and then you retrieve the **Value** (the book's contents)

---
# Why three, and not one?

This is the question worth sitting with, because "three matrices" looks arbitrary until you see what each one buys you.

Consider the sentence: *"The animal didn't cross the street because **it** was too tired."*

For the model to handle "it" correctly, that token needs to reach back and pull in "animal". So "it" has to **ask a question**: ~={blue}"I'm a pronoun, I need a nearby animate noun."=~

Meanwhile "animal" has to **advertise**: "I'm an animate noun, subject position."

And once they match, what actually gets passed to "it" isn't the advertisement — it's the **content**: the semantic substance of "animal".

> [!SUCCESS] Core idea
> Those are three genuinely different jobs, so they get three different [[Linear Projection|projections]]:
> - **Query** = what I'm looking for 🔍
> - **Key** = what I'm advertising 🏷️
> - **Value** = what I actually hand over 📦 ^three-roles

If you used one vector for all three, "what I'm looking for" and "what I'm offering" would be forced to be the same thing. A pronoun would only be able to attract *other pronouns*. The whole mechanism would collapse into similarity matching rather than the flexible query-answer routing you actually want.

---
# The mechanics

Every token's embedding $x$ gets projected three ways — three separate learned matrices:
$$Q = XW_Q \qquad K = XW_K \qquad V = XW_V$$

Then:
$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Walk it left to right, it's four steps:

1. **$QK^T$** — every query dotted with every key. A dot product is a similarity score, so this gives you a grid: *how much does token $i$ care about token $j$?* For $n$ tokens that's an $n \times n$ matrix. ← remember this, it's the whole problem [[Flash Attention]] exists to solve
2. **$\div \sqrt{d_k}$** — the scaling (below)
3. **softmax** — turn each row of raw scores into weights that sum to 1. Now each token has a proper distribution over "who I'm listening to"
4. **$\times V$** — take the weighted average of everyone's *values* using those weights

The output for each token is a **weighted blend of every other token's content**, where the blend is decided by how well queries matched keys.

> [!NOTE] Why divide by $\sqrt{d_k}$?
> Dot products of two random $d_k$-dimensional vectors have a variance that grows with $d_k$. So for a big head dimension you get some scores that are *huge* — and softmax of huge numbers is brutally peaky. One token gets weight $\approx 1.0$ and everyone else $\approx 0$.
>
> That's bad for two reasons: the model can only look at one thing, and softmax's gradient in a saturated region is near zero, so it **can't learn its way out**. Dividing by $\sqrt{d_k}$ rescales the variance back to ~1 and keeps the distribution soft enough to train. ^why-sqrt-dk

---
# Self-attention vs cross-attention

The formula never changes. **Only where Q, K and V come from changes** — and that one distinction explains most transformer architecture diagrams.

| | Q comes from | K, V come from | What it means |
|---|---|---|---|
| **Self-attention** | the sequence itself | the *same* sequence | "Let every token look at every other token in this sentence" |
| **Cross-attention** | the decoder | the **encoder** | "Let the output I'm generating look back at the input" |

Cross-attention is the piece that made [[Seq2Seq models|seq2seq]] work — it's how a translation decoder looks back at the source sentence instead of squeezing everything through one fixed bottleneck vector.

> [!TIP] Reading any transformer diagram
> Find the three arrows going into each attention block. If all three come from the same place, it's self-attention. If Q comes from one place and K/V from another, it's cross-attention. That's genuinely all there is to it. 📐

---
# Multi-head — why not just one big attention?

One attention operation produces **one** set of weights per token. But "it" in that sentence needs to track several relationships at once: what it refers to, its grammatical role, its position in the clause.

So instead of one attention with dimension 512, you run **8 heads** of dimension 64 each, in parallel, then concatenate. Each head gets its own $W_Q, W_K, W_V$ and is free to specialise — in practice some heads learn syntax, some track positional patterns, some do coreference.

Same total compute, far more expressive. ~={pink}It's the difference between asking one broad question and asking eight sharp ones.=~

---
# The KV cache — where this becomes an infra problem

When generating text one token at a time, every new token attends to all previous tokens. Recomputing every previous K and V each step would be enormously wasteful — so you **cache** them.

That cache is why serving LLMs is memory-hungry. It grows linearly with sequence length *and* batch size, and for long contexts it can dwarf the model weights themselves. Notice it's the **K** and **V** you cache — not Q, because each new token's query is only needed once, right now.

This is why so much inference work (GQA, MQA, paged attention) is fundamentally about ~={blue}making the KV cache smaller=~. See [[GPU processing]] and [[ML Infrastructure]].

---
### Context from Kidan
The **Vanilla Encoder** from the tranformer paper created a unique Q, K, and V using [[Linear Projection]] layers, which is standard practice. But the CM2 Encoder of Kidan uses the same input as Q, K and V.

> [!WARNING] Why that hurt
> Reusing one tensor for all three collapses the three roles described above into one. The token can no longer ask for something different from what it advertises — the attention matrix is forced toward symmetry, and effectively becomes plain similarity matching. It also means one shared representation is carrying three conflicting gradient signals, which is likely part of why it trained poorly and couldn't exploit [[Mixed Precision training|BFLOAT16]]. ^kidan-shared-qkv

---
# ⁉️
This lets every token see every other token — including ones that come **after** it. Fine when reading a whole sentence. Fatal when you're supposed to be *predicting* the next word, since the answer is sitting right there.

→ [[Causal Attention]]
