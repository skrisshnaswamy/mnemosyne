---
aliases:
  - HMM
  - HMMs
  - Hidden Markov Models
  - Forward Algorithm
  - Forward-Backward Algorithm
  - Viterbi Algorithm
  - Viterbi
  - Baum-Welch
  - Emission Probability
  - Regime Switching
tags:
  - estimation
  - decision-sciences
  - probability
  - bayesian
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A hidden state that hops between a **few discrete labels**, which you never see — only a noisy symptom of it. Two small tables (how it **moves**, what it **emits**) are the whole model.
> **Metaphor:** A windowless office. You can't see the weather; you can see whether your colleague walks in with an umbrella.
> **Where it bites:** Pre-deep-learning speech and POS tagging, gene finding, market-regime detection, inferring user intent from clicks. And it's the [[Kalman Filter]]'s discrete twin.

---
You work in a basement office with no windows. ☂️

You can't see the weather. What you *can* see is whether your colleague arrives carrying an umbrella. You know three things from experience:

- weather is sticky — if it rained yesterday, there's a **70%** chance it's raining today
- when it rains, she brings the umbrella **90%** of the time
- when it's dry she still brings it **20%** of the time (she's cautious)

Monday: umbrella. Tuesday: umbrella.

~={blue}How sure are you that it's raining on Tuesday — and is it more or less sure than you were on Monday?=~

---
# The windowless office

![[hmm_umbrella_office.png]]

Start on Monday knowing nothing: 50/50.

**Monday, update.** She has the umbrella. Rain explains that well (0.9), dry explains it badly (0.2):

$$P(\text{rain}) = \frac{0.5 \times 0.9}{0.5 \times 0.9 + 0.5 \times 0.2} = \frac{0.45}{0.55} = \mathbf{0.818}$$

**Overnight, predict.** Weather is sticky but not certain, so the belief *relaxes* a little:

$$P(\text{rain tomorrow}) = 0.818 \times 0.7 + 0.182 \times 0.3 = 0.627$$

**Tuesday, update.** Umbrella again:

$$P(\text{rain}) = \frac{0.627 \times 0.9}{0.627 \times 0.9 + 0.373 \times 0.2} = \mathbf{0.883}$$

More sure than Monday — two umbrellas in a row are better evidence than one. And look at the rhythm: **predict (spread), update (sharpen)**. It's exactly the [[Bayes Filter]], with the bell curves swapped for two-row tables.

![[hmm_umbrella_filtering.png]]
> [!TIP] Reading the chart
> One umbrella-free day (day 3) drops the belief from 0.88 to **0.19** — evidence moves it fast. But three dry days in a row only reach 0.06, not zero, and a single umbrella on day 10 bounces it back to 0.68. It never becomes *certain*, because the weather can always have changed overnight.

> [!NOTE] Hidden Markov Model
> A hidden state taking one of $N$ discrete values, evolving by a **transition table** $P(\text{state}_t \mid \text{state}_{t-1})$, and producing an observation at each step through an **emission table** $P(\text{obs}_t \mid \text{state}_t)$. The state obeys the [[Markov Property]]; you only ever see the emissions. ^hmm-def

> [!SUCCESS] Core idea
> ~={pink}An HMM is a [[State-Space Model]] whose state is a label instead of a vector.=~ Swap the tables for matrices and Gaussians and you have the [[Kalman Filter]]. Same two equations, same predict–update loop, different bookkeeping. ^hmm-is-discrete-kalman

---
# The three questions people ask of one

| Question | Plain English | Algorithm | The trick |
|---|---|---|---|
| **Filtering** | "what's the state *right now*?" | **Forward algorithm** | the loop above — *sum* over where you could have come from |
| **Decoding** | "what's the single most likely *sequence* of states?" | **Viterbi** | same loop, but take the ***max*** instead of the sum |
| **Learning** | "I don't know the tables — fit them" | **Baum–Welch** (EM) | guess tables → infer states → refit tables → repeat |

> [!TIP] Sum vs max — one character apart
> Forward and Viterbi are the *same recursion*. Replace $\sum$ with $\max$ and "how probable is it that I'm here?" becomes "what's the best path that ends here?". That $\max$-over-predecessors, cached per state and swept through time, is [[Dynamic Programming]] — Viterbi is one of its most famous applications. (The same sum-vs-max split shows up as [[Markov Decision Process#max vs argmax 🏙️|V vs V*]].) ^sum-vs-max

**Smoothing** is the fourth, hindsight version: *"now that I've seen the whole week, what was Tuesday?"* — forward pass plus a backward pass (**forward–backward**). Later evidence legitimately changes your view of earlier days.

---
# Where you'll meet them

- **Speech recognition**, for thirty years: hidden = phoneme, observed = a slice of audio.
- **Part-of-speech tagging**: hidden = noun/verb/adjective, observed = the word.
- **Genomics**: hidden = gene / not-gene, observed = the base pair.
- **Finance and ops**: hidden = *regime* (calm / volatile, healthy / degrading), observed = returns or metrics. Still very much alive.
- **Product analytics**: hidden = intent (browsing / comparing / ready to buy), observed = clicks.

Deep sequence models replaced HMMs wherever there's lots of data. They're still the right tool when you have **little data, a handful of meaningful states, and you want to read the tables afterwards**.

> [!WARNING] Which thing is "Markov"?
> The **hidden state** is Markov. The **observations are not** — and that's the point. Today's umbrella depends on the whole history of umbrellas, *through* the hidden weather. If you model the umbrellas directly as a Markov chain you throw that away. The hidden layer is what lets a simple model have a long memory. ^observations-are-not-markov

---
> [!SUCCESS] If you remember one thing
> Two tables — how the hidden label **moves**, and what it **emits** — and the same predict–update heartbeat as every other filter. ~={pink}Sum to track it; max to decode it.=~

---
# ⁉️
That closes out estimation: whatever the world looks like, we can now hold a sensible belief about *where we are*. None of it says a word about what to **do**. And Viterbi just smuggled in the trick we need — *cache the best answer at each step and sweep*.

→ [[Dynamic Programming]]
