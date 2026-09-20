---
aliases:
  - KL
  - Kullback-Leibler Divergence
  - Relative Entropy
tags:
  - fundamentals
  - information-theory
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** KL divergence is the **extra surprise you pay** for using the wrong probability distribution to describe reality.
> **Metaphor:** Navigating a city with a slightly wrong map — KL is the extra distance you walk because of the map's errors.
> **Where it bites:** VAEs, RLHF/PPO ("don't drift too far from the base model"), [[Distillation]], variational inference. It's the standard leash for "stay close to *that* distribution."

---
# Picking up exactly where cross-entropy left off

In [[Cross Entropy]] we had the weather example:
- **Reality** ($P$): sunny ☀️ **90%**, rainy 🌧️ **10%**
- **Our model** ($Q$): sunny ☀️ **70%**, rainy 🌧️ **30%**

And we said three things:
1. **Entropy** $H(P)$ = the average surprise if you use reality's own probabilities. This is the *floor*. The irreducible minimum. Nobody can do better than this.
2. **Cross-Entropy** $H(P, Q)$ = the average surprise when you use *your model's* probabilities to describe *reality's* outcomes. Always ≥ the floor.
3. ~={blue}"The extra amount of surprise is our loss."=~

That third line — **that extra bit** — is the KL divergence. That's the whole thing.

$$D_{KL}(P \parallel Q) = \underbrace{H(P, Q)}_{\text{surprise with your model}} - \underbrace{H(P)}_{\text{unavoidable floor}}$$

> [!SUCCESS] Core idea
> **Cross-entropy = the floor + the penalty. KL divergence = just the penalty.**
> It's cross-entropy with the unavoidable part subtracted out, so what's left is purely *your model's fault*. ^kl-is-ce-minus-entropy

---
# The map metaphor 🗺️

Imagine you're walking across a city you don't know.

**Reality** ($P$) is the actual street layout — where the roads truly are.
**Your map** ($Q$) is what you *think* the layout is.

Even with a *perfect* map you still have to walk the distance. That unavoidable walking is the **entropy** $H(P)$ — it's a property of the city, not of you.

But your map is a bit wrong. So you take a turn that dead-ends, backtrack, take a detour. **The extra distance you walk purely because your map was wrong** — that's the KL divergence.

And notice the consequences that fall straight out of this picture:
- If your map is **perfect**, you walk zero extra. $D_{KL} = 0$. ✅
- You can never walk *negative* extra distance. $D_{KL} \geq 0$, always. Your wrong map cannot make the city smaller.
- The more wrong the map, the more you walk. It's a measure of **wrongness**.

---
# So it's a distance between distributions?

Almost — and this is the part worth getting right, because it's the thing people get pulled up on.

It is **not** a distance. It's called a *divergence* precisely because it fails the rules a distance has to obey:

> [!WARNING] KL is not symmetric
> $$D_{KL}(P \parallel Q) \neq D_{KL}(Q \parallel P)$$
> "How wrong is my map about the city" is a **different number** from "how wrong is the city about my map." ^kl-asymmetric

That sounds like pedantry. It absolutely is not — the asymmetry is the single most practically important property of KL, and choosing the wrong direction will visibly change what your model learns.

## Forward KL vs Reverse KL

Look at the formula and where the weighting sits:

$$D_{KL}(P \parallel Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$$

The sum is weighted by $P(x)$ — **the first distribution decides which outcomes get to matter.**

### Forward KL — $D_{KL}(P \parallel Q)$, "**mode-covering**"

Here $P$ is reality and it does the weighting. So for any outcome where reality says $P(x)$ is meaningfully large, if your model says $Q(x) \approx 0$, then $\log \frac{P}{Q} \to \infty$ and you get an **enormous** penalty.

Translation: ~={blue}you are severely punished for assigning zero probability to something that actually happens.=~

So the model plays it safe and **smears itself over everything** reality does. If reality has two separate peaks, $Q$ stretches to blanket both — including the empty valley in between, where nothing actually happens.

> [!TIP] Forward KL = the paranoid model
> "I must not be caught off guard by *anything*. I'll put a little probability everywhere." Covers all the modes, but blurry. 🫥

This is what you're minimising when you train with ordinary [[Cross Entropy]] on a dataset — maximum likelihood *is* forward KL.

### Reverse KL — $D_{KL}(Q \parallel P)$, "**mode-seeking**"

Now the *model* does the weighting. The penalty only applies where $Q(x)$ is large. Wherever your model says "this basically never happens," it pays almost nothing — it doesn't matter what reality thinks there.

So the safe strategy flips: **pick one peak, sit on it confidently, ignore the rest.** No penalty for the ignored regions, because you assigned them ~zero mass.

> [!TIP] Reverse KL = the confident specialist
> "I'll bet everything on one answer I'm sure about, and pretend the other options don't exist." Sharp, but it collapses. 🎯

> [!WARNING] This *is* [[Mode Collapse]]
> The failure where a generative model only ever produces cats when it was trained on cats, dogs and hamsters — that's reverse-KL behaviour. It found one mode, it's not being punished for missing the others, so it stays there. ^reverse-kl-mode-collapse

| | **Forward** $D_{KL}(P \parallel Q)$ | **Reverse** $D_{KL}(Q \parallel P)$ |
|---|---|---|
| **Weighted by** | Reality $P$ | Model $Q$ |
| **Punishes** | Missing something real | Inventing something fake |
| **Behaviour** | Mode-**covering** (blurry, hedges) | Mode-**seeking** (sharp, collapses) |
| **Analogy** | Paranoid generalist | Confident specialist |
| **Shows up in** | Max-likelihood training, cross-entropy loss | VAE objective, variational inference, RLHF |

---
# Where you'll actually meet it

### 1. As a leash — RLHF and PPO 🦮
This is the one that comes up most in conversation. When you fine-tune a language model with reinforcement learning, the reward model is imperfect, and the policy will happily find degenerate nonsense that scores highly ("reward hacking").

The fix: add a KL penalty against the **original, pre-RL model**.
$$\text{objective} = \mathbb{E}[\text{reward}] - \beta \cdot D_{KL}(\pi_{\text{new}} \parallel \pi_{\text{original}})$$

Read it in plain English: ~={pink}"Get a high reward — but don't become a *different model* while doing it."=~ The $\beta$ is how short the leash is.

### 2. As a teaching signal — [[Distillation]] 👩‍🏫
The student model isn't trained to match the teacher's *answer*, it's trained to match the teacher's whole **distribution** over answers. The loss that measures "how close are these two distributions" is KL. That's why distillation transfers so much more than plain labels — the teacher's uncertainty (its "I think cat, but honestly it could be a fox") is *in* the distribution, and KL makes the student copy it.

### 3. As a regulariser — VAEs
The VAE loss has two terms: "reconstruct the input well" + "keep your latent distribution close to a standard Gaussian." That second term is a KL. It's doing exactly the job described in [[Regularization#The deeper way to see it: a regularizer is a prior|regularization as a prior]] — pulling the model toward a simple assumed shape.

### 4. As the definition of "close enough" — variational inference
When the true posterior is impossible to compute, you pick a simple family of distributions and find the member of that family with the smallest KL to the truth. It's the formal version of "I can't have the real answer, give me the nearest tractable one."

---
# The gotchas

> [!WARNING] Three things that will bite you
> 1. **Direction matters and the notation is easy to misread.** $D_{KL}(P \parallel Q)$ — the *first* argument is the one doing the weighting. Say out loud which one is reality before you write the code.
> 2. **KL blows up to infinity** the moment $Q(x) = 0$ where $P(x) > 0$. In practice this means adding an $\epsilon$, or clamping, or using a distribution family with full support.
> 3. **KL ignores geometry.** It has no idea whether two outcomes are "near" each other. Predicting 9 when the answer is 10 is penalised exactly as hard as predicting 900. When distance between outcomes matters, you want something like **Wasserstein** distance instead.

If you need symmetry, there's **Jensen–Shannon divergence** — essentially the average of both directions against a midpoint distribution. Symmetric, bounded, and never infinite. It's the divergence the original GAN objective turns out to be minimising.

---
> [!SUCCESS] If you remember one thing
> KL is **the price of being wrong about the world**, and *which direction you write it in* decides whether your model becomes a blurry hedger or a confident specialist.

---
# ⁉️
KL measures the gap between what we believe and what's true. But it quietly assumes we *have* a distribution to begin with — that the model knows how unsure it is. Where does that come from, and what are the different flavours of "not sure"?

That's [[Uncertainty]].
