---
aliases:
  - Regularisation
  - Weight Decay
tags:
  - fundamentals
  - training
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Regularization is anything you deliberately do to make it *harder* for a model to be complicated — so it's forced to learn the pattern instead of memorising the answers.
> **Metaphor:** The student who memorises past exam papers vs. the one who actually learns the subject.
> **Where it bites:** Train loss keeps dropping, validation loss starts climbing. That gap *is* the thing regularization closes.

---
# The problem it solves

Imagine two students preparing for an exam. 📚

**Student A** memorises every single question from the last 10 years of past papers. Word for word. Ask them a question from those papers, they'll nail it perfectly — 100%.
**Student B** actually learns the subject. On the past papers they get maybe 85%, because they occasionally slip on a detail.

Now exam day comes and the questions are *new*.
Student A collapses. 💀 They never learned the subject, they learned the paper.
Student B does fine.

Student A has **overfit**. ^overfitting

> [!NOTE] Overfitting
> When a model learns the *noise and the accidents* of the training data instead of the underlying pattern. You can spot it because training error keeps going down while validation error starts going back up. ^overfit-def

And here's the cruel part — a *bigger, more powerful* model is **better** at memorising. Give a neural network enough parameters and it can memorise pure random noise perfectly. So capacity alone isn't the answer. Something has to push back.

That push-back is **regularization**.

---
# The core idea

You see, normally we ask the optimizer to do one thing:
$$\text{minimize} \quad \text{Loss}(\text{predictions}, \text{truth})$$

Regularization changes the ask to *two* things at once:
$$\text{minimize} \quad \underbrace{\text{Loss}(\text{predictions}, \text{truth})}_{\text{"be accurate"}} + \lambda \cdot \underbrace{\text{Complexity}(\text{model})}_{\text{"but stay simple"}}$$

That's it. That's the whole concept. ~={blue}You are adding a second term to the objective that the model has to pay a price for being complicated.=~

And $\lambda$ (lambda) is the **regularization strength** — the dial that says *how much* you care about simplicity vs. accuracy.
- $\lambda = 0$ → "I don't care, memorise away." (back to plain overfitting)
- $\lambda$ huge → "Be so simple you say nothing useful." (now you're **underfitting**)
- Somewhere in the middle → the sweet spot you tune for.

> [!SUCCESS] Core idea
> Regularization is not a technique. It's a *category*. Anything that biases the model toward simpler explanations counts — even things that don't look like maths, like stopping training early or feeding it distorted images.

---
# But what counts as "complicated"?

This is the interesting question. Because "complexity" has to become a number the optimizer can differentiate.

The answer people landed on: **big weights = complicated.**

Why? Think about what a weight *does*. A weight is how strongly the model reacts to a feature. A huge weight means "when this one feature twitches slightly, my output swings wildly." That's a jagged, spiky function — exactly the shape you get when a model is contorting itself to pass through every single training point.

Small weights mean a smooth, gentle function. And smooth functions generalise.

So: **penalise the size of the weights.**

## L2 penalty (Ridge / Weight Decay)

Add up the *squares* of all the weights:
$$\text{Loss} + \lambda \sum_i w_i^2$$

Because it squares, the penalty grows fast for large weights and is nearly free for small ones. So the optimizer's incentive is: ~={blue}shrink the big weights hard, leave the small ones roughly alone.=~

Every weight gets nudged toward zero on every single step — which is why in deep learning it's usually called **weight decay**. The weights literally decay a little bit each update.

> [!TIP] What L2 does to your weights
> It **shrinks** everything toward zero but almost never *to* zero. You end up with lots of small non-zero weights — the model spreads its bets across many features rather than betting big on a few. ^l2-shrinks

## L1 penalty (Lasso)

Add up the *absolute values* instead:
$$\text{Loss} + \lambda \sum_i |w_i|$$

This looks like a tiny change. It isn't. Remember from [[Loss, Objectives, and Business Alignment#**L1 vs. L2 Loss**|L1 vs L2]] that $|x|$ has a **constant slope** of $1$ or $-1$ right up until zero — it doesn't shrink as you approach the bottom.

So L1 keeps pushing a weight toward zero *with the same force* no matter how small the weight already is. And eventually it shoves it clean **through** zero and pins it there.

The result: L1 doesn't shrink weights, it **deletes** them. You get a **sparse** model where most weights are exactly $0$ — which is automatic feature selection. The model is telling you "these 40 features out of 1000 are the ones that matter."

> [!WARNING] Don't confuse the two L1/L2s!
> This trips everyone up, and it's worth having straight in your head:
> - **L1/L2 _loss_** = how you measure the error on your *predictions*. (MAE vs MSE. About outliers.) → [[Loss, Objectives, and Business Alignment]]
> - **L1/L2 _penalty_** = how you measure the size of your *weights*. (Lasso vs Ridge. About overfitting.) → this note
>
> Same maths ($|x|$ vs $x^2$), applied to completely different things, solving completely different problems. ^l1-l2-confusion

| | **L1 (Lasso)** | **L2 (Ridge / Weight decay)** |
|---|---|---|
| **Penalty** | $\sum \|w_i\|$ | $\sum w_i^2$ |
| **Effect on weights** | Pushes them to *exactly* zero | Shrinks them *toward* zero |
| **Result** | Sparse model, few features survive | Dense model, all features slightly damped |
| **Use when** | You want feature selection / interpretability | You want stability, correlated features |
| **Gradient near 0** | Constant ($\pm 1$) — keeps pushing | Proportional ($2w$) — eases off |
| **Differentiable at 0?** | ❌ (needs sub-gradient) | ✅ |

And you can use **both** at once — that's called **Elastic Net**.

---
# The deeper way to see it: a regularizer is a prior

Here's the perspective that makes all of this click, and it's worth the detour.

When you add an L2 penalty, you are mathematically doing the *exact same thing* as saying:

> "Before I saw any data, I already believed the weights were probably small — clustered around zero."

That "before I saw any data" statement is a **prior** — a [[Beliefs|belief]] you hold going in. L2 is a Gaussian prior on the weights. L1 is a Laplace prior.

> [!SUCCESS] Core idea
> **Regularization = injecting what you already believe, before the data gets a say.**
> The $\lambda$ dial is literally "how strongly do I hold this belief vs. how much do I let the data change my mind?" ^reg-is-prior

This is why regularization matters most when data is **scarce**. With a billion samples the data overwhelms your prior and $\lambda$ barely matters. With 200 samples, your prior is doing most of the work — and if it's a good prior, that's a *feature*, not a hack.

It's the same trade the [[Kalman Filter]] makes, actually. The Kalman gain weighs "what I predicted" against "what I measured" by their uncertainties. Regularization weighs "what I assumed" against "what I observed" by $\lambda$. Same shape of idea.

---
# Regularization in deep learning (where it stops looking like maths)

In deep nets, most of the regularization you actually use isn't a penalty term at all. It's structural.

### Dropout
During training, randomly switch off a fraction of neurons (say 50%) on every forward pass.

Why on earth would breaking your own network help? Because a neuron can no longer *rely* on any specific other neuron being there — its partner might vanish next step. So it can't form fragile co-dependent chains ("I only fire correctly if neuron 47 also fires"). Every neuron is forced to be independently useful.

> [!TIP] The team analogy
> It's like a football team where the coach randomly benches half the players every practice. Nobody can build a strategy that depends on one star player. Everyone becomes individually competent, and the team gets robust. ⚽

At test time you turn dropout **off** and use the whole network — which is roughly like averaging over an ensemble of all the thinned networks you trained.

### Early stopping
Watch the validation loss. The moment it stops improving and starts creeping up, **stop training**. That's the point where the model switched from learning the pattern to memorising the noise.

It's the laziest regularizer and it's shockingly effective. It's also regularization by *time* rather than by penalty — you're limiting how far the weights are allowed to travel from their small random starting point.

### Data augmentation
Instead of constraining the model, expand the data. Flip the image, crop it, rotate it, change the brightness. Now the model sees 10× more "different" pictures of the same cat and is forced to learn *cat-ness* rather than *that exact arrangement of pixels*.

The cheapest and often the strongest regularizer you have — because you're attacking the root cause (not enough data) rather than the symptom.

### Label smoothing
Instead of training toward a hard target of $[0, 0, 1, 0]$, train toward $[0.02, 0.02, 0.94, 0.02]$.

You're telling the model "be confident, but not *infinitely* confident." Hard targets push the model to drive the correct logit toward $+\infty$, which makes it arrogant and badly calibrated. Smoothing caps that. Connects directly to [[Cross Entropy]] — you're softening the target distribution before you compute the loss.

### Normalisation layers (a side effect)
BatchNorm and LayerNorm weren't designed as regularizers, but BatchNorm accidentally is one — each sample's normalisation depends on the random other samples in its batch, which injects noise. Same flavour as dropout.

---
# The gotcha: weight decay ≠ L2 in Adam

This one costs people real experiments, so it's worth knowing.

With plain SGD, "add $\lambda \sum w^2$ to the loss" and "shrink every weight by a bit each step" are **the same operation**. The maths works out identically.

With **Adam**, they are *not* the same. Adam divides each gradient by a running estimate of its magnitude. So if you fold the L2 penalty into the loss, that penalty gets divided too — meaning weights with big gradients get *less* decay, which is backwards from what you want.

**AdamW** fixes this by decoupling: compute the Adam step from the loss gradient only, then apply the weight shrink separately, afterwards.

> [!WARNING] Practical rule
> If you're using Adam, use **AdamW**. If a paper reports a weight-decay value, check which one they meant — the numbers are not transferable between them. ^adamw-gotcha

---
# Summary

| Method | What it constrains | Cost |
|---|---|---|
| **L2 / weight decay** | Size of weights | Basically free, one hyperparameter |
| **L1 / Lasso** | Number of non-zero weights | Free, but non-smooth gradient |
| **Dropout** | Co-dependence between neurons | Slower convergence |
| **Early stopping** | How long weights can travel | Free — you needed a val set anyway |
| **Data augmentation** | Nothing — it adds data instead | Domain knowledge + preprocessing time |
| **Label smoothing** | Model's confidence | One hyperparameter |

> [!SUCCESS] If you remember one thing
> Every regularizer is answering the same question in a different accent: ~={pink}"what do I believe about the solution before I look at the data?"=~ — and then charging the model for disagreeing.

---
# ⁉️
So regularization is about admitting we might be wrong and building that humility into the objective. But that raises the obvious next question — **how do we actually measure how wrong we might be?** How does a model represent "I'm not sure"?

That leads to [[Uncertainty]].
