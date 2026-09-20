---
aliases:
  - Cross-Entropy
  - CE Loss
  - Log Loss
tags:
  - fundamentals
  - information-theory
  - loss
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** The **average surprise** you get when you use your model's probabilities to describe what actually happens.
> **Metaphor:** Weather forecasting. Predict 70% sun when it's really 90% sun, and you'll be "surprised" more often than you needed to be.
> **Where it bites:** The default loss for every classification task and every language model. Gotcha: `nn.CrossEntropyLoss` wants **logits**, not probabilities.

---
Cross-Entropy is a [[Loss Function]] used in classification tasks, and its name comes from the concept of [[Entropy]] in [[Information Theory]]. My initial thought was that entropy meant "information loss," but it's more accurate to think of it as a measure of **uncertainty** or **average surprise**.
## Intuition
Imagine our only goal is to predict if tomorrow will be sunny ☀️ or rainy 🌧️.
And let's say we know the **ground truth**: in reality, it's sunny **90%** of the time and rains only **10%** of the time.
### Step 1: Entropy (Reality - The Best Possible Scenario)
**Entropy** is about information. Think of it as the measure of **average surprise**.
- A sunny day (90% probability) is very predictable, so it has **low surprise**.
- A rainy day (10% probability) is less common, so it has **high surprise**.
### Step 2: Cross-Entropy (Our Model vs. Reality)

Now, let's say we build a machine learning model to predict the weather. Our model isn't perfect yet, and it predicts:
- Sunny ☀️: 70% chance
- Rainy 🌧️: 30% chance

**Cross-entropy** measures the average surprise if we use our **model's predictions** (70/30) to describe the **actual outcomes** (90/10).

Because our model thinks rain is more likely (30%) than it actually is (10%), it's going to be "less surprised" by rain than it should be. Conversely, it will be "more surprised" by sun.
Cross-entropy takes this mismatch and calculates the total average surprise. This value will always be higher than the true entropy unless the model is perfect.

~={blue}The extra amount of "surprise" is our **loss**.=~ Our goal during training is to adjust the model's predictions to minimize this cross-entropy loss, pushing its predictions closer and closer to the ground truth.


So, **Cross-Entropy** measures the average surprise or "penalty" we get when we use a model's _predicted_ probabilities to describe the _actual_ outcomes. If our model predicts a 70/30 chance of sun/rain, it's not perfectly aligned with the 90/10 ground truth. Cross-entropy quantifies this misalignment. The goal of training is to minimize this value, pushing the model's predictions to match the truth as closely as possible.

---
## A Key Doubt: Certain vs. Probabilistic Ground Truth

So in the weather analogy, the ground truth was a **probability** (90/10). But in a real classification task, like identifying a cat in a photo, the ground truth is a **fact**. The image _**is**_ a cat. So what do we compare the model's prediction to?

You see there are two "types" of ground truth:
1. **Population/Dataset Truth:** This is the overall distribution, like the 90/10 weather pattern. It's useful for high-level intuition.
2. **Instance Truth:** This is the factual label for a single training example. For an image that is a cat, we express this truth with 100% certainty using a **one-hot vector**: `[1, 0]`. This means it's "100% a cat" and "0% a dog."

When training a model, we always use the **Instance Truth**. The entropy of this certain `[1, 0]` label is actually **0**, because there is zero uncertainty.

We then compare our model's probabilistic prediction (e.g., `[0.9, 0.1]`) against this certain, zero-entropy ground truth.

> [!NOTE] The Math: Why the Logarithm?
> This got me thinking, why do we use `log(p)`? Why not a simpler operation to measure the error? The answer comes back to the idea of **"surprise"** from [[Information Theory]].
>
The key insight is that the "surprise" of two independent events should be the **sum** of their individual surprises. However, their probabilities are **multiplied**.
>
 $f(prob_m1 * prob_m2) = f(prob_m1) + f(prob_m2)$
>
The only function that satisfies this property is the **logarithm**. This makes it the natural, mathematically correct way to measure surprise. The negative sign is added ($-log(p)$) simply to make the loss a positive number.
>
> **Why not a linear penalty?**
If we used a simpler penalty like the absolute error, `|y - p|`, it wouldn't punish confident mistakes harshly enough.
>
> - A model predicting `p=0.1` for a true `y=1` has an error of `0.9`.
> - A model predicting `p=0.01` has an error of `0.99`. 
>
The penalty barely increases. But with `-log(p)`:
> - Loss for $p=0.1$ is $-log(0.1) ≈ 2.3$.
> - Loss for $p=0.01$ is $-log(0.01) ≈ 4.6$.
>
The penalty **doubled**. It explodes as the model gets more confidently wrong, creating a massive [[Derivative#Gradient|gradient]] that forces the model to correct its biggest mistakes quickly.

---

## Putting It All Together: A Practical Example

Here's how this works in a single step of training a model to classify a **Cat** (class 1) vs. a **Dog** (class 0). The model's job is to output a single probability, `p`, for the "Cat" class.

### Step 1: The Forward Pass (Prediction)

We give the model an image of a **cat**.

- **Ground Truth (y):** 1
    
- **Model Prediction (p):** The untrained model outputs `p = 0.6` (it's 60% sure it's a cat).
    

### Step 2: Calculating the Loss

Because the ground truth `y` is 1, the loss is just `-log(p)`.

`Loss = -log(0.6) ≈ 0.51`

This value, 0.51, is the error for this one example.

### Step 3: Learning (The Backward Pass)

The loss is used to calculate a [[Derivative#Gradient|gradient]]. This gradient is a set of directions for the model's internal parameters. The directions essentially say: "Adjust yourselves so that next time you see this cat image, your prediction `p` is higher than 0.6."

By repeating this process thousands of times with different cat and dog images, the model slowly learns to produce a high `p` value for cats and a low `p` value for dogs.

---
# The relationship to entropy and KL — all three in one place

These three get muddled constantly. They're the same quantity sliced differently:

| | What it measures | Floor |
|---|---|---|
| **Entropy** $H(P)$ | Average surprise using **reality's own** probabilities | The irreducible minimum |
| **Cross-entropy** $H(P,Q)$ | Average surprise using **your model's** probabilities | $\geq H(P)$, always |
| **[[KL Divergence]]** $D_{KL}(P \parallel Q)$ | The **gap** between them — purely your model's fault | $\geq 0$, zero iff perfect |

$$H(P,Q) = \underbrace{H(P)}_{\text{unavoidable}} + \underbrace{D_{KL}(P \parallel Q)}_{\text{your error}}$$

> [!SUCCESS] Why we minimise cross-entropy and not KL
> They differ by $H(P)$ — and $H(P)$ is a property of the **data**, not of your model. It's a constant. Minimising cross-entropy and minimising KL are therefore ~={blue}the exact same optimisation problem=~, and cross-entropy is simply cheaper to compute (you never need to know the true entropy).
>
> Which means: **training with cross-entropy loss *is* minimising forward KL to the data distribution.** That's why it's mode-covering — see [[KL Divergence#Forward KL vs Reverse KL|forward vs reverse KL]]. ^ce-vs-kl

---
# The practical gotchas

> [!WARNING] `nn.CrossEntropyLoss` takes **logits**, not probabilities
> The single most common bug in this area. PyTorch's `CrossEntropyLoss` applies `log_softmax` **internally**. If you put a softmax in your model's final layer and then use this loss, you've applied softmax twice — training will still "work", just badly and slowly, with no error raised. 🐛
>
> - Raw logits → `nn.CrossEntropyLoss`
> - Already log-probabilities → `nn.NLLLoss`
> - Binary, raw logits → `nn.BCEWithLogitsLoss` (**not** `BCELoss`) ^logits-not-probs

**Why the fused version exists.** Computing softmax then log separately overflows for large logits. Combining them allows the *log-sum-exp trick* — subtract the max logit before exponentiating — which is numerically stable. This is why you should essentially never hand-roll it.

**Class imbalance.** With 99% negatives, predicting "negative" always gives a low loss and a useless model. Fixes: class weights in the loss, or **focal loss**, which down-weights already-easy examples so the gradient focuses on the hard ones.

**Label smoothing.** Training toward $[0,0,1,0]$ pushes the correct logit toward $+\infty$ — the model becomes overconfident and badly calibrated. Softening the target to $[0.02,0.02,0.94,0.02]$ caps that. See [[Regularization#Label smoothing|label smoothing]] and [[Uncertainty#How you actually get a number out of a model|calibration]].

**Perplexity.** For language models you'll usually see perplexity rather than loss: $\text{PPL} = e^{\text{cross-entropy}}$. It's the same number, exponentiated, and it has a nice reading — *"the model is effectively choosing uniformly among this many tokens."* A perplexity of 20 means it's about as uncertain as picking from 20 equally likely words. 📖

---
# ⁉️
Cross-entropy measures the gap between prediction and truth. The isolated gap itself — and the crucial fact that it is **asymmetric** — is [[KL Divergence]].
