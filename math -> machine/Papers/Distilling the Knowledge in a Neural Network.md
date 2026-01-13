---
title: "Distilling the Knowledge in a Neural Network"
authors: ["Geoffrey Hinton", "Oriol Vinyals", "Jeff Dean"]
year: 2015
arxiv: "1503.02531"
url: https://arxiv.org/abs/1503.02531
priority: Must-Read
read_on: 2026-08-22
tags: [paper]
---
## The Core Idea

A trained classifier knows more than its labels say. When a big network looks at a photo of a BMW, it outputs maybe $0.99$ for "BMW", $10^{-3}$ for "garbage truck", and $10^{-9}$ for "carrot". The hard label only says "BMW". But the *ratio* between garbage-truck and carrot is real knowledge: it encodes which classes look like which. Hinton calls this the **dark knowledge** — the structure hidden in the tiny wrong-class probabilities.

The trick: train a small network to copy the big network's full output distribution, not the one-hot labels. Because the small model gets a whole probability vector per example instead of a single index, each training case carries far more bits of information, and the gradient varies less between cases. So the small model can learn from less data, at a higher learning rate, and it inherits the big model's *way of generalising* rather than just its answers.

The problem that blocked this before: those informative probabilities are so close to zero that they contribute almost nothing to [[Cross Entropy|cross-entropy]]. Caruana's earlier "model compression" fixed it by regressing on the raw logits with squared error. Hinton's fix is cleaner — divide the logits by a **temperature** $T$ before the softmax, which flattens the distribution and drags the small probabilities up into a range where the loss can see them. Matching logits turns out to be the $T \to \infty$ limit of this, so the old method is a special case.

What it unlocks: you can train a monstrous, slow ensemble to squeeze structure out of data, then pour that structure into one small deployable net. The "larval form vs adult form" analogy from the intro — the training model and the serving model no longer have to be the same animal.

> [!NOTE] Knowledge distillation — training a small "student" network to match the softened output distribution of a large "teacher" network, so the student inherits the teacher's generalisation behaviour rather than just the training labels. ^knowledge-distillation

> [!NOTE] Temperature — a divisor $T$ applied to logits inside the softmax. $T=1$ is normal; higher $T$ makes the distribution flatter and reveals the relative sizes of the small wrong-class probabilities. ^softmax-temperature

## The Methodology

**The softened softmax.** For logits $z_i$:

$$q_i = \frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$$

Set $T=1$ and you get the ordinary softmax. Raise $T$ and everything moves toward uniform, but the *ordering and relative gaps* survive.

**The loss.** Two terms, weighted:

1. **Soft loss** — cross-entropy between the student's distribution at temperature $T$ and the teacher's distribution at the *same* temperature $T$.
2. **Hard loss** — ordinary cross-entropy between the student at $T=1$ and the true label.

$$\mathcal{L} = T^2 \cdot \alpha \cdot H\big(p^{\text{teacher}}_T,\, q^{\text{student}}_T\big) + (1-\alpha)\cdot H\big(y,\, q^{\text{student}}_{T=1}\big)$$

The $T^2$ factor is the detail people forget. The gradient through a $1/T$-scaled softmax shrinks as $1/T^2$, so without rescaling, cranking up $T$ silently kills the soft term and you are just training on hard labels. Multiplying by $T^2$ keeps the two terms in a fixed ratio while you tune $T$.

Best results came from putting a **considerably lower weight on the hard-label term** (speech experiments used relative weight 0.5). Keeping some hard-label signal helps because the student cannot exactly match the teacher, and "erring toward the correct answer" is a good place to err.

**Why matching logits is a special case.** The gradient of the soft cross-entropy with respect to student logit $z_i$, with teacher logits $v_i$:

$$\frac{\partial C}{\partial z_i} = \frac{1}{T}\left(q_i - p_i\right)$$

If $T$ is large compared to the logit magnitudes, expand $e^x \approx 1+x$:

$$\frac{\partial C}{\partial z_i} \approx \frac{1}{T}\left(\frac{1+z_i/T}{N+\sum_j z_j/T} - \frac{1+v_i/T}{N+\sum_j v_j/T}\right)$$

Assume logits are zero-meaned per example ($\sum_j z_j = \sum_j v_j = 0$) and this collapses to

$$\frac{\partial C}{\partial z_i} \approx \frac{1}{NT^2}(z_i - v_i)$$

which is exactly the [[Derivative#Gradient|gradient]] of $\tfrac{1}{2}(z_i - v_i)^2$. So high-$T$ distillation *is* least-squares logit matching.

At **lower** $T$, distillation largely ignores logits far below average. That is a design choice with two sides: very negative logits are essentially unconstrained by the teacher's own training loss, so they may be pure noise — but they might also be real similarity information. The paper says this is an empirical question and the answer depends on student capacity (see ablations).

**MNIST setup.** Teacher: 2 hidden layers of 1200 ReLU units, trained on all 60k cases with dropout, weight constraints, and inputs jittered up to 2 pixels. Student: 2 hidden layers of 800 ReLU, no other regularisation.

**Speech setup.** 8 hidden layers × 2560 ReLU, softmax over 14,000 HMM tri-phone-state targets, ~85M parameters. Input is 26 frames of 40 Mel filterbank coefficients (10ms hop), predicting the state of frame 21. Trained on ~2000 hours of English (~700M examples) with distributed SGD. Teacher ensemble = 10 copies of the same architecture, differing only in random init — that alone gave enough diversity. Varying the *data* each model saw did not help further, so they dropped it.

**Specialist ensembles (JFT).** JFT = 100M images, 15,000 labels; the baseline conv net took ~6 months to train, so a normal ensemble was impossible. Instead: keep one generalist, then add many **specialists**, each covering ~300 confusable classes plus one "dustbin" class absorbing everything else. Class groupings came from online K-means on the columns of the covariance matrix of the generalist's predictions — no ground-truth labels needed. Clusters came out sensible: {Tea party, Easter, Bridal shower, Baby shower}, {Bridge, Cable-stayed bridge, Suspension bridge, Viaduct}, {Toyota Corolla E100, Opel Astra, Mazda Familia}.

Each specialist is initialised from the generalist's weights, then trained on 50% examples from its special subset and 50% sampled from everything else. Afterwards you correct the sampling bias by adding $\log(\text{oversampling factor})$ to the dustbin logit.

At test time: run the generalist, take its top-1 class, activate every specialist whose subset contains it, then solve for the combined distribution $\mathbf{q}$ by gradient descent on

$$KL(\mathbf{p}^g, \mathbf{q}) + \sum_{m \in A_k} KL(\mathbf{p}^m, \mathbf{q})$$

using [[KL Divergence]], with $\mathbf{q} = \text{softmax}(\mathbf{z})$ at $T=1$. Dustbin handling: sum all of $\mathbf{q}$'s mass over the classes that specialist $m$ lumped together. There is no closed form, so this optimisation runs *per image*.

## Ablation Studies and Experiments

**MNIST — the headline table.**

| Model | Test errors |
|---|---|
| Big net (1200×2, dropout + jitter) | 67 |
| Small net (800×2, no regularisation) | 146 |
| Small net distilled at $T=20$ | **74** |

The distilled 800-unit net closes almost the whole gap. Note what transferred: the teacher learned translation invariance from jittered inputs, but the transfer set contained **no translated images**. That knowledge came through the soft targets alone.

**Temperature vs student size — the most informative ablation.** With 300+ units per hidden layer, every $T > 8$ worked about the same. Drop the student to **30 units per layer** and the sweet spot moves to $T \in [2.5, 4]$ — high temperatures got worse. Reading: when the student has capacity to spare, keep all the dark knowledge including the very negative logits. When the student is too small to hold everything, low $T$ helps by *throwing away* the far-negative logits, so the limited capacity is spent on the distinctions that matter. This is the empirical answer to the noise-vs-information question raised in §2.1.

**The mythical digit.** Remove every example of "3" from the transfer set. The distilled student — which has literally never seen a 3 — makes 206 test errors, 133 of them on the 1010 threes. The failure is almost entirely a **bias** problem: bump the 3-class bias by $+3.5$ and errors drop to 109, with only 14 on threes. So the student gets **98.6% of test 3s correct having never seen one**. Harder version: transfer set contains only 7s and 8s → 47.3% test error, falling to 13.2% after adjusting the 7 and 8 biases by $-7.6$. The shape of the function transfers; the output offsets do not.

**Speech.**

| System | Frame accuracy | WER |
|---|---|---|
| Baseline (single) | 58.9% | 10.9% |
| 10× ensemble | 61.1% | 10.7% |
| Distilled single model | **60.8%** | **10.7%** |

Over 80% of the ensemble's frame-accuracy gain survives into one model of the *same size* as the baseline. Temperatures tried: $[1, \mathbf{2}, 5, 10]$; $T=2$ won. Note the small WER gain relative to frame accuracy — frame-level cross-entropy is a proxy objective, mismatched with the real decoding objective, so improvements there do not translate one-for-one. Classic [[Loss, Objectives, and Business Alignment|objective mismatch]].

For contrast, prior work (Li et al. 2014) distilled at $T=1$ on a large unlabelled set and recovered only **28%** of the large/small gap. Temperature is doing real work.

**Soft targets as regularisers — the cleanest result in the paper.**

| System & training set | Train frame acc | Test frame acc |
|---|---|---|
| Baseline, 100% of data | 63.4% | 58.9% |
| Baseline, 3% of data (~20M examples) | 67.3% | **44.5%** |
| Soft targets, 3% of data | 65.4% | **57.0%** |

Same 85M-parameter model, 3% of the data. Hard labels overfit catastrophically (train 67.3%, test 44.5%, and they needed early stopping because test accuracy *fell* after that point). Soft targets on the same 3% reach 57.0% — within 2 points of full-data training — and needed **no early stopping at all**; it simply converged. Soft targets act as a powerful [[Regularization|regulariser]] because they transmit the regularities the teacher found in the other 97%.

**JFT specialists.**

| System | Conditional test acc | Test acc |
|---|---|---|
| Baseline | 43.1% | 25.0% |
| + 61 specialists | 45.9% | **26.1%** |

4.4% relative gain overall. Broken down by how many specialists cover the correct class: 0 specialists → 0.0% change (by construction); 1 → +3.4%; 3 → +8.8%; 5 → +11.1%; 9 → +16.6%. More coverage, more gain — and since specialists train independently, coverage is cheap to buy with parallelism. Specialists took **days** where JFT training took **weeks to months**.

**What did not work / was abandoned:**
- Adding ensemble diversity by giving each model different data subsets — no significant change over plain random-init diversity, so they used the simpler thing.
- Modifying the soft targets themselves using the true labels — worse than just adding a separate weighted hard-label loss term.
- High temperature with a very small (30-unit) student — actively harmful.
- Computing the actual confusion matrix to define specialist clusters — they used prediction-covariance clustering instead, which needs no labels and worked just as well. Several clustering algorithms gave similar clusters.

## Worth Remembering

**Admitted limitations.** They never distilled the specialists back into a single large net — that was left as future work, and it is the obvious missing half of the JFT story. The specialist inference procedure also requires solving a per-image optimisation (gradient descent on $\mathbf{z}$), which is not cheap and undercuts the "efficient deployment" pitch. The full-softmax specialist idea (§6.1), where soft targets from the generalist prevent overfitting on non-special classes, is described as "currently exploring" — untested.

**Practical caveats.**
- Forget the $T^2$ rescale and your soft loss quietly evaporates as you raise $T$. This is the single most common implementation bug.
- $T$ is not a free-lunch hyperparameter — it interacts with student capacity. Small student → low $T$. Roomy student → $T$ barely matters above 8.
- The teacher's soft targets must be generated at the *same* $T$ as the student's soft loss. Then serve the student at $T=1$.
- Distillation transfers the *function shape* but not necessarily the output biases, as the missing-3 experiment shows. If your transfer set is class-imbalanced relative to deployment, expect calibration drift.

**Connections.** Dropout is framed here as training an exponentially large weight-sharing ensemble — so distilling from a heavily-dropout-regularised single model is the same move as distilling from a real ensemble. The specialist scheme is explicitly contrasted with mixtures of experts: MoE learns assignments jointly with a gating network, which makes training hard to parallelise (each expert's effective training set keeps shifting). Fixing the clusters up front from the generalist's confusion structure gives up adaptivity to buy embarrassing parallelism. Worth holding next to modern sparse MoE layers, which went back to the learned-gating side.

**Surprises worth keeping.** The 98.6% accuracy on a digit the model never saw. The 3%-data speech result — soft targets recovering 57.0% vs 44.5%. Both point at the same thing: a probability vector is a much richer teaching signal than an integer, and much of what we call "regularisation" is really "the model was told too little".

**Follow-up questions.** Does dark knowledge survive when teacher and student have different architectures (conv → transformer)? What happens when the teacher is *badly calibrated* — does distillation transfer the miscalibration? Modern LLM distillation typically uses full-sequence token distributions at $T=1$ plus vast synthetic transfer sets; is temperature obsolete when your vocabulary is 100k and the entropy is already high?

## Links

Related: [[Distillation]] · [[Cross Entropy]] · [[KL Divergence]] · [[Regularization]] · [[Deep Learning]] · [[Backpropagation]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Loss, Objectives, and Business Alignment]] · [[Derivative]] · [[Uncertainty]]

New topics worth writing: Dark knowledge, Softmax temperature, Model compression (Buciluǎ & Caruana 2006), Mixture of Experts, Dropout as implicit ensembling, Ensemble methods, Model calibration, Hidden Markov Model acoustic modelling, Word Error Rate, Online K-means clustering, Label smoothing (as a cousin of soft targets)
