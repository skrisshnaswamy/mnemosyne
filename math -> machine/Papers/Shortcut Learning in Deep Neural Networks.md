---
title: "Shortcut Learning in Deep Neural Networks"
authors: ["Robert Geirhos", "Jörn-Henrik Jacobsen", "Claudio Michaelis", "Richard Zemel", "Wieland Brendel", "Matthias Bethge", "Felix A. Wichmann"]
year: 2020
arxiv: "2004.07780"
url: https://arxiv.org/abs/2004.07780
priority: Must-Read
read_on: 2026-08-25
tags: [paper, rl, vision]
---
## The Core Idea

Lots of separate-looking failures of deep nets are the same failure. A model finds a rule that works on the training data and on the test data drawn the same way, but that rule is not the rule you wanted. The paper gives this one name: **shortcut learning**.

The classic examples:

- A pneumonia detector on chest X-rays got great accuracy. It had learned to spot a **metal token** that one hospital stamps on its scans, plus that hospital's base rate of pneumonia. It knew almost nothing about lungs.
- A cow classifier fails when the cow stands on a beach. "Grass" was the feature.
- BERT looked like it could evaluate arguments. It was mostly checking whether the word **"not"** appeared.
- Amazon's résumé screener preferred men. Deleting names did not fix it — it inferred gender from all-women college names.
- A Tetris agent learned to **pause the game forever**, because a paused game never loses.

Why is this one problem and not five? Because the standard test protocol cannot tell a shortcut from the real thing. We split data randomly into train and test, so the test set is **i.i.d.** with the training set — same distribution, same biases. Any rule that exploits a bias present in training is *also* rewarded at test time. The leaderboard is blind by construction.

> [!NOTE] Shortcut
> A decision rule that gets high accuracy on the training set **and** on an i.i.d. test set, but breaks on out-of-distribution (o.o.d.) data, where the intended rule would still hold. ^shortcut

The unlock is a change of default: **o.o.d. testing should be the rule, not the exception**, and "the model achieved X% on benchmark Y" should never be read as "the model has ability Z". They borrow a rule of thumb from comparative psychology — Morgan's Canon, "never explain behaviour by a high mental process if a low one will do" — and restate it for ML: *never attribute to high-level abilities that which can be adequately explained by shortcut learning.*

Also worth internalising: this is not a bug unique to silicon. Rats "discriminating colours" in a maze were actually smelling the paint. Students who rote-learn ("surface learning") get *better* multiple-choice grades than students who understand. Any learner that is graded narrowly will optimise the grading, not the goal.

## The Methodology

This is a perspective paper, not an experimental one. It has one toy experiment, one taxonomy, and one framework for where shortcuts come from.

**The toy experiment (Figure 2).** A $200 \times 200$ canvas, one shape per image: a star or a moon. Training set = 4000 images. During training, **stars only appear in the top-right or bottom-left quadrant; moons only in the top-left or bottom-right**. So location is perfectly predictive of class, and so is shape, and so is white-pixel count (moons are smaller). Three valid rules, one dataset.

The i.i.d. test set (2000 images) keeps the same location rule. The o.o.d. test set places shapes anywhere on the canvas.

Two models, both trained 5 epochs, batch size 100, [[Cross Entropy|cross-entropy]] loss, Adam with a *very* small learning rate $10^{-5}$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$:

- **Three-layer ReLU MLP**, 1024 units per layer, 2 outputs. 100% train accuracy → **51.0% on the o.o.d. test set**. Chance. It learned location.
- **Three-layer CNN**, 128 channels, stride 2, $5\times5$ filters, global average pooling, then a linear layer. **100% on both** train and o.o.d. test.

The [[ImageNet Classification with Deep CNNs (AlexNet)#^convolutional-layer|convolution]] is doing all the work here. Weight sharing makes location *hard* to represent, so the network is pushed toward shape. This is the concrete demo that architecture — not data — decides which of the equally-valid rules gets picked.

**The taxonomy (Figure 3).** Nested sets of decision rules, each narrower than the last:

1. **All possible rules**, most of which are non-solutions using *uninformative features* (predict "star" when you see a white pixel — both classes have white pixels).
2. **Rules that fit the training set**, including ones using *overfitting features* — they fit train but fail the i.i.d. test.
3. **Rules that also pass the i.i.d. test** — these are what tops leaderboards. Includes both shortcuts and the intended solution. **Nothing in i.i.d. testing separates these two.**
4. **The intended solution**, which additionally survives o.o.d. tests.

The point of drawing it this way: the intended solution is a strict subset, and standard practice only ever constrains you to level 3.

**Where shortcuts come from — two ingredients.**

*Ingredient one: shortcut opportunities in the data.* Real correlations create them (cows really do stand on grass). They do **not** vanish with scale — "systematic biases are still present even in Big Data." Some are invisible: high-frequency texture patterns humans cannot see. This connects to [[Understanding Deep Learning Requires Rethinking Generalization]] — data alone does not pin down a function.

*Ingredient two: how discriminative learning combines features.* A discriminative model picks *whatever suffices to separate classes*. It has no notion of what an object typically looks like, so it can equate "cat" with "cat texture" and drop shape entirely — an ImageNet-trained CNN calls a cat-shaped image with elephant skin an elephant. Taken to the limit, some models key on a **single predictive pixel** and ignore everything else. The authors call this **excessive invariance**: the model is invariant to nearly all features a human would use.

> [!NOTE] Excessive invariance
> Ignoring evidence is sometimes what you want (shift invariance in object recognition). It becomes a failure when the model is invariant to the features that *define* the class. Adversarial examples are the flip side: the model is hyper-sensitive to features humans ignore, and hyper-insensitive to features humans use. ^excessive-invariance

**The four levers (Box II).** What makes a solution "easy to learn" is the model's [[An Image is Worth 16x16 Words (ViT)#^inductive-bias|inductive bias]], and that has exactly four components:

- **Architecture.** Convolutions penalise location. [[Attention Is All You Need|Transformer]] attention encodes relations between tokens. Even ReLUs matter — they produce unwarranted high confidence far from the training data.
- **Training data.** Blocking a shortcut in the data works: adversarial training reduces adversarial vulnerability; style-transfer augmentation reduces texture bias.
- **Loss function.** Cross-entropy *encourages the network to stop learning once a simple predictor is found*. Once the loss is near zero the gradient is near zero, and the remaining evidence is never used.
- **Optimisation.** SGD biases toward simple functions. Large learning rates → simple shared patterns; small learning rates → complex patterns and memorisation.

## Ablation Studies and Experiments

The toy experiment *is* the ablation, and it is a clean one: same data, same loss, same optimiser, same depth. Swap MLP → CNN and o.o.d. accuracy goes **51.0% → 100%**. Architecture alone flipped which of three equally-valid rules the network chose. No amount of extra data would have fixed the MLP, because location remains perfectly predictive in the training distribution.

**What did not work, from the literature they collect:**

- **More data does not fix it.** Shortcut opportunities "rarely disappear by adding more data." Scale attacks sampling noise, not systematic bias.
- **Unsupervised pre-training does not fix it.** BERT and GPT-2 still lean on cue words like "not" for argument comprehension. See [[BERT- Pre-training of Deep Bidirectional Transformers]] and [[Language Models are Few-Shot Learners (GPT-3)]] — pre-training buys a lot, but not shortcut-freedom.
- **Deleting the sensitive attribute does not fix fairness.** Amazon's tool re-derived gender from proxies after names were removed.
- **Generative modelling alone does not fix it.** Learning $p(x)$ does not automatically give representations useful downstream, let alone o.o.d.-robust ones. This is why disentanglement research exists. Relevant to [[Auto-Encoding Variational Bayes (VAE)]] and [[Generative Adversarial Networks]].
- **Benchmarks decay.** The Winograd Schema Challenge was *designed* to be shortcut-proof, and turned out to be full of shortcuts once language models started scoring suspiciously well. O.o.d. tests must evolve alongside models.

**The diagnostic they push hardest: the surprisingly strong baseline.** Build a model that deliberately cannot see the intended feature, and see how well it does. If it does well, your benchmark is broken.

- Bag-of-local-features ImageNet classifiers (no global shape) do "surprisingly well" — so ImageNet mostly measures counting texture patches, not object recognition.
- Hypothesis-only baselines in Natural Language Inference: read only the hypothesis, never the premise, still beat chance by a lot.
- Answer MovieQA questions **without showing the model the movie**.
- Generate image captions **without looking at the image**.

**O.o.d. benchmarks they endorse (Box I),** with what each isolates:

| Benchmark | What it removes / controls |
|---|---|
| ImageNet-C | 15 corruption types (noise, blur, weather, JPEG) |
| ImageNet-A | natural images that SOTA models consistently get wrong |
| ObjectNet | scientific controls on background, rotation, viewpoint |
| Cue-conflict images | shape vs texture pitted against each other; directly comparable to humans |
| PACS | trained on photos, tested on cartoons/sketches |
| ARCT (shortcuts removed) | the known cue words are deleted from the dataset |
| Shift-MNIST / biased CelebA / unfair dSprites | a class-predictive pixel or quality artefact is *injected*, then you measure the accuracy drop on clean data |

That last row is the most useful as an engineering tool: it turns "how shortcut-prone is this architecture + loss?" into a number you can measure.

**Three conditions for a good o.o.d. test:** (1) a genuine distribution shift; (2) a well-defined intended answer — training on photos and testing on white noise is o.o.d. but meaningless; (3) current models actually struggle. Most conceivable o.o.d. tests are uninteresting.

## Worth Remembering

- **Generalisation is not absent, it is misdirected.** DNNs generalise *enormously* — the set of images classified "guitar" with high confidence includes abstract patterns, noise, and things that look like food. The failure is not "cannot generalise", it is "generalises along the wrong axis". Robustness and shortcut learning are the same coin.
- **The cross-entropy point is underrated and actionable.** Cross-entropy rewards the model for stopping as soon as *one* sufficient predictor is found. If you want the model to use all the evidence, cross-entropy is actively working against you. Relevant when you design [[Loss, Objectives, and Business Alignment|objectives]].
- **Reward hacking is shortcut learning in [[Markov Decision Process|RL]] clothes.** The Tetris-pausing agent is the same phenomenon as the grass-detecting cow classifier, just with a reward function instead of a label. The sim-to-real gap in robotics is too — domain randomisation (randomise colour, size, texture, lighting) works because it destroys the simulator-specific shortcuts.
- **Not all fairness problems are shortcut problems**, and the authors are careful about this. But many are: once a feature is predictive, even if it is a dataset artefact, the rule may depend on it entirely. Bias amplification means the model can end up *more* biased than the humans whose decisions it was trained on.
- **The "same strategy assumption" is the trap.** Human-level performance does not imply human-like strategy. This is anthropomorphism, and comparative psychologists have been warning about it since 1903.
- **Practical caveat:** this paper gives you a vocabulary and a checklist, not a method. If you want an actual algorithm, the pointers are Invariant Risk Minimization, adversarial training, conditional variance penalties, group-equivariant CNNs, AutoAugment, and disentangled generative models — all referenced, none developed here.
- **Follow-up question worth sitting with:** the CNN solved the toy problem because someone knew, in advance, that location was irrelevant. The four levers only help if you already know which invariance you want. For problems where you do not know, what is the move? The paper's honest answer is: test o.o.d. and find out empirically.
- Written in 2020, pre-scaling-boom. Worth reading against [[Scaling Laws for Neural Language Models]] and [[The Bitter Lesson (essay)]] — those say "add compute and data"; this says "data alone rarely constrains a model sufficiently, and data cannot replace making assumptions". Both can be true: scale kills *some* shortcuts (rare-case coverage) and leaves *systematic* ones untouched.

## Links

Related: [[Understanding Deep Learning Requires Rethinking Generalization]] · [[An Image is Worth 16x16 Words (ViT)]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Cross Entropy]] · [[Regularization]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Counterfactual Reasoning and Learning Systems]] · [[Unbiased Learning-to-Rank with Biased Feedback]] · [[Offline Reinforcement Learning- Tutorial, Review, and Perspectives]] · [[Why do tree-based models still outperform deep learning on tabular data]] · [[Reconciling Modern ML Practice and the Bias-Variance Trade-off]] · [[The Bitter Lesson (essay)]] · [[Uncertainty]]

New topics worth writing: Invariant Risk Minimization, Adversarial examples and adversarial training, Domain adaptation and domain generalisation, Domain randomisation for sim-to-real, Texture bias vs shape bias in CNNs, ImageNet-C and corruption robustness, Reward hacking and specification gaming, Disentangled representation learning, Covariate shift, Clever Hans effect, Group-equivariant CNNs, Annotation artefacts in NLI datasets, Bias amplification in ML systems
