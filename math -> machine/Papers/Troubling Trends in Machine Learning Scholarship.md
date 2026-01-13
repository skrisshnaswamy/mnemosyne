---
title: "Troubling Trends in Machine Learning Scholarship"
authors: ["Zachary C. Lipton", "Jacob Steinhardt"]
year: 2018
arxiv: "1807.03341"
url: https://arxiv.org/abs/1807.03341
priority: Good-To-Read
read_on: 2026-08-29
tags: [paper, transformers, llm]
---
## The Core Idea

This is not a method paper. It has no model, no loss, no benchmark. It is a critique of how machine learning papers are *written*, and it names four specific bad habits that had become normal by 2018.

The argument: a paper is valuable when it serves the reader. That means (i) giving intuition but labelling it as intuition, (ii) ruling out alternative explanations, (iii) being clear about where maths ends and hand-waving begins, and (iv) picking words that do not smuggle in claims you never proved. Lipton and Steinhardt claim the field was drifting away from all four, and that strong empirical results were being used as a licence to skip them.

The four trends:

1. **Explanation vs speculation.** Authors give a story for *why* their method works, dressed as a fact, with no experiment testing it.
2. **Failure to identify the source of gains.** Ten changes are proposed at once, one of them is doing all the work, and nobody knows which.
3. **Mathiness.** Equations that impress reviewers rather than clarify anything, with loose joints between the formal claims and the English claims.
4. **Misuse of language.** Words picked for their flattering connotations, or technical words redefined into mush.

Why this matters even now: the failure modes compound. A speculative story cited a thousand times becomes "well-known". A benchmark redefined as "reading comprehension" makes an unsolved problem look solved. Both corrupt what the next generation of researchers thinks is already settled. The authors are explicit that they are insiders indicting themselves — they use their own papers as examples, and otherwise pick on senior, well-defended researchers so junior students do not get hurt.

## The Methodology

There is no experiment. The method is: define the trend, give real examples from real papers (including positive counter-examples), state the consequence.

**Trend 1 — speculation dressed as explanation.**
The headline example is [[Batch Normalization]]. The paper says BatchNorm works by reducing *internal covariate shift* — a change in the distribution of hidden activations during training. Lipton and Steinhardt ask the obvious question: **by which divergence measure is this change quantified?** It is never said. The claim has no truth value you could check. Yet it propagated: Noh et al. (2015) write "it is well-known that a deep neural network is very hard to optimize due to the internal-covariate-shift problem." Santurkar et al. later showed the explanation is probably wrong — see [[How Does Batch Normalization Help Optimization]].

Steinhardt indicts himself twice: claiming high dimensionality "gives the attacker more room" in a data-poisoning paper without measuring the effect of dimension, and using an undefined notion of "coverage" as an explanation.

The positive example is [[Dropout- A Simple Way to Prevent Overfitting]], which speculates at length about sexual reproduction as an analogy — but quarantines it inside a section labelled "Motivation". You can speculate. Just fence it off.

**Trend 2 — where did the gains come from?**
Melis, Dyer and Blunsom (2018) re-tuned hyperparameters on a series of "architectural innovations" in language modelling, and a vanilla [[Long Short-Term Memory (Neural Computation)|LSTM]] — essentially unchanged since 1997 — topped the leaderboard. The improvements were tuning, not architecture. The same has been shown for deep RL (Henderson et al.) and for GANs (Lucic et al., "Are GANs Created Equal?").

The sharp line: proposing many tweaks without ablations "can give the false impression that the authors did more work, when in fact they did not do enough."

They also note ablation is not the only route to understanding. Robustness checks work (Cotterell et al. find language models handle inflectional morphology badly). Qualitative error analysis works. And pure analysis papers with no new algorithm can be the most valuable: Chen, Bolton and Manning found that on the CNN/Daily Mail "reading comprehension" set, **73% of questions are answerable from a single sentence and only 2% need multiple sentences** (25% were ambiguous or had coreference errors) — and simple linear classifiers beat the complicated architectures published on it.

**Trend 3 — mathiness.**

> [!NOTE] Mathiness
> Mixing symbols and words without tight links between them, so that vague definitions hide holes in the theory while the appearance of formalism props up weak prose. Term borrowed from economist Paul Romer. ^mathiness

Three sub-species:
- **Spurious theorems.** Steinhardt again names his own paper, where "staged strong Doeblin chains" had little bearing on the algorithm but conferred theoretical gravity. The more famous case: [[Adam- A Method for Stochastic Optimization]] proves convergence in the convex case for an optimizer used almost entirely on non-convex problems — and the proof was later shown to be wrong (Reddi et al., 2018).
- **Claims that are neither formal nor informal.** Dauphin et al. cite a statistical physics result and state that in high dimensions "all local minima are likely to have an error very close to that of the global minimum." Is that a limit statement, or a numerical observation at typical parameter settings? Without a theorem you cannot tell.
- **Theorems invoked out of scope.** The *no free lunch theorem* gets cited to excuse heuristics, even though it does not formally rule out learning procedures with guarantees.

The positive example is Bottou et al.'s counterfactual reasoning tutorial — see [[Counterfactual Reasoning and Learning Systems]] — heavy maths, but every piece tied to a real applied problem.

**Trend 4 — misuse of language**, in three flavours:

> [!NOTE] Suggestive definition
> Coining a technical term whose everyday meaning imports connotations you never argued for. "Curiosity", "fear", "thought vectors", "consciousness prior". ^suggestive-definition

Once a suggestive term acquires a technical meaning, every later paper is stuck: adopt it and confuse readers, or rename it and fragment the literature.

The "human-level" family is the worst offender. "Dermatologist-level classification of skin cancer" hides the fact that the classifier and the dermatologist are doing *different tasks*: the dermatologist handles unpredictable variety, the classifier gets low error on i.i.d. test data. [[Delving Deep into Rectifiers (He init, PReLU)]] is credited as the careful case — it qualifies its human-level claim to the ImageNet classification task specifically — and even that was not enough to stop the press. Meanwhile networks misclassify "Asians dressed in red" as ping-pong balls.

> [!NOTE] Overloaded terminology
> Taking a word with an exact technical meaning and using it loosely or contradictorily. ^overloaded-terminology

- **Deconvolution** properly means reversing a convolution. In deep learning it now usually means a transpose/up-convolution. A paper using the word could mean either, or be trying to fix the confusion ("upconvolution (deconvolution)").
- **Generative model** properly means a model of $p(x)$ or $p(x, y)$, versus a discriminative model of $p(y \mid x)$. It now means "anything that emits realistic-looking structured data" — which hides that a [[Generative Adversarial Networks|GAN]] or a [[Auto-Encoding Variational Bayes (VAE)|VAE]] cannot do conditional inference like sampling $p(x_2 \mid x_1)$ for two input features.
- **Covariate shift** properly means $p(x)$ changes while $p(y \mid x)$ stays fixed. The BatchNorm paper uses it for any change in input distribution — and is influential enough that Google Scholar returned it as the *first* result for "covariate shift".

> [!NOTE] Suitcase word
> Minsky's term: a word packing several unrelated meanings into one bag, so two papers can appear to be in dialogue while discussing different things. ^suitcase-word

Examples: *interpretability* (no agreed meaning, references disjoint methods and goals), *bias* in the fairness literature, and *generalization* — which means both train→test and the colloquial train→other population or lab→real world. Conflating them makes systems look more capable than they are. Bostrom's *Superintelligence* writes an equation containing "intelligence" and "optimization power" as if both were scalars.

## Ablation Studies and Experiments

There are none — this is a position paper for the ICML Debates workshop. What stands in for experiments is the cited evidence and the diagnosis of causes.

**Proposed causes** (explicitly labelled as speculation, practising what they preach):
- *Complacency*: strong numbers excuse weak arguments. Single-round reviewing makes it worse — a reviewer who rejects a flawed paper has no guarantee the flaw is fixed next cycle, so accepting it feels like the least bad option.
- *Growing pains*: since ~2012 the community exploded; the reviewer pool thinned both in ratio and in experience. Inexperienced reviewers demand architectural novelty, are impressed by spurious theorems, and miss language misuse. Overburdened experienced reviewers fall into checklist mode, rewarding formulaic work.
- *Misaligned incentives*: press coverage rewards anthropomorphism (Wired calling an autoencoder a "simulated brain"; NYT saying a captioning system "mimics human levels of understanding"), and investors sometimes fund startups off a single well-covered paper.

**The countervailing considerations they concede** — this is the honest part:
- Maybe noisy fast progress beats clean slow progress. "SGD converges faster than gradient descent." [[ImageNet Classification with Deep CNNs (AlexNet)]] proposed many techniques with no ablations, several later shown unnecessary — but the result was so big and the compute so expensive that waiting would have cost the community more.
- High standards may block original, speculative ideas. In economics, high standards mean multi-year revision cycles.
- Maybe specialisation is fine: the people generating ideas need not be the people distilling them.

Their response: these are heuristics, not rules. "If an idea cannot be shared without violating these heuristics, we prefer the idea be shared." But the field is not on the Pareto frontier of growth vs quality — most of the fixes cost a few extra days of experiments and more careful writing.

**Concrete tests they propose.**
For authors: *would I rely on this explanation to make a prediction or get a system working?* If not, the theorem is there to please reviewers.
For reviewers: *might I have accepted this paper if the authors had done a worse job?* A simple idea with a gain plus two honest negative results should beat three ideas bundled without ablations for the same gain.

## Worth Remembering

- The three practices they say separate strong empirical papers from weak ones: **error analysis, ablation studies, robustness checks** (to hyperparameters and ideally to dataset choice).
- Papers that add knowledge with *no new algorithm* are held up as exemplary: [[Understanding Deep Learning Requires Rethinking Generalization]] (networks fit random labels, so learning-theoretic complexity does not explain generalisation), and Goodfellow et al. finding that straight-line paths in parameter space from init to solution usually have monotonically decreasing loss.
- The historical framing is the part most people forget. Drew McDermott, 1976, on AI's suggestive definitions: *"if we can't criticize ourselves, someone else will save us the trouble."* Cohen and Howe, 1988, asking AI to publish performance evaluations at all. Armstrong et al., 2009, on information retrieval comparing against the same weak baselines for a decade so that "improvements" never accumulated — exactly the finding of [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] ten years later. Psychology's 2015 reproducibility study. N-rays.
- Real-world consequence they cite: the European Parliament passed a report considering regulations for if "robots become or are made self-aware". Anthropomorphic language in peer-reviewed papers is at least partly to blame.
- A caveat for using this note as ammunition: the authors are careful that exhibiting one of these patterns does not make a paper bad or indict its authors. BatchNorm and Adam are enormously useful *despite* the flaws named here. The point is that both would be stronger papers without them.
- Open questions they leave: does open review help or hurt? Do reviewer point systems align with these values? They also want more authoritative retrospective surveys that strip exaggerated claims, rename anthropomorphic terms, and standardise notation — and want critical/position papers to have a venue at ML conferences, since neither an algorithm nor an experiment can address whether the *problem formulation* is valid.
- Practical use: this is a checklist to run over your own drafts, and a lens for reading. When a paper tells you *why* its method works, ask what measurement would falsify that story. Usually there is none.

## Links

Related: [[Batch Normalization]] · [[How Does Batch Normalization Help Optimization]] · [[Adam- A Method for Stochastic Optimization]] · [[Dropout- A Simple Way to Prevent Overfitting]] · [[Long Short-Term Memory (Neural Computation)]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[Understanding Deep Learning Requires Rethinking Generalization]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[On the Difficulty of Evaluating Baselines]] · [[Counterfactual Reasoning and Learning Systems]] · [[Generative Adversarial Networks]] · [[Auto-Encoding Variational Bayes (VAE)]] · [[Delving Deep into Rectifiers (He init, PReLU)]] · [[Shortcut Learning in Deep Neural Networks]] · [[Do ImageNet Classifiers Generalize to ImageNet]] · [[Towards Quantifying Benchmark Optimization in ASR Models]] · [[Hidden Technical Debt in Machine Learning Systems (NeurIPS)]] · [[Inherent Trade-Offs in the Fair Determination of Risk Scores]] · [[Playing Atari with Deep Reinforcement Learning (DQN)]] · [[The Bitter Lesson (essay)]]

New topics worth writing: On the State of the Art of Evaluation in Neural Language Models (Melis et al.), Deep Reinforcement Learning That Matters (Henderson et al.), Are GANs Created Equal? (Lucic et al.), A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task (Chen et al.), On the Convergence of Adam and Beyond (Reddi et al.), The Mythos of Model Interpretability, Mathiness in the Theory of Economic Growth (Romer), Strong Inference (Platt 1964), Artificial Intelligence Meets Natural Stupidity (McDermott 1976), Random Search for Hyper-Parameter Optimization (Bergstra & Bengio), Identifying and Attacking the Saddle Point Problem, Learning in Implicit Generative Models, Winner's Curse? On Pace, Progress and Empirical Rigor
