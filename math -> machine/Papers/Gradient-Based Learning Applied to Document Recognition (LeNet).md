---
title: "Gradient-Based Learning Applied to Document Recognition (LeNet)"
authors: ["LeCun et al."]
year: 1998
url: http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf
priority: Good-To-Read
read_on: 2026-08-27
tags: [paper, transformers, llm, vision, scaling]
---
## The Core Idea

Two ideas, one paper. Most people remember only the first.

**Idea one: stop hand-designing features.** In 1998 the standard recogniser was a fixed, hand-built feature extractor feeding a small trainable classifier. The accuracy of the whole system was capped by how clever the human was at inventing features, and the invention had to be redone for every new task. LeCun et al. argue that a network fed near-raw pixels can *learn* its own feature extractor by [[Backpropagation]], if — and this is the crucial qualifier — you build the right prior into the architecture. Fully connected nets on images are wasteful (a 28×28 input with 100 hidden units is already ~80,000 weights), have no built-in tolerance to shifts, and completely ignore that pixels near each other are correlated. The fix is three architectural constraints: **local receptive fields**, **shared weights**, and **subsampling**. LeNet-5 has 340,908 connections but only ~60,000 free parameters because of sharing. That is the whole trick: bake translation invariance and locality into the wiring so the learner does not have to spend data discovering them.

**Idea two: train the whole document pipeline, not just the classifier.** A real check reader is a field locator → segmenter → character recogniser → language model. Normally each module is tuned separately and glued together by hand. LeCun's claim is that if every module is differentiable *and* the thing passed between modules is a graph with numbers on the arcs (not a fixed-size vector), you can backpropagate a single global loss through the entire pipeline. These are **Graph Transformer Networks (GTNs)**. This unlocks something specific and valuable: you never have to hand-label where one character ends and the next begins. You label the string `"345"` and the system figures out the segmentation itself, because bad segmentations get pushed down by the same gradient that pushes the right answer up.

> [!NOTE] Weight sharing
> All units in a feature map use the *same* kernel, so they detect the same feature at every location. Consequence: shift the input, and the feature map shifts by the same amount, unchanged otherwise. This is where shift-robustness comes from, and it also slashes the parameter count (i.e. the capacity), narrowing the train/test gap. ^weight-sharing

> [!NOTE] Graph Transformer Network
> A module network where the state passed between modules is a directed acyclic graph whose arcs carry numeric penalties (and labels, images, whatever). Each path is one hypothesis about the input. Gradients flow with respect to the *numbers on the arcs*, so discrete-looking operations like Viterbi are still trainable. ^gtn

## The Methodology

### LeNet-5, layer by layer

Input is 32×32, bigger than the 20×20 digit centred in a 28×28 field, so that stroke endpoints can land in the *centre* of a top-layer receptive field. Pixels are scaled so background = −0.1 and foreground = 1.175, giving mean ≈ 0 and variance ≈ 1 — the same "normalise your inputs" motive later formalised by [[Batch Normalization]].

| Layer | What | Size | Trainable params | Connections |
|---|---|---|---|---|
| C1 | conv, 5×5 | 6@28×28 | 156 | 122,304 |
| S2 | subsample, 2×2 | 6@14×14 | 12 | 5,880 |
| C3 | conv, 5×5, **sparse** | 16@10×10 | 1,516 | 151,600 |
| S4 | subsample, 2×2 | 16@5×5 | 32 | 2,000 |
| C5 | conv, 5×5 (full) | 120@1×1 | 48,120 | 48,120 |
| F6 | fully connected | 84 | 10,164 | — |
| Out | Euclidean RBF | 10 | fixed | — |

Two details people usually get wrong when re-implementing this:

**Subsampling is not max-pooling.** Each S-unit *averages* its four inputs, multiplies by one trainable coefficient, adds one trainable bias, and passes through a sigmoid. If the coefficient is small the layer just blurs; if large, it acts like a noisy-OR or noisy-AND depending on the bias.

**C3 is deliberately not fully connected to S2.** Table I gives the map: the first six C3 maps see contiguous triples of S2 maps, the next six see contiguous quadruples, three see discontiguous quadruples, and the last sees all six. Reason given: keep connections bounded, but *more importantly* break symmetry, so different maps are forced to compute different things because they see different inputs.

The squashing function is $f(a) = A\tanh(Sa)$ with $A = 1.7159$, $S = 2/3$, chosen so $f(1)=1$ and $f(-1)=-1$, and so the second derivative is maximal at ±1. Weights initialised uniform in $[-2.4/F_i,\ 2.4/F_i]$ where $F_i$ is the fan-in — a direct ancestor of Xavier init.

### The output layer and the loss

There is no softmax. The 10 outputs are **Euclidean RBF units**:

$$y_i = \sum_j (x_j - w_{ij})^2$$

i.e. squared distance from F6's 84-dim state to a fixed target vector. The 84 comes from a 7×12 bitmap: each class's target is a *stylised picture of the character*, components set to ±1. So confusable classes (uppercase O, lowercase o, zero) get similar codes, which helps a downstream language model disambiguate. Because ±1 are the max-curvature points of the F6 sigmoids, this also keeps F6 out of saturation.

The naive loss is just the correct class's RBF output:
$$E(W) = \frac{1}{P}\sum_{p=1}^{P} y_{D_p}(Z^p, W)$$

This has no competition between classes and, if RBF centres are trainable, has a **collapse** solution: all centres equal, F6 constant, all outputs zero, input ignored. The fix is a MAP-style discriminative loss that also pushes *up* the wrong classes:
$$E(W)=\frac{1}{P}\sum_p \Big( y_{D_p}(Z^p,W) + \log\big(e^{-j} + \textstyle\sum_i e^{-y_i(Z^p,W)}\big)\Big)$$
The constant $j$ is a "rubbish class" penalty that stops the system wasting effort pushing up already-huge penalties. Note the framing throughout: everything is a **penalty**, not a probability. Normalisation is postponed to the final decision. This is the seed of [[A Tutorial on Energy-Based Learning]].

### Optimisation

Stochastic gradient, 20 passes over 60,000 patterns. Global learning rate schedule: 0.0005 (2 passes), 0.0002 (3), 0.0001 (3), 0.00005 (4), 0.00001 (rest). Per-parameter step size from **stochastic diagonal Levenberg–Marquardt**:
$$\eta_k = \frac{\epsilon}{\mu + h_{kk}}$$
with $\mu = 0.02$ and $h_{kk}$ a Gauss–Newton diagonal estimate of the second derivative, re-estimated on just 500 samples before each epoch. Effective learning rates spanned $7\times10^{-5}$ to $0.016$ across parameters. The Gauss–Newton backward pass drops the $f''$ term so estimates stay non-negative:
$$\frac{\partial^2 E_p}{\partial a_i^2} = f'(a_i)^2 \sum_k u_{ki}^2 \frac{\partial^2 E_p}{\partial a_k^2}$$
Weight sharing makes the loss surface badly conditioned — one early-layer parameter influences everything — and this per-parameter scaling is the compensation. Gradients for shared weights are computed per-connection then summed, which is exactly what [[Pytorch Autograd]] does today.

### MNIST, which this paper invented

NIST's SD-3 (Census Bureau employees) was much cleaner than SD-1 (high schoolers), and NIST had used them as train/test. That is a broken split. LeCun et al. unscrambled SD-1 by writer, put writers 1–250 in train and 251–500 in test, then topped both up from SD-3 to 60,000 each. Digits size-normalised to fit a 20×20 box (grey levels appear from anti-aliasing), centred by centre of mass in 28×28. Two variants: *regular* and *deslanted* (sheared so the principal axis of inertia is vertical).

### The GTN half

For strings, **heuristic over-segmentation** generates far more candidate cuts than needed (minima of the vertical projection profile; a "hit and deflect" procedure that drops lines from the top and follows contours). Alternative segmentations become a DAG. The **recognition transformer** replaces each arc with one arc per class, carrying the recogniser's penalty. The **Viterbi transformer** finds the min-penalty path:
$$v_n = \min_{i \in U_n} (c_i + v_{s_i})$$

Backprop through Viterbi is legal because Viterbi is just `min` and `+` glued together, and `min` is differentiable except on a measure-zero set. Gradient is 1 on arcs in the winning path, 0 elsewhere.

Four training losses were tried, in increasing order of quality:

1. **Viterbi training**: $E_{vit} = C_{cvit}$, the penalty of the best path *constrained* to spell the correct label sequence. Suffers collapse and gives no usable confidence.
2. **Discriminative Viterbi**: $E_{dvit} = C_{cvit} - C_{vit}$. Now wrong paths get pushed up. But the gradient hits exactly zero the moment the correct path merely *ties* the best path — no margin.
3. **Forward training**: replace `min` with `logadd`,
$$f_n = \text{logadd}_{i \in U_n}(c_i + f_{s_i}), \qquad \text{logadd}(x_1..x_n) = -\log\sum_i e^{-x_i}$$
so *all* paths spelling the right answer get credit, weighted by how good they are. Same collapse problem though.
4. **Discriminative forward** (the one they ship): $E_{dforw} = C_{cforw} - C_{forw}$. Always ≥ 0, zero when the correct interpretations carry essentially all the probability mass. The authors note the direct analogy to the clamped and free phases of a Boltzmann machine. Backward pass:
$$\frac{\partial E}{\partial c_i} = \frac{\partial E}{\partial f_{d_i}} e^{-c_i - f_{s_i} + f_{d_i}}$$
A soft version of Viterbi backprop: every arc gets gradient, low-penalty arcs get more. Training automatically concentrates on the images that cause errors and on the *pieces of the image* causing the error.

## Ablation Studies and Experiments

### MNIST test error, everything on the same data

| Method | Test error |
|---|---|
| Linear classifier | 12.0% |
| Linear, deslanted | 8.4% |
| Pairwise linear (45 one-vs-one units) | 7.6% |
| K-NN Euclidean | 5.0% |
| K-NN, deslanted, k=3 | 2.4% |
| 40 PCA + quadratic | 3.3% |
| 1000 RBF + linear | 3.6% |
| MLP 784-300-10 | 4.7% |
| MLP 784-1000-10 | 4.5% |
| MLP 784-300-100-10 | 3.05% |
| MLP 784-1000-150-10 | 2.95% |
| MLP 300-100-10, + distortions | 2.50% |
| MLP 20×20-300-10, deslanted | 1.6% |
| Tangent Distance (16×16) | 1.1% |
| SVM poly-4 | 1.1% |
| Reduced-Set SVM poly-5 | 1.1% |
| Virtual SVM poly-9, + distortions | 0.8% |
| LeNet-1 (16×16 input, ~2,600 params) | 1.7% |
| LeNet-4 (~17,000 params) | 1.1% |
| **LeNet-5** | **0.95%** |
| **LeNet-5 + distortions** | **0.80%** |
| **Boosted LeNet-4 + distortions** | **0.70%** |

The one number that should stick: **LeNet-1, with ~2,600 free parameters, beats a 1000-hidden-unit MLP with ~800,000.** That is the architecture prior paying rent, not scale.

### What the ablations actually show

**More data helps, and distortions are a cheap substitute.** Training on 15k / 30k / 60k gives a curve still sloping down at 60k. Adding 540,000 affinely distorted copies (translate, scale, squeeze, shear) took LeNet-5 from 0.95% → 0.80%, over the same 20 passes — so the network effectively sees each real sample only twice.

**No overfitting was observed.** Test error flattens after ~10 passes and never turns up. Their explanation: the learning rate was kept large, weights never settle into a narrow minimum, and SGD noise acts like a regulariser favouring *broad* minima. This is a 1998 statement of an argument still being litigated — see [[Understanding Deep Learning Requires Rethinking Generalization]] and [[Reconciling Modern ML Practice and the Bias-Variance Trade-off]].

**They also observed early implicit regularisation in MLPs**: the origin of weight space is an attractive saddle, so weights shrink first, sigmoids stay quasi-linear, and the network behaves like a low-capacity linear model until the weights grow. They call this "an almost perfect, if fortuitous, implementation of structural risk minimisation."

**Two-hidden-layer beats one-hidden-layer** at matched parameter count (3.05% vs 4.7%), despite universal-approximation theory saying one layer suffices.

### Things that did **not** work

- **LeNet-4 + K-NN on the penultimate layer**, and **LeNet-4 + Bottou–Vapnik local learning**: neither improved raw error at all. They only improved *rejection* performance (fewer rejects needed to hit 0.5% error).
- **Distortions barely help MLPs**: 300-hidden went 4.7% → 3.6%, 1000-hidden 4.5% → 3.8%. Deslanting helped the MLP far more (1.6%). Data augmentation pays off much more for the architecture that can exploit it.
- **Second-order batch optimisers are useless here.** Appendix B is blunt: Gauss–Newton and Levenberg–Marquardt are $O(N^3)$ per update, quasi-Newton $O(N^2)$, and even the $O(N)$ methods (conjugate gradient, L-BFGS) need accurate conjugate directions, which only makes sense in batch mode. On large redundant datasets, nothing they tried beat a well-tuned stochastic gradient. Relevant background for [[Adam- A Method for Stochastic Optimization]] and [[Momentum]].
- **SDNN never beat heuristic over-segmentation.** Sweeping a replicated convnet across the whole word (a Space Displacement Neural Network) is elegant, handles interlocked characters that over-segmentation cannot cut, and correctly groups disconnected ink — the paper shows it separating a "4" and "0" that touch, and reassembling a broken "4". Yet: "so far it has not yielded better results than Heuristic Over-Segmentation." A rare, honest negative result about the more beautiful method.
- **SVMs are architecture-blind and expensive.** Poly-4 SVM matches LeNet-4 at 1.1% *and would do just as well if you permuted the pixels* — which the authors flag as remarkable. Cost: ~14 million multiply-adds per digit versus ~401k for LeNet-5, and ~24,000 stored variables versus ~60 thousand... actually the memory comparison favours SVM less: nearest-neighbour needs ~24MB, LeNet-5 ~60k variables at roughly a byte each.
- **Boosting is cheaper than it looks.** Three LeNet-4s, each trained on what the previous ones got wrong, outputs summed: 0.7%. Naively 3× the cost, but if the first net answers confidently the others are never called — average 1.75× a single net.

### Global training, measured

Online handwriting (writer-independent, 881 test words):

| Setting | Char error before global training | After |
|---|---|---|
| SDNN/HMM, no lexicon | 12.4% | **8.2%** |
| Over-segmentation, no lexicon | 8.5% | **6.3%** |
| Over-segmentation, 25,461-word lexicon | 2.0% | **1.4%** |

Word errors dropped 24–32% relative in every configuration. Separately, normalising the **whole word** instead of each segment cut word error 7.3% → 4.6% and char error 3.5% → 2.0%.

The deployed check reader: **82% correct / 1% error / 17% reject** on 646 machine-print business checks, against the previous system's 68% / 1% / 31%. The bank's economic threshold was 50% correct at <1% error. Credited to three things, in order: a bigger recogniser trained on more data; the GTN letting grammar constraints be used properly; and — the authors stress this — the framework separating algorithmic machinery from task heuristics made experimentation fast. Global training itself touched only a small subset of parameters here and contributed least.

## Worth Remembering

- **Confidence for free.** Because $\exp(-E_{dforw})$ is an estimate of the posterior of the chosen label sequence, you get a rejection score by re-running the discriminative forward loss with the *Viterbi answer* as the desired sequence. No extra model.

- **Why penalties, not probabilities.** The authors deliberately avoid locally normalised scores. Normalising per-character destroys the ability to reject a segment as *not a character at all* (all classes should be bad simultaneously, which a softmax cannot express). And enforcing normalisation during gradient training is "complex, inefficient, time consuming, and creates ill-conditioning of the loss function." This argument is the direct ancestor of energy-based modelling — see [[A Tutorial on Energy-Based Learning]].

- **Collapse is a real failure mode**, not a theoretical one. Any non-discriminative sequence loss over a trainable recogniser can be minimised by ignoring the input. Fixed RBF centres prevent full collapse but leave an attractive flat saddle. HMMs escape it only because their generative normalisation forces probabilities to trade off. If you are training a sequence model with a "score the correct path" loss and nothing pushing wrong paths up, you should expect this.

- **The pooling that isn't.** Modern reimplementations of "LeNet-5" almost always use max-pooling and softmax + [[Cross Entropy]]. The original has trainable-coefficient average subsampling and squared-distance RBF outputs. Worth knowing if you compare parameter counts against a paper.

- **The 84-unit distributed output code** is a genuinely interesting idea that mostly died: representing each class as a bitmap so that visually confusable classes have nearby codes, letting a downstream language model resolve them. It also gives better rejection than one-hot, because RBFs only fire in a bounded region while sigmoids are on everywhere outside their off-region.

- **Where this lands historically.** Everything here — learned features beating hand-crafted ones, more data always helping, architecture priors that encode invariance — is the thesis of [[The Bitter Lesson (essay)]] arriving 20 years early with a check reader attached. AlexNet ([[ImageNet Classification with Deep CNNs (AlexNet)]]) is essentially this architecture with ReLU, dropout, and GPUs; [[Deep Residual Learning for Image Recognition (ResNet)]] fixes the depth problem; [[An Image is Worth 16x16 Words (ViT)]] argues the local-receptive-field prior can be dropped once you have enough data.

- **The forgotten half.** The GTN material is the intellectually harder contribution and almost nobody reads it. But "make the pipeline differentiable end to end and train one global loss" is now the default in speech, OCR, and detection. Note also the sharp remark that "training time is of little interest to the final user" — LeNet-5 took 2–3 days on a single 200MHz R10000.

- **Practical caveat.** The stochastic diagonal Levenberg–Marquardt trick exists specifically because weight sharing wrecks conditioning. Modern practice solves the same problem with adaptive optimisers and normalisation layers. You do not need to implement it, but understanding *why* it was needed explains why plain SGD on a naive convnet is fragile.

Open questions the paper leaves: why local minima are not a problem (they only conjecture "extra dimensions in an oversized network"); why SDNN, which is strictly more general, loses to a heuristic segmenter; and whether a graph transformer can learn the *structure* of its output graph rather than just the numbers on it.

## Links

Related: [[Deep Learning]] · [[Backpropagation]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[A Tutorial on Energy-Based Learning]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Understanding Deep Learning Requires Rethinking Generalization]] · [[Reconciling Modern ML Practice and the Bias-Variance Trade-off]] · [[Batch Normalization]] · [[Derivative]] · [[Vector Jacobian Product]] · [[Pytorch Autograd]] · [[Adam- A Method for Stochastic Optimization]] · [[Regularization]] · [[Cross Entropy]] · [[The Bitter Lesson (essay)]] · [[Markov Property]]

New topics worth writing: Max pooling vs average subsampling, Hidden Markov Models, Viterbi algorithm and dynamic programming, Weighted finite-state transducers, Levenberg–Marquardt and Gauss–Newton approximations, AdaBoost and boosting, Tangent distance, MNIST dataset construction, Receptive field arithmetic, Neocognitron, Xavier/He initialisation, Data augmentation via affine distortions, CTC loss (the modern descendant of discriminative forward training)
