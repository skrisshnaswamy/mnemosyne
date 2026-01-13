---
title: Scaling Laws for Neural Language Models
authors:
  - Jared Kaplan
  - Sam McCandlish
  - Tom Henighan
  - Tom B. Brown
  - Benjamin Chess
  - Rewon Child
  - Scott Gray
  - Alec Radford
  - Dario Amodei
year: 2020
arxiv: "2001.08361"
url: https://arxiv.org/abs/2001.08361
priority: Must-Read
read_on: 2026-08-21
tags:
  - paper
  - transformers
  - llm
  - optimization
  - scaling
  - GPT3
---
>[!ABSTRACT] 
>The "Scaling Laws for Neural Language Models" paper by **OpenAI** is famous because ==it mathematically proved that bigger models are systematically and predictably better==, providing the literal engineering blueprint that justified OpenAI building GPT-3.
>
>While GPT-3 was indeed a massive parameter count hike over GPT-1 and GPT-2, OpenAI did not just arbitrarily pick a larger number. The 2020 scaling laws paper gave them the exact mathematical confidence to invest millions of dollars into training a 175-billion-parameter model.
>
>#### What the Paper is Famous For
>
>- Predictable Performance: It showed that cross-entropy loss (error rate) scales as a power-law with model size, dataset size, and compute. This meant researchers could train a tiny model for a few dollars and accurately predict exactly how well a massive model would perform before spending millions.
>- The "More is Better" Philosophy: It argued that architectural details (like network width vs. depth) barely mattered. The only things that truly drove performance were three variables: compute, dataset size, and parameters.
>- Overemphasizing Model Size: The paper concluded that if your budget increases, you should allocate 73% of it to making the model bigger and only 27% to adding more text data.
>
>#### The Direct Link to GPT-3
>
>Because of this specific 73/27 math, OpenAI concluded that the most efficient way to use their budget was to make the model parameters massive, even if they didn't have enough data to train it fully.
>
>This directly resulted in GPT-3 (175B parameters) being trained on only 300 billion tokens—a ratio that the later Chinchilla paper proved was severely undertrained, but one that completely shifted the AI industry toward the "brute-force" scaling paradigm.

---

## The Core Idea

Language model quality is not a mysterious art. It is a straight line on a log-log plot.

Train a decoder-only Transformer to predict the next token. Measure the test [[Cross Entropy|cross-entropy]] loss in nats. That loss falls as a **power law** in three things: the number of non-embedding parameters $N$, the number of training tokens $D$, and the amount of training compute $C$. Each relation holds over six to eight orders of magnitude with no bend at the top end.

$$L(N) = \left(\frac{8.8\times10^{13}}{N}\right)^{0.076} \qquad L(D) = \left(\frac{5.4\times10^{13}}{D}\right)^{0.095} \qquad L(C_{\min}) = \left(\frac{3.1\times10^{8}}{C_{\min}}\right)^{0.050}$$

> [!NOTE] Power law
> A relation of the form $y = (a/x)^{\alpha}$. On a log-log plot it is a straight line with slope $-\alpha$. The exponent is what matters: it tells you what fraction of loss you buy per doubling. Here $2^{-0.076}=0.95$, so doubling the model shaves 5% off the loss — forever, as far as the data shows. ^power-law

The second, sharper claim: **model shape barely matters**. Hold $N$ fixed and vary depth, width, number of heads, feed-forward width — the loss moves by a few percent. A 6-layer model with $d_{\text{model}}=4288$ lands within 3% of the 48-layer, $d_{\text{model}}=1600$ GPT-2 shape. Aspect ratio can vary by 40× and nothing much happens. The architecture debates of 2015–2019 were, at this level of measurement, noise. Scale is the signal.

The third claim is the one that changed how labs spend money. Given a fixed compute budget and no other constraint, the best thing to do is **train a very large model and stop long before it converges**. Not train a small model to convergence. Large models are dramatically more sample-efficient — they hit any given loss in fewer optimisation steps and fewer tokens. So most of a compute increase should go into $N$:

$$N \propto C_{\min}^{0.73}, \qquad B \propto C_{\min}^{0.24}, \qquad S \propto C_{\min}^{0.03}, \qquad D \propto C_{\min}^{0.27}$$

A billion-fold increase in compute buys you a ~5×10⁷-fold larger model, a much bigger batch, and almost no extra serial steps.

Why did this not exist before? Nobody had run the sweep. It required training hundreds of models from 768 parameters to 1.5B on a fixed pipeline, plus the earlier gradient-noise-scale work that let them normalise away batch-size effects. What it unlocks: you can now *predict* the loss of a model you have not trained, and budget a run before you buy the GPUs. GPT-3 was, in a real sense, an extrapolation of Figure 1.

## The Methodology

**Model.** Decoder-only Transformer ([[Attention Is All You Need]] architecture, no encoder), context $n_{\text{ctx}}=1024$, byte-pair-encoded vocab of 50,257. Compare [[Improving Language Understanding by Generative Pre-Training (GPT-1)#^decoder-only]].

**The key accounting trick — exclude embeddings.** Model size is defined as

$$N \approx 2 d_{\text{model}} n_{\text{layer}} (2 d_{\text{attn}} + d_{\text{ff}}) = 12 n_{\text{layer}} d_{\text{model}}^2$$

The $n_{\text{vocab}} d_{\text{model}}$ embedding matrix and $n_{\text{ctx}} d_{\text{model}}$ positional embeddings are **left out**. This is not cosmetic. Figure 6 shows that with embeddings included, models of different depth sit on separate curves and the trend is muddy; exclude them and every depth collapses onto one line. The implication is that embedding parameters are cheap filler — you can shrink them without hurting loss (which is exactly what ALBERT did).

**Compute accounting.** Forward pass costs $\approx 2N$ FLOPs per token (the 2 is multiply-accumulate). Backward is roughly twice forward. So

$$C \approx 6NBS$$

with $B$ the batch in tokens and $S$ the number of parameter updates. Context-dependent attention terms ($2 n_{\text{layer}} n_{\text{ctx}} d_{\text{attn}}$) are dropped because $d_{\text{model}} \gg n_{\text{ctx}}/12$ for the models studied. Compute is quoted in PF-days $= 8.64\times10^{19}$ FLOPs.

**Data.** WebText2 — outbound Reddit links with ≥3 karma, scraped through October 2018. 20.3M documents, 96 GB, $2.29\times10^{10}$ tokens, with $6.6\times10^8$ held out. Also evaluated (never trained) on Books Corpus, Common Crawl, Wikipedia, and internet books.

**Training.** [[Momentum|Adam]], $2.5\times10^5$ steps, batch of 512 sequences × 1024 tokens = $2^{19}$ tokens. Models over 1B params used Adafactor for memory. 3000-step linear warmup then cosine decay to zero. Learning rate set by a rule of thumb $\text{LR}(N) \approx 0.003239 - 0.0001395\log N$ — bigger models need smaller LR or they diverge. 10% dropout for the $L(N,D)$ overfitting sweep.

**The joint law for overfitting.** Vary $N$ and $D$ together, early-stop on test loss, and one equation fits everything:

$$L(N,D) = \left[\left(\frac{N_c}{N}\right)^{\alpha_N/\alpha_D} + \frac{D_c}{D}\right]^{\alpha_D}$$

with $\alpha_N = 0.076$, $\alpha_D = 0.103$, $N_c = 6.4\times10^{13}$, $D_c = 1.8\times10^{13}$. Three principles motivated the form: (1) changing the tokeniser should just rescale the loss, so $N_c, D_c$ must be free scale constants with no fundamental meaning; (2) $N\to\infty$ must recover $L(D)$ and $D\to\infty$ must recover $L(N)$; (3) the loss should be analytic in $1/D$ at $D=\infty$, because overfitting tracks the variance of a finite sample, which goes as $1/D$. Principle 3 is the shakiest and is what breaks the symmetry between $N$ and $D$ in the formula.

The practical output: overfitting depends only on the ratio $N^{0.74}/D$. To keep the penalty below the ~0.02 run-to-run noise floor you need

$$D \gtrsim (5\times10^3)\, N^{0.74}$$

**Sub-linear.** Grow the model 8×, grow the data ~5×. (Note for later: [[Language Models are Few-Shot Learners (GPT-3)|GPT-3]] followed this. Chinchilla later argued the exponent should be ~1.0, i.e. data and params scale together — see caveats below.)

**Batch size and the step-count normalisation.** Most runs used a fixed $B = 2^{19}$ tokens, which is not optimal for every loss level. They correct for this using the critical batch size from the gradient-noise-scale paper. Empirically, for a target loss $L$, steps $S$ and examples $E=BS$ obey

$$\left(\frac{S}{S_{\min}} - 1\right)\left(\frac{E}{E_{\min}} - 1\right) = 1, \qquad B_{\text{crit}}(L) \equiv \frac{E_{\min}}{S_{\min}} = \frac{B_*}{L^{1/\alpha_B}}$$

with $B_* \approx 2\times10^8$ tokens, $\alpha_B \approx 0.21$. Critical batch size **doubles for every 13% drop in loss** and does not depend on model size at all — only on the loss reached. Roughly 1–2M tokens at convergence for the largest models. This lets them define $S_{\min}(S) = S/(1 + B_{\text{crit}}/B)$ and $C_{\min}(C) = C/(1 + B/B_{\text{crit}})$, which is what makes the compute trends clean.

**Learning curves.** In the infinite-data limit, after warmup,

$$L(N, S_{\min}) = \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{S_c}{S_{\min}}\right)^{\alpha_S}$$

with $\alpha_S = 0.76$, $S_c = 2.1\times10^3$. Note this one is a **sum**, not a product-like blend: a capacity term plus an optimisation term. You can fit the early part of a training curve and predict where it lands much later.

**Deriving the compute allocation.** Substitute $S_{\min} = C_{\min}/(6NB)$ into the equation above and minimise over $N$ at fixed $C$. Two things fall out. First,

$$\alpha_C^{\min} = \frac{1}{1/\alpha_S + 1/\alpha_B + 1/\alpha_N} \approx 0.054$$

which matches the measured 0.050. Second, at the optimum the loss sits a **fixed 10% above the converged loss**, since $\alpha_N/\alpha_S \approx 0.10$. That is the whole "stop early" result in one number. And $N(C_{\min}) \propto C_{\min}^{0.71}$ predicted vs $0.73$ measured.

## Ablation Studies and Experiments

**Shape sweep (Figure 5).** Fix $N$, vary one shape hyperparameter, compensate with $d_{\text{model}}$ to hold $N \approx 12 n_{\text{layer}} d_{\text{model}}^2$. Varied $n_{\text{heads}}$ freely, $n_{\text{layer}}$ from 6 to 207, $d_{\text{ff}}$. Loss moves only a few percent across a 40× range of aspect ratio. **Nothing about shape does the work.** The exceptions: models with fewer than 2 layers, or truly extreme depth-to-width ratios, fall off the trend.

**Including vs excluding embeddings (Figure 6).** With embeddings in the count, loss looks like it depends on depth on top of parameter count. Take them out and all depths collapse to one curve. This is the single most important measurement hygiene choice in the paper.

**Functional form (Figure 23).** They tried logarithms and other forms for $L(N)$, $L(C)$, $L(D)$. Power laws fit qualitatively much better. Not a subtle win.

**Learning rate schedules (Figure 22).** Cosine decay, linear decay, faster and slower decays, all on a 3M-parameter model. **Choice of schedule is essentially irrelevant** provided (a) the total summed LR is large enough, (b) there is a warmup, (c) it decays to near zero at the end. Differences between schedules are the same size as the run-to-run seed noise, ~0.05 in loss for small models, ~0.02 for the larger sweeps. This is a genuinely useful negative result — do not spend a week tuning the schedule.

**LSTMs vs Transformers (Figure 7).** Same data, same context length, plotted against $N$. LSTMs match Transformers on tokens **early** in the context but lose badly on later tokens. The Transformer advantage is specifically about using long-range context. Related: [[Causal Attention]].

**Universal / recurrent Transformers (Figure 17).** Parameter reuse means they do slightly *better* per parameter and slightly *worse* per FLOP. A wash, and it confirms $N$ is the right axis only when you also account for compute.

**Where the fit fails.** The $L(N,D)$ fit is excellent everywhere **except** the smallest dataset, $\approx 2\times10^7$ tokens (WebText2 cut by 1024×), where one epoch is only 40 parameter updates. Overfitting happens almost immediately there and the law does not describe it. The authors admit they never properly explored the small-data regime and never tuned dropout or augmentation as $N$ and $D$ changed.

**1-layer models are excluded from the $L(C)$ and $L(N)$ fits.** There is a visible lump at $10^{-5}$ PF-days where the 1→2 layer transition happens.

**Sub-optimal model sizes (Appendix B.4).** Anything between 0.6× and 2.2× the compute-optimal size costs only 20% extra compute. A 2.2× oversized model reaches the same loss in 45% fewer steps for that 20% compute premium — which is the right trade if you have the parallelism. And going *under*-sized is the right trade if you care about inference cost.

**Efficient vs conventional training (Appendix B.3).** Training to 10% above converged loss (efficient) versus 2% (what people actually did): the efficient recipe uses **2.7× more parameters, 7.7× fewer parameter updates, and 65% less total compute** to reach the same loss.

**Transfer to other distributions (Figure 8).** Trained only on WebText2, evaluated on Books, Common Crawl, Wikipedia, internet books. Loss on each improves as a power law in $N$ with **nearly identical exponent**, offset by a constant. Generalisation depends only on the in-distribution validation loss — not on how long you trained, not on whether you converged, not on depth. Getting better on your training distribution is getting better everywhere, at a fixed tax.

**Per-token position (Figures 20, 21).** Loss at position $T$ in the 1024-token context is itself a power law in $T$. Models trained with a tiny $n_{\text{ctx}}=8$ beat the largest 1024-context models on the first few tokens — they spend all capacity there. During training, a model learns short-range structure first and long-range structure later.

## Worth Remembering

**The self-contradiction the authors flag (Section 6.3).** Extrapolate the two laws far enough and they cross. Compute-efficient training grows data as $D \propto C_{\min}^{0.27}$, but avoiding overfitting requires $D \propto N^{0.74} \propto C_{\min}^{0.54}$. The compute-optimal curve $L(C_{\min}) \propto C_{\min}^{-0.050}$ eventually dives below the data-limited floor $L(D(C_{\min})) \propto C_{\min}^{-0.03}$. They meet at

$$C^* \sim 10^4 \text{ PF-days}, \quad N^* \sim 10^{12}, \quad D^* \sim 10^{12} \text{ tokens}, \quad L^* \sim 1.7 \text{ nats/token}$$

The laws must break before that. The speculative reading: $L^*$ is roughly the entropy per token of natural language, the point where a Transformer has extracted all reliable information from text. The numbers are uncertain by an order of magnitude in either direction — they are exponentially sensitive to the fitted $\alpha$'s.

**The exponent that later turned out wrong.** $D \propto N^{0.74}$ came from an overfitting analysis at fixed batch size with only two orders of magnitude of $D$ variation, and the batch size was *not* varied along with $D$. Chinchilla (Hoffmann et al., 2022) re-ran this with the learning-rate schedule matched to each run's length and found $N \propto C^{0.5}$, $D \propto C^{0.5}$ — data and parameters should scale *together*. GPT-3 (175B params, 300B tokens) is badly undertrained by the later standard. The likely culprit: Kaplan et al. used a fixed cosine schedule length across runs of different lengths, which systematically penalises the shorter (data-heavy) runs. Read this paper knowing its headline allocation is superseded, but its *framework* — power laws, embedding exclusion, $C=6ND$, critical batch size — is not.

**Caveats the authors list themselves.** No theory for any of it; the $N$ and $C$ scalings are "especially mysterious." No confidence in $B_{\text{crit}}(L)$ outside the measured loss range. $C \approx 6NBS$ ignores attention terms and will mislead when $n_{\text{ctx}} \gtrsim 12 d_{\text{model}}$ — which is exactly the long-context regime everyone moved to later. They may have failed to tune some hyperparameter (initialisation scale, momentum) that matters for scaling. They never tried a *higher* learning rate for the short, non-converged runs that the compute-optimal recipe calls for.

**The "ideal gas law" analogy.** The authors' own framing: these relations connect macroscopic quantities ($N$, $D$, $C$, $L$) universally, without caring about the microscopic details (which head attends where, how deep the stack is). They ask for the "statistical mechanics underneath the thermodynamics." Fifteen-plus years of that theory work is still open.

**"More is different."** The final discussion point, and the one that aged best. Smooth loss improvement can hide sharp capability jumps. Nothing in a straight log-log line tells you that in-context learning appears at some scale. That paper came seven months later — [[In Context Learning]], [[Language Models are Few-Shot Learners (GPT-3)]].

**Practical takeaways if you are budgeting a run today.** Count non-embedding params. Use $C = 6ND$ FLOPs. Do not obsess over depth/width. Do not obsess over the LR schedule shape — do warm up and do decay to near zero. Do scale batch size with the gradient noise scale, not by feel. And check whether your compute-optimal point is Kaplan's or Chinchilla's before you commit money.

## Links

Related: [[Attention Is All You Need]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Improving Language Understanding by Generative Pre-Training (GPT-1)]] · [[BERT- Pre-training of Deep Bidirectional Transformers]] · [[Cross Entropy]] · [[Foundation Models]] · [[In Context Learning]] · [[Causal Attention]] · [[Regularization]] · [[Momentum]] · [[Derivative#Hessian|Hessian]] · [[Auto-regressive models]] · [[Mixed Precision training]] · [[Uncertainty]]

New topics worth writing: Chinchilla compute-optimal scaling, Gradient noise scale and critical batch size, Adafactor, Emergent abilities and discontinuous scaling, Neural tangent kernel, Byte-pair encoding, Noisy quadratic model of optimisation, Data-constrained scaling and repeated epochs
