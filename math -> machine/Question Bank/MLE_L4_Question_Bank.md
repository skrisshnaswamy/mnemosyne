
> [!TIP] **Built for:** Parvathy Krishnaswamy — representation learning, ranking, tabular/sequence modeling, distributed training, experimentation. Target: Senior MLE / Applied ML (one below Lead IC).
>
>**What this is.** A working reference, not a cram sheet. It's built to serve three different modes:
>
>1. **Interview prep** — answers are written to be *spoken*. Read one out loud; if it sounds recited, cut the second half. The second half is usually the "if they push, go here" material.
>2. **Team discussion** — the trade-off paragraphs are the ones you'll actually use in design reviews. They're written to be defensible, not just correct.
>3. **Refresher while reading papers** — when a paper assumes you remember what logQ correction or dimensional collapse or NTK scaling is, the relevant entry is here with enough context to reload it in thirty seconds.
>
>**How answers are structured.** Every one follows the same shape: **plain-language core → why it matters → what I'd actually do → the trade-off.** That last part is the one that separates levels. A mid-level engineer says what a thing is. A senior one says what it costs. When you add to this bank later, keep that shape.
>
>**Conventions:**
>- Jargon is always followed by a one-line plain meaning in parentheses. Keep that habit when you talk too — it reads as mastery, not as dumbing down.
>- `↪ Your hook:` marks a question that maps to something on your CV, plus the follow-up an interviewer will ask.
>- Where the field's consensus has shifted or is contested, that's said explicitly rather than papered over. Those are the highest-value things to know and the first to go stale — check them when you revisit.
>
>**Maintenance.** Content ages. Parts 1-11 and 21-22 are stable for years. Parts 18-20 and 26 (compression, fine-tuning, inference, generative) turn over every 12-18 months — assume anything version-specific there needs re-checking. See the closing section for a maintenance loop.


>[!ABSTRACT] **Contents**
>
>*Foundations*
>1. Gradients & Optimization
>2. Regularization & Normalization
>3. Gates & Sequence Models
>4. Attention & Transformer Architecture
>5. Positional Encoding
>6. Tokenization, Vocabulary & Embedding Tables
>
>*Representations*
>
>1. Extracting, Storing & Serving Embeddings
>2. Collapse, Anisotropy & Contrastive Learning
>3. Dimensionality Reduction
>
>*Applied Systems*
>
>1. Ranking & Evaluation
>2. Distributed Training & Systems
>3. Tabular Deep Learning & Foundation Models
>4. Failure Modes, Anti-Patterns & Debugging
>5. Bandits, Uncertainty & the Decision Intelligence Bridge
>6. System Design & Behavioral
>
>*Modeling Depth*
>
>1. Loss Functions
>2. Retrieval & Two-Tower Systems
>3. Model Compression
>4. Fine-Tuning & Adaptation
>5. Inference Optimization
>
>*Fundamentals & Craft*
>
>1. Probability & Statistics for MLEs
>2. Classical ML Depth
>3. Coding Round Prep (with implementations)
>4. Case Study Drills
>5. Behavioral & Narrative
>6. Generative Systems & Modern LLM Practice
>7. Fairness, Privacy & Governance

*Closing: maintaining this bank*

---

## Part 1 — Gradients & Optimization

### Q1. What is a gradient, and what is backprop really doing?

A gradient is just a direction and a size. For every weight in the network it tells me: if I nudge this weight up a tiny bit, does the loss go up or down, and by how much. So I move each weight the opposite way, by a small step.

Backprop is nothing more than the chain rule applied backwards through the compute graph — *chain rule meaning if a affects b and b affects c, the effect of a on c is the product of the two.* The clever part isn't the math, it's the bookkeeping: it computes the gradient for every parameter in roughly the same cost as one forward pass, because it reuses the intermediate results going backward instead of recomputing them per parameter.

The practical thing I care about day to day is not the direction, it's the **magnitude**. Directions are usually fine. Magnitudes are what blow up, vanish, or oscillate — and almost every training bug I've debugged has been a magnitude problem wearing a costume.

---

### Q2. Explain vanishing gradients. Where do they come from and how do you fix them?

Vanishing gradients happen because backprop *multiplies*. To get the gradient at layer 1 of a 40-layer network, you multiply 40 numbers together. If each of those is around 0.8, you end up at roughly 0.0001 — the early layers get essentially no signal and just sit there while the late layers learn.

Three main causes:

1. **Saturating activations.** Sigmoid and tanh flatten out at the extremes — *saturate meaning the output stops changing even when the input changes a lot.* The derivative of sigmoid maxes out at 0.25, so you're multiplying by ≤0.25 per layer. That's fatal at depth.
2. **Bad initialization.** If weights start too small, the forward signal shrinks layer over layer, and so does the backward one.
3. **Long sequences in recurrent models.** Same problem but through *time* instead of depth — a 200-step sequence is a 200-term product.

What I actually reach for:

- **Non-saturating activations** — ReLU, GELU, SiLU. Their derivative is around 1 in the active region, so the product doesn't decay.
- **Residual connections.** This is the single biggest fix. A skip connection makes the layer compute `x + f(x)`, so the gradient has a path where it gets multiplied by 1 — it can flow to the bottom of the network untouched. This is exactly why we can train 100-layer nets now and couldn't in 2013.
- **Normalization layers** to keep activations in a sane range so nothing saturates.
- **Variance-preserving init** — He init for ReLU-family, Xavier/Glorot for tanh. The whole idea is to set the initial weight variance so signal variance stays roughly constant layer to layer.
- **Architectural**: gates in LSTM/GRU, or just switching to attention, which gives every position a direct path to every other position instead of a 200-step chain.

How I'd diagnose it: log the gradient norm per layer group, not just globally. If layer 1's grad norm is three orders of magnitude below layer 20's and stays there, that's vanishing, not "still warming up."

---

### Q3. And exploding gradients?

Same mechanism, opposite direction — the per-layer factors are above 1, so the product blows up. You see it as the loss going to NaN, or a sudden vertical spike in the loss curve, or weights suddenly becoming huge.

Common triggers: too high a learning rate, a bad batch with extreme values, a division-by-near-zero somewhere (log of zero, softmax over huge logits), fp16 overflow, or a poorly conditioned recurrent weight matrix.

The standard fix is **gradient clipping**, and there are two flavors and they are not equivalent:

- **Clip by value** — clamp each individual gradient component to `[-c, c]`. This *changes the direction* of the update, because you're squashing some coordinates and not others. I mostly avoid it.
- **Clip by global norm** — compute the norm of the whole gradient vector across all parameters; if it exceeds a threshold, scale the entire thing down proportionally. This preserves the direction and only shortens the step. This is what I default to; 1.0 is a reasonable starting threshold for transformers.

Beyond clipping: lower LR, add warmup, switch fp16 → bf16 (*bf16 has the same exponent range as fp32, so it doesn't overflow the way fp16 does*), and check your data pipeline for garbage rows. Honestly, in my experience about half of "exploding gradient" incidents turn out to be a single corrupted or extreme-valued feature in one batch, not an optimizer problem.

One nuance worth saying out loud: clipping is a **symptom mask**. If I'm clipping on 90% of steps, my effective learning rate is now being set by the clip threshold, not by my scheduler, and I should fix the underlying cause instead.

---

### Q4. Why do residual connections help so much?

Two reasons, and people usually only give the first.

**Gradient path.** `y = x + f(x)` means `dy/dx = 1 + df/dx`. That "1" is a gradient highway — even if `f` contributes almost nothing, the gradient reaches earlier layers intact.

**Optimization surface.** With a residual block, the identity function is free — the network just has to learn `f(x) = 0`. Without it, every layer has to learn to reproduce its input just to not make things worse. So residuals make "do nothing" the default, and the network only spends capacity on the delta. That's why very deep plain networks were often *worse* than shallow ones, which isn't an overfitting story at all — it's an optimization story.

The subtlety at depth: residual streams **accumulate variance** — each block adds to the stream, so activations grow layer over layer. That's why normalization placement matters, and why things like scaled residuals or LayerScale exist for very deep nets.

---

### Q5. Walk me through the common activations and when you'd pick each.

- **Sigmoid** — squashes to (0,1). Use it for a binary output or a gate. Never for hidden layers in a deep net: saturates, and its output isn't zero-centered, which makes gradients for a whole layer share a sign and zig-zag.
- **Tanh** — zero-centered, still saturates. Fine for small recurrent nets and gates, mostly historical elsewhere.
- **ReLU** — `max(0,x)`. Cheap, non-saturating on the positive side, induces sparsity. The failure mode is **dying ReLU** — *a unit whose input is always negative outputs zero forever and gets zero gradient forever, so it's permanently dead.* Usually caused by too-high LR or a badly-negative bias. Leaky ReLU fixes it with a small negative slope.
- **GELU / SiLU (Swish)** — smooth, non-monotonic near zero, small negative values allowed. Standard in transformers now. The smoothness helps optimization and there's no hard dead zone.
- **GLU family (SwiGLU, GeGLU)** — a gated variant where one linear projection multiplies another that's been passed through an activation. Used in the feedforward block of most modern LLMs; empirically better quality per parameter, at the cost of a third weight matrix. This is where "gates" show up in modern architectures, which I think is worth flagging because people associate gates only with LSTMs.

Practically: for anything transformer-shaped I use GELU or SwiGLU without thinking hard. For tabular MLPs, ReLU or GELU. The activation is rarely where the wins are.

---

### Q6. Why does initialization matter, and what do Xavier and He actually do?

Both are answering the same question: what variance should I draw initial weights from so that the *variance of activations stays roughly constant* as I go up the layers, and the variance of gradients stays constant as I go down?

If the variance shrinks per layer, you get vanishing. If it grows, you get exploding. So it's the same problem as Q2/Q3, just at step zero.

- **Xavier/Glorot** balances forward and backward variance assuming a roughly linear/symmetric activation like tanh — variance ∝ `2/(fan_in + fan_out)`.
- **He/Kaiming** accounts for ReLU killing half the units, so it doubles the variance — ∝ `2/fan_in`.

In transformers there's an extra wrinkle: many implementations scale the initialization of the output projection of each residual block by `1/sqrt(2 * n_layers)`, precisely to stop the residual stream variance from compounding with depth. And embedding tables are typically initialized much smaller (e.g. std 0.02) than a generic Xavier would suggest.

The reason I'd raise this in an interview: init is a "free" fix. If your first 200 steps look unstable and you're reaching for the LR, check init first.

---

### Q7. SGD, momentum, Nesterov — what's the actual difference?

- **Plain SGD**: `w ← w - lr * g`. Noisy, but the noise is a feature — it helps escape sharp bad minima. Very sensitive to LR.
- **Momentum**: keep a running average of past gradients (velocity) and step along that. `v ← β·v + g; w ← w - lr·v`. Two effects — it accelerates along directions where gradients consistently agree, and it cancels out oscillation in directions where the gradient keeps flipping sign. Classic picture: a narrow ravine where plain SGD bounces off the walls and momentum slides down the floor. β around 0.9 means roughly a 10-step effective average.
- **Nesterov**: peek ahead — compute the gradient at where momentum is *about to take you*, not where you are. Slightly better-behaved, mild gain, rarely the deciding factor.

The thing worth saying: **SGD + momentum still wins on some vision workloads for final generalization**, and there's a real argument that it finds flatter minima than Adam. But it needs careful LR tuning and schedules. Adam is more forgiving, which matters more when you're iterating fast or the loss landscape is heterogeneous (like embeddings + dense layers in one model).

---

### Q8. Explain Adam end to end.

Adam keeps two running averages per parameter:

- **m** — exponential moving average of the gradient. This is momentum: direction with smoothing.
- **v** — exponential moving average of the gradient *squared*. This is a per-parameter estimate of how big gradients have typically been.

The update is roughly `w ← w - lr * m / (sqrt(v) + ε)`.

The intuition: divide by the typical gradient size, so every parameter gets a step that's normalized to its own scale. A parameter that always sees tiny gradients still gets a meaningful step; one that sees huge gradients gets damped. That's why Adam works so well on models with wildly different parameter groups — an embedding table and a LayerNorm gain live on completely different gradient scales, and Adam handles both without me hand-tuning per-group LRs.

Then there's **bias correction**. Both m and v start at zero, so in the first steps they're biased toward zero — an average of "zero and one real gradient" underestimates. Adam divides by `1 - β^t` to undo that. Without it, the first few hundred steps take wrongly-sized steps, usually too small for m and too small for v, and the ratio misbehaves. It matters most early, and it decays to nothing as `t` grows.

The mental model I use: **Adam is momentum plus per-parameter learning rate normalization, with a correction for the cold start.**

---

### Q9. What do β1 and β2 actually control, and when would you change them from the defaults?

- **β1 (default 0.9)** — how much gradient history goes into the direction. Higher = smoother, more inertia, slower to react to a genuine change in direction. Roughly `1/(1-β1)` steps of memory, so 0.9 ≈ 10 steps, 0.95 ≈ 20.
- **β2 (default 0.999)** — how much history goes into the *scale* estimate. 0.999 ≈ 1000 steps of memory. This one is the sneaky one.

When I'd change them:

- **Large-batch training** → raise β1 slightly (0.95). Gradients are already less noisy, and you want the momentum window to cover a comparable amount of *data*, not a comparable number of steps.
- **Loss spikes / instability in transformer pretraining** → *lower* β2 to 0.95 or 0.98. With β2=0.999 the variance estimate is very slow to react, so if a single batch produces a huge gradient, `v` doesn't rise fast enough to damp it and you take a giant step. Lowering β2 makes the denominator react faster and absorbs the spike. This is a well-known LLM pretraining fix and a good thing to have an opinion about.
- **Very sparse gradients (embeddings)** → β2 close to 1 is actively harmful for rare features, because `v` decays toward zero between the rare updates, and then a rare gradient gets divided by a tiny number → enormous step. This is exactly the motivation for SparseAdam / lazy Adam.
- **Fine-tuning a pretrained model** → I mostly leave them and lower the LR instead.

**ε (default 1e-8)** is there to stop division by zero, but it's not purely cosmetic — it caps the maximum effective step size at roughly `lr/ε`-ish behavior for tiny gradients. In fp16, 1e-8 can underflow, so people raise it to 1e-6 or 1e-7. RL practitioners often use 1e-5 deliberately to damp the adaptivity.

---

### Q10. Adam vs AdamW. Why does the difference matter?

L2 regularization and weight decay are the *same thing* for plain SGD, and **different things** for Adam. That's the whole point.

Classic L2 adds `λ·w` to the gradient. Adam then divides that by `sqrt(v)`. So a parameter with a large gradient history gets its decay *shrunk*, and a parameter with a small gradient history gets its decay *amplified*. The regularization strength ends up coupled to the gradient statistics, which is not what anyone intends.

**AdamW decouples it**: the decay is applied directly to the weight, outside the adaptive normalization — `w ← w - lr·(m/(sqrt(v)+ε)) - lr·λ·w`. Now decay means the same thing for every parameter.

In practice AdamW generalizes better and its weight decay hyperparameter is actually tunable in a meaningful way. It's the default for transformers for good reason. If someone tells me they're using Adam with `weight_decay` set on the optimizer in old-style code, that's a small red flag worth checking.

**Which params to exclude from decay:** biases, LayerNorm/BatchNorm gains and biases. Decaying a normalization scale toward zero is just fighting the normalization layer. Most training scripts get this wrong by default; it's a cheap win.

---

### Q11. What is SparseAdam and when do you need it?

This is a real problem in any model with big ID embedding tables — which is basically every recsys and every merchant/user representation model I've built.

Say I have 20 million merchant IDs and each batch touches 4,000 of them. The gradient for the embedding table is **sparse** — only 4,000 rows have nonzero gradient. But standard Adam is *dense*: it updates `m` and `v` for all 20 million rows every step, applying momentum and decay to rows that got zero gradient. Two problems:

1. **Cost.** You're doing 20M row updates for 4K rows of signal, every step. That's a huge waste of memory bandwidth and it's often the actual bottleneck.
2. **Correctness for rare IDs.** For a row that gets updated once every 10,000 steps, `v` has decayed to nearly nothing by the time the next gradient arrives. Dividing a normal gradient by `sqrt(tiny)` gives a massive step. Rare merchants get violently thrashed while common ones update smoothly.

**SparseAdam** (and "lazy Adam" variants) only touch the rows present in the current batch, and handles the moment updates in a way appropriate to that. It's dramatically faster and better behaved on the tail.

Caveats to mention, because they show you've actually used it:
- SparseAdam in PyTorch requires `sparse=True` on the embedding, doesn't support weight decay, and doesn't play nicely with all distributed setups — sparse gradient all-reduce is awkward, so many large-scale systems use a parameter-server or sharded-embedding design instead.
- You typically end up with **two optimizers** — SparseAdam for embedding tables, AdamW for dense layers — which means two schedulers and two sets of hyperparameters to keep in sync. That's an operational cost, not just a code detail.
- Alternative approaches: row-wise Adagrad (what a lot of production recsys uses — simpler, monotonically decaying per-row LR, very robust for the tail), or hashing/bucketing the tail IDs so no row is ultra-rare in the first place.

↪ **Your hook:** merchant representation learning at Grab — high-cardinality merchant IDs with a long tail is exactly this scenario. Talk about how you handled the tail.

---

### Q12. Learning rate schedules — what do you use and why?

The learning rate is the single most important hyperparameter, and the schedule matters almost as much as the peak value.

**Warmup** — start near zero, ramp linearly to peak over a few hundred to a few thousand steps. Why: at step zero the model is random, gradients are large and mutually inconsistent, and Adam's `v` estimate is based on almost no data so it's unreliable. Taking full-size steps on unreliable curvature estimates is how you get an early divergence that never recovers. Warmup buys time for the second-moment estimate to become meaningful. Pre-LN transformers need it less than Post-LN, but I still use it.

**Decay** — the reason is that early on you want to explore, and late you want to settle into a minimum. A constant LR keeps bouncing around the basin and never converges tightly.

- **Cosine** — smooth decay to near zero. My default for a fixed-budget run. Downside: it's tied to a fixed total step count, so if you want to train longer later you've already annealed.
- **Linear decay** — nearly as good, simpler, common for fine-tuning.
- **Step decay** — divide by 10 at fixed milestones. Old-school, works, produces those characteristic staircase loss drops.
- **Inverse sqrt** — the original transformer schedule; doesn't require knowing total steps, so it's good for open-ended training.
- **WSD (warmup-stable-decay)** — hold a constant LR for most of training then decay sharply at the end. Increasingly popular because you can branch a decay off any checkpoint without having committed to a horizon.
- **ReduceLROnPlateau** — reactive, based on validation. Fine for smaller supervised jobs, awkward for large distributed runs.

Rule of thumb I'd state: if you change the batch size, change the LR. Roughly **linear scaling** for SGD (2× batch → 2× LR) and closer to **square-root scaling** for Adam. And always re-warm up after a big batch change.

---

### Q13. Gradient accumulation — what is it and what breaks?

Gradient accumulation means running several forward/backward passes, summing the gradients, and only stepping the optimizer once. It simulates a large batch on limited memory.

Two things people get wrong:

1. **Averaging.** If your loss is a mean over the batch, you have to divide by the number of accumulation steps, otherwise your effective LR is silently multiplied by that number.
2. **BatchNorm.** Accumulation does *not* give you large-batch BatchNorm statistics — each micro-batch normalizes over itself. So accumulation is a true large-batch equivalent for LayerNorm models and a lie for BatchNorm models. This is one of several reasons transformers use LayerNorm.

Also worth knowing: accumulation doesn't reduce compute, only memory pressure and communication frequency (you only all-reduce on the step boundary), so it can actually improve throughput in a communication-bound distributed setup.

---

### Q14. Mixed precision — fp16 vs bf16, and what's loss scaling?

Mixed precision keeps weights in fp32 (the "master copy") but does the matmuls in 16-bit, which roughly doubles throughput on tensor cores and halves activation memory.

- **fp16** has high precision but a narrow exponent range. Small gradients underflow to zero. So you need **loss scaling** — multiply the loss by a big constant before backward, so gradients land in representable range, then divide it back out before the optimizer step. Dynamic loss scaling adjusts that constant automatically and skips steps that produce infs.
- **bf16** has the same exponent range as fp32 but fewer mantissa bits. So it can represent tiny and huge numbers fine — no loss scaling needed — at the cost of precision. On A100 and newer this is what I'd default to, because it removes an entire class of instability bugs.

Things that must stay in fp32 regardless: optimizer states, the master weights, the softmax and loss computation, and typically the normalization layers. Accumulating a long reduction in 16-bit is where silent quality loss creeps in.

↪ **Your hook:** multi-A100 training — bf16 is the natural choice on A100 and you can say why.

---

### Q15. Your loss suddenly went to NaN at step 40k. Walk me through debugging.

I'd go in order of cheapest-to-check:

1. **Is it reproducible?** Restart from the last good checkpoint with the same seed and data order. If it NaNs at the same step, it's data or a deterministic numerical issue. If not, it's a race/nondeterminism/hardware thing.
2. **Find the first NaN, not the last.** Register hooks or use `torch.autograd.set_detect_anomaly` on a short repro to locate which op produced the first non-finite value. Loss NaN is the end of the story, not the beginning.
3. **Check the batch.** Pull the exact batch at that step. Extremely common causes: an all-padding sequence (so the attention mask makes an entire softmax row `-inf` → NaN), a zero-length sequence, a divide-by-zero in a normalization of a constant feature, `log(0)` in a loss, or an out-of-range index into an embedding table.
4. **Check for fp16 overflow** — if you're on fp16, look at the loss scale history. A collapsing loss scale is the tell. Switch to bf16 and see if it survives.
5. **Check gradient norms just before.** If the grad norm was climbing for a few hundred steps, it's an instability — lower LR, lower β2, tighten clipping. If it was flat and then one spike, it's a bad batch.
6. **Check for accumulating drift** — e.g. a growing `v` denominator underflowing, or a running BatchNorm statistic being poisoned.

The L4 answer adds the operational piece: **I want NaN detection built into the training loop** — check loss finiteness each step, skip-and-log the batch rather than crash, and alert if the skip rate exceeds a threshold. A 40k-step run dying at hour 9 is an availability problem, not just an ML problem.

---

### Q16. Why don't we use second-order optimization?

Second-order methods use curvature — the Hessian tells you not just the slope but how the slope is changing, so you can take a smartly-sized step instead of guessing. In theory it's much faster in iterations.

The problem is cost: the Hessian is N×N for N parameters. For a 1B parameter model that's not storable, let alone invertible. Approximations exist — K-FAC, Shampoo, Sophia, and more recently Muon — that approximate curvature block-wise or via matrix structure. Shampoo and Muon have shown real wall-clock wins on large training runs recently, so I'd say the honest answer today is "we're starting to."

Also worth noting: **Adam is already a crude diagonal second-order method.** Dividing by `sqrt(v)` is approximating the diagonal of the Fisher information — *a matrix that measures how sensitive the model's output distribution is to each parameter.* So we do use curvature, just the cheapest possible version of it.

---

## Part 2 — Regularization & Normalization

### Q17. L1 vs L2 — what's the real difference?

Both add a penalty on weight size to the loss, but they penalize differently and the consequence is geometric.

- **L2 (ridge)** penalizes the sum of squared weights. The gradient of the penalty is proportional to `w`, so it shrinks large weights hard and small weights barely at all. Result: everything gets smaller, nothing hits exactly zero. Weights get spread out — if two features are correlated, L2 splits the weight between them.
- **L1 (lasso)** penalizes the sum of absolute weights. The gradient is a constant `±λ` regardless of how small `w` is. So a weight near zero gets pushed the same amount as a large one, and it lands exactly at zero and stays. Result: **sparsity** — actual feature selection. With correlated features, L1 tends to pick one and zero the others.

The geometric picture, if they want it: the L1 constraint region is a diamond with corners on the axes; the loss contours are likeliest to first touch a corner, and a corner means a coordinate is zero. The L2 region is a sphere with no corners, so no zeros.

When I pick which: L2 (as decoupled weight decay) is my default for deep nets. L1 when I want an interpretable sparse model or genuine feature pruning, or in a linear/GBDT-adjacent setting. **Elastic net** is both, with a mixing ratio — useful when features are correlated and pure L1 is unstable in which one it picks.

One honest caveat for deep learning: in modern nets, weight decay's benefit is arguably less about the classical "shrink the hypothesis class" story and more about controlling the effective learning rate through weight norm — especially with normalization layers, where scaling a weight doesn't change the function at all, only the gradient scale.

---

### Q18. Where do Frobenius, spectral, and nuclear norms show up?

These are all matrix norms — ways of measuring "how big is this matrix" — and each one measures something different.

- **Frobenius norm** — square root of the sum of all squared entries. It's just L2 on a flattened matrix. This is what "weight decay on a weight matrix" is doing. Cheap, differentiable, no structure assumed. It's also what people mean by "matrix reconstruction error" in PCA and in low-rank approximations — PCA is exactly the best rank-k approximation in Frobenius norm.
- **Spectral norm** — the largest singular value. That's the maximum amount the matrix can stretch any vector, i.e. its worst-case amplification. This is the one that matters for **stability**: a layer whose spectral norm exceeds 1 amplifies, and stacking amplifiers is how you explode. Spectral normalization (dividing the weight by its top singular value, estimated with a couple of power iterations) was the key trick that made GAN discriminator training stable, and it's also central to Lipschitz-constrained models — *Lipschitz meaning the output can't change faster than some fixed multiple of the input change.*
- **Nuclear norm** — sum of all singular values. This is the "L1 of the singular values," so penalizing it produces **low rank** the same way L1 produces sparsity. Shows up in matrix completion and low-rank recommendation models. Expensive because it needs an SVD.

The one-line summary I'd give: Frobenius controls total size, spectral controls worst-case amplification, nuclear controls rank.

---

### Q19. Dropout — how does it work and where would you *not* use it?

At training time, randomly zero out a fraction `p` of activations, independently each forward pass. At inference, use everything. This prevents co-adaptation — *units learning to rely on the presence of specific other units* — so each unit has to be independently useful. Loosely, it's training an exponential ensemble of subnetworks and averaging them at test time.

The implementation detail worth knowing: **inverted dropout**, which is what every framework does. During training, after zeroing, divide the survivors by `(1-p)` so the expected activation magnitude is unchanged. That means inference needs no adjustment at all — you just turn dropout off. If you ever hand-roll this and forget the scaling, train and eval will disagree by a constant factor and you'll chase it for a day.

Where I would *not* use it, or would be careful:

- **Right before or after BatchNorm.** Dropout changes the variance of activations, so the BN running statistics collected during training don't match inference. This is the classic "great val loss, terrible test loss" variance shift bug.
- **In very large models trained on very large data.** LLM pretraining often uses dropout 0.0 — with a single epoch over trillions of tokens you can't really overfit, and dropout just slows convergence. Dropout comes back for *fine-tuning* on small data.
- **On embeddings, naively.** Zeroing random dimensions of an embedding is different from zeroing a whole token. Usually you want structured dropout: whole-token/whole-feature dropout, which also doubles as robustness training for missing features at serving time.

Variants worth naming: **attention dropout** (on the attention probabilities), **stochastic depth / DropPath** (drop entire residual blocks — the standard regularizer for deep vision transformers), and **feature-level dropout for tabular models**, which conveniently simulates the missing-feature case you'll hit in production anyway.

---

### Q20. What is label smoothing and when does it help?

Instead of training toward a hard target of 1.0 for the correct class and 0.0 for everything else, you train toward something like 0.9 and spread the remaining 0.1 across the others.

Why: with hard targets, the cross-entropy loss is minimized by pushing the correct logit to infinity, so the model keeps growing logits forever, becomes wildly overconfident, and the gap between the correct and incorrect logits gets pathological. Label smoothing caps that — it says "be right, but don't be infinitely sure."

Benefits: better **calibration** (*predicted probabilities that actually match observed frequencies*), slightly better accuracy on classification and translation, and more stable training.

The cost, and this is the part worth knowing for your embedding work: label smoothing **tightens clusters and shrinks the within-class variance of the penultimate representation.** That's great for classification and can be actively bad if you plan to use those representations for retrieval or transfer, because you've thrown away within-class structure. There's a known result that label smoothing hurts knowledge distillation for exactly this reason. So: if the model is the product, smooth. If the *embedding* is the product, be careful.

↪ **Your hook:** this connects directly to extracting merchant embeddings from a classification-style head.

---

### Q21. Explain BatchNorm, including train vs inference.

BatchNorm normalizes each feature across the batch dimension — for feature `j`, compute the mean and variance over all samples in the batch, subtract the mean, divide by the std, then apply a learnable scale `γ` and shift `β` so the network can undo the normalization if it wants to.

**Train vs inference is the crux.** At training time it uses the current batch's statistics. At inference you often have a batch of one, and you also want deterministic outputs, so it uses **running estimates** of mean and variance accumulated during training (an exponential moving average). This means BatchNorm behaves differently in `train()` and `eval()` mode, and forgetting to call `.eval()` is one of the most common bugs in the field.

Where it breaks:

- **Small batches.** With batch size 2, the batch statistics are pure noise. Quality falls off a cliff.
- **Variable-length sequences.** Normalizing across the batch when different sequences have different lengths mixes real tokens with padding. Deeply awkward.
- **Distributed training.** Each GPU normalizes over its local shard, so effective batch size for BN is per-device, not global. SyncBatchNorm fixes it but adds a communication step per BN layer.
- **Train/serve distribution shift.** The running stats are frozen snapshots of the training distribution. If production traffic shifts, your normalization is now wrong in a way that no amount of retraining the head will fix.
- **Any dependency between examples in a batch** — it leaks information across examples, which matters for contrastive setups and can be an actual leakage bug.

That list is basically the argument for LayerNorm.

---

### Q22. LayerNorm vs RMSNorm vs GroupNorm.

- **LayerNorm** normalizes across the *feature* dimension, within a single example. No cross-example dependency at all, so train and inference are identical, batch size is irrelevant, and sequence length is irrelevant. That's why transformers use it.
- **RMSNorm** drops the mean-subtraction and only divides by the root-mean-square, with a learnable scale and no bias. It's cheaper (one reduction instead of two) and empirically just as good. Most recent LLMs use it. Nice thing to mention because it shows you're current.
- **GroupNorm** splits channels into groups and normalizes within each group per example. It's the compromise for vision when batches are small — you get BN-like benefits without batch dependence. InstanceNorm is GroupNorm with one channel per group; used in style transfer where you specifically want to remove per-image contrast/style information.

---

### Q23. Pre-LN vs Post-LN. Why did everyone switch?

The original transformer was **Post-LN**: `x → attention → add residual → LayerNorm`. The norm sits on the residual path.

**Pre-LN** moves it inside: `x → LayerNorm → attention → add residual`. Now the residual stream itself is never normalized, so there's a completely clean identity path from input to output.

Consequences:

- Post-LN needs careful warmup or it diverges, because the gradient at initialization is badly scaled at depth. Pre-LN trains stably with much less babysitting.
- Pre-LN's residual stream grows in magnitude with depth (each block adds to it), so you need a final LayerNorm before the output head. And very deep Pre-LN nets can suffer from later blocks contributing proportionally less — sometimes called representation "collapse toward the residual stream."
- Post-LN, when it *does* train, sometimes reaches slightly better final quality. Which is why hybrid schemes exist (DeepNet, sandwich norm, normalizing both before and after).

Practical position: I default to Pre-LN with a final norm, because training stability at scale is worth more than a small quality delta I might not even measure reliably.

---

### Q24. Why does normalization help at all? The "internal covariate shift" story is contested.

Right — the original claim was that BN reduces internal covariate shift, meaning the distribution each layer sees keeps changing as the layers below it update. That story is intuitive but the evidence didn't hold up; people showed you can inject noise after BN to deliberately *increase* covariate shift and it still trains fine.

The better explanations:

1. **Smoother loss landscape.** Normalization bounds how much the loss and gradients can change per step — it improves the Lipschitz properties, so you can use a larger learning rate safely. This is the Santurkar et al. result and it's the one I'd lead with.
2. **Scale invariance.** With a norm layer downstream, scaling a weight matrix doesn't change the function output at all — it only changes the gradient magnitude. That effectively decouples the direction of the weight from its magnitude and makes optimization better-conditioned. It also means weight decay's real job in a normalized net is controlling effective learning rate.
3. **It keeps activations off the saturating parts of nonlinearities**, which is the plain-English version.

Being able to say "the original explanation is probably wrong and here's what the field thinks now" is a genuinely strong signal in a senior interview.

---

### Q25. How do you regularize an embedding table specifically?

Different from dense layers, because of the long tail.

- **L2 on embeddings is applied only to the rows in the batch** in most efficient implementations — otherwise you're touching the whole table every step. But be careful: this means frequent IDs get decayed far more often than rare ones, which is a hidden frequency bias.
- **Dimension choice as regularization.** The cheapest regularizer is a smaller embedding dim. Rule of thumb people use is something like `dim ≈ cardinality^0.25`, but I'd tune it.
- **Frequency-based handling**: bucket or hash tail IDs together, or fall back to content/feature-based embeddings for cold entities instead of giving every rare ID its own free parameters to overfit with.
- **ID dropout** — randomly drop the ID embedding and force the model to rely on content features. This directly trains cold-start robustness, which is the thing that will actually bite you in production.
- **Norm constraints** — clip embedding rows to a max L2 norm. Prevents popular entities from dominating dot-product retrieval purely through magnitude.
- **Shared/hierarchical structure** — e.g. merchant embedding = own embedding + category embedding + region embedding, so rare merchants inherit useful structure.

↪ **Your hook:** with 20M+ merchants and a heavy tail, the cold-start and tail-overfitting story is the most interesting thing you can talk about here. Have a number ready — what fraction of merchants had fewer than N events.

---

## Part 3 — Gates & Sequence Models

### Q26. Why do vanilla RNNs fail, and what do LSTM gates fix?

A vanilla RNN computes `h_t = tanh(W·h_{t-1} + U·x_t)`. To get the gradient from step 100 back to step 1, you multiply the same recurrent Jacobian ~100 times. If its largest eigenvalue is below 1, the signal vanishes; above 1, it explodes. Either way, long-range dependencies don't get learned — the model effectively has a memory of maybe 10 steps.

LSTM's insight is to add a **cell state** that flows through time with *addition* rather than repeated multiplication by a learned matrix. That's the same trick as a residual connection: give the gradient a highway.

Three gates control that highway. Each is a sigmoid producing a number between 0 and 1 per dimension, which multiplies something — that's all a gate is, a learned soft switch.

- **Forget gate** — how much of the previous cell state to keep. Output 1 = remember perfectly, 0 = wipe. This is arguably the most important gate; when people ablate the LSTM, removing the forget gate hurts most.
- **Input gate** — how much of the new candidate value to write into the cell.
- **Output gate** — how much of the cell state to expose as the hidden state this step.

So the cell update is `c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`. The gradient through the cell state is multiplied by `f_t`, which the network can learn to keep near 1 for dimensions it wants to preserve. That's how it gets long memory. Not magic — just a learned, per-dimension, per-timestep decision about whether to keep multiplying by ~1.

One implementation detail that shows hands-on experience: **initialize the forget gate bias to 1 or 2.** At random init the forget gate sits at ~0.5, so memory halves every step and the model can't learn long dependencies before it learns to open the gate. Biasing it open fixes a well-documented slow start.

---

### Q27. GRU vs LSTM — which and why?

GRU merges the cell and hidden state and uses two gates instead of three: an **update gate** (interpolates between keeping the old state and taking the new candidate — it does the job of forget and input together) and a **reset gate** (how much of the previous state to use when computing the candidate).

Fewer parameters, ~25% faster, less memory. Quality is usually comparable; LSTM sometimes edges it out on very long sequences or tasks needing precise counting, because the separate cell state gives it a place to store something the output gate hides.

My honest take: for most problems the choice doesn't matter much, and in 2026 the real answer is "I'd probably use a transformer or an SSM unless the sequences are short, the latency budget is brutal, or I need true streaming with O(1) state per step." Streaming inference is a genuine reason RNNs and state-space models are still around — a transformer's KV cache grows with sequence length, an RNN's state doesn't.

---

### Q28. Gates show up outside RNNs too. Where?

Worth raising because it shows you understand gating as a *principle*, not an LSTM trivia item.

- **Highway networks** — the predecessor of ResNets; a gate decides how much of the layer output vs the input passes through. ResNet is basically a Highway net with the gate hardcoded open.
- **GLU / SwiGLU / GeGLU in the transformer FFN.** The feedforward block becomes `(W1·x) ⊙ σ(W2·x) · W3` — one branch produces content, the other produces a multiplicative gate. Almost every modern LLM uses SwiGLU. It costs a third matrix (so you shrink the hidden dim to ~2/3 to keep parameter count constant) and reliably improves quality.
- **Mixture-of-Experts routing** — the router is a hard/sparse gate choosing which experts see this token.
- **Attention itself** is a form of soft gating — the softmax weights are data-dependent multipliers on values.
- **Mamba / selective SSMs** — the "selection" mechanism is explicitly input-dependent gating of the state transition, which is precisely the LSTM forget-gate idea brought back with a parallel-scan implementation.

The unifying idea: **multiplicative, data-dependent control lets a network decide what to route where, instead of applying the same transform to everything.**

---

### Q29. What's teacher forcing and what's exposure bias?

**Teacher forcing**: when training a sequence generator, feed the *ground-truth* previous token as input rather than the model's own prediction. It makes training parallelizable and stable, since the model never compounds its own errors during training.

**Exposure bias** is the consequence: at inference the model consumes its *own* outputs, a distribution it never saw in training. One mistake shifts it off-distribution and errors compound — the classic degenerate repetition loop.

Mitigations: **scheduled sampling** (gradually mix in the model's own predictions during training — though it has a known theoretical inconsistency), **minimum risk training / sequence-level objectives**, **backtranslation-style data augmentation**, and at inference, **beam search** with length normalization, or sampling with temperature / top-k / nucleus. In modern LLMs, RLHF-style training is partly addressing exposure bias too, since the model is scored on its own generations.

↪ **Your hook:** your mBART / FloRes NMT project. Backtranslation is literally an exposure-bias and data-scarcity mitigation, and you used it. Worth framing it that way rather than just "I did backtranslation."

---

## Part 4 — Attention & Transformer Architecture

### Q30. Explain self-attention from scratch.

Every token produces three vectors via learned linear projections:

- **Query** — what I'm looking for.
- **Key** — what I have to offer.
- **Value** — what I'll actually hand over if you pick me.

For a given token, I dot its query against every token's key. That gives a similarity score per token. Softmax those scores into weights that sum to 1, then take the weighted sum of the value vectors. That's the output for that token.

So: attention is **content-based soft lookup**. Instead of a hard dictionary lookup where a key either matches or doesn't, every key matches a bit and you take a weighted blend.

The whole thing is `softmax(QKᵀ/√d)V`, and it's two matmuls and a softmax, which is why it's so GPU-friendly.

**Why divide by √d:** if Q and K entries are roughly independent with unit variance, their dot product over `d` dimensions has variance `d`, so the scores grow with dimension. Large scores push softmax into a near one-hot regime where its gradient is nearly zero — saturated, so nothing learns. Dividing by √d keeps the scores at roughly unit variance regardless of head dimension. It's a variance-control fix, exactly the same family of reasoning as initialization and normalization.

---

### Q31. Why multiple heads?

A single softmax attention distribution can really only focus on one thing at a time — it's a single weighted average, so it has one "opinion" per token. Multiple heads let the model attend to different relationships in parallel: one head might track syntactic dependency, one might track the previous occurrence of the same token, one might be a positional offset head.

Mechanically, you split the model dimension into `h` chunks, run attention independently within each, concatenate, and project. So total compute is roughly the same as one big head — it's a *refactoring* of the same budget into multiple lower-dimensional subspaces, not extra capacity.

Two honest caveats worth stating:
- **Many heads are redundant.** There's solid work showing you can prune a large fraction of heads at inference with little loss. So "more heads" isn't monotonically better.
- **Head dimension has a floor.** If you take a 512-dim model and use 64 heads, each head has 8 dimensions and can't represent much. There's a rank/expressiveness argument here — very small head dims hurt.

---

### Q32. Causal attention — what is it and how is it implemented?

Causal (or masked) attention means a token can only attend to itself and earlier tokens, never future ones. It's what makes autoregressive generation valid — *autoregressive meaning predicting the next element conditioned on all previous ones.*

Implementation: build a lower-triangular mask, and before the softmax, set all disallowed positions to `-inf` (in practice a large negative number like -1e9, or -inf handled carefully). Softmax then assigns them exactly zero weight.

Why do it *inside* the softmax rather than zeroing after: because softmax normalizes. If you zero the weights afterward, the remaining weights no longer sum to 1 and you've silently changed the scale of every output. Masking pre-softmax means the normalization only ever runs over legal positions.

The reason this matters so much for efficiency: it lets you compute the loss for **every position in the sequence in a single forward pass.** A 2048-token sequence gives you 2048 training signals at once, all correctly conditioned, with no leakage. That's the entire reason decoder-only pretraining scales.

**The bug to watch for:** combining a causal mask with a padding mask. If a padded row ends up with *every* position masked, the softmax over all `-inf` produces NaN. That's the NaN cause I mentioned earlier. Handle it by ensuring at least one valid position, or by zeroing those rows explicitly after.

Also worth knowing: **causal masking is what makes the KV cache possible.** Since position `t` never looks forward, the keys and values for positions `1..t-1` never change as you generate, so you cache them and each new token costs O(t) instead of O(t²).

---

### Q33. What is FlashAttention and why is it not an approximation?

This is a favorite question because a lot of people get it wrong and say it's a sparse or approximate attention method. It isn't. **FlashAttention computes exactly the same output as standard attention** — bit-for-bit up to floating point reordering. It's a systems optimization, not a math approximation.

The problem it solves: standard attention materializes the full `n × n` attention matrix in GPU high-bandwidth memory (HBM). For a 8k sequence that's 64M entries per head per layer. Writing and reading that matrix is *memory-bandwidth bound* — the GPU spends most of its time moving data, not computing. The FLOPs were never the bottleneck; the memory traffic was.

The trick has three parts:

1. **Tiling.** Split Q, K, V into blocks that fit in on-chip SRAM (fast, tiny) and compute attention block by block, so the big intermediate matrix never touches HBM.
2. **Online softmax.** Softmax needs a max and a sum over the whole row, which naively requires seeing the whole row. The online/streaming formulation keeps a running max and running sum and *rescales* previously accumulated results when a new larger max appears. So you get exact softmax without ever holding the full row.
3. **Recomputation in the backward pass.** Rather than storing the attention matrix for backward, recompute it from the cached statistics. Trading FLOPs for memory traffic is a *win* here precisely because you're bandwidth-bound.

Result: memory goes from O(n²) to O(n) and wall-clock speeds up several-fold, with identical outputs. FlashAttention-2 improved the work partitioning across warps; FlashAttention-3 exploits Hopper-specific async features.

The transferable lesson I'd draw out: **know whether your kernel is compute-bound or memory-bound before optimizing it.** The arithmetic intensity analysis is the reusable skill; FlashAttention is one instance of it. That framing plays very well for a systems-heavy candidate.

---

### Q34. What other approaches exist for long context?

- **Sparse / local attention** — each token attends to a window plus a few global tokens (Longformer, BigBird). Linear in `n`, approximate, loses some long-range detail.
- **Linear attention** — reorder the matmuls using a kernel feature map so you never form the `n×n` matrix (Performer, Linear Transformer). O(n) but quality has historically lagged; it's having a comeback in hybrid architectures.
- **Low-rank projection** — Linformer projects keys/values down to a fixed length. Cheap but assumes the attention matrix is low-rank and breaks for variable-length inputs.
- **Sliding window + attention sinks** — StreamingLLM's observation that keeping the first few tokens plus a recent window preserves quality, because models dump excess attention mass on the first tokens ("attention sinks"). Nice detail to know.
- **State space models (Mamba, S4)** — O(n) with a recurrent state, parallelizable in training via scan. Often now used *hybrid*, a few attention layers among many SSM layers.
- **Multi-Query / Grouped-Query Attention** — doesn't reduce training cost but massively shrinks the **KV cache** at inference by sharing keys and values across heads. MQA = one KV head total, GQA = a few groups. GQA is the standard compromise: near-MHA quality at near-MQA memory. Since decoding is memory-bandwidth bound on the KV cache, this is often the single biggest inference win.

The framing: FlashAttention makes exact attention *cheap*; these make attention *asymptotically cheaper by changing the math*. Different trade, and I'd try FlashAttention first because it costs nothing in quality.

---

### Q35. Walk through a transformer block.

Pre-LN block, which is the modern default:

```
x = x + MultiHeadAttention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

- **The residual stream** is the backbone — a running representation that each block reads from and writes to. Everything else is additive edits to it. I find this mental model much more useful than "stacked layers."
- **Attention** is the mixing operation across positions — it's the only place information moves between tokens.
- **Feedforward** is per-position — it processes each token independently. Typically expands 4× and projects back (or ~8/3× with SwiGLU to hold parameters constant). This is where most of the parameters live, and there's decent evidence it functions as a key-value memory storing factual associations.
- **Norm** keeps scales controlled; **final LayerNorm** before the output head in Pre-LN.

The clean summary: **attention moves information between positions, the FFN transforms information within a position.** Alternating them is the whole architecture.

Parameter split, useful to have at hand: attention is `4·d²` per layer (Q, K, V, O), FFN is `8·d²` for a 4× ratio. So roughly two thirds of the non-embedding parameters are in the FFN.

---

### Q36. Encoder-only, decoder-only, encoder-decoder — when do you pick each?

- **Encoder-only (BERT-style)** — bidirectional attention, every token sees every other token. You get a rich contextual representation of a *given* input, but it can't generate. Use for classification, retrieval, entity representations, scoring, tagging. If my product is an embedding or a label, this is the natural fit.
- **Decoder-only (GPT-style)** — causal attention, trained to predict the next token. Generates naturally, scales beautifully, and empirically turns out to be a strong general learner. Use for generation and anything you can frame as generation.
- **Encoder-decoder (T5, mBART, original transformer)** — encoder builds a bidirectional representation of the source, decoder generates the target while cross-attending to the encoder output. Use when the input and output are genuinely different objects and the input is fully available up front: translation, summarization, seq2seq. The bidirectional encoder is a real advantage for understanding the source.

Why the field drifted to decoder-only: one objective, one stack, trivially scalable, no architectural asymmetry, and next-token prediction on everything turns out to be a remarkably general training signal. But for **translation specifically**, encoder-decoder remains very competitive, which is directly relevant to your mBART work — you can say you chose an encoder-decoder because the source is fully observed and bidirectional source encoding is worth it, and that's a defensible engineering answer rather than a fashion answer.

↪ **Your hook:** mBART is an encoder-decoder denoising autoencoder. Be ready to explain *why that architecture for low-resource MT* — pretraining on monolingual data in many languages, then fine-tuning on the scarce parallel pairs.

---

### Q37. BERT vs GPT — compare them properly.

The real difference is the **training objective**, and everything else follows from it.

**BERT — Masked Language Modeling (MLM).** Mask ~15% of tokens and predict them using both left and right context. Because it's bidirectional, each token's representation is informed by the entire sentence. Great representations. But:
- It only gets a learning signal from 15% of tokens per sequence, so it's sample-inefficient.
- It has a **pretrain/finetune mismatch**: `[MASK]` appears in pretraining and never at inference. (The 80/10/10 trick — 80% replaced by `[MASK]`, 10% by a random token, 10% left alone — exists to soften exactly this.)
- It can't generate, because generation needs a causal factorization.

**GPT — Causal Language Modeling (CLM).** Predict the next token. Signal from every position, no mismatch between pretraining and use, and it generates. The cost is that each token's representation only sees the left context, which is theoretically weaker for understanding.

Other differences worth naming: BERT adds `[CLS]` and `[SEP]` tokens and a segment embedding for sentence-pair tasks; original BERT had Next Sentence Prediction, which **RoBERTa showed was useless** — RoBERTa dropped NSP, trained longer on more data with dynamic masking and bigger batches, and got substantially better with the same architecture. That's a good thing to cite because it's a clean example of "the recipe mattered more than the architecture."

**What I'd actually use in 2026 for a representation task:** honestly, a modern encoder (something in the ModernBERT / E5 / GTE family) or a decoder-based embedding model with the last-token or mean pooling and a contrastive fine-tune. Decoder-only models *can* produce excellent embeddings once you either remove the causal mask during embedding or train them properly for it — LLM2Vec-style. But if latency and cost matter and my domain is narrow, a small bidirectional encoder is still the pragmatic pick.

---

### Q38. What are the other pretraining objectives besides MLM and CLM?

- **Span corruption (T5)** — mask contiguous spans and generate them, rather than individual tokens. Harder and more useful than single-token masking, because predicting one masked token in "New [MASK] City" is trivial.
- **Denoising autoencoding (BART/mBART)** — corrupt the input arbitrarily (delete, shuffle, infill, rotate) and reconstruct the original. More general than MLM and it trains the decoder too, which is why it's the natural pretraining for a seq2seq model.
- **Replaced token detection (ELECTRA)** — a small generator swaps in plausible tokens, and the main model classifies *every* token as original or replaced. Signal from 100% of positions instead of 15%, so it's dramatically more compute-efficient. Great answer if someone asks how to improve on MLM.
- **Permutation LM (XLNet)** — predict tokens in a random order, so you get bidirectional context without a `[MASK]` token.
- **Contrastive objectives** — SimCSE, sentence-level contrastive learning. Crucial for embeddings specifically, and I'd argue MLM alone does *not* give you good sentence embeddings (see the anisotropy section).
- **Masked sequence modeling on non-text** — which is essentially what you did: mask events in a merchant's behavior sequence and predict them, semi-supervised. Same family as MLM, different modality.

↪ **Your hook:** frame your merchant work as "MLM adapted to event sequences," then discuss what's *different* — no natural vocabulary, mixed continuous and categorical fields, and the fact that you have to decide what "masking" even means for a multi-field event. That last point is a genuinely interesting design question and interviewers love it.

---

## Part 5 — Positional Encoding

### Q39. Why does a transformer need positional information at all?

Because attention is **permutation-equivariant** — *shuffle the input tokens and the output just shuffles the same way, with identical values.* Attention computes a weighted sum over a set; a set has no order. So "dog bites man" and "man bites dog" would produce identical representations, which is obviously useless.

Note that the FFN doesn't help — it's per-position and identical for every position. So the *only* way order enters the model is if you inject it into the representations or into the attention scores themselves. That framing matters: it explains why there are two whole families of solutions — ones that modify the **inputs** (absolute) and ones that modify the **attention scores** (relative, ALiBi) or the **Q/K vectors** (RoPE).

---

### Q40. Explain sinusoidal positional encoding.

The original transformer adds a fixed vector to each token embedding, built from sines and cosines at geometrically-spaced frequencies. Low-dimension components oscillate fast, high-dimension components oscillate slowly — so together they form something like a smooth binary counter, giving each position a unique fingerprint.

Two properties they wanted:

1. **Bounded and consistent scale** — values stay in [-1,1] regardless of sequence length, unlike just adding the integer index.
2. **Linear relative-offset structure** — the encoding at position `p+k` is a fixed linear (rotation) transform of the encoding at position `p`. So in principle the model can learn to attend by relative offset, because "shift by k" is a learnable linear operation.

Practical view: it needs no parameters and in theory extrapolates past the training length. In practice it extrapolates badly — the model learns to use specific frequency patterns from the range it saw, and quality degrades outside it. That's the motivation for everything after.

---

### Q41. Learned absolute positional embeddings — trade-offs?

Just a lookup table with one vector per position, added to the token embedding, trained like any other parameter. That's what BERT and GPT-2 used.

Upside: simple, and the model learns whatever positional structure actually helps rather than what we guessed.

Downsides, and they're serious:
- **Hard length ceiling.** There's no embedding for position 513 if you trained to 512. You cannot extend the context without adding and training new rows.
- **Poor generalization across positions.** Position 480 may be badly trained if most training sequences were short, so quality is uneven along the sequence.
- **Absolute, not relative.** What usually matters linguistically is "three tokens back," not "token number 417." Absolute encodings have to learn relative structure indirectly.

---

### Q42. What are relative positional encodings?

Instead of tagging each token with where it is, you modify the attention score based on the **distance between** the query and key positions. That directly encodes "how far apart are these two tokens," which is usually what actually matters.

- **Shaw et al.** — add a learned vector per relative offset into the key (and optionally value) computation.
- **T5 relative bias** — much simpler and very widely copied: add a learned **scalar bias** to the attention logit, indexed by a bucketed relative distance. The buckets are logarithmic, so nearby distances get fine resolution and far distances get coarse shared buckets. Cheap, effective, and it extrapolates reasonably because far distances all collapse into the same bucket.
- **Transformer-XL** — a relative scheme with a decomposition of the attention score into content-content, content-position, etc.

Cost: you're adding a term per query-key pair, which is more bookkeeping than adding a vector once at the input, and it complicates fused attention kernels (though FlashAttention supports bias terms).

---

### Q43. Explain RoPE. Why is it the default now?

**Rotary Position Embedding.** Instead of adding anything, it **rotates** the query and key vectors by an angle proportional to their position.

Concretely: take the head dimension and pair it up into 2D slices. For a token at position `m`, rotate each 2D pair by angle `m·θ_i`, where each pair `i` has its own frequency `θ_i` (again geometrically spaced, like sinusoidal).

The magic is what happens in the dot product. When you dot a query at position `m` against a key at position `n`, the two rotations combine and what survives depends only on `(m - n)`. So:

> **RoPE is applied absolutely but acts relatively.**

That's the one-sentence answer, and it's the reason it's popular. You get relative-position behavior with the implementation simplicity of an absolute scheme — no `n×n` bias matrix, no extra parameters, and it composes cleanly with FlashAttention because it's applied to Q and K *before* the attention kernel runs.

Other properties: rotation preserves vector norms, so it doesn't perturb the scale of the representations; and the attention score naturally decays with distance for most frequency pairs, which is a sensible inductive bias.

Applied to Q and K only, **not V** — the values carry content, not position.

---

### Q44. What is ALiBi, and how does it compare?

**Attention with Linear Biases.** No positional embeddings at all. Instead, subtract a penalty from the attention logit proportional to the distance between query and key: `score = q·k - m·|i-j|`, where `m` is a fixed head-specific slope. Different heads get different slopes, so some heads are very local and some are near-global.

It's essentially a hardcoded "prefer nearby tokens" prior, with the strength varying by head.

Why people liked it: it extrapolates to sequences much longer than training with almost no degradation, because a linear penalty is well-defined at any distance. It's also trivially cheap and parameter-free.

Why RoPE mostly won anyway: ALiBi's monotonic decay is a strong assumption — it makes genuinely long-range retrieval harder, because a token 5,000 positions back is always penalized regardless of how relevant it is. On tasks that need precise long-range recall (retrieval, in-context learning over long documents), that hurts. RoPE plus a context-extension method gives you long context without hardcoding "far = unimportant."

---

### Q45. How do you extend a model's context window after training?

This comes up a lot and it's a nice "I've read the current literature" question.

If a RoPE model was trained to 4k and you feed it 32k, the rotation angles go far outside the range it ever saw, and quality collapses. Approaches:

- **Position Interpolation (PI)** — instead of extrapolating, *compress*: divide the position index by the extension factor, so position 32,000 maps to angle-position 4,000. Every angle is now in-distribution. Needs a short fine-tune. Downside: it squashes high-frequency components, blurring the model's ability to distinguish adjacent tokens.
- **NTK-aware scaling** — the insight that PI treats all frequencies equally, when the *high-frequency* dimensions are the ones handling local precision and the *low-frequency* ones handle long-range. So scale the frequency base non-uniformly: barely touch high frequencies, interpolate low ones. Works better, sometimes without fine-tuning.
- **YaRN** — refines NTK scaling further with a per-dimension ramp plus a temperature adjustment on attention. Currently one of the stronger recipes, and it needs far fewer fine-tuning tokens.
- **Just train with a bigger RoPE base** (θ = 500k or 1M instead of 10k) from the start, which is what many recent long-context models do.

The honest caveat I'd add: **extending the window is not the same as the model actually using it.** Needle-in-a-haystack and "lost in the middle" evaluations regularly show models with nominal 128k context that degrade badly for information placed in the middle. So I'd always measure retrieval quality across position, not just check that it doesn't crash.

---

### Q46. Do decoder-only models actually need positional encoding?

Great question to have an opinion on. There's solid work showing decoder-only transformers with **no explicit positional encoding (NoPE)** still learn position, and sometimes extrapolate better than models with explicit encodings.

The reason: the causal mask itself leaks position. Token 1 attends to 1 thing, token 5 attends to 5 things. The attention distribution's entropy and normalization differ by position, so the model can infer "how far along am I" from the mask alone and build implicit positional features in the lower layers.

Encoder-only models genuinely can't do this — bidirectional attention is fully symmetric, so they *must* have explicit positions.

In practice most production decoder models still use RoPE because it's a helpful inductive bias and speeds up learning. But knowing NoPE exists — and knowing *why* it works — is a strong signal.

---

### Q47. How would you encode position for non-text sequences, like user or merchant event streams?

This is the version of the question that matters most for your work, and it's where the standard answers don't transfer cleanly.

Text positions are integers with uniform spacing. Event sequences aren't. Three things differ:

1. **Order index vs wall-clock time.** "Fifth-most-recent order" and "an order 40 days ago" are different signals. I'd usually encode **both**: an ordinal position and a time-based one.
2. **Irregular gaps carry meaning.** A merchant with 50 orders in one day is behaviorally different from one with 50 orders over a year, even though the sequences are identical positionally. So encode the **delta** between consecutive events, and typically `log(1 + Δt)` because gap distributions are heavy-tailed.
3. **Time-of-day and day-of-week are cyclic.** Encode as sin/cos pairs so 23:00 and 01:00 are close, rather than as an integer where they're maximally distant.

Concrete options: **Time2Vec** (a learned linear term plus learned periodic terms — a general-purpose time encoding), sinusoidal encoding of continuous timestamps rather than integer positions, bucketed recency embeddings (last hour / day / week / month), or a decay-based feature so old events naturally fade.

There's also a design question about whether you want **absolute recency** ("how long ago from *now*") rather than relative-between-events, because for a prediction made at serving time, staleness relative to the prediction moment is often the strongest signal — and it's the one that silently breaks if your training data computes recency relative to the label timestamp and serving computes it relative to request time. That's a real train/serve skew bug.

↪ **Your hook:** this is a strong, specific, senior-sounding answer that's directly yours. If you made any of these choices in the merchant sequence model, lead with the decision and the ablation you ran on it.

---

## Part 6 — Tokenization, Vocabulary & Embedding Tables

### Q48. Word-level vs character-level vs subword. Why did subword win?

- **Word-level** — vocabulary explodes, you get a huge softmax, and anything unseen becomes `[UNK]`, which is a total information loss. Morphologically rich languages (Finnish, Tamil, Turkish) are hopeless because every inflection is a new word.
- **Character-level** — tiny vocabulary, zero OOV, but sequences become 5-10× longer. With O(n²) attention that's a 25-100× cost increase, and the model has to spend capacity learning that "c-a-t" is a unit.
- **Subword** — the compromise. Frequent words stay whole, rare words split into meaningful pieces. `unhappiness` → `un + happi + ness`. Fixed vocabulary, no OOV, reasonable sequence lengths, and morphological structure is at least partially exposed.

Subword won because it's the only option that bounds *both* vocabulary size and sequence length.

---

### Q49. BPE vs WordPiece vs Unigram — what's the actual difference?

All three produce subwords; they differ in **how they decide the merges/splits**.

- **BPE (Byte-Pair Encoding)** — start from characters, repeatedly merge the most **frequent** adjacent pair, until you hit the target vocab size. Purely greedy frequency counting. Deterministic encoding by applying the learned merges in order. Used by GPT models. **Byte-level BPE** starts from raw bytes instead of Unicode characters, which guarantees zero OOV for any input on earth — that's why GPT-2 onward can tokenize emoji, code, and arbitrary binary-ish text without ever emitting `[UNK]`.
- **WordPiece** — same greedy merging, but the merge criterion is **likelihood gain** rather than raw frequency: it merges the pair that most increases the likelihood of the training corpus under a unigram model. In effect it favors merges where the pair occurs together more than their individual frequencies would predict. Used by BERT. Marks continuation with `##`.
- **Unigram LM (SentencePiece)** — works *backwards*. Start with a large candidate vocabulary and iteratively **prune** the tokens whose removal least hurts corpus likelihood. Because it's probabilistic, it can produce multiple valid segmentations with probabilities, which enables **subword regularization** — sample different segmentations during training as a data augmentation. That's a genuine advantage for low-resource languages.

**SentencePiece** is worth distinguishing: it's the *library*, not an algorithm — it implements both BPE and Unigram, and its key contribution is treating the input as a raw stream (encoding spaces as `▁`) so it needs no language-specific pre-tokenizer. Essential for languages without whitespace, like Thai, Chinese, Japanese.

↪ **Your hook:** for low-resource NMT, Unigram with subword regularization (BPE-dropout is the BPE equivalent) is a well-documented win, precisely because you have little data and need augmentation. If you tried it, say so; if you didn't, say it's what you'd try next and why.

---

### Q50. How do you choose vocabulary size?

It's a trade-off with real terms on both sides:

**Bigger vocab** → shorter sequences (cheaper attention, more content per context window), better handling of rare words as whole units. But: bigger embedding matrix and bigger output softmax (both `V × d`, and for a small model that can dominate the parameter count), more rarely-updated rows that stay poorly trained, and a slower softmax.

**Smaller vocab** → fewer parameters, every embedding row is well-trained, but longer sequences and more fragmentation of rare words.

The metric I'd use to decide is **fertility** — *average number of tokens per word*. Compute it per language or per domain on held-out data. If fertility for your target domain is 3.5 and for English it's 1.2, your tokenizer is effectively charging that domain 3× more for the same content.

Numbers for orientation: BERT-era English models used ~30k; GPT-2 ~50k; modern multilingual LLMs run 128k-256k. The trend is upward because compute grew and sequence length became the binding cost.

For a **domain-specific** model, training your own tokenizer often beats reusing a general one — a clinical or code or SEA-languages corpus tokenized with a general English tokenizer will have terrible fertility.

---

### Q51. What goes wrong with tokenization for low-resource and multilingual models?

This is directly your FloRes territory, so worth having sharp.

- **Vocabulary allocation is driven by data volume.** If English is 90% of your corpus, the greedy merge process spends almost all vocabulary slots on English. Low-resource languages get fragmented into near-character-level pieces — fertility of 4-8 tokens per word is common. That means those languages consume more context, more compute, and get weaker representations per unit of meaning. There's a real fairness/cost argument here too: users of under-represented languages literally pay more per API call.
- **The fix is temperature-based sampling.** Sample training data for the tokenizer (and for pretraining) with probabilities proportional to `p^(1/T)` — T=1 is natural distribution, higher T flattens it toward uniform. mBART/XLM-R use something like T=5. It upsamples low-resource languages so they get vocabulary share and training signal.
- **Script sharing helps.** Languages sharing a script (Devanagari, Latin, Cyrillic) share subwords, which gives genuine positive transfer. Languages with unique scripts get less.
- **The curse of multilinguality** — as you add languages at fixed capacity, per-language quality eventually *drops*. Positive transfer for low-resource, negative interference for high-resource. Mitigations: more capacity, language-specific adapters, or grouping related languages.

↪ **Your hook:** you can connect this to your focal loss investigation — token imbalance in long-sequence translation is *partly a tokenization artifact*. If low-resource target text fragments into many short subwords, the token distribution is dominated by a few high-frequency pieces, and that's exactly the imbalance focal loss is trying to address. Framing it that way — "the loss fix was treating a symptom whose root cause was vocabulary allocation" — is a very senior thing to say.

---

### Q52. What's weight tying and why do it?

Share the input embedding matrix with the output projection (the softmax layer), since both are `V × d`. Instead of two matrices, use one (usually transposed).

Reasons: it halves a large chunk of parameters — for a small model with a 50k vocab, embeddings can be a third of all parameters. And there's a conceptual argument: both matrices are learning a map between vocabulary and vector space, so tying them regularizes and typically *improves* perplexity, not just saves memory.

Caveats: the input embedding and the output space want slightly different scales, so implementations often scale the embedding by `√d` on input. And for very large models with huge hidden dims, the parameter savings become proportionally small and some models untie them for a little extra quality.

---

### Q53. How do you choose embedding dimensions for categorical features?

Common heuristics: `min(50, cardinality/2)`, or `cardinality^0.25` (the fast.ai-ish rule), or just powers of two chosen by tuning.

But I'd frame it by *information*, not by a formula: the dimension should be big enough to separate the entities that need separating, and small enough that rare entities don't have more parameters than they have data. A category with 8 values does not need 64 dimensions. A merchant ID space of 20M might need 64-128 — but the *tail* of that space certainly doesn't.

Which points to the more interesting answer: **use mixed dimensions.** Give high-frequency IDs large embeddings and low-frequency ones small embeddings projected up to a common size — that's the "mixed dimension embeddings" idea, and it's a real memory and quality win at recsys scale.

Also, hardware matters: pad to multiples of 8 or 16 for tensor-core efficiency. A 100-dim embedding is measurably slower than 128 on a GPU.

---

### Q54. Explain the hashing trick and its trade-offs.

Instead of maintaining an ID→row dictionary for 20M merchants, hash the ID into a fixed table of, say, 1M rows. No dictionary, fixed memory, and **new IDs work automatically** — a merchant that appeared five minutes ago hashes to some row and gets a usable embedding.

The cost is **collisions**: two merchants share a row and their gradients interfere. For head entities that's damaging; for the long tail it's often fine, because a rare merchant with three events wasn't going to learn a good embedding anyway, and sharing with other rare merchants is closer to a useful prior than to noise.

Mitigations:
- **Multiple hash functions** — hash into `k` tables and sum or concatenate the results. Collisions in all `k` simultaneously become very unlikely, so entities are distinguished by their *combination* of rows. This is the "hash embeddings" / QR-embedding family and it's the standard fix.
- **Hybrid**: dedicated rows for the top-N frequent IDs, hashing for everything else. Best of both, and what I'd default to in a production recsys.
- **Frequency thresholding** — anything below N events maps to a shared "rare" bucket plus content features.

The bigger point: at production scale, embedding tables are usually the memory bottleneck of the whole model, and hashing/quantization/mixed-dimension is where the engineering actually happens.

---

### Q55. How do you handle cold-start entities?

Ranked by how much I'd rely on each:

1. **Content-based features.** The best answer is that a new merchant is not really cold — you know its category, location, price band, menu text, opening hours. Build the representation from a **content tower** and treat the ID embedding as an *additive refinement* that only becomes meaningful once there's data. Then day-zero quality is decent and it improves smoothly.
2. **Hierarchical fallback.** Merchant embedding initialized from its category × region centroid rather than randomly. New entity starts at a sensible prior.
3. **ID dropout during training.** Randomly drop the ID embedding so the model is forced to be accurate without it. This directly trains the cold-start pathway rather than hoping it works.
4. **Meta-learning / fast adaptation** — heavier machinery, sometimes worth it.
5. **Explicit exploration.** Cold items get zero engagement, so they get no data, so they stay cold — a self-reinforcing loop. Breaking it needs deliberate exploration budget (see the bandits section), not just a better model.

That last point is the one that separates a modeling answer from a systems answer. Cold start is only half a representation problem; the other half is a data-acquisition problem.

↪ **Your hook:** merchant onboarding at Grab is a textbook cold-start pipeline. Any number you have — time-to-first-order, fraction of catalog that's new per month — makes this concrete.

### Q56. ID embeddings vs content embeddings — when do you pick which?

- **Pure ID embeddings** memorize. They're great when the entity set is stable, you have lots of interactions per entity, and behavior isn't predictable from attributes. They fail on cold start and they can't generalize at all.
- **Pure content embeddings** generalize. They handle cold start natively and transfer across markets, but they can't capture "this specific merchant is unusually popular on Friday nights for reasons no feature explains."
- **Hybrid is almost always right**: `e = f(content) + ID_embedding`, or concatenate and project. Memorization *and* generalization — this is exactly the Wide & Deep argument.

The design question worth raising: which one dominates depends on data density, and it varies *within* your catalog. Head merchants should be ID-dominated, tail merchants content-dominated. A single global architecture that gets that balance right by construction — e.g. ID dropout with frequency-dependent rate — is more elegant than two models.

---

## Part 7 — Extracting, Storing & Serving Embeddings

### Q57. You have a trained transformer. How do you get one vector per entity out of it — CLS, mean pooling, or last token?

The honest answer is: **it depends entirely on what the model was trained to do**, and the most common mistake is picking a pooling strategy that the training objective never optimized.

- **`[CLS]` token.** Works well *if* the training objective put a loss on the CLS position — e.g. a classification head, or NSP, or a contrastive objective on CLS. Then CLS has been explicitly trained to summarize. If you take an off-the-shelf MLM-only BERT and use raw CLS, it's typically **worse than averaging GloVe vectors**, which is a famous and slightly embarrassing result. That's because the only gradient CLS ever got was from NSP, which RoBERTa showed was a near-useless objective.
- **Mean pooling.** Average the token vectors, masked properly. Robust, no training required, and usually the best zero-shot default. It's what Sentence-BERT settled on after testing alternatives.
- **Last token.** The right choice for a **decoder-only** model, because with causal attention only the final position has seen the whole sequence. Every earlier token is a partial-context representation. This is why decoder-based embedding models use last-token pooling (often with an explicit `<EOS>`) — or remove the causal mask entirely during embedding, as LLM2Vec does.
- **Max pooling / attention pooling / a learned pooler.** Attention pooling — a small learned query that attends over the token outputs — is often the best if you're training the pooling anyway. It lets the model learn *which* events matter rather than treating all of them equally.

**The pitfall I'd call out explicitly:** mean pooling must be **mask-aware**. If you average over the padded positions too, then the same entity produces a different embedding depending on the batch it was padded in. I've seen this ship. It's silent — the embeddings look fine, the retrieval quality is just mysteriously mediocre, and the bug only shows up if you compare an entity embedded alone against the same entity embedded in a padded batch. Cheap regression test: **embed the same entity twice under different padding and assert the vectors match.**

For **entity sequence models** like yours, there's a fourth option people forget: a **dedicated entity token** prepended to the sequence (a CLS by another name), *trained with a loss on it*. If you want a merchant vector, giving the merchant its own token and putting a contrastive or predictive loss there is much better than hoping mean-pooled event vectors happen to be a good merchant summary.

---

### Q58. Which layer should you pull embeddings from?

Not automatically the last one. There's consistent evidence that the **last layer is over-specialized to the pretraining objective** — for MLM, the final layers are shaped for predicting masked tokens, which isn't the same as representing meaning.

Empirically the **middle-to-upper-middle layers** often give the best general-purpose representations. Lower layers are more lexical/surface, middle layers are more semantic/syntactic, top layers are task-specific.

Options: pick a layer by validating on your downstream task; concatenate a few layers; or learn a weighted sum across layers (the ELMo-style scalar mix), which is nice because the learned weights tell you *which* layers your task actually cares about — a free interpretability signal.

The practical answer for an interview: **I treat "which layer" as a hyperparameter and validate it, and I've seen the answer move by several points of downstream metric.** That framing is better than asserting a number, because it shows you measured.

↪ **Your hook:** if you ran layer ablations for the merchant embeddings, this is a great thing to have quantified. It's exactly the kind of ablation your CV mentions.

---

### Q59. Should you L2-normalize embeddings? Cosine or dot product?

**Cosine similarity** = dot product after L2-normalizing both vectors, so it only measures *direction*. **Dot product** keeps magnitude in play.

That difference matters more than people expect:

- In many trained models, **embedding norm correlates with frequency or popularity.** So dot-product retrieval will systematically favor popular items regardless of relevance — magnitude becomes a hidden popularity prior. Sometimes you want that (popularity genuinely predicts engagement). Usually you don't want it *unmodeled*.
- **Normalize if** you want pure semantic similarity, you're using an ANN index tuned for cosine, or you're doing contrastive learning (almost all contrastive objectives normalize, which is why they pair with a temperature).
- **Don't normalize if** magnitude carries real signal you want — e.g. a confidence or intensity — and you've verified it's not just frequency.

Big consistency requirement: **whatever you choose must match between training, index building, and query time.** Training with cosine and serving with dot product is a classic silent quality killer. Same for the ANN index metric.

One more: if you normalize, note that **maximum inner product search becomes equivalent to nearest-neighbor search**, which simplifies your ANN choices considerably.

---

### Q60. How do you evaluate an embedding, especially without a downstream task?

Two layers, and I'd want both.

**Intrinsic (properties of the space itself):**
- **Alignment** — do things that should be similar land close? Measured on known positive pairs.
- **Uniformity** — do embeddings spread over the space, or clump? Measured as the log of average pairwise Gaussian potential.
- **Average cosine similarity between random pairs.** If random entities have cosine 0.9, your space is anisotropic and nearly degenerate. Healthy is near 0.
- **Effective rank / singular value spectrum.** Run SVD on a sample of embeddings. If 90% of variance is in 5 of 128 dimensions, you have dimensional collapse.
- **Nearest-neighbor sanity checks.** Pick 20 entities you know well, look at their top-10 neighbors, and read them. Unglamorous and consistently the fastest way to find that something is broken.

**Extrinsic (does it help?):**
- Frozen-embedding probes on downstream tasks — train a linear model or small MLP on top and measure. This is the real test, and it's what your `emb-bench` system sounds like it's for.
- Retrieval metrics — Recall@k, NDCG on a held-out task.
- **Stability over retraining** — retrain the model and check whether the neighbor sets are stable. Unstable embeddings break every downstream consumer.

The senior framing: **the embedding is a product with consumers, so it needs an SLA, not just a metric.** Which leads directly to the next question.

---

### Q61. How do you productionize embeddings? Walk me through the system concerns.

This is where an L4 answer separates from an L3 answer, so I'd take my time here.

**Versioning and compatibility.** Embedding spaces from two training runs are **not comparable** — even with identical data and hyperparameters, different random seeds produce different rotations of the space. So a v2 embedding cannot be compared against a v1 embedding, and any downstream model trained on v1 will silently degrade if you swap in v2 without retraining it. Concretely: version the embedding namespace, never overwrite in place, and treat an embedding refresh as a **coordinated migration** across every consumer, not a data update. Alternatives if that's too expensive: anchor the space (regularize toward the previous version), or learn an alignment/rotation matrix between versions (Procrustes-style).

**Freshness vs stability.** Two update modes: full periodic retrain (stable, stale) and incremental/streaming inference (fresh, drifting). Most systems do both — periodic full retrains for the space itself, plus fresh inference for entities whose features changed. You need an explicit policy on how stale an embedding is allowed to be.

**Backfill.** When you ship v2, you need to re-embed the entire catalog. At 20M merchants that's a real batch job with a cost and a duration, and it needs to complete before the consumers cut over. Plan the compute.

**Serving path.** Precomputed lookup (fast, stale) vs on-the-fly inference (fresh, expensive). For entities, precompute; for queries or sessions, compute live. That asymmetry is exactly the two-tower design.

**ANN index.** HNSW or IVF-PQ depending on scale and memory. Index build time, recall-vs-latency tuning, and the fact that **your ANN recall is a silent quality cap** — if the index returns 90% recall@100, you've lost 10% before ranking even runs. Measure it separately from model quality; people conflate them constantly.

**Monitoring.** Track embedding norm distribution, average pairwise similarity, effective rank, and neighbor-set stability over time. Drift in these is your early warning that either the data or the model has shifted. Also monitor the *coverage* — what fraction of requests hit an entity with no embedding.

**Train/serve consistency.** Same preprocessing, same feature computation, same pooling, same normalization. Ideally the same code path, not a reimplementation. This is the single most common source of "offline great, online mediocre."

↪ **Your hook:** this whole answer is your CV — 20 years of data systems plus embedding work is *exactly* the profile that should crush this question. Most ML candidates give a modeling answer here. Give the systems answer; it's your differentiator for an L4 role.

---

## Part 8 — Collapse, Anisotropy & Contrastive Learning

### Q62. What is representation collapse, and what are its varieties?

Collapse is when the model finds a shortcut that satisfies the loss while destroying the information you wanted.

- **Complete collapse** — every input maps to (nearly) the same vector. Classic in non-contrastive self-supervised learning: if the objective is "make two views of the same thing similar," outputting a constant is a perfect solution. Detect it by average pairwise cosine → 1, or embedding variance → 0.
- **Dimensional collapse** — the embeddings don't all coincide, but they occupy a low-dimensional subspace of the space you allocated. You paid for 128 dimensions and are effectively using 12. Detect it via the **singular value spectrum**: plot the sorted singular values of your embedding matrix; a sharp cliff means collapse. This one is insidious because all your loss curves look fine.
- **Cluster/mode collapse in generative models** — a GAN generator produces only a few modes of the data distribution because that's enough to fool the discriminator. Different mechanism, same word, so it's worth distinguishing them explicitly if asked.
- **Neural collapse** — a well-studied end-state of long classification training where each class's features converge to their class mean and the means arrange into a maximally-separated simplex. It's *good* for classification accuracy and *bad* if you want the features for anything else, because within-class variation has been deliberately annihilated. This is the theoretical version of the label-smoothing warning from Q20.

---

### Q63. What is anisotropy in embedding space and why does it happen?

**Anisotropy** means the embeddings occupy a narrow cone rather than spreading through the space — *anisotropic just means "not the same in all directions."* The symptom: two randomly-chosen, semantically unrelated items have cosine similarity of 0.6, 0.8, sometimes 0.95. Since everything is similar to everything, cosine similarity loses its discriminative power, and downstream retrieval degrades badly.

This isn't hypothetical — it's the documented default behavior of BERT, GPT-2, and most language models trained with only MLM or CLM.

Why it happens: the leading explanation is the **word frequency / softmax bottleneck** effect. In the output softmax, rare tokens get pushed in a consistent direction (they're almost always the negative class, so they get pushed away from the common region), while frequent tokens cluster elsewhere. Frequency ends up encoded as a dominant global direction, and that direction dominates the variance for every embedding. There's a common-direction component shared by all embeddings that swamps the actual semantic differences.

The key insight, and this is the one to land: **a language modeling objective never asks for a well-shaped similarity space.** It asks for good next-token or masked-token prediction. Getting a good similarity space is a *separate* requirement that needs a *separate* objective or a post-hoc fix. That's why "just take BERT embeddings and use cosine" underperforms and why SBERT/SimCSE exist.

---

### Q64. How do you fix anisotropy? Explain whitening.

Two families: post-hoc corrections, and training-time fixes.

**Post-hoc:**

- **Remove the top principal components.** The "all-but-the-top" approach — subtract the mean and remove the first few PCs, which are usually the frequency/common direction rather than semantics. Crude, cheap, and often gives a surprisingly large gain.
- **Whitening.** Transform the embeddings so the covariance matrix becomes the identity — *i.e. zero mean, unit variance in every direction, and no correlation between dimensions.* Concretely: center the embeddings, compute the covariance, and multiply by `W = Σ^(-1/2)` (via SVD or Cholesky). Every direction now contributes equally, so the cone is expanded into a ball and cosine becomes meaningful again. **BERT-whitening** showed this matches the more complicated **BERT-flow** (which learned an invertible normalizing flow to map embeddings to a Gaussian) at a fraction of the complexity.
- Practical whitening caveats worth mentioning: it's fit on a data sample, so it's a **learned artifact you have to version and ship alongside the model**; near-zero eigenvalues blow up under `Σ^(-1/2)` so you need shrinkage or a small ridge term; and it's a linear transform, so it can't fix nonlinear structure. Also, whitening can *amplify noise* by boosting low-variance directions that were low-variance for good reason.

**Training-time (better, if you can):**

- **Contrastive learning** with in-batch negatives directly optimizes for a spread-out space — the uniformity term does exactly the anti-anisotropy job. SimCSE is the minimal version: two dropout passes over the same sentence as a positive pair, everything else in the batch as negatives. It's almost embarrassingly simple and it works.
- **Regularizers that penalize off-diagonal covariance** — Barlow Twins and VICReg build this directly into the loss.

My default recommendation: if I control training, fix it in training. If I'm consuming a frozen model, whiten. And I'd measure it before and after with average random-pair cosine and the singular value spectrum, so the fix is verified rather than assumed.

---

### Q65. Explain contrastive learning and the InfoNCE loss.

The idea: pull together things that should be similar, push apart things that shouldn't. You need positive pairs, negatives, and a similarity function.

**InfoNCE** is a softmax over similarities: for an anchor, the loss is the negative log probability of picking the true positive out of the positive plus all the negatives, with similarities scaled by a temperature. Effectively it's a classification problem where the classes are "which of these candidates is the real match."

Two forces fall out of it, and this is the Wang & Isola decomposition worth citing:
- **Alignment** — positives get pulled together.
- **Uniformity** — everything else gets pushed apart, spreading embeddings over the hypersphere.

That second term is why contrastive learning fixes anisotropy for free.

**Temperature (τ)** is the hyperparameter that matters most and that people undertune. It divides the similarities before the softmax. Low τ (0.01-0.05) makes the distribution peaky, so the loss concentrates almost entirely on the **hardest** negatives — strong separation, but very sensitive to false negatives and can be unstable. High τ (0.5+) treats all negatives more equally — smoother, more tolerant, but weaker separation. There's a documented **uniformity-tolerance trade-off**: low temperature produces a more uniform space but is less tolerant of semantically similar items being labeled as negatives. I'd tune τ explicitly and I'd expect it to matter more than the architecture.

---

### Q66. Talk about negatives. In-batch, hard, false negatives.

Negatives are where most of the quality lives in a contrastive system.

- **In-batch negatives** — use the other examples in the batch as negatives. Free (no extra forward passes) and effective. This is why **contrastive learning loves large batches** — batch size *is* your number of negatives, and quality scales with it. Techniques to get more without the memory: MoCo's momentum queue (a rolling bank of encoded negatives from previous batches), or cross-GPU gathering of in-batch negatives, which is basically free in a distributed setup and often overlooked.
- **Hard negatives** — items that are similar but wrong. These carry far more gradient signal than random negatives, because a random negative is already easy and contributes almost nothing. Mining them (e.g. retrieve top-k with the current model, exclude known positives) is usually a bigger win than any architecture change. ANCE and RocketQA are the canonical references.
- **False negatives** — and this is the trap. If you mine hard negatives from an unlabeled corpus, many of your "hard negatives" are actually **unlabeled positives**. You are now explicitly training the model that a correct answer is wrong. This is the number one failure of naive hard negative mining. Mitigations: denoise with a cross-encoder or a stronger teacher model, sample from a mid-range of ranks rather than the very top, use a higher temperature, or use a loss that tolerates label noise.
- **Popularity/frequency bias** — with in-batch negatives sampled from traffic, popular items appear as negatives far more often, so they get systematically pushed down. The fix is the **logQ correction** (sampled softmax bias correction): subtract `log(sampling probability)` from the logit of each negative to make the sampled softmax an unbiased estimate of the full one. This is standard in production two-tower retrieval and is a great specific detail to bring up.

↪ **Your hook:** if your merchant model used any contrastive component, the negatives design is the most interesting thing to talk about. If it used masked modeling instead, you can say why — and "I chose masked sequence modeling because defining a clean positive pair for a merchant is genuinely ambiguous, whereas masking gives dense supervision for free" is a strong, defensible answer.

---

### Q67. How do non-contrastive methods avoid collapse without negatives?

Worth knowing because the obvious question is "if you only pull positives together, why doesn't everything collapse to a constant?"

- **BYOL** — two networks, an online one and a target one. The target is an exponential moving average of the online network, and there's a predictor head on the online side only, plus a stop-gradient. The asymmetry is what prevents collapse; the exact reason was debated for a while and it seems to be a subtle optimization dynamic rather than an explicit anti-collapse term.
- **SimSiam** — shows you don't even need the momentum encoder; a stop-gradient plus a predictor head is enough. Very clean ablation of what actually matters.
- **Barlow Twins** — make the **cross-correlation matrix** between two views' embeddings equal the identity. The diagonal term enforces invariance (same feature agrees across views), the off-diagonal term enforces **decorrelation**, which directly prevents dimensional collapse. No negatives needed at all, and it's very interpretable.
- **VICReg** — three explicit terms: **V**ariance (each dimension must have variance above a threshold — this directly forbids complete collapse), **I**nvariance (positives agree), **C**ovariance (decorrelate dimensions). It's essentially "write down what you want the space to look like and regularize toward it," which I find the most intellectually honest of the family.

Why it matters practically: these methods work with **small batches**, which contrastive methods struggle with. If you're compute-constrained or your positive pairs are expensive, this family is worth trying.

---

### Q68. How would you detect collapse in a training run, live?

I'd log these every N steps, on a fixed held-out sample of entities so the numbers are comparable across steps:

1. **Mean and std of embedding L2 norms.** Norms collapsing toward zero or exploding are both bad.
2. **Average pairwise cosine similarity of random pairs.** Trending toward 1 = collapse/anisotropy. Should sit near 0 for a healthy normalized space.
3. **Singular value spectrum / effective rank.** The single most informative one. Log the ratio of the top-k singular values to the total, or the entropy of the normalized spectrum. A slow decline in effective rank over training is dimensional collapse in progress.
4. **Alignment and uniformity** if you have positive pairs.
5. **Downstream probe metric** on a small frozen task, run periodically. Slower but it's the ground truth.

The important part of the answer: **loss going down does not rule out collapse.** In several of these regimes the loss looks perfectly healthy while the representation degrades. So these have to be first-class training metrics, not a post-hoc investigation. I'd wire them into the same MLflow tracking as loss and treat a falling effective rank as a training alert.

↪ **Your hook:** this is exactly what a benchmarking system like `emb-bench` is for. Framing it as "I built this because loss curves don't tell you whether your representation is healthy" is a much stronger narrative than "I built a benchmarking tool."

---

## Part 9 — Dimensionality Reduction

### Q69. Explain PCA properly.

PCA finds the directions along which the data varies most, and re-expresses the data in terms of those directions.

Mechanically: center the data, compute the covariance matrix, take its eigenvectors and eigenvalues. The eigenvectors are the principal components (the new axes), the eigenvalues are how much variance each captures. Sort by eigenvalue, keep the top `k`, project onto them. In practice you use SVD on the centered data matrix instead of forming the covariance explicitly — it's more numerically stable and doesn't require an `n×n` or `d×d` intermediate.

Three properties worth stating because they're what make PCA the default:

1. It's the **optimal linear** reduction in a precise sense — it minimizes reconstruction error in Frobenius norm (ties back to Q18). No other rank-k linear projection does better.
2. The components are **orthogonal and uncorrelated**, so you've removed linear redundancy.
3. It's **deterministic and cheap**, and it gives you an invertible-ish map back, so you can reconstruct and inspect.

Requirements people forget: **you must center**, or the first component just points at the mean. And you should usually **standardize** if features are on different scales, or PCA will just find whichever feature has the biggest units.

---

### Q70. What is whitening, and PCA whitening vs ZCA?

**PCA whitening** = project onto the principal components, then divide each by its standard deviation (the square root of the eigenvalue). Result: unit variance in every direction, zero correlation. The covariance becomes the identity.

**ZCA whitening** does the same but then **rotates back** into the original coordinate frame. It's the whitening transform that stays closest to the original data — useful for images, where you want whitened data that still looks like an image, and for cases where the original axes are interpretable.

Both are whitening; they differ by an arbitrary rotation, since whitening is only defined up to rotation.

When it helps: making cosine similarity meaningful in an anisotropic embedding space (Q64), decorrelating inputs so gradient descent converges faster on a better-conditioned surface, and as preprocessing before methods that assume isotropy.

When it hurts, and I'd want to say this part: whitening **boosts low-variance directions to the same scale as high-variance ones.** If those directions were noise — and low-variance directions frequently are — you've just amplified noise to the level of signal. That's why practical whitening uses shrinkage: add a small `ε` to the eigenvalues, or cap the number of components. Whitening without regularization on an ill-conditioned covariance is a good way to make things worse and be confused about it.

---

### Q71. When does PCA fail?

- **Nonlinear structure.** PCA is linear. A spiral, a manifold, or an XOR-like structure won't be captured — PCA will happily report that the variance is spread over many dimensions when actually the data lies on a simple 1D curve.
- **Variance ≠ importance.** This is the big one. PCA optimizes for variance, which is not the same as optimizing for what you care about. A high-variance direction can be pure noise (a broken sensor with huge range), and the signal you need can live in a low-variance direction that PCA discards first. For a *supervised* problem, **LDA** (which maximizes class separation) or a supervised feature selection method can be much better.
- **Outliers.** Covariance is very sensitive to them; a handful of extreme points can rotate your entire component basis. Robust PCA exists for this.
- **Scale sensitivity** without standardization.
- **Interpretability.** Components are dense linear combinations of everything, so "PC3" usually means nothing to a human. Sparse PCA or NMF (which gives non-negative, parts-based, additive components) are more interpretable alternatives.
- **It's not a clustering method**, though people use it as one and then over-read the plot.

---

### Q72. PCA vs t-SNE vs UMAP — when do you use each?

The critical framing: **PCA is a data transformation; t-SNE and UMAP are visualization tools.** They are not interchangeable, and most mistakes come from treating them as if they were.

- **PCA** — linear, fast, deterministic, preserves global structure, gives you a reusable projection matrix you can apply to new data. Use it for actual dimensionality reduction in a pipeline: compressing features, denoising, preprocessing before clustering or ANN.
- **t-SNE** — nonlinear, focuses on preserving *local* neighborhoods. Beautiful cluster visualizations. But: **distances between clusters and cluster sizes in a t-SNE plot are not meaningful.** The `perplexity` hyperparameter changes the picture substantially, different random seeds give different layouts, and it doesn't naturally embed new points. It's O(n log n) with Barnes-Hut but still slow at scale. Use it to *look* at 10k points, never as a feature transform.
- **UMAP** — also nonlinear and neighborhood-based, but grounded in a topological construction. Faster than t-SNE, preserves more global structure, scales to millions of points, and importantly it **can transform new data** with the fitted model. Also supports supervised and semi-supervised variants. It's my default over t-SNE. It still has the caveat that inter-cluster distances are only loosely meaningful, and it can produce spurious cluster separation on data that's actually continuous.

The senior addition: **for large embedding sets I'd run PCA first (to say 50 dims) and then UMAP.** It's the standard recipe — denoises, massively speeds up the neighbor search, and usually improves the result.

And the warning I'd give a team: never make a modeling decision from a UMAP plot alone. Validate with a metric.

---

### Q73. What other dimensionality reduction options should be on the table?

- **Random projection.** The Johnson-Lindenstrauss lemma says you can project into `O(log n / ε²)` dimensions with a random matrix and preserve all pairwise distances within `1±ε`. Requires no training and no data pass. Startlingly effective and criminally underused when you just need to make things smaller fast.
- **Autoencoders.** A nonlinear generalization of PCA — in fact a linear autoencoder with squared loss recovers the PCA subspace. Learns nonlinear structure, but needs training, tuning, and can overfit. Worth it when the structure genuinely is nonlinear and you have data.
- **Matrix factorization (SVD/ALS/NMF)** — the classic for interaction matrices in recsys. NMF for interpretable, additive, parts-based components.
- **Feature selection** rather than transformation, when you need interpretability or when downstream cost is per-feature.
- **Matryoshka Representation Learning (MRL).** Train so that the first 64 dimensions are *by themselves* a good embedding, the first 128 better, the full 768 best — via a loss applied at multiple nested truncations. Then at serving time you can simply slice the vector to whatever dimension your latency budget allows, with no retraining and no separate projection. This is genuinely elegant and it's the modern answer to "we need a smaller embedding" — worth naming because it signals you're current.
- **Quantization** — a different axis of compression. Scalar quantization to int8 gives 4× with minimal loss; **product quantization (PQ)** splits the vector into subvectors and codebook-encodes each, giving 10-50× compression for ANN indexes. Note PCA is often applied *before* PQ to decorrelate dimensions, because PQ assumes subvectors are roughly independent — a nice example of these composing.

The framing I'd use: ask what the reduction is *for*. Storage, latency, denoising, visualization, and downstream accuracy all point to different tools.

---

## Part 10 — Ranking & Evaluation

### Q74. Explain NDCG.

NDCG answers: how good is this ranked list, given that (a) more relevant items should be higher, and (b) higher positions matter more?

Build it in three steps:

1. **Gain** — each item has a relevance score. Usually `2^rel - 1`, so higher relevance grades count disproportionately.
2. **Discount** — divide by `log2(position + 1)`, so an item at rank 1 counts fully and rank 10 counts about a third. Sum these up: that's **DCG**.
3. **Normalize** — divide by the DCG of the ideal ranking (the same items sorted perfectly). That gives **NDCG**, between 0 and 1, so you can average across queries with different numbers of relevant items.

Why the normalization matters: without it, a query with 50 relevant items scores much higher than one with 2, and averaging across queries becomes meaningless.

Things worth knowing that people miss:
- **The log discount is a modeling assumption**, not a fact. It encodes a specific belief about how attention decays with position. If your UI is a 3-item carousel, that discount curve is wrong for you.
- **NDCG@k truncation** interacts with the ideal-DCG computation, and different libraries handle the case of "fewer relevant items than k" differently. Check your implementation.
- **It's not differentiable** (it depends on sort order), which is why learning-to-rank uses surrogates: LambdaRank/LambdaMART weight pairwise gradients by the NDCG change from swapping the pair — a neat trick that optimizes the metric without differentiating it.

↪ **Your hook:** your CV says NDCG improvements on downstream tasks. Be ready for "how much, at what k, on what baseline, and was it significant?" Have the number and the confidence interval.

---

### Q75. NDCG vs MRR vs MAP vs Recall@k — when does each apply?

Pick the metric that matches the user's actual behavior:

- **Recall@k** — did we get the relevant items into the candidate set at all? The right metric for a **retrieval/candidate-generation stage**, where order doesn't matter yet because a ranker comes after. This is the one I'd use to evaluate a two-tower retriever.
- **MRR (Mean Reciprocal Rank)** — `1/rank` of the *first* relevant item. Right when there's essentially one correct answer and the user stops when they find it: navigational search, QA, "find my order."
- **MAP** — averages precision at each relevant item's position. Good for binary relevance with multiple relevant items.
- **NDCG** — handles **graded** relevance (a 5-star match vs a 3-star match) and multiple relevant items. The default for ranking with rich labels.
- **Hit Rate@k** — did any relevant item appear in the top k. Blunt, but interpretable to non-ML stakeholders.

The point to make: metrics encode a **user model**. NDCG assumes the user scans down with decaying attention; MRR assumes they stop at the first hit. Choosing the metric is a product decision disguised as a technical one, and I'd want to make it explicitly with the PM rather than defaulting to NDCG because it's standard.

---

### Q76. Your offline NDCG improved 5% but the A/B test was flat. What happened?

I'd work through the likely causes in order of how often they're the culprit:

1. **Train/serve skew.** Different feature computation, different preprocessing, a feature that's fresh offline and stale online, a different pooling or normalization. Check by scoring the *same* requests through both paths and diffing the outputs. This finds it more often than anything else.
2. **Position and presentation bias in the offline labels.** Your offline data was logged under the *old* model's policy. Items the old model never showed have no positive labels, so the offline metric rewards agreeing with the old model. A genuinely better model that surfaces different items will look worse offline and can look better online — or the reverse. This is the fundamental offline-evaluation problem.
3. **Candidate set mismatch.** Offline you reranked a fixed logged candidate set; online your retrieval stage feeds different candidates. You improved a stage that wasn't the bottleneck.
4. **The metric isn't the business objective.** NDCG on clicks improved; the experiment measured orders or GMV. Better click prediction can genuinely fail to move revenue — or move clicks toward cheaper items.
5. **Not enough power.** A 5% NDCG lift may translate to a 0.1% metric change that your test cannot detect. Check the MDE (minimum detectable effect) *before* concluding flat. "Flat" and "underpowered" look identical.
6. **Latency.** The new model is slower, and the latency cost ate the quality gain. Very common with bigger models.
7. **Novelty/feedback effects** — the metric moves for a week then reverts, or vice versa.
8. **Segment cancellation.** Big win in one segment, loss in another, netting to zero. Always cut the results.

The senior version of this answer ends with process, not diagnosis: **I'd want the offline evaluation to be predictive of online results, and if it isn't, that's a bug in the evaluation system, which is a higher priority than any individual model.** Then I'd invest in counterfactual/off-policy offline evaluation and in logging exploration data specifically to de-bias it.

↪ **Your hook:** this is your experimentation-platform experience. You've been on the *other* side of this — you built the systems that adjudicate it. Say so.

---

### Q77. What is position bias and how do you correct for it?

Users click the top result more often *because it's at the top*, not only because it's better. So your click logs conflate relevance with position. Train naively on clicks and you learn to reproduce the existing ranking — a self-fulfilling loop.

Corrections:

- **Inverse Propensity Scoring (IPS)** — weight each observed click by `1/P(examined | position)`. A click at position 10 is rare, so it counts for more. This gives an unbiased estimate of relevance under some assumptions. The catch: **high variance** when propensities are small, so you clip/cap the weights, which trades a little bias for a lot of variance reduction.
- **Estimating the propensities** — you need `P(examined|position)`. Gold standard is **result randomization** (swap positions occasionally), which costs you some user experience. Cheaper: intervention harvesting from natural ranking variation, or an EM-based joint estimate.
- **Two-tower / position-as-a-feature debiasing** — train a model with position as an input feature in a *separate* shallow tower, then at serving time set it to a constant. The position tower absorbs the positional effect; the main tower learns relevance. Cheap, no randomization needed, widely used in production, and my usual first choice.
- **Pairwise comparisons within the same position** to sidestep the issue.

Related biases in the same family, worth naming: **selection bias** (you only observe items you showed), **trust bias** (users trust top results and click them even when less relevant), and **presentation bias** (image quality, badges).

---

### Q78. How do you handle multi-task ranking?

You're predicting several things at once — click, conversion, dwell time, rating, cancellation — usually because the business objective is a combination and because tasks share signal.

**Architecture options:**
- **Shared bottom + per-task heads.** Simplest. Risk of **negative transfer** — tasks fighting over shared capacity.
- **MMoE (Multi-gate Mixture-of-Experts).** Shared experts, a per-task gate that learns which experts each task uses. Handles loosely-related tasks much better than a hard shared bottom. This is the standard in production recsys.
- **PLE / CGC** — adds task-*specific* experts alongside shared ones, explicitly separating what's shared from what isn't. Usually beats MMoE when task correlation is low.

**Loss weighting — the real problem:**
- Fixed weights, tuned. Works, expensive to tune, brittle as data shifts.
- **Uncertainty weighting (Kendall et al.)** — learn a per-task weight parameterized as a task noise term, so noisy tasks get down-weighted automatically.
- **GradNorm** — balance the *gradient magnitudes* across tasks rather than the loss values, which is more principled since losses on different scales aren't comparable.
- **PCGrad** — when two tasks' gradients conflict (negative cosine similarity), project one onto the normal plane of the other so they stop directly fighting. Good to name.

**The practical issues people hit:**
- **Different label densities.** Clicks are 100× more common than conversions. The dense task dominates the shared representation. Solutions: re-weight, or use separate batches/samplers per task.
- **Label delay.** Conversion arrives hours or days after the click. So the "label" for recent data is systematically incomplete, and if you train on it naively you'll learn that recent items don't convert. This is a genuinely nasty and very common bug — handle it with delayed-feedback modeling or an attribution window with a holdout.
- **Combining at serving time.** You end up with a formula like `score = p(click)^a · p(convert)^b · value^c`. Those exponents are a business decision, they need tuning online, and **your model scores must be calibrated for that formula to mean anything.**

---

### Q79. What does it mean for a model to be calibrated, and when do you care?

Calibrated means: among all the cases where the model says 0.3, about 30% are actually positive. It's about the *probabilities being trustworthy*, not about ranking.

You **don't** need calibration if you only ever sort by the score. You **absolutely** need it when:
- You combine scores in a formula (like the multi-task one above) — multiplying two uncalibrated scores is meaningless.
- You multiply by a value/price to get expected revenue — this is bidding, and miscalibration directly costs money.
- You threshold for a decision ("auto-approve above 0.9").
- A human reads the number.

Why models are miscalibrated: negative downsampling (very common in ads/recsys — it shifts the base rate and you must correct it back with the known sampling rate), class imbalance handling, label smoothing, focal loss (which deliberately trades calibration for focus), and modern deep nets simply being overconfident.

Fixes: **Platt scaling** (fit a logistic regression on the logits, one or two parameters), **isotonic regression** (nonparametric, more flexible, needs more data, can overfit), or **temperature scaling** (single parameter dividing the logits — the standard for neural nets, preserves ranking exactly, so it can never hurt your NDCG). All fit on a **held-out** set, never the training set.

Measure with **ECE (Expected Calibration Error)** — bin predictions and compare average predicted vs actual rate — plus a reliability diagram, which is more informative than the single number. Check calibration **per segment**, because a model can be perfectly calibrated overall and badly miscalibrated for new users or a specific city.

---

## Part 11 — Distributed Training & Systems

### Q80. DataParallel vs DistributedDataParallel vs FSDP/ZeRO — explain the progression.

- **DataParallel (DP)** — single process, multiple GPUs, one Python thread scattering batches and gathering outputs. Bottlenecked by the GIL and by GPU 0 doing all the gathering. Effectively deprecated; if I see it in a codebase I'd change it.
- **DistributedDataParallel (DDP)** — one process per GPU, each with a full model replica. Each computes gradients on its own shard of the batch, then an **all-reduce** averages gradients across all ranks. Crucially, the all-reduce is **overlapped with the backward pass** — gradients for later layers are ready first and start communicating while earlier layers are still computing, which hides most of the communication cost. This is the workhorse. Requirement: the model plus optimizer states must fit on one GPU.
- **ZeRO / FSDP** — attacks the memory redundancy in DDP, where every GPU stores identical copies of everything. Three stages:
  - **Stage 1**: shard the optimizer states. For Adam, that's `m` and `v` — 8 bytes per parameter in fp32, often more memory than the model itself. Biggest win per unit of complexity.
  - **Stage 2**: also shard the gradients.
  - **Stage 3 (= FSDP)**: also shard the **parameters**. Each GPU holds only a slice; before a layer runs, an all-gather reconstructs the full weights for that layer, then they're freed again.
  
  So FSDP trades **communication for memory** — you can train a model far larger than one GPU, at the cost of gathering weights on every forward and backward. With enough interconnect bandwidth (NVLink, InfiniBand) that trade is very favorable.

The decision rule I'd state: **model fits comfortably → DDP. Optimizer states are the problem → ZeRO-1/2. Model itself doesn't fit → FSDP/ZeRO-3.** Don't reach for the complex one first; DDP is faster when it's viable.

---

### Q81. What about tensor, pipeline, and sequence parallelism?

Those shard along different axes, and at large scale you combine them ("3D parallelism").

- **Tensor parallel** — split individual matrices across GPUs. A 4096×4096 matmul becomes four 4096×1024 matmuls with an all-reduce to combine. Very communication-heavy (every layer, every step), so it's used **within a node** where NVLink bandwidth is high.
- **Pipeline parallel** — put different *layers* on different GPUs and stream micro-batches through. Low communication (only activations at the boundaries), so it works **across nodes**. The problem is the **pipeline bubble** — idle time while the pipeline fills and drains. Interleaved schedules (1F1B) reduce it. Bubble fraction is roughly `(stages-1)/micro_batches`, so you want many micro-batches.
- **Sequence/context parallel** — shard the sequence dimension. Necessary for very long context, where activations for a single sequence exceed one GPU. Ring Attention is the notable implementation.
- **Expert parallel** — for MoE, put different experts on different GPUs; routing becomes an all-to-all.

Typical large-scale layout: tensor parallel within a node, pipeline parallel across nodes, data parallel across the whole thing.

---

### Q82. What is gradient checkpointing / activation recomputation?

During the forward pass you normally save every intermediate activation because backward needs them. For a long sequence and a deep model, activations — not parameters — are usually what fills the GPU.

Gradient checkpointing saves only a subset (say, one per transformer block) and **recomputes** the rest during backward. Memory drops roughly from O(L) to O(√L) with optimal placement, at the cost of about one extra forward pass — typically 20-30% more compute.

It's a straight memory-for-compute trade, and it's almost always worth it when it lets you increase batch size enough to more than recover the throughput, or when it's the difference between fitting and not fitting. Selective checkpointing (recompute cheap ops, store expensive ones like attention outputs) is the refined version.

Same underlying idea as FlashAttention's backward pass, which is a nice connection to draw.

---

### Q83. Your GPUs are at 40% utilization. What do you do?

I'd resist the urge to touch the model and profile first. The usual culprits, roughly in order of frequency:

1. **The data loader.** By far the most common. Too few workers, CPU-bound preprocessing, decoding or tokenizing on the fly, no prefetch, no pinned memory. Check whether GPU idle time correlates with batch fetches. Fixes: more workers, `pin_memory=True`, `persistent_workers`, prefetch factor, precompute/cache preprocessing, move preprocessing to GPU, or switch to a columnar format optimized for sequential reads.
2. **Storage/IO.** Reading millions of small files from network storage. Fix with sharded formats (WebDataset, Parquet, TFRecord-style) and sequential reads.
3. **Small batch size** — kernels too small to saturate the GPU. Increase batch, or use gradient accumulation less and real batch more.
4. **Communication not overlapped** — check whether all-reduce is blocking. Bucket sizes matter in DDP.
5. **Synchronization points in the training loop** — a `.item()`, a `.cpu()`, or a print inside the loop forces a sync every step and kills pipelining. Extremely common and easy to fix.
6. **Not using mixed precision**, or not using fused kernels/`torch.compile`.
7. **Stragglers** — in a distributed job, every rank waits for the slowest. One bad node, or uneven shard sizes, throttles the whole cluster. Log per-rank step times.

The tooling answer: PyTorch profiler with the trace viewer, or Nsight, plus `nvidia-smi dmon` for a quick read on whether it's compute, memory, or idle.

↪ **Your hook:** "reducing training time by 75%" is on your CV. Be extremely ready to explain *what the bottleneck actually was* and how you identified it. Interviewers will push on whether that number came from real optimization or just from adding GPUs — have the breakdown.

---

### Q84. What breaks when you scale to very large batch sizes?

Large batches give better hardware utilization, but there's a well-known **generalization gap** — past a critical size, quality degrades.

Explanations: fewer optimizer steps for the same number of epochs; less gradient noise, and that noise seems to help find flatter minima; and the fact that the gradient becomes so accurate that further batch increases add nothing but cost.

Mitigations:
- **Scale the LR** — linear scaling for SGD, closer to square-root for Adam.
- **Warmup**, which becomes essential rather than optional at large batch.
- **LARS / LAMB** — layer-wise adaptive rate scaling. They normalize the update by the ratio of weight norm to gradient norm *per layer*, which lets you push to very large batches (LAMB trained BERT with batch 32k). Worth naming.
- Accept that there's a **critical batch size** beyond which you're buying nothing — it's data- and task-dependent, and measuring it saves real money.

---

### Q85. How do you make a long training run survivable?

This is the systems question hiding inside "tell me about training at scale," and it's where your background should shine.

- **Checkpointing** — periodic, atomic (write to temp then rename, so a crash mid-write doesn't corrupt), including model, optimizer state, scheduler state, RNG state, and **data loader position**. Missing the last two means your "resumed" run silently repeats or skips data. Async/sharded checkpoint writing so you're not stalling all ranks on IO.
- **Fault tolerance** — elastic training so a lost node doesn't kill the job; automatic restart from the last checkpoint; detection of hung ranks (NCCL timeouts) rather than silently hanging for hours.
- **Determinism when you need it** — fixed seeds, deterministic kernels, fixed data order. Note that full determinism costs throughput, so I'd make it a debug mode rather than the default, but I *would* want it available, because non-reproducible bugs at this scale are extremely expensive.
- **Monitoring** — loss, grad norm, LR, throughput, GPU utilization, per-rank step time, and the representation-health metrics from Q68. Alert on NaN, on loss spikes, on throughput drops, on rank divergence.
- **Experiment tracking** — every run tied to a code commit, a data version, and a config. This is the thing that makes ablations trustworthy; without it, six weeks in, nobody can reproduce the number in the deck.

↪ **Your hook:** MLflow + config-driven pipelines is on your CV. The framing that lands: **"the reason I could run credible ablations is that the infrastructure made runs comparable."** That's an L4-to-Lead argument.

---

## Part 12 — Tabular Deep Learning & Foundation Models

### Q86. Why does gradient boosting still beat deep learning on tabular data?

It genuinely often does, and pretending otherwise in an interview reads as naive. The reasons are structural:

1. **Tabular data has no spatial or sequential structure to exploit.** Deep learning's wins came from architectures with the right inductive bias — convolution for locality, attention for sequences. In a table, column order is arbitrary and there's no locality to bias toward. So the main advantage evaporates.
2. **Irregular target functions.** Real tabular targets are often piecewise-constant with sharp thresholds ("under 18 vs over 18"). Trees represent axis-aligned splits natively; MLPs are biased toward smooth functions and have to work hard to approximate a step.
3. **Uninformative features.** Tables are full of junk columns. Trees ignore them almost for free via split selection; MLPs have to learn to zero them out and often don't fully.
4. **Heavy-tailed and mixed-type features.** Trees are invariant to any monotonic transform of a feature, so skew and outliers just don't matter. Neural nets need careful normalization per column and are sensitive to getting it wrong.
5. **Small data.** Most tabular problems have thousands to millions of rows, not billions. That's GBDT's home turf.
6. **Practical**: GBDTs need almost no tuning to get 95% of their performance, train on a CPU in minutes, and handle missing values natively.

The Grinsztajn et al. paper ("Why do tree-based models still outperform deep learning on tabular data?") is the canonical citation and it's worth naming.

---

### Q87. So when *does* deep learning win on tabular?

This is where the answer gets interesting, and it's where your work lives:

- **When there's sequence or event structure.** A merchant's ordered event stream is not a table — it's a sequence with a temporal structure that a transformer can exploit and a GBDT cannot, short of hand-crafted aggregate features. This is the single strongest case, and it's the honest framing of "tabular" deep learning in industry: most wins come from data that was *flattened into* a table and shouldn't have been.
- **High-cardinality categoricals.** Millions of merchant IDs are painful for GBDT (target encoding leaks, one-hot explodes) and natural for embeddings.
- **Multi-modal features** — text descriptions, images, geo — where you want to jointly learn with tabular features in one model.
- **Multi-task and transfer.** You can train one representation and reuse it across many downstream models. GBDTs don't give you a reusable representation. This is often the *real* business case: not that the deep model beats GBDT on task A, but that it produces an embedding that improves tasks A through F and lets each team ship faster.
- **Very large datasets** where GBDT's scaling gets awkward.
- **When you need an embedding as the deliverable** — which is your exact situation.

The mature answer: **I'd always build the GBDT baseline first.** If the deep model can't beat a tuned LightGBM, that's important information, and shipping the simpler thing is the right call. Where I'd argue for deep is where the *representation* is the product, or where the structure genuinely isn't tabular.

↪ **Your hook:** this is the strongest possible framing for your merchant representation work — you're not claiming deep learning beats trees on tables; you're saying the data was sequential and shared across many downstream consumers, which is exactly when representation learning pays.

---

### Q88. What are tabular foundation models, and what do you think of TabPFN?

**TabPFN** is the interesting one. It's a transformer pretrained on **millions of synthetic tabular datasets** generated from structural causal models. At inference you feed the entire training set *as context* along with the test points, and it predicts in a single forward pass — **no gradient training at all**. It's in-context learning applied to tabular data.

Why it works: it's approximating Bayesian inference. It was trained to do posterior prediction over a prior of plausible datasets, so given a new small dataset it produces something close to the Bayesian-optimal prediction under that prior.

Strengths: on small datasets (originally under ~1,000 rows and ~100 features; v2 extended this considerably) it matches or beats tuned GBDT ensembles, in *seconds*, with zero hyperparameter tuning. Well-calibrated too.

Limits, which I'd state plainly: bounded by context length, so it doesn't yet cover the million-row regime most industrial problems live in; the synthetic prior may not match your domain; inference cost grows with the training set since it's all in context; and it doesn't handle high-cardinality IDs or sequences well.

Other things in this space worth naming: **TabTransformer** and **FT-Transformer** (attention over feature embeddings — FT-Transformer is the stronger, more reliable baseline), **SAINT** (attention over both features and rows), **NODE** (differentiable trees), and self-supervised approaches like **VIME** and **SCARF** that adapt masking/contrastive pretraining to tables.

My take, said honestly: tabular foundation models are the most intellectually exciting direction in the space, and simultaneously not yet the thing I'd bet a production system on for large-scale industrial data. That balance of enthusiasm and judgment is the right register for L4.

---

### Q89. How do you design masked modeling for tabular / event data? What's different from text?

This is your differentiated question, so it deserves a strong answer. The design choices that don't exist in NLP:

1. **What is a token?** In text a token is obvious. In an event stream, is a token one event (with all its fields), or one field? If one event, you need to fuse heterogeneous fields — categorical embeddings, continuous values, timestamps — into a single vector. Usually: embed each field, then sum or concatenate-and-project.
2. **What do you mask?** Options: mask a whole event, mask individual fields within an event, or mask a contiguous span of events. These teach very different things. Whole-event masking teaches sequence dynamics; field masking teaches within-event correlations, which risks a shortcut where the model infers a masked field from a correlated field in the same event and learns nothing about the sequence. I'd mask at the **event** level for representation quality, and possibly mix in field masking as an auxiliary.
3. **What's the reconstruction loss?** Text has one softmax. Here you have a categorical head per categorical field and a regression head per continuous field, and now you have a **loss-weighting problem** across heterogeneous heads with different scales. Cross-entropy on a 10-way category and MSE on a log-transformed amount are not comparable. This is genuinely one of the hardest design decisions, and uncertainty-based weighting (Q78) is a reasonable answer.
4. **Continuous values are hard to reconstruct.** Options: regress directly (loss scale problems, dominated by outliers), **bin/discretize into quantile buckets and classify** (usually more stable, and it makes the loss comparable to categorical fields — this is my default), or use a contrastive objective instead of reconstruction.
5. **Masking rate.** Text uses 15% by design. There's no reason that's right here — event sequences are far more redundant than text (a merchant's Tuesday looks like their Monday), so a low mask rate makes the task trivial and the model learns nothing. I'd expect to need a **much higher rate**, and I'd tune it. Higher masking rates being better in redundant modalities is exactly the MAE finding in vision (75%).
6. **Leakage through time.** Any feature computed with a lookahead window will leak. Event sequences make this easy to get wrong because feature pipelines aggregate over windows.

↪ **Your hook:** if you can speak to *two or three of these as decisions you made with an ablation behind each*, that is a genuinely senior answer, and it's specific enough that no one can suspect you're reciting a blog post.

---

## Part 13 — Failure Modes, Anti-Patterns & Debugging

### Q90. What is train/serve skew and how do you prevent it structurally?

The model sees different inputs in production than it did in training, so quality silently degrades. The word *silently* is the problem — nothing errors, the metrics just aren't what you expected.

Common causes: feature computed differently in the batch pipeline (Spark/SQL) vs the online path (Java/Go service); different default/missing-value handling; different preprocessing (a scaler fit on training data but recomputed online); time-zone or unit mismatches; a feature that's aggregated over 30 days offline and 7 days online; different tokenization or pooling.

Structural preventions, in order of strength:
1. **A feature store with a single definition** used by both paths, with point-in-time-correct offline retrieval. This is the real fix.
2. **Log the actual features at serving time** and train on those logged features rather than recomputing them. Now training data is by construction what serving produces.
3. **Shadow scoring / dual-run** — run the offline pipeline on production requests and diff the feature vectors and scores. Alert on divergence beyond a threshold. Cheap and catches most of it.
4. **Same code path.** Export the preprocessing with the model (as part of the graph or a shared library) rather than reimplementing it.
5. **Monitor feature distributions** in production against training, per feature — PSI or KL divergence with alerts.

---

### Q91. Talk about leakage. What kinds are there?

- **Target leakage** — a feature that encodes the label. Sometimes obvious (`is_refunded` when predicting refunds), often subtle (`num_support_tickets` where the ticket was created *because* of the outcome you're predicting).
- **Temporal leakage** — using information from after the prediction time. Aggregates computed over a window that includes the future, a "customer lifetime value" feature computed on the full history, or just a random train/test split on time-series data. **Any time-ordered problem needs a time-based split**, and honestly needs a walk-forward evaluation.
- **Preprocessing leakage** — fitting a scaler, imputer, PCA, or vocabulary on the full dataset before splitting. Now test statistics have influenced training. Fit on train only, transform everything.
- **Group leakage** — the same user, merchant, or session appears in both train and test. The model memorizes the entity and you measure memorization instead of generalization. Use grouped splits.
- **Duplicate leakage** — near-duplicate rows across the split.
- **Leakage through embeddings** — the sneaky one for your work. If entity embeddings were *pretrained on data that includes your evaluation period*, then a downstream model using those embeddings has indirect access to the future, even though its own features are clean. The embedding is a feature and it has a timestamp. I'd enforce that the representation model's training cutoff precedes the downstream training window.

The tell for leakage: **a metric that's too good.** If AUC jumps to 0.99, my first hypothesis is always leakage, not brilliance. Debugging move: rank feature importances and look at the top one skeptically.

---

### Q92. What are feedback loops and popularity bias in a deployed model?

The model shapes the data that trains the next model. Popular items get shown more, so they get more clicks, so they look more relevant, so they get shown more. Meanwhile good items that were never shown accumulate no evidence and stay invisible. Over generations the catalog effectively shrinks and the system converges on a narrow, self-confirming set.

Consequences: falling diversity, cold-start items never escaping, filter bubbles, an offline evaluation that increasingly measures agreement with the previous model, and metrics that look stable while the actual user experience narrows.

Mitigations:
- **Deliberate exploration** — an epsilon of randomized or bandit-driven traffic. This is a cost you pay in the short term to buy unbiased data, and it needs to be argued as an investment, which is a product conversation.
- **Propensity correction** on logged data (IPS), which requires having logged the propensities in the first place. **If you don't log propensities, you can't debias later** — so I'd push to log them from day one even before anyone needs them.
- **Diversity/novelty terms** in the ranking objective, or MMR-style re-ranking.
- **Popularity de-biasing** — sampling corrections, or explicitly modeling popularity as a separate term you can turn down.
- **Monitor catalog coverage and Gini/entropy of impressions over time.** If coverage is falling month over month, the loop is tightening.

The framing that lands: **this is a systems-and-incentives problem, not a loss-function problem.** A better model trained on biased data reproduces the bias more efficiently.

---

### Q93. How do you handle severe class imbalance? And what's focal loss?

Options, in the order I'd consider them:

1. **Do nothing, and use the right metric.** Imbalance itself isn't the problem; using accuracy is. With PR-AUC or a proper scoring rule, a well-trained model on imbalanced data is often fine.
2. **Class weighting** in the loss. Simple, keeps all data, but can hurt calibration.
3. **Downsampling the majority.** Very common at scale because it's also a compute win. Critical detail: **you must correct the intercept afterward** to restore calibration, using the known sampling rate.
4. **Oversampling / SMOTE.** I'm lukewarm — SMOTE interpolates between minority points, which is dubious in high dimensions and with categorical features, and it can create points inside the majority region. Rarely my choice for deep models.
5. **Focal loss** — down-weight the easy examples. It multiplies cross-entropy by `(1 - p_t)^γ`, where `p_t` is the predicted probability of the true class. If the model is already confident and correct, `(1-p_t)` is near zero, so that example contributes almost nothing. The gradient budget shifts to hard examples. `γ=2` is standard; γ=0 recovers plain cross-entropy. It was designed for dense object detection, where the imbalance is 1000:1 background to object.

Focal loss caveats worth stating: it **degrades calibration** (that's the trade — it deliberately stops pushing confident predictions), it can over-focus on **mislabeled** examples since noisy labels look exactly like hard examples, and it adds a hyperparameter that interacts with the class weighting α.

↪ **Your hook:** your NMT project used focal loss for token imbalance. Be ready for "did it actually help, and how did you know?" — and the strongest version of that answer includes the honest possibility that it helped less than expected, plus what you'd do differently (see Q51 on tokenization being the root cause).

---

### Q94. What is catastrophic forgetting and when does it bite?

A model fine-tuned on new data loses capability on the old distribution, because nothing in the objective preserves it.

Where it bites in production: continual retraining on recent data (the model becomes excellent at last month and poor at seasonal patterns), fine-tuning a general model on a narrow domain, or sequentially adding markets/languages.

Mitigations: **replay** (mix in a sample of old data — simplest and most effective), **regularization toward the old weights** (EWC weights the penalty by parameter importance via the Fisher information), **parameter-efficient tuning** (LoRA/adapters — freeze the base and add a small trainable delta, so the original capability is literally still there), **multi-task training** rather than sequential, and **model merging/averaging**.

The practical version for a retraining pipeline: always evaluate the retrained model on a **frozen historical benchmark**, not just recent data. Otherwise you have no way to see the forgetting.

---

### Q95. What is shortcut learning?

The model solves the task using a spurious correlation that happens to work on your data and won't hold in production. Classic examples: classifying pneumonia from a hospital-specific scanner artifact; detecting wolves via snow in the background; a resume screener keying on a school name.

In recsys/embedding contexts: a model that "predicts" merchant category from a feature that was derived from the category; a sequence model that predicts the masked event by reading a correlated field in the same event rather than learning temporal structure (see Q89); or a model exploiting an ID that encodes creation order.

How I'd catch it: feature importance / SHAP with a skeptical eye, ablating suspicious features and seeing whether performance survives, evaluating on a **deliberately shifted** slice (a new city, a new time period, a new device type), and adversarial/counterfactual probes. And group-wise evaluation — a shortcut usually shows up as excellent performance on one segment and poor on another.

---

### Q96. The model performs worse in production than in your offline eval. Walk me through it.

I'd want to be systematic rather than start guessing, and I'd say that out loud:

1. **Is it actually worse, or is it noise?** Check the confidence interval and the sample size first. Half of these turn out to be underpowered readings.
2. **Is the model the one you think it is?** Verify the deployed artifact hash matches the evaluated one. This is embarrassing and it happens.
3. **Score parity check.** Take 10k production requests, score them through the offline pipeline, and diff against the logged online scores. If they differ → skew (Q90), and now you have exact examples to bisect. If they match → the model is fine and the *evaluation* was wrong.
4. **If scores match**: the offline eval was biased. Look at candidate-set mismatch, position bias in the labels, a metric that doesn't match the business objective, or a distribution difference between your eval set and live traffic.
5. **Segment it.** Cut by new vs returning users, city, device, time of day, item popularity decile. Aggregate regressions are usually one segment falling over.
6. **Check the operational path** — latency, timeouts, fallbacks. If 5% of requests time out and fall back to a default ranking, that's a 5% quality hit that has nothing to do with the model. Look at feature *availability* online too: a feature that's null 30% of the time in production but never null in training is a very common cause.
7. **Check freshness** — stale embeddings, stale features, a feature pipeline that's lagging.

Then the part that makes it a senior answer: **write up the root cause and add the missing guardrail**, whether that's a parity test in CI, a feature-null-rate alert, or a change to how offline eval sets are constructed. The individual bug matters less than closing the class of bug.

---

## Part 14 — Bandits, Uncertainty & the Decision Intelligence Bridge

*This section is your strategic bet. You want to move toward decision-making under uncertainty, and you already have more relevant experience than you're giving yourself credit for — Bayesian optimization, adaptive experimentation, and a Thompson Sampling project. Use these to make the transition sound like a continuation, not a leap.*

### Q97. Explain the explore-exploit trade-off and the multi-armed bandit setup.

You have several options ("arms"), each with an unknown payoff. Every time you pick one you learn a little about it and earn its reward. **Exploit** = pick the one that currently looks best. **Explore** = pick something uncertain to learn more. Pure exploitation locks in on whatever got lucky early; pure exploration wastes traffic on known-bad options.

The metric is **regret** — how much worse you did than always picking the true best arm. Good algorithms achieve regret that grows logarithmically with time rather than linearly, which means the cost of learning becomes negligible in the long run.

The reason this matters for recsys specifically: it's the principled answer to the feedback-loop and cold-start problems in Q92 and Q55. A cold item has *high uncertainty*, and a bandit will deliberately show it — not out of fairness, but because the information is valuable. That reframing of exploration as an investment with a computable return is the useful business argument.

---

### Q98. Epsilon-greedy vs UCB vs Thompson Sampling.

- **Epsilon-greedy** — exploit with probability 1-ε, pick uniformly at random otherwise. Trivial to implement and reason about. The flaw: it explores *uniformly*, wasting the exploration budget equally on arms already known to be terrible. Decaying ε helps.
- **UCB (Upper Confidence Bound)** — pick the arm with the highest *optimistic* estimate: mean plus a confidence bonus that shrinks as you gather data. "Optimism in the face of uncertainty." Deterministic, with clean theoretical regret bounds. Downsides: the confidence term needs tuning to your reward scale, and being deterministic it's awkward for batched/delayed feedback — it'll pick the same arm for the whole batch.
- **Thompson Sampling** — maintain a posterior distribution over each arm's reward. Each round, **sample** a value from each posterior and play the argmax. So an arm is chosen in proportion to the probability it's actually the best.

Why I'd usually pick Thompson Sampling in practice: it's naturally randomized, which means it handles **batched and delayed feedback** gracefully (different requests in a batch sample differently, so you get natural diversity), it produces **propensities you can log for later off-policy evaluation** (huge — see Q92), it's often empirically stronger than UCB, and it extends cleanly to contextual and structured settings. The cost is needing a posterior, which is easy for Beta-Bernoulli and harder for a neural model — approximations include Bayesian linear heads on learned features, bootstrapped ensembles, or MC dropout.

↪ **Your hook:** your Tree-CNN + BAO query optimizer project used Thompson Sampling on a contextual bandit. **This is real RL-adjacent experience and you should stop describing it as a database project.** Frame it as: "I've built a system that made sequential decisions under uncertainty, with a learned value estimator and a posterior-sampling exploration policy." That's a decision-intelligence project.

---

### Q99. What's a contextual bandit, and when do you actually need full RL?

A **contextual bandit** adds features: the reward depends on both the arm and the context (user, time, query). So you're learning a policy `context → action` rather than a single best arm. LinUCB and Thompson Sampling with a linear or neural reward model are the standard approaches.

**The key distinction from full RL:** in a bandit, your action **does not change the state**. You show a recommendation, you get a reward, and the next user arrives independently. In RL, actions change the environment and rewards can be delayed across many steps, so you need to reason about long-horizon consequences — hence value functions, discounting, and credit assignment.

When you genuinely need RL: multi-step sessions where early actions shape later opportunities, sequential decisions with delayed payoff (driver dispatch, inventory, pricing over time), or anything where you're optimizing a trajectory rather than a single choice.

When a bandit is enough — and this is the mature answer: **most recommendation and ranking problems are bandits, and treating them as full RL adds enormous complexity for little gain.** The honest failure mode in industry is teams reaching for RL when a contextual bandit plus a good offline evaluation pipeline would have shipped in a fifth of the time. I'd want to be the person who knows the difference.

This is also a great answer to "you don't have RL experience" — because it lets you show you understand the *decision framework* and that you have judgment about when it's warranted, which is more valuable than having trained a PPO agent once.

---

### Q100. What is off-policy evaluation and why does it matter?

The problem: you logged data under policy A, and you want to estimate how policy B would perform, **without running it**. Because running every candidate policy as an experiment is slow and expensive.

Estimators:
- **IPS (Inverse Propensity Scoring)** — reweight logged rewards by `π_new(a|x) / π_logged(a|x)`. Unbiased if the logging policy had nonzero probability on every action the new policy takes (the **overlap/support** condition), but **high variance** when the two policies disagree a lot, since a few samples get enormous weights.
- **Self-normalized IPS / capped IPS** — divide by the sum of weights, or clip them. Slightly biased, much lower variance, and usually what's used in practice.
- **Direct Method** — just fit a reward model and evaluate the new policy against it. Low variance, but biased by whatever the reward model gets wrong, especially in regions with little data.
- **Doubly Robust** — combine both: use the reward model as a baseline and use IPS to correct its residuals. Consistent if *either* the propensity model or the reward model is right, hence "doubly robust." This is the standard recommendation.

The critical operational prerequisite: **you must log the propensities at decision time.** If your production policy is deterministic, propensities are all 1 and off-policy evaluation is essentially impossible — you have no support outside what you did. That's a strong argument for keeping some stochasticity in the serving policy, and it's the kind of forward-looking infrastructure argument an L4 should be making.

↪ **Your hook:** this is the direct bridge between your experimentation platform work and RL. Off-policy evaluation is *causal inference applied to sequential decisions*. You already do the causal inference; this is the same machinery pointed at policies instead of treatments.

---

### Q101. How does Bayesian optimization work, and how does it relate to bandits?

Bayesian optimization is for optimizing an expensive black-box function — like a hyperparameter search or an experiment configuration — where each evaluation costs real time or money.

Two pieces:
1. **A surrogate model** (usually a Gaussian Process, sometimes a tree-based model like in TPE/SMAC) that models the function *and its uncertainty* at every point you haven't tried.
2. **An acquisition function** that decides where to sample next by trading off exploiting where the surrogate predicts high value against exploring where it's uncertain. Expected Improvement, Upper Confidence Bound, and Probability of Improvement are the common ones — and note UCB is literally the same idea as in bandits.

So the relationship: **Bayesian optimization is a bandit problem with a continuous action space and a smoothness assumption.** GP-UCB is exactly the bandit algorithm applied to continuous domains, with regret bounds to match. The surrogate's job is to let you generalize across untried actions rather than treating each as independent.

↪ **Your hook:** your CV literally says you architected ETL feeding **Bayesian optimization models for continuous adaptive experimentation across 80+ pipelines.** That is decision-making under uncertainty at production scale. Say the words "explore-exploit" when you describe it, because that's what it was — and most candidates claiming RL interest have never run anything like it in production.

---

### Q102. "You don't have RL experience. Why should we consider you for decision-intelligence work?"

The version I'd actually say out loud, adapted:

> Fair — I haven't shipped a deep RL agent. But the core of decision intelligence is making choices under uncertainty and knowing whether the choice was right, and that's most of what I've done for the last four years. I've built adaptive experimentation infrastructure driving Bayesian optimization across 80+ concurrent pipelines, which is explore-exploit at production scale. I've built large-scale causal inference for AB and switchback experiments, which is the counterfactual reasoning that off-policy evaluation depends on. And on the modeling side I built a contextual bandit with Thompson Sampling and a learned Tree-CNN value estimator for query optimization.
>
> What I'd need to pick up is the sequential-credit-assignment half — value functions, policy gradients, the practical instability of on-policy methods. That's a real gap and I'm not going to pretend otherwise. But my read is that the hard part of applied RL in industry usually isn't the algorithm — it's the evaluation, the simulation environment, the logging, and knowing when a bandit is sufficient. That part I'd be strong on from day one.

That answer works because it's specific, it concedes the real gap without flinching, and it reframes your systems experience as the scarce skill rather than the consolation prize.

---

## Part 15 — System Design & Behavioral

### Q103. Design a merchant embedding system for a food delivery platform.

Structure the answer; don't free-associate. I'd spend the first two minutes on requirements.

**1. Clarify.** Who consumes this — search ranking, recommendations, fraud, ops? How many merchants and how heavy is the tail? Latency budget? How fresh must embeddings be? Is this replacing hand-crafted features or a previous model?

That question — "who consumes it" — is the important one, because a general-purpose embedding serving five teams is a different product from a task-specific one, and the trade-off (general is reusable but weaker per task) should be stated explicitly.

**2. Data.** Order events, search and browse behavior, merchant attributes (category, price, location, menu text), temporal patterns. Define the entity, the event schema, and — critically — the **time-based split** with a cutoff, so the representation model never sees data after the downstream training window.

**3. Modeling.** Sequence model over merchant events. Fuse per-field embeddings into an event vector, add time-aware positional encoding (Q47), transformer encoder, masked event modeling as the pretraining objective (Q89), with an optional contrastive auxiliary. Merchant vector from a trained entity token or attention pooling (Q57). Hybrid ID + content representation for cold start (Q56).

**4. Training infra.** Distributed with DDP or FSDP depending on size, bf16, gradient checkpointing if activation-bound. Config-driven so ablations are comparable. MLflow tracking tied to code and data versions.

**5. Evaluation.** Intrinsic health metrics (effective rank, random-pair cosine, alignment/uniformity) as *training-time* metrics; frozen probes on 3-4 real downstream tasks as the acceptance gate; and a clear statement of what a "good" embedding means before you start.

**6. Serving.** Batch precompute for all merchants, incremental refresh for changed entities, ANN index for retrieval, versioned namespace with a coordinated migration plan (Q61).

**7. Monitoring and rollout.** Shadow first, then a downstream A/B on one consumer, then expand. Monitor embedding drift, coverage, and neighbor stability.

**8. Trade-offs I'd volunteer** without being asked: general vs task-specific; freshness vs stability; embedding dimension vs storage and latency; and the organizational cost of every consumer being coupled to your version schedule. Naming that last one — that an embedding is a *dependency you're asking other teams to take* — is a strong L4 signal.

---

### Q104. How do you decide a model is ready to ship?

- **Offline gates**: beats the baseline on the primary metric by more than the noise; no regression on key segments; calibration checked; fairness/consistency checks if relevant.
- **Systems gates**: latency at p99 within budget, memory within budget, feature availability verified in production, score parity between offline and online paths, fallback behavior defined for missing features and timeouts.
- **Online**: shadow mode first (score real traffic, don't serve it) to catch skew; then a small-percentage experiment; then a properly powered A/B with a pre-registered primary metric and guardrail metrics.
- **Decision**: the primary metric moves beyond the MDE, guardrails don't regress, and the effect holds across segments. Then a staged rollout with a rollback plan and an owner.

The thing I'd emphasize: **the guardrails matter as much as the win.** A model that lifts clicks 2% and raises p99 latency 40ms or reduces catalog coverage 15% is often not a win. Deciding what the guardrails are *before* seeing results is what keeps that honest.

---

### Q105. Why switchback experiments, and what is an SRM check?

**Switchback** experiments randomize *time periods* across the whole market rather than randomizing users. You need them when there's **interference** — when treating one user affects another. In a marketplace that's the norm: if you give some riders better matching, they consume the supply that other riders would have used, so a user-randomized test measures a mix of the treatment effect and the cannibalization, and it's usually biased. Switching the entire region on and off in time blocks makes the treatment unit the region-time cell, which contains the spillover.

The costs: far fewer effective units (so much less power), temporal autocorrelation between adjacent blocks, carryover effects across the boundary (usually handled with burn-in periods), and the need for cluster-robust variance estimation.

**SRM (Sample Ratio Mismatch)** is the check that your actual traffic split matches the intended one. If you designed a 50/50 test and observe 50.4/49.6 with millions of users, that tiny imbalance is statistically impossible under correct randomization — a chi-square test flags it. It matters because SRM means the randomization or the logging is broken, and **any result from an SRM-failing experiment is untrustworthy regardless of how significant it looks.** Common causes: a bug in bucketing, differential crashes or timeouts in one arm, bot filtering applied unevenly, or a redirect that drops users.

The reason I'd bring SRM up unprompted: it's the highest-value cheap check in experimentation, and treating it as a hard gate rather than a warning is what separates a mature platform from one that ships false positives.

↪ **Your hook:** you did this at scale — 1,200+ daily metric pipelines, 10-15 TB/day. Very few ML candidates can speak to experimentation infrastructure with that depth. It's a genuine differentiator, so don't bury it.

---

### Q106. How do you frame the career transition question?

You'll get some version of "you've been a data engineer for 20 years — why ML now, and can you actually do the modeling?" Don't be defensive, and don't over-apologize for the background. The framing that works:

> I've spent four years now doing hands-on ML — representation learning, distributed training, ablations, evaluation systems — plus a Master's in AI. So it's less a transition than a shift in where I spend my time. What I'd say is that the systems background isn't a thing I'm leaving behind; it's why my ML work ships. Most of the hard problems I hit in the merchant embedding work weren't architecture problems, they were data leakage, train/serve skew, evaluation validity, and throughput — and those are exactly the things 20 years of data systems prepared me for. What I'm optimizing for now is depth on the modeling side, which is why I'm looking for a role where the model is the product rather than one where I'm the infrastructure person adjacent to a modeling team.

Adjust the specifics, but keep three properties: (1) claim the ML experience concretely, (2) reframe the systems background as an asset with a specific example, (3) be clear about what you *want*, since ambiguity there reads as uncertainty.

**The other thing to prepare:** they may probe whether you'd be bored or overqualified doing IC modeling work, or whether you'd resist being more junior in ML than your years suggest. Have an honest answer. Something like "I'd rather be the strongest engineer on a team where I'm learning modeling depth than the ML person on a team where I'm the ceiling."

---

### Q107. Questions worth asking them.

Asking good questions is graded, and these signal seniority:

- How do you decide what gets built? Where do modeling projects originate?
- What does the offline-to-online correlation look like for your team — do offline wins usually translate?
- Who consumes the models this team builds, and how tightly coupled are those consumers?
- What's the retraining and deployment cadence, and how much of it is automated?
- What's the split between modeling, infrastructure, and analysis for someone in this role?
- What's a recent project that didn't work, and what happened afterward?
- How does the team think about the bandit/RL space — is anyone working on sequential decision-making?

That last one plants your interest in decision intelligence without making it sound like you're using the role as a stepping stone.


---

## Part 16 — Loss Functions

### Q108. Why cross-entropy for classification instead of MSE?

Three reasons, and the gradient one is the important one.

**1. Gradients.** If you put MSE on top of a sigmoid or softmax, the gradient contains the derivative of the activation as a factor. When the model is confidently *wrong* — sigmoid output near 0 when the label is 1 — the sigmoid derivative is near zero, so the gradient vanishes exactly when you most need a big correction. Learning stalls. With cross-entropy, the sigmoid/softmax derivative cancels algebraically against the loss derivative, and the gradient of the logit becomes simply `(prediction - target)`. Clean, proportional to the error, and it never saturates. That cancellation is the single best reason.

**2. It's the right objective.** Cross-entropy is the negative log-likelihood of the data under the model's predicted distribution. Minimizing it *is* maximum likelihood estimation for a categorical model. MSE is the MLE for Gaussian noise, which isn't what a class label is.

**3. It's a proper scoring rule** — *meaning it's minimized exactly when your predicted probabilities equal the true probabilities*, so it encourages honest calibration. MSE (Brier score) is also proper, so this one is weaker, but cross-entropy penalizes confident wrongness much more aggressively, which is usually what you want.

**The exception worth knowing:** MSE isn't insane for classification, and there's work showing it's competitive with careful tuning. And cross-entropy's aggressive penalty on confident errors makes it **sensitive to label noise** — one mislabeled example the model is confident about generates a huge gradient. If your labels are noisy, that's an argument for label smoothing, a bounded loss like symmetric CE or generalized CE, or MAE-flavored variants.

---

### Q109. Explain the relationship between entropy, cross-entropy, and KL divergence.

They're one family and it's worth being able to draw the line between them.

- **Entropy** `H(p)` — the average surprise of a distribution. How many bits you need on average to encode samples from `p` using the optimal code for `p`. It's a property of one distribution.
- **Cross-entropy** `H(p,q)` — the average bits needed if you encode samples from `p` using a code optimized for `q`. Always ≥ H(p), with equality only when `q = p`.
- **KL divergence** `D(p‖q) = H(p,q) - H(p)` — the *extra* bits you pay for using the wrong code. So it's the penalty for your model `q` being different from reality `p`.

Why this matters practically: when you minimize cross-entropy with respect to model parameters, `H(p)` is a constant (it's a property of the data, not the model), so **minimizing cross-entropy and minimizing KL divergence are the same optimization problem.** That's the connective tissue between "train with cross-entropy" and "fit the data distribution."

Properties to remember: KL is **not symmetric** and not a distance. `D(p‖q) ≠ D(q‖p)`, and the difference is meaningful:
- **Forward KL** `D(data‖model)` is **mode-covering** — it's infinite wherever the data has mass and the model doesn't, so the model spreads out to cover everything, including averaging over modes it can't represent. This is what MLE does, and it's why a poorly-specified generative model produces blurry averages.
- **Reverse KL** `D(model‖data)` is **mode-seeking** — it's penalized for putting mass where the data has none, so it collapses onto one mode and ignores the rest. This is what variational inference and some RL objectives use, and it's why they produce sharp but less diverse outputs.

That distinction shows up constantly in generative modeling and in RLHF (where the KL penalty to the reference model is a specific direction chosen for specific reasons). **JS divergence** is the symmetrized, bounded version, and it's what the original GAN objective implicitly minimizes.

---

### Q110. Binary vs categorical cross-entropy vs sampled softmax — when does each apply?

- **Binary CE** — independent yes/no per output. Use for binary classification and for **multi-label** problems, where an item can have several labels at once (a merchant is both "halal" and "late-night"). Each output gets its own sigmoid; they don't compete.
- **Categorical CE with softmax** — mutually exclusive classes. Softmax makes outputs compete for a fixed probability budget. Use when exactly one label is correct.

The mistake people make is using softmax for multi-label, which forces the classes to compete when they shouldn't, or BCE for mutually exclusive classes, which throws away the constraint that probabilities sum to one.

- **Sampled softmax / negative sampling** — when the number of classes is enormous (a 20M-item vocabulary or catalog), computing the full softmax denominator every step is prohibitive. Instead, compute it over the true class plus a sample of negatives. This is the standard for large-scale retrieval and word embeddings.

The catch, and it's the part that separates people who've read about it from people who've shipped it: **sampled softmax is a biased estimator unless you correct for the sampling distribution.** If you sample negatives proportional to popularity (which happens naturally if you use in-batch negatives from real traffic), popular items are over-represented as negatives and get systematically pushed down. The **logQ correction** fixes it: subtract `log Q(item)` from each logit, where `Q` is that item's sampling probability. In practice you estimate `Q` from a streaming frequency counter. Without it, your retrieval model quietly learns to under-rank the head of the catalog. Related: **NCE** and **negative sampling** (word2vec-style) are different approximations with different bias properties — negative sampling doesn't actually approximate softmax at all, it solves a different binary classification problem that happens to produce good embeddings.

---

### Q111. Explain hinge loss and where you'd still use it.

Hinge loss is `max(0, 1 - y·f(x))` for labels in {-1, +1}. It's zero once an example is correctly classified **with a margin of at least 1**, and linear in the violation otherwise.

The key property: **once an example is comfortably right, it contributes exactly nothing.** Only the examples near or across the boundary matter — those are the support vectors. Compare with cross-entropy, which keeps pushing on every example forever, always wanting more confidence.

Consequences: hinge produces a max-margin decision boundary and is more robust to a few outliers far on the correct side. But it's **not a proper scoring rule**, so it doesn't give you calibrated probabilities — an SVM's decision value isn't a probability without Platt scaling on top.

Where it survives: classical SVMs, some ranking settings, and — importantly for you — the **margin idea** propagates everywhere. Triplet loss, contrastive loss with a margin, and margin-based softmax variants (ArcFace, CosFace, which add an angular margin to the softmax and are the standard for face recognition and strong for any metric-learning problem) are all hinge thinking applied to embeddings.

---

### Q112. Explain triplet loss and its practical difficulties.

Triplet loss takes an anchor, a positive (should be similar), and a negative (should be dissimilar), and asks: `d(a,p) + margin < d(a,n)`. The loss is the violation of that inequality, floored at zero.

So it's explicitly **relative** — it doesn't care about absolute distances, only that positives are closer than negatives by a margin. That's often exactly right for retrieval.

The practical difficulty is entirely about **triplet mining**, and this is what interviewers probe:

- Randomly sampled triplets are almost all **easy** — already satisfying the constraint, contributing zero gradient. After the first epoch you're computing forward passes on triplets that teach nothing. Training stalls not because it converged but because it ran out of signal.
- **Hardest-negative mining** (pick the closest negative) gives strong gradients but is unstable and prone to collapse, and it's very sensitive to label noise — the "hardest negative" is often a mislabeled positive.
- **Semi-hard mining** (FaceNet's answer) picks negatives that are farther than the positive but still inside the margin. Enough signal, less instability. Usually done **within the batch**, so batch size and batch composition matter enormously — you need multiple examples per class in each batch, which means a custom sampler, not random shuffling.

Because of all this, most people now prefer **InfoNCE / batch-softmax** over triplet loss: it uses *all* negatives in the batch at once instead of picking one, so it's less sensitive to mining strategy and generally better behaved. I'd default to InfoNCE and reach for triplet only when the relative-comparison framing is genuinely the right structure. **Multi-similarity loss** and **circle loss** are the refined descendants worth knowing about.

---

### Q113. Pointwise, pairwise, listwise ranking losses — compare them.

- **Pointwise** — predict a score or relevance for each item independently, with MSE or cross-entropy. Simple, scales trivially, reuses any classifier. But it optimizes absolute prediction, not order, and it's blind to the fact that ranking is a *comparative* task. Still, a well-calibrated pointwise CTR model is the backbone of most production systems, because you need the calibrated probability for other purposes anyway (Q79).
- **Pairwise** — for each pair of items with different relevance, penalize getting their order wrong. **RankNet** uses a logistic loss on the score difference. **LambdaRank** adds the crucial refinement: weight each pair's gradient by how much swapping them would change NDCG. That means pairs at the top of the list matter much more than pairs at the bottom, which aligns the gradient with the actual metric without needing the metric to be differentiable. **LambdaMART** is LambdaRank gradients inside GBDT, and it remains a very strong baseline — often still the thing to beat.
- **Listwise** — operate on the whole list at once. **ListNet/ListMLE** model a probability distribution over permutations; **Softmax cross-entropy over the list** (treating the relevant item as the correct class) is the simple practical version and it's what most neural rankers actually use. **ApproxNDCG** and **NeuralNDCG** use smooth relaxations of the sort operation to differentiate the metric directly.

**How I'd choose:** start pointwise if you need calibrated scores and the list structure is weak. Move to listwise softmax if the objective is purely ranking quality — it's usually the best quality-per-complexity in a neural setting. Use LambdaMART if you're in a GBDT world with rich features and modest data. And note you can combine: a multi-task model with a pointwise calibration head and a listwise ranking head gets you both.

---

### Q114. Regression losses: MSE, MAE, Huber, quantile.

- **MSE (L2)** — penalizes squared error, so large errors dominate. Its minimizer is the **conditional mean**. Differentiable everywhere, and it's the MLE under Gaussian noise. Very sensitive to outliers — one bad label can dominate the loss for a whole batch.
- **MAE (L1)** — penalizes absolute error. Its minimizer is the **conditional median**, so it's robust to outliers. Downside: the gradient has constant magnitude regardless of error size, so it converges poorly near the optimum and the gradient is undefined at zero.
- **Huber** — quadratic near zero, linear beyond a threshold δ. Gets MSE's nice gradients near the optimum and MAE's robustness on outliers. δ is a hyperparameter you should actually set from the data (e.g. a percentile of residuals), not leave at the default.
- **Quantile / pinch loss** — asymmetric absolute loss, weighting over- and under-prediction differently by a factor τ. Its minimizer is the **τ-th quantile**. This is how you get prediction *intervals* from a point model: train three heads for the 10th, 50th, and 90th percentiles. Enormously useful for ETA prediction, demand forecasting, and anywhere the cost of over- and under-prediction is asymmetric — which is most business problems.
- **Log-transform first** — if your target is heavy-tailed and multiplicative (revenue, order counts, durations), predicting `log(y)` with MSE is often far better than predicting `y`. But remember the retransformation bias: `exp(E[log y]) ≠ E[y]`, so you need a correction if you care about the mean rather than the median.

**Framing that matters:** the loss encodes what kind of error you're willing to accept. Asking "what does a 10% over-prediction cost versus a 10% under-prediction?" is a product question, and the answer picks the loss. Defaulting to MSE is defaulting to "I want the conditional mean and I don't mind outliers dominating" — often not what anyone wanted.

---

### Q115. How do you approach designing a loss for a new problem?

The framework I'd articulate:

1. **What decision does the output feed?** If a human reads the number, you need calibration. If it's sorted, you need ranking quality. If it's thresholded, you need accuracy near the threshold specifically. If it's multiplied by a value, you need calibration *and* the right scale.
2. **What's the noise model?** Gaussian → MSE. Heavy-tailed → Huber or log-transform. Categorical → cross-entropy. Count data → Poisson loss. This gets you to the MLE-appropriate choice.
3. **What's the asymmetry?** Are false positives and false negatives equally costly? Almost never. Encode the ratio in class weights or a quantile loss rather than post-hoc threshold tuning — though post-hoc threshold tuning is a perfectly good, simpler answer that I'd try first.
4. **What structure exists that a naive loss ignores?** Ordinal labels (1-5 stars) where predicting 4 for a 5 is better than predicting 1 — plain cross-entropy treats those as equally wrong. Hierarchical labels. List structure. Time structure.
5. **What shortcuts does the loss permit?** Every loss has a degenerate minimizer. Ask what the laziest possible model that minimizes this loss looks like — predicting the base rate, predicting a constant embedding, memorizing an ID. If that's cheap, add a term that forbids it (Q67 is the canonical version of this).
6. **Then check the simplest thing first.** Most "we need a custom loss" situations are actually solved by fixing the labels, re-weighting, or changing the threshold. Custom losses add tuning surface and are hard to debug. I'd want evidence that the standard loss is the binding constraint before writing a new one.

---

## Part 17 — Retrieval & Two-Tower Systems

### Q116. Explain the two-tower architecture and why it exists.

Two separate encoders: a **query tower** (user, context, search string) and an **item tower** (merchant, product, document). Each produces a vector in a shared space, and the score is a dot product or cosine similarity.

The whole point is **precomputation**. Because the item tower doesn't depend on the query, you can embed all 20M items offline, load them into an ANN index, and at request time only run the query tower once and do an approximate nearest-neighbor lookup. That turns "score 20M items" into "one small forward pass plus a sublinear index lookup" — which is the difference between possible and impossible at production latency.

The cost is that **the towers cannot interact.** The only place query and item information meet is the final dot product. So the model can't learn "this user likes spicy food AND this restaurant's *third* menu item is spicy" — any interaction has to be compressed into the two independent vectors before they meet. That's a genuine and severe expressiveness limit, and it's why two-tower is used for *retrieval* (get 500 plausible candidates from 20M) and a heavier model does *ranking* (order those 500 precisely).

Design details that matter:
- **Shared vs separate parameters.** If both towers encode the same kind of object (sentence-sentence similarity), share weights. If they encode different things (user vs item), separate towers, though shared feature embeddings for common features are fine.
- **Symmetric vs asymmetric training.** Some setups use a shared "in-batch softmax" in both directions (query→item and item→query), which is usually a small free win.
- **Normalization and temperature** — almost always L2-normalize and use a learned or tuned temperature (Q65).

---

### Q117. Bi-encoder vs cross-encoder vs late interaction — what's the spectrum?

This is the key framing for modern retrieval and worth having crisp.

- **Bi-encoder (two-tower)** — encode separately, compare with a dot product. Fast, precomputable, less accurate. Retrieval stage.
- **Cross-encoder** — concatenate query and item into one sequence and run a full transformer over both together, so every query token can attend to every item token. Dramatically more accurate because interactions are modeled directly. But it requires a full forward pass **per query-item pair**, so it cannot be precomputed and cannot scale past a few hundred candidates. Reranking stage.
- **Late interaction (ColBERT)** — the interesting middle. Encode query and document into *per-token* vectors independently (so documents are still precomputable), then compute a cheap interaction at query time: for each query token, take the max similarity over all document tokens, and sum. So you get token-level interaction with precomputed document representations. Much better than a bi-encoder, much cheaper than a cross-encoder. The cost is storage — you're storing a vector per token instead of per document, which is 50-100× more index. ColBERTv2 and PLAID address that with compression.

The standard production pattern is a **cascade**: cheap bi-encoder retrieves thousands → optional late-interaction or lightweight model narrows to hundreds → expensive cross-encoder or full ranking model orders the final list. Each stage buys precision at increasing cost per candidate, and you tune the funnel widths against your latency budget.

**Distillation ties it together:** train a cross-encoder as the teacher, distill into the bi-encoder. The bi-encoder learns from a model that *could* see the interactions, which meaningfully closes the gap. This is one of the highest-leverage tricks in retrieval and I'd bring it up unprompted.

---

### Q118. How do you train a retrieval model? Walk through the practical decisions.

1. **Positives.** What counts as a match? Clicks, conversions, dwell-time-thresholded clicks, explicit relevance labels. Every choice bakes in a bias — clicks bake in position bias (Q77), conversions are sparse and delayed. I'd usually use a weighted combination and be explicit about it.
2. **Negatives.** The single biggest lever (Q66). In-batch negatives for free scale, cross-GPU gathering to multiply them, plus **mined hard negatives** from the current model's top-k excluding known positives. Watch for false negatives.
3. **The logQ correction** for popularity-biased sampling (Q110). Not optional at scale.
4. **Loss** — in-batch softmax cross-entropy (InfoNCE) with a tuned temperature.
5. **The train/serve mismatch that's specific to retrieval:** at training time your negatives come from the batch — maybe a few thousand items. At serving time the model competes against the **entire catalog** of 20 million. The model has never been asked to discriminate against most of them. That gap is why offline in-batch metrics look great and full-catalog recall is worse, and it's why hard negative mining from the full corpus matters so much. Always evaluate against the **full catalog**, not the batch.
6. **Index-aware evaluation** — your true recall is `model recall × ANN recall`. Measure them separately (Q61) so you know which one to fix.
7. **Refresh cadence** — the item index has to be rebuilt when the model changes, and both towers must be the same version. Serving a v2 query tower against a v1 item index produces garbage silently, because the dot products are still well-formed numbers. That deserves an explicit version-compatibility assertion at load time.

---

### Q119. Compare ANN algorithms: HNSW, IVF, PQ, ScaNN.

- **HNSW (Hierarchical Navigable Small World)** — a multi-layer proximity graph. Search starts at a sparse top layer and descends, greedily walking toward the query. Excellent recall-latency trade-off, and it's the default in most vector databases. Costs: high memory (the graph itself is large, often more than the vectors), slow index build, and **awkward deletes** — you can't cleanly remove a node from a graph, so you tombstone and periodically rebuild.
- **IVF (Inverted File)** — cluster the vectors with k-means, and at query time search only the `nprobe` nearest clusters. Simple, fast to build, tunable via `nprobe` (recall vs latency). Weakness: vectors near cluster boundaries get missed, so recall plateaus below HNSW at equal latency.
- **PQ (Product Quantization)** — a *compression* method, not a search structure, and people conflate these. Split each vector into m subvectors, run k-means on each subspace, store only the centroid IDs. A 768-dim float32 vector (3KB) becomes 96 bytes. Distances are computed approximately from lookup tables. Almost always combined: **IVF-PQ** is the standard for billion-scale indexes where the vectors don't fit in RAM uncompressed.
- **ScaNN** — Google's, with **anisotropic vector quantization**: the insight that when you care about maximum inner product, quantization error *parallel* to the vector matters more than error perpendicular to it, so weight the quantization loss accordingly. Consistently strong benchmarks.
- **Flat / brute force** — exact. Genuinely the right answer under ~100k vectors, especially on a GPU. Don't add an approximate index to a problem that doesn't need one; you've added a recall ceiling and an infrastructure dependency for nothing.

**How I'd choose:** under 100k → flat. Up to a few million and RAM is fine → HNSW. Tens of millions or more, or memory-constrained → IVF-PQ or ScaNN. High write/delete rate → think hard, because HNSW handles churn badly and you may need a segmented index with periodic merges (the Lucene model).

---

### Q120. What's hybrid retrieval and why does it beat pure dense?

**Sparse retrieval (BM25)** matches on exact terms with frequency weighting. **Dense retrieval** matches on learned semantics. They fail differently, and that's the point.

Dense fails on: rare proper nouns, product codes, exact identifiers, out-of-domain terms it never trained on, and anything requiring precise lexical match. It'll happily return something semantically adjacent when you needed the literal string.

Sparse fails on: paraphrase, synonyms, cross-lingual matching, and anything where the useful signal isn't a shared token.

Combining them beats either. Methods:
- **Reciprocal Rank Fusion (RRF)** — score each document as the sum of `1/(k + rank)` across the two lists. It ignores the raw scores entirely, which is exactly why it works: dense and sparse scores are on incomparable scales and normalizing them is fiddly. RRF sidesteps the whole problem. It's embarrassingly simple and it's a very strong baseline.
- **Learned linear combination** with a tuned weight, requiring score normalization.
- **SPLADE** — learned *sparse* representations: use a transformer to produce a sparse vector over the vocabulary, so you get semantic expansion with the efficiency of an inverted index. A genuinely nice idea that gets both properties in one model.

The takeaway I'd offer: **most "our vector search is bad" problems in production are actually cases where a keyword match would have worked.** Hybrid is close to free and should usually be the default rather than an optimization.

---

### Q121. What are the common failure modes of a deployed retrieval system?

Worth having a checklist, because these recur:

1. **Version skew between towers or between model and index** (Q118). Silent, produces plausible-looking garbage.
2. **ANN recall ceiling** mistaken for model quality (Q61).
3. **Popularity collapse** — the index returns the same head items for everything, because of unnormalized magnitudes (Q59) or uncorrected sampled softmax (Q110).
4. **Stale index** — new items aren't retrievable until the next rebuild, so a fresh catalog is invisible. Usually needs a real-time path for new items alongside the batch index.
5. **Query-side cold start** — a new user has no history, so the query tower gets mostly-default features and returns generic results.
6. **Embedding drift** — the item distribution shifts (new categories, new markets) while the model stays fixed, so a growing fraction of the catalog sits in a region the model never trained on.
7. **False-negative contamination** in training that suppresses whole categories.
8. **Missing normalization at query time** while the index was built normalized, or vice versa.
9. **Filtering interacting badly with ANN** — if you retrieve top-100 and then filter by availability/geography, you may be left with 3 results. Pre-filtering inside the index (which most vector DBs now support) or over-retrieving are the fixes, and this is a very common real-world bug.

That last one is worth spelling out in an interview because it's the kind of thing you only hit once you've actually run one of these systems.

---

## Part 18 — Model Compression

### Q122. Explain knowledge distillation. Why can a student trained on a teacher beat the same student trained from scratch?

Distillation trains a small **student** to match the outputs of a large **teacher**, instead of (or in addition to) matching the hard labels.

The mechanism is **dark knowledge** — the information in the teacher's *wrong* answers. A hard label says "this is a 7." The teacher's soft distribution says "92% a 7, 6% a 1, 1.5% a 9, and essentially zero for everything else." That relative structure encodes which classes are confusable, which is real information about the input that the one-hot label throws away. The student learns the teacher's similarity structure over classes, not just the answer.

Mechanics: soften both distributions with a **temperature** T (divide logits by T before softmax), take KL divergence between them, and scale the gradient by T² because softening shrinks gradients by roughly 1/T². Usually combine with the normal hard-label loss at some mixing weight.

Why it beats from-scratch training:
- **Denser supervision per example.** One label gives log(C) bits; a full distribution gives much more. So the student is more sample-efficient.
- **Label noise smoothing.** The teacher's output on a mislabeled example is often closer to the truth than the label.
- **A better-conditioned optimization target.** Soft targets create a smoother loss surface than one-hot targets.
- **Regularization** — soft targets act like a learned, input-dependent label smoothing.

Variants worth knowing:
- **Self-distillation** — student and teacher are the same architecture, and it *still* helps. That rules out the "capacity transfer" explanation and supports the regularization one.
- **Feature/hint distillation (FitNets)** — match intermediate representations, not just outputs. Needs a projection if dimensions differ.
- **Attention transfer** — match attention maps.
- **Relational KD** — match the *pairwise relationships* between examples rather than individual outputs. Especially relevant for embeddings: what you want preserved is the geometry, not the absolute vectors.
- **Online/mutual distillation** — train several models simultaneously that teach each other, no pretrained teacher required.

↪ **Your hook:** for embedding models, relational distillation is the right framing — you want the *neighbor structure* preserved when you shrink the model, and that's a different objective than matching vectors.

---

### Q123. Explain pruning. What's structured vs unstructured, and what's the lottery ticket hypothesis?

Pruning removes parameters that contribute little.

- **Unstructured pruning** — zero out individual weights, usually by magnitude. Can remove 80-95% of weights with minimal quality loss. The problem: you get a **sparse matrix with an irregular pattern**, and standard GPU kernels get no speedup from that. You've saved memory in theory and nothing in wall-clock. Making it real requires sparse kernels or hardware support — NVIDIA's **2:4 semi-structured sparsity** (exactly 2 of every 4 weights zero) is the practical middle ground, with actual tensor-core support for roughly 2× speedup.
- **Structured pruning** — remove whole units: attention heads, FFN neurons, channels, entire layers. Less compression at the same quality, but the result is a *smaller dense model* that runs faster on any hardware with no special kernels. For production, this is usually what you want.
- **Layer dropping** for transformers is surprisingly effective — many models tolerate dropping a meaningful fraction of middle layers with a short recovery fine-tune, which tells you something about how much redundancy is in there.

Process: train → prune → fine-tune to recover, often iteratively. **Gradual magnitude pruning** during training (ramping sparsity up on a schedule) generally beats one-shot pruning at the end.

**The Lottery Ticket Hypothesis** (Frankle & Carbin): inside a randomly initialized dense network there exists a sparse subnetwork ("the winning ticket") that, *when trained from the original initialization*, matches the full network's accuracy in comparable time. The startling part is the dependence on the original init — reinitialize the same sparse structure randomly and it trains much worse. So the claim is that the value is in the structure *plus* its specific starting point.

Why it matters, and the honest caveat: it suggests overparameterization's role is providing many candidate subnetworks to select from, which reframes why big models train easily. But finding the ticket requires training the dense model first, so it doesn't yet give you a cheaper training recipe. And the original result needed **iterative magnitude pruning** and doesn't hold cleanly at large scale without modifications ("late resetting" — rewinding to an early training step rather than to init). Knowing that nuance is worth more than knowing the headline.

---

### Q124. Explain quantization: PTQ vs QAT, and what actually breaks.

Quantization stores and computes with lower-precision numbers — fp16, int8, int4 — instead of fp32. You get memory savings proportional to the bit reduction and, if the hardware has the kernels, real throughput gains.

**Post-Training Quantization (PTQ)** — quantize an already-trained model. Needs a small **calibration set** to determine the scale factors (the mapping from float range to integer range). Fast, no retraining. Usually fine for int8 on most architectures.

**Quantization-Aware Training (QAT)** — simulate quantization during training with "fake quant" nodes, so the model learns weights that are robust to the rounding. Uses the **straight-through estimator** to get gradients through the non-differentiable rounding operation (*just pass the gradient through as if rounding were the identity*). More accurate, especially below int8, but requires a training run.

**What actually breaks**, which is the interesting part:

- **Outlier activation channels.** In large transformers, a small number of feature dimensions have activation magnitudes 10-100× everything else. Since quantization scale is set by the max, those outliers force a coarse scale and everything else gets crushed into a couple of quantization levels. This is *the* reason naive int8 fails on LLMs above a certain size. **LLM.int8()** solves it by keeping outlier dimensions in fp16 and quantizing the rest (mixed-precision decomposition). **SmoothQuant** migrates the difficulty from activations to weights by rescaling — since weights are easier to quantize than activations, that's a good trade.
- **Weights vs activations are different problems.** Weights are static, known, and well-behaved — easy. Activations are dynamic and input-dependent — hard. Weight-only quantization (very common for LLMs, since decoding is memory-bandwidth-bound on weights) gets most of the benefit with much less risk.
- **Per-tensor vs per-channel vs per-group scaling.** Finer granularity handles outliers better at a small metadata cost. Per-channel for weights is near-free and should basically always be on.
- **Sensitive layers.** The first and last layers, the embedding table, and LayerNorm are typically kept in higher precision.

**Modern LLM quantization methods:** **GPTQ** (layer-by-layer, uses second-order/Hessian information to choose rounding that minimizes output error), **AWQ** (activation-aware — protects the small fraction of weight channels that matter most, identified by activation magnitude), and **bitsandbytes NF4** (the 4-bit normal-float format used in QLoRA, designed to be information-theoretically optimal for normally-distributed weights).

**The evaluation trap worth calling out:** perplexity barely moves under quantization while task performance on reasoning or long-context tasks can degrade noticeably. Never validate a quantized model on perplexity alone.

---

### Q125. How do you decide which compression technique to use?

Start from the actual constraint, because these solve different problems:

- **Memory / model doesn't fit** → quantization first. Best ratio of benefit to risk, and weight-only int8 or int4 is often nearly free.
- **Latency, memory-bandwidth-bound** (which LLM decoding is) → quantization again, because you're moving fewer bytes per token.
- **Latency, compute-bound** (which batch inference and most encoders are) → structured pruning or distillation to a genuinely smaller architecture. Quantization won't help much if you're not bandwidth-bound.
- **Cost per request at scale** → distillation into a purpose-built small model. The biggest wins come from the student being *designed* small rather than the teacher being shrunk.
- **You need one model, many deployment targets** → Matryoshka-style nested training (Q73) or a single model with multiple exit points.

**The order I'd apply them:** distill → prune → quantize, with recovery fine-tuning between stages. They compose, but the interactions are real — pruning after quantization is worse than before it, and quantizing a heavily-pruned model is more fragile.

**And the meta-point:** before compressing, check whether you need the big model at all. A well-tuned smaller model trained on the right data frequently beats a compressed large one, and it's simpler to maintain. Compression is what you do when you've verified the big model's quality is genuinely necessary.

---

## Part 19 — Fine-Tuning & Adaptation

### Q126. Fine-tune, prompt, or RAG — how do you choose?

The clean framing: **RAG adds knowledge, fine-tuning adds behavior, prompting adds instructions.** Most people reach for fine-tuning when they wanted RAG.

- **Prompting / in-context learning** — zero cost, instant iteration, no infrastructure. Try this first, always. Limits: consumes context (so it costs tokens on every request), inconsistent for complex formats, and can't teach genuinely new skills.
- **RAG** — retrieve relevant documents and put them in the context. Right when the need is **factual, changing, or proprietary knowledge**. Advantages: update the knowledge base without retraining, cite sources, control access per user, and no risk of the model confidently making things up about your internal data (well — less risk). Fine-tuning is a *bad* way to inject facts: it's expensive, the facts get entangled with everything else, you can't update one fact, and the model still hallucinates around the edges.
- **Fine-tuning** — right when you need **consistent behavior, format, tone, or a domain-specific skill** that prompting can't reliably produce; when you want to shrink the model (a fine-tuned small model can beat a prompted large one at a specific task, much cheaper); or when the task is far enough out of distribution that in-context examples don't suffice.

They compose. The strong production pattern is often a fine-tuned model that's been taught *how to use retrieved context well*, plus RAG supplying the facts.

**Decision heuristics:** if the answer changes when the underlying data changes → RAG. If you can write down the rule → prompt. If you have 1,000+ examples of the desired input/output behavior and prompting is inconsistent → fine-tune. If latency and unit cost are the binding constraint → fine-tune a small model.

---

### Q127. Explain LoRA. Why does it work, and what are the knobs?

**LoRA (Low-Rank Adaptation)** freezes the pretrained weights and learns a low-rank update alongside them. Instead of updating a `d×k` matrix W, you learn two small matrices `A` (r×k) and `B` (d×r), and the effective weight becomes `W + BA·(α/r)`. With r=8 and d=4096, you're training ~0.1% of the parameters.

**Why it works:** the hypothesis is that the *update* needed to adapt a pretrained model to a task has low **intrinsic rank** — you're not learning a new function, you're nudging an existing one, and that nudge lives in a small subspace. Empirically this holds remarkably well.

**Practical advantages beyond memory:**
- **No inference latency**, because you can merge `BA` back into `W` after training. Adapters (Q128) can't do this — they add layers.
- **Swappable.** One base model in memory, many LoRA adapters loaded per request. That's a completely different serving economics than one fine-tuned model per task.
- **Less catastrophic forgetting**, since the base weights are untouched.
- **Tiny checkpoints** — megabytes instead of gigabytes.

**The knobs:**
- **Rank `r`** — capacity. 8-16 for most tasks; 64+ if you're teaching a substantially new skill or doing continued pretraining. Higher rank helps less than people expect; going from 8 to 64 often changes little, which is itself evidence for the low-intrinsic-rank claim.
- **Alpha `α`** — a scaling factor; the effective update is scaled by `α/r`. The common convention is α = 2r. What matters is the ratio, so if you change r, change α or you've silently changed your learning rate.
- **Which modules.** The original paper only adapted attention Q and V. Later work found **adapting all linear layers, including the FFN, is meaningfully better** — and the FFN is where most parameters and most stored knowledge live. Default to all linear layers unless memory forbids.
- **Learning rate** — LoRA wants a much *higher* LR than full fine-tuning, typically 1e-4 to 1e-3 rather than 1e-5. This trips people up constantly; a LoRA run at full-fine-tune LR just doesn't learn.
- **Initialization** — A random, B zero, so the update starts at exactly zero and the model begins as the unmodified base. Getting this backwards breaks training.

**QLoRA** adds: quantize the frozen base model to 4-bit NF4, keep LoRA parameters in bf16, plus **double quantization** (quantize the quantization constants) and **paged optimizers** to survive memory spikes. Result: fine-tuning a 65B model on a single 48GB GPU. It's the reason fine-tuning large models became broadly accessible.

**Limits worth stating:** LoRA is weaker than full fine-tuning when you need substantial new knowledge rather than behavior change; multiple merged LoRAs interfere with each other; and there's evidence LoRA "learns less and forgets less" — a real trade, not a free lunch. **DoRA** (decomposing into magnitude and direction) and **rsLoRA** (fixing the rank-scaling behavior) are the refinements to know.

---

### Q128. What are the other parameter-efficient methods, and how do they compare?

- **Adapters (Houlsby/Pfeiffer)** — insert small bottleneck MLPs inside each transformer block, train only those. The original PEFT method. Downside: they add sequential layers, so they add **inference latency** that can't be merged away. LoRA largely replaced them for that reason.
- **Prefix tuning** — prepend learned continuous vectors to the keys and values at every layer. No architecture change, but it consumes context length and is notoriously finicky to optimize.
- **Prompt tuning** — the minimal version: learn soft embedding vectors prepended only at the input. Extremely few parameters. Works well at very large scale, poorly at small scale, and is hard to train.
- **BitFit** — train only the bias terms. Astonishingly, competitive on some tasks. Mostly interesting as evidence about how much of fine-tuning is just re-calibration.
- **IA³** — learn per-channel rescaling vectors for keys, values, and FFN activations. Even fewer parameters than LoRA, mergeable, competitive on few-shot tasks.
- **(IA)³ / LoRA hybrids and MoE-LoRA** — the current frontier.

**The comparison that matters:** LoRA and IA³ are mergeable (no latency cost), adapters and prefix tuning are not. LoRA has the best quality-per-parameter across the widest range of tasks and is the pragmatic default. Full fine-tuning still wins when you need maximum quality and have the compute.

---

### Q129. What actually matters in fine-tuning data?

More than the method, in my experience.

- **Quality beats quantity, dramatically.** The LIMA result — 1,000 carefully curated examples producing a strong instruction-following model — is the canonical demonstration. A thousand excellent examples routinely beat fifty thousand mediocre ones, because the model is learning a *style and a decision procedure*, and inconsistent examples teach an inconsistent procedure.
- **Consistency is the thing you're actually teaching.** If two examples handle the same situation differently, you're teaching the model to be arbitrary. I'd rather remove an example than include an inconsistent one.
- **Diversity of task types** matters more than volume within a type.
- **Match the inference distribution.** If your production inputs are messy and your training examples are clean, you've trained for a distribution you'll never see.
- **How much:** hundreds to low thousands for format/tone; thousands to tens of thousands for a genuine skill; millions of tokens for domain adaptation via continued pretraining.
- **Continued pretraining vs instruction tuning** are different things. Continued pretraining (plain next-token prediction on domain text) teaches domain *knowledge and vocabulary*; instruction tuning (input/output pairs) teaches *behavior*. If your domain has unusual terminology, you may need both, in that order.
- **Watch for catastrophic forgetting** (Q94) — mix in general-domain data at some ratio, evaluate on a frozen general benchmark, and prefer LoRA if forgetting is a concern.
- **Evaluation is the hard part.** For generative fine-tuning there's often no clean metric. Build a fixed eval set of 100-200 representative cases with either reference answers or a rubric, and check it every run. Without that you're tuning blind, and "it seems better" is not a result you can defend.

---

## Part 20 — Inference Optimization

### Q130. What dominates LLM inference cost? Explain prefill vs decode.

These are two completely different computational regimes and conflating them is the most common mistake.

**Prefill** — processing the input prompt. All tokens are known, so you process them **in parallel** in one big matmul. This is **compute-bound**: high arithmetic intensity, GPU well-utilized. Cost scales with prompt length. Determines **time to first token (TTFT)**.

**Decode** — generating output tokens one at a time. Each step processes exactly **one** token, so the matmuls are matrix-*vector* products. You have to read all the model weights from HBM to compute one token's worth of output. That's an arithmetic intensity of roughly 1 — catastrophically **memory-bandwidth-bound**. The GPU's compute units sit almost idle. Determines **time per output token (TPOT)**.

The consequence that drives every optimization decision: **for decode, the cost is dominated by moving weights and KV cache from memory, not by FLOPs.** Which explains why:
- Weight-only quantization gives near-linear speedups in decode (fewer bytes to move) but little in prefill.
- **Batching is nearly free in decode** — reading the weights once and applying them to 32 sequences costs barely more than one sequence. Throughput scales almost linearly with batch size until you run out of KV cache memory. This is *the* reason serving systems batch aggressively.
- MQA/GQA matter enormously (Q34) — they shrink the KV cache, which is the other thing being read every step.

So: prefill wants fewer FLOPs, decode wants fewer bytes. Optimizations that help one often don't help the other.

---

### Q131. Explain the KV cache and PagedAttention.

**The KV cache**: during decode, attention needs the keys and values of all previous tokens. Since causal attention means those never change, you compute them once and cache them. Without it, generating token N requires re-encoding all N-1 previous tokens — O(n²) total instead of O(n).

**The size**, and it's worth being able to compute this:
`2 (K and V) × layers × kv_heads × head_dim × seq_len × batch × bytes_per_element`

For a 70B model with GQA at 8 KV heads, 80 layers, head_dim 128, fp16, a single 8k-token sequence is roughly 2×80×8×128×8192×2 bytes ≈ **2.7 GB**. With MHA instead of GQA it'd be 8× that. Multiply by batch size and you see instantly why KV cache, not weights, is the binding memory constraint at serving time.

**PagedAttention (vLLM)** solves the *fragmentation* problem. Naive serving pre-allocates a contiguous buffer per sequence sized to the max possible length. If a request generates 100 tokens out of a 4096 reservation, 97% is wasted, and you can't use it for another request because it's contiguous and reserved. Real systems were wasting 60-80% of KV memory this way.

PagedAttention borrows virtual memory: split the cache into fixed-size **blocks**, keep a per-sequence **block table** mapping logical positions to physical blocks, and allocate blocks on demand. Non-contiguous physical storage, near-zero waste. Bonus: **prefix sharing** — two requests with the same system prompt point at the same physical blocks, copy-on-write. That's a large win when every request shares a long system prompt, which is most production deployments.

The result is 2-4× throughput purely from memory management, no model changes. It's a beautiful example of a systems insight beating a modeling one.

---

### Q132. Static vs dynamic vs continuous batching.

- **Static batching** — collect N requests, run them together to completion, return. Problem: the batch runs until the *longest* sequence finishes. Short requests sit idle in the batch, holding GPU memory and adding latency. Utilization is poor when output lengths vary, which they always do.
- **Dynamic batching** — a short wait window to accumulate requests, then run. Better utilization, adds queuing latency. Standard for encoder models where all outputs are the same size.
- **Continuous / in-flight batching** — the one that matters for LLMs. Operate at the **iteration** level rather than the request level: after every decode step, evict finished sequences and admit new ones into the batch. A request that finishes at step 50 frees its slot immediately instead of waiting for a peer at step 2000.

Combined with PagedAttention, continuous batching is why vLLM/TGI/TensorRT-LLM get order-of-magnitude throughput improvements over naive serving. The scheduler also has to handle **prefill/decode interference** — a long prefill blocks decode steps for everyone in the batch, spiking their TPOT. **Chunked prefill** splits long prompts into pieces interleaved with decode steps to smooth this out.

---

### Q133. Explain speculative decoding.

The insight: decode is memory-bandwidth-bound, so the GPU has spare compute. Verifying multiple tokens in parallel costs nearly the same as generating one.

Mechanism: a small **draft model** (or a cheaper method) proposes k tokens quickly. The large **target model** then evaluates all k in a single forward pass — which is a parallel prefill-style operation, so it's cheap. A **rejection sampling** scheme accepts the longest prefix consistent with what the target model would have sampled, and resamples at the first disagreement.

The critical property: it is **mathematically exact.** The accepted output distribution is identical to sampling from the target model directly. You're not trading quality for speed; you're exploiting idle compute. That's the thing to emphasize, because people assume it's an approximation.

Speedup depends on the **acceptance rate** — how often the draft agrees. A well-matched draft model gets 2-3×. A poorly matched one can be *slower* than no speculation, because you pay for the draft and reject everything.

Variants: **Medusa** (extra prediction heads on the target model itself — no separate draft model to maintain), **EAGLE** (speculate in feature space rather than token space, higher acceptance), **lookahead decoding** (no draft model at all, uses n-gram parallelism), and **prompt lookup decoding** (draft by copying from the prompt — trivially simple and remarkably effective for summarization and code editing, where output heavily overlaps input).

**When it doesn't help:** at high batch sizes. Speculation spends compute to save memory bandwidth, but large batches already saturate compute. So it's a **low-latency, low-batch** optimization — great for interactive single-user serving, less useful for throughput-maximizing batch jobs. Knowing that boundary is the senior part of the answer.

---

### Q134. What metrics do you use for an inference service, and how do they trade off?

- **TTFT (time to first token)** — dominated by prefill and queuing. What users perceive as responsiveness.
- **TPOT / ITL (time per output token / inter-token latency)** — dominated by decode. Determines whether streaming feels smooth. ~50ms is roughly reading speed.
- **End-to-end latency** = TTFT + TPOT × output_length.
- **Throughput** — tokens/sec or requests/sec across the whole service. This is what determines cost per request.
- **Goodput** — throughput *of requests that met their SLO*. The metric that actually matters, because a system with great raw throughput that misses latency targets is failing.

**The fundamental tension:** larger batches → higher throughput, worse per-request latency. You cannot maximize both. So you set an SLO (e.g. p95 TTFT < 500ms, p95 TPOT < 60ms) and maximize throughput *subject to* it. Framing capacity planning that way, rather than as "make it fast," is the mature version.

Other levers: **prefix caching** for shared system prompts (huge and often overlooked — can cut TTFT dramatically); **request prioritization** for interactive vs batch traffic; **disaggregated prefill/decode** (run them on separate GPU pools since they have opposite bottlenecks — the current frontier in serving architecture); and **autoscaling on queue depth** rather than GPU utilization, because a decode-bound GPU shows low utilization while being fully saturated on bandwidth.

---

### Q135. What about non-LLM inference — a ranking model at high QPS?

Different problem, different toolkit, and this is closer to what you'd own day to day:

- **Export and compile** — ONNX Runtime, TensorRT, or `torch.compile`. Operator fusion and kernel selection routinely give 2-5× over eager PyTorch for free.
- **Batch on the server side** with a short accumulation window. For a ranking model scoring 500 candidates per request, that's already a natural batch.
- **Precompute everything that doesn't depend on the request.** Item-side embeddings, static features — this is the two-tower argument (Q116) applied generally.
- **Cache aggressively.** Feature caches, embedding caches, and full result caches for repeated queries. A cache hit is infinitely faster than any model optimization.
- **Quantize to int8** — for CPU inference especially, this is often 2-4×.
- **Distill to a smaller model** if the big one is genuinely too slow (Q122).
- **Right-size the hardware.** Many ranking models are faster on CPU at low batch than on GPU, once you account for transfer overhead. Measure rather than assume.
- **Cascade** — a cheap model filters, an expensive model refines only the survivors (Q117). Usually the biggest architectural win available.
- **Watch p99, not the mean.** Tail latency is driven by garbage collection, cold caches, feature-store timeouts, and stragglers — not by the model. Instrument the whole request path, because the model is often a minority of the latency budget and optimizing it is the wrong place to spend effort.

---

## Part 21 — Probability & Statistics for MLEs

### Q136. MLE vs MAP vs full Bayesian.

- **Maximum Likelihood Estimation (MLE)** — pick the parameters that make the observed data most probable. `argmax P(data|θ)`. It's what almost all standard training does: minimizing cross-entropy is MLE for a categorical model, minimizing MSE is MLE under Gaussian noise. Weakness: with little data it overfits hard, and it can produce degenerate answers (three coin flips, all heads → "P(heads) = 1.0").
- **Maximum A Posteriori (MAP)** — add a prior: `argmax P(data|θ)·P(θ)`. Take logs and the prior becomes an additive penalty. **This is exactly what regularization is.** A Gaussian prior on weights gives you L2; a Laplace prior gives you L1. That equivalence is worth being able to state, because it reframes weight decay from a hack into a modeling assumption.
- **Full Bayesian** — don't pick a single θ at all. Keep the whole posterior `P(θ|data)` and integrate over it when predicting. You get genuine uncertainty quantification, which is what you actually want for decision-making. The cost is that the integral is intractable for anything deep, so you approximate: MCMC (accurate, slow), variational inference (fast, biased), Laplace approximation, deep ensembles, or MC dropout.

**Why this matters for you specifically:** Thompson Sampling (Q98) needs a posterior. Bayesian optimization (Q101) needs a posterior. The bridge from "I train point-estimate models" to "I make decisions under uncertainty" runs directly through this material. The practical approximations that work at scale — **deep ensembles** (train 5 models, use their disagreement as uncertainty; consistently the strongest practical method) and **bootstrapped heads** (one shared trunk, several output heads trained on bootstrap samples; nearly free) — are the ones worth knowing.

Also worth distinguishing: **aleatoric uncertainty** (irreducible noise in the data — more data won't help) vs **epistemic uncertainty** (uncertainty about the model, which more data *does* reduce). Exploration should target epistemic uncertainty. Confusing the two means you'll explore forever in genuinely noisy regions.

---

### Q137. Explain the bias-variance decomposition, and the modern caveat.

Expected prediction error decomposes into three parts:
- **Bias²** — error from the model being too simple to represent the truth. A linear model on a curved relationship.
- **Variance** — error from sensitivity to the particular training sample. Retrain on different data, get a very different model.
- **Irreducible noise** — the floor.

The classical story: model complexity increases → bias falls, variance rises → there's a U-shaped test error curve with a sweet spot in the middle. This drives the traditional intuitions: bagging reduces variance (average many high-variance models), boosting reduces bias (sequentially fit residuals), regularization trades a little bias for a lot of variance reduction.

**The modern caveat, which you should raise because it shows you're current: this story is incomplete for deep learning.** **Double descent** is the observed phenomenon where test error follows the classical U-curve up to the interpolation threshold (where the model has just enough capacity to fit the training data exactly), spikes there, and then **decreases again** as you keep adding parameters. So a massively overparameterized network can generalize better than a "right-sized" one — which the classical picture says is impossible.

The explanation involves implicit regularization: among the many parameter settings that fit the training data perfectly, SGD tends to find low-norm, "simple" ones, and having more parameters gives it more such solutions to choose from. There's also **epoch-wise double descent** — the same curve over training time, not just model size.

What to take from it practically: **don't stop scaling a model because classical intuition says you'll overfit.** Measure. And be aware that early stopping and model selection heuristics built on the U-curve can be actively misleading in the overparameterized regime.

---

### Q138. What does a p-value actually mean, and what does a confidence interval mean?

Both are routinely misstated, and stating them correctly is a cheap credibility signal.

**A p-value** is: assuming the null hypothesis is true, the probability of observing data at least this extreme. That's it.

It is **not** the probability the null is true, and **not** the probability your result is a fluke. A p-value of 0.03 does not mean 3% chance you're wrong — the actual false discovery rate depends on how plausible your hypothesis was a priori and on your test's power. If you run experiments where only 10% of ideas genuinely work, a large fraction of your p<0.05 "wins" are false positives regardless of the p-value.

**A 95% confidence interval**: if you repeated this experiment many times and built an interval each time, 95% of those intervals would contain the true value. It is a statement about the *procedure*, not about this particular interval. You cannot say "there's a 95% probability the true effect is in this range" — that's a **credible interval**, which is the Bayesian object and requires a prior.

**Why to care in practice:** report intervals, not just p-values, because the interval tells you about **magnitude** and p-values don't. A statistically significant 0.02% lift and a non-significant but plausibly 3% lift are very different business situations, and only the interval reveals that. I'd rather see "+1.2% [-0.1%, +2.5%]" than "p = 0.08."

---

### Q139. Explain power analysis and MDE. Why run one before the experiment?

**Power** is the probability of detecting a real effect of a given size. **MDE (minimum detectable effect)** is the smallest effect your experiment can reliably detect at your sample size, significance level, and target power (conventionally 80%).

Sample size scales roughly with `σ² / MDE²`. The squared term is the crucial intuition: **detecting an effect half as large requires four times the sample.** That's why detecting small effects is so expensive, and why the answer to "can we just run it longer" is usually "much longer than you think."

**Why before, not after:**
1. If your MDE is 5% and you expect a 1% effect, the experiment **cannot succeed** and you shouldn't run it. Better to know that before spending three weeks.
2. Post-hoc power analysis is statistically meaningless — it's a deterministic function of the p-value you already have.
3. An underpowered experiment that comes back non-significant is uninformative, but everyone treats it as evidence the idea failed. That's how good ideas get killed.
4. The **winner's curse**: in an underpowered experiment, any effect that *does* reach significance must be large by chance, so the effect size is **systematically overestimated**. This is why so many shipped wins don't replicate — you're measuring the noise that got you over the line. Type M (magnitude) and Type S (sign) errors are the terms.

Ways to buy power without more traffic: variance reduction (Q140), a more sensitive metric, a better unit of randomization, or a one-sided test if the direction is genuinely predetermined.

---

### Q140. What is CUPED and what other variance reduction techniques matter?

**CUPED (Controlled-experiment Using Pre-Experiment Data)** uses pre-period data to remove predictable variance. If a user spent a lot last month, they'll probably spend a lot this month regardless of the treatment — that's variance in your metric that has nothing to do with your experiment.

Mechanically: `Y_adjusted = Y - θ(X - E[X])`, where X is a pre-experiment covariate (usually the same metric measured before the experiment) and θ is chosen as `Cov(X,Y)/Var(X)` — the value that minimizes the variance of the adjusted metric. The adjusted metric has the **same expected treatment effect** (because X is pre-treatment, it can't be affected by the treatment) but **lower variance**.

Variance reduction is roughly `ρ²`, the squared correlation between pre-period and in-period metric. Correlation of 0.7 → ~50% variance reduction → you need half the sample, or equivalently you can detect a 30% smaller effect. That's an enormous win for free.

The critical requirement: **X must be strictly pre-treatment.** Using any in-experiment covariate biases the estimate, and this is the mistake people make.

Related techniques:
- **Stratification / post-stratification** — block on known high-variance dimensions (country, platform, user tenure) and estimate within strata. Guarantees balance rather than relying on randomization.
- **Regression adjustment (CUPAC / ML-based)** — generalize CUPED by predicting the outcome from many pre-period features with an ML model and using the prediction as the covariate. Strictly stronger than single-covariate CUPED.
- **Trigger analysis** — only analyze users who could actually have been affected. If a change only affects the checkout page, including users who never reached checkout dilutes the effect and adds pure noise. Often the single biggest sensitivity gain available, and it's usually a data-availability problem rather than a statistical one.
- **Winsorization / capping** on heavy-tailed metrics like revenue, where one whale can dominate the variance. Introduces a little bias, removes a lot of variance.
- **Choosing a better metric** — a proxy metric with lower variance and a known relationship to the business metric.

↪ **Your hook:** with 1,200+ daily metric pipelines, whether the platform implemented CUPED and trigger analysis is a very natural interview question. Have a position on it.

---

### Q141. What's the peeking problem, and how do sequential tests fix it?

If you check an experiment repeatedly and stop when it hits p < 0.05, your actual false positive rate is far above 5% — with continuous monitoring it approaches 100% given enough time, because a random walk will eventually cross any fixed boundary. This is the single most common way experimentation platforms produce false results, and it's especially insidious because the incentive to peek is enormous.

Fixes:
- **Fixed-horizon discipline** — decide the sample size in advance and only look at the end. Statistically correct, organizationally very hard to enforce.
- **Group sequential testing** — pre-specify a small number of interim analyses with adjusted (stricter) boundaries, e.g. O'Brien-Fleming, which is very conservative early and relaxes toward the planned end. Standard in clinical trials.
- **Always-valid inference / sequential p-values** — construct confidence sequences that are valid at *every* point in time, so you genuinely can monitor continuously and stop whenever. Based on mixture sequential probability ratio tests or e-values. Costs some power relative to a fixed-horizon test at the planned endpoint, and buys you the ability to stop early safely. This is what modern experimentation platforms (Optimizely, Statsig, Eppo) implement and it's the right default for a self-serve platform where you cannot enforce discipline.

**The organizational point** worth making: this isn't really a statistics problem, it's an incentives problem. The right fix is to make the correct behavior the default in the tooling — hide the p-value until the horizon, or use always-valid inference so peeking is safe — rather than training people not to peek.

---

### Q142. What is multiple testing and how do you correct for it?

Test 20 metrics at α=0.05 and you expect one false positive by chance alone. Test 100 and you expect five. Any experimentation platform showing dozens of metrics has this problem structurally.

- **Bonferroni** — divide α by the number of tests. Controls **family-wise error rate** (the probability of *any* false positive). Simple and very conservative — with 50 metrics you're testing at α=0.001 and you'll miss real effects.
- **Holm-Bonferroni** — a step-down version, uniformly more powerful, same guarantee. Strictly better than plain Bonferroni; there's no reason to use plain Bonferroni.
- **Benjamini-Hochberg** — controls **false discovery rate** (the expected *proportion* of discoveries that are false). Much more powerful, and usually the right guarantee for exploratory analysis: "at most 10% of the things I flag are wrong" is a more useful promise than "there is at most a 5% chance I flagged anything wrong."

**The practical structure that actually solves this:** designate **one primary metric** in advance, tested at the full α. Everything else is either a **guardrail** (tested one-sided for harm, with a different threshold, because the cost of missing a regression differs from the cost of missing a win) or **exploratory** (flagged with FDR control, never used alone to justify a decision). Pre-registration of the primary metric is what makes this credible.

Related traps: **multiple variants** (an A/B/C/D test has more comparisons than you think), **repeated segment slicing** (twenty segments × ten metrics = 200 tests, and someone will find a "significant" result), and **sequential experiments on the same idea** until one works.

---

### Q143. What experiment design pitfalls should you watch for beyond the statistics?

- **Novelty and primacy effects** — users react to *change* itself. A new UI gets a temporary engagement bump that decays; a change to a familiar workflow gets a temporary dip that recovers. Detect by plotting the effect over time rather than looking only at the pooled average, and by holding a long-running holdback.
- **Network effects and interference** — treatment leaks between users. Marketplaces are the canonical case (Q105). Requires cluster or switchback randomization.
- **Survivorship / selection in the analysis population** — analyzing only users who completed an action, when the treatment changed who completes it, breaks randomization. Always analyze by **intent to treat**: everyone assigned to the arm, regardless of whether they experienced the treatment.
- **Simpson's paradox** — an effect that's positive in every segment can be negative in aggregate if segment sizes differ between arms. Usually a symptom of a randomization or logging bug (which SRM should catch), but it can also be real when there's a shift in traffic mix.
- **Metric dilution** — including users who couldn't have been affected (Q140's trigger analysis).
- **Long-term vs short-term divergence** — engagement-maximizing changes frequently degrade retention. If the only measurable metric is short-term, you're structurally biased toward changes that borrow from the future. The mitigation is a long-term holdback group, and it needs organizational commitment because it costs real traffic.
- **Multiple experiments interacting** — with 80+ concurrent experiments, interaction effects exist. Most platforms assume they're negligible (usually reasonable) but you want an interaction detection mechanism for the cases where they're not.

---

## Part 22 — Classical ML Depth

### Q144. Bagging vs boosting — what's the actual difference?

**Bagging (bootstrap aggregating)** — train many models **in parallel** on bootstrap samples (random samples with replacement), then average. Each model is a full-strength, low-bias, high-variance learner. Averaging independent-ish errors reduces variance without increasing bias. **Random Forest** adds a second randomization: at each split, only consider a random subset of features, which decorrelates the trees further (correlated errors don't average away, so decorrelation is the whole game).

**Boosting** — train models **sequentially**, each one correcting the previous ensemble's errors. In gradient boosting, each new tree fits the **negative gradient of the loss** with respect to the current predictions — for squared loss that's literally the residuals, but the framework generalizes to any differentiable loss, which is what makes it so flexible. Each learner is deliberately **weak** (shallow trees, high bias), and the ensemble reduces bias by accumulating many small corrections.

The consequences:
- Bagging is **parallelizable**; boosting is inherently sequential.
- Bagging is **hard to overfit** by adding more trees (you're averaging, so more is asymptotically fine). Boosting **can** overfit with too many rounds, which is why early stopping on a validation set is essential.
- Random Forest works well out of the box with near-default settings. GBDT needs tuning but has a higher ceiling — a tuned GBDT beats a tuned RF on most tabular problems.
- Boosting is more sensitive to noisy labels, because it keeps focusing on the examples it gets wrong, and mislabeled points are permanently wrong.

**Reference to Q137:** bagging attacks variance, boosting attacks bias. That's the one-line summary.

---

### Q145. XGBoost vs LightGBM vs CatBoost — what's actually different?

All three are gradient boosting on trees. The differences are in the engineering, and each one has a specific innovation worth naming.

**XGBoost**
- **Second-order optimization** — uses both gradient and Hessian in the split criterion, giving a better-justified split score than first-order alone.
- **Explicit regularization in the objective** — an L2 penalty on leaf weights and a complexity penalty per leaf (γ), so regularization is part of the tree-growing criterion rather than bolted on.
- **Sparsity-aware split finding** — learns a default direction for missing values at each split rather than imputing.
- **Level-wise growth** — grows all nodes at a depth before going deeper. More balanced, more conservative.

**LightGBM** — the speed one. Two key ideas:
- **Histogram-based splitting** — bucket continuous features into ~255 bins and find splits over bins instead of over sorted values. Turns split-finding from O(n log n) sorting into O(bins), and the memory drops massively. (XGBoost adopted this too, as `tree_method=hist`.)
- **Leaf-wise (best-first) growth** — split whichever leaf gives the biggest loss reduction, anywhere in the tree. Produces deeper, asymmetric trees and reaches lower loss with fewer leaves. The trade-off: **it overfits more easily on small data**, which is why `num_leaves` and `min_data_in_leaf` are the critical hyperparameters. `num_leaves` should be well below `2^max_depth`.
- Plus **GOSS** (sample rows by gradient magnitude — keep large-gradient rows, subsample small-gradient ones) and **EFB** (bundle mutually-exclusive sparse features into one).

**CatBoost** — the categorical one, and it solves a real statistical problem.
- **Ordered target statistics.** Naive target encoding (replace a category with its mean target) **leaks** — the row's own label is in its encoding, so the model overfits catastrophically. CatBoost imposes a random ordering and encodes each row using only rows *before* it, like an online estimate. This is a genuinely elegant fix.
- **Ordered boosting** — the same principle applied to the gradient estimates, addressing the subtle "prediction shift" bias where standard GBDT computes gradients on the same data used to fit the tree.
- **Symmetric (oblivious) trees** — the same split condition at every node of a level. Acts as strong regularization and makes inference extremely fast (the tree becomes an index lookup).
- Best defaults of the three; often strong with no tuning at all.

**How I'd choose:** LightGBM when speed and dataset size dominate and I'm going to tune. CatBoost when there are many high-cardinality categoricals or I want strong results without much tuning. XGBoost when I want the most battle-tested option with the widest ecosystem. Honestly, tuned properly on the same data they land within noise of each other most of the time — the differences that matter are training speed and how much tuning each needs.

---

### Q146. Which GBDT hyperparameters matter, in what order?

Roughly the order I'd tune:

1. **Learning rate × number of trees.** These trade off directly — lower LR needs more trees. Standard practice: set LR low-ish (0.05, or 0.01 if you can afford it), then use **early stopping on a validation set** to pick the number of trees automatically. This gives you one fewer thing to tune and it's the highest-value single practice.
2. **Tree complexity** — `max_depth` (XGBoost/CatBoost) or `num_leaves` (LightGBM). The main capacity knob. 6-8 depth or 31-127 leaves are reasonable starting points.
3. **`min_child_weight` / `min_data_in_leaf`** — minimum data per leaf. The main defense against fitting noise, and underrated. Raise it if you're overfitting.
4. **Subsampling** — `subsample` (rows) and `colsample_bytree` (features). 0.7-0.9 for both adds randomization and usually helps. Free variance reduction.
5. **Regularization** — `lambda` (L2), `alpha` (L1), `gamma` / `min_split_gain`. Useful but usually a smaller lever than the above.
6. **`scale_pos_weight`** for imbalance, if you're using it.

**Tuning method:** random search or Bayesian optimization (Q101) beats grid search decisively, because most hyperparameters don't matter and grid search wastes its budget exploring them exhaustively. Optuna with a pruner is the practical default.

---

### Q147. Explain SHAP, and how it differs from other feature importance methods.

**Built-in gain importance** (how much each feature reduced loss across all splits) — free, but **biased toward high-cardinality features**, because a continuous feature with many possible splits gets more chances to look good. Also inconsistent: it can decrease when a model genuinely relies on a feature more. Fine for a rough sanity check, not for anything you'd present.

**Permutation importance** — shuffle a feature's values and measure how much performance drops. Model-agnostic, measures actual predictive contribution on held-out data. Problems: **correlated features** — shuffling one when a correlated twin remains shows near-zero importance for both, so genuinely important features look useless. And shuffling creates unrealistic feature combinations the model never saw, so you're evaluating off-distribution.

**SHAP (SHapley Additive exPlanations)** — from cooperative game theory. The Shapley value of a feature is its average marginal contribution to the prediction across all possible orderings of feature inclusion. It's the unique attribution satisfying a set of desirable axioms (efficiency — attributions sum to the prediction minus a baseline; symmetry; dummy; additivity).

Why it's better: it's **local** (per-prediction, not just global), **consistent** (if a model relies on a feature more, its SHAP value won't decrease), and **signed** (tells you direction, not just magnitude). Global importance is just the mean absolute SHAP value across the dataset, so you get both views from one computation.

The catch: exact Shapley values are exponential in the number of features. **TreeSHAP** computes them exactly in polynomial time for tree ensembles, which is why SHAP is standard in GBDT workflows. For neural nets you use DeepSHAP or KernelSHAP, both approximations, both much slower.

**Caveats worth stating**, because uncritical SHAP use is common:
- **Correlated features split credit arbitrarily** between themselves. SHAP doesn't solve this; it just distributes it differently than permutation importance does.
- **SHAP explains the model, not the world.** A high SHAP value means the model uses that feature, not that the feature causes the outcome. Presenting SHAP as causal evidence to a stakeholder is a serious and common error.
- The choice of **background/baseline distribution** materially changes the values, and people rarely think about it.

---

### Q148. When would you still choose a linear model?

More often than the field's instincts suggest:

- **Very little data.** With a few hundred rows, a regularized linear model will often beat anything fancier and won't pretend to know things it doesn't.
- **You need genuine interpretability** — not "here's a SHAP plot," but "this coefficient is the effect, here's its confidence interval, and here's the statistical assumption behind it." Regulated domains (credit, insurance, healthcare) frequently require this, and a GBDT with SHAP is not a substitute for a model whose functional form you can defend.
- **You need extrapolation.** Trees cannot predict outside the range of the training targets — they output leaf averages, so a GBDT trained on prices up to $100 will never predict $150. For forecasting or any extrapolative task, that's disqualifying, and it's a limitation people forget until it bites.
- **Inference latency at extreme scale** — a dot product is hard to beat.
- **As the baseline.** A regularized logistic regression on good features is the number that everything else has to justify itself against. If your deep model beats it by 1%, that's important information about whether the complexity is worth it.
- **When the relationship genuinely is roughly linear**, which is more common than the ML community's priors suggest, especially after sensible feature transforms.

**Generalized additive models (GAMs)**, and especially **EBM (Explainable Boosting Machine)**, are the underused middle ground: they fit a separate smooth function per feature (plus optional pairwise interactions), so you get much of GBDT's accuracy with a model you can plot feature by feature and actually explain. Worth naming — it signals you think about interpretability as a design constraint rather than a post-hoc report.

---

## Part 23 — Coding Round Prep

*These are the five implementations that come up most. Write them once by hand, then reread them before an interview. The commented gotchas are the parts interviewers probe.*

### Q149. Implement scaled dot-product attention and multi-head attention from scratch.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0):
    """
    q, k, v: (batch, heads, seq_len, head_dim)
    mask:    (batch, 1, seq_q, seq_k) or broadcastable. True = KEEP, False = MASK OUT.
    """
    d_k = q.size(-1)

    # (B, H, Sq, D) @ (B, H, D, Sk) -> (B, H, Sq, Sk)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # ^ divide by sqrt(d_k): dot product of two unit-variance d-dim vectors has
    #   variance d, which saturates softmax and kills the gradient.

    if mask is not None:
        # Use a large negative rather than -inf: -inf can produce NaN if an entire
        # row is masked (all-padding sequence). This is THE classic NaN source.
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)

    attn = F.softmax(scores, dim=-1)   # normalize over KEYS (last dim), not queries
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)

    return torch.matmul(attn, v), attn   # (B, H, Sq, D)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0, causal=False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads = d_model, n_heads
        self.d_head = d_model // n_heads
        self.causal = causal

        # One fused projection for Q,K,V is faster than three separate matmuls.
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.dropout_p = dropout

    def forward(self, x, key_padding_mask=None):
        """
        x: (B, S, d_model)
        key_padding_mask: (B, S), True = real token, False = padding.
        """
        B, S, _ = x.shape

        qkv = self.qkv(x)                                  # (B, S, 3*d_model)
        qkv = qkv.view(B, S, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)                    # (3, B, H, S, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Build the combined mask. Both are "True = keep".
        mask = None
        if self.causal:
            causal = torch.ones(S, S, dtype=torch.bool, device=x.device).tril()
            mask = causal.view(1, 1, S, S)
        if key_padding_mask is not None:
            pad = key_padding_mask.view(B, 1, 1, S)         # mask KEYS
            mask = pad if mask is None else (mask & pad)

        ctx, attn = scaled_dot_product_attention(
            q, k, v, mask=mask, dropout_p=self.dropout_p if self.training else 0.0
        )

        # (B, H, S, D) -> (B, S, H*D). contiguous() is required before view()
        # because permute returns a non-contiguous tensor.
        ctx = ctx.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.out(ctx), attn
```

**Points interviewers probe:**
- *Why divide by √d_k?* Variance control — see Q30.
- *Why mask before softmax, not after?* Softmax normalizes; zeroing after breaks the sum-to-one. See Q32.
- *What if a whole row is masked?* NaN with `-inf`. Use `finfo.min`, or guarantee at least one valid key.
- *Why `.contiguous()`?* `permute`/`transpose` return views with non-standard strides; `view` requires contiguous memory. Use `.reshape()` if you want it handled automatically.
- *In production?* `F.scaled_dot_product_attention` dispatches to FlashAttention automatically. Write it by hand to show you understand it; use the built-in to ship.

---

### Q150. Implement an LSTM cell.

```python
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # One fused matmul for all four gates, then chunk. Much faster than 4 separate.
        self.x2h = nn.Linear(input_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            nn.init.uniform_(w, -std, std)
        # Forget-gate bias = 1.0. At random init the forget gate sits near 0.5,
        # so memory halves every step and long dependencies never get learned.
        # This is a real, well-documented trick -- see Q26.
        with torch.no_grad():
            self.x2h.bias[self.hidden_size:2 * self.hidden_size].fill_(1.0)

    def forward(self, x, state):
        h_prev, c_prev = state
        gates = self.x2h(x) + self.h2h(h_prev)              # (B, 4H)
        i, f, g, o = gates.chunk(4, dim=1)

        i = torch.sigmoid(i)      # input gate:  how much new info to write
        f = torch.sigmoid(f)      # forget gate: how much old cell state to keep
        g = torch.tanh(g)         # candidate:   the new content itself
        o = torch.sigmoid(o)      # output gate: how much cell state to expose

        c = f * c_prev + i * g    # ADDITIVE update -- this is the gradient highway
        h = o * torch.tanh(c)
        return h, c
```

**Points interviewers probe:**
- *Why does this fix vanishing gradients?* `c_t = f*c_{t-1} + ...` — the gradient path through the cell state is multiplied by `f`, which the network can learn to hold near 1. Additive, not repeated matrix multiplication.
- *Why tanh on the candidate but sigmoid on the gates?* Gates must be in [0,1] to act as soft switches. The candidate is content, so it wants a zero-centered range.
- *Why fuse the four gate matmuls?* One `(input, 4H)` GEMM is far more efficient than four `(input, H)` ones.

---

### Q151. Implement NDCG@k.

```python
import numpy as np

def dcg_at_k(relevances, k):
    """relevances: array of graded relevance in RANKED order."""
    rel = np.asarray(relevances, dtype=float)[:k]
    if rel.size == 0:
        return 0.0
    # gain = 2^rel - 1 (exponential gain), discount = log2(rank + 1), ranks 1-indexed
    discounts = np.log2(np.arange(2, rel.size + 2))
    return float(np.sum((2 ** rel - 1) / discounts))

def ndcg_at_k(y_true, y_score, k=10):
    """
    y_true:  true relevance grades, order corresponds to y_score
    y_score: model scores
    """
    order = np.argsort(-np.asarray(y_score), kind="stable")   # stable = deterministic ties
    ranked = np.asarray(y_true)[order]

    dcg = dcg_at_k(ranked, k)
    idcg = dcg_at_k(sorted(y_true, reverse=True), k)          # ideal ranking

    if idcg == 0.0:
        return 0.0     # no relevant items -- convention varies; some libs return NaN.
                       # Say which convention you're using; it changes the mean.
    return dcg / idcg
```

**Points interviewers probe:**
- *Why normalize?* Queries with different numbers of relevant items aren't comparable otherwise (Q74).
- *Why `2^rel - 1`?* Exponential gain makes highly-relevant items count disproportionately. Linear gain (`rel`) is the other convention — state which you're using.
- *Ties?* Use a stable sort, or average over random permutations of ties. Unstable sorting makes your metric non-deterministic across runs, which is a real and confusing bug.
- *No relevant items?* Skipping vs returning 0 changes the mean substantially. Be explicit.

---

### Q152. Write a correct training loop.

```python
def train_one_epoch(model, loader, optimizer, scheduler, scaler, device,
                    accum_steps=1, max_grad_norm=1.0, log_every=50):
    model.train()                       # BatchNorm/Dropout in train mode
    running_loss, n_batches = 0.0, 0
    optimizer.zero_grad(set_to_none=True)   # set_to_none is faster than zeroing

    for step, batch in enumerate(loader):
        # non_blocking=True needs pin_memory=True in the DataLoader to actually help
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            loss = loss / accum_steps    # CRITICAL: else effective LR scales with accum

        scaler.scale(loss).backward()

        if (step + 1) % accum_steps == 0:
            scaler.unscale_(optimizer)   # unscale BEFORE clipping, or you clip the
                                         # scaled gradients and the threshold is wrong
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()             # per-step scheduler goes here, not per-epoch

        # .item() forces a GPU->CPU sync. Doing it every step stalls the pipeline.
        # Accumulate on-device and sync only when logging.
        running_loss += loss.detach() * accum_steps
        n_batches += 1

        if step % log_every == 0:
            val = (running_loss / n_batches).item()   # one sync, every log_every steps
            if not math.isfinite(val):
                raise RuntimeError(f"Non-finite loss at step {step}")

    return (running_loss / max(n_batches, 1)).item()


@torch.no_grad()                        # no graph, big memory saving
def evaluate(model, loader, device):
    model.eval()                        # BatchNorm uses running stats, Dropout off
    total_loss, total_n = 0.0, 0
    for batch in loader:
        x = batch["x"].to(device); y = batch["y"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x), y, reduction="sum")
        total_loss += loss.item(); total_n += y.numel()
    return total_loss / total_n
```

**The gotchas, which are the actual content of this question:**
1. `model.train()` / `model.eval()` — forgetting `.eval()` is the single most common bug in PyTorch.
2. `zero_grad()` at the right point — gradients accumulate by default.
3. Divide loss by `accum_steps`, or your effective LR is silently multiplied.
4. **Unscale before clipping** with AMP, or the clip threshold is meaningless.
5. `scheduler.step()` per optimizer step for warmup/cosine, per epoch for epoch-based schedules. Mixing these up is common and silently ruins the schedule.
6. Avoid `.item()` in the hot loop — it forces a device sync (Q83).
7. `@torch.no_grad()` on eval.
8. Reduction consistency — `mean` per batch weights small final batches too heavily; use `sum` and divide by total count for a correct epoch average.

---

### Q153. Write a collate function for variable-length sequences.

```python
from torch.nn.utils.rnn import pad_sequence

def collate_variable_length(batch, pad_id=0, max_len=None):
    """
    batch: list of dicts with "tokens" (1-D LongTensor) and "label".
    Returns padded tensors plus an attention mask (True = real token).
    """
    seqs = [b["tokens"] for b in batch]
    if max_len is not None:
        seqs = [s[:max_len] for s in seqs]     # truncate BEFORE padding

    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded = pad_sequence(seqs, batch_first=True, padding_value=pad_id)

    # mask: (B, S), True where real. arange broadcast against lengths.
    mask = torch.arange(padded.size(1))[None, :] < lengths[:, None]

    return {
        "tokens": padded,
        "mask": mask,
        "lengths": lengths,
        "labels": torch.stack([b["label"] for b in batch]),
    }


def masked_mean_pool(hidden, mask):
    """
    hidden: (B, S, D); mask: (B, S) True = real token.
    THE bug from Q57: averaging over padding makes an entity's embedding
    depend on what else was in its batch.
    """
    m = mask.unsqueeze(-1).to(hidden.dtype)           # (B, S, 1)
    summed = (hidden * m).sum(dim=1)                  # (B, D)
    counts = m.sum(dim=1).clamp(min=1e-9)             # avoid div-by-zero
    return summed / counts
```

**Points interviewers probe:**
- *Why truncate before padding?* Otherwise you pad to the longest sequence then cut, wasting memory and possibly cutting real tokens after padding.
- *What's the mask convention?* PyTorch is inconsistent — `nn.MultiheadAttention` uses `key_padding_mask` where **True means ignore**, the opposite of most other APIs. Always state your convention in a comment. This causes real bugs.
- *How do you reduce padding waste?* **Length-grouped batching** (a sampler that buckets similar-length sequences together). Can cut padding compute by more than half on skewed length distributions. Trade-off: less randomness in batch composition, so shuffle within buckets.
- *Ragged alternatives?* Nested tensors or packed sequences avoid padding entirely, but with narrower op support.

---

### Q154. What PyTorch gotchas would you check in a code review?

A checklist worth keeping:

- `model.eval()` missing at inference; `model.train()` not restored afterward.
- `torch.no_grad()` missing in eval or in feature-extraction loops → memory blowup.
- In-place ops (`x += y`, `relu_()`) on tensors needed for backward → autograd error or, worse, silently wrong gradients.
- Loss reduction mismatch between training and eval.
- `.detach()` vs `.item()` vs `.cpu()` confusion causing syncs or memory leaks. Accumulating a *tensor* (with graph attached) across a loop leaks the whole graph — a classic OOM cause.
- DataLoader with `num_workers=0`, no `pin_memory`, no `persistent_workers`.
- Seeds not set for `random`, `numpy`, and `torch` — and in DataLoader workers, which need `worker_init_fn`.
- Device mismatches handled by silent `.to(device)` scattered everywhere rather than one place.
- Learning rate scheduler stepped at the wrong granularity.
- Weight decay applied to biases and norm parameters (Q10).
- Masks built with the wrong polarity.
- Broadcasting bugs: `(B,1)` vs `(B,)` in a loss silently produces a `(B,B)` tensor and a meaningless mean. Assert shapes.
- Not checking `loss.isfinite()` in long runs.
- Checkpoint saving the model but not optimizer/scheduler/RNG/data-position state (Q85).

---

## Part 24 — Case Study Drills

### The framework

Every ML system design question gets the same skeleton, and using it visibly is itself the signal:

1. **Clarify** (2-3 min) — the objective and how it's measured, scale, latency budget, who consumes the output, what exists today. Never skip this; jumping to architecture is the most common failure.
2. **Frame as an ML problem** — what's the unit of prediction, the label, the training data, the loss? State the objective/proxy-metric gap explicitly.
3. **Data** — sources, labels, the time-based split, leakage risks.
4. **Baseline first** — heuristic, then simple model. State what you'd have to beat.
5. **Model** — architecture and why, features, cold start.
6. **Evaluation** — offline metrics, online metrics, guardrails, why offline should predict online.
7. **Serving** — latency, batching, caching, staleness, fallbacks.
8. **Monitoring and iteration** — drift, retraining cadence, what alerts.
9. **Trade-offs you'd volunteer** — this is the part that separates levels.

---

### Q155. Design search ranking for a food delivery app.

**Clarify:** what's the objective — clicks, orders, GMV, long-term retention? Query volume and latency budget? How much of the traffic is navigational ("McDonald's") vs exploratory ("something spicy")? Personalized or not?

**Framing:** it's a multi-stage cascade, not one model.
- **Query understanding** — spell correction, intent classification (navigational vs category vs dish), entity linking.
- **Retrieval** — hybrid (Q120): BM25 over merchant and menu text plus a two-tower dense retriever, fused with RRF. Filter hard constraints (delivery radius, open now, availability) *inside* the index (Q121).
- **Ranking** — a multi-task model predicting p(click), p(order), and expected basket value, combined into an expected-value score. GBDT or a DNN over dense + embedding features.
- **Re-ranking** — diversity (don't show eight burger places), business rules, sponsored placement blending.

**Features:** query-merchant relevance signals, merchant quality (rating, cancellation rate, prep time), personalization (the user's history embedding, prior orders from this merchant), context (time of day, weather, day of week), and geo/ETA.

**Key trade-offs to volunteer:**
- **Relevance vs conversion.** Optimizing clicks pushes toward clickbait imagery; optimizing orders pushes toward cheap and familiar. Both drift from "found what I wanted." Need a relevance guardrail.
- **Personalization vs discovery** — a well-personalized system narrows the catalog over time (Q92).
- **Marketplace balance** — ranking affects merchant livelihoods, so supply-side fairness and new-merchant exposure are real constraints, not nice-to-haves.
- **ETA as both a feature and a promise** — ranking on ETA means you're making a commitment the ops system has to keep.

**Evaluation:** offline NDCG on logged relevance with position debiasing (Q77); online A/B on orders with guardrails on relevance, latency, catalog coverage, and merchant-side metrics.

---

### Q156. Design a fraud detection system.

**Clarify:** what fraud — stolen cards, promo abuse, collusion between driver and merchant, fake accounts? What's the cost asymmetry between a false positive (blocking a real customer, very expensive in lifetime value) and a false negative (chargeback, bounded)? Real-time block or async review? Regulatory constraints on explainability?

**What makes this different from ranking**, and I'd lead with these because they're the interesting part:

1. **Adversarial.** Your adversary adapts. A model that works today degrades because fraudsters probe it and route around it. This means **fast retraining cadence**, monitoring for sudden distribution shifts, and holding some detection logic back from any single model so it can't be fully reverse-engineered.
2. **Extreme imbalance** — often 0.1% or less (Q93). PR-AUC, not ROC-AUC; ROC-AUC looks great at 0.99 and means nothing here.
3. **Delayed and biased labels.** Chargebacks arrive 30-90 days later, so recent data is unlabeled. Worse: **you only observe outcomes for transactions you approved.** Blocked transactions have no label, ever. That's a selection bias that compounds every retraining cycle. The fix is a deliberate small **random-approval holdout** — you accept some known fraud loss to buy unbiased labels. That's a business conversation with a computable cost, and being able to frame it that way is the senior move.
4. **Graph structure matters more than features.** Fraud is relational — shared devices, cards, addresses, IPs. A graph model or even simple graph-derived features (connected component size, shared-device count) usually beats a better tabular model on the same rows.

**Approach:** rules engine for known patterns and hard blocks (fast, auditable, instantly updatable) + supervised GBDT on engineered features including velocity/aggregation features + unsupervised anomaly detection for novel patterns + graph features. Ensemble with tiered actions: allow / step-up authentication / manual review / block. **Step-up auth is the key product primitive** — it converts a binary decision into a graduated one and dramatically reduces the cost of false positives.

**Serving:** hard latency budget (this is inline with checkout), so real-time feature computation with a feature store, and a fail-open or fail-closed decision you should make explicitly.

**Evaluation:** precision at the operating threshold, recall on known fraud, dollar loss prevented vs dollars of legitimate volume blocked. Monitor the review queue's precision — it's your fastest feedback loop.

---

### Q157. Design a driver dispatch / matching system.

**Why this is interesting:** it's genuinely not a prediction problem, it's an **optimization under uncertainty** problem, and saying so early is the whole point.

**Framing:** at any moment you have open requests and available drivers. You want an assignment maximizing total value. That's a **bipartite matching** problem (Hungarian algorithm or min-cost flow), where the edge weights come from ML predictions — ETA, acceptance probability, completion probability, expected trip value.

**The layered structure:**
- **ML layer** predicts the quantities: ETA (quantile loss, Q114 — you care about the tail, not the mean), p(accept), p(cancel), expected duration.
- **Optimization layer** does the assignment given those predictions.
- **Policy layer** decides *when* to match — and this is the subtle part. Matching immediately is greedy; **batching requests over a short window** (10-30 seconds) lets the optimizer see more options and produces better global matches. That's a direct latency-vs-quality trade-off with real user experience implications.

**Where sequential decision-making enters** (and this is your decision-intelligence hook): assigning a driver now removes them from the pool for future requests. So a greedy match can be globally suboptimal — you might want to hold a driver for a high-value request likely to arrive. That's genuinely RL-shaped: state (supply/demand distribution), action (assignment), delayed reward. Approaches range from simple (value-function-based dispatch: estimate the future value of a driver's location and time, subtract it from the match score) to full RL. **Spatiotemporal value functions** are the pragmatic middle and what most production systems actually use.

**Evaluation:** this is where I'd be emphatic — **you cannot A/B test this per-user**, because of interference (Q105). Assigning a driver to a treatment user removes them from control's pool. You need **switchback experiments** or region-based randomization, plus a simulator for offline policy evaluation.

**Trade-offs to volunteer:** rider wait time vs driver utilization vs driver earnings fairness; short-term efficiency vs long-term supply retention (over-utilizing your best drivers burns them out); surge pricing as a supply lever with real fairness and PR consequences.

↪ **Your hook:** this is Grab's actual problem domain. If you have context on it, this is the case study to be strongest on.

---

### Q158. Design an offline evaluation platform.

**Why this one matters for you:** it's the case study that plays directly to your experimentation background, and it's a system design question most ML candidates can't answer well.

**The goal:** let teams estimate the online impact of a model or policy change *without* running an experiment, with a known and monitored relationship between offline estimates and online outcomes.

**Components:**

1. **Logging layer.** The foundation, and the part people skip. Log at decision time: the candidate set, the features used, the scores, the action taken, and — critically — **the propensity** (Q100). If the serving policy is deterministic, inject controlled randomization on a small traffic slice specifically to make evaluation possible. This is an upfront cost that pays for everything downstream.
2. **Replay / counterfactual evaluation.** Given logged data and a new policy, estimate its performance with IPS, self-normalized IPS, and doubly robust estimators (Q100). Report all three plus their variance — when they disagree, that disagreement is information.
3. **Point-in-time correct feature retrieval.** Reconstruct exactly the features that *were available* at decision time. This is the hardest engineering piece and the biggest source of silent leakage (Q91).
4. **Simulation** where replay isn't enough — for policies whose actions change the environment (dispatch, pricing), you need an environment model, and you need to be honest about its fidelity.
5. **Metric library** with consistent definitions shared with the online experimentation platform. If offline NDCG and online NDCG are computed differently, no comparison means anything.
6. **The calibration loop, which is the actual product.** Store every offline prediction alongside the eventual online result, and **track the correlation over time.** If offline says +3% and online says +0.2%, that discrepancy is the platform's most important output. The platform's job isn't to produce numbers, it's to produce numbers that have been *shown* to predict reality.

**Trade-offs:** exploration cost (randomization degrades near-term experience to buy evaluation capability); variance vs bias in estimator choice; and the organizational reality that a platform which frequently says "your model isn't better" will be unpopular until it's been right a few times publicly.

---

## Part 25 — Behavioral & Narrative

### The structure

**STAR** — Situation, Task, Action, Result — but the version that works at senior level weights it differently. Most people spend 80% on Situation. Invert it:

- **Situation/Task: 20%.** Just enough context. Two sentences.
- **Action: 50%.** Specifically what *you* did. First person singular. "We" is where senior interviews go to die — if the interviewer can't tell what you personally contributed, you get graded down regardless of how impressive the project was.
- **Result: 20%.** Numbers. Even approximate ones, clearly labeled as approximate.
- **Reflection: 10%.** What you'd do differently. This is the part junior candidates skip and it's the part that reads as senior.

**What L4 signals look like**, and these are what you're actually being scored on:
- **Scope** — you influenced beyond your own tasks.
- **Judgment under ambiguity** — you made a call without complete information and can explain the reasoning, not just the outcome.
- **Trade-off articulation** — you can name what you gave up.
- **Ownership of failure** — without excessive self-flagellation or blaming others.
- **Influence without authority** — you changed what other teams did through argument, not org chart.

**A note on preparation.** Write out 6-8 stories once, properly. Most behavioral questions are the same 8 stories re-angled. Prepare the *stories*, not the answers to specific questions, then select and re-frame live.

---

### Q159. The core stories to prepare

For each, write out the STAR and the number. Prompts are drawn from your CV:

**1. The biggest technical thing you built.**
→ The merchant representation system. Lead with *why it existed*, not what it was. The strongest framing: "multiple teams were each hand-crafting merchant features; a shared representation let them all move faster." That's scope and impact, not just a model.

**2. A project that failed or didn't ship.**
→ Everyone has one. What matters: did you kill it yourself, and how fast? "I ran the ablation, it showed the gain came from a leaked feature, I killed the project and wrote up the leakage pattern so the team wouldn't repeat it" is a *stronger* story than a success. Don't pick a fake failure like "I worked too hard."

**3. A technical disagreement.**
→ Structure: what the disagreement was, what data you brought, what you conceded, what the outcome was. Best version ends with you having been partly wrong and saying so plainly. Interviewers are testing whether you can be disagreed with.

**4. Influencing without authority.**
→ Getting a downstream team to adopt your embeddings, or getting an org to standardize on your Datalake design. The Saltside/Grab datalake work is a natural fit — that's a governance and adoption story as much as a technical one.

**5. Mentoring or leveling someone up.**
→ At 20 years in, they will expect this. Specific person, specific change, specific outcome.

**6. Handling ambiguity / a poorly-defined problem.**
→ "Make merchant recommendations better" → how you turned it into a measurable problem. The framing move (defining the metric, defining the baseline) is the content here.

**7. A production incident.**
→ Detection, diagnosis, mitigation, root cause, and the *systemic* fix. The last part is what separates levels — "I fixed it" vs "I fixed it and added the guardrail that makes this class of bug impossible."

**8. Pushing back on a stakeholder.**
→ Saying no to a request, or refusing to ship something. In your case, likely an experimentation story: telling someone their result wasn't valid. Those are high-stakes and show spine.

**9. The transition story.** (Q106 has a scripted version.)

---

### Q160. How do you handle "what's your weakness" and similar traps?

Answer honestly with something real, then show the mechanism you've built around it. The structure is: **the weakness → a concrete instance where it cost something → what you now do about it.**

For your profile, plausible honest answers:
- "I default to building infrastructure when the situation calls for a quick hack to test an idea. I've had to learn to explicitly ask 'is this a throwaway or a foundation' before starting, because I'd previously spent three weeks on a framework for an experiment that got killed in week two."
- "Depth in modern modeling relative to my systems depth. I'm four years in on ML versus twenty in data systems, and there are areas — RL in particular — where I know the concepts but haven't shipped. I've been closing that deliberately."

**What not to do:** a disguised strength ("I'm a perfectionist"), something disqualifying for the role, or something with no remediation attached.

**The related trap** for your profile: *"Won't you be bored / are you overqualified?"* Answer it directly rather than deflecting. Something like: "I've been the most senior person in the room on data infrastructure for a while. I'd rather be the person learning modeling depth on a strong team than the ceiling on a weaker one. If the concern is whether I'll be frustrated taking direction on ML from someone more junior in years — I won't, because they'd be more senior in the thing I'm here to get better at."

---

## Part 26 — Generative Systems & Modern LLM Practice

### Q161. Design a RAG system and name its failure modes.

**Pipeline:** ingest → chunk → embed → index → retrieve → rerank → assemble context → generate → (optionally) cite and verify.

**The failure modes, which is where the real content is:**

1. **Retrieval failure is the dominant one.** The overwhelming majority of "the LLM hallucinated" complaints in RAG systems are actually "the retriever didn't find the right chunk." Always evaluate retrieval **separately** from generation — measure recall@k of the gold chunk. If retrieval recall is 60%, no amount of prompt engineering fixes your system.
2. **Chunking destroys context.** A chunk split mid-table, mid-list, or mid-argument is unusable. Fixed-size chunking is the default and it's bad. Better: structure-aware chunking (by section/heading), overlapping windows, and **contextual retrieval** — prepend a short LLM-generated summary of the document context to each chunk before embedding, which substantially improves retrieval on ambiguous chunks. Also consider **small-to-big**: embed small precise chunks for retrieval but return the larger surrounding context to the LLM.
3. **Lost in the middle.** LLMs attend better to the beginning and end of the context than the middle. So chunk *ordering* matters — put the highest-relevance chunk first or last, not buried at position 7.
4. **No answer is a valid answer.** The model will confabulate from irrelevant context rather than say "not found." Needs explicit instruction, and ideally a relevance threshold that short-circuits generation when retrieval scores are all low.
5. **Query-document mismatch.** A user's question and the answer passage often share few words and different framing. Fixes: **HyDE** (generate a hypothetical answer, embed *that*, and retrieve against it), query expansion, or multi-query generation.
6. **Multi-hop questions** that need information combined from several documents. Single-shot retrieval fails structurally. Needs iterative retrieval or an agentic loop.
7. **Stale index** vs source of truth.
8. **Conflicting sources** — the model has no principled way to adjudicate. Needs metadata (recency, authority) surfaced in the context.

**Evaluation:** separate metrics for retrieval (recall@k, MRR) and generation (faithfulness — is the answer supported by the retrieved context; answer relevance; and groundedness/citation accuracy). RAGAS-style frameworks operationalize this. **Build a fixed eval set of 100-200 real questions with known answers before building the system**, not after — otherwise every change is a vibe check.

---

### Q162. Explain the RLHF pipeline, and DPO vs PPO.

**RLHF (Reinforcement Learning from Human Feedback)** has three stages:

1. **SFT (supervised fine-tuning)** — train on human-written demonstrations of good responses. Gets the model into the right output space.
2. **Reward model** — collect human *preferences* (given two responses, which is better), and train a model to predict them, typically with a Bradley-Terry loss on the score difference. Preferences are used rather than absolute ratings because humans are far more reliable at comparison than at scoring.
3. **RL optimization** — use the reward model as the reward signal and optimize the policy with **PPO**, plus a **KL penalty** to the SFT reference model. That KL term is essential: without it the policy drifts to whatever exploits the reward model, producing degenerate high-reward gibberish. This is **reward hacking**, and it's the central practical problem.

**PPO (Proximal Policy Optimization)** works by clipping the policy update so each step can't move the policy too far from the previous one — that's the "proximal" part, and it's what makes on-policy RL stable enough to use. In the RLHF setting it needs four models in memory (policy, reference, reward, value), it's sensitive to hyperparameters, and it's operationally painful.

**DPO (Direct Preference Optimization)** is the important simplification. The insight: under the RLHF objective there's a closed-form relationship between the optimal policy and the reward function — so you can **reparameterize the reward in terms of the policy itself** and optimize preferences directly with a simple classification-style loss. No reward model, no sampling, no RL loop. Just a supervised objective on preference pairs.

**Trade-offs, which is what you'd actually be asked:**
- DPO is far simpler, cheaper, and more stable. It's the default for most fine-tuning now.
- PPO still tends to win at the frontier, plausibly because on-policy sampling lets the model explore and get feedback on *its own current outputs*, whereas DPO learns from a fixed offline preference dataset and can be off-distribution.
- DPO is sensitive to the quality and coverage of the preference data, and it has a known tendency to reduce the probability of *both* the chosen and rejected responses (it optimizes the margin, not the absolute likelihood), which can degrade the model in unexpected ways.
- **Variants to know:** IPO (fixes a DPO overfitting issue), KTO (works with binary good/bad labels instead of pairs — much easier data collection), ORPO (merges SFT and preference optimization into one stage), and **RLAIF / Constitutional AI** (AI-generated preferences instead of human ones, which is how this scales).

↪ **Your hook:** the reward-hacking problem is a specification-gaming problem, which is the same failure class as optimizing a proxy metric that diverges from the business objective (Q76). You've lived that on the experimentation side.

---

### Q163. What's an agent, and what actually breaks?

An agent is an LLM in a loop with tools: it observes, decides on an action (usually a tool call), executes it, observes the result, and repeats until done. The core loop is trivially simple; making it reliable is not.

**What breaks:**

1. **Compounding error.** If each step is 95% reliable, a 20-step task succeeds 36% of the time. This is the fundamental constraint and every other issue is downstream of it. Implications: minimize step count, add verification steps, and design for recovery rather than perfection.
2. **No recovery from bad states.** Models tend to keep going rather than backtrack. Explicit checkpointing and the ability to abandon a path help.
3. **Context growth.** Every observation accumulates. Long trajectories blow the context window and dilute attention. Needs summarization, or externalizing state to a scratchpad/file rather than holding it in context.
4. **Tool misuse** — wrong arguments, hallucinated tools, misinterpreting errors. Tool descriptions matter enormously; treat them as prompts, not documentation.
5. **Loops** — the same failing action repeated. Needs loop detection and a step budget.
6. **Evaluation is genuinely hard.** Trajectories vary, there are many valid paths, and success is often partial. You need end-state checks (did the task actually get done, verified independently), not trajectory matching.
7. **Cost and latency** are unbounded by default. A step budget and a cost ceiling are operational requirements, not nice-to-haves.

**The engineering judgment:** most tasks framed as "agentic" are better served by a **fixed pipeline with LLM steps** than by an open-ended loop. Determinism where you can have it, agency only where the branching genuinely can't be enumerated. That's the same instinct as Q99 — don't reach for the general framework when the constrained one solves your problem.

---

### Q164. How do you evaluate an LLM system?

The hardest unsolved practical problem in the space, and having a structured answer is valuable.

**Levels:**
- **Unit level** — deterministic checks: does it output valid JSON, does it call the right tool, does it stay within length. Cheap, run on every change.
- **Reference-based** — compare to a gold answer. Exact match for extraction, F1 for spans. Fails for open-ended generation because there are many correct answers; BLEU/ROUGE correlate poorly with quality.
- **LLM-as-judge** — a strong model scores outputs against a rubric. Currently the practical workhorse. **Known biases you must control for:** position bias (judges favor the first option — randomize order), verbosity bias (judges favor longer answers), self-preference (models favor their own outputs), and poor calibration on fine-grained scales. Mitigations: pairwise comparison rather than absolute scoring, order randomization, a detailed rubric with explicit criteria, and **validating the judge against human labels on a sample** before trusting it. A judge you haven't validated is a random number generator with good manners.
- **Human evaluation** — the ground truth, expensive, needs clear guidelines and inter-annotator agreement measurement.
- **Online** — A/B on real user metrics. Everything above is a proxy for this.

**The practical setup I'd build:** a fixed eval set of 100-300 real cases, versioned in git alongside the prompts and code. Run it on every change. Track scores over time. **Add every production failure to the eval set** — that's the flywheel, and it's the single highest-value habit. Without a regression suite, prompt engineering is undirected and every fix silently breaks something else.

---

## Part 27 — Fairness, Privacy & Governance

### Q165. Explain the main fairness definitions and why you can't satisfy them all.

The three common ones:

- **Demographic parity** — the positive prediction rate is equal across groups. `P(ŷ=1|A=a)` is the same for all a. Ignores whether the base rates genuinely differ.
- **Equalized odds** — true positive rate and false positive rate are equal across groups. Conditions on the actual outcome, so it permits different prediction rates if the base rates differ. **Equal opportunity** is the weaker version requiring only equal TPR.
- **Calibration within groups** — a predicted 0.7 means 70% for every group.

**The impossibility result** (Kleinberg et al.; Chouldechova): if base rates genuinely differ between groups, you **cannot** simultaneously satisfy calibration and equalized odds, except in degenerate cases. This is a mathematical theorem, not an engineering limitation. So "make the model fair" is not a well-posed request — someone has to choose *which* fairness definition applies, and that's a normative decision that belongs with policy, legal, and affected stakeholders, not with the ML engineer alone.

The COMPAS recidivism debate was exactly this: the tool was calibrated across races, and it had unequal false positive rates. Both claims were true simultaneously. The argument was about which definition mattered, and that argument is not resolvable by better modeling.

**What I'd actually do in practice:** measure several definitions and report them rather than optimizing one silently; involve the people who own the policy decision; and be very clear about the difference between *fairness of the model* and *fairness of the system it sits in*. A perfectly fair model deployed into a biased process doesn't produce fair outcomes.

---

### Q166. Where does bias enter an ML pipeline?

Worth enumerating because "the data is biased" is too coarse to act on:

- **Historical bias** — the data faithfully records a world with existing inequity. The model is accurate *and* harmful. No amount of data cleaning fixes this; it's a question about whether to build the thing.
- **Representation bias** — some groups are underrepresented, so the model is simply less accurate for them. This is often the most tractable one: measure per-group performance and fix the sampling.
- **Measurement bias** — the proxy label differs in quality across groups. Arrests as a proxy for crime; clicks as a proxy for relevance where UI treats groups differently.
- **Aggregation bias** — one model for groups that need different models. Simpson's-paradox-shaped.
- **Learning bias** — the objective itself. Optimizing overall accuracy sacrifices minority-group performance because they contribute less to the average.
- **Deployment bias** — the model is used differently than intended, or in a context it wasn't validated for.
- **Feedback loops** (Q92) that amplify all of the above over time.

**The practical minimum I'd insist on:** *always* report metrics disaggregated by relevant groups, not just in aggregate. Most fairness problems are invisible in the aggregate number and obvious the moment you slice. That's a cheap, non-controversial practice that catches a lot.

---

### Q167. What's differential privacy and when would you use it?

**Differential privacy** gives a mathematical guarantee: the output of an analysis is nearly the same whether or not any single individual's data is included. Formally, for any two datasets differing in one record, the probability of any output changes by at most a factor of `e^ε`. **ε (epsilon)** is the privacy budget — smaller is more private. It's a *worst-case* guarantee that holds regardless of what auxiliary information an attacker has, which is what makes it stronger than anonymization.

That last point is the important one: **anonymization by removing identifiers does not work.** Netflix Prize data was de-anonymized against public IMDB reviews; supposedly anonymous location traces are re-identifiable from a handful of points. DP exists because ad-hoc anonymization repeatedly fails.

**DP-SGD** is the ML version: clip each *per-example* gradient to a fixed norm (bounding any one example's influence), add calibrated Gaussian noise, and track the cumulative privacy budget across steps with a privacy accountant. The costs are real and worth stating: accuracy drops, training is slower (per-example gradient clipping is expensive), and **the accuracy cost falls disproportionately on underrepresented groups** — the tail is exactly what noise erases. So there's a genuine privacy/fairness tension.

**When you'd use it:** regulatory requirement, training on sensitive data, publishing aggregate statistics, or releasing a model trained on user data externally. **When it's overkill:** most internal models on already-access-controlled data, where the right controls are access management, retention limits, and purpose limitation rather than DP.

Related: **federated learning** (train on-device, aggregate updates centrally — often combined with DP and secure aggregation, since raw gradients leak information), and **machine unlearning** (removing a user's influence post-hoc, which GDPR's right-to-erasure arguably requires and which nobody has solved efficiently).

---

### Q168. Can embeddings leak private information?

Yes, and this is underappreciated — worth raising because it applies directly to embedding systems like yours.

- **Embedding inversion** — given an embedding, reconstruct the input. For text embeddings this works alarmingly well; there's work recovering substantial portions of the original text from sentence embeddings alone. So a vector database of embedded documents is closer to a database of the documents than people assume, and should be access-controlled accordingly.
- **Membership inference** — determine whether a specific record was in the training set. Relevant under GDPR and for any model trained on sensitive data.
- **Attribute inference** — an embedding trained for one purpose encodes attributes you never intended. A merchant embedding trained on order patterns may encode neighborhood demographics; a user embedding may encode protected attributes even though they were never features. Then any downstream model using that embedding has indirect access to them, which is a fairness problem *and* a compliance problem, and it's invisible in a feature audit because the protected attribute never appears in a column.
- **Sequence models memorize.** Models trained on user event sequences can memorize and regurgitate rare sequences — the extraction attacks demonstrated on LLMs apply in principle to any sequence model with rare, distinctive training examples.

**Practical mitigations:** treat embeddings as sensitive data with the same access controls as the source; test explicitly for protected-attribute predictability from your embeddings (train a probe — if a linear classifier recovers a protected attribute at high accuracy, that's a finding you need to act on); consider adversarial removal of specific attributes if required; and apply retention policies to embedding stores, not just raw data.

---

### Q169. What does model governance actually require?

The unglamorous part that matters increasingly, especially with the EU AI Act and similar regimes:

- **Lineage and reproducibility** — every deployed model traceable to a code commit, a data snapshot, a config, and an environment. If you can't rebuild a model from scratch, you can't defend it.
- **Model cards / documentation** — intended use, out-of-scope uses, training data description, evaluation results **disaggregated by group**, known limitations. Increasingly a compliance requirement rather than a nicety.
- **Approval workflow** proportionate to risk — a ranking tweak and a credit decision model should not have the same gate.
- **Monitoring in production** — performance, drift, per-segment metrics, and alerting with a named owner.
- **A rollback path** that's been tested, not just theoretically available.
- **Human oversight** for consequential decisions, plus an appeals mechanism where individuals are affected. "A human reviews it" only counts if the human has the information and authority to actually overturn the model — rubber-stamping is not oversight.
- **Retention and deletion** policies covering training data, features, embeddings, and model artifacts.
- **Periodic revalidation** — models decay, and a model approved two years ago on data from three years ago has not been meaningfully validated.

The framing that makes this land in an interview: **governance is the thing that lets you move fast safely, not the thing that slows you down.** Good lineage means you can debug a production incident in an hour instead of a week. Good monitoring means you find out from an alert rather than from a stakeholder. Framing it as an engineering asset rather than a compliance tax is both true and the more senior position.

---

## Closing — Maintaining This Bank

**What decays and how fast:**

| Section | Half-life | Recheck trigger |
|---|---|---|
| Parts 1-9 (gradients, norms, attention, embeddings) | Years | Rarely — mostly stable theory |
| Parts 10-15 (ranking, distributed, tabular, bandits) | 2-3 years | New framework generation |
| Parts 16-17 (losses, retrieval) | 2-3 years | New retrieval architectures |
| Parts 18-20 (compression, fine-tuning, inference) | **12-18 months** | Every major model release |
| Parts 21-22 (stats, classical ML) | Decades | Basically never |
| Part 23 (code) | 1-2 years | PyTorch major versions |
| Part 26 (generative) | **6-12 months** | Constantly |
| Part 27 (fairness, privacy) | 1-2 years | Regulatory changes |

**A maintenance loop that actually works:**

1. **When you read a paper that changes your mind about something in here, edit the entry.** Don't add a new one — replace the outdated claim and note what changed. The bank's value is that it's current, not that it's complete.
2. **After every interview, add the questions you couldn't answer well.** That's the highest-signal source of gaps you'll ever get, and it's free.
3. **After every production incident, check whether Part 13 covers it.** If not, add it. Your own failures are more valuable content than anything from a paper.
4. **Every 6 months, skim Parts 18-20 and 26** and mark anything you're no longer confident is current.
5. **Keep the `↪ Your hook:` lines updated as your CV changes.** They're what convert generic knowledge into a specific answer, and they go stale fastest of all.

**The things in here most likely to be wrong in two years:** specific quantization methods, the DPO/PPO balance, agent architectures, RoPE extension recipes, and anything with a version number. The things most likely to still be true: everything in Parts 1-2 and 21-22, and every trade-off paragraph.

**One habit worth adopting:** when you learn something new, try writing the entry in this format before you're sure you understand it. The "what does it cost" paragraph is the hard one, and if you can't write it, you don't understand the thing yet — you've only understood the abstract.
