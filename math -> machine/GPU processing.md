---
aliases:
  - GPU
  - GPUs
  - GPU Bottlenecks
  - Why training is slow
tags:
  - infrastructure
  - gpu
  - performance
  - training
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A GPU is judged by **memory bandwidth** and **tensor-core throughput at your dtype** — not core count. And slow training is always one of three things: **input-bound**, **compute-bound**, or **communication-bound**.
> **Metaphor:** A kitchen with thousands of cooks and one narrow pantry door. The door — not the head-count — sets the pace.
> **Where it bites:** `nvidia-smi` shows 8 devices and one is busy. Utilisation sawtooths 90% → 10%. Adding GPUs stops helping. Or you went to a bigger batch and quietly forgot the learning rate.

---
## Part 1 — What a GPU actually is

- Forget **"cores"** as a unit of parallelism you control. Here's the real hierarchy, from smallest to largest:
- **CUDA core** — a tiny arithmetic unit. One multiply-add at a time. An A100 has 6,912 of them. You never address these; they're plumbing.
- **SM _(Streaming Multiprocessor)_** — a cluster of CUDA cores plus tensor cores, registers, and a scratchpad of shared memory. The A100 has 108 SMs. The scheduler hands work to SMs automatically.
- **Tensor core** — specialized hardware that does a small matrix multiply in a single instruction instead of many scalar operations. This is where nearly all your training FLOPs happen. It only activates for the right dtypes — FP16 or BF16, not FP32. This matters enormously and I'll come back to it.
- **VRAM** — the GPU's own memory. Your model weights, gradients, optimizer state, and activations all live here. Run out and you get CUDA OOM.
- **Memory bandwidth** — how fast data moves between VRAM and the SMs. This is the most underrated number on the spec sheet, and often the actual limit.
- The **device** — the whole card. This is _**cuda:0**_. This is the smallest unit PyTorch, Lightning, or Ray can assign work to.

So when you saw eight entries in `nvidia-smi`, you were looking at eight devices — eight whole A100 cards in one `p4d.24xlarge` server.

---

## Part 2 — Your three cards


```
┌────────────────────┬───────────────┬───────────────┬─────────────────┐
│                    │   T4 (g4dn)   │   A10G (g5)   │ A100 40GB (p4d) │
├────────────────────┼───────────────┼───────────────┼─────────────────┤
│ VRAM               │ 16 GB GDDR6   │ 24 GB GDDR6   │ 40 GB HBM2      │
├────────────────────┼───────────────┼───────────────┼─────────────────┤
│ Memory bandwidth   │ 320 GB/s      │ 600 GB/s      │ 1,555 GB/s      │
├────────────────────┼───────────────┼───────────────┼─────────────────┤
│ CUDA cores         │ 2,560         │ 9,216         │ 6,912           │
├────────────────────┼───────────────┼───────────────┼─────────────────┤
│ Tensor cores       │ 320 (2nd gen) │ 288 (3rd gen) │ 432 (3rd gen)   │
├────────────────────┼───────────────┼───────────────┼─────────────────┤
│ FP16 tensor TFLOPS │ ~65           │ ~125          │ ~312            │
├────────────────────┼───────────────┼───────────────┼─────────────────┤
│ GPU-to-GPU link    │ none (PCIe)   │ none (PCIe)   │ NVLink 600 GB/s │
├────────────────────┼───────────────┼───────────────┼─────────────────┤
│ Per AWS instance   │ 4 (g4dn.12xl) │ 4 (g5.12xl)   │ 8 (p4d.24xl)    │
└────────────────────┴───────────────┴───────────────┴─────────────────┘
```

**Look at the CUDA core row**. The A10G has 9,216 — more than the A100's 6,912. And the A100 is roughly 2.5× faster for training.

That single comparison should permanently kill "core count" as a mental model. The A100 wins on the things that actually matter:

- Memory bandwidth: 1,555 vs 600 GB/s. Deep learning moves enormous amounts of data — weights in, activations out, gradients back. HBM sits on the same package as the chip. GDDR6 sits beside it on the board. That gap is most of the difference.
- Tensor core throughput: 312 vs 125 TFLOPS.
- NVLink. On p4d, all 8 GPUs are wired together through NVSwitch at ~600 GB/s, all-to-all. On g4dn and g5, GPUs talk over PCIe at ~32 GB/s — roughly 20× slower.

That last row is why your 4× speedup was near-linear. When four GPUs average their gradients every step, they're moving hundreds of megabytes between each other. On NVLink that's nearly free. On PCIe it would have eaten a large chunk of your gains. You got clean scaling partly because you were on the right hardware.

---

## Part 3 — Why training is slow: three bottlenecks

This is the framework your team was rediscovering through trial and error. There are only three places time goes.

### Bottleneck 1: Input-bound (starving)

The GPU finishes its batch and waits for the next one. CPU can't decode, transform, and transfer fast enough.

Signature: GPU utilization low and sawtoothing — 90%, 10%, 90%, 10%.
Fixes: more dataloader workers, prefetching, pin_memory=True, caching preprocessed data, faster storage.

This was your team's first discovery — "first it was the loader itself."

### Bottleneck 2: Compute-bound (the GPU is genuinely busy)

The GPU is saturated and the math simply takes that long.

Signature: utilization pinned at 95–100%.
Fixes: mixed precision — the big one — bigger batches, better kernels, or more GPUs.

Mixed precision deserves a note. In plain FP32, tensor cores sit idle. Switching to FP16/BF16 with automatic mixed precision routes matmuls through them, typically 2–3× faster, and halves activation memory. Lightning enables this with one flag (precision="bf16-mixed"). When you said "it was sped up with Lightning itself" — this is very likely what happened. BF16 is the safer choice on A100: same exponent range as FP32, so no loss-scaling drama.

### Bottleneck 3: Communication-bound (multi-GPU only)

GPUs spend their time exchanging gradients instead of computing.

Signature: adding GPUs stops helping. 4 GPUs give 3.9×, 8 give 4.5×.
Fixes: NVLink, gradient bucketing, larger batches per step.

Your team's actual path

▎ loader was slow → fixed the loader → then it/s was slow → fixed with Lightning → then GPU utilization was bad → adopted Ray Train → utilization ~80% sustained

That's textbook iterative profiling: fix one bottleneck and the next one becomes visible. You were input-bound, then you were compute-bound-with-idle-tensor-cores, then you were leaving seven of eight GPUs idle.

"80% sustained throughout the epoch" is the key phrase. Not the peak — the sustained number, with no sawtooth. That means each of the four GPUs had its own properly-fed data pipeline. That's the real win, and it's worth saying exactly that way in an interview.

How to see this yourself

```
nvidia-smi -L                  # one line per GPU + UUID — quickest inventory
watch -n 1 nvidia-smi          # live; watch the GPU-Util column
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv -l 1
```

The `Bus-Id` column you don't remember seeing is just PCIe addressing — different driver versions and terminal widths lay the table out differently. The columns that matter are GPU-Util % and Memory-Usage.

---

## Part 4 — What Ray Train actually did

Two different Ray systems appear in your setup, and the distinction is the whole answer to the MIG question.

`ScalingConfig(num_workers=4, use_gpu=True)` — this is Ray Train. It launches 4 worker processes, and use_gpu=True gives each worker one full GPU by default. So: 4 workers, 4 whole A100s, one per worker. Ray Train then wires them together with PyTorch DDP under the hood.

`@ray.remote(num_gpus=0.25)` decorators — this is Ray Core, a different mechanism. Fractional GPU requests let several actors share one physical GPU. You used these on your modularized service methods, not on the training workers.

So no MIG anywhere. You didn't need it. Ray Train took 4 whole GPUs; the fractional decorators packed your auxiliary actors onto the remaining ones. That's a clean design, and you don't need to check anything.

What DDP does each step:
1. Every worker holds a full copy of the model.
2. Each gets a different slice of the batch — 4 workers × batch 256 = global batch 1,024.
3. Each computes gradients on its own slice.
4. Gradients are averaged across all 4 via all-reduce over NVLink.
5. All 4 apply the identical averaged gradient, so the copies stay in lockstep.

The clever part: DDP buckets gradients and starts all-reducing the early layers' gradients while the backward pass is still computing later ones. Communication hides behind computation instead of stalling after it.

---

## Part 5 — The learning rate tune-up

Here's what you half-remember doing, and why it was necessary.

The problem

> 4 workers → global batch is 4× larger → you take 4× fewer optimizer steps per epoch.

steps_per_epoch = dataset_size / global_batch_size

Same data, a quarter of the updates. If you change nothing else, the model learns less per epoch and your metrics get worse — which looks like "distributed training broke my model" but is really just a step-count problem.

### Fix 1: the linear scaling rule

From Goyal et al. 2017 (Accurate, Large Minibatch SGD): when you multiply batch size by N, multiply the learning rate by N.

> batch 256, lr 0.001   →   batch 1024, lr 0.004

The intuition: each step now averages 4× more examples, so the gradient is a lower-variance estimate — you can trust it 4× further. Fewer steps, each proportionally bigger, gets you to roughly the same place.

Caveat worth knowing: linear scaling is derived for SGD. For Adam/AdamW — likely what you used — many practitioners find √N scaling works better, because Adam already normalizes by gradient magnitude. So 4× batch → 2× LR, not 4×. If you tuned by hand and landed somewhere between, that's why.

### Fix 2: warmup (this is the non-negotiable one)

You cannot jump straight to 4× LR at step 0. Early in training, gradients are large and inconsistent, and a big LR diverges immediately.

Gradual warmup: start at the original LR and ramp linearly to the scaled LR over the first few hundred steps or ~5 epochs, then hand off to your normal schedule.

Goyal et al.'s central finding was that large-batch training matches small-batch training only with warmup. Without it, it just fails.

### Fix 3: recompute your scheduler's step count ← probably what bit you

This is the one that silently breaks things, and I'd bet it's what "tweak the LR scheduler" actually was.

Cosine annealing, OneCycle, and linear decay are all parameterized by total steps:

`CosineAnnealingLR(optimizer, T_max=total_steps)`

Your total_steps just dropped by 4×. If you don't recompute it, the schedule thinks it has 4× more steps than it will ever get — so it decays a quarter of the way down and training ends with the LR still high, mid-schedule. The model underperforms and nothing errors out.

And if you're stepping a scheduler per-batch under DDP, each worker sees only its own shard — so len(dataloader) per worker is already divided by 4. Getting this wrong in either direction is a very common bug.

#### How to check you got it right

Plot loss versus epoch — not versus step — for the 1-GPU run and the 4-GPU run on the same axes.

- Curves overlay → LR scaling is correct. You bought 4× wall-clock for free.
- 4-GPU converges worse per epoch → LR too low, or scheduler step count wrong.
- 4-GPU diverges early → LR too high, or warmup missing/too short.

Two DDP correctness details while you're here: your loss must be mean-reduced consistently (sum-reduction silently gives you 4× the gradient), and your DistributedSampler needs set_epoch(epoch) each epoch or every worker replays the identical shard forever.

---

The version of this story to tell in an interview

▎ Training was on a p4d.24xlarge — 8 A100s, NVLink. We profiled iteratively: fixed the data loader, then got a big win from mixed precision via Lightning, then realized we were using one of eight GPUs. We moved to Ray Train with 4 workers, one full A100 each, DDP underneath. Sustained GPU utilization went to ~80% across the epoch and training time dropped 75%. The global batch went 4×, so we scaled the learning rate, added warmup, and recomputed the scheduler's total step count — then verified loss-per-epoch matched the single-GPU baseline.

That's a systems answer with a diagnosis, a mechanism, and a correctness check. It's considerably stronger than any GPU count.

Our fleet in practice was A100 (40 GB) or A10G (24 GB) or T4 (16 GB) or V100 (16/32 GB), depending on what capacity existed that month. Four architectures, three memory tiers, wildly different interconnects.

That is what Ray Train bought us. Not raw speed — **portability under scarcity**. `ScalingConfig(num_workers=N, use_gpu=True)` is the same line of code whether it lands on 4 T4s or 4 A100s. **Hand-rolled DDP with hardcoded device lists and `torchrun` launch scripts would have needed rework every time capacity moved.** This is a much stronger and more distinctive interview answer than "we went 4× faster."

---

### AWS GPU Instances

Here's the decoder for AWS GPU instances:

```
┌───────────┬───────────────┬──────────────┬──────────────────────┐
│  Family   │      GPU      │     Arch     │        Notes         │
├───────────┼───────────────┼──────────────┼──────────────────────┤
│ p3 / p3dn │ V100 16/32 GB │ Volta (2017) │ NVLink; FP16 only    │
├───────────┼───────────────┼──────────────┼──────────────────────┤
│ p4d       │ A100 40 GB    │ Ampere       │ NVSwitch 600 GB/s    │
├───────────┼───────────────┼──────────────┼──────────────────────┤
│ p4de      │ A100 80 GB    │ Ampere       │ very limited regions │
├───────────┼───────────────┼──────────────┼──────────────────────┤
│ p5        │ H100 80 GB    │ Hopper       │ this is the H100 box │
├───────────┼───────────────┼──────────────┼──────────────────────┤
│ g4dn      │ T4 16 GB      │ Turing       │ PCIe only            │
├───────────┼───────────────┼──────────────┼──────────────────────┤
│ g5        │ A10G 24 GB    │ Ampere       │ PCIe only            │
└───────────┴───────────────┴──────────────┴──────────────────────┘
```

`p4de` is concentrated in `us-east-1`/`us-west-2` and effectively absent from `ap-southeast-1`. 

### A portability gotcha you should know about

`BF16` requires `Ampere` (compute capability 8.0+). So:

```
┌──────────────────────┬───────┬─────────────────────────┐
│                      │ BF16? │          FP16?          │
├──────────────────────┼───────┼─────────────────────────┤
│ A100, A10G           │ ✅    │ ✅                      │
├──────────────────────┼───────┼─────────────────────────┤
│ T4 (7.5), V100 (7.0) │ ❌    │ ✅ (needs loss scaling) │
└──────────────────────┴───────┴─────────────────────────┘
```

If precision="bf16-mixed" was set and a job landed on `g4dn` or `p3`, it either errored or silently fell back. Given your fleet churn, someone on your team almost certainly hit this. Worth knowing — it's exactly the kind of question a systems interviewer asks to see whether you've actually run on mixed hardware.

### The embedding memory math explains everything

Here's why merchants and passengers were different problems. Embedding table cost, at d=128, FP32:

```
params  = num_ids × d × 4 bytes
Adam    = params × 3   (weights + m + v)
```

```
┌────────────┬────────────┬────────┬───────────┬──────────────────┐
│  Workload  │ Unique IDs │ Table  │ With Adam │ Fits 40 GB A100? │
├────────────┼────────────┼────────┼───────────┼──────────────────┤
│ Merchants  │ 1M         │ 0.5 GB │ ~1.5 GB   │ Easily           │
├────────────┼────────────┼────────┼───────────┼──────────────────┤
│ Passengers │ 400M       │ 205 GB │ ~615 GB   │ Not remotely     │
└────────────┴────────────┴────────┴───────────┴──────────────────┘
```

---
> [!SUCCESS] If you remember one thing
> The **A10G has more CUDA cores than the A100, and the A100 is ~2.5× faster.** That one comparison kills "core count" as a mental model for good. ~={pink}Deep learning is a data-movement problem that happens to involve arithmetic=~ — so fix bottlenecks in order (loader → precision → communication), because fixing one is what makes the next one visible.

---
# ⁉️
This was the original walkthrough. The superset — AWS instance families, Graviton, storage, interconnect, the full memory math — is [[ML Infrastructure]].
The same bandwidth-not-FLOPs story, told from the *inference* side → [[Prefill and Decode]].
And once one node isn't enough → [[Distributed Training]] and [[Collective Communication]].
