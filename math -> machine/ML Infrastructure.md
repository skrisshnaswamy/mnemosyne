---
aliases:
  - Compute Infrastructure
  - GPU Infrastructure
  - AWS ML Infrastructure
  - Training Infrastructure
tags:
  - infrastructure
  - gpu
  - aws
  - distributed-training
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** The whole stack under a training run — what a GPU really is, which spec-sheet numbers matter (**bandwidth**, **tensor throughput at your dtype**, **interconnect**), how AWS packages it, how work is split across devices, and how to tell *why* a run is slow.
> **Metaphor:** A car brochure. Core count is the horsepower figure printed in big type — but the gearbox and the road (memory bandwidth, NVLink) decide your lap time.
> **Where it bites:** `bf16-mixed` landing on a T4 or V100 and erroring or silently falling back. One GPU busy out of eight. 8 GPUs giving 4.5×. Picking an instance family. A 400M-row embedding table that fits nowhere.

> [!INFO] 🗺️ What's in here
> The full infrastructure picture underneath model training: what a GPU actually is, which numbers on a spec sheet matter, how AWS packages accelerators and CPUs, how work is split across devices, and how to tell *why* a training run is slow.
> [[GPU processing]] holds the original walkthrough of Parts 1–5. This note is the superset — same material plus AWS, Graviton, storage, interconnect, capacity, and the memory math.

---

# 1. The hardware hierarchy

The single most common mental-model error is treating "cores" as a unit of parallelism you control. You don't. Here is the real ladder, smallest to largest.

| Level                             | On an A100 40GB             | Can your framework address it?      |
| --------------------------------- | --------------------------- | ----------------------------------- |
| **CUDA core**                     | 6,912                       | No — plumbing                       |
| **Tensor core**                   | 432 (3rd gen)               | No — invoked via kernels/dtypes     |
| **SM** (Streaming Multiprocessor) | 108                         | No — the driver schedules onto them |
| **VRAM**                          | 40 GB HBM2                  | Indirectly (allocations)            |
| **Device**                        | **1** — this is `cuda:0`    | **Yes — smallest addressable unit** |
| **MIG instance**                  | up to 7, only if enabled    | Yes, appears as its own device      |
| **Node**                          | 8× A100 in a `p4d.24xlarge` | Yes                                 |

> [!IMPORTANT] `cuda:0` is a *device* index, not a core index
> A plain PyTorch or [[Deep Learning|Lightning]] job on one A100 **already uses all 108 SMs and all 6,912 CUDA cores**. The CUDA runtime schedules kernels across the whole die automatically. There is no core-level parallelism to opt into.
> If you have 8 GPUs and see only one busy, that is not "one core" — it is **one device out of eight**, because nothing wrapped the model in [[#15. Distributed training taxonomy|DDP]].

**The two numbers that actually predict training speed:**

1. **Memory bandwidth** — how fast weights, activations and gradients move between VRAM and the SMs. Deep learning is overwhelmingly memory-bound at the kernel level. This is the most underrated line on any spec sheet.
2. **Tensor-core throughput at your dtype** — where essentially all your FLOPs happen, *and only if you are in FP16/BF16*. In pure FP32 the tensor cores sit idle. See [[Mixed Precision training]].

> [!WARNING] Core count is a trap
> The **A10G has 9,216 CUDA cores. The A100 has 6,912.** The A100 is roughly 2.5× faster for training.
> It wins on bandwidth (1,555 vs 600 GB/s), tensor throughput (312 vs 125 TFLOPS), memory technology (HBM2 on-package vs GDDR6 on-board), and NVLink. Never rank GPUs by core count.

---

# 2. NVIDIA generations and compute capability

**Compute capability (CC)** is the version number of a GPU's feature set. It gates which dtypes and instructions exist. Memorise the boundary at **8.0**.

| Arch | Year | CC | Example GPUs | Gained |
|---|---|---|---|---|
| Pascal | 2016 | 6.x | P100, K80-era successors | FP16 storage |
| **Volta** | 2017 | **7.0** | **V100** | 1st-gen tensor cores (FP16) |
| **Turing** | 2018 | **7.5** | **T4**, RTX 20xx | 2nd-gen tensor cores, INT8/INT4 |
| **Ampere** | 2020 | **8.0** (A100) / **8.6** (A10G) | **A100, A10G**, RTX 30xx | **BF16**, **TF32**, structural sparsity, MIG |
| **Ada** | 2022 | **8.9** | **L4, L40S**, RTX 40xx | **FP8**, 4th-gen tensor cores |
| **Hopper** | 2022 | **9.0** | **H100, H200** | FP8, Transformer Engine, NVLink 900 GB/s |
| Blackwell | 2024 | 10.x | B100/B200, GB200 | FP4, 2nd-gen Transformer Engine |

> [!DANGER] The portability gotcha that will bite you
> **BF16 and TF32 require compute capability ≥ 8.0 (Ampere).**
>
> | GPU | CC | BF16 | FP16 |
> |---|---|---|---|
> | V100 | 7.0 | ❌ | ✅ (needs loss scaling) |
> | T4 | 7.5 | ❌ | ✅ (needs loss scaling) |
> | A10G | 8.6 | ✅ | ✅ |
> | A100 | 8.0 | ✅ | ✅ |
> | L4 / L40S | 8.9 | ✅ | ✅ |
> | H100 | 9.0 | ✅ | ✅ |
>
> If your config says `precision="bf16-mixed"` and the job lands on a `g4dn` (T4) or `p3` (V100) because that was the only capacity available, it errors or silently falls back. **On a heterogeneous fleet, precision must be selected at runtime from the device, not hardcoded in config.**

```python
import torch
def best_precision() -> str:
    if not torch.cuda.is_available():
        return "32-true"
    return "bf16-mixed" if torch.cuda.is_bf16_supported() else "16-mixed"
```

---

# 3. Precision formats

| Format | Bits | Exponent / Mantissa | Range | Notes |
|---|---|---|---|---|
| **FP32** | 32 | 8 / 23 | wide | Default. Tensor cores idle. |
| **TF32** | 19 stored as 32 | 8 / 10 | = FP32 | Ampere+. Automatic on matmuls. Free ~2–3×. |
| **FP16** | 16 | 5 / 10 | **narrow** | Needs **loss scaling** — gradients underflow to zero. |
| **BF16** | 16 | **8** / 7 | **= FP32** | Same exponent as FP32 → no loss scaling. Less mantissa precision. **Preferred on Ampere+.** |
| **FP8** | 8 | E4M3 / E5M2 | very narrow | Ada/Hopper. Mostly inference + LLM training. |
| **INT8** | 8 | integer | — | Quantised inference. |

> [!TIP] Why BF16 beats FP16 for training
> FP16's 5-bit exponent means small gradients underflow to zero, so you need a `GradScaler` multiplying the loss by ~$2^{16}$ and unscaling before the optimizer step. BF16 keeps FP32's 8-bit exponent — the same dynamic range — so gradients never underflow and **no scaler is needed**. You trade mantissa bits (7 vs 10), which training tolerates well.

**Mixed precision means mixed.** Weights are kept in an FP32 *master copy*; forward/backward run in 16-bit; the optimizer updates the FP32 master. You get speed and memory savings without the accumulated rounding error of pure 16-bit training. See [[Mixed Precision training]].

---

# 4. The cards that matter

| | **T4** | **A10G** | **L4** | **V100 SXM2** | **A100 40GB** | **H100 SXM5** |
|---|---|---|---|---|---|---|
| Arch / CC | Turing 7.5 | Ampere 8.6 | Ada 8.9 | Volta 7.0 | Ampere 8.0 | Hopper 9.0 |
| VRAM | 16 GB GDDR6 | 24 GB GDDR6 | 24 GB GDDR6 | 32 GB HBM2 | 40 GB HBM2 | 80 GB HBM3 |
| **Bandwidth** | 320 GB/s | 600 GB/s | 300 GB/s | 900 GB/s | **1,555 GB/s** | **3,350 GB/s** |
| CUDA cores | 2,560 | 9,216 | 7,680 | 5,120 | 6,912 | 16,896 |
| Tensor cores | 320 (2nd) | 288 (3rd) | 240 (4th) | 640 (1st) | 432 (3rd) | 528 (4th) |
| FP16/BF16 TFLOPS | ~65 | ~125 | ~121 | ~125 (FP16 only) | ~312 | ~990 |
| BF16 | ❌ | ✅ | ✅ | ❌ | ✅ | ✅ |
| GPU↔GPU | PCIe | PCIe | PCIe | NVLink 300 GB/s | **NVLink 600 GB/s** | **NVLink 900 GB/s** |
| TDP | 70 W | 300 W | 72 W | 300 W | 400 W | 700 W |
| AWS instance | `g4dn` | `g5` | `g6` | `p3` | `p4d` | `p5` |

> [!NOTE] Why NVLink decides whether multi-GPU scales
> Under [[#15. Distributed training taxonomy|DDP]], every GPU exchanges a full copy of the gradients every single step — hundreds of MB, thousands of times per epoch.
> - **NVLink/NVSwitch (p4d, p5, p3dn):** ~600–900 GB/s, all-to-all inside the node. Communication hides behind computation. Scaling is near-linear.
> - **PCIe only (g4dn, g5, g6):** ~32 GB/s and it is a *shared* bus. Communication becomes the bottleneck fast. 4 GPUs might give you 2.5×, not 3.9×.
>
> **A near-linear multi-GPU speedup is partly a statement about your interconnect, not just your code.**

---

# 5. Reading `nvidia-smi`

`nvidia-smi` has **no concept of cores**. Its top table lists **devices**, one row each.

```
| GPU  Name                    Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|   0  NVIDIA A100-SXM4-40GB   00000000:07:00.0 Off |                    0 |
|   1  NVIDIA A100-SXM4-40GB   00000000:0F:00.0 Off |                    0 |
...
|   7  NVIDIA A100-SXM4-40GB   00000000:BE:00.0 Off |                    0 |
```

Eight rows = **eight physical GPUs**. `Bus-Id` is just PCIe addressing; different driver versions and terminal widths lay the table out differently. The columns that matter are **`GPU-Util %`** and **`Memory-Usage`**.

```bash
nvidia-smi -L                    # one line per GPU + UUID — fastest inventory
watch -n 1 nvidia-smi            # live; watch GPU-Util
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
           --format=csv -l 1     # scriptable, 1s interval
nvidia-smi topo -m               # interconnect matrix: NV# = NVLink, PIX/PHB = PCIe
nvidia-smi mig -lgi              # list MIG instances, if any
```

> [!WARNING] `nvidia-smi` shows the *host's* GPUs, not your entitlement
> It reports every GPU on the machine regardless of what your job may actually touch. Quota, scheduler allocation and `CUDA_VISIBLE_DEVICES` decide what your process can use — and none of that changes what `nvidia-smi` prints.
> **"I saw 8 in `nvidia-smi`" does not mean "I had 8."** Check `torch.cuda.device_count()` for what your process can actually see.

`nvidia-smi topo -m` is the one people forget. It prints the interconnect matrix and tells you immediately whether GPU pairs talk over NVLink (`NV1`…`NV12`) or crawl over PCIe (`PIX`, `PHB`, `SYS`).

---

# 6. AWS instance naming, decoded

```
        m 7 g d . 4xlarge
        │ │ │ │      │
        │ │ │ │      └── size
        │ │ │ └───────── attributes
        │ │ └─────────── processor family
        │ └───────────── generation
        └─────────────── family
```

**Family:**

| Letter | Meaning |
|---|---|
| `t` | Burstable (credits) |
| `m` | General purpose (balanced) |
| `c` | Compute optimised (high CPU:RAM) |
| `r` | Memory optimised |
| `x`, `u` | Extra / ultra high memory |
| `i`, `im`, `is` | Storage optimised (NVMe, high IOPS) |
| `d` | Dense HDD storage |
| `z` | High **frequency** cores |
| `p` | **GPU — training class** (P for "parallel") |
| `g` | **GPU — graphics/inference class** |
| `inf` | AWS Inferentia |
| `trn` | AWS Trainium |
| `hpc` | HPC, EFA-heavy, no hyperthreading |

**Attribute suffixes** — these compose, e.g. `c7gn`, `m7gd`, `p4de`:

| Suffix | Meaning |
|---|---|
| **`g`** | **Graviton (ARM64)** — see [[#9. Graviton]] |
| `a` | AMD EPYC |
| `i` | Intel (when disambiguating) |
| `d` | Local **NVMe instance store** attached |
| `n` | Enhanced **network** bandwidth |
| `e` | **Extra** capacity — more RAM, or more GPU memory (`p4de` = 80 GB A100) |
| `q` | Qualcomm accelerator |
| `flex` | Flexible / partial-throughput |

> [!TIP] Read any instance name in three seconds
> `m7gd.4xlarge` → general purpose, gen 7, **Graviton**, with local NVMe, 16 vCPU.
> `c7gn.16xlarge` → compute optimised, gen 7, Graviton, enhanced network.
> `p4de.24xlarge` → GPU training, gen 4, **e = 80 GB** A100s.
> `g5.12xlarge` → GPU inference class, gen 5 (A10G), 48 vCPU.

Sizes go `nano → micro → small → medium → large → xlarge → 2xlarge → … → 48xlarge → metal`. **vCPU roughly doubles each step**, and `metal` is the bare-metal whole host.

---

# 7. AWS GPU instances

## P family — training class

| Instance | GPU | Count | GPU RAM | vCPU | Host RAM | Interconnect | Network |
|---|---|---|---|---|---|---|---|
| `p3.2xlarge` | V100 | 1 | 16 GB | 8 | 61 GiB | — | up to 10 Gbps |
| `p3.8xlarge` | V100 | 4 | 16 GB | 32 | 244 GiB | NVLink | 10 Gbps |
| `p3.16xlarge` | V100 | 8 | 16 GB | 64 | 488 GiB | NVLink | 25 Gbps |
| `p3dn.24xlarge` | V100 | 8 | **32 GB** | 96 | 768 GiB | NVLink | **100 Gbps EFA** |
| **`p4d.24xlarge`** | **A100** | **8** | **40 GB** | **96** | **1,152 GiB** | **NVSwitch 600 GB/s** | **400 Gbps EFA** |
| `p4de.24xlarge` | A100 | 8 | **80 GB** | 96 | 1,152 GiB | NVSwitch | 400 Gbps EFA |
| `p5.48xlarge` | H100 | 8 | 80 GB | 192 | 2,048 GiB | NVSwitch 900 GB/s | 3,200 Gbps EFA |
| `p5e` / `p5en` | H200 | 8 | 141 GB | 192 | 2,048 GiB | NVSwitch | 3,200 Gbps EFA |

> [!NOTE] `p4d` is the one to know
> **One `p4d.24xlarge` = 8× A100 40 GB + 96 vCPU + 1.1 TiB host RAM + 8 TB local NVMe.**
> That 1.1 TiB of *host* RAM is what makes CPU-offloading a giant [[#18. Memory math|embedding table]] viable — it is ~28× the memory of a single GPU.

## G family — inference / graphics class, but perfectly usable for training

| Instance | GPU | Count | GPU RAM | vCPU | Host RAM |
|---|---|---|---|---|---|
| `g4dn.xlarge` | T4 | 1 | 16 GB | 4 | 16 GiB |
| `g4dn.12xlarge` | T4 | 4 | 16 GB | 48 | 192 GiB |
| `g4dn.metal` | T4 | 8 | 16 GB | 96 | 384 GiB |
| `g5.xlarge` | A10G | 1 | 24 GB | 4 | 16 GiB |
| `g5.12xlarge` | A10G | 4 | 24 GB | 48 | 192 GiB |
| `g5.48xlarge` | A10G | 8 | 24 GB | 192 | 768 GiB |
| `g6.12xlarge` | L4 | 4 | 24 GB | 48 | 192 GiB |
| `g6e.12xlarge` | L40S | 4 | **48 GB** | 48 | 384 GiB |
| `g5g.*` | **T4G** | 1–2 | 16 GB | — | — |

> [!IMPORTANT] `g5g` is the only Graviton + NVIDIA combination
> `g5g` pairs **Graviton2 (ARM64)** with **NVIDIA T4G** GPUs. Everything else in the P and G families is x86. If you are on `g5g` your entire CUDA stack must be built for `aarch64` — a much smaller ecosystem. Rarely worth it.

**G vs P in practice:** P has NVLink and EFA and is built for multi-node training. G is PCIe-only, cheaper, far easier to get capacity for, and completely fine for single-GPU training, fine-tuning, and batch inference. **Most real training runs do not need a P instance.**

---

# 8. AWS custom silicon

| | **Inferentia** | **Trainium** |
|---|---|---|
| Instances | `inf1` (Inf1), `inf2` (Inf2) | `trn1`, `trn1n`, `trn2` |
| Purpose | Inference only | Training |
| Chips per instance | up to 12 (`inf2.48xlarge`) | 16 (`trn1.32xlarge`) |
| SDK | **AWS Neuron** | **AWS Neuron** |
| Framework support | PyTorch/TF via `torch-neuronx` | PyTorch via `torch-neuronx`, XLA |
| Interconnect | NeuronLink | NeuronLink, EFA for multi-node |

> [!CAUTION] The Neuron tax
> These are genuinely cheaper per unit of throughput, but they are **not CUDA**. You compile your model through the Neuron SDK (XLA-based), and any op without a Neuron kernel either falls back to CPU or fails to compile. Debugging is harder and the ecosystem is thinner.
> Sensible for a stable, high-volume production model. A bad choice for research iteration.

---

# 9. Graviton

> [!ABSTRACT] What Graviton is
> **AWS's own ARM64 (aarch64) server CPUs**, designed in-house by Annapurna Labs.
> It is a **CPU**, not an accelerator. There is no GPU and no CUDA on a Graviton instance (the sole exception being `g5g`, which bolts NVIDIA T4Gs onto Graviton2). Graviton competes with Intel Xeon and AMD EPYC, not with the A100.

## Generations

| Gen | Year | Core | Cores | Key advance | Instances |
|---|---|---|---|---|---|
| **Graviton1** | 2018 | Cortex-A72 | ≤16 | Proof of concept | `a1` (retired) |
| **Graviton2** | 2019 | Neoverse **N1**, 7 nm | 64 | ~40% better price-performance vs x86 | `m6g`, `c6g`, `r6g`, `t4g`, `x2gd`, `im4gn`, `is4gen`, `g5g` |
| **Graviton3** | 2022 | Neoverse **V1**, 5 nm | 64 | **DDR5** (+50% bandwidth), **SVE 256-bit**, **native BF16** → ~3× ML perf vs Gen2 | `c7g`, `m7g`, `r7g`, `c7gn` |
| **Graviton3E** | 2022 | Neoverse V1+ | 64 | +35% vector/HPC performance | `hpc7g`, `c7gn` |
| **Graviton4** | 2024 | Neoverse **V2** | **96** | 12× DDR5-5600 channels, ~30% faster than Gen3 | `r8g`, `c8g`, `m8g`, `x8g`, `i8g` |

## Why it matters for ML

Graviton will never train your model. It will do almost everything *around* it, cheaper:

- **Data preprocessing and ETL.** Spark, Trino, Kafka, Airflow workers, Polars/Arrow jobs. This is the single biggest Graviton win for an ML org.
- **The dataloader side of training.** [[#14. Diagnosing slow training|Input-bound training]] is a CPU problem. `c7g`/`m7g` give you more CPU per dollar to feed the GPU.
- **CPU inference.** Graviton3+ has **SVE with native bfloat16**, so PyTorch on aarch64 (via **Arm Compute Library** through oneDNN) gives real throughput for small models, embeddings lookups, tree ensembles, and classical ML.
- **Serving and API layers.** Anything not on the GPU hot path.

**Economics:** typically ~20% cheaper on-demand than the equivalent x86 instance, with better performance per watt, and AWS quotes ~40% better price-performance. On a large ETL fleet this is a material line item.

## The gotchas — this is the part that costs days

> [!DANGER] It is `aarch64`, not `x86_64`
> Every container image, Python wheel, compiled extension and binary must exist for ARM64.
> - Build multi-arch: `docker buildx build --platform linux/amd64,linux/arm64 …`
> - A single-arch `linux/amd64` image will **not run**, or will run under emulation at catastrophic speed.
> - Some Python packages ship no `manylinux_aarch64` wheel → pip falls back to a source build, which needs a toolchain and may simply fail.
> - Anything using x86 intrinsics (AVX2/AVX-512) needs a NEON/SVE code path. Mature libraries have one; niche ones do not.
> - **No CUDA.** If a job needs a GPU, Graviton is the wrong instance (except `g5g`).

```bash
uname -m                    # aarch64 vs x86_64
docker image inspect <img> --format '{{.Architecture}}'
python -c "import platform; print(platform.machine())"
```

> [!TIP] The natural split for an ML platform
> **Graviton (`c7g`/`m7g`/`r8g`) for data prep, feature pipelines, ETL and serving; x86 + NVIDIA (`g5`/`p4d`) for training.**
> Two container build targets, one codebase. The cost saving lands on the CPU fleet, which is usually the larger fleet by far.

---

# 10. CPU instance families worth knowing

| Family | Shape | Typical ML use |
|---|---|---|
| `c7g` / `c8g` | High CPU, low RAM ratio (2 GiB/vCPU) | Preprocessing, tokenisation, Spark executors |
| `m7g` / `m8g` | Balanced (4 GiB/vCPU) | Airflow, general workers, small services |
| `r7g` / `r8g` | Memory heavy (8 GiB/vCPU) | Polars/pandas on large frames, joins, feature stores |
| `x2gd` / `x8g` | Very memory heavy (16–32 GiB/vCPU) | Huge in-memory datasets, big embedding tables on CPU |
| `i4i` / `im4gn` | NVMe-dense, high IOPS | Shuffle-heavy Spark, local dataset caches |
| `z1d` | Highest clock speed | Single-threaded bottlenecks, licensing-bound work |
| `hpc7g` | EFA, no hyperthreading | Tightly-coupled MPI/HPC |

> [!NOTE] The RAM:vCPU ratio is the thing to internalise
> `c` = 2 GiB per vCPU, `m` = 4, `r` = 8, `x` = 16–32. When a Polars job OOMs, you usually want to move *sideways* from `c` to `r`, not *up* a size.

---

# 11. Interconnect

Ordered by bandwidth, because this ordering explains most scaling behaviour.

| Link | Bandwidth | Scope |
|---|---|---|
| **HBM ↔ SM** (on-package) | 1,555 GB/s (A100) – 3,350 GB/s (H100) | Inside one GPU |
| **NVLink / NVSwitch** | 300 (V100) / 600 (A100) / 900 (H100) GB/s | GPU ↔ GPU, **same node** |
| **PCIe Gen4 ×16** | ~32 GB/s bidirectional | GPU ↔ CPU, GPU ↔ GPU without NVLink |
| **EFA** | 100 / 400 / 3,200 Gbps (= 12.5 / 50 / 400 GB/s) | Node ↔ node |
| **Standard ENA** | 10–100 Gbps | Node ↔ node, no OS bypass |

**EFA (Elastic Fabric Adapter)** is AWS's OS-bypass network adapter. It lets NCCL and MPI write straight to the NIC without going through the kernel, which is what makes **multi-node** training viable. Without EFA, multi-node all-reduce is usually slower than not bothering.

**Cluster placement groups** pack instances into the same rack/spine for lowest latency. Required for serious multi-node work — and a common reason capacity requests fail, since AWS must find contiguous capacity.

**NCCL** (NVIDIA Collective Communications Library) is the library that actually performs `all_reduce`, `all_gather`, `broadcast`. DDP is built on it. `gloo` is the CPU/fallback backend.

```bash
export NCCL_DEBUG=INFO           # prints the topology and algorithm chosen
export NCCL_P2P_DISABLE=1        # diagnostic: force off NVLink peer-to-peer
export FI_PROVIDER=efa           # use EFA
```

---

# 12. Storage

| Option | Throughput | Latency | Use |
|---|---|---|---|
| **Instance store NVMe** (`d` suffix) | GB/s | µs | **Fastest.** Ephemeral — gone when the instance stops. Ideal scratch/cache. |
| **EBS gp3** | 125 MB/s baseline, tunable to 1,000 MB/s | ms | Root volumes, checkpoints. IOPS/throughput provisioned independently of size. |
| **EBS io2 Block Express** | up to 4,000 MB/s | sub-ms | Databases. Rarely needed for training. |
| **S3** | very high aggregate | **high per-object** | Dataset of record, checkpoints, artifacts. Terrible for random small reads. |
| **FSx for Lustre** | 100s of GB/s | low | Parallel FS that mounts an S3 bucket. The standard answer for large multi-node training datasets. |

> [!TIP] The storage pattern that fixes input-bound training
> **S3 (source of truth) → one-time preprocess into sharded columnar files → cache on local NVMe → stream with prefetch.**
> Reading many small files directly from S3 every epoch is the single most common cause of a starved GPU. Shard into large sequential files (Parquet, WebDataset tars), and cache locally so epoch 2 onward reads from NVMe.

---

# 13. Capacity and pricing

| Model | Discount | Commitment | Interruption |
|---|---|---|---|
| **On-Demand** | — | none | none |
| **Spot** | up to **90%** | none | **2-minute warning, any time** |
| **Savings Plans** | up to 72% | 1 or 3 years, $/hr | none |
| **Reserved Instances** | up to 72% | 1 or 3 years, specific type | none |
| **On-Demand Capacity Reservation (ODCR)** | none (you pay regardless) | none | **guarantees capacity exists** |
| **Capacity Blocks for ML** | — | reserve GPU cluster for 1–182 days | none |

> [!WARNING] Capacity is a real constraint, not a theoretical one
> GPU capacity in secondary regions (`ap-southeast-1`, `eu-west-2`, …) is genuinely scarce and fluctuates month to month. `p4d`/`p5` are frequently unavailable outside `us-east-1`/`us-west-2`; `p4de` barely exists outside them. Teams routinely fall back `g5` → `g4dn` → `p3` on whatever they can get.
>
> **This is an architectural constraint, not an ops annoyance.** It is the strongest argument for a device-count-agnostic training abstraction like [[#16. Ray Train vs Ray Core|Ray Train]] over hardcoded `torchrun` launch scripts: the same code has to land on 4× T4, 4× A10G, or 8× A100 without a rewrite.

**Spot for training** works only if you checkpoint frequently and can resume. For a run that fits in a few hours with checkpoints every N steps, spot is a 60–90% saving. For a 3-day run with no resume path, it is a trap.

---

# 14. Diagnosing slow training

There are only three places time goes. Identify which before changing anything.

## Bottleneck 1 — Input-bound (GPU starving)

The GPU finishes a batch and waits. CPU cannot decode, transform and transfer fast enough.

- **Signature:** GPU utilisation low and *sawtoothing* — 90%, 10%, 90%, 10%.
- **Fixes:** more `num_workers`, `pin_memory=True`, `prefetch_factor`, `persistent_workers=True`, cache preprocessed data on NVMe, larger sequential shards, do preprocessing once offline instead of per-epoch, move preprocessing to a [[#9. Graviton|Graviton]] fleet.

## Bottleneck 2 — Compute-bound (GPU genuinely busy)

- **Signature:** utilisation pinned at 95–100%.
- **Fixes:** **mixed precision first** (2–3× and usually free), then `torch.compile`, bigger batches, [[Flash Attention]] for attention-heavy models, gradient checkpointing if memory-limited, then more GPUs.

## Bottleneck 3 — Communication-bound (multi-GPU only)

- **Signature:** adding GPUs stops helping. 4 GPUs → 3.9×, 8 GPUs → 4.5×.
- **Fixes:** NVLink-equipped instances, larger per-GPU batch (fewer all-reduces per epoch), gradient accumulation, gradient compression, `NCCL_DEBUG=INFO` to check the chosen algorithm.

> [!EXAMPLE] Iterative profiling in practice
> A real sequence from the Kidan work:
> `loader slow → fixed loader → it/s still slow → fixed with Lightning (mixed precision) → GPU util still bad → adopted Ray Train → ~80% sustained`
>
> **Fix one bottleneck and the next becomes visible.** That is not thrashing; that is how profiling works. The phrase that matters is *"80% sustained across the epoch"* — not the peak, the sustained figure with no sawtooth.

## Tools

```python
from torch.profiler import profile, ProfilerActivity, schedule
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    schedule=schedule(wait=1, warmup=1, active=3),
    on_trace_ready=torch.profiler.tensorboard_trace_handler("./tb"),
    record_shapes=True, profile_memory=True, with_stack=True,
) as prof:
    for step, batch in enumerate(loader):
        train_step(batch); prof.step()
        if step > 10: break
```

```bash
nvidia-smi dmon -s pucm          # scrolling per-second utilisation/power/mem/clocks
dcgmi dmon                       # DCGM — richer, includes tensor-core activity
py-spy top --pid <pid>           # is the bottleneck actually Python on the CPU side?
```

> [!TIP] The cheapest possible diagnostic
> Run one training step on **synthetic random tensors** instead of the real dataloader.
> - Much faster → you are **input-bound**.
> - Same speed → you are **compute-bound**.
> Thirty seconds of work, removes all guessing.

---

# 15. Distributed training taxonomy

| Mode | Splits | When |
|---|---|---|
| **DataParallel (DP)** | batch, single process | **Never.** Deprecated, GIL-bound, one process drives all GPUs. |
| **DistributedDataParallel (DDP)** | batch, one process per GPU | **Default.** Model fits on one GPU. |
| **FSDP / ZeRO-3** | batch **+ params, grads, optimizer states** | Model does *not* fit on one GPU. |
| **Tensor parallel** | individual matmuls across GPUs | Very large layers. Needs NVLink. |
| **Pipeline parallel** | layers across GPUs | Very deep models; introduces bubbles. |
| **Expert parallel** | MoE experts | Mixture-of-experts. |
| **Embedding sharding** | rows/columns of the embedding table | **Recsys.** See [[#18. Memory math]]. |

## What DDP does each step

1. Every worker holds a **full replica** of the model.
2. Each receives a **different shard** of the global batch — 4 workers × 256 = **global batch 1,024**.
3. Each computes gradients on its own shard.
4. Gradients are **averaged across all workers** via ring all-reduce (NCCL).
5. All workers apply the identical averaged gradient, so replicas stay bit-identical.

The engineering cleverness is **gradient bucketing**: DDP groups gradients into buckets and begins all-reducing early-layer buckets *while the backward pass is still computing later layers*. Communication hides behind computation instead of serialising after it.

> [!BUG] Two DDP correctness traps
> - **Loss reduction must be `mean`, consistently.** With `sum` reduction you silently get N× the gradient magnitude.
> - **`DistributedSampler` needs `sampler.set_epoch(epoch)` every epoch**, or every worker replays the identical shard forever and your shuffling silently does nothing.

---

# 16. Ray Train vs Ray Core

Two different systems that both mention GPUs. Confusing them causes real misunderstandings.

```python
# --- Ray Train: distributed TRAINING ---
from ray.train import ScalingConfig
ScalingConfig(num_workers=4, use_gpu=True)
# 4 worker processes, each gets ONE FULL GPU by default.
# Ray Train wires them together with PyTorch DDP underneath.
# To request fractions or multiple GPUs per worker:
ScalingConfig(num_workers=4, use_gpu=True,
              resources_per_worker={"GPU": 1, "CPU": 8})

# --- Ray Core: general distributed actors/tasks ---
@ray.remote(num_gpus=0.25)
class FeatureActor:
    ...
# Fractional GPU request — several actors SHARE one physical GPU.
# Used for auxiliary/serving work, not for DDP training workers.
```

> [!IMPORTANT] `use_gpu=True` means one whole GPU per worker
> `ScalingConfig(num_workers=4, use_gpu=True)` on an 8-GPU node consumes **4 entire GPUs**. It has nothing to do with MIG, and nothing to do with the fractional `num_gpus=0.x` decorators elsewhere in the codebase.

**Why Ray Train over bare `torchrun`:** the worker count is a *config value*, not a launch-script topology. The same code runs on 4× T4 in a sandbox, 4× A10G on `g5`, or 8× A100 on `p4d`, and adapts to whatever capacity exists. On a [[#13. Capacity and pricing|capacity-constrained fleet]] that portability is worth more than raw speed.

---

# 17. Learning rate scaling under large batch

The consequence everyone forgets when they turn on multi-GPU:

$$\text{steps per epoch} = \frac{\text{dataset size}}{N \times \text{batch per worker}}$$

**4 workers → 4× larger global batch → 4× *fewer* optimizer steps per epoch.** Same data, a quarter of the updates. Change nothing else and the model learns less per epoch — which looks like *"distributed training broke my model"* but is really a step-count problem.

## Fix 1 — Scale the learning rate

**Linear scaling rule** (Goyal et al. 2017, *Accurate, Large Minibatch SGD*): multiply batch by $N$ → multiply LR by $N$.

$$\text{batch } 256,\ \text{lr } 0.001 \quad\longrightarrow\quad \text{batch } 1024,\ \text{lr } 0.004$$

Each step now averages $N\times$ more samples, so the gradient is a lower-variance estimate — the variance of the mean gradient falls as $1/(NB)$ — and you can trust a proportionally larger step.

> [!NOTE] Adam wants $\sqrt{N}$, not $N$
> Linear scaling is derived for SGD. **Adam/AdamW already normalise by gradient magnitude**, so the variance argument does not transfer cleanly. In practice $\sqrt{N}$ scaling is usually the better starting point: 4× batch → **2×** LR, not 4×.

## Fix 2 — Warmup (non-negotiable)

You cannot jump to the scaled LR at step 0; early gradients are large and inconsistent and it diverges. Ramp linearly from the base LR to the scaled LR over the first few hundred steps (or ~5 epochs), then hand off to the normal schedule. **Goyal et al.'s central finding was that large-batch training matches small-batch training *only* with warmup.**

## Fix 3 — Recompute the scheduler's total step count

> [!BUG] The silent one
> `CosineAnnealingLR(optimizer, T_max=total_steps)`, `OneCycleLR`, and linear decay are all parameterised by **total steps**. Your `total_steps` just dropped 4×. If you do not recompute it, the schedule believes it has 4× more steps than it will ever get, decays a quarter of the way down, and training ends **mid-schedule with the LR still high**. The model underperforms and *nothing errors out*.
>
> Compounding it: under DDP each worker's `len(dataloader)` is **already divided by N**. Getting this wrong in either direction is extremely common.

## How to verify you got it right

Plot **loss vs. epoch** (not vs. step) for the 1-GPU and N-GPU runs on the same axes:

- **Curves overlay** → scaling is correct; you bought N× wall-clock for free.
- **N-GPU worse per epoch** → LR too low, or scheduler step count wrong.
- **N-GPU diverges early** → LR too high, or warmup missing/too short.

---

# 18. Memory math

## Training memory per parameter

| Component | FP32 | Mixed precision (BF16) |
|---|---|---|
| Weights | 4 B | 2 B (+ 4 B FP32 master) |
| Gradients | 4 B | 2–4 B |
| Adam $m$ | 4 B | 4 B |
| Adam $v$ | 4 B | 4 B |
| **Total** | **≈ 16 B/param** | **≈ 16–18 B/param** |

> [!TIP] The rule of thumb worth memorising
> **Adam training costs roughly 16 bytes per parameter**, before activations.
> 1 B parameters ≈ **16 GB** of optimizer footprint alone. Mixed precision saves activation memory, not optimizer memory.

Activations are separate and scale with $\text{batch} \times \text{seq len} \times \text{hidden} \times \text{layers}$. Gradient checkpointing trades ~30% extra compute to recompute them instead of storing them.

## Embedding tables — the recsys memory problem

$$\text{table bytes} = N_{\text{ids}} \times d \times \text{bytes per element}$$

At $d=128$, FP32, with dense Adam (16 B/param):

| Workload | Unique IDs | Params | Weights | **+ grads + Adam** | Fits 40 GB A100? |
|---|---|---|---|---|---|
| Merchants | 1 M | 128 M | 0.5 GB | **~2 GB** | ✅ easily — even on a T4 |
| Drivers | ~10 M | 1.28 B | 5 GB | **~20 GB** | ⚠️ tight |
| Passengers | **400 M** | **51.2 B** | **205 GB** | **~819 GB** | ❌ not remotely |

> [!IMPORTANT] Why 400 M IDs forces CPU offload
> ~819 GB of dense optimizer state fits nowhere near a 40 GB GPU — but a `p4d.24xlarge` has **1,152 GiB of host RAM**. Keeping the table in CPU memory while its parameters stay in the autograd graph is not an optimisation; it is the only way the model trains at all.

> [!NOTE] Sparse gradients change the arithmetic entirely
> Only the rows appearing in a batch receive gradients. With `nn.Embedding(..., sparse=True)` plus `SparseAdam` (or row-wise Adagrad, which is what **TorchRec** uses), you never materialise dense gradients or dense optimizer state for all 400 M rows.
> Dense state is then $O(\text{rows touched})$, not $O(N_{\text{ids}})$ — often a 10–100× reduction. **Production recsys always does this.** The dense figures above are the naive worst case.

**Sharding strategies** (TorchRec vocabulary): *table-wise* (whole tables to different GPUs), *row-wise* (ID ranges split across GPUs), *column-wise* (embedding dimensions split), *data-parallel* (small tables replicated). Real systems mix them via a planner.

---

# 19. Containers, drivers, CUDA versions

The layer cake, bottom to top — a mismatch anywhere breaks everything:

```
NVIDIA driver  (host)          e.g. 535.x  ─ must be ≥ CUDA runtime requirement
    └─ nvidia-container-toolkit (host)     ─ exposes /dev/nvidia* into containers
        └─ CUDA runtime (in image)  12.1
            └─ cuDNN / NCCL (in image)
                └─ PyTorch build    cu121   ─ must match the CUDA runtime
```

- **Driver is backward compatible**, runtime is not: a 12.x driver runs 11.x containers, but not the reverse.
- `torch.__version__` shows the build, e.g. `2.4.0+cu121`. That `cu121` must line up.
- Run GPU containers with `docker run --gpus all …`.
- **Architecture and CUDA are independent axes.** `linux/arm64` + CUDA exists only for Grace/Jetson/`g5g`; for everything else GPU means `linux/amd64`.

```bash
nvidia-smi                        # driver + max supported CUDA (top-right)
nvcc --version                    # CUDA toolkit in the image
python -c "import torch; print(torch.__version__, torch.version.cuda, \
  torch.cuda.is_available(), torch.cuda.device_count(), \
  torch.cuda.get_device_capability())"
```

---

# 20. Cheat sheet

> [!SUMMARY] Fast lookups
> - **`cuda:0` = device 0**, the whole card. Not a core.
> - **`nvidia-smi` rows = physical GPUs**, and it shows the *host's*, not your entitlement. Use `torch.cuda.device_count()`.
> - **BF16 needs CC ≥ 8.0** → A100/A10G/L4/H100 yes; **V100/T4 no**.
> - **`p4d.24xlarge` = 8× A100 40 GB, 96 vCPU, 1.1 TiB RAM, NVSwitch, 400 Gbps EFA.**
> - **`p3` = V100. `p4d` = A100. `p5` = H100. `g4dn` = T4. `g5` = A10G. `g6` = L4.**
> - **`g` suffix in a CPU instance name = Graviton = ARM64.** `d` = NVMe, `n` = network, `e` = extra.
> - **Graviton has no GPU** (except `g5g`). It is for ETL, dataloading, CPU inference, serving.
> - **RAM:vCPU** — `c`=2, `m`=4, `r`=8, `x`=16–32 GiB.
> - **Adam ≈ 16 bytes/param.** 1 B params ≈ 16 GB before activations.
> - **N× batch → N× fewer steps.** Scale LR ($N$ for SGD, $\sqrt{N}$ for Adam), add warmup, **recompute `T_max`**.
> - **Sawtoothing GPU util = input-bound. Pinned 95–100% = compute-bound. Adding GPUs stops helping = communication-bound.**
> - **Synthetic-tensor test** distinguishes input-bound from compute-bound in 30 seconds.
> - **Near-linear multi-GPU scaling implies NVLink.** On PCIe, expect sub-linear.

---

# 21. Case study — the Kidan training stack

> [!EXAMPLE] Reconstructed 2026-09-03. The reference example for everything above.
> **Hardware:** two `p4d.24xlarge` provisioned for the org — one per team. Each is **8× A100 40 GB** with NVSwitch and 1.1 TiB host RAM. Sandbox work ran on self-provisioned `g4dn` (T4) / `g5` (A10G). Production sometimes landed on `p3` (V100) when `g5` capacity was unavailable in `ap-southeast-1`.
>
> **The misconception:** "we had 1 A100" meant *one 8-GPU node*, not one GPU. `nvidia-smi` showing 8 devices was 8 physical A100s. Plain PyTorch used `cuda:0` — **one device out of eight, seven idle** — which is exactly what DDP exists to fix.
>
> **The bottleneck chase:** loader → Lightning (mixed precision) → GPU utilisation → Ray Train → **~80% sustained across the epoch**, 75% wall-clock reduction ≈ a clean **4× on 4 workers**, consistent with near-linear NVSwitch scaling.
>
> **Ray's real value:** not speed — **portability**. `ScalingConfig(num_workers=N, use_gpu=True)` is identical code on T4, A10G or A100. On a fleet where capacity moved month to month, that mattered more than the speedup.
>
> **Unit of work:** one country × one entity per run; 8 countries × {merchant, driver, passenger}. Independent runs, shared codebase, different Hydra configs. No global model.
>
> **The model output was embeddings, not predictions.** Average-pooled from the **final encoder layer, not the ID embedding table** — so representations encode behavioural sequence rather than identity. Unknown entities map to a shared UNK bucket, so a few days of behaviour at batch-inference time yields a usable embedding **without retraining**. This makes the system **inductive rather than transductive**, which is the whole cold-start story. Consumed downstream as ranking features.
>
> **Why merchants and passengers were different problems:** 1 M merchant IDs ≈ 2 GB with Adam — fits a T4. 400 M passenger IDs ≈ 819 GB dense — fits nowhere but host RAM. See [[#18. Memory math]].

---

# 22. Open questions

- [ ] Was it **3 or 4** Ray workers? 75% implies 4; 3 implies ~67%.
- [ ] Why 4 GPUs and not all 8? (per-worker memory / batch limits / node contention / measured diminishing returns)
- [ ] Was the LR scaled **linearly or $\sqrt{N}$** when the global batch went 4×, and did metrics hold per epoch?
- [ ] Did the passenger table use **sparse gradients**, or dense with CPU offload?
- [ ] Was `precision` selected at runtime from device capability, or hardcoded? (V100/T4 fallbacks would have broken BF16.)
- [ ] How was embedding quality validated beyond downstream NDCG — any metrics **bucketed by entity frequency** to measure cold-start coverage directly?

---

> [!SUCCESS] If you remember one thing
> Never rank GPUs by core count. Rank them by **memory bandwidth, tensor-core throughput at your dtype, and GPU↔GPU link** — and memorise the line at **compute capability 8.0**, because ~={pink}BF16 lives above it and loss-scaling drama lives below it.=~ On a mixed fleet, pick precision at runtime from the device, never from config.

## Related

[[GPU processing]] · [[Mixed Precision training]] · [[Flash Attention]] · [[Backpropagation]] · [[Pytorch Autograd]] · [[Deep Learning]] · [[Recommender Systems - Evolution]] · [[Distributed Training]] · [[Embedding Tables]]

---
# ⁉️
This is the **rent-it-by-the-hour** picture. The other tradition — a shared cluster you *book time on* — is [[HPC]], and the booking system is [[Slurm]].
How a model actually gets split across all these devices → [[Distributed Training]]. What they say to each other while doing it → [[Collective Communication]].
And the same hardware seen from the serving side → [[Prefill and Decode]], [[KV Cache]].
