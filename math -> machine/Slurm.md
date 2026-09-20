---
aliases:
  - SLURM
  - sbatch
  - Workload Manager
  - Job Scheduler
tags:
  - infrastructure
  - hpc
  - engineering
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** The booking system for a shared cluster — you write down what you need in a script, submit it, and it runs when resources free up.
> **Metaphor:** The theatre list coordinator from [[HPC]]. You hand in a request form; they pack the timetable.
> **Where it bites:** `sbatch` to submit, `squeue` to watch, `scancel` to kill. Ask for **less** and you'll start sooner.

---
# The form you fill in

A Slurm job is an ordinary shell script with special comments at the top. Those `#SBATCH` lines are the booking form — everything below them is what actually runs.

```bash
#!/bin/bash
#SBATCH --job-name=train-7b
#SBATCH --partition=gpu           # which pool of nodes
#SBATCH --nodes=4                 # how many machines
#SBATCH --gres=gpu:8              # GPUs per machine
#SBATCH --ntasks-per-node=8       # one process per GPU
#SBATCH --cpus-per-task=8         # CPU cores per process — for dataloaders
#SBATCH --mem=0                   # 0 = all the memory on the node
#SBATCH --time=24:00:00           # walltime. HARD limit
#SBATCH --output=logs/%x-%j.out   # %x = job name, %j = job id
#SBATCH --requeue                 # let it restart if the node dies

module load cuda/12.4
srun python train.py --config cfg.yaml
```

Submit with `sbatch train.sh`. It prints a job ID and returns **immediately** — the job hasn't run, it's been *queued*.

> [!NOTE] `sbatch` vs `srun`
> - **`sbatch`** submits a script to the queue and returns. Fire and forget.
> - **`srun`** launches tasks *across your allocated nodes*. Inside a batch script it's what actually spawns your 32 processes.
>
> This trips everyone up: `srun` inside `sbatch` is not redundant. `sbatch` got you the rooms; `srun` puts a person in each one. ^sbatch-vs-srun

---
# The six commands you'll actually use

| Command | What it does |
|---|---|
| `sbatch job.sh` | Submit. Returns a job ID |
| `squeue -u $USER` | What am I running or waiting on? |
| `scancel 12345` | Kill a job |
| `sinfo` | What partitions exist, what's idle |
| `sacct -j 12345` | What happened after the fact — exit code, memory used, runtime |
| `salloc` / `srun --pty bash` | Get an **interactive** shell on a compute node |

> [!TIP] `sacct` is the one people forget
> After a job dies, `sacct -j <id> --format=JobID,State,ExitCode,MaxRSS,Elapsed` tells you whether it was `TIMEOUT`, `OUT_OF_MEMORY` or `FAILED` — and how much memory it actually used. That last number is how you stop over-requesting. ^sacct

Reading `squeue`, the **reason** column is the useful part:

| Reason | Meaning |
|---|---|
| `Resources` | Waiting for enough nodes to free up. Normal |
| `Priority` | Others are ahead of you |
| `QOSMaxJobsPerUserLimit` | You've hit a per-user cap |
| `AssocGrpBillingMinutes` | ⚠️ Your allocation has run out |
| `ReqNodeNotAvail` | You asked for something that doesn't exist, or nodes are in maintenance |

---
# Why asking for less gets you running sooner ⏱️

This is the single most practical thing to understand about schedulers.

Slurm doesn't run jobs strictly in order. It runs **backfill**: while it's assembling nodes for a big high-priority job, it looks for small jobs that can finish *entirely within the gap* before the big one is ready.

So a job asking for **1 node for 1 hour** slots into gaps constantly. A job asking for **64 nodes for 48 hours** waits for a rare alignment.

> [!SUCCESS] Core idea
> Your walltime request is not a safety margin — it's a **promise the scheduler plans around**. Requesting 48 hours "just in case" for a 5-hour job means you're only ever eligible for 48-hour-shaped gaps, and you'll sit in the queue for days. ~={blue}Ask for what you need plus a modest buffer, and checkpoint so being wrong is survivable.=~ ^ask-for-less-start-sooner

The same applies to memory and CPUs. Over-requesting delays you *and* burns your allocation faster, since most sites bill on what you **reserved**, not what you used.

---
# Job arrays — the underrated feature

Running the same script over 200 hyperparameter settings? Don't write 200 scripts and don't write a loop inside one job.

```bash
#SBATCH --array=0-199%20        # 200 jobs, at most 20 running at once
python sweep.py --idx $SLURM_ARRAY_TASK_ID
```

Each array element is an independent job. They backfill beautifully, they fail independently, and the `%20` throttle stops you monopolising the cluster and tanking your fair-share priority.

> [!TIP] This is the natural home for a hyperparameter sweep
> Most sweeps are **embarrassingly parallel** — no communication between runs. That's exactly the shape a job array is built for, and it needs none of the [[Collective Communication|collective machinery]] that multi-node training does. ^job-arrays

---
# Wiring Slurm to PyTorch

The bit that's genuinely fiddly. PyTorch's [[Distributed Training|DDP]] needs every process to know: the total world size, its own rank, and where to find rank 0. Slurm has all of this in environment variables:

```bash
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
srun python -c '
import os, torch.distributed as dist
os.environ["RANK"]       = os.environ["SLURM_PROCID"]
os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
dist.init_process_group("nccl")
'
```

| Slurm variable | Meaning |
|---|---|
| `SLURM_PROCID` | Global rank — 0 to world_size−1 |
| `SLURM_LOCALID` | Rank **within this node** — use this to pick the GPU |
| `SLURM_NTASKS` | World size |
| `SLURM_NODEID` | Which node this is |

> [!WARNING] `RANK` vs `LOCAL_RANK`
> `RANK` identifies you across the *whole job*. `LOCAL_RANK` identifies you *on your machine*, and it's the one you pass to `torch.cuda.set_device()`. Mixing them up on a multi-node job means several processes grab GPU 0 and the rest sit idle — a very confusing OOM. ^rank-vs-local-rank

---
# Surviving preemption and walltime

On a shared cluster your job **will** be interrupted. Build for it:

```bash
#SBATCH --signal=B:USR1@300     # send SIGUSR1 five minutes before the axe falls
#SBATCH --requeue
```

Trap that signal, save a checkpoint, and call `scontrol requeue $SLURM_JOB_ID`. Your job goes back in the queue and restarts from where it stopped.

> [!SUCCESS] If you remember one thing
> ~={pink}A cluster job that cannot restart from a checkpoint is a job that will eventually lose all its work.=~ At scale this isn't caution — it's arithmetic. See [[HPC#^hpc-failure-modes|the failure modes]].

---
# ⁉️
Slurm hands you the machines. What your processes then *say* to each other over the fast network is [[Collective Communication]] — and how the model is split across them is [[Distributed Training]].
