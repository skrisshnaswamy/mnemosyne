---
aliases:
  - venv
  - Virtual Environments
  - Local ML Setup
  - pip
tags:
  - python
  - tooling
  - environment
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A venv is a **symlink to Python plus a five-line `pyvenv.cfg`** that changes `sys.prefix` — giving each project its own private `site-packages`. Delete the folder and the uninstall is complete.
> **Metaphor:** A signpost, not a second house. `.venv` contains no Python of its own — it just points the real interpreter at a private cupboard of packages.
> **Where it bites:** `ModuleNotFoundError` right after installing (wrong interpreter, every time). "Works on my machine." A venv that breaks when you move the folder. And venv vs pyenv vs conda vs uv — they solve *different* problems.

> [!INFO] 🗺️ What's in here
> What a virtual environment actually *is* mechanically, how to create and use one, how dependency pinning works, what the alternatives (uv, conda, poetry, pyenv) each solve, and the exact steps used to build the `ml-capstone` PyTorch + MPS environment.
> Hardware and GPU context lives in [[ML Infrastructure]] and [[GPU processing]].

---

# 1. The problem it solves

`pip install torch` with no environment puts torch into your **system Python** — the one every other project and some OS tooling shares.

That breaks in three ways:

1. **Version conflicts.** Project A needs `numpy 1.x`, project B needs `numpy 2.x`. Only one can win.
2. **No clean uninstall.** `pip uninstall` leaves orphaned transitive dependencies behind. Over a year the system Python silently accumulates hundreds of packages nobody can account for.
3. **Non-reproducibility.** "Works on my machine" because your machine has an accumulated pile nobody can reconstruct.

A **virtual environment** gives each project its own private `site-packages`. Delete the folder and the uninstall is complete and exact.

---

# 2. What a venv actually is

This is the part worth understanding, because it is far less magical than it looks.

```bash
python3.11 -m venv .venv
```

`venv` is a **standard-library module**. It creates a directory that looks like this:

```
.venv/
├── bin/
│   ├── python3.11 -> /opt/homebrew/opt/python@3.11/bin/python3.11   # a SYMLINK
│   ├── python3    -> python3.11
│   ├── python     -> python3.11
│   ├── pip, pip3, pip3.11
│   └── activate, activate.fish, Activate.ps1
├── lib/
│   └── python3.11/site-packages/     # <- where installs land. starts empty.
├── include/
└── pyvenv.cfg                        # <- the entire mechanism
```

> [!IMPORTANT] The interpreter is **not** copied
> `.venv/bin/python3.11` is a symlink to the real Python. A venv contains no Python of its own. It is a *redirect*, not a duplicate — which is why creating one takes a fraction of a second.

## `pyvenv.cfg` is the whole trick

Five lines. From the actual `ml-capstone` environment:

```ini
home = /opt/homebrew/opt/python@3.11/bin
include-system-site-packages = false
version = 3.11.5
executable = /opt/homebrew/Cellar/python@3.11/3.11.5/Frameworks/Python.framework/Versions/3.11/bin/python3.11
command = /opt/homebrew/opt/python@3.11/bin/python3.11 -m venv /Users/.../ml-capstone/.venv
```

**How it works:** at startup, Python looks for a `pyvenv.cfg` next to (or one level above) its own executable. If it finds one, it sets `sys.prefix` to that directory instead of the base installation. And `sys.prefix` is what determines where `site-packages` is searched.

That's it. The entire isolation mechanism is *one config file changing one variable*.

```python
# system python
sys.prefix = /opt/homebrew/opt/python@3.11/Frameworks/Python.framework/Versions/3.11
# venv python
sys.prefix = /Users/.../ml-capstone/.venv
```

`include-system-site-packages = false` is what stops it from also seeing globally installed packages. Leave it false.

---

# 3. Creating and using one

```bash
cd /path/to/project
python3.11 -m venv .venv          # name it .venv by convention
```

Then there are **two ways to use it**, and they are equivalent:

## Option A — explicit path (no state, no surprises)

```bash
.venv/bin/python train.py
.venv/bin/pip install torch
```

Unambiguous. Works in scripts, cron jobs, Makefiles and CI, where "did I activate?" has no answer. **Prefer this in anything automated.**

## Option B — activate (convenient for interactive work)

```bash
source .venv/bin/activate          # bash/zsh
# prompt becomes (.venv) ...
python train.py                    # now means .venv/bin/python
deactivate
```

> [!NOTE] `activate` is not magic either
> It is a shell script that does roughly three things: prepend `.venv/bin` to `$PATH`, set `$VIRTUAL_ENV`, and modify `$PS1` so you can see it's on. It changes **nothing** about Python itself — `.venv/bin/python` behaves identically whether or not you activated. Activation is purely a `PATH` convenience.

## Checking which Python you're actually on

```bash
which python                       # should be .../.venv/bin/python
python -c "import sys; print(sys.prefix)"
python -c "import sys; print(sys.executable)"
pip -V                             # prints the pip AND the python it belongs to
```

---

# 4. Installing and pinning

```bash
.venv/bin/pip install --upgrade pip setuptools wheel   # do this first
.venv/bin/pip install torch torchvision numpy matplotlib polars
```

## Recording what you installed

```bash
.venv/bin/pip freeze > requirements.txt     # exact versions of EVERYTHING
```

```bash
# rebuilding the same environment elsewhere
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

> [!TIP] `pip freeze` vs a hand-written requirements file
> `pip freeze` dumps **every** installed package including transitive dependencies you never asked for — reproducible, but unreadable, and it hides which packages you actually chose.
> The cleaner discipline is two files: `requirements.in` with your *direct* dependencies (`torch`, `polars`), compiled into a fully-pinned `requirements.txt`. `pip-compile` (pip-tools) or `uv pip compile` does that.
> For a solo project `pip freeze` is fine. For anything a team touches, use the two-file pattern.

## Pinning styles

| Style | Meaning | Use when |
|---|---|---|
| `torch` | anything | never, in a project |
| `torch>=2.0` | at least | libraries you publish |
| `torch~=2.14.0` | 2.14.x, not 2.15 | usually right |
| `torch==2.14.0` | exactly | applications, reproducibility |

## Always gitignore the venv

```gitignore
.venv/
data/          # downloaded datasets too
```

The venv is ~1 GB, platform-specific, and fully reconstructible from `requirements.txt`. It has no business in git. **Commit the recipe, not the meal.**

---

# 5. The alternatives, and what each actually solves

These get conflated constantly. They solve **different** problems.

| Tool | Solves | Notes |
|---|---|---|
| **venv** | package isolation | stdlib, zero install, always available |
| **pyenv** | multiple *Python versions* | orthogonal to venv — use pyenv to get 3.11, then venv to isolate |
| **uv** | isolation + install speed + locking | Rust, from Astral (the ruff people). 10–100× faster than pip. Increasingly the default. |
| **poetry / pdm** | dependency resolution + lockfile + packaging | good for libraries; heavier workflow |
| **conda / mamba** | Python **and** non-Python binaries | handles CUDA toolkits, MKL, compilers. Heavy. miniforge is the sane variant |
| **pip-tools** | lockfiles for plain pip | `pip-compile` turns `.in` into pinned `.txt` |

> [!TIP] If starting fresh today, look at `uv`
> ```bash
> uv venv                          # create (near-instant)
> uv pip install torch             # drop-in pip replacement
> uv pip compile requirements.in -o requirements.txt
> ```
> It is a straight speed win with the same mental model. `venv` + `pip` remains the right choice for *learning*, because it's what every tutorial, error message and Stack Overflow answer assumes.

**When you genuinely need conda:** when a dependency ships non-Python binaries that pip can't provide — a specific CUDA toolkit, MKL-linked numerics, some geospatial and bioinformatics stacks. On Apple Silicon with PyTorch, pip is sufficient.

---

# 6. The `ml-capstone` setup, exactly

Machine: **MacBook Air, Apple M2, macOS 15.6.1, arm64**, Homebrew Python 3.11.5.

```bash
# 1. isolated interpreter
cd ~/projects/ml-capstone
python3.11 -m venv .venv

# 2. modern installer first — old pip picks worse wheels
.venv/bin/pip install --upgrade pip setuptools wheel

# 3. the stack
.venv/bin/pip install torch torchvision numpy matplotlib polars

# 4. freeze
.venv/bin/pip freeze > requirements.txt

# 5. keep the big stuff out of git
echo ".venv/" >> .gitignore
echo "data/"  >> .gitignore
```

Result: torch 2.14.0, torchvision 0.29.0, numpy 2.4.6, matplotlib 3.11.1, polars 1.44.1.

> [!IMPORTANT] On macOS, plain `pip install torch` already has MPS
> No index URL, no special build. The default macOS wheel ships the Metal backend.
> This is **not** true elsewhere:
> ```bash
> # NVIDIA, CUDA 12.1
> pip install torch --index-url https://download.pytorch.org/whl/cu121
> # CPU-only (smaller, e.g. for a serving container)
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```
> Installing the default wheel on a Linux GPU box and wondering why `torch.cuda.is_available()` is False is one of the most common setup failures.

## Verifying it

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
# 2.14.0 True
```

`is_built()` means *compiled with MPS support*. `is_available()` means *usable right now*. Check the second.

---

# 7. Failure modes

> [!BUG] `ModuleNotFoundError: No module named 'torch'` — immediately after installing torch
> You installed into the venv and ran the **system** Python. Check `which python`. This is the single most common venv confusion, and it is always this.

> [!BUG] Jupyter can't see your packages
> The notebook kernel points at a different interpreter. Register the venv as a kernel:
> ```bash
> .venv/bin/pip install ipykernel
> .venv/bin/python -m ipykernel install --user --name ml-capstone
> ```
> then pick `ml-capstone` in the kernel menu.

> [!BUG] `pip` vs `pip3` vs `python -m pip`
> Bare `pip` may belong to a different Python than the one you're about to run. **`python -m pip install ...` is always unambiguous** — it uses the pip belonging to that exact interpreter. Use this form when debugging.

> [!BUG] The venv breaks after a `brew upgrade python`
> The venv symlinks the base interpreter. Upgrade or relocate it and the symlink dangles. The fix is always to delete and recreate — `rm -rf .venv`, remake, `pip install -r requirements.txt`. Never worth debugging; that's the whole point of the recipe being committed.

> [!BUG] Moving or renaming the project directory
> Paths inside `pyvenv.cfg` and the console scripts are absolute. Recreate rather than repair.

---

# 8. Cheat sheet

> [!SUMMARY]
> ```bash
> python3.11 -m venv .venv               # create
> source .venv/bin/activate              # optional convenience
> .venv/bin/pip install -U pip           # always first
> .venv/bin/pip install torch            # install
> .venv/bin/pip freeze > requirements.txt
> .venv/bin/pip install -r requirements.txt   # rebuild elsewhere
> which python && python -c "import sys;print(sys.prefix)"   # which am I on?
> rm -rf .venv                           # complete uninstall
> ```
> - A venv is a **symlink + `pyvenv.cfg`**, not a copy of Python.
> - `pyvenv.cfg` changes `sys.prefix`, which changes where `site-packages` is found. That is the whole mechanism.
> - `activate` only edits `$PATH`. `.venv/bin/python` works either way.
> - Commit `requirements.txt`, gitignore `.venv/`.
> - On macOS, `pip install torch` includes MPS. On Linux+NVIDIA you need the CUDA index URL.
> - `ModuleNotFoundError` right after installing = wrong interpreter, every time.

> [!SUCCESS] If you remember one thing
> The entire isolation mechanism is **one config file changing one variable**. `activate` only edits `$PATH` — `.venv/bin/python` behaves identically whether you activated or not. ~={pink}Commit the recipe, not the meal.=~

## Related

[[ML Infrastructure]] · [[GPU processing]] · [[Mixed Precision training]] · [[Deep Learning]]

---
# ⁉️
All of this assumes a laptop where you own the disk. On a shared cluster, an environment is ~100,000 small files hammering a filesystem that 400 other people depend on — which is why HPC pushes you to containers instead → [[HPC#Software: modules and containers|modules and containers]].
And whether the `torch` you just installed can actually see an accelerator is a hardware question → [[ML Infrastructure]].
