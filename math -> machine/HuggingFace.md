---
aliases:
  - Hugging Face
  - HF Hub
  - transformers library
  - AutoModel
  - from_pretrained
tags:
  - llm
  - engineering
  - infrastructure
  - tooling
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Two things bolted together — a **hosting platform** where every model is a git repo, and a set of **Python libraries** that give one uniform API over hundreds of different architectures.
> **Metaphor:** GitHub, but the repos contain weights instead of source — plus a standard adaptor so every one of them plugs into the same socket.
> **Where it bites:** Pinning a revision for reproducibility, the tokenizer that must match its model, and loading a 200 GB dataset on a 16 GB laptop.

---
**HuggingFace** is a company, and what people usually mean by the name is two separate things that happen to live under it.

The first is the **Hub** — a website that hosts models, datasets, and Spaces (which are small running demo apps, usually Gradio or Streamlit).

The second is a set of **Python libraries** — `transformers`, `datasets`, `tokenizers` and a few others — that download things from the Hub and run them.

Most descriptions stop at "it's a site with pretrained models you can download". That's true, and it's the less interesting half.

---
# The Hub is git. Actually git.

Every model on the Hub is a **git repository**. Not "like" a git repository. This works:

```
git clone https://huggingface.co/bert-base-uncased
```

Commits, branches, tags, history — all ordinary git.

The weights are handled by **git-LFS**, which stands for Large File Storage.
	Here's why that's needed. Git keeps every version of every file forever, and for code that's cheap, because git stores compact diffs between versions. But a 400 MB weights file has no useful diff — change it and you've stored another whole 400 MB. So LFS puts a tiny *pointer* file inside git, and keeps the real blob on a separate server. `git clone` fetches the pointers; LFS then fetches the actual files.

## What `from_pretrained` is actually doing

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("bert-base-uncased")
```

It is **not** running `git clone`. It fetches individual files over HTTP from the same storage the repo sits on. Four steps:

1. Download `config.json` — the architecture blueprint. How many layers, how wide, how many heads.
2. Build an empty network in memory from that blueprint.
3. Download `model.safetensors` — the actual weights.
4. Pour those numbers into the empty network.

Everything downloaded lands in `~/.cache/huggingface`. Run it a second time and nothing is fetched.

---
# Why git matters: pinning

A plain download site gives you "the latest weights". Git gives you something better, and it comes down to one distinction you already know from code.

**A branch is live. A commit is not.**

`main` moves every time the maintainer pushes. A commit hash cannot move, because the hash *is* the content — change one byte and it's a different hash.

```python
AutoModel.from_pretrained("bert-base-uncased")                      # tracks main. live.
AutoModel.from_pretrained("bert-base-uncased", revision="a1b2c3d")  # frozen forever.
```

That second line is called **pinning a revision**.

> [!NOTE] But isn't a downloaded copy frozen too?
> Yes — and this is the right objection. If you download the weights file and keep it, it can't change either.
> The difference is **who the freeze works for**. Your downloaded copy is frozen *for you*; it's a file on your disk, and if someone asks what you ran, the best you can say is "bert, I downloaded it in March". A commit hash is frozen *for everybody* — `a1b2c3d` is the same bytes on your machine, mine, and in CI. ^pinning-is-for-others

## The frame worth keeping

Put these two lines side by side:

```
model = AutoModel.from_pretrained("bert-base-uncased")   # no version
torch                                                    # no version
```

You would never ship that second line in a `requirements.txt`. You'd pin `torch==2.4.1` — see [[Python Environments]].

But almost everyone writes the first line unpinned, every day.

**A model checkpoint is a dependency.** It just doesn't live in `requirements.txt`, so nobody treats it like one.

And note *which* kind of reproducibility this affects. There are two:

| What you want to reproduce | What you have to send |
|---|---|
| The **inference** — same input, same output | the fine-tuned artefact (an MLflow link, a saved model) |
| The **training** — run it again, get 91% again | the code, the data, **and the base checkpoint's revision** |

The second row is where an unpinned `from_pretrained` quietly breaks you, six months later, for better or for worse.

---
# Why "Auto"? 🤖

You hand `AutoModel` a string. Something has to decide "this one is a BERT".

That something is `config.json`, which is downloaded *before* any weights:

```json
{
  "model_type": "bert",
  "architectures": ["BertForMaskedLM"],
  "num_hidden_layers": 12,
  "hidden_size": 768
}
```

`AutoModel` reads `model_type`, looks `"bert"` up in a registry of architectures, and builds a `BertModel`.

So it isn't magic and it isn't a giant if-statement. ~={blue}The repo tells the library what it is.=~

And here's the payoff:

```python
AutoModel.from_pretrained("bert-base-uncased")
AutoModel.from_pretrained("roberta-base")
AutoModel.from_pretrained("meta-llama/Llama-3.1-8B")
```

Same line. Completely different architectures. Your code doesn't change.

That uniform interface is what made the Hub worth having. Without it, half a million models would mean half a million loading scripts.

---
# The tokenizer and the weights are one unit 🪤

The model expects integers. Those integers are indices into a **vocabulary** — a dictionary of subwords. For `bert-base-uncased` it holds exactly 30,522 entries. (On what subwords are and why they beat whole words: [[Tokenization]] and [[Neural Machine Translation of Rare Words with Subword Units]].)

There is no universal vocabulary. Each model ships its own:

| Model | Vocabulary size | Algorithm |
|---|---|---|
| `bert-base-uncased` | 30,522 | WordPiece |
| `roberta-base` | 50,265 | byte-level BPE |

Different sizes, different algorithms, and **different words sitting at every index**.

Each vocabulary was frozen during training. The weights learned what integer `2003` means *for that specific vocabulary*. So the vocabulary ships in the repo beside the weights, and you load it with the same string:

```python
AutoTokenizer.from_pretrained("bert-base-uncased")   # same string as the model
```

> [!WARNING] Mix them and nothing crashes
> Load bert's weights with roberta's tokenizer. Both are real. Both load without complaint. Roberta encodes `"the cat sat"` as, say, `[100, 123, 145]` — and bert reads those indices as entirely different subwords. You get fluent, confident, wrong output.
>
> There's a nastier wrinkle. Bert's vocabulary stops at 30,521, roberta's goes to 50,264. An id **above** 30,521 throws an index error and you're lucky. An id **below** it is silently wrong — and common words have small ids. ~={red}The more ordinary your text, the more likely it sails straight through and lies to you.=~ ^tokenizer-mismatch-is-silent

---
# A 200 GB dataset on a 16 GB laptop 💾

In one epoch every sample is used exactly once. So there's no frequency distribution and nothing worth caching in the usual sense — an LRU cache is useless when nothing is ever read twice.

The answer is to never load the whole thing. `datasets` stores data on disk in **Apache Arrow** format and **memory-maps** it.
	Memory-mapping means the file is mapped into the process's address space, and the operating system pages in only the bytes you actually touch. So the caching instinct is right after all — it just happens in the kernel's page cache rather than in the library.

Three consequences:

- `load_dataset()` on a 200 GB corpus returns almost immediately and uses almost no RAM.
- RAM usage stays roughly **flat regardless of dataset size**.
- You still get random access — `ds[5_000_000]` is fast, not a sequential scan.

And if the dataset is too big even to download, `load_dataset(..., streaming=True)` reads shards over HTTP as you iterate, storing nothing.

---
> [!SUCCESS] If you remember one thing
> HuggingFace is a hosting platform **and** a uniform API, and the second is what made the first useful. ~={pink}The repo carries its own config and its own vocabulary, which is why one line of code loads any architecture — and why the tokenizer must always come from the same string as the weights.=~

---
# ⁉️
`from_pretrained` downloads a file from a stranger on the internet and loads it into your Python process. So why did HuggingFace bother inventing a *new* file format — **safetensors** — when `torch.save` already existed?

→ [[Tokenization]] · [[LLM Engineering]] · [[Python Environments]]
