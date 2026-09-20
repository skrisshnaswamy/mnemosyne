---
aliases:
  - Seq2Seq
  - Sequence to Sequence
tags:
  - fundamentals
  - transformers
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A **task shape** — sequence in, sequence out. It says nothing about *how* the output is produced.
> **Metaphor:** "Translate this paragraph." That's the job description. Whether you write it word-by-word or draft-and-revise is a separate decision.
> **Where it bites:** The seq2seq / [[Auto-regressive models|autoregressive]] distinction, and encoder-decoder vs decoder-only.

---
So the term **sequence-to-sequence (seq2seq)** simply means that a model **takes a sequence as input and produces another sequence as output**.  
In other words, it is a **mapping from one ordered set of tokens to another ordered set of tokens**.

The defining property of seq2seq is **~={red}not=~ _how_ the output is generated**, but **what the task is**:

- Input: a sequence (text, audio frames, tokens, etc.)
- Output: a sequence (text, labels, tokens, etc.)
- The output sequence is **conditioned on the input sequence as a whole**

In that sense, _seq2seq is more about the problem formulation than the generation strategy_.

---

### Seq2Seq as a task, not a generation rule

Unlike **auto-regression**, seq2seq does **not** imply:

- Sequential decoding
- Token-by-token generation
- Causality
- Any specific architecture

A seq2seq model **may** generate outputs:
- Autoregressively (one token at a time)
- Non-autoregressively (in parallel)
- Or even in multiple refinement passes

---

### Architectural implications

Historically, seq2seq models are most commonly implemented using an **encoder–decoder architecture**:

- The **encoder** reads the entire input sequence and produces a contextual representation
- The **decoder** generates the output sequence conditioned on:
    - The encoder’s representations
    - Previously generated output tokens (if autoregressive)

This separation naturally supports tasks where:

- Input and output lengths differ
    
- Input and output are in different “spaces” (e.g., languages, modalities)
    

---

### Relation to Transformers

Transformer-based models that include **both an encoder and a decoder** are naturally suited for seq2seq tasks.

Key points:

- The **encoder** uses _bidirectional self-attention_ (can see the full input)
    
- The **decoder** typically uses:
    
    - **Causal self-attention** (cannot see future output tokens)
        
    - **Cross-attention** to attend over encoder outputs
        

However, it is important to note:

> **Seq2seq does not require a Transformer, nor does it require attention.**

The Transformer is simply the most common modern instantiation.

---

### Seq2Seq vs Decoder-Only Models

- **Decoder-only models** can _simulate_ seq2seq behavior by concatenating input and output into a single sequence and applying causal masking.
    
- However, conceptually:
    
    - Decoder-only models are **language models**
        
    - Seq2seq models are **conditional sequence transducers**
        

So while a decoder-only model can perform seq2seq tasks, it is not _architecturally_ seq2seq in the classical sense.

---

### Ultra-compact takeaway

If you want a single mental rule when reading papers:

> **Seq2seq means “sequence in → sequence out,” regardless of how the output is generated.**

And the follow-up questions you should always ask are:

- Is the decoding **autoregressive or parallel**?
    
- Is this **encoder-decoder or decoder-only**?
    
- How is the input sequence **conditioned on**?

---
# Seq2Seq vs Autoregressive — the contrast box

These two get used interchangeably and they are **not** the same kind of claim. This is the distinction worth having sharp:

| | **Seq2Seq** | **[[Auto-regressive models\|Autoregressive]]** |
|---|---|---|
| It describes | the **task** | the **generation method** |
| Answers | *what* am I mapping? | *how* do I produce output? |
| Sequence in? | ✅ required | not necessarily |
| Sequence out? | ✅ required | ✅ |
| One token at a time? | **not required** | ✅ by definition |
| Example that is only this | non-autoregressive translation | unconditional text generation |

> [!SUCCESS] The one-line rule
> ~={blue}Seq2seq is a **problem statement**. Autoregression is a **solution strategy**.=~ Most seq2seq models happen to be autoregressive, which is why the two get conflated — but a diffusion-based translator is seq2seq and **not** autoregressive, and a language model completing a prompt is autoregressive without being seq2seq in the classical sense. ^seq2seq-vs-ar

---
# The bottleneck problem — why cross-attention exists

Worth knowing, because it's the historical reason attention was invented in the first place.

The original (2014) encoder-decoder squeezed the **entire** input sequence into one fixed-size vector, and the decoder generated everything from that. It worked for short sentences and fell apart on long ones — you cannot compress a 50-word sentence into 512 numbers without losing something. 🍾

Bahdanau's fix (2015): let the decoder look back at **all** the encoder's hidden states, and learn which to focus on at each output step.

That was **attention**, and it was invented specifically to remove this bottleneck — two years before [[Attention Is All You Need|"Attention Is All You Need"]] proposed dropping the recurrence and keeping only the attention.

In the transformer this became **cross-attention**: the decoder supplies $Q$, the encoder supplies $K$ and $V$. See [[Query, Key, and Value (QKV)#Self-attention vs cross-attention|self vs cross attention]] — the formula is identical, only the source of the three inputs changes.

---
# Encoder-decoder vs decoder-only, in practice

| | **Encoder-decoder** (T5, BART) | **Decoder-only** (GPT, LLaMA) |
|---|---|---|
| Input processing | bidirectional — sees the whole input at once | causal — even over the prompt |
| Separation | explicit input/output boundary | one flat concatenated sequence |
| Strong at | translation, summarisation, fixed transductions | general-purpose, open-ended |
| Parameters | two stacks | one, so simpler to scale |

> [!TIP] Why decoder-only won anyway
> Encoder-decoder is arguably the *better fit* for seq2seq tasks — the input genuinely deserves bidirectional treatment. Decoder-only models won because **one stack scales more simply**, they train on any text without needing input/output pairs, and [[In Context Learning|in-context learning]] lets one model handle every task without a task-specific architecture.
>
> A good example of the more elegant design losing to the more scalable one. 📈

---
# ⁉️
The masking that makes the decoder half work is [[Causal Attention]]; the mechanism it uses to look back at the encoder is [[Query, Key, and Value (QKV)]].
