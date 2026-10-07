---
aliases:
  - Chunk Size
  - Text Splitting
  - Semantic Chunking
  - Late Chunking
  - Chunk Overlap
tags:
  - llm
  - retrieval
  - engineering
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A document only becomes retrievable once you **cut it up**, and the cut decides two separate things — what can be *found*, and whether what's found is *enough to answer*.
> **Metaphor:** Turning a reference book into index cards. Too big and the card is about five things at once; too small and the answer is split across three cards, two of which you didn't get.
> **Where it bites:** Most "our RAG doesn't work" is a chunking bug wearing an embedding-model costume.

---
You index the employee handbook. Someone asks:

> *"How much notice do I need to give for unpaid leave?"*

The answer is one 180-word paragraph that happens to straddle the bottom of page 12 and the top of page 13.

Retrieval returns five chunks:

```
rank 1   "Leave policy — overview"        score 0.88   ← no notice period in it
rank 2   "Annual leave entitlement"       score 0.86   ← wrong kind of leave
rank 3   "Parental leave"                 score 0.84
rank 4   "Sickness absence"               score 0.81
rank 5   "Leave policy — overview (ctd)"  score 0.80   ← first half of the answer
...
rank 9   "…must be submitted 21 days…"    score 0.71   ← the actual answer
```

The model answers from the first half and invents the number.

Nothing here is the embedding model's fault, and nothing is the [[Vector Database|vector store]]'s fault. ~={blue}Both did exactly what you asked. You just asked about the wrong pieces of text.=~

---
# The index cards

You've got a 400-page reference book and a box of blank index cards. You have to copy the book onto the cards, and later you'll only ever be handed **five cards**.

Cut a card per **paragraph** and every card is crisply about one thing — easy to file, easy to match. But the answer to a real question often takes three consecutive paragraphs, and you get five cards from the whole book, chosen independently. You might get two of the three.

Cut a card per **chapter** and the whole answer is definitely on one card. But now the card is "about" twenty things, so when you file it by topic, its topic is a *blur*. Ask for the notice period and this card doesn't look any more relevant than the four other chapter cards.

![[chunking_index_cards.png]]

So the small card is **findable but incomplete**. The big card is **complete but unfindable**. That tension doesn't go away — you can only move along it.

> [!NOTE] Chunking
> Splitting a source document into the units that get embedded, stored and retrieved. A chunk is the **atom of retrieval**: it is what gets a vector, what gets ranked, and what gets pasted into the [[Context Window]]. Parameters: **size** (tokens or words), **overlap** (how much consecutive chunks share), and **boundary rule** (where you're allowed to cut). ^chunking-def

> [!SUCCESS] Core idea
> An embedding is a **single point** for the whole chunk, so it's an *average* of everything in it. Add irrelevant text to a chunk and you don't add context — you ~={pink}drag its vector towards the middle of the space, away from every specific question.=~ Chunk size is a dilution dial, not a context dial. ^chunk-dilution

---
# The arithmetic 🧮

Take a 40-page policy document, ~20,000 words.

```
chunk size S = 256 words, overlap 25%  →  stride = 192  →  ~104 chunks
chunk size S = 1024 words, overlap 25% →  stride = 768  →  ~26 chunks
```

Now push the answer through. The paragraph you need is 180 words.

- At **S = 256**: the paragraph is 70% of a chunk. Its vector is dominated by the answer. It ranks.
- At **S = 1024**: the paragraph is 18% of a chunk. The other 82% is about parental leave and sick pay. Its vector is dominated by *those*. It doesn't rank.

That 18% is the whole problem, and it's why "just use bigger chunks so the model has more context" is exactly backwards for *retrieval*. It's right for *generation* — which is why the best systems retrieve small and expand large (below).

> [!WARNING] "Found something relevant" is not "found the answer"
> These are two different metrics and teams only measure the first. In the simulation at the bottom of this note, 512-word chunks retrieved *something* from the answer span **87%** of the time — but retrieved the **whole** span only **82%** of the time. At 2,048 words it's 72% vs 66%. ~={red}The gap is answers that got cut in half=~, and the model fills the missing half from its own priors. That's a [[Hallucination]] with a citation attached. ^partial-retrieval

---
# The strategies, ranked by what they cost you

| Strategy | The cut rule | Cost | Verdict |
|---|---|---|---|
| **Fixed-size** | Every $N$ tokens, blindly | Nothing | Baseline. Splits sentences and tables. Never ship it alone. |
| **Recursive character** | Try `\n\n`, then `\n`, then `. `, then chars — first separator that fits | Nothing | **Default.** Respects paragraphs for free. |
| **Document-structure aware** | Split on Markdown headings / HTML sections / PDF layout; prepend the heading path to each chunk | An afternoon | **The biggest cheap win.** A chunk that says `Leave policy > Unpaid leave > Notice` retrieves on words it doesn't contain. 🥇 |
| **Semantic** | Embed each sentence, cut where consecutive similarity drops | Embedding pass at ingest | Good on unstructured prose. Marginal when the doc already has headings. |
| **Late chunking** | Embed the **whole document** with a long-context encoder, *then* pool per chunk | A long-context embedder | Each chunk's vector carries document context — fixes the "it says *it*, and *it* was defined two pages ago" failure. |
| **Parent-document / small-to-big** | Embed small, but return the **parent** section to the model | Two stores | ~={blue}The answer to the whole trade-off.=~ Retrieve on precision, generate on completeness. |

**Overlap** is the cheap insurance against cutting mid-answer. In the simulation, going from 0% to 25% overlap at 256 words took whole-answer coverage from **65% to 91%** — 26 points for a one-line config change. Past ~25% you're mostly paying to store the same sentences several times.

> [!TIP] Where to start, in order
> 1. **Recursive splitter, 256–512 tokens, 15–25% overlap.** That's the sane default and it's two lines.
> 2. **Prepend the heading path** to every chunk's text before embedding.
> 3. **Small-to-big**: embed 256-token children, return the 2,000-token parent.
> 4. Only *then* consider semantic or late chunking — and measure, because on structured docs they often change nothing.
>
> And measure the right thing: **not** "was a relevant chunk retrieved", but "**could the answer have been written from what we retrieved**". 📏

Tables, code blocks and lists deserve a special mention: a fixed-size splitter cuts a table in half and both halves become nonsense — the header row lives in one chunk and the numbers in the other. Extract those as whole units before you split anything else.

Related: [[RAG]] for the pipeline this sits in, [[Embeddings]] for what a chunk's vector actually is, [[Reranking]] for recovering from a mediocre cut, and [[Lost in the Middle]] for why sending 20 chunks instead of 5 doesn't rescue you.

---

---
![[chunk_size_vs_recall.png]]
> [!TIP] Reading the chart
> There's an interior optimum and it's not where intuition puts it — 128–256 words beat both 64 and 2,048. The dashed grey line is the metric you're probably tracking; the green line is the one that decides whether the user gets a correct answer.

---
> [!SUCCESS] If you remember one thing
> The cut you make at ingest sets a ceiling on everything downstream. Measure **could the answer have been written from what we retrieved** — not *was something relevant retrieved*. ~={pink}Those two numbers diverge exactly where your users are.=~

---
# ⁉️
Chunks are one thing that competes for room in the window. On turn 40 of an agent run there are six others — tool schemas, stale observations, a plan nobody has re-read, half a file. Deciding what stays is its own discipline now.

→ [[Context Engineering]]
