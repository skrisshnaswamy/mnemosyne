The three roles a piece of data plays in an [[Attention]] mechanism.
Think of it like a library search:
- your **Query** is your question
- you match it against the **Keys** (book titles) to find the right book
- and then you retrieve the **Value** (the book's contents)

---

### Context from Kidan
The **Vanilla Encoder** from the tranformer paper created a unique Q, K, and V using [[Linear Projection]] layers, which is standard practice. But the CM2 Encoder of Kidan uses the same input as Q, K and V.