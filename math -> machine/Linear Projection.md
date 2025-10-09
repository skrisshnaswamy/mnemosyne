A simple way to transform data by multiplying it with a matrix (a learnable set of weights). It's like converting a list of ingredients into a nutritional summary; the information is the same, but it's in a new, more useful format.

---

### Context from Kidan
The "Vanilla Encoder" uses linear projections to create the [[Query, Key, and Value (QKV)]] from the input. This is the standard, efficient way of doing it as described in the original "[Attention Is All You Need](https://arxiv.org/pdf/1706.03762)" paper.