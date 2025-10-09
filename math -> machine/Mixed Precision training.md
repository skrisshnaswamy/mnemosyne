### FP32 vs BFLOAT16 (Numerical Precision)

- `FP32` (Floating Point 32) is the standard, highly precise way computers store numbers.
- `BFLOAT16` (Brain Float 16) is a "half-precision" format that uses less memory and is much faster on modern GPUs, with a tiny loss in precision that often doesn't hurt model performance.

---

### Kidan's experiments
- The Vanilla Encoder gets a massive **120% throughput increase** by switching to `BFLOAT16`.
- The old CM2 encoder couldn't take advantage of it for some reason.
	- But then by switching to [[Flash Attention]], we noticed similar drop in resource usages.