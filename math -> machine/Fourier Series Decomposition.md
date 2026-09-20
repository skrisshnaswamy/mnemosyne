---
aliases:
  - Fourier Series
  - Fourier Transform
  - FFT
  - Fourier Decomposition
tags:
  - fundamentals
  - signal-processing
  - time-series
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Any repeating pattern can be rewritten as a **sum of simple sine and cosine waves**, each with its own frequency and amplitude.
> **Metaphor:** A smoothie machine run in reverse 🍓 — feed it the blended drink, get back the recipe and the proportions.
> **Where it bites:** Seasonality in forecasting, transformer positional encodings, spectral bias (networks learn low frequencies first), and MP3/JPEG compression.

---
> [!ABSTRACT] **Fourier Series Decomposition: The Musical Score of Signals**
>
>Imagine you’re listening to a beautiful symphony. The music you hear is actually made up of many different notes played together. A Fourier Series is like a musical score that breaks down a complex sound (or any repeating pattern) into its individual musical notes.
>
>In the world of math and signals, it’s a way to take any repeating wave—like the sound of a guitar string, the rhythm of a heartbeat, or even the up-and-down pattern of a stock market—and express it as a sum of simple, smooth waves (called sines and cosines). Each of these simple waves has its own speed (frequency) and height (amplitude).
>
>**Why is this useful?** Just as a musician can understand a song by looking at its sheet music, engineers and scientists can understand complex signals by breaking them down into these simple waves. This helps in everything from compressing music files (like MP3s) to analyzing brain waves in medicine.

---

**In one sentence:** Fourier Series Decomposition is a way to break down any repeating pattern into a collection of simple, building-block waves.

Think of a complex wave like a fruit smoothie.
- The final drink looks and tastes like one blended mixture.
- However, if you look at the recipe, it is actually made of specific amounts of strawberries, bananas, and yogurt.

Fourier Series is like a magic machine that works **in reverse**. You feed it a weird, jagged, complicated wave, and it hands you a list telling you exactly which simple wave ingredients were mixed together to make it, and in what proportions.

## Why Is This Useful?

Breaking complex waves into simple parts makes them much easier to study, clean up, or use in technology. Here is where you encounter it every day:

- **Noise-Canceling Headphones:** They use this concept to identify the steady background hum of an airplane, flip it upside down, and cancel it out.
- **Digital Media:** Technologies like MP3 audio and JPEG images break down data into wave frequencies to shrink file sizes without losing quality.
- **Telecommunications:** Your Wi-Fi and phone signals rely on separating overlapping waves so information doesn't get scrambled.
---
# Why it matters in machine learning

Beyond the signal-processing uses above, this shows up in a few places that are worth recognising:

**Positional encodings.** The original transformer encodes each token's position using sines and cosines at many different frequencies. A position becomes a *pattern across frequencies* — which is exactly a Fourier basis. It works because relative offsets become simple linear relationships in that space, so the model can learn "three tokens back" as one consistent operation rather than memorising every pair.

**Spectral bias.** Neural networks learn **low frequencies first**. Train on a wiggly function and the network fits the broad shape early and the fine detail much later — which is a form of implicit [[Regularization]], and part of why early stopping works so well. It also explains why plain networks struggle to represent sharp detail, and why Fourier features are fed in explicitly to fix it.

**Seasonality.** Any forecasting problem with repeating cycles — daily, weekly, yearly — is naturally expressed as a small set of frequencies rather than a huge number of lag terms. See [[Auto-regressive lags]].

> [!TIP] The transferable idea
> ~={blue}A hard problem in one representation can be easy in another.=~ Convolution is expensive in the time domain and a plain multiplication in the frequency domain. Changing basis doesn't change the information — it changes which operations are cheap. That's the same instinct as a [[Linear Projection|linear projection]]: re-express the data so the *next* step becomes easy. ^change-of-basis

> [!NOTE] Series vs Transform
> - **Fourier *Series*** — for **periodic** signals. Gives a discrete set of frequencies.
> - **Fourier *Transform*** — for **non-periodic** signals. Gives a continuous spectrum.
> - **FFT** — the fast algorithm for computing the discrete version, $O(n \log n)$ instead of $O(n^2)$. It's the reason any of this is practical.

---
# ⁉️
Decomposing a signal into frequencies is one way to model something that repeats. The other way — predict the next value from the previous ones — is [[Auto-regressive models]].
