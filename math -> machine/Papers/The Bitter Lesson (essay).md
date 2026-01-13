---
title: "The Bitter Lesson (essay)"
authors: ["Sutton"]
year: 2019
url: http://www.incompleteideas.net/IncIdeas/BitterLesson.html
priority: Must-Read
read_on: 2026-08-24
tags: [paper, llm, optimization, scaling]
---
## The Core Idea

Sutton looks back at 70 years of AI and finds one pattern that keeps repeating. Researchers try to hand-code what they know about a problem. It works, for a while. Then someone throws a general, dumb-looking method at the same problem with ten times more compute, and wins by a mile.

The reason is economics, not philosophy. The cost of one unit of computation has been falling exponentially for decades. But almost all research is done as if compute were fixed. If compute is fixed, then the only lever you have is your own knowledge of the domain — so you build in edges, phonemes, chess openings, whatever. That is a rational choice for a two-year project. It is a losing choice over twenty years, because the compute you were treating as a constant grew by a factor of a thousand while your hand-built knowledge grew by a factor of one.

> [!NOTE] The Bitter Lesson
> General methods that scale with computation beat methods that encode human domain knowledge, given enough time. It is "bitter" because the win comes at the expense of the elegant, human-centred approach that researchers were emotionally and professionally invested in. ^bitter-lesson

The second, sharper claim: the contents of a mind are *irredeemably complex*. Our tidy concepts — objects, space, symmetry, agents — are compressions we invented, and they are always lossy. So do not build in your discoveries. Build in the **meta-method** that can make discoveries. "We want AI agents that can discover like we can, not which contain what we have discovered."

And the practical payload: only two families of methods scale arbitrarily with compute.

> [!NOTE] Search and Learning
> **Search** = spend compute at decision time, exploring possible futures (tree search, sampling, test-time compute). **Learning** = spend compute at training time, fitting parameters to data (gradient descent, self-play). Both convert raw FLOPs into performance without a human in the loop. Everything else has a ceiling set by how much a person can hand-write. ^search-and-learning

What this unlocks, in hindsight, is the entire scaling era. [[Scaling Laws for Neural Language Models]] and [[Training Compute-Optimal Large Language Models (Chinchilla)]] are the bitter lesson written as equations rather than as history.

## The Methodology

There is no methodology. This is a 1,100-word blog post with no experiments, no math, and no citations. Treating it as a paper is a category error — it is a **prior**, an argument about how to allocate research effort.

The structure of the argument is four claims:

1. AI researchers have repeatedly tried to build knowledge into their agents.
2. This always helps in the short term, and feels good.
3. In the long run it plateaus, and then actively blocks progress.
4. The breakthrough eventually comes from the opposite direction — scaling compute via search and learning.

Point 3 is the non-obvious one. Sutton does not just say hand-built knowledge stops helping; he says it *inhibits*. The mechanism he gives is that domain knowledge "tends to complicate methods in ways that make them less suited to taking advantage of general methods." A system with hand-written phoneme rules is not a system you can just feed 10,000 more hours of audio into. The special-case code becomes a load-bearing wall you cannot remove.

There is also a time-budget argument. Search-and-learning and knowledge-engineering "need not run counter to each other, but in practice they tend to." Time spent on one is time not spent on the other, and people form psychological commitments to whichever one they started with.

## Ablation Studies and Experiments

The essay's "experiments" are four historical case studies. Each has the same shape: knowledge-based approach leads, compute-based approach overtakes, incumbents complain.

**Chess.** Deep Blue beat Kasparov in 1997 using massive deep search plus custom hardware (roughly 200 million positions per second), not chess understanding. The computer-chess research community, which had spent years on human-style positional reasoning, called it "brute force" and said it was not how people play. Sutton's note: they "were not good losers."

**Go.** The same story, delayed 20 years. Huge effort went into encoding Go-specific pattern knowledge to *avoid* search. All of it became irrelevant once search worked at scale. The extra ingredient here was **self-play learning of a value function** — which Sutton points out is the same kind of thing as search, another way to pour compute into the problem. (AlphaGo 2016, then AlphaZero 2017, which dropped human games entirely and got better.) The value-function framing connects to [[Markov Decision Process]] and to [[A Distributional Perspective on Reinforcement Learning]].

**Speech recognition.** The 1970s DARPA competition pitted systems built on knowledge of words, phonemes and the human vocal tract against statistical hidden Markov models that did far more computation. HMMs won, and that outcome slowly reshaped all of NLP over the following decades. Deep learning for speech is the same arrow, further along: even less human knowledge, even more compute, much bigger training sets.

**Computer vision.** Edges, generalized cylinders, SIFT features — "today all this is discarded." Modern nets keep only convolution and a few invariances and do much better. See [[ImageNet Classification with Deep CNNs (AlexNet)]] (top-5 error 15.3% vs 26.2% for the best hand-engineered feature pipeline in ILSVRC-2012), and then [[An Image is Worth 16x16 Words (ViT)]], which removed even the convolution and won again once the data was large enough.

**What the "ablation" reveals.** The component doing the work is never the clever domain-specific part; it is the part that turns FLOPs into accuracy. Notice the recursion in the vision line: convolution was itself the bitter lesson applied to SIFT, and then ViT applied the bitter lesson to convolution. The [[An Image is Worth 16x16 Words (ViT)#^inductive-bias|inductive bias]] you keep today is the hand-engineering you delete tomorrow.

**What did not work, generalised:** every attempt to make a system work "the way the researchers thought their own minds worked." Sutton calls this "a colossal waste of researcher's time."

## Worth Remembering

- **The essay is often mis-cited as "just use more compute."** It is not. It says use methods that *keep improving as compute grows*. A method that saturates is useless no matter how many GPUs you give it. The test is the slope of the scaling curve, not the size of the cluster.
- **Sutton assumes compute is the binding constraint. Often data is.** [[Training Compute-Optimal Large Language Models (Chinchilla)]] showed the 280B-parameter Gopher was badly undertrained: a 70B model on 1.4T tokens beat it, with tokens scaling roughly one-to-one with parameters. If high-quality tokens run out, the "just scale" strategy stops being free. Sutton had nothing to say about this in 2019.
- **Architecture is smuggled-in human knowledge, and it matters enormously.** Self-attention in [[Attention Is All You Need]] is a design choice a human made. So is the residual connection in [[Deep Residual Learning for Image Recognition (ResNet)]] and the gating in [[Long Short-Term Memory (Neural Computation)]]. The honest reading is that these count as *meta-methods* — they are how compute gets used, not what the answer is. But the line between "meta-method" and "domain knowledge" is fuzzy and Sutton never draws it.
- **RLHF is a direct counterexample worth chewing on.** [[Training language models to follow instructions with human feedback]] injects human preference data on purpose, and a 1.3B InstructGPT was preferred to the 175B base GPT-3. That is human knowledge beating a 100× compute gap. The reconciliation: humans supplied the *objective*, not the *method* — but it does show that where the signal comes from is not settled.
- **The essay is at its weakest on why the pattern should continue.** Moore's law in its original form is over; gains now come from specialised silicon and parallelism, which is a slower and lumpier curve. Sutton offers no argument beyond induction over four case studies.
- **Practical use for an engineer.** When you are picking between two approaches, ask: if I get 10× the compute and 10× the data next year, which one gets better and which one is capped? Prefer the one with headroom. This is also the cleanest lens on [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] — a lot of hand-tuned neural recommenders lost to well-tuned nearest-neighbour baselines, which is the bitter lesson pointing the *other* way when the data is small. Scale-first is a claim about the limit, not about your Tuesday.
- **Follow-up reading.** Rodney Brooks wrote "A Better Lesson" as a direct rebuttal, arguing the compute-heavy path is economically and environmentally unsustainable and that Sutton ignores how much structure is baked into modern architectures.

## Links

Related: [[Scaling Laws for Neural Language Models]] · [[Training Compute-Optimal Large Language Models (Chinchilla)]] · [[ImageNet Classification with Deep CNNs (AlexNet)]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Attention Is All You Need]] · [[Language Models are Few-Shot Learners (GPT-3)]] · [[Playing Atari with Deep Reinforcement Learning (DQN)]] · [[Markov Decision Process]] · [[Foundation Models]] · [[Deep Learning]] · [[Are We Really Making Much Progress- A Worrying Analysis (RecSys, best paper)]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Training language models to follow instructions with human feedback]]

New topics worth writing: Monte Carlo Tree Search, AlphaGo and AlphaZero self-play, Deep Blue and minimax search with alpha-beta pruning, Hidden Markov Models for speech, SIFT and hand-crafted visual features, Moore's Law and the end of Dennard scaling, A Better Lesson (Brooks rebuttal), test-time compute scaling, the data wall
