---
aliases:
  - Observable
  - Unobservable
  - Controllability
  - Controllable
  - Observability and Controllability
  - Identifiability
  - Unidentifiable
tags:
  - control-theory
  - estimation
  - decision-sciences
  - failure-mode
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** A quantity is **unobservable** if it leaves no trace in anything you measure — and then no sensor upgrade, no cleverer algorithm and no amount of extra data will ever recover it. Its twin, **controllability**, asks whether your inputs can reach every part of the state.
> **Metaphor:** One thermometer at the shower head, two taps. 38° — but is it a trickle or a torrent?
> **Where it bites:** *"Is the information even in the data?"* Logging decisions you can't backfill, parameters that trade off perfectly, single video frames with no velocity in them.

---
A shower with two taps, hot and cold, and **one** thermometer at the shower head. It reads **38°**. Lovely.

Now a question: **are both taps wide open, or both barely cracked?**

Have a think before reading on.

...

You can't tell. Balance the taps at a trickle and you get 38°. Balance them at full blast and you get 38°.

| Hot tap | Cold tap | Thermometer reads |
|---|---|---|
| wide open | wide open | 38° |
| barely open | barely open | 38° |

Fine — so what would fix it? ~={blue}Go through the options you'd actually reach for:=~

---
# The thermometer that can't tell 🚿

![[shower_two_taps_one_thermometer.png]]

- **A better thermometer?** It'll say 38.0000° instead of 38°. Precision was never the issue.
- **A cleverer filter?** There is no flow-rate information *anywhere* in a temperature reading. You can't compute something out of nothing.
- **More readings?** A thousand readings of 38° say exactly what one did.

The only thing that works is a **second sensor** — a flow meter. You have to go and measure the other thing.

> [!NOTE] Observability
> A system is **observable** if, by watching its outputs for long enough (knowing the inputs), you can work out its entire internal [[State-Space Model|state]]. If two different states would produce identical outputs forever, the difference between them is **unobservable**. ^observability-def

> [!SUCCESS] Core idea
> ~={pink}It isn't a tuning problem. It's a **plumbing** problem.=~ You don't fix it with better estimation; you fix it by adding a sensor or changing what you measure. This is the concept-note version of [[Decision Sciences#^observability|the chapter]], and the habit it teaches is [[Decision Sciences#^is-the-info-even-there|asking whether the information is even there]]. ^plumbing-not-tuning

---
# The subtle part — time can rescue you

Here's where it gets interesting, because *"my sensor doesn't measure it"* is **not** the same as *"it's unobservable"*.

**Case 1 — a position sensor, and you want velocity.** The radar never reports speed. But position *changes because of* velocity: 100 m, then 120 m a second later → 20 m/s. The hidden variable leaks into the measured one through the dynamics. ✅ **Observable.**

**Case 2 — a speedometer, and you want position.** Two cars, both doing a steady 20 m/s. One is at the 100 m mark, the other at the 5,000 m mark. Their speedometers read identically — now, in a minute, forever. Position never feeds back into speed, so it leaves no fingerprint. ❌ **Unobservable**, however long you watch.

> [!TIP] The rule of thumb
> A hidden variable is observable when **it influences something you measure**, directly *or through the dynamics over time*. Formally: stack $C$, $CA$, $CA^2, \ldots$ — "what I see now, what I'll see after one step, after two…" — and check the stack has full rank. In plain words: *does every direction of the state eventually leave a fingerprint?* ^observability-test

---
# The twin — controllability

Flip the question around. Not *"can I **see** every part of the state?"* but *"can I **steer** every part of it?"*

Imagine a shower with a single lever that moves the hot and cold valves **together**. You can set any flow you like. You can never change the temperature. One direction of the state is simply out of reach of your input.

| | Asks | Fails when | The fix |
|---|---|---|---|
| **Observability** | can outputs reveal the whole state? | two states look identical forever | **add a sensor** |
| **Controllability** | can inputs drive the state anywhere? | some direction ignores every input | **add an actuator** |

They're mathematical mirror images — one is about $C$ and $A$, the other about $B$ and $A$ — and between them they answer the two questions you should ask before designing *anything*: can I see it, and can I move it?

---
# Where it bites outside control 🪤

| Situation | What's unobservable | The only real fix |
|---|---|---|
| A model $y = a \cdot b \cdot x$ | $a$ and $b$ separately — only the product matters (**non-identifiability**) | reparameterise, or measure one independently |
| One Atari frame | the ball's **velocity** | stack 4 frames → [[DQN]] |
| Logs with no record of *why* an item was shown | the logging policy's propensities | **log them at serve time** — you can't backfill → [[Off-Policy Evaluation]] |
| Revenue moved; price and an ad campaign changed the same day | which one caused it | run an experiment → [[AB Testing]] |
| Behaviour → the reward that produced it | many rewards explain the same behaviour | → [[Inverse Reinforcement Learning]] |
| An agent that can't see the full state | the state itself | act on a belief instead → [[POMDP]] |

> [!WARNING] The expensive mistake
> The instinct, when an estimate is bad, is to reach for a bigger model or more data. ~={red}If the quantity is unobservable, every hour of that is wasted.=~ So ask it first, of any system: *"if I already had the perfect method, could I even answer this from what I'm collecting?"* Often the honest answer is no — and knowing that on day one saves you weeks. ^ask-it-first

---
> [!SUCCESS] If you remember one thing
> Before tuning the estimator, check the plumbing. ~={pink}If two different truths would give you identical data forever, the fix is a new measurement — never a better algorithm.=~

---
# ⁉️
Everything so far has quietly assumed the world is *linear* — straight-line physics, a sensor that reports a simple slice of the state. Most real sensors measure something bent: a radar gives you range and bearing, not $x$ and $y$. What happens to a tidy bell-curve belief when you push it through a curve?

→ [[Extended Kalman Filter]]
