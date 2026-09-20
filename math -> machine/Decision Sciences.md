---
aliases:
  - Decision Science
  - Control to RL
tags:
  - decision-sciences
  - reinforcement-learning
  - control-theory
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** The story of how machines learned to act well — feedback → filtering → planning → learning → learning the *goal itself*.
> **Metaphor:** Each chapter is a thing the previous one couldn't handle. Control can't see through noise. Filtering can't plan. Planning needs a known world. Learning needs a known reward.
> **Where it bites:** It's the map. When someone says "should this be a bandit or full RL?", the answer lives in this progression.

> [!INFO] 📖 A guided path
> This note is being built chapter by chapter as a learning path. Each section follows the same shape: **the historical problem → the intuition → the vocabulary → a toy example → what it connects to → why it wasn't enough.**

>[!TIP] **How should an agent act when the world is noisy, the future matters, and the goal itself is unclear?**

> [!TIP] 🗺️ Looking for one specific term?
> This note is the **story** — read it to rebuild the intuition. The **map** — every concept as its own note, a *symptom → note* table, and the full reading order — is [[Control and Reinforcement Learning]].

---
### 1. The first instinct: feedback before intelligence

**Control theory (early 1900s–1950s)** 🎛️

> [!ABSTRACT] 📚 Concept notes for this chapter
> [[Feedback Loop]] · [[PID Controller]] · [[Step Response]] · [[Stability]] · [[PID Tuning]] · [[Feedforward Control]] · [[Transfer Function]]

This all begins with machines, not minds.

Engine governors, thermostats, autopilots. The problem was brutally practical:  
“How do I keep a system stable when the world pushes it around?” or rather "How do we get a machine to "behave" without a human constantly turning the knobs?"

#### 1. The Historical Problem
In the early days of the Industrial Revolution, engineers faced a major issue with steam engines. If you fed the engine more coal, it sped up. If the load changed (like a mill grinding more grain), it slowed down. A human could stand there and adjust the steam valve, but humans are slow, they get tired, and they make mistakes.
#### 2. The Core Intuition: The Feedback Loop 🔄
The solution was to create a **Feedback Loop**.

Imagine you are taking a shower. You want the water to be a perfect $38^\circ\text{C}$.
1. **Sense:** You feel the water on your skin.
2. **Compare:** Your brain compares the current temperature to your "target" (the perfect $38^\circ\text{C}$).
3. **Act:** If it’s too cold, you turn the hot handle up.
4. **Repeat:** You feel the water again and adjust further.

> [!TIP] This loop — **Sense → Think → Act** — is the heartbeat of all control systems.

[[PID Controller]] live here. They don’t “think.” They react.
A PID controller says:
- I see an error
- I push back
- I adjust how hard I push based on past and present error

> [!TIP] No uncertainty modeling. No planning. No notion of _future reward_. Just **feedback**.

This era gave us the most important primitive of all:  
> [!TIP] 👉 _**closed-loop control**_ → _action affects the world, which affects the next action._

Everything later inherits this loop.


#### 3. The distinction that matters most — open vs closed loop

This is the single idea to carry out of this chapter.

| | **Open loop** 🍞 | **Closed loop** 🔥 |
|---|---|---|
| Example | A **toaster** — set 3 minutes, heat for 3 minutes | An **oven** — measure the temperature, keep adjusting |
| Does it look at the result? | ❌ never | ✅ constantly |
| Frozen bread? | burnt, or raw — it cannot tell | fine, it just takes longer |
| Needs to be right… | **in advance** | only *eventually* |

> [!SUCCESS] Core idea
> ~={blue}A closed loop is a machine that is allowed to be wrong and recover.=~ An open loop has to be correct before it starts. That is the entire advantage, and it's why feedback took over the industrial world. ^open-vs-closed-loop

#### 4. The vocabulary — six words

Using the shower, so none of these stay abstract:

| Term | In the shower |
|---|---|
| **Plant** | the shower — the system being controlled |
| **Setpoint** | $38^\circ\text{C}$ — the target |
| **Sensor** | your skin |
| **Error** | setpoint − measurement. *The gap* |
| **Actuator** | the tap — what you can physically change |
| **Controller** | you — the rule that turns error into action |

> [!TIP] **Error** is the star of this whole field.
> Control theory is very nearly just the study of *what to do with the error*. [[PID Controller|P, I and D]] are three different answers: react to it **now**, accumulate it from the **past**, or anticipate where it's **going**.

#### 5. A toy example — proportional control

Room is $15^\circ$. You want $20^\circ$. Rule: $\text{heater power} = 10 \times \text{error}$.

| Room temp | Error | Heater power |
|---|---|---|
| $15^\circ$ | 5 | 50 |
| $18^\circ$ | 2 | 20 |
| $19.5^\circ$ | 0.5 | 5 |

Two things fall out of this, and neither was designed in:

1. **It eases off as it closes in.** It glides to the target instead of slamming into it. That's free — it's just a consequence of "push in proportion to the gap."
2. **It never quite arrives.** At zero error there's zero power, but the room is always leaking heat. So it settles *just below* $20^\circ$, forever. This is **steady-state error**, and it is precisely why the **I** term exists → [[PID Controller#**I** for Integral|Integral]].

#### 6. The dark side — feedback can make things worse 🚿

Old house, bad shower. You turn the hot tap. Nothing. Turn it further. Still nothing. Further. Suddenly **scalding**. You yank it to cold. Nothing... then **freezing**.

You are oscillating — and *you* caused it. Doing nothing would have been better.

The culprit is **delay**. You are reacting to information about the *past* and applying it to the *present*.

> [!WARNING] The fundamental danger of every feedback loop
> ~={red}Delay + a strong reaction = instability.=~ The harder you push on stale information, the wilder the swings. This is why controller **gain** has to be tuned rather than maximised, and why the **D** term exists — to anticipate instead of merely react. ^delay-causes-instability

This has a name worth knowing: a loop that oscillates or runs away is **unstable**, and most of classical control theory is the mathematics of proving a loop *won't* do that.

#### 7. What feedback can and cannot do

> [!SUCCESS] What is genuinely remarkable about it
> A thermostat **knows no thermodynamics**. Watt's governor knows no physics. They hold a target without any model of the system they're controlling. ~={blue}Feedback buys you competence without understanding=~ — which is why PID still runs most of the physical world, a century later. ^competence-without-understanding

> [!WARNING] **But then, why was this idea insufficient?**
Simple controllers like this are amazing for things that **stay the same** — like keeping a room at one temperature. But they struggle when:
>1. **The world is "noisy"** (the thermometer is broken or jittery). → the controller chases the noise. **Chapter 2**
>2. **Things are hidden** (you can't see the temperature, only the steam). → **Chapter 2**
>3. **The future matters more than the now** (if I turn the heater on now, it takes 20 minutes to warm up). → it has no concept of a future at all. **Chapter 3**

> [!TIP] 🌱 And a fourth limitation — the seed of this entire note
> A controller cannot **choose its own target**. Someone has to walk up and say "$38^\circ$".
>
> That sounds trivial now. It is the thing we spend the next eight chapters slowly taking away. By **Chapter 8** we won't know the target either — and we'll have to infer it from human beings who can't articulate it. ^who-sets-the-setpoint

---
### 2. Reality intrudes: noise, partial observation

**State-space models & [[Kalman Filter|Kalman filters]] (1950s–1960s)**

> [!ABSTRACT] 📚 Concept notes for this chapter
> [[State-Space Model]] · [[Kalman Filter]] · [[Bayes Filter]] · [[Beliefs]] · [[Uncertainty]] · [[Observability]] · [[Extended Kalman Filter]] · [[Particle Filter]] · [[Hidden Markov Model]]

Then engineers hit a wall: **sensors lie**.

You don’t observe the true state of the world. You see noisy shadows of it.

>[!INFO] Self-driving car's cruise control
>Imagine you’re designing a self-driving car using just this PID controller logic.
>It works great for speed, but what happens if the "eye" of the car (the camera 📸) is a bit blurry? Or if the sensor that measures speed is jittery, jumping between 59 and 61 mph every second?

Kalman’s insight was subtle and profound:  
> [!TIP] “Let’s separate **what the world is** from **what we observe**, and reason probabilistically about the gap.”

This is where **state** becomes a formal concept:
- Hidden state (what actually matters)
- Observations (what you can measure)
- Dynamics (how state evolves)

Kalman filters didn’t choose actions.  
They answered a different question:
> [!TIP] “Given uncertainty, what do I _believe_ about the world right now?”

This belief-state idea quietly became foundational later.

#### The three obvious answers, and why each one fails

You have two sources of truth about where you are, and both are wrong. So which do you believe?

| Instinct | What goes wrong |
|---|---|
| **"Trust the sensor"** | You chase **noise**. And remember [[Decision Sciences#1. The first instinct: feedback before intelligence\|Chapter 1]] — a controller reacts to error. Feed it a jittery sensor and it thrashes the actuator fighting phantoms. You've rebuilt the bad shower 🚿 |
| **"Trust the model"** | You **drift**. Nothing ever corrects you, so small errors compound silently. You become confidently, increasingly wrong |
| **"Average them 50/50"** | Closer — but obviously wrong when one source is a precision instrument and the other is a rough guess |

> [!SUCCESS] Kalman's answer
> Average them — but ~={blue}weight each source by how much you trust it.=~ And "how much you trust it" is simply **how uncertain it is**. That weight is the **Kalman gain**, and it is recomputed at every single step. ^weight-by-uncertainty

#### The actual big idea — stop storing a number, start storing a belief

This is bigger than the averaging trick, and it's the part that echoes through the rest of this note.

In Chapter 1 the controller stored a **number**:
> `temperature = 19.5`

From here on we store a **belief**:
> `temperature is probably 19.5, give or take 0.3`

**Two numbers instead of one.** The first is your best guess; the second is *how sure you are* — and that second number is what tells you how much to trust yourself versus a new measurement.

> [!SUCCESS] Core idea
> You have stopped tracking **the world**. You are now tracking ~={pink}**what you know about the world**.=~ Everything later in this note — exploration, regret, Thompson sampling, RLHF — is downstream of that one move. See [[Beliefs]] and [[Uncertainty]]. ^track-what-you-know

#### The rhythm — predict, then update

Everything from here runs on a two-beat loop:

| Beat | What happens | Your uncertainty |
|---|---|---|
| **Predict** | Step forward using your model of the dynamics | **Grows** 📈 — time passed, you didn't look, you know less |
| **Update** | Take a measurement, fold it in | **Shrinks** 📉 — you looked, you know more |

Breathe out, breathe in. Forever.

The quietly beautiful part: ~={blue}uncertainty decays on its own as time passes, and evidence is the only thing that pulls it back.=~

#### A toy example, with numbers

Tracking a car's position:

1. **You believe:** $100\text{m} \pm 10\text{m}$
2. **Predict** one second forward at ~20 m/s → $120\text{m} \pm 14\text{m}$
	- the spread **grew** from 10 to 14 — you didn't look, so you're less sure
3. **Radar reports:** $130\text{m} \pm 5\text{m}$
	- radar ($\pm 5$) is more certain than your prediction ($\pm 14$), so you lean toward it — without abandoning your own estimate
4. **New belief:** roughly $128\text{m} \pm 4.7\text{m}$

> [!TIP] Look at that last number 🔍
> $\pm 4.7$ is **smaller than both** $\pm 5$ and $\pm 14$. The combined estimate is *more certain than either source alone*.
>
> Two blurry photographs, stacked, give a sharper picture than either one. That is the entire payoff of the method. ^combining-beats-both

#### Vocabulary

| Term | Plain meaning |
|---|---|
| **State** | What is actually true. Hidden from you |
| **Observation** | The noisy shadow of it that you get to see |
| **Dynamics** | Your model of how the state evolves |
| **Process noise** | How wrong your **model** is |
| **Measurement noise** | How wrong your **sensor** is |
| **Belief** | Your guess **+** your uncertainty about it |
| **Kalman gain** | The trust dial between prediction and measurement |

#### One more thing — and this one surprised me

Let's go back to the shower. 🚿

You have **one** thermometer, sitting at the shower head. It reads $38^\circ$. Lovely.

Now I ask you a question: **are both taps wide open, or are both barely cracked?**

Have a think about it before reading on.

...

You can't tell. And it's worth seeing *why* not:

| Hot tap | Cold tap | Thermometer reads |
|---|---|---|
| wide open | wide open | $38^\circ$ |
| barely open | barely open | $38^\circ$ |

Both taps balanced gives you $38^\circ$ either way. A **trickle** at $38^\circ$ and a **torrent** at $38^\circ$ produce **the exact same reading**.

So now the useful question: what would fix this?

- Buy a **more expensive thermometer**? No — it'll just tell you $38.0000^\circ$ instead of $38^\circ$. Precision was never the problem.
- Use a **cleverer filter**? No. There is genuinely no information about flow rate anywhere in a temperature reading. You cannot compute something out of nothing. 🚫
- Take **more readings over time**? Still no. A thousand readings of $38^\circ$ tell you the same thing one did.

The only thing that fixes it is **a second sensor** — a flow meter. You have to go and *measure the other thing*.

> [!SUCCESS] And now the word
> When the truth you care about **cannot be worked out from what you're measuring**, no matter how good your sensor or how clever your maths — that thing is **unobservable**.
>
> ~={blue}It isn't a tuning problem. It's a *plumbing* problem.=~ You don't fix it with better estimation, you fix it by adding a sensor or redesigning what you measure. ^observability

> [!TIP] Why this is worth carrying around
> This was the bit that reframed things for me. I'd assumed that if my estimate was bad, I needed a better algorithm. Sometimes that's true. But sometimes ~={pink}the information simply isn't in the data=~, and then every hour spent tuning the model is wasted.
>
> It's a question worth asking early of any system: *"if I already had the perfect method — could I even answer this from what I'm collecting?"* Often the honest answer is no, and that saves you weeks. ^is-the-info-even-there

#### How this connects back to Chapter 1

The architecture becomes two boxes instead of one:

```
world → sensor → [ FILTER ] → belief → [ CONTROLLER ] → action → world
```

The Chapter 1 controller **does not change at all**. It simply receives a cleaned-up *estimate* instead of a raw sensor reading.

Perception and decision have split into separate jobs — and that separation survives all the way into modern robotics and RL, where it reappears as "representation learning" versus "policy learning".

> [!WARNING] **Why was this still not enough?**
> 1. **It never chooses anything.** A Kalman filter tells you where you *are*. It has no opinion whatsoever about what you should *do*. It is perception, not decision.
> 2. **It assumes a tidy world** — linear dynamics, bell-curve noise. Reality often isn't, which is why you'll meet the EKF, UKF and particle filters. All patches on the same idea.
> 3. **Still no future.** It is the best possible answer to *"where am I now?"* — and cannot reason about consequences at all.
>
> That third gap is **Chapter 3**. ^ch2-limits


---

### 3. Decisions become mathematical

**Operations Research & Dynamic Programming (1940s–1960s)**

> [!ABSTRACT] 📚 Concept notes for this chapter
> [[Dynamic Programming]] · [[Bellman Equation]] · [[Curse of Dimensionality]] · [[LQR]] · [[Model Predictive Control]] · [[System Identification]]

In parallel—mostly driven by wartime logistics—another thread emerged.

The question here was not control but choice:  
> [!TIP] “How do I make a sequence of decisions that **minimizes cost** or **maximizes reward** **over time**?”

Richard Bellman introduces **dynamic programming** and the Bellman equation.

This is the first time someone formalizes:
- Long-term consequences
- Recursive value of decisions
- Optimality over trajectories, not steps

No learning yet.
The model is assumed known.  
But the _logic of planning_ is born.

---
#### A morning you've probably had 🚗

You're leaving home for work. First junction: **left** onto the quiet back road, or **right** onto the motorway slip.

You glance at both:
- the motorway leg takes **5 minutes**
- the quiet road leg takes **8 minutes**

So which do you take?

Right, obviously. It's faster. 5 beats 8.

...

Except the motorway feeds into Junction 2. And at 8:45am, Junction 2 is a car park. From there it's another **32 minutes**.

The quiet road just goes to the office. **20 minutes**, door to door.

| Choice | This leg | Everything after | **Total** |
|---|---|---|---|
| Motorway | 5 | 32 | **37 min** |
| Quiet road | 8 | 12 | **20 min** ✅ |

~={red}The option that was faster at the moment you chose it was worse by 17 minutes.=~

#### So what went wrong?

You compared the wrong thing.

You compared *"how long is this next bit?"*, when the question you actually cared about was *"how long until I'm sitting at my desk?"*

And notice — neither of the previous two chapters could have caught this:
- A **controller** reacts to the error *right now*
- A **filter** tells you where you are *right now*

Neither has any machinery at all for *"this looks good now and costs you later."*

> [!NOTE] And now the word
> Choosing whatever looks best at this instant, ignoring where it leads, is called acting **greedily**. It's not a stupid strategy — it's the *obvious* one, and it's right surprisingly often. It just fails exactly when consequences are delayed. ^greedy

#### "Fine, I'll just check every route"

Reasonable. Evaluate every possible route, pick the shortest.

Your commute has 10 junctions with 3 choices at each:
$$3^{10} = 59{,}049 \text{ routes}$$

Annoying, but a computer could do it. Now make it a delivery van with 30 stops. Or chess, where the number of possible games exceeds the number of atoms in the observable universe. ♟️

> [!WARNING] Enumerating the future never scales
> Every extra step **multiplies** the work rather than adding to it. This is why "just simulate everything" is never the answer, in any of the chapters that follow. ^enumeration-explodes

#### Bellman's move — start at the end 🔙

Here's the trick. Watch it happen rather than taking my word for it.

**Forget the start.** Walk to the *last* junction — the one 5 minutes from the office.

Standing there, what's the best route in? You can just **look**. Left is 5 minutes, right is 9. Best from here: **5**.

Write `5` on that junction. 📍

Now step back to the second-to-last junction. From here you can reach:
- Junction **X** — 3 min to get there, and X is already labelled `5`
- Junction **Y** — 2 min to get there, and Y is already labelled `8`

So: $3 + 5 = 8$, or $2 + 8 = 10$. Best is **8**. Write `8`.

Now notice what you **didn't** do.

You didn't re-explore anything beyond X or Y. You didn't consider a single one of the thousands of routes past them. You read ~={blue}one number=~ off each neighbour.

Keep walking backwards. At every junction you compare three numbers. Never 59,049.

#### And now the words

That number you wrote on each junction — *"the best total time from here to the end"* — is the **value** of that place.

And the rule you kept applying:

> **value of here = cost of one step + value of where that step lands you** (taking the best option)

That is the **Bellman equation**.

> [!SUCCESS] Core idea
> The future is infinite and branching — but you never have to look at it, because ~={pink}the value of your neighbour already has the entire future baked into it.=~
>
> You've swapped *"think about everything that could happen"* for *"think one step, then read a number."* ^bellman-intuition

#### A name that confused me

The method is called **dynamic programming**, and it has ~={red}nothing to do with programming.=~

In the 1950s, "programming" meant **planning** or **scheduling** — the same sense as a TV programme, or "linear programming". "Dynamic" meant it unfolded over multiple stages in time.

Bellman later admitted he partly chose the name because it sounded impressive and hard to object to — he was working under a Secretary of Defense who was hostile to anything resembling mathematical research. So the name is essentially **marketing**.

> [!TIP] What it actually means
> **Dynamic programming = planning, over time, by working backwards.** If you'd been imagining code that rewrites itself at runtime — that was me too. ^dp-naming

#### What it needs — and why that's a problem

Go back and look at what you had to know to fill in those numbers.

You needed the entire **map**. Every road, every junction, and how long each leg takes — *before you set off*.

> [!WARNING] **Why was this still not enough?**
> That's perfectly fine for a chess board or a warehouse layout. But:
> - a robot in a building it has never entered
> - an ad system meeting a user it has never seen
> - a trader in a market that has never had exactly this day
>
> **There is no map.** And with no map, there are no numbers to write on the junctions.
>
> [[Decision Sciences#4. The unifying abstraction\|Chapter 4]] gives this whole picture a proper name and vocabulary. [[Decision Sciences#5. Reality intrudes again: the model is unknown\|Chapter 5]] takes the map away. ^ch3-limits

---

### 4. The unifying abstraction

**Markov Decision Processes (1950s–1970s)**

> [!ABSTRACT] 📚 Concept notes for this chapter
> [[Markov Property]] · [[Markov Decision Process]] · [[Policy]] · [[Reward Function]] · [[Discount Factor]] · [[Value Function]] · [[Value and Policy Iteration]] · [[Credit Assignment]] · [[POMDP]]

MDPs are where everything clicks together.

They fuse:
- State (from control + filtering)
- Actions (from control)
- Rewards (from economics & OR)
- Dynamics (from physics)
- Planning (from dynamic programming)

The Markov property is the key simplifying assumption:

> _“The future only depends on the present state and action.”_

From this one assumption spill out:

- Value functions
- Q functions
- Policies
- Bellman expectation and optimality equations

This is the **mathematical skeleton of RL**, even though no learning has happened yet.

At this point, decision science exists—but only if you know the world model.

---
#### Why this chapter is the hinge 🔗

Everything before this was a *different tradition solving its own problem*. This is where they turn out to have been solving the same one.

| Came from                                                                                 | Contributed                                                      |
| ----------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| [[Decision Sciences#1. The first instinct: feedback before intelligence\|Ch 1]] — control | **actions** that affect the world, in a loop                     |
| [[Decision Sciences#2. Reality intrudes: noise, partial observation\|Ch 2]] — filtering   | **state** as a formal thing, separate from what you observe      |
| [[Decision Sciences#3. Decisions become mathematical\|Ch 3]] — dynamic programming        | **value**, and the recursion that makes long horizons computable |
| economics / OR                                                                            | **reward** — a single number saying how good an outcome was      |

Snap those together and you get one object that all four traditions can write in: the **[[Markov Decision Process|MDP]]**.

#### The two questions that separate a chain from an MDP

Take a shop whose stock drifts between `Low`, `Medium` and `High` as customers buy things. That's a [[Markov Property|Markov chain]] — a system evolving on its own. The only question you can ask is *"where will it probably end up?"* That's **prediction**.

Now put yourself in charge. You can order stock, or discount, or do nothing. Suddenly the question changes to *"what should I do?"* That's **optimisation**.

> [!SUCCESS] Core idea
> ~={blue}A Markov chain describes. An MDP decides.=~ The whole difference is whether there's someone in the picture who can act. ^chain-describes-mdp-decides

#### The thing I had backwards

I assumed an action *causes* a transition — order stock, therefore move to `High`.

It doesn't. **An action changes the odds.**

You're in `Low Inventory`. "Order more" doesn't move you to `High`; it swaps out the entire row of probabilities you're about to draw from. You still land somewhere random — just from a *better distribution*.

> [!TIP] The slot machine 🎰
> Your action picks **which lever to pull**. The lever decides which set of outcomes you're gambling on. The outcome itself is still drawn at random.
>
> ~={pink}You don't choose where you land. You choose the distribution you land from.=~
>
> Which is exactly why it's written $P(s' \mid s, a)$ — *given* the state *and* the action. Full detail in [[Markov Decision Process#The bit I got wrong first time|the MDP note]]. ^action-changes-odds

#### The vocabulary this unlocks

Four objects, and it's worth keeping their **types** straight — that's where I got tangled:

| | Takes | Returns | Asks |
|---|---|---|---|
| **$V(s)$** | a state | a **number** | how good is it to be here? |
| **$Q(s,a)$** | state + action | a **number** | how good is doing *this*, from here? |
| **$\pi(s)$** | a state | an **action** | what should I do? |

$V$ and $Q$ are scores. $\pi$ is a **rule** — a different kind of object entirely.

And the pair that ties them together:
$$V^*(s) = \max_a Q^*(s,a) \qquad \pi^*(s) = \arg\max_a Q^*(s,a)$$
**max** says *how good the best option is*. **argmax** says *which option it was*. ~={blue}Once you have $Q$, the policy is free.=~ → [[Markov Decision Process#max vs argmax 🏙️|max vs argmax]]

> [!WARNING] **Why was this still not enough?**
> Look back at the store table — the one with `0.8`, `0.1`, `0.9` in it.
>
> **Who gave you those numbers?**
>
> An MDP is only solvable if you already know $P$ (how the world responds) and $R$ (what things are worth). Every method so far has quietly assumed someone handed you both.
>
> Nobody hands you those. That's [[Decision Sciences#5. Reality intrudes again: the model is unknown\|Chapter 5]]. ^ch4-limits

---

### 5. Reality intrudes again: the model is unknown

**Reinforcement Learning (1980s–1990s)**

> [!ABSTRACT] 📚 Concept notes for this chapter
> [[Reinforcement Learning]] · [[Model-Based vs Model-Free RL]] · [[Monte Carlo Methods]] · [[Temporal Difference Learning]] · [[Q-Learning]] · [[On-Policy vs Off-Policy]] · [[Policy Gradient]] · [[Actor-Critic]] · [[Exploration vs Exploitation]] · [[Regret]]

Now comes the leap.

What if:

- You don’t know the transition dynamics?
- You don’t know the reward function exactly?
- You only learn by acting?

This is where RL is born—not as a new goal, but as a concession to ignorance.

Temporal Difference learning, Q-learning, policy gradients emerge.

Key shift:

> From _planning with a model_ → _learning through interaction_

Exploration vs exploitation becomes unavoidable.  
Regret becomes a metric.  
Learning replaces inference.

---
#### Who gave you those probabilities?

Go back to the store table with `0.8` and `0.9` in it. Somebody had to fill that in.

The honest answer is **experience** — a manager who ran the shop long enough to notice the patterns. And that's a real strategy with a real name: **model-based** learning. Estimate $P$ and $R$ from data, then plan with them using [[Decision Sciences#3. Decisions become mathematical|Chapter 3]]'s machinery.

But it raises a question: **what exactly should you write down?**

#### Two tables 📋

Both are keyed on **(current state, action)**. They differ in what comes out.

**Table A — the mechanics.** From `Low`:

| Action | → L | → M | → H |
|---|---|---|---|
| Do nothing | 0.9 | 0.1 | 0.0 |
| Order more | 0.0 | 0.8 | 0.2 |

**Table B — the verdict.** From `Low`:

| Action | worth |
|---|---|
| Do nothing | 12 |
| Order more | **40** |

> [!WARNING] Table A contains no notion of worth. None.
> This took me a while. Table A never says `Low` is bad — it only says where you go.
>
> Two managers run identical shops, so their Table A is **character-for-character identical**. But one is paid on profit and the other on never running out of stock. Their Table B's come out completely different.
>
> ~={blue}Table A is the physics. The rewards are what *you* care about. Table B is both, combined and cached.=~ ^table-a-has-no-worth

Where does `12` come from? Say $V(L)=10$, $V(M)=40$, $V(H)=60$. Do-nothing costs $-1$ in lost sales:
$$-1 + (0.9 \times 10) + (0.1 \times 40) = 12$$

> [!SUCCESS] That arithmetic *is* the Bellman equation
> Immediate reward, plus the weighted average of wherever you land. And `12` isn't tied to any one destination — every possible future is summed inside that single number.
>
> Table B has a name: it's the **Q-function**. $Q(\text{Low}, \text{OrderMore}) = 40$. → [[Markov Decision Process#V, Q and π — three things that are easy to blur|V, Q and π]] ^table-b-is-q

#### The leap — you can skip Table A entirely

Here's the thing. The manager never multiplied a probability by a value in their life.

They hit `Low`, ordered stock, cleared £38. Months later: £45. Then £37. Then £41.

And they did the obvious thing with that pile of numbers — they **averaged** them. Not on paper; as a slowly-updated gut feel.

Written as a rule:

$$\text{new estimate} = \text{old estimate} + \alpha \times (\underbrace{\text{what I just saw} - \text{what I expected}}_{\text{the surprise}})$$

You held a belief. Reality arrived. You nudged the belief toward it by a fraction $\alpha$. Run it a couple of hundred times and it lands on 40 — ~={pink}without ever knowing a single transition probability.=~

> [!SUCCESS] Three names earned at once
> - That surprise term is the **TD error** — *temporal difference*, the gap between predicted and observed.
> - Learning this way is **temporal difference (TD) learning**.
> - And because Table A never got built, this is **model-free** learning. ^td-learning

#### 🎯 The callback

> [!TIP] Look at these two side by side
> **[[Kalman Filter|Kalman]]:**  `new estimate = prediction + gain × (measurement − prediction)`
> **TD learning:** `new estimate = old estimate + α × (observation − old estimate)`
>
> **The same equation.** One corrects a belief about *where you are*; the other corrects a belief about *how good this is*. Sixty years apart, different fields, identical shape.
>
> This is the moment the whole note snaps together. ^kalman-td-same-shape

#### The catch — you only learn about what you try

You've hit `Low` two hundred times and ordered stock every single time. You're confident it's worth 40.

You have never once tried **Discount**. How would you ever discover it's worth 55?

That gap — between what you earned and what you'd have earned had you known the best action all along — is **regret**. It's the opportunity cost of your own ignorance.

And here's why [[Beliefs|belief]] matters rather than just an estimate:

| Action | Tried | Estimate |
|---|---|---|
| Order more | 200× | $40 \pm 1$ |
| Discount | 2× | $40 \pm 30$ |

Same number. You should obviously try **Discount** — it might be 70. The thing that separates them is not the estimate, it's the ~={blue}**width**=~.

> [!SUCCESS] Chapter 2, returning
> In Chapter 2 the width of your belief told you *how much to trust a sensor*. Here it tells you *what is worth trying*. Same object, new job. See [[Uncertainty#Exploration vs exploitation|exploration vs exploitation]]. ^width-tells-you-what-to-try

| Strategy | How it explores |
|---|---|
| **ε-greedy** | At random. Crude, and completely blind to the width |
| **UCB** | Acts on `estimate + uncertainty` — optimism in the face of uncertainty |
| **Thompson sampling** | Draws one plausible world from the belief and acts as if it's true. Wide beliefs get tried, narrow ones settle — ~={pink}the exploration rate tunes itself=~ |

> [!WARNING] **Why was this still not enough?**
> Everything here still assumes a **table** — one row per state. Chess has $10^{47}$ positions; the Earth has $10^{50}$ atoms. And it's worse than storage: you'd have to *visit* every row to fill it, and after ten moves you're in a position never played in history.
>
> Most rows would stay blank forever. A blank row means *"I have never seen this, I have no idea what to do."*
>
> The fix is the one a club player uses without thinking: judge an unfamiliar position by its **features** — king safety, material, structure — not its identity. That's **generalisation**, and it's [[Decision Sciences#7. Deep learning bends the curve|Chapter 7]]. ^ch5-limits

---

### 6. Bandits split off as a special case

**Multi-armed bandits & contextual bandits**

> [!ABSTRACT] 📚 Concept notes for this chapter
> [[Multi-Armed Bandit]] · [[Upper Confidence Bound]] · [[Thompson Sampling]] · [[Contextual Bandit]] · [[AB Testing]] · [[Off-Policy Evaluation]] · [[Bayesian Optimization]]

Bandits are not simpler RL. They are **RL with no state transitions**.

They exist because many real problems don’t have long horizons:

- Ads
    
- Experiments
    
- Recommendations
    
- A/B testing
    

Contextual bandits add partial state back in, without dynamics.

Bayesian optimization, Thompson sampling, UCB—all live here.

This is where **decision sciences meets statistics** in a very practical way.

---

### 7. Deep learning bends the curve

**Deep RL (2013–present)**

> [!ABSTRACT] 📚 Concept notes for this chapter
> [[Function Approximation]] · [[DQN]] · [[Experience Replay]] · [[PPO]] · [[SAC]] · [[Monte Carlo Tree Search]] · [[Offline RL]]

Everything before assumed small state spaces or handcrafted features.

Neural networks change that.

Now:

- States can be images, text, trajectories
    
- Value functions become approximators
    
- Policies become neural programs
    

This is not a new theory—just more expressive function approximation.  
But it unlocks scale.

---

### 8. Humans re-enter the loop

**RLHF, preference learning, inverse RL**

> [!ABSTRACT] 📚 Concept notes for this chapter
> [[Imitation Learning]] · [[Inverse Reinforcement Learning]] · [[Preference Learning]] · [[RLHF]] · [[DPO]] · [[GRPO]] · [[Reward Hacking]]

Here the reward function itself becomes uncertain.

Instead of:  
“Maximize this explicit reward”

We say:  
“Figure out what humans _seem to prefer_, then optimize that.”

This pulls in:

- Inverse reinforcement learning
    
- Preference modeling
    
- Contrastive learning over trajectories
    
- Human feedback as noisy reward signals
    

We are back, philosophically, to Kalman’s original insight:

> You never observe the true objective—only signals about it.

---

### So… is your progression “right”?

Yes in spirit. But the **true backbone** is:

- Feedback loops
    
- State vs observation
    
- Long-term optimization
    
- Uncertainty
    
- Learning under partial knowledge
    

PID → Kalman → MDPs → RL is not a strict lineage.  
It’s multiple traditions converging on the same problem from different angles.

---
> [!SUCCESS] If you remember one thing
> Every chapter is the same loop — **sense → decide → act** — with one more assumption taken away. First the sensor stops being trustworthy, then the future starts to matter, then the rules are unknown, and finally ~={pink}the goal itself is something you only get noisy signals about.=~

---
# ⁉️
This note is the story; the map is [[Control and Reinforcement Learning]] (full reading order: [[Control and Reinforcement Learning#^reading-chain|the chain]]). The backbone of it, in order:

> [[PID Controller]] → [[Kalman Filter]] → [[Beliefs]] → [[Markov Property]] → [[Markov Decision Process]] → [[Uncertainty#Exploration vs exploitation|Exploration vs exploitation]] → [[Dec-POMDP]] → [[Multi-Agent Reinforcement Learning]] ^decision-sciences-chain

And Chapter 8 — humans re-entering the loop — is where this whole story runs straight into LLMs → [[RLHF]], [[DPO]], [[Reward Hacking]].
