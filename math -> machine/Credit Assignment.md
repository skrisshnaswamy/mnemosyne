---
aliases:
  - Credit Assignment Problem
  - Temporal Credit Assignment
  - Structural Credit Assignment
  - Delayed Reward
  - Delayed Rewards
  - Eligibility Traces
  - Eligibility Trace
  - Process Reward Model
  - PRM
tags:
  - reinforcement-learning
  - decision-sciences
  - fundamentals
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** You get **one score at the end** and made **thousands of decisions** on the way. Which of them deserve the credit — or the blame?
> **Metaphor:** A last-minute winning goal. Was it the striker? The pass ten seconds earlier? The save in minute 12 that kept you in the game?
> **Where it bites:** Sparse rewards, long-horizon agents, and LLM reasoning chains where only the final answer is checked.

---
Minute 90. Your team scores. One–nil. 🎉

Now do the post-match analysis. **Who won you that game?**

The striker who put it away? Obviously. But the through-ball from midfield ten seconds earlier made it a tap-in. And that midfielder only had the ball because a defender won a tackle in your own half. And none of it would have mattered without the goalkeeper's save in minute 12.

You have **one number** — "we won" — and about five thousand decisions made by eleven people over ninety minutes. ~={blue}How do you split one number across all of that?=~

---
# The post-match analysis ⚽

![[credit_assignment_football.png]]

The lazy answer is *"whoever touched it last"*. It's also the worst one: it would train strikers and nobody else. Chess makes the point even more sharply — a queen sacrifice looks **terrible** at the moment it's played, and is proved right twelve moves later ([[Markov Decision Process#The one hard part — delayed reward|see the MDP note]]).

> [!NOTE] The credit assignment problem
> Working out which earlier decisions were responsible for an outcome that only became visible later. It comes in two flavours: **temporal** — *which moment?* — and **structural** — *which component?* ^credit-assignment-def

| Flavour | The question | Solved by |
|---|---|---|
| **Temporal** | which of my 5,000 **time steps** mattered? | value functions, TD learning — this note |
| **Structural (parameters)** | which of my 70 billion **weights** mattered? | [[Backpropagation]] — the chain rule *is* a credit-assignment algorithm |
| **Structural (agents)** | which of my 11 **players** mattered? | [[Value Function Factorization]] |

> [!SUCCESS] Core idea
> ~={pink}Delayed reward is the only thing that makes sequential decisions hard.=~ If feedback were instant you'd try each action and keep the best. Everything in RL's toolbox — discounting, value functions, TD errors, advantages — is machinery for getting a signal from *where it arrived* back to *where it was earned*. ^delay-is-the-difficulty

---
# How RL moves credit backwards

**1 · Discount it.** The crudest rule: recent decisions get more. $\gamma^k$ — see [[Discount Factor]]. Better than nothing; often wrong (the save in minute 12 gets $0.99^{4000} \approx 0$).

**2 · Turn the delayed signal into an immediate one.** This is the big idea. Suppose you keep a running estimate of your chance of winning — a [[Value Function]]. Before the through-ball it was 0.30. After it, 0.70. **That jump of +0.40 happened at the moment of the pass**, and *that* is the credit — handed out eighty seconds before the scoreboard changed.

That jump is the **TD error** → [[Temporal Difference Learning]]. A good value function converts one late reward into a stream of small, well-timed surprises.

**3 · Remember where you've just been.** When a surprise arrives, don't only update the *current* state — also update the ones you passed through a moment ago, a little less each. That fading memory is an **eligibility trace**, controlled by $\lambda$:

![[eligibility_traces.png]]
> [!TIP] Reading the chart
> $\lambda = 0$: only the step right before the surprise gets updated (pure TD — credit crawls back one step per visit). $\lambda = 1$: every step in the episode shares it equally (Monte Carlo). In between, credit fades backwards. Most practical systems live around $\lambda \approx 0.9$–$0.95$ — the same dial as [[Actor-Critic|GAE]].

**4 · Subtract what was going to happen anyway.** "We won" credits *everyone*, including the player who did nothing. Ask instead: *was this action better than my usual?* That's the [[Value Function|advantage]], and it's why [[Policy Gradient]] needs a baseline.

**5 · Add honest intermediate rewards** — carefully → [[Reward Function#Sparse, dense, shaped|potential-based shaping]].

---
# The LLM version — and it's a live research problem 🤖

A model writes 800 tokens of reasoning and gets one bit back: *answer right* or *answer wrong*. Which of the 800 tokens deserve it?

| Approach | Credit goes to | Catch |
|---|---|---|
| **Outcome reward** — score the final answer | every token equally | simple and hard to game; extremely coarse → [[GRPO]] |
| **Process reward model (PRM)** — score each reasoning *step* | the step that went wrong | needs step-level labels; the PRM itself can be [[Reward Hacking\|gamed]] |
| **A learned critic** | each token, via a value model | that's [[PPO]] in [[RLHF]] — costs a second large model |

See [[Chain of Thought]], [[Test-Time Compute]], and in the vault [[DRACO- Fine-Grained Credit Assignment with Dynamic Rubrics for Long-Horizon Agent Training]] and [[Eligibility Traces for Off-Policy Policy Evaluation (ICML)]].

> [!WARNING] "Whatever happened just before the reward caused it"
> The most natural assumption and a reliably bad one. It rewards the striker and ignores the build-up; it blames the last deploy for an outage seeded three weeks ago; it credits the final prompt tweak for a gain that came from the data fix. ~={red}Proximity in time is not causation.=~ ^proximity-is-not-cause

---
> [!SUCCESS] If you remember one thing
> One late score, many early decisions. ~={pink}A value function is the device that turns that one late score into credit delivered *at the moment it was earned*.=~

---
# ⁉️
All of this assumed the agent can **see** the state it's assigning credit to. What if it can't — if it's acting through a keyhole?

→ [[POMDP]]
