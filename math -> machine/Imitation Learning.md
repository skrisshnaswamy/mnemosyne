---
aliases:
  - Behaviour Cloning
  - Behavior Cloning
  - Behavioural Cloning
  - Behavioral Cloning
  - BC
  - DAgger
  - Dataset Aggregation
  - Learning from Demonstrations
  - LfD
  - Exposure Bias
tags:
  - reinforcement-learning
  - deep-rl
  - alignment
  - decision-sciences
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** Skip the reward. Record an expert, and train a model to predict **what they did in each situation** — ordinary supervised learning. It works until the learner drifts somewhere the expert never went.
> **Metaphor:** Learning to drive from a perfect driver's dashcam footage. You never see how to recover from a skid — because they never skid.
> **Where it bites:** Self-driving, robotics — and **supervised fine-tuning of LLMs**, which is exactly this.

---
You want a car that drives itself. Writing the [[Reward Function|reward]] for "drive well" is hopeless.

But you have **100 hours of an excellent human driver**: a camera image and the steering angle, thirty times a second.

So don't do RL at all. Train a network: *image → steering angle*. Plain supervised regression. It reaches 98% accuracy on held-out frames.

You put it on the road. For the first thirty seconds it's flawless.

~={blue}Then it drifts ten centimetres to the left. What has it just done, that the expert never did?=~

---
# The dashcam 🚗

![[imitation_learning_drift.png]]

It's put itself in a situation **that isn't in the training data**. The expert was never ten centimetres off-centre — that's what made them an expert. So the network has never seen this view, and its output here is a guess. A slightly wrong one. Now it's twenty centimetres off. *Even less* familiar. A worse guess.

Supervised learning assumes the test data looks like the training data ([[Random variable#Why this underpins all of ML|the i.i.d. assumption]]). Here, **the learner's own mistakes change what it sees next** — and they change it toward places the expert never visited.

![[bc_compounding_error.png]]
> [!TIP] Reading the chart
> Forty runs each. The grey band is where the demonstrations live. **Behaviour cloning** (red) is fine *inside* it — but every small wobble that carries it outside is unrecoverable, because out there it has no idea what to do. By step 400, **half the runs have left the road**. **DAgger** (green), trained on the same task, doesn't leave at all.

> [!NOTE] Imitation learning · behaviour cloning
> **Imitation learning:** learn a policy from expert demonstrations instead of from a reward. **Behaviour cloning (BC)** is its simplest form: supervised learning on (state → expert action) pairs. ^imitation-def

> [!SUCCESS] Core idea
> ~={pink}A 2% error rate doesn't stay at 2% — it compounds.=~ In ordinary supervised learning, $\varepsilon$ error per step over $T$ steps costs about $\varepsilon T$. In behaviour cloning it grows like $\varepsilon T^2$, because each mistake pushes you into states where the *next* mistake is more likely (Ross & Bagnell, 2010). Same arithmetic as [[LLM Engineering#^compounding-error|the agent-reliability number]], with an extra twist: the errors *feed* each other. ^errors-compound

---
# The fixes

| Fix | Idea |
|---|---|
| **DAgger** (Dataset Aggregation) | let the **learner** drive. Record the states *it* gets into. Ask the expert: *"what would you have done **here**?"* Add those to the dataset. Retrain. Repeat. The training distribution becomes *the learner's own* — including its mistakes and how to recover from them |
| **Recovery data by construction** | NVIDIA's 2016 car had two extra cameras angled left and right, labelled with *"steer back towards the centre"*. Synthetic off-centre examples, for free |
| **Inject noise while recording** | perturb the expert a little so they *have* to demonstrate recoveries |
| **Learn the goal, not the moves** | infer what the expert was *trying to achieve* → [[Inverse Reinforcement Learning]] |
| **Clone first, then improve with RL** | BC as a warm start; RL to go beyond → AlphaGo, and every LLM pipeline |

DAgger's catch is practical: you need an expert **on call**, willing to label states they didn't choose to be in. Humans find that surprisingly hard and tiring.

---
# The LLM version — you already know this one 🤖

**Supervised fine-tuning is behaviour cloning.** The state is the text so far; the expert action is the next token a human wrote; the loss is [[Cross Entropy]]. → [[Instruction Tuning]], [[Fine-Tuning]].

And it has exactly the disease above, under the name **exposure bias**. During training the model always continues from *perfect, human-written* context. At generation time it continues from **its own** output. One odd token, and it's in a context it has never seen — and the next token is more likely to be odd too. That's a big part of why long generations drift, and why [[Hallucination|a confident wrong turn]] tends to stay wrong.

| What BC / SFT can't do | Why | What comes next |
|---|---|---|
| Exceed the demonstrator | it's trained to *match* them | RL against a reward |
| Learn from its own mistakes | it never sees them during training | **on-policy** training: sample from the model, then score → [[RLHF]], [[GRPO]] |
| Know *why* the expert did it | it copies actions, not intentions | [[Inverse Reinforcement Learning]], [[Preference Learning]] |
| Tell a good demonstration from a sloppy one | every example counts equally | it's a [[Policy Gradient#The thread to the rest of the vault 🧵\|policy gradient with every reward set to 1]] |

That table is, more or less, the argument for why LLM training doesn't stop at SFT.

> [!WARNING] "With enough demonstrations, cloning is enough"
> More expert data makes you better *inside* the expert's distribution — and leaves you just as lost outside it. The problem isn't the **amount** of data, it's **whose states** it covers. 10,000 hours of perfect driving contains roughly zero minutes of "how to recover from being half off the road". ^more-demos-dont-fix-it

---
> [!SUCCESS] If you remember one thing
> Copying an expert teaches you what to do **where the expert went**. ~={pink}Your own small errors take you somewhere else — so you must either train on *your own* states (DAgger, on-policy RL) or learn what the expert was actually trying to achieve.=~

---
# ⁉️
A cloned policy has the expert's *moves* and none of their *reasons*. Put it in a new city and it's lost. But if you could work out **what the expert was optimising for**, you could re-plan anywhere. Can a goal be recovered from behaviour alone?

→ [[Inverse Reinforcement Learning]]
