---
aliases:
  - Rectified Flows
  - Reflow
  - Straight Flows
  - InstaFlow
  - Optimal Transport
  - OT Coupling
  - Optimal Transport Flow Matching
  - OT-CFM
  - Minibatch OT
tags:
  - generative-models
  - diffusion
  - performance
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** [[Flow Matching]] paths curve because randomly-paired training lines **cross**. **Re-pair** each noise sample with the image *the model itself* sends it to, retrain on those pairs — the lines stop crossing, the flow goes straight, and a straight flow needs **one step**.
> **Metaphor:** A tangle of strings stretched between two walls. Swap which nail each string ends on until none of them cross — every string is now taut and straight.
> **Where it bites:** Few-step and one-step generators (InstaFlow, SD3-Turbo-style models). And the link between generative modelling and **optimal transport**.

---
[[Flow Matching]] trains on perfectly straight lines between noise and data. Yet the flow it learns is curved.

Here's why. Take two noise points, $a$ (on the left) and $b$ (on the right), and two images, $A$ (left) and $B$ (right). Random pairing will sometimes give you $a \to B$ and $b \to A$. Those two lines form an **X**.

At the centre of the X, one line says *go right* and the other says *go left*. The network, trained with squared error, learns the average: *go nowhere in particular*. And an ODE's trajectories [[Neural ODE|can never cross]] — so the learned flow can't follow either line. It has to swerve.

~={blue}The network isn't at fault. The **pairing** is. What pairing would give it lines that never cross?=~

---
# Untangling the strings 🧵

![[rectified_flow_reflow.png]]
> [!TIP] Reading the chart
> **Left:** noise paired with *random* data points. Blue lines end in the top cluster, orange in the bottom — and they cross everywhere, because a noise point near the bottom is as likely to be paired with a top image as a bottom one. **Right:** after re-pairing. Points that start low go low; points that start high go high. **Nothing crosses**, so there's nothing to average away, and every path can be a genuine straight line.

Where do the good pairs come from? **From the model you already have.** That's the *reflow* procedure (Liu, Gong & Liu, 2022):

1. Train an ordinary flow-matching model (the "1-rectified flow"). Its paths are curved.
2. Draw fresh noise $x_0$, and **run the model's ODE** to get the image $x_1 = \text{ODE}(x_0)$ it produces. Save the pair.
3. Those pairs come from an ODE — whose paths *don't cross*. Retrain flow matching on **them**: straight lines between each $x_0$ and *its own* $x_1$.
4. The new flow is much straighter. Repeat if you like.

> [!NOTE] Rectified flow · reflow
> **Rectified flow:** a flow-matching model on straight-line paths. **Reflow:** iteratively retraining on (noise, generated-sample) couplings produced by the previous model, which provably doesn't increase — and in practice sharply reduces — the transport cost and path curvature. ^rectified-flow-def

> [!SUCCESS] Core idea
> ~={pink}Curvature comes from crossing; crossing comes from the pairing; so fix the pairing.=~ A straight, constant-speed trajectory is integrated **exactly** by one Euler step: $x_1 = x_0 + v_\theta(x_0, 0)$. One network call, one image. The cost has moved from *inference* (many steps, forever) to *training* (generate a dataset of pairs, once). ^fix-the-pairing

The measurement, from [[Diffusion Sampling#How few steps can you get away with?|the step-count experiment]]: at **one step**, stochastic diffusion's error is 6.4, the diffusion ODE's 1.2, flow matching's 1.9 — and the rectified flow's is **0.013**, already at the noise floor.

---
# The idea underneath: optimal transport

Ask the question abstractly: *you have a pile of sand shaped like a Gaussian, and you want it shaped like your data. What's the cheapest way to move every grain?* That's **optimal transport** (Monge, 1781), and its solution has exactly the property we want — **grains' paths never cross**, because if two did, swapping their destinations would be cheaper.

| Pairing | How | Paths |
|---|---|---|
| **Independent** (plain [[Flow Matching]]) | each noise sample with a random image | cross a lot → curved flow |
| **Minibatch OT** (OT-CFM) | within each batch, solve a small assignment problem to pair noise with nearby images | cross less → straighter. Cheap; no extra model |
| **Reflow** | pair noise with the model's own output | essentially non-crossing → near-straight |
| **True OT map** | the exact solution | perfectly straight. Intractable in high dimensions |

> [!TIP] In one dimension you can see the answer
> The optimal pairing in 1-D is simply **sort both lists and match in order** — smallest noise to smallest data, and so on. That's what the right-hand panel above is. In a million dimensions there's no "sorting", which is why you approximate it with batches or with reflow.

---
# What it costs

> [!WARNING] No free lunch
> - **Reflow needs a generated dataset.** Millions of (noise, image) pairs, each costing a full multi-step sample from the teacher. It's a big one-off bill.
> - **The student inherits the teacher's flaws.** Round 2 trains on the round-1 model's *outputs*, not on real data — errors compound, and quality typically drifts down a little with each reflow. It's [[Distillation|distillation]], with the usual caveats → [[Distillation#^self-distillation-mechanism|self-distillation]].
> - **One step is still not free.** In practice 1-step rectified flows lag multi-step quality; the common sweet spot is **2–8 steps**, often combined with an extra distillation stage. ^reflow-costs

---
# Where you'll meet it

- **Stable Diffusion 3 and Flux** are rectified-flow models in the narrow sense (straight-line paths, velocity prediction) — the base models still sample in ~20–50 steps.
- **InstaFlow, and the "Turbo" / "Schnell" / "Lightning" variants** are where reflow-style straightening and distillation buy 1–4 steps.
- It's one of three routes to few-step generation, and they're often combined:

| Route | Idea | Note |
|---|---|---|
| Better solvers | follow a curved path more cleverly | [[Diffusion Sampling]] — 10–25 steps |
| **Straighter paths** | remove the curve | this note — 1–8 steps |
| Learn the jump | skip the path; map any point straight to the end | [[Consistency Models]] — 1–4 steps |

---
> [!SUCCESS] If you remember one thing
> **Paths curve because training pairs cross. Re-pair so they don't, and the flow goes straight — and a straight flow is a one-step generator.** ~={pink}You're moving cost from every future inference into one round of training.=~

---
# ⁉️
Reflow straightens the road so one step covers it. There's a more direct idea: leave the road as curvy as it likes, and train a network that — from *any* point on it — tells you where the road ends.

→ [[Consistency Models]]
