---
title: "Embedding Physics Priors in Robot Learning: A Survey"
authors: ["Mattia Piccinini", "Lucas Schulze", "Alice Plebe", "Matteo Saveriano", "Thomas Beckers", "Yuan Gao", "Oleg Arenz", "Baha Zarrouki", "Dingrui Wang", "Finn Rasmus Schäfer", "Jan Peters", "Johannes Betz", "Gastone Pietro Rosati Papini"]
year: 2026
arxiv: "2609.22319"
url: https://arxiv.org/abs/2609.22319
priority: Good-To-Read
read_on: 2026-09-28
tags: [paper, transformers, vision]
---
## The Core Idea

Robotics does not have the data that language and vision have. Real experiments cost money, break hardware, and can hurt people. A dexterous hand has 20+ joints and its dynamics are nonlinear, underactuated, and full of contact. So the scaling recipe that worked for text — more data, more compute, weak priors — does not transfer cleanly.

But robots have something text does not: they obey physics. Newton's laws, energy conservation, rotational symmetry, the fact that an inertia matrix must be positive definite. That knowledge is free. The question this survey answers is **where in a learning system you can put it**, and what each choice costs you.

The contribution is a taxonomy with three slots, adapted from Faroughi et al. (2024):

1. **Physics-guided** — physics shapes the *inputs, data, or representations*. You compute features from an analytical model, project states onto a manifold, generate training data with a differentiable simulator.
2. **Physics-encoded** — physics is baked into the *architecture*. The network cannot represent a non-positive-definite inertia matrix because of how it is wired.
3. **Physics-informed** — physics enters the *loss function* as a residual penalty. The governing equation is a soft constraint.

> [!NOTE] Lifecycle of a physics prior ^prior-lifecycle
> The three slots differ in *when the prior is active*. Physics-guided data curation is active only before training. Physics-informed losses are active only during training — at inference they survive only through the learned weights. Physics-encoded architecture is active at training **and** inference. This is the practical distinction: a low physics-residual loss at training time does not guarantee the model respects physics on a new input. An architectural constraint does.

Why this did not exist before: the literature is split across communities that do not share vocabulary. "Physics-informed", "physics-guided", "structured", "hybrid", "grey-box", "model-based deep learning" have all been used for overlapping and non-overlapping things. Earlier surveys covered either fluid/solid mechanics (no robots) or rigid-body dynamics only (16 of 108 references on robotics). This one has 232 of 329 references actually embedding physics in robot learning.

The framing argument: physics is a **robotics-specific [[An Image is Worth 16x16 Words (ViT)#^inductive-bias|inductive bias]]**, in the same way that convolution encodes translation invariance and [[Attention Is All You Need|attention]] encodes permutation-equivariant set mixing. It complements data, it does not replace it. That is a deliberate push against [[The Bitter Lesson (essay)|the Bitter Lesson]] — but a narrow one, because robotics is data-poor in a way Go and ImageNet were not.

The distribution of the field, which is the single most useful number in the paper:

| Route | Share of reviewed methods |
|---|---|
| Physics-encoded architectures | 71% |
| Physics-guided inputs/data | 16% |
| Physics-informed losses | 13% |
| Combining **two** routes | 4% |

## The Methodology

This is a survey, so "methodology" means the actual mechanisms. Here are the ones worth being able to sketch.

### Lagrangian and Hamiltonian architectures

Start with mechanics. The Lagrangian of a robot is kinetic minus potential energy:

$$\mathcal{L}(\mathbf q,\dot{\mathbf q}) = \tfrac12 \dot{\mathbf q}^{\mathrm T}\mathbf M(\mathbf q)\dot{\mathbf q} - \mathcal V(\mathbf q)$$

Push that through the Euler–Lagrange equation $\frac{d}{dt}\frac{\partial\mathcal L}{\partial\dot{\mathbf q}} - \frac{\partial\mathcal L}{\partial\mathbf q} = \boldsymbol\tau$ and you get the equation every roboticist knows:

$$\mathbf M(\mathbf q)\ddot{\mathbf q} + \mathbf c(\mathbf q,\dot{\mathbf q}) + \mathbf g(\mathbf q) = \boldsymbol\tau$$

**Deep Lagrangian Networks (DeLaN)**, Lutter et al. 2019, learn only two things: $\mathbf M(\mathbf q)$ and $\mathcal V(\mathbf q)$. Two small MLPs, each taking *only* $\mathbf q$ as input. Everything about how the dynamics depend on velocity and acceleration comes out of the Euler–Lagrange algebra, not from data. That is the whole trick: the velocity dependence is *derived*, so it generalises to velocities you never trained on.

Positive definiteness of $\mathbf M$ is enforced structurally, not penalised:

$$\mathbf M(\mathbf q) = \mathbf L(\mathbf q)\,\mathbf L(\mathbf q)^{\mathrm T}$$

with $\mathbf L$ lower triangular and its diagonal forced positive (e.g. by an exponential or softplus). You cannot produce an invalid inertia matrix. This is physics-encoded in the strict sense.

Because one learned model gives you energy, forward dynamics, *and* inverse dynamics, you can train on whichever you have labels for, and then drop it straight into a computed-torque controller or [[Model Predictive Control|MPC]].

Extensions worth knowing:
- **Sparsity.** Schulze et al. (2026) use Featherstone's reordered Cholesky $\mathbf M = \mathbf L^{\mathrm T}\mathbf L$, which lets you zero out entries according to the *branching structure of the kinematic tree*. This gives FeLaN, the floating-base extension for quadrupeds and humanoids.
- **Non-conservative forces.** Vanilla DeLaN assumes $\boldsymbol\tau$ is fully measured. Reality has friction, backlash, hydraulics. Fixes: a black-box MLP residual (Wu et al.), a structured Coulomb/viscous/Stribeck friction layer (Lahoud, Hu), a temporal CNN residual, Gaussian basis functions for backlash.
- **Lagrangian Neural Networks (LNN)** drop the $\tfrac12\dot{\mathbf q}^{\mathrm T}\mathbf M\dot{\mathbf q}$ assumption and learn $\mathcal L(\mathbf q,\dot{\mathbf q})$ as one unstructured network, recovering $\ddot{\mathbf q}$ by differentiating. More general, less structure, less interpretable.

The Hamiltonian route uses the Legendre transform $\mathcal H = \dot{\mathbf q}^{\mathrm T}\frac{\partial\mathcal L}{\partial\dot{\mathbf q}} - \mathcal L$ and gives

$$\dot{\mathbf q} = \frac{\partial\mathcal H}{\partial\mathbf p},\qquad \dot{\mathbf p} = -\frac{\partial\mathcal H}{\partial\mathbf q} + \boldsymbol\tau$$

**Hamiltonian Neural Networks** parameterise $\mathcal H_\theta(\mathbf q,\mathbf p)$ with an MLP and train on the residual of those two equations:

$$\Big\|\dot{\mathbf q} - \tfrac{\partial\mathcal H_\theta}{\partial\mathbf p}\Big\|_2 + \Big\|\dot{\mathbf p} + \tfrac{\partial\mathcal H_\theta}{\partial\mathbf q}\Big\|_2$$

Note what this is: the gradients come from [[Pytorch Autograd|automatic differentiation]] of the network, and the targets come from data. So HNN as originally written is *both* physics-encoded (Hamiltonian structure) and physics-informed (residual loss). **Port-Hamiltonian** models extend this with explicit energy ports for dissipation, actuation, and multi-domain coupling (mechanical + electrical + thermal), which connects naturally to passivity-based control. See [[Energy-Based Models]] for the general "learn a scalar, differentiate it" pattern.

The practical catch: $\mathbf p$ (generalised momentum) is almost never measured. You need $\mathbf M$ or a learned map to get it from velocities.

### Model-structured networks (MSNN)

The largest, messiest, most engineering-heavy family. You design layers that correspond to physical terms. Examples from vehicle dynamics:

- A layer whose output is quadratic in speed, because aerodynamic drag is.
- A neuro-fuzzy **FIR** layer over past throttle/brake inputs, because longitudinal acceleration depends on input history.
- Sigmoid activations that squash learned parameters into physically plausible ranges (so the identified cornering stiffness cannot come out negative).
- Force superposition: separate branches for traction, braking, drag, summed at the end.

The reward is that after training, individual weights *mean* something. The cost is that this is hand-built per application by someone who knows the domain. No design methodology exists.

### Physics-informed losses (PINNs)

The Raissi et al. recipe. Write the governing equation as $\partial_t u + \mathcal N[u;\lambda] = 0$, define the residual on the network's own output $\hat u$:

$$f(t,x) := \partial_t\hat u(t,x) + \mathcal N[\hat u(t,x)]$$

and train with

$$\mathcal J = \underbrace{\frac{1}{N_u}\sum_i |u(t_i,x_i)-\hat u(t_i,x_i)|^2}_{\text{data}} + \underbrace{\frac{1}{N_f}\sum_j |f(t_j,x_j)|^2}_{\text{physics}}$$

The physics term needs no labels — only *collocation points*, which you can sample anywhere. That is where the data efficiency comes from. Architecture stays a plain MLP, so this bolts onto anything: MLPs, LSTMs, temporal CNNs, even a VLA trajectory planner.

Two robotics-specific auxiliary losses worth stealing (Lutter et al. 2019a):

**Power consistency** — the rate of change of total energy must equal input power:
$$\mathbf l_p = \big\|\dot{\mathbf q}^{\mathrm T}\boldsymbol\tau - \dot{\hat{\mathcal E}}\big\|_2^2$$
Especially useful when you model friction separately, because it enforces passivity.

**Temporal coherence** — penalise the gap between the predicted energy at $t+\Delta t$ and its first-order Taylor extrapolation from $t$, with a stop-gradient on the target:
$$\mathbf l_e = \big\|\mathcal T_{t+\Delta t} - \mathrm{sg}(\mathcal T_t + \dot{\mathcal T}_t\Delta t)\big\|_2^2 + \big\|\mathcal V_{t+\Delta t} - \mathrm{sg}(\mathcal V_t + \dot{\mathcal V}_t\Delta t)\big\|_2^2$$
This is a bootstrapped target in the same spirit as a target network — smooths the learned energy surface.

### Neural ODEs and variational integrators

The [[Neural ODE]] observation: a [[Deep Residual Learning for Image Recognition (ResNet)|ResNet]] block is an Euler step.

$$\underbrace{\frac{d\mathbf x}{dt} = \mathbf f_\theta(\mathbf x,t)}_{\text{NODE}} \;\longrightarrow\; \mathbf x(t{+}\Delta t) = \mathbf x(t) + \underbrace{\mathbf f_\theta(\mathbf x,t)\Delta t}_{\text{ResNet}}$$

Useful in robotics because robots are continuous-time and sensors are irregular. But a generic Runge–Kutta solver destroys energy over long rollouts. **Variational Integrator Networks** fix this by using symplectic integrators derived from discrete variational principles, making symplecticity and momentum conservation hard architectural constraints. **Forced VINs** add control inputs and damping via discrete d'Alembert. [[Flow Matching]] and [[Rectified Flow]] sit on the same NODE foundation and are now the default for imitation-learning policies.

### Koopman operators (physics-encoded neural operators)

Lift a nonlinear system into a higher-dimensional space where it is approximately linear, then use linear control. The learnable part is the **dictionary** of lifting functions. Physics enters by *designing* that dictionary:

- Kronecker products of polynomials derived from configuration-space topology (Shi & Karydis).
- Lie-bracket constructions respecting nonholonomic constraints on an Ackermann vehicle.
- $\cos(\psi), \sin(\psi)$ embeddings for yaw, so heading geometry survives the lift.
- $SE(3)$-consistent observables built from rotation matrices for quadrotors.

Then run Koopman-MPC on the lifted linear model. This is how soft and continuum robots get controlled without a minimal-coordinate model.

### SINDy and topology learning

Pick a library of candidate functions, fit a *sparse* linear combination to $\dot{\mathbf x}$. Physics enters through the library: odd polynomials for lateral tyre force (because it is odd in slip angle), trigonometric terms for rotational soft-robot dynamics, Euler–Lagrange-derived terms for manipulators, hydrodynamic terms for marine vehicles. You get a symbolic equation you can read.

Related: First-Order-Principle networks (Díaz Ledezma & Haddadin, 2017 — notably *predating* DeLaN) that assemble a topology from Newton–Euler operators, and Equation Learner Networks that learn which activation functions to combine.

### Hybrid physics-learning

The least glamorous and probably most deployed family:
- **Learn a hard subsystem.** Replace the Pacejka tyre model — which needs expensive rig testing to parameterise — with an MLP, keeping the rest of the vehicle model analytical.
- **Residual learning.** $\hat y = f_{\text{physics}}(x) + g_\theta(x)$. Used with MLPs, GPs, GRUs, transformers, temporal CNNs. TossingBot is the clean example: analytical ballistics predicts release velocity, a learned residual corrects for grasp offset, mass distribution, aerodynamics.
- **Learned sensor pre-processing.** A CNN or LSTM cleans raw sensor data, and a [[Kalman Filter]] or [[Extended Kalman Filter|UKF]] with an analytical vehicle model consumes the output.

### Physics-guided generative models

The newest slice, and where almost all foundation-model work lands. Three mechanisms:
- **Equivariant architectures.** $SE(3)$-, $SO(2)$-, $SIM(3)$-equivariant [[Diffusion Policy|diffusion policies]] for manipulation.
- **Inference-time guidance.** Backpropagate a differentiable dynamics or kinematics metric into the denoising steps — structurally identical to [[Classifier-Free Guidance|classifier guidance]], but the guide is a physics model rather than a classifier. Payload-aware corrections for aerial manipulation, kinematic-feasibility projection for whole-arm control.
- **Physically consistent world representations.** [[Video Diffusion|Video world models]] conditioned on 4D occupancy, or on physical parameters (mass, friction, restitution) identified via a differentiable simulator plus Gaussian-splatting rendering loss.

## Ablation Studies and Experiments

A survey has no experiments of its own, so the useful reading is: **what the literature consistently reports, and what it reports failing.**

### What the accumulated comparisons show

- **DeLaN beats black-box MLPs on sample efficiency and extrapolation** on the same systems, in the same papers (Ramesh & Ravindran; Gupta et al.; Kotecha et al. on quadruped MPC). The mechanism is clear: velocity and acceleration dependence is derived rather than fitted, so the model extrapolates to unseen $\dot{\mathbf q}$, $\ddot{\mathbf q}$.
- **MSNNs beat MLPs and RNNs from limited data**, repeatedly, across vehicle dynamics work (Da Lio, Piccinini, Mungiello, Djeumou).
- **Physics-informed losses improve out-of-distribution behaviour specifically.** Saviolo et al.'s quadrotor model generalises beyond its training distribution; Bolderman et al. designed a regulariser that *deliberately* makes the model defer to the analytical prior in data-scarce regions, and proved input-to-state stability for the resulting controller.
- **Data efficiency numbers are striking where reported.** Djeumou et al. control a hexacopter from three minutes of flight data with physics-constrained neural SDEs. Dikici et al. identify race-car dynamics in under a minute of track data.
- **Bianchi et al.'s PINN parameter estimator beats an EKF baseline** for quadrotor thrust/drag with limited flight data.

### What did not work

This is the more valuable half.

**"Data does not replace geometry."** Rosenfelder et al. (2025) tried to learn Koopman observables for a differential-drive robot purely from data. The learned model *failed to recover the nonholonomic rolling-without-slipping geometry*. The structure had to be supplied by hand. This is the single sharpest negative result in the survey, and it generalises: physics-guided structure must be given, not discovered.

**Foundation models do not pick up physics for free.** Schulze Buschoff et al. found VLMs do not acquire intuitive physics from interaction alone — they needed RL against a physics-engine stability reward.

**$SE(3)$ body coordinates are a trap at scale.** Finzi et al.'s trick of learning Lagrangians/Hamiltonians in $SE(3)$ body coordinates with Lagrange-multiplier constraints simplifies $\mathbf M$ (spatial rather than joint-space inertia) but introduces redundant states and requires knowing the kinematic constraints in advance. Consequence: every application stayed on pendulums, quadrotors, and wheeled robots.

**Soft constraints are soft.** Low physics-residual loss at training time does not mean the governing equation holds at inference. This is stated repeatedly and is the honest ceiling on PINN-style methods. See [[Regularization#^reg-is-prior|regularisation as a prior]] — a penalty biases, it does not forbid.

**Residual learning inherits its base model's blind spots.** If the analytical part omits the dominant error source — friction, contact, compliance, actuator dynamics — the residual can only partially compensate, and the physical parameters lose their interpretation. Guarantees from hybrid models apply only to the analytical half.

**Validation is concentrated on easy systems.** The distribution:

| Robot class | Coverage |
|---|---|
| Manipulators + vehicles | 62% of methods |
| Legged robots (quadruped/humanoid, 20+ DoF) | 7% (16 papers) |
| Canonical systems *only* (pendulum, cart-pole, acrobot) | 17 papers |

So most of the claimed benefits are measured on systems under 10 degrees of freedom.

**SINDy is brittle in practice.** It regresses on state derivatives estimated from noisy measurements; it becomes ill-conditioned when excitation trajectories are uninformative (Omar et al. had to add inequality constraints to cope); and contact, hysteresis, and backlash do not admit compact symbolic libraries.

**Only 4% of methods combine two routes, and those combinations are ad hoc.** Lutter et al. pair a DeLaN architecture with a power-consistency loss; Bolderman et al. combine analytical model + MLP residual + physics-regularised loss. No one has systematically studied whether the routes are complementary or redundant.

### The gap the ablation literature cannot close

There is **no standardised physical-consistency metric and no shared benchmark.** Energy methods report energy drift; equivariant methods report momentum conservation; contact methods report penetration and complementarity violations; PINNs report ODE/PDE residuals. Different horizons, different perturbations, different tasks. Direct comparison is not possible. For world models this is acute — a rollout can be visually perfect and physically nonsense, which is the same failure mode as [[Evaluating Generative Models#^fid-blind-spots|FID's blind spots]].

## Worth Remembering

**The decision table is the takeaway.** The survey's Table 7 is the part to keep. Compressed:

| Family | Use when | Costs you |
|---|---|---|
| DeLaN | Rigid-body-like, non-conservative forces known or small, want interpretable $\mathbf M$/$\mathcal V$ | Needs collocated coordinates; dissipation and contact need bolt-ons |
| HNN / port-Hamiltonian | Energy exchange, dissipation, passivity are central | Needs generalised momenta $\mathbf p$, rarely measured |
| LNN | General Lagrangians (charged particles, exotic systems) | Weaker guarantees, less interpretable than DeLaN |
| MSNN | You know specific dependencies / force decompositions | Heavy manual design, zero transferability |
| SINDy / topology learning | You believe a compact symbolic equation exists | Library design is manual; dies on contact |
| Koopman operators | You want to use linear control on nonlinear dynamics | Lifting design is per-platform; lifted dimension explodes |
| NODE / VIN | Continuous time, long-horizon rollouts | Little embedded physics unless you add VIN structure |
| Hybrid / residual | Dynamics partly known, one hard-to-model piece | Guarantees apply only to the analytical half |

**Strength of prior is a dial, and the right setting is problem-dependent.** Smooth rigid-body dynamics tolerate strong constraints. Contact-rich dynamics need flexibility. Stronger priors buy interpretability and data efficiency; they cost expressiveness, and they often demand extra measurements (torques, accelerations, momenta) you may not have.

**Safety guarantees are almost absent.** Out of 232 physics-embedded robotics papers, the survey names essentially two with formal results: Bolderman et al. (input-to-state stability for feedforward motor control) and Liu et al. (passivity-based stability for soft-robot control with learned energy). No one has connected physics-embedded learning to Lyapunov certification, control barrier functions, or reachability analysis in a general way. [[Uncertainty]] quantification is also rare — a handful of GP, neural-SDE, and Gaussian-basis-function approaches.

**Sim-to-real: physics priors help exactly when the physics is invariant.** In DeLaN, the Euler–Lagrange skeleton transfers for free; only $\mathbf M$ and $\mathcal V$ need fine-tuning on real data. When the omitted effect (friction, contact, compliance) *is* the dominant error, the prior becomes a systematic bias that more data cannot remove. Usefully, failures are diagnostic — they tell you which assumption broke.

**The software ecosystem is fragmented, and there is a notable hole.** The survey's Table 10 is a good practical reference. Highlights: **nnodely** and **NeuroMANCER** for model-structured and hybrid architectures; **torchdiffeq / torchdyn / DiffEqFlux.jl** for NODEs; **PySINDy** for equation discovery; **deeptime** for Koopman; **DeepXDE / NeuralPDE.jl / PhysicsNeMo / NeuroDiffEq** for PINNs; **Brax / Warp / Nimble / Dojo.jl** for differentiable simulation. There is **no dedicated library for DeLaN, LNN, or HNN** — everyone rebuilds them on raw autodiff. And no framework spans all three embedding routes in one workflow.

**Worth stealing regardless of robotics.** Two ideas travel well beyond robots:
1. *Architectural constraint beats loss penalty when you need the property to hold at inference.* The Cholesky trick for positive definiteness is the model example — you cannot violate it, so you never need to check.
2. *A physics prior that determines how the output depends on an input is worth more than one that merely penalises bad outputs.* DeLaN's win is not the loss; it is that velocity dependence is algebraic, not learned.

**Where this connects to your line.** Nothing here is about [[Off-Policy Evaluation|OPE]] or bandits. But the underlying question — *how do you inject structure to get more out of scarce, expensive, partially observed data?* — is the same question variance reduction and doubly-robust estimation answer with a different tool. The model-based half of a doubly-robust estimator is exactly a "physics prior" in the sense used here, and it fails the same way: badly when the model omits the dominant effect, and you cannot tell from the training loss alone.

**Open questions the authors flag, and the honest ones:**
- Nobody knows how to pick which priors to embed, or how hard, for a given task. No guidelines exist.
- Reusable priors across embodiments remain unsolved. Everything transfers only within closely related morphologies.
- Contact-rich dynamics — impacts, friction, compliance, hybrid mode switching — is the largest genuine gap.
- Interpretability results are qualitative. There is no quantitative interpretability metric anywhere in the field.
- Automated discovery of physics-encoded architectures (from libraries of admissible physical building blocks) is proposed but barely attempted beyond FOP-nets and SINDy.

**Practical caveat if you wanted to use this.** The survey's own numbers say: 71% of the field builds bespoke architectures, and those architectures are validated mostly on sub-10-DoF systems by people with deep domain expertise. If you are not in a position to hand-design a mechanics-aware network, the cheapest entry points are (a) residual learning on top of whatever analytical model you already have, and (b) a physics-informed loss term, since neither requires changing your architecture. Both are weaker, and both are honest about it. The paper's repository is maintained and takes contributions: `github.com/TUM-AVS/survey-physics-embedded-robot-learning`.

## Links
Related: [[Neural ODE]] · [[Flow Matching]] · [[Energy-Based Models]] · [[Model Predictive Control]] · [[System Identification]] · [[State-Space Model]] · [[Kalman Filter]] · [[Extended Kalman Filter]] · [[Function Approximation]] · [[Diffusion Policy]] · [[Classifier-Free Guidance]] · [[Video Diffusion]] · [[Imitation Learning]] · [[Model-Based vs Model-Free RL]] · [[Control and Reinforcement Learning]] · [[The Bitter Lesson (essay)]] · [[An Image is Worth 16x16 Words (ViT)]] · [[Deep Residual Learning for Image Recognition (ResNet)]] · [[Regularization]] · [[Uncertainty]] · [[Evaluating Generative Models]] · [[Pytorch Autograd]] · [[Stability]] · [[Observability]] · [[LQR]] · [[Mastering Diverse Domains through World Models (DreamerV3)]] · [[Shortcut Learning in Deep Neural Networks]]

New topics worth writing: Deep Lagrangian Networks, Hamiltonian Neural Networks, Port-Hamiltonian Systems, Physics-Informed Neural Networks, Koopman Operator Theory, SINDy (sparse identification of nonlinear dynamics), Variational Integrators and Symplectic Integration, Gaussian Process Regression, Reproducing Kernel Hilbert Space, Equivariance and Group Symmetry in Neural Networks, Neural Operators (DeepONet / FNO), Differentiable Simulation, Sim-to-Real Transfer, Cosserat Rod Theory and Soft Robot Modelling, Cholesky Factorisation as an Architectural Constraint, Contact Dynamics and Complementarity Constraints, Control Barrier Functions, Residual Learning
