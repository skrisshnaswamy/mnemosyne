# Derivative

A derivative of a function is basically a way to tell how much does the function's outcome change, when the input is changed by a unit measure.
Suppose we've a function $f(x) = x^2$ , here we know the "rate of change" i.e. **how much** change, is calculated by $\dfrac{d}{dx} (x^2)$ which is $2x$.

So it essentially says for every unit change in $x$, $f(x)$ changes by twice of $x$.

But then how about functions that depend on more than 1 variable. Example: $f(x,y)$. Here the "rate of change" isn't as straightforward as the derivative of $x$ or $y$, and more importantly it (here it refers to the change rate itself which ironically is a function too 😏 ) isn't just a function of 1 variable too. Meaning if we've a function which depends on 2 variable, the function itself can be plotted on a 2D plane. And such a function changes on both axes, no? So the "rate of change" is infact also breaks down to
- rate of change _with respect to_ $x$ (or rather rate of change _in the direction of_ $x$)
- rate of change _with respect to_ $y$

Which also means there doesn't exist a **derivative** so to speak, but rather 2 [[Derivative#Partial Derivative|partial derivative]] in 2 axes, no? There isn't a single number that can describe the rate of change in **all** directions at once. 

---

# Partial Derivative

If you're here, you must have read about [[Derivative#Derivative|derivative]] first. So you already know, a partial derivative is simply the derivative of a [[Fundamentals#Multi-variate function|multi-variate function]]. As we already know, when a function is modelled as a machine that takes in many inputs, it's "rate of change" cannot be crammed into a single scalar value.
Such a function has derivates (rate of change) across all the different axes in which the function exists.
What it means is, a function $f(x_1, x_2, ... x_n)$ exist in an $n$ dimension space and so has $n$ axes. And it's "rate of change" in each of these $n$ direction is defined by $n$ different partial derivatives.
- rate of change in the direction of $x_1$ would be $\dfrac{\delta}{\delta x_1} f$ 
- rate of change in the direction of $x_2$ would be $\dfrac{\delta}{\delta x_2} f$ 
- ...
- rate of change in the direction of $x_n$ would be $\dfrac{\delta}{\delta x_n} f$ 
This explanation neatly leads to our next topic - [[Derivative#Gradient|gradients]]

And when you start learning about partial derivatives, you will learn terms like
- [[Derivative#Jacobian|Jacobian]]
- [[Derivative#Hessian|Hessian]]

---
# Gradient

Gradients are nothing but the [[Derivative#Partial Derivative|partial derivatives]] of a [[Fundamentals#Multi-variate function|mutli-variate function]]. Well, very similar to how a single equation of multi-variate function is represented as an _algebraic equation_, but if you have $n$ such equations they are called a _system of algebraic equations_. And such a _system of algebraic equations_ can be alternatively represented in the form of a [[Fundamentals#Matrix|matrix]]. Exactly like that, when you think of representing all the $n$ partial derivatives of a single multi-variate function - we end up with a _vector_ of size $n$
For a function $f$ where $f(x_1, x_2, ... x_n)$
- rate of change in the direction of $x_1$ would be $\dfrac{\delta}{\delta x_1} f$ 
- rate of change in the direction of $x_2$ would be $\dfrac{\delta}{\delta x_2} f$ 
- ...
- rate of change in the direction of $x_n$ would be $\dfrac{\delta}{\delta x_n} f$ 
So a gradient $\nabla f$ is $\begin{bmatrix}\dfrac{\delta}{\delta x_1} f\\ \dfrac{\delta}{\delta x_2} f\\ ...\\ \dfrac{\delta}{\delta x_n} f\end{bmatrix}$

---
# Jacobian
So when you've a system of multi-variate equations all neatly written down in the form of a matrix, then we don't just get a Gradient you see. Gradient is just a vector representation of 1 single multi-variate equation. When you deal with a system of equations, you will end up with a matrix of partial derivatives. Such a matrix is called a Jacobian.

Simply put, it's just a matrix of first order partial derivative of each equation in the direction of each feature.

---

# Hessian
It's the second order partial derivative of a system of equation or [[Fundamentals#Multi-variate function|multi-variate function]], all neatly arranged in the form of a matrix. But here is the catch, the equations should all be of the form $f(x_1, x_2, .., x_n)$ s.t. $f: \mathbb{R}^3 \to \mathbb{R}^1$. (meaning the output of each of these equations should be a single scalar value)

It is also a square, symmetric matrix.

### Application & Intuition

While the [[Derivative#Gradient|gradient]] tells us the slope or "direction" of a function, the Hessian tells us about its **curvature**. It helps us understand if a function's critical point is a minimum, maximum, or a saddle point.

### Role in Machine Learning & Statistics

The Hessian is also crucial to [[Optimization]]. 
#### The Hessian's True Role in Regression

Think about how a statistical package finds the "best" coefficients for a regression. It's trying to minimize a [[Loss Function]] (like the Sum of Squared Errors) or maximize a **Log-Likelihood Function**. This is an optimization problem.

Let's use an analogy: finding the lowest point in a hilly landscape.
- The **Loss Function** is the elevation of the landscape.
- The **Gradient** of the Loss Function is a vector that points in the direction of the steepest _uphill_ slope. Gradient Descent follows the opposite direction to find the bottom.
- The **Hessian** of the Loss Function describes the **curvature** of the landscape at your current position. It tells you if you're in a steep, narrow valley or a wide, flat bowl.

**So, in regression, the Hessian is used for:**

1. **Smarter Optimization:** Algorithms like Newton's Method use the Hessian (curvature information) to take more direct and intelligent steps toward the minimum loss, often converging much faster than simple Gradient Descent.
2. **Calculating Standard Errors:** After the optimization finds the best coefficients, the inverse of the Hessian matrix (evaluated at the minimum) is used to calculate the variance-covariance matrix of your estimated coefficients. This is where the standard errors, t-statistics, and p-values that you see in regression output come from!


---

### Understanding **Curvature**

Let's try to understand **Hessian in one dimension** and then progress to N-dimension.

#### 1. Let's Start in 1 Dimension (with a Ball and a Hill)

Imagine you have a function $f(x)$ that describes the height of a hilly terrain along a single line.

- The **function $f(x)$** is your **elevation**.
- The **first derivative $f'(x)$** (the slope) is how **steep** the ground is.
	- If you place a ball, $f'(x)$ tells you which way it will start to roll and how fast.
	- If $f'(x) = 0$, the ground is flat, and the ball stays put.
- Now, the **second derivative $f''(x)$** is the **curvature**.
	- It answers the question: **"What is the _shape_ of the ground where the ball is resting?"**
	- If $f''(x)$ is POSITIVE ( $> 0$ ):
		- The ground is **shaped like a bowl or a valley** (∪). It "curves up". ![[pos_curvature.png]]
		- If you place a ball at a flat spot (where $f'(x) = 0$), it's at the bottom. It's a minimum.
		- The bigger the value of $f''(x)$, the narrower and steeper the walls of the bowl.
	- If $f''(x)$ is NEGATIVE ( $< 0$ ):
		- The ground is **shaped like a hilltop** (∩). It "curves down". ![[neg_curvature.png]]
		- If you place a ball at a flat spot (where $f'(x) = 0$), it's balanced precariously at the very top. It's a maximum.
	- If $f''(x)$ is ZERO ( $= 0$ ):
		- There is no curvature. The ground is a straight line (either flat or a constant slope).
	- **In short: The second derivative tells you which way the slope _itself_ is changing.**

---

### 2. Now, Let's Go to Higher Dimensions (The Hessian)

In two or more dimensions, our "terrain" is a complex surface, not just a line.

- The **Gradient $∇f$** is still the direction of steepest ascent (the "uphill" direction). And zero gradient means you're at a flat spot (a critical point).

But now, the question "What is the _shape_ of the ground?" is much more complicated. It's not just "a bowl" or "a hill". It could be a bowl, a hill, or something like a horse's saddle or a mountain pass.

The **Hessian matrix** captures all of this multi-dimensional shape information. When we analyze the Hessian at a flat spot (where the gradient is zero), it tells us what kind of point we're at:

- If the Hessian is "Positive Definite" (all its eigenvalues are positive):
	- This is the multi-dimensional equivalent of $f''(x) > 0$.
	- The surface **curves up in every direction**. You are at the bottom of a bowl. This is a local minimum. ~={pink}This is what optimizers want to find!=~
- If the Hessian is "Negative Definite" (all its eigenvalues are negative):
	- This is the equivalent of $f''(x) < 0$.
	- The surface **curves down in every direction**. You are on top of a hill. This is a local maximum.
- If the Hessian is "Indefinite" (it has both positive and negative eigenvalues):
	- This is the new, interesting case!
	- The surface curves up in some directions, but curves down in others.
	- This is a saddle point, like a mountain pass.
	- It's a minimum along one path, but a maximum along another.

#### Why This Matters for Optimization

The Hessian gives our optimization algorithm a "map of the terrain" instead of just a compass.

1. **It confirms a minimum:** It verifies that a point with a zero gradient is a true valley floor, not a tricky saddle point where an algorithm could get stuck.
    
2. **It helps us move smarter:**
    - If the curvature is very **high** (a narrow canyon), the Hessian tells the algorithm to take smaller, more careful steps to avoid bouncing from side to side.
    - If the curvature is very **low** (a wide, flat plain), the gradient is tiny and Gradient Descent would crawl. The Hessian tells the algorithm the ground is flat, so it's safe to take a much bigger "leap" to make progress.

So, think of the Hessian as the tool that lets us understand the full, rich geometry of a function's surface.

### Hessian and Momentum

~={red} **No, Hessian and Momentum aren't exactly the same!** =~
But it's a good segue to understand [[Momentum]] after understanding Hessian. You see, in a way, hessian and momentum, both are ways to navigate a complex loss function landscape - they both try to solve similar problem with gradient decent. I.e. whether to speed up or slow down the learning, when navigating the terrain. ~={blue} But, they use very different information to do this.=~

#### The Core Idea: Past vs. Present

- **Second-Order Methods (Hessian):** These methods use information about the **present**.
	- They analyze the _local curvature_ of the loss function _right now_ to decide on the best step to take.
	- It's like a sophisticated hiker who uses a special device to measure the exact shape of the valley floor to plot the most intelligent path down.
- **Momentum:** This is a **first-order** method.
	- It doesn't know anything about curvature.
	- Instead, it uses information about the **past**. It builds up inertia by keeping a "memory" of the previous gradients.
	- It's like a heavy ball rolling downhill—it doesn't analyze the valley's shape, but its own past motion prevents it from making sudden, jerky moves.

##### The "Heavy Ball" Analogy

This is the best way to build an intuition for momentum.

1. **Vanilla [[Gradient Descent]]:** Imagine a person hiking down a valley. At every single step, they stop, look at the ground to find the steepest downhill direction, take exactly one step, and then stop again.
	- They have no memory of their previous direction.
	- This is slow, especially on long, flat plains, and in a narrow canyon, they might just zig-zag from one wall to the other, making very slow progress downwards.
2. **Gradient Descent with Momentum:** Now, imagine a heavy bowling ball rolling down that same valley.
    - **On a long, gentle slope:** The gradient is always pointing in the same direction. The ball picks up speed and barrels through these flat regions much faster than the hiker. This is momentum **accelerating convergence**.
    - **In a narrow canyon (ravine):** The gradient points sharply from side to side. The hiker would oscillate wildly. But the heavy ball's inertia (its momentum) wants to keep it going straight down the main direction of the ravine. The side-to-side gradients aren't strong enough to change its course completely; they just nudge it slightly.
	    - The momentum effectively averages out the oscillations and helps the ball stay on a more direct path to the bottom. This is momentum **dampening oscillations**.
        
#### How The Math Differs

This is where you can see there's no second derivative.

- Vanilla GD Update: 
	$parameter = parameter - learning\_rate * current\_gradient$
    
- **Momentum Update (simplified):**
    1. $velocity = (beta * velocity) + (learning\_rate * current\_gradient)$
    2. $parameter = parameter - velocity$
        

Notice that the momentum update only uses the `current_gradient` (the first derivative) and the `velocity` from the last step. The `beta` term is a friction-like constant (e.g., 0.9) that determines how much of the past velocity is kept. There's no Hessian (`H`) anywhere in the calculation.

~={green}**But here is the catch!
Gradient is like the property of the landscape (in ML terms - the feature space) but velocity is like the property of the learning algo which is trying to model this feature space.
---
You see, when you analyze a function, you have fundamental properties like the function's value, its derivative (slope), and its second derivative (curvature). These exist regardless of how you analyze the function. But **"velocity" is a concept we borrow from physics** and **not** a fundamental property of the loss function. It is a **stateful variable created and maintained by the optimization algorithm itself.** It's just an **Exponentially Weighted Moving Average (EWMA) of the past gradients**.
=~
### Summary Table

| Feature       | **Gradient Descent**                                  | **Momentum**                                              | **Second-Order Methods (Newton's)**                                         |
| ------------- | ----------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Core Idea** | Take a small step in the steepest downhill direction. | Accumulate a "velocity" based on past steps.              | Model the surface as a bowl and jump to the bottom.                         |
| **Info Used** | 1st Derivative (Gradient) **only**.                   | 1st Derivative (Gradient) over **multiple past steps**.   | 1st Derivative (Gradient) AND 2nd Derivative (Hessian).                     |
| **Analogy**   | A cautious, memoryless hiker.                         | A heavy ball rolling downhill with inertia.               | An intelligent drone that maps the terrain's curvature.                     |
| **Strength**  | Simple, easy to implement.                            | Faster on flat surfaces, dampens oscillations in ravines. | Converges very quickly on friendly, bowl-shaped loss surfaces.              |
| **Weakness**  | Can be very slow; oscillates in ravines.              | Requires tuning an extra hyperparameter (`beta`).         | Calculating the Hessian is extremely expensive/impossible for large models. |

**Conclusion:** Momentum is a brilliant and computationally cheap _trick_ that uses the history of first derivatives to navigate complex surfaces more effectively. It mimics some of the benefits of knowing the curvature without ever needing to calculate the expensive second derivative.
