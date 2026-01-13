>[!SUCCESS] _How models are told what “good” means?_

---
# **L1 vs. L2 Loss**

>[!WARNING] In a regression task for house price prediction, why would you choose L1 (MAE) over L2 (MSE) if your dataset has extreme outliers like 'luxury mansions'?"

In a dataset where most houses are $300k and a few are $10M, those mansions are **outliers**.

1. **L2 (The "Magnifier"):** Because L2 squares the error, a mistake on a $10M mansion creates an **enormous** penalty.
    - Example: If the model predicts $8M for a $10M mansion, the error is $2M. L2 squares that to $4,000,000,000,000$.
    - The optimizer sees this massive number and panics. It will "pull" the entire model's weights toward the mansions to reduce that huge penalty, which ends up **ruining the predictions for the 99% of normal $300k homes.**
2. **L1 (The "Steady"):** L1 only penalizes the $2M error as $2,000,000$. It is much more "robust." It acknowledges the mistake but doesn't let the outlier dominate the entire learning process.

> [!WARNING] L1 is robust, but look at the graph of $|x|$. At the very bottom (where error is 0), the tip is 'pointy' (not differentiable). **How does this affect our gradient descent compared to the smooth curve of L2?**

## 1. The "Subgradient" Method (The Industry Standard)

Since the derivative doesn't exist at exactly 0, we use a **Subgradient**.
- **Intuition:** Instead of saying the slope is undefined, we _choose_ a value from the set of all possible slopes that "touch" that point.
- For L1 ($|x|$), the slope is $-1$ for $x < 0$ and $+1$ for $x > 0$. At exactly $x = 0$, we simply pick any value between $-1$ and $1$ (usually **0**).
- **Why it works:** Modern deep learning frameworks like **PyTorch** and **TensorFlow** have this "hack" built-in. If you look at their source code, they essentially use an `if-else` block:
$$
\text{grad} = 1 \text{ if } x > 0 \text{ else } (-1 \text{ if } x < 0 \text{ else } 0)
$$

> [!INFO] In calculus, a **Derivative** is a single line that is a "perfect fit" (tangent) to a curve at a point. For a smooth bowl (L2), there is only **one** such line at the bottom. For a "pointy" V-shape (L1), at the very bottom point, you **can't** have one "perfect" tangent line. 
>
Instead, you can fit **an entire family of lines** that stay underneath the V without poking through it.
> - Any line with a slope between **-1** and **1** will work.
> - Because there is a _set_ of lines rather than just _one_, each individual line in that set is called a **sub-gradient**.
> - The collection of all these lines is called the **sub-differential**.
>
> **In plain English:** It’s called a "sub"-gradient because any of these lines stay **below** (sub) the function.
> ##### Is it just a `coalesce()` hack?
Essentially, **yes**. That's the "developer secret" of ML.
Mathematically, the subgradient is a set: $[-1, 1]$. But a computer cannot multiply a weight by a "set." It needs a single number. So, PyTorch developers basically said:
>_"Since any number between -1 and 1 is technically a valid subgradient, let's just pick **0** to keep it simple."_
>
So, the logic inside the C++ code of these libraries looks exactly like the `coalesce()` SQL analogy:
>
> ```python
>if error > 0:
>    return 1
>elif error < 0:
>    return -1
>else:
>    return 0  # This is the "coalesced" default value
>```
>
> ##### Why?
This is important to understand **Gradient Stability**.
> - **The Problem:** If you used a pure mathematical derivative, your code would throw an error at $0$.
> - **The Solution:** We use subgradients because they allow us to keep training even when we hit the "pointy" bits of L1 or other non-smooth functions (like ReLU). 
> - **The Trade-off:** As you correctly identified, because the subgradient doesn't "shrink" as you get closer to zero (it stays at 1 or -1), your model won't converge as smoothly as L2 unless you use a **Learning Rate Scheduler**.

Now, Imagine your model is very close to the perfect prediction ($x=0$). Because L1 has a **constant slope** of 1 or -1 right until the very end, what happens to the weights if your learning rate is a bit too high?

If the slope is always **1** or **-1** and your learning rate is **0.1**, the model will just keep hopping back and forth over the line by **0.1** units. It can never "settle" into the groove unless you decay the learning rate to zero.
> [!TIP] Subgradients vs. Learning Rate Because a subgradient doesn't "shrink" as you get closer to zero (it's always 1, -1, or 0), the model will never naturally slow down. This is why **Learning Rate Decay** is non-negotiable for L1-style losses—you have to manually shrink the steps because the math won't do it for you.
## 2. The "Smooth Approximation" (The Engineering Fix)

If you are working with a sensitive second-order optimizer (like Newton's method) that needs the slope to be smooth, you can't have a "pointy" corner. In this case, we use the **Huber Loss** or a **Pseudo-Huber Loss**.

> [!INFO] **Newton’s Method: The "Curvature" Insight**
>
Standard Gradient Descent is "First-Order"—it only knows the slope. It's like walking down a hill with a blindfold, only feeling the tilt of the ground under your feet.
>
> - **The Intuition:** Newton's Method is "Second-Order." It looks at the **Hessian** (the rate at which the slope is changing). 
> - **Why it matters:** It can tell if the hill is getting steeper or flatter. This allows it to take much smarter, faster steps toward the bottom. 
> - **The Catch:** It requires the function to be **twice-differentiable** (smooth). You can't use Newton's Method on "pointy" L1 or ReLU without a smooth approximation like Pseudo-Huber.

### Huber & Pseudo-Huber: The "Best of Both Worlds"
Sometimes you want the robustness of L1 (to ignore those $10M mansion outliers) but you want the smoothness of L2 so the model doesn't "jitter" at the end.

- **Huber Loss:** It's literally a piecewise function: L2 (curved) near the bottom, and L1 (straight lines) for the outliers.
- **Pseudo-Huber Loss:** This is the MLE prod favorite. It's a single, continuous mathematical formula that **looks** like Huber but is differentiable everywhere.
    - It avoids the `if-else` logic of Huber, making it faster to compute on GPUs and compatible with **Newton's Method**.

### 3. Coordinate Descent

For some specific models (like **LASSO** regression), we don't use Gradient Descent at all. We use **Coordinate Descent**, where we optimize one feature at a time while holding the others constant. This bypasses the need for a global derivative entirely.