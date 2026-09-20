---
aliases:
  - R2
  - Coefficient of Determination
tags:
  - R2
  - metrics
  - statistics
---
> [!ABSTRACT] 🧠 Recall
> **In one line:** What fraction of the outcome's variation does my model explain, compared to just guessing the average?
> **Metaphor:** Predicting exam scores. The class average is the baseline; $R^2$ says how much of the spread your model accounts for beyond it.
> **Where it bites:** It **always** rises when you add a variable (use **adjusted** $R^2$), it says nothing about causation, and a high $R^2$ does not mean a good model.

---
---
tags:
  - R2
---
It's called a **Coefficients of Determination**.
It's a stats measure that helps you explain - like what proportion of variation in outcome / dependent variable ($Y$) can be _explained_ by the predictor(s) / independent variables ($X$) in a [[Regression Analysis|regression model]]. Or rather how better is my model at predicting than making simple guesses?

> [!NOTE] Building Intuition by example
>So, imagine we have a dataset of 100 people's exam scores. If you had to predict the score of a new, random student—but you knew _nothing_ about them (like how much they studied, their past grades, etc.)—what would be your single best guess for their score?
>
>Probably the class average, no?
>That's a _really simple guess_!
>And then if we treat this as our **baseline** then we can see how far off (errors / variations) is our guess from the actual score. And essentially whatever _intelligent_ solution we build using these complex models, need to beat this average error - basically your model's prediction should be better than this random guess. 
>
>Now imagine we've a model that predicts the exams score using just 1 variable - _# hours studied_.
We already have our baseline. And say our model uses a simple regression to fit a line.
So higher the number of hours studied, higher the exam score - simple!
> ![[exam_score_vs_hours_studied_with_reg_line.png]]
What $R^2$ tells us is, ~={green}**how much this _variation_ / _error_ is reduced by using the model's regression line as opposed to our really simple guess**=~.


So, if we were to get an $R^2 = 0.70$ i.e. $70\%$ then it means our model would **explain away $70\%$ of our variations**.
But what does it mean?
It means, say if our random guess / really simple guess would have been say 10% of the times wrong (i.e. 10 out of 100 exam scores are incorrectly guessed), then a model with $R^2 = 0.7$ is only getting 3 out of those 100 exam scores incorrect - only 3% error / variation.

So an $R^2$ of $0$ would mean our model is as worse as our guess!
**But it also tells you that may be the $# hours studied$ and $# exam scores$ don't have a linear relationship

---
# The formula, and what the pieces are

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

- $SS_{tot}$ — squared error of the **baseline** (always guess the mean $\bar{y}$). Your "really simple guess"
- $SS_{res}$ — squared error of **your model**

So the fraction is *"how much error is left over"*, and $R^2$ is one minus that: ~={green}the proportion of variance you explained away.=~ Exactly the intuition in your exam-score example, written out.

> [!NOTE] Can it be negative?
> Yes — and it's a genuinely useful signal. If $SS_{res} > SS_{tot}$, your model is doing **worse than guessing the mean**. This can't happen for OLS on its own training data (the mean is always available to it), but it absolutely happens on a **test set**. A negative test $R^2$ means the model has learned something actively harmful. ^negative-r2

---
# The traps

> [!WARNING] 1. $R^2$ never goes down when you add a variable
> Add a column of **random noise** and $R^2$ will rise, or at worst stay flat. It can *never* fall, because the model can always assign the useless variable a coefficient of zero and be no worse off.
>
> So "my $R^2$ went up" is **not** evidence the new feature helps. This is [[Regularization#^overfitting|overfitting]] visible in a single number. 📈
>
> **Fix: adjusted $R^2$**, which penalises each additional predictor:
> $$R^2_{adj} = 1 - \frac{(1-R^2)(n-1)}{n - p - 1}$$
> This one *can* fall — and if it does when you add a feature, that feature is not earning its place. ^r2-always-increases

> [!WARNING] 2. High $R^2$ ≠ good model
> - A model can have $R^2 = 0.95$ and be **systematically wrong** — fit a straight line to a curve and you'll get a high $R^2$ with obviously patterned residuals. **Always plot the residuals.** 📉
> - Time-series regressions of two trending variables routinely produce $R^2 > 0.9$ and mean nothing at all — **spurious regression**.
> - $R^2$ is scale-free, which sounds nice but hides the practical question. An $R^2$ of 0.6 with errors of ±£2 and one with errors of ±£2,000 read identically. Report **RMSE or MAE** alongside it.

> [!WARNING] 3. Low $R^2$ ≠ bad model
> In human-behaviour domains, $R^2 = 0.15$ can be an excellent result — people are noisy, and most variance is genuinely irreducible ([[Uncertainty#^aleatoric|aleatoric]]). What matters is whether the coefficient is precisely estimated and meaningful, not whether the model explains most of the spread.
>
> ~={blue}What counts as a good $R^2$ is entirely domain-dependent.=~ Physics: 0.99. Marketing: 0.3 is a win.

> [!TIP] And the one everyone forgets
> $R^2$ measures **fit**, never **causation**. A high $R^2$ on ice-cream sales predicting drownings tells you nothing except that both follow temperature. See [[Regression Analysis#^coefficient-is-not-causal|why a coefficient is not an effect]].

---
# ⁉️
$R^2$ scores a fit. The design questions — is this relationship causal, and does the model generalise — run through [[Regression Analysis]] and [[Regularization]].
