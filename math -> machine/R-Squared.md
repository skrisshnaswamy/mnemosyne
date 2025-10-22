---
tags:
  - R2
---
It's called a **Coefficients of Determination**.
It's a stats measure that helps you explain - like what proportion of variation in outcome / dependent variable ($Y$) can be _explained_ by the predictor(s) / independent variables ($X$) in a [[Regression Analysis|regression model]]. Or rather how better is my model at predicting than making simple guesses?

> [[!NOTE]] Understanding Intuition using an example
So, imagine we have a dataset of 100 people's exam scores. If you had to predict the score of a new, random student—but you knew _nothing_ about them (like how much they studied, their past grades, etc.)—what would be your single best guess for their score?
>
Probably the class average, no?
That's a _really simple guess_!
And then if we treat this as our **baseline** then we can see how far off (errors / variations) is our guess from the actual score. And essentially whatever _intelligent_ solution we build using these complex models, need to beat this average error - basically your model's prediction should be better than this random guess. 
>
Now imagine we've a model that predicts the exams score using just 1 variable - _# hours studied_.
We already have our baseline. And say our model uses a simple regression to fit a line.
So higher the number of hours studied, higher the exam score - simple!
> ![[exam_score_vs_hours_studied_with_reg_line.png]]
What $R^2$ tells us is, ~={green}**how much this _variation_ / _error_ is reduced by using the model's regression line as opposed to our really simple guess**=~.
>
>So, if we were to get an $R^2 = 0.70$ i.e. $70\%$ then it means our model would **explain away $70\%$ of our variations**.
>But what does it mean?
>It means, say if our random guess / really simple guess would have been say 10% of the times wrong (i.e. 10 out of 100 exam scores are incorrectly guessed), then a model with $R^2 = 0.7$ is only getting 3 out of those 100 exam scores incorrect - only 3% error / variation.

>So an $R^2$ of $0$ would mean our model is as worse as our guess!
>**But it also tells you that may be the $# hours studied$ and $# exam scores$ don't have a linear relationship 