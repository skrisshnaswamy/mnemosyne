---
tags:
---
PID stands for **P**roprotional **I**ntegral **D**erivative Controller

![[pid.png]]

The story of **control systems** and **stochastic models** is really about how we've learned to make things work the way we want them to, even when the world is messy and unpredictable. It all began with a simple idea: ~={blue}how do you get a system to hit a target and stay there?=~

The **PID controller** is one of the earliest and most successful answers to that question.

## **P** for Proportional

Imagine you're driving a car and you want to keep it at a constant speed, say 60 mph. Your speedometer shows you're at 55 mph. 🐢
How would you adjust the gas pedal to get to 60 mph? 
You press down on the gas pedal!
But what happens if you press the pedal too much?
The car speeds up past 60 mph. 🚙💨
So you ease off the pedal.
It's a constant cycle of adjusting based on the "error" (the difference between your current speed and your target speed.)

That's the essence of **proportional control**!
> _The amount of correction you make is **directly proportional** to how big the error is._
> $correction \propto error$

**Proportional control** is the **process** of adjusting a system based on the error. It's a control **law** or **rule** that says: "The amount of adjustment you make should be proportional to the size of the error."
- The **expected outcome** (the target) is to be at 60 mph.
- The **current state** is the car's actual speed, say 55 mph.
- The **error** is the difference between the target and the current state: 60−55=5 mph.
- The **controller** is the part that calculates the error and decides on the adjustment. In our simple human analogy, the **driver** is the proportional controller.
- The **adjustment** is how much you press or release the gas pedal.
	- this adjustment is a direct function of the error $adjustment = (some number) * error$
	- And that $some number$ is called the **proportional gain** ($K_p$​).
	- $P = K_p e(t)$
## **I** for Integral
Now imagine your car's engine has a slight headwind.
You press the gas pedal to get to 60 mph, but the headwind keeps slowing you down.
So you **p**roportional controller could never quite hit the target 60mph...
Say, it settles at 58 mph, because at that point, the force from the engine is just enough to counteract the headwind and the controller sees the small error as a "good enough" state.
Then, how would we adjust our strategy to make sure the car _eventually_ reaches exactly 60 mph, even with that constant headwind?
And even with shifting gears for more power, the proportional controller would still face the same issue. It would find a new "good enough" point, but it might not be the _exact_ target of 60 mph.
As long as there's a non-zero error, the proportional controller applies some adjustment, but it may *not be enough* to completely eliminate the error, especially when facing a persistent disturbance like a headwind.
This persistent, uncorrected error is what's called a **steady-state error**.
So what else can we do?
May be apply some kind of force to counter the headwind? But then **how much force to apply?**
May be simply looking at the *current error* isn't good enough.
May be it needs to look at the *past* - to counter the steady-state error problem, we need to "remember" the history of errors. Something that can **build up over time**.
	So currently it looks at 58 mph and thinks the adjustment only needs to counter 2mph. But what it needs to know is, it's been 2 mph for a while now.
	A component to the adjustment, that builds over time as long as the error is non-zero.
If you were to plot the error over time as a graph, it would be a line graph which is not at 2mph for a while. What's the sum of the area under this curve?
It's the sum of all the past errors over time.
![[pid_error_decay_curve.png]]
If the cumm. error is positive, the integral term keeps getting bigger and bigger, pushing the controller to increase the adjustment
If the cumm. error is consistently negative, the controller needs to decrease the adjustment.
$$
adjustment = (some\_number * current\_error) + (another\_number * sum\_of\_all\_past\_errors)
$$
where $some\_number$ is the proportional gain $K_p$
and $another\_number$ is the integral gain $K_i$
and the $sum\_of\_all\_past\_errors$ is basically $\int_{0}^{t} e(\tau)d\tau$ 
$I = K_i \int_{0}^{t} e(\tau)d\tau$
and so
$PI = K_p.e(t) + K_i \int_{0}^{t} e(\tau)d\tau$

## **D** for Derivative

