The Kalman filter is a way to deal with uncertainty by combining two sources of information that are both imperfect.

---

So the **state** of the system is a set of variables that can perfectly describe the system at any given moment. Think of it like the current snapshot in time - like the environment, what they are in a given point in time.
For a simple moving car, the state might be its position and its velocity (speed and direction).
![[kalman_filter.png]]

🚀 Imagine you're trying to figure out the state of a rocket at a point in time. 
You've 2 ways to do this:
1. A **prediction** based on our understanding of physics. We know how fast the rocket's engine is burning, so we can predict where it should be next. This prediction is good, but it's not perfect—there are outside forces like wind that we can't account for.
2. A **measurement** from a sensor, like a radar (perhaps **noisy**). This measurement gives us a direct reading, but it's also imperfect because of static and interference.

The Kalman filter's genius is that it figures out **how to combine these two pieces of information** to get a much better, more accurate estimate of the rocket's true state.

Imagine your physics prediction tells you the rocket should be at 1,000 feet, but the noisy sensor reading says it's at 1,100 feet.
How would you decide which piece of information to trust more to come up with your best guess? What would make you favor one over the other?
Like a weighted average of the two - where the weights are the inverse of the uncertainty.
This is exactly what Kalman filters do!
They don't look at the numbers per se, but rather look at the uncertainty associated with them.
And this so called weight is called the **Kalman Gain**

It's a number that the filter calculates at every single step based on the uncertainty of both its prediction and the new measurement.
- If the sensor is very **uncertain** (noisy), the Kalman Gain is low, so the filter trusts the **prediction** more.
- If the sensor is very **certain** (reliable), the Kalman Gain is high, so the filter trusts the **measurement** more.

The filter does need some initial information about the uncertainty of its inputs. It doesn't start from zero and learn everything from scratch. Instead, it relies on two key pieces of information you have to give it from the very beginning:

1. **Process Uncertainty:** How much uncertainty or randomness is there in the system itself? For our rocket, this would be the uncertainty in our physics model—the unpredictable forces like wind or tiny engine variations. This tells the filter how much to "doubt" its own predictions over time.
    
2. **Measurement Uncertainty:** How much uncertainty is there in the sensor readings? This is a property of the sensor itself. A cheap radar will have a much higher uncertainty than an expensive one. This tells the filter how much to "doubt" the new measurements it receives.
    

So, the filter starts with a guess about the initial state and its uncertainty, and it uses those two values you provided to figure out the first Kalman Gain.