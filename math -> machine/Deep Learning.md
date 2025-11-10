So, way back in the 50s Rosenblatt came up with an idea of [[#^perceptron|perceptron]].

> [!NOTE] Perceptron
> A perceptron is basically supposed to be a single unit of neuron which is capable of learning and recognizing patterns. And it does that by assuming that different patterns are all linearly separable. So if it can learn this linear plane of separation, it can basically recognize and classify which category the sample belongs to. ^perceptron

The preceptron brought in the idea of [[#^weights|weights and biases]]. 

Think of **weights** and **biases** as the scalar coefficients of the features. So imagine we're trying to teach the perceptron how to determine whether or not a person is overweight. Now we use the person's _weight_ ($x_1$) and _height_ ($x_2$) as features. Now if we were to plot the height and weight of every person (sample) in a xy-plane, and then we find a line that separates the individuals who are normal weight and overweight, then this line can be defined by some linear combination of $x_1$ and $x_2$, right? Say for example the line is $3.2x_1 + 2.5x_2 + 6$ now $3.2$ and $2.5$ are the **weights** the perceptron assigns to these features (basically the perceptron tells that the weight of the individual is more important than the height of the individual when we're trying to determine whether the person is overweight or not). And the **bias** term is like a default value when $x_1$ and $x_2$ are 0, so we still have a non-zero outcome. And it is these **weights** and **biases** that the perceptron adjusts when it is trying to learn. ^weights

But then what happens when the boundaries aren't linear? That's where the famous **XOR** problem popped up and showed that when the perceptron is given a _really simple XOR_ problem it fails to learn!

So then decades later, folks realized, **What if we stack perceptrons?** This stack of perceptron is called the [[#^mlp|multi-layer preceptron]].

> [!NOTE] Multi-layer perceptron / Feedforward neural net
> An **MLP** or a **Multi-layer perceptron** or alternatively called a **Feed forward neural network**, is basically a stack of preceptrons. And the idea here was may be each layer could learn a different level of abstraction. Like in image recognition, some layers are responsible to detect edges, while some recognize shapes and others recognize objects. Same concept! ^mlp

But then the question came, ~={blue}How do you teach the layers in the middle?=~ You can see what the output layer does (because it gives you a prediction), but how do you tell the hidden layers how wrong they were?

That’s where [[#^backprop|backpropagation]] came in.

>[!NOTE] [[Backpropagation]]
> Around 1986, **Rumelhart, Hinton, and Williams** rediscovered and popularized this technique (it was mathematically known earlier). Backpropagation (or “backprop”) means exactly what it sounds like:  you take the **error** (the difference between the model’s prediction and the true answer) and **propagate it backward** through the network to adjust every layer’s weights. ^backprop

---
### **Backprop, Gradients, and the Vector-Jacobian Trick**

Here’s the intuition:

The **loss** is a single number — like ~={red}“how wrong am I?”=~
The **gradient** tells you: ~={blue}“how should each weight change to reduce this loss?”=~

For the output layer, you can compute this easily.  
But for earlier layers, you have to use the **chain rule** — the fundamental rule of calculus that tells you how changes in one part affect another indirectly.

If we write it out carefully:

- The loss depends on the output of the network.
- The output depends on each layer’s weights.
- So, the loss depends on the weights _through_ all those layers.

That’s a mess to differentiate directly.  
So we use a clever trick: **the Vector-Jacobian Product (VJP)**.

Instead of writing full Jacobian matrices (which are huge), we multiply the “vector” of derivatives of the loss (∂L/∂output) by each layer’s Jacobian (∂output/∂weights).  
This efficiently computes ∂L/∂weights — the actual gradient needed to update the weights.

So in essence: the VJP simplifies how we compute gradients layer by layer.