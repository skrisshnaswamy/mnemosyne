The backbone of machine learning. Without backprop, machines simply cannot learn. It's the process of finding how far away from truth is your prediction, and using this to tell your model, how to adjust it's weights and biases so that it can try again.

And you can't really learn backpropagation, without learning about [[chain rule]]. You see, in a deep learning network, we've many layers. And in such a network, backpropagation needs to use chain rule to compute the gradient at every single layer.

And on theory, backpropagation is done by computing the [[Derivative#Gradient|gradients]] of the outcome of each layer in the direction of the weights matrix. But unfortunately, a neural network is super complex with many layers - with each layer comprising of linear and non linear computation. So doing this is not really easy. So what Hinton suggested was this brilliant idea of [[Vector Jacobian Product]].

Since computing gradient with respect to the weights matrix directly isn't easy, we can simply compute the [[Vector Jacobian Product|VJP]] instead. This was 